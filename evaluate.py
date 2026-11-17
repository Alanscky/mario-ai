"""
Evalúa un agente como lo haría el evaluador del profe: el juego avanza de a
UN fotograma, se le pasa al Agent la imagen cruda y él devuelve la acción.
(No usa los wrappers de entrenamiento: así se comprueba que Agent es autónomo.)

Uso:
    python evaluate.py --model runs/config_a_XXXX --episodes 30
    python evaluate.py --model runs/config_a_XXXX/checkpoints/ppo_400000_steps.zip --episodes 10
    python evaluate.py --model runs/config_a_XXXX --deterministic --episodes 1 --video results/a.mp4
    python evaluate.py --model runs/config_a_XXXX --render          # ventana en vivo (PC local)

Métricas (sección 3.8 / 4.2):
    completado, x máximo, progreso normalizado, fotogramas, causa de fin.

Nota: el juego es determinista. Con --deterministic, todos los intentos salen
IGUALES, así que solo tiene sentido 1 intento. Por defecto la política muestrea
(estocástica) con una semilla distinta por intento, lo cual mide qué tan robusto
es el agente.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from mario_ai.agent import Agent
from mario_ai.env import FLAG_X_1_1, MarioGymnasium


def end_reason(info: dict, terminated: bool, frames: int, max_frames: int) -> str:
    """Causa de fin del intento. 'caida' y 'enemigo' son una HEURÍSTICA basada en la
    altura de Mario al morir (en el suelo y_pos≈79; en un hueco queda por debajo)."""
    if info.get("flag_get"):
        return "bandera"
    if not terminated and frames >= max_frames:
        return "limite_frames"
    if info.get("time", 1) <= 0:
        return "tiempo"
    if info.get("y_pos", 79) < 70:
        return "caida"
    return "enemigo"


def run_episode(env, agent, max_frames, video=None, render=False):
    frame, _ = env.reset()
    agent.reset()
    max_x, frames, info, terminated = 0, 0, {}, False
    while frames < max_frames:
        action = agent.act(frame)
        frame, _, terminated, truncated, info = env.step(action)
        frames += 1
        max_x = max(max_x, info.get("x_pos", 0))
        if video is not None:
            video.write(frame[:, :, ::-1])       # RGB -> BGR para OpenCV
        if render:
            env.render()
        if terminated or truncated:
            break
    return {
        "completed": int(bool(info.get("flag_get", False))),
        "max_x": int(max_x),
        "progress": round(min(max_x / FLAG_X_1_1, 1.0), 4),
        "frames": frames,
        "end_reason": end_reason(info, terminated, frames, max_frames),
        "x_end": int(info.get("x_pos", 0)),
        "y_end": int(info.get("y_pos", 0)),
        "score": int(info.get("score", 0)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="carpeta del run o checkpoint .zip")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--max-frames", type=int, default=10000, help="AJUSTAR al anexo técnico")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", help="nombre de la carpeta en results/")
    p.add_argument("--video", help="guarda un .mp4 del PRIMER intento")
    p.add_argument("--render", action="store_true", help="muestra ventana (requiere pantalla)")
    args = p.parse_args()

    agent = Agent(args.model, deterministic=args.deterministic)
    env = MarioGymnasium(agent.cfg["action_set"], render_mode="human" if args.render else None)

    label = args.label or Path(args.model).with_suffix("").name
    if label in ("", "."):
        label = "eval"
    out_dir = Path("results") / label
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    t0 = time.time()
    for ep in range(args.episodes):
        agent.model.set_random_seed(args.seed + ep)
        video = None
        if args.video and ep == 0:
            import cv2
            Path(args.video).parent.mkdir(parents=True, exist_ok=True)
            video = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), 60, (256, 240))
        r = run_episode(env, agent, args.max_frames, video, args.render)
        if video is not None:
            video.release()
        r["episode"] = ep
        rows.append(r)
        print(f"[{ep + 1:>3}/{args.episodes}] completo={r['completed']}  x_max={r['max_x']:>4}  "
              f"progreso={r['progress']:.0%}  frames={r['frames']:>5}  puntos={r['score']:>5}  fin={r['end_reason']}")
    env.close()

    with open(out_dir / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    done = [r for r in rows if r["completed"]]
    reasons = {}
    for r in rows:
        reasons[r["end_reason"]] = reasons.get(r["end_reason"], 0) + 1
    summary = {
        "model": args.model,
        "episodes": len(rows),
        "deterministic": args.deterministic,
        "max_frames": args.max_frames,
        "completion_rate": len(done) / len(rows),
        "max_x_best": max(r["max_x"] for r in rows),
        "max_x_mean": float(np.mean([r["max_x"] for r in rows])),
        "progress_mean": float(np.mean([r["progress"] for r in rows])),
        "frames_mean": float(np.mean([r["frames"] for r in rows])),
        "frames_mean_completed": float(np.mean([r["frames"] for r in done])) if done else None,
        "frames_best_completed": min((r["frames"] for r in done), default=None),
        "score_mean": float(np.mean([r["score"] for r in rows])),
        "score_max": max(r["score"] for r in rows),
        "score_mean_completed": float(np.mean([r["score"] for r in done])) if done else None,
        "end_reasons": reasons,
        "eval_seconds": round(time.time() - t0, 1),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nResultados en {out_dir}/")


if __name__ == "__main__":
    main()
