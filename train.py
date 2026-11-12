"""
Entrena un agente PPO para Mario 1-1.

Uso:
    python train.py --config configs/config_a.yaml
    python train.py --config configs/config_a.yaml --resume runs/config_a_XXXX/checkpoints/ppo_400000_steps.zip
    python train.py --config configs/config_a.yaml --timesteps 500000      # sobrescribe total

Genera en runs/<nombre>_<fecha>/:
    model.zip, agent_config.json   -> modelo final (lo que carga Agent)
    checkpoints/                   -> modelos intermedios
    episodes.csv                   -> un renglón por episodio (para gráficas del reporte)
    tb/                            -> logs de TensorBoard
    config.yaml, meta.json         -> configuración y recursos usados (pide la rúbrica)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import time
from datetime import datetime
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

from mario_ai.env import make_train_env


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def env_factory(cfg: dict, rank: int):
    def _init():
        env = make_train_env(cfg)
        env = Monitor(env, info_keywords=("x_pos", "flag_get", "frames"))
        env.reset(seed=cfg["seed"] + rank)
        return env
    return _init


class EpisodeLogger(BaseCallback):
    """Guarda cada episodio en CSV y manda progreso a TensorBoard."""

    def __init__(self, csv_path: Path):
        super().__init__()
        self.csv_path = csv_path
        self.n_episodes = 0
        self.recent = []  # últimos 100 episodios

    def _on_training_start(self):
        new = not self.csv_path.exists()
        self._f = open(self.csv_path, "a", newline="")
        self._w = csv.writer(self._f)
        if new:
            self._w.writerow(["episode", "timesteps", "reward", "steps", "frames", "x_pos", "flag_get"])

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is None:
                continue
            self.n_episodes += 1
            row = [self.n_episodes, self.num_timesteps, round(ep["r"], 3), ep["l"],
                   ep.get("frames", 0), ep.get("x_pos", 0), int(bool(ep.get("flag_get", False)))]
            self._w.writerow(row)
            self.recent = (self.recent + [row])[-100:]
        if self.n_episodes and self.n_calls % 200 == 0:
            xs = [r[5] for r in self.recent]
            self.logger.record("mario/x_pos_mean_100", sum(xs) / len(xs))
            self.logger.record("mario/x_pos_max_100", max(xs))
            self.logger.record("mario/flag_rate_100", sum(r[6] for r in self.recent) / len(self.recent))
            self.logger.record("mario/episodes", self.n_episodes)
            self._f.flush()
        return True

    def _on_training_end(self):
        self._f.close()


def hardware_info() -> dict:
    import torch
    info = {
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        import stable_baselines3, gymnasium, gym_super_mario_bros, nes_py  # noqa
        info["stable_baselines3"] = stable_baselines3.__version__
        info["gymnasium"] = gymnasium.__version__
    except Exception:
        pass
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume", help="checkpoint .zip para continuar un entrenamiento")
    p.add_argument("--timesteps", type=int, help="sobrescribe total_timesteps")
    p.add_argument("--n-envs", type=int, help="sobrescribe n_envs")
    p.add_argument("--run-dir", help="carpeta de salida (por defecto runs/<nombre>_<fecha>)")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.timesteps:
        cfg["total_timesteps"] = args.timesteps
    if args.n_envs:
        cfg["n_envs"] = args.n_envs

    run_dir = Path(args.run_dir or f"runs/{cfg['name']}_{datetime.now():%Y%m%d_%H%M}")
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "config.yaml")

    # Lo mínimo que necesita Agent para reproducir el preprocesamiento
    agent_cfg = {k: cfg[k] for k in ("action_set", "frame_skip", "frame_size", "frame_stack")}
    with open(run_dir / "agent_config.json", "w") as f:
        json.dump(agent_cfg, f, indent=2)

    # Entornos en paralelo
    n = cfg["n_envs"]
    fns = [env_factory(cfg, i) for i in range(n)]
    venv = SubprocVecEnv(fns) if n > 1 else DummyVecEnv(fns)
    venv = VecFrameStack(venv, n_stack=cfg["frame_stack"], channels_order="last")

    if args.resume:
        print(f"Reanudando desde {args.resume}")
        model = PPO.load(args.resume, env=venv, tensorboard_log=str(run_dir / "tb"))
    else:
        model = PPO("CnnPolicy", venv, verbose=1, seed=cfg["seed"],
                    tensorboard_log=str(run_dir / "tb"), **cfg["ppo"])

    callbacks = CallbackList([
        CheckpointCallback(save_freq=max(cfg["checkpoint_every"] // n, 1),
                           save_path=str(run_dir / "checkpoints"), name_prefix="ppo"),
        EpisodeLogger(run_dir / "episodes.csv"),
    ])

    meta = {"config": args.config, "started": datetime.now().isoformat(timespec="seconds"),
            "hardware": hardware_info(), "resumed_from": args.resume}
    print(json.dumps(meta, indent=2))
    t0 = time.time()
    try:
        model.learn(total_timesteps=cfg["total_timesteps"], callback=callbacks,
                    reset_num_timesteps=not args.resume, tb_log_name="ppo")
    except KeyboardInterrupt:
        print("\nInterrumpido: guardando el modelo actual...")
    finally:
        model.save(run_dir / "model.zip")
        meta["finished"] = datetime.now().isoformat(timespec="seconds")
        meta["train_seconds"] = round(time.time() - t0)
        meta["timesteps_done"] = int(model.num_timesteps)
        with open(run_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        venv.close()
        print(f"Modelo guardado en {run_dir / 'model.zip'}  ({meta['train_seconds']} s)")


if __name__ == "__main__":
    main()
