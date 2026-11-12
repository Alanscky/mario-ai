"""
Verificación de instalación. Es lo PRIMERO que corre cada integrante.

    python check_env.py

Comprueba:
  1. Que las librerías importan y en qué versiones.
  2. Que el juego corre (300 fotogramas con acciones al azar).
  3. Que el entorno de entrenamiento cumple la API de gymnasium (checker de SB3).
  4. Que el Agent ve EXACTAMENTE lo mismo que vio la red al entrenar
     (preprocesamiento + frame-skip + apilado de frames).
  5. Guarda results/check_frame.png para ver cómo queda la imagen preprocesada.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

OK, FAIL = "  [OK]", "  [FALLA]"


def step(title):
    print(f"\n== {title}")


def main():
    step("1. Librerías")
    import cv2, gymnasium, torch, stable_baselines3  # noqa: E401
    import gym_super_mario_bros  # noqa: F401
    print(f"  python {sys.version.split()[0]} | numpy {np.__version__} | torch {torch.__version__} | "
          f"sb3 {stable_baselines3.__version__} | gymnasium {gymnasium.__version__} | opencv {cv2.__version__}")
    print(f"  GPU CUDA disponible: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else " -> se entrenará en CPU"))
    if int(np.__version__.split(".")[0]) >= 2:
        print(FAIL, "numpy >= 2 rompe nes-py. Instala 'numpy<2'.")
        return 1

    from mario_ai.env import MarioGymnasium, make_train_env

    step("2. El juego corre")
    env = MarioGymnasium("SIMPLE_MOVEMENT")
    frame, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(300):
        frame, r, term, trunc, info = env.step(int(rng.integers(env.action_space.n)))
        if term or trunc:
            frame, _ = env.reset()
    env.close()
    print(OK, f"frame {frame.shape} {frame.dtype}, info de ejemplo: {info}")

    step("3. API gymnasium del entorno de entrenamiento")
    from stable_baselines3.common.env_checker import check_env
    cfg = {"action_set": "SIMPLE_MOVEMENT", "frame_skip": 4, "frame_size": 84, "frame_stack": 4,
           "max_frames": 2000, "reward": {"scale": 0.1}}
    tenv = make_train_env(cfg)
    check_env(tenv, warn=True)
    tenv.close()
    print(OK, f"observación {tenv.observation_space.shape}, acciones {tenv.action_space.n}")

    step("4. Agent ve lo mismo que en entrenamiento")
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    from mario_ai.env import SkipFrame
    from mario_ai.preprocessing import FrameStacker

    # Camino de entrenamiento: wrappers + VecFrameStack
    venv = VecFrameStack(DummyVecEnv([lambda: make_train_env(cfg)]), n_stack=4, channels_order="last")
    train_obs = [venv.reset()[0]]
    # Camino de evaluación: juego crudo, 1 fotograma por paso, apilado como en Agent
    raw = MarioGymnasium("SIMPLE_MOVEMENT")
    stacker = FrameStacker(4, 84)
    frame, _ = raw.reset()
    eval_obs = [stacker.push(frame)]

    actions = [int(a) for a in rng.integers(7, size=60)]
    for a in actions:
        o, _, done, _ = venv.step(np.array([a]))
        if done[0]:
            break
        train_obs.append(o[0])
        for _ in range(4):                   # el Agent repite la acción 4 fotogramas
            frame, *_ = raw.step(a)
        eval_obs.append(stacker.push(frame))
    venv.close(); raw.close()
    diffs = [int(np.abs(t.astype(int) - e.astype(int)).max()) for t, e in zip(train_obs, eval_obs)]
    if max(diffs) == 0:
        print(OK, f"{len(diffs)} observaciones idénticas entre entrenamiento y Agent")
    else:
        print(FAIL, f"diferencias máximas por paso: {diffs}")
        return 1

    step("5. Imagen preprocesada")
    import cv2
    Path("results").mkdir(exist_ok=True)
    last = eval_obs[-1]
    strip = np.concatenate([last[:, :, i] for i in range(last.shape[-1])], axis=1)
    cv2.imwrite("results/check_frame.png", cv2.resize(strip, None, fx=3, fy=3,
                                                     interpolation=cv2.INTER_NEAREST))
    print(OK, "results/check_frame.png (los 4 frames apilados que ve la red, del más viejo al más nuevo)")

    print("\nTodo listo. Siguiente paso: python train.py --config configs/smoke_test.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
