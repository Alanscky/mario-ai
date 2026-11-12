"""
Entorno de Super Mario Bros. 1-1 para el proyecto.

Problema que resuelve este archivo:
  - `gym-super-mario-bros` usa la librería vieja `gym` (API de 2021).
  - `stable-baselines3` (SB3) usa la librería nueva `gymnasium`.
  Aquí creamos el juego "crudo" y lo envolvemos en un `gymnasium.Env`
  para que SB3 lo acepte, sin depender de los wrappers de `gym`.

Capas (de adentro hacia afuera):
  MarioGymnasium      -> juego crudo, 1 paso = 1 fotograma, acciones discretas
  SkipFrame           -> repite la acción `skip` fotogramas (1 paso = 4 frames)
  Preprocess          -> RGB 240x256 -> gris 84x84 (misma función que usa Agent)
  ShapeReward         -> (opcional) bonificación por bandera / castigo por morir
  FrameLimit          -> corta el episodio al llegar al límite de fotogramas
El apilado de 4 frames se hace afuera con VecFrameStack (en train.py).
"""
from __future__ import annotations

import warnings

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .preprocessing import preprocess_frame

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Posición X aproximada del asta de la bandera en el 1-1 (para normalizar progreso).
# Ajustar si el anexo técnico del profe define otro valor.
FLAG_X_1_1 = 3161

# Bits del control de NES (igual que nes_py.JoypadSpace)
_BUTTONS = {
    "right": 0b10000000, "left": 0b01000000, "down": 0b00100000, "up": 0b00010000,
    "start": 0b00001000, "select": 0b00000100, "B": 0b00000010, "A": 0b00000001,
    "NOOP": 0b00000000,
}


def get_action_set(name: str) -> list[list[str]]:
    """Devuelve la lista de combinaciones de botones de un conjunto de acciones."""
    from gym_super_mario_bros import actions

    sets = {
        "RIGHT_ONLY": actions.RIGHT_ONLY,              # 5 acciones
        "SIMPLE_MOVEMENT": actions.SIMPLE_MOVEMENT,    # 7 acciones
        "COMPLEX_MOVEMENT": actions.COMPLEX_MOVEMENT,  # 12 acciones
    }
    if name not in sets:
        raise ValueError(f"Conjunto de acciones desconocido: {name}. Opciones: {list(sets)}")
    return sets[name]


def buttons_to_byte(combo: list[str]) -> int:
    value = 0
    for b in combo:
        value |= _BUTTONS[b]
    return value


def make_raw_mario(world: int = 1, stage: int = 1):
    """Juego crudo (API vieja de gym): reset() -> obs, step() -> (obs, r, done, info)."""
    from gym_super_mario_bros import SuperMarioBrosEnv

    # target=(1,1) => "single stage": el episodio termina al morir o al tocar la bandera
    return SuperMarioBrosEnv(rom_mode="vanilla", target=(world, stage))


class MarioGymnasium(gym.Env):
    """Adaptador del juego crudo a gymnasium. 1 paso = 1 fotograma."""

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, action_set: str = "SIMPLE_MOVEMENT", render_mode: str | None = None):
        super().__init__()
        self._env = make_raw_mario()
        self.action_names = get_action_set(action_set)
        self._action_bytes = [buttons_to_byte(c) for c in self.action_names]
        self.action_space = spaces.Discrete(len(self._action_bytes))
        self.observation_space = spaces.Box(0, 255, (240, 256, 3), dtype=np.uint8)
        self.render_mode = render_mode
        self._last_obs = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._env.reset()
        self._last_obs = obs
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self._env.step(self._action_bytes[int(action)])
        self._last_obs = obs
        # En 1-1 "done" siempre es por un evento del juego (morir o bandera) -> terminated
        return obs, float(reward), bool(done), False, info

    def render(self):
        if self.render_mode == "human":
            return self._env.render(mode="human")
        return self._last_obs

    def close(self):
        self._env.close()


class SkipFrame(gym.Wrapper):
    """Repite la misma acción `skip` fotogramas y suma la recompensa.
    Lleva la cuenta total de fotogramas en info['frames']."""

    def __init__(self, env, skip: int = 4):
        super().__init__(env)
        self.skip = skip
        self.frames = 0

    def reset(self, **kwargs):
        self.frames = 0
        obs, info = self.env.reset(**kwargs)
        info["frames"] = 0
        return obs, info

    def step(self, action):
        total = 0.0
        for _ in range(self.skip):
            obs, r, terminated, truncated, info = self.env.step(action)
            self.frames += 1
            total += r
            if terminated or truncated:
                break
        info["frames"] = self.frames
        return obs, total, terminated, truncated, info


class Preprocess(gym.ObservationWrapper):
    """RGB (240,256,3) -> gris (84,84,1) usando la MISMA función que el Agent."""

    def __init__(self, env, size: int = 84):
        super().__init__(env)
        self.size = size
        self.observation_space = spaces.Box(0, 255, (size, size, 1), dtype=np.uint8)

    def observation(self, obs):
        return preprocess_frame(obs, self.size)


class ShapeReward(gym.Wrapper):
    """Modificaciones a la recompensa (configuración B).

    La recompensa original del entorno por fotograma es:
        v (avance en X) + c (castigo por reloj) + d (-15 al morir), recortada a [-15, 15]
    Aquí se puede:
        - multiplicar todo por `scale`
        - sumar `flag_bonus` al tocar la bandera
        - sumar `death_penalty` (negativo) al morir
    """

    def __init__(self, env, scale: float = 1.0, flag_bonus: float = 0.0, death_penalty: float = 0.0):
        super().__init__(env)
        self.scale = scale
        self.flag_bonus = flag_bonus
        self.death_penalty = death_penalty

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        if terminated:
            if info.get("flag_get", False):
                r += self.flag_bonus
            else:
                r += self.death_penalty
        return obs, r * self.scale, terminated, truncated, info


class FrameLimit(gym.Wrapper):
    """Trunca el episodio si se rebasa el límite de fotogramas (como hará la competencia)."""

    def __init__(self, env, max_frames: int):
        super().__init__(env)
        self.max_frames = max_frames

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        if info.get("frames", 0) >= self.max_frames and not terminated:
            truncated = True
        return obs, r, terminated, truncated, info


def make_train_env(cfg: dict, render_mode: str | None = None) -> gym.Env:
    """Construye el entorno completo de entrenamiento a partir de la configuración."""
    env = MarioGymnasium(cfg["action_set"], render_mode=render_mode)
    env = SkipFrame(env, cfg["frame_skip"])
    env = Preprocess(env, cfg["frame_size"])
    r = cfg.get("reward", {})
    env = ShapeReward(env, r.get("scale", 1.0), r.get("flag_bonus", 0.0), r.get("death_penalty", 0.0))
    env = FrameLimit(env, cfg["max_frames"])
    return env
