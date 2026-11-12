"""
Agente final: lo que se entrega al evaluador del profe.

Interfaz (lo que pide la sección 3.10 del proyecto):
    agent = Agent("models/config_b")   # 1. inicializar (carga modelo)
    agent.reset()                      # 2. reiniciar estado interno
    action = agent.act(frame)          # 3-4. recibe observación -> acción válida
    agent.close()                      # 5. finalizar

`frame` es el fotograma CRUDO del juego: numpy uint8 (240, 256, 3).
Todo el preprocesamiento, apilado de frames y frame-skip ocurre AQUÍ ADENTRO,
así el agente funciona aunque el evaluador solo le pase la imagen del juego.

Cuando salga el anexo técnico, solo hay que adaptar esta clase (o escribir un
pequeño adaptador encima) a los nombres de métodos/formatos que pida el profe.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .env import buttons_to_byte, get_action_set
from .preprocessing import FrameStacker


class Agent:
    def __init__(self, model_dir: str | Path, deterministic: bool = True,
                 frame_skip: int | None = None, seed: int | None = None, device: str = "cpu"):
        """
        model_dir      carpeta del run (con `model.zip` y `agent_config.json`, los genera
                       train.py) o un checkpoint .zip concreto de runs/<run>/checkpoints/
        deterministic  True: siempre la acción más probable. False: muestrea de la política.
        frame_skip     cada cuántos fotogramas decide. None = el mismo que en entrenamiento.
                       Si el evaluador YA aplica frame-skip por su cuenta, poner 1.
        """
        from stable_baselines3 import PPO

        path = Path(model_dir)
        if path.suffix == ".zip":                 # un checkpoint concreto
            model_file = path
            cfg_file = next(p / "agent_config.json" for p in (path.parent, path.parent.parent)
                            if (p / "agent_config.json").exists())
        else:                                     # carpeta del run
            model_file, cfg_file = path / "model.zip", path / "agent_config.json"
        with open(cfg_file) as f:
            self.cfg = json.load(f)

        self.model = PPO.load(model_file, device=device)
        self.deterministic = deterministic
        self.frame_skip = frame_skip if frame_skip is not None else self.cfg["frame_skip"]
        self.frame_size = self.cfg["frame_size"]
        self.n_stack = self.cfg["frame_stack"]
        self.action_names = get_action_set(self.cfg["action_set"])
        if seed is not None:
            self.model.set_random_seed(seed)
        self.reset()

    # ------------------------------------------------------------------ interfaz
    def reset(self) -> None:
        """Limpia la memoria de frames. Llamar al inicio de cada intento."""
        self._stacker = FrameStacker(self.n_stack, self.frame_size)
        self._step = 0
        self._last_action = 0

    def act(self, frame: np.ndarray) -> int:
        """Recibe un fotograma crudo y devuelve el índice de acción (0..n-1)."""
        # Solo decide cada `frame_skip` fotogramas; en los demás repite la acción
        # anterior (igual que el SkipFrame con el que se entrenó).
        if self._step % self.frame_skip == 0:
            obs = self._stacker.push(frame)                       # (84, 84, 4)
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            self._last_action = int(action)
        self._step += 1
        return self._last_action

    def close(self) -> None:
        pass

    # ------------------------------------------------------------- utilidades
    def action_buttons(self, action: int) -> list[str]:
        """Botones de una acción, p. ej. ['right', 'A']. Útil para adaptadores."""
        return self.action_names[action]

    def action_byte(self, action: int) -> int:
        """Acción como byte del control de NES, por si el evaluador lo pide así."""
        return buttons_to_byte(self.action_names[action])
