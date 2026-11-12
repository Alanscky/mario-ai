"""
Preprocesamiento de imagen. Se usa EXACTAMENTE igual en entrenamiento (env.py)
y en evaluación (agent.py). Si el agente ve imágenes distintas a las que vio al
entrenar, falla, así que cualquier cambio aquí obliga a re-entrenar.
"""
import cv2
import numpy as np


def preprocess_frame(frame: np.ndarray, size: int = 84) -> np.ndarray:
    """RGB uint8 (240, 256, 3) -> escala de grises uint8 (size, size, 1)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return small[:, :, None]


class FrameStacker:
    """Apila los últimos `n_stack` frames preprocesados -> (size, size, n_stack).

    Reproduce el comportamiento de VecFrameStack(channels_order="last") de SB3:
    al reiniciar, la pila se llena de ceros y el primer frame queda al final.
    """

    def __init__(self, n_stack: int = 4, size: int = 84):
        self.n_stack = n_stack
        self.size = size
        self.reset()

    def reset(self) -> None:
        self._stack = np.zeros((self.size, self.size, self.n_stack), dtype=np.uint8)

    def push(self, raw_frame: np.ndarray) -> np.ndarray:
        self._stack = np.roll(self._stack, -1, axis=-1)
        self._stack[:, :, -1:] = preprocess_frame(raw_frame, self.size)
        return self._stack.copy()
