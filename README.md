# Mario AI: agente PPO para Super Mario Bros. 1-1

Agente autónomo que juega el Mundo 1-1 usando **aprendizaje por refuerzo (PPO)**
con observaciones de imagen (84×84 gris, 4 frames apilados).
Proyecto para el *Mario AI Challenge* del curso de Inteligencia Artificial.

---

## 1. Estructura

```
mario-ai/
├── mario_ai/                # paquete del agente
│   ├── env.py               # juego + wrappers (frame-skip, recompensa, límite)
│   ├── preprocessing.py     # imagen -> 84x84 gris y apilado de frames
│   └── agent.py             # clase Agent: LO QUE SE ENTREGA AL EVALUADOR
├── configs/
│   ├── config_a.yaml        # configuración A (base)
│   ├── config_b.yaml        # configuración B (comparación)
│   └── smoke_test.yaml      # prueba rápida del pipeline
├── check_env.py             # verifica instalación (correr primero)
├── train.py                 # entrena
├── evaluate.py              # evalúa como el evaluador del profe (métricas + video)
├── notebooks/
│   └── colab_train.ipynb    # entrenar en Google Colab con GPU
├── runs/                    # salidas de entrenamiento (NO se suben a git)
├── models/                  # modelos FINALES que sí se suben
└── results/                 # métricas de evaluación
```

## 2. Instalación

> **Requisito clave:** Python 3.10 y `numpy<2`. `nes-py` (el emulador) **truena con
> NumPy 2** (`OverflowError: ... out of bounds for uint8`). Probado también con Python 3.12.

### Linux (Fedora / Mint) y macOS
```bash
git clone https://github.com/USUARIO/mario-ai.git
cd mario-ai
# Fedora: el emulador se compila, hacen falta compilador y headers de Python
sudo dnf install -y gcc-c++ python3.10 python3.10-devel     # Mint/Ubuntu: build-essential python3.10-dev python3.10-venv
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```
Luego **PyTorch según tu hardware** (antes del resto):
```bash
# Con GPU NVIDIA:
pip install torch==2.3.1
# Sin GPU NVIDIA (incluye GPUs AMD viejas tipo RX 470/570/580: PyTorch ya no las soporta):
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
```
Y el resto:
```bash
pip install -r requirements.txt
python check_env.py
```

> Si `python3.10` no está en los repos, usa [uv](https://docs.astral.sh/uv/):
> `uv python install 3.10 && uv venv -p 3.10 venv`.

### Windows
1. Instala Python 3.10 (python.org) y **"Desktop development with C++"** de
   Visual Studio Build Tools (nes-py se compila).
2. En PowerShell:
   ```powershell
   py -3.10 -m venv venv
   venv\Scripts\activate
   pip install torch==2.3.1            # o la versión CPU, ver arriba
   pip install -r requirements.txt
   python check_env.py
   ```
   Si da error con `SubprocVecEnv`, entrena con `--n-envs 1` o pon el código dentro de `if __name__ == "__main__":` (ya está así en train.py).

### Google Colab
Abrir `notebooks/colab_train.ipynb` y seguir las celdas (incluye reinicio de sesión para `numpy<2`).

### ¿Funcionó?
`python check_env.py` debe terminar con `Todo listo.` y mostrar
`[OK] N observaciones idénticas entre entrenamiento y Agent`.

## 3. Uso

```bash
# Prueba del pipeline completo (~1 min): entrena poquito, guarda, evalúa
python train.py --config configs/smoke_test.yaml --run-dir runs/smoke
python evaluate.py --model runs/smoke --episodes 3 --video results/smoke.mp4

# Entrenamiento real
python train.py --config configs/config_a.yaml
tensorboard --logdir runs                 # ver progreso en http://localhost:6006

# Reanudar tras un corte (Colab, apagón, Ctrl+C)
python train.py --config configs/config_a.yaml --run-dir runs/config_a_XXXX \
                --resume runs/config_a_XXXX/checkpoints/ppo_800000_steps.zip

# Evaluación formal (30 intentos, política estocástica con semilla por intento)
python evaluate.py --model runs/config_a_XXXX --episodes 30 --label config_a
# Un intento determinista + video
python evaluate.py --model runs/config_a_XXXX --deterministic --episodes 1 --video results/a.mp4
# Ver jugar en una ventana (PC local con pantalla)
python evaluate.py --model runs/config_a_XXXX --deterministic --episodes 1 --render
```

`Ctrl+C` durante el entrenamiento **guarda el modelo** antes de salir.

## 4. Cómo funciona

### Ciclo del agente (sección 3.5)
```
fotograma crudo (240x256 RGB) ──► Agent.act()
                                   ├─ ¿toca decidir? (cada 4 fotogramas)
                                   │    ├─ gris + 84x84
                                   │    ├─ apilar con los 3 anteriores -> (84,84,4)
                                   │    └─ red CNN (PPO) -> acción
                                   └─ si no, repetir la acción anterior
                              ◄── índice de acción (0..n-1)
```

### Formalización (resumen, para el reporte)
| Elemento | Definición |
|---|---|
| Estado | últimos 4 frames en gris 84×84 (el movimiento se infiere de la diferencia entre frames) |
| Acciones | `RIGHT_ONLY` (5) en A, `SIMPLE_MOVEMENT` (7) en B |
| Transición | el emulador; el agente decide cada 4 fotogramas |
| Recompensa | avance en X − castigo por reloj − 15 al morir (del entorno) + shaping en B |
| Éxito | tocar la bandera (`flag_get`) |
| Fracaso | morir (enemigo, hueco, tiempo) o llegar al límite de fotogramas |
| Política | red convolucional (CnnPolicy de SB3) entrenada con PPO |

### Configuraciones a comparar (sección 3.7)
| | A (base) | B |
|---|---|---|
| Acciones | RIGHT_ONLY (5) | SIMPLE_MOVEMENT (7) |
| Bono bandera | 0 | +500 (×0.1) |
| Castigo muerte extra | 0 | −250 (×0.1) |
| Todo lo demás | igual | igual |

## 5. Salidas

**`runs/<run>/`** (entrenamiento): `model.zip`, `agent_config.json`, `checkpoints/`,
`episodes.csv` (un renglón por episodio: recompensa, pasos, fotogramas, x, bandera),
`tb/` (TensorBoard), `config.yaml`, `meta.json` (hardware, versiones, duración: la rúbrica pide documentar recursos).

**`results/<label>/`** (evaluación): `episodes.csv` y `summary.json` con:
tasa de finalización, x máximo, progreso normalizado (x / 3161), fotogramas
(media y mejor en intentos completados) y causas de fin.

> La causa `caida` vs `enemigo` es una heurística por la altura de Mario al morir.
> Para el análisis de errores conviene también graficar `x_end` (dónde muere).

## 6. Modelos finales

`runs/` no se sube a git (pesa). Cuando un modelo sea el bueno:
```bash
mkdir -p models/config_a
cp runs/config_a_XXXX/{model.zip,agent_config.json,config.yaml,meta.json} models/config_a/
```
Un `model.zip` de PPO pesa ~20 MB (GitHub acepta hasta 100 MB por archivo).

## 7. Integración con el evaluador del profe

La interfaz pedida en la sección 3.10 ya existe en `mario_ai/agent.py`:

```python
from mario_ai.agent import Agent
agent = Agent("models/config_b")   # inicializar / cargar
agent.reset()                      # reiniciar estado interno
action = agent.act(frame)          # observación cruda -> acción
agent.close()                      # finalizar
```
Cuando salga el anexo técnico, revisar:
- **Formato de la observación.** Si no es el RGB 240×256, adaptar `preprocess_frame`; **cambiar el preprocesamiento obliga a reentrenar**.
- **Si el evaluador ya aplica frame-skip**, usar `Agent(..., frame_skip=1)`.
- **Formato de la acción.** Si pide índice de otro conjunto o botones, usar `agent.action_buttons(a)` / `agent.action_byte(a)`.
- **Límite de fotogramas.** Actualizar `max_frames` en configs y `--max-frames` en evaluate.

## 8. Entorno probado

| | versión |
|---|---|
| Python | 3.10.20 (también 3.12 con numpy 1.26) |
| numpy | 1.26.4 |
| gym / gym-super-mario-bros / nes-py | 0.26.2 / 7.4.0 / 8.2.1 |
| gymnasium | 0.29.1 |
| stable-baselines3 | 2.3.2 |
| torch | 2.3.1 |

Al importar aparece un aviso de que *Gym está sin mantenimiento*; es normal, se
puede ignorar (solo se usa para el emulador; el entrenamiento usa gymnasium).
