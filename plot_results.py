"""
Genera las gráficas del avance / reporte a partir de lo que ya está en el repo.

    python plot_results.py

Lee:
    models/<config>/episodes.csv          -> curvas de aprendizaje (entrenamiento)
    results/<config>_100/episodes.csv     -> evaluación de 100 intentos
Escribe en results/figuras/:
    1_aprendizaje.png      progreso y % de banderas durante el entrenamiento
    2_comparacion.png      finalización, puntos y tiempo de cada configuración
    3_muertes.png          dónde muere cada configuración (análisis de errores)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("results/figuras")
FLAG_X = 3161

# Un color fijo por configuración (C refinada es la misma entidad que C)
COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
TEXT, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"

# nombre en la gráfica -> (carpetas de entrenamiento en models/, carpeta de evaluación en results/)
CONFIGS = {
    "A":              (["config_a"],                "config_a_100"),
    "B":              (["config_b"],                "config_b_100"),
    "C":              (["config_c"],                "config_c_100"),
    "C refinada (4M)": (["config_c", "config_c_4m"], "config_c_4m_100"),
}


def color(name: str) -> str:
    return COLORS[name[0]]


def style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, loc="left", fontsize=11, color=TEXT, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def load_training(folders):
    parts = [pd.read_csv(f"models/{f}/episodes.csv") for f in folders if Path(f"models/{f}/episodes.csv").exists()]
    if not parts:
        return None
    return pd.concat(parts).sort_values("timesteps")


def plot_learning():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in ("A", "B", "C refinada (4M)"):
        d = load_training(CONFIGS[name][0])
        if d is None:
            continue
        label = "C" if name.startswith("C") else name
        roll = d.rolling(300, min_periods=50)
        x = d["timesteps"] / 1e6
        axes[0].plot(x, roll["x_pos"].mean() / FLAG_X * 100, color=color(name), lw=2, label=label)
        axes[1].plot(x, roll["flag_get"].mean() * 100, color=color(name), lw=2, label=label)
    for ax, t, y in ((axes[0], "Progreso promedio en el nivel", "% del nivel (media móvil de 300 episodios)"),
                     (axes[1], "Episodios que llegan a la bandera", "% de episodios (media móvil de 300)")):
        style(ax, t, "Pasos de entrenamiento (millones)", y)
        ax.set_ylim(0, 100)
        ax.axvline(2, color=MUTED, lw=1, ls=":")
        ax.legend(frameon=False, fontsize=9)
    axes[1].text(2.05, 4, "A y B terminan; C sigue (refinamiento)", color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "1_aprendizaje.png", dpi=160)
    plt.close(fig)


def eval_rows():
    rows = []
    for name, (_, ev) in CONFIGS.items():
        p = Path(f"results/{ev}/episodes.csv")
        if not p.exists():
            continue
        d = pd.read_csv(p)
        done = d[d["completed"] == 1]
        rows.append({"name": name, "completion": d["completed"].mean() * 100,
                     "score": done["score"].mean(), "frames": done["frames"].median(), "data": d})
    return rows


def plot_comparison(rows):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    metrics = (("completion", "Completan el nivel", "% de 100 intentos", "{:.0f}%"),
               ("score", "Puntos al completar", "promedio", "{:.0f}"),
               ("frames", "Tiempo al completar", "fotogramas (mediana, menos es mejor)", "{:.0f}"))
    names = [r["name"] for r in rows]
    for ax, (key, title, ylab, fmt) in zip(axes, metrics):
        vals = [r[key] for r in rows]
        if key == "frames":
            # Diferencias de pocos fotogramas: puntos (no barras) para no exagerarlas con un eje recortado
            ax.scatter(names, vals, s=110, color=[color(n) for n in names], edgecolor="white",
                       linewidth=2, zorder=3)
            for n, v in zip(names, vals):
                ax.text(n, v + 1.5, f"{v:.0f}\n({v / 60:.2f} s)", ha="center", va="bottom",
                        fontsize=8, color=TEXT)
            ax.set_ylim(min(vals) - 10, max(vals) + 15)
        else:
            bars = ax.bar(names, vals, color=[color(n) for n in names], width=0.6,
                          edgecolor="white", linewidth=2)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v),
                        ha="center", va="bottom", fontsize=9, color=TEXT)
            ax.set_ylim(0, max(vals) * 1.15)
        style(ax, title, ylabel=ylab)
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "2_comparacion.png", dpi=160)
    plt.close(fig)


def plot_deaths(rows):
    fig, ax = plt.subplots(figsize=(11, 3.2))
    for i, r in enumerate(rows):
        dead = r["data"][r["data"]["completed"] == 0]
        jitter = (np.random.default_rng(i).random(len(dead)) - 0.5) * 0.35
        ax.scatter(dead["x_end"], i + jitter, s=60, color=color(r["name"]), alpha=0.7,
                   edgecolor="white", linewidth=2, zorder=3)
        ax.text(FLAG_X + 40, i, f"{len(dead)} muertes", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(rows)), [r["name"] for r in rows])
    ax.set_xlim(0, FLAG_X + 300)
    ax.axvline(FLAG_X, color=MUTED, lw=1, ls=":")
    ax.text(FLAG_X, -0.55, "bandera", ha="center", va="bottom", fontsize=8, color=MUTED)
    style(ax, "Dónde muere Mario (cada punto es un intento fallido de 100)", "Posición en el nivel (x)")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / "3_muertes.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plot_learning()
    rows = eval_rows()
    if rows:
        plot_comparison(rows)
        plot_deaths(rows)
        print("\nResumen de la evaluación (100 intentos):")
        print(pd.DataFrame([{k: v for k, v in r.items() if k != "data"} for r in rows])
              .round(1).to_string(index=False))
    print(f"\nGráficas guardadas en {OUT}/")


if __name__ == "__main__":
    main()
