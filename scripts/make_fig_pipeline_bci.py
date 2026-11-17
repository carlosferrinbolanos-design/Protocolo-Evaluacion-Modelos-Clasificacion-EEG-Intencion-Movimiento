"""Figura: pipeline de una BCI motora para el control de una órtesis de mano (solo español).
Resalta lo que evalúa el artículo (evaluación offline de los pasos 3 a 5) y marca lo que queda fuera del alcance.
Uso: python scripts/make_fig_pipeline_bci.py  -> figures/Figura_pipeline_BCI.png (300 dpi) y .svg."""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
INK = "#333333"; EVAL_FC = "#fff4e0"; EVAL_EC = "#d9822b"; OUT_EC = "#8a8a8a"

fig, ax = plt.subplots(figsize=(5.9, 1.98), dpi=300)   # ~15 cm de ancho y ~5.0 cm de alto
ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")
TF, BF = 5.1, 4.6                                        # tamaño de título y de texto (pt)

def box(x, y, w, h, title, body, fc, ec=INK, ls="-", tcol="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec, lw=0.7, ls=ls))
    ax.text(x + w/2, y + h - 1.2, title, ha="center", va="top", fontsize=TF, fontweight="bold", color=tcol, linespacing=1.1)
    nt = title.count("\n") + 1
    ax.text(x + w/2, y + h - 1.9 - 2.75*nt, body, ha="center", va="top", fontsize=BF, color=tcol, linespacing=1.25)

def arrow(x0, y0, x1, y1, color=INK, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=color, linestyle=ls, shrinkA=0, shrinkB=0, mutation_scale=6))

# --- columnas (x, ancho) ---
C = {1: (0.6, 11.6), 2: (15.2, 12.4), 3: (30.6, 12.8), 4: (46.4, 20.6), 5: (69.8, 13.5), 6: (86.3, 13.1)}
Y0, H = 7.8, 23.0            # caja estándar
ymid = Y0 + H/2

# zona evaluada (pasos 3 a 5)
ex0 = C[3][0] - 1.4; ex1 = C[5][0] + C[5][1] + 1.2
ax.add_patch(FancyBboxPatch((ex0, 5.0), ex1 - ex0, 36.3, boxstyle="round,pad=0.2,rounding_size=1.5",
                            fc=EVAL_FC, ec=EVAL_EC, lw=1.0, ls=(0, (4, 2))))
ax.text((ex0 + ex1)/2, 39.5, "Evaluado en este trabajo (offline, pasos 3 a 5)", ha="center", va="center", fontsize=TF + 0.3,
        fontweight="bold", color="#a35a12")
ax.text((ex0 + ex1)/2, 35.3, "partición disjunta por participante · control de fuga de información\nmétricas (BA, MCC, kappa) · prueba global y diagrama DC",
        ha="center", va="center", fontsize=BF - 0.1, color="#a35a12", linespacing=1.25)

# fuera del alcance (paso 6 y lazo cerrado)
ax.text(C[6][0] + C[6][1]/2, 36.6, "Fuera del alcance:\nvalidación en línea\ncon la órtesis", ha="center", va="center",
        fontsize=BF, color="#555555", style="italic", linespacing=1.2)

x, w = C[1]; box(x, Y0, w, H, "1. Adquisición\ndel EEG", "Electrodos o casco;\nseñal multicanal", "#dbe9f6")
t = np.linspace(0, 1, 80)                                   # trazas EEG esquemáticas
for k in range(4):
    yy = Y0 + 1.8 + k*1.6 + 0.5*np.sin(2*np.pi*(6 + 3*k)*t + k) * (0.6 + 0.4*np.sin(7*t + k))
    ax.plot(x + 1.4 + t*(w - 2.8), yy, lw=0.4, color="#2b5d8a")
x, w = C[2]; box(x, Y0, w, H, "2. Preprocesa-\nmiento", "Filtrado pasabanda\nre-referencia\nremoción de\nartefactos", "#dbe9f6")
x, w = C[3]; box(x, Y0, w, H, "3. Segmentación\nen épocas", "Ventana alrededor\ndel evento o de la\nintención de\nmovimiento", "#e3f1df")
x, w = C[4]; hh = H/2 - 0.8
box(x, Y0 + H - hh, w, hh, "4a. Características\n+ clasificador", "CSP, FBCSP o covarianzas (geometría\nde Riemann) → LDA, SVM o MDM", "#fdf0d5")
box(x, Y0, w, hh, "4b. Aprendizaje profundo\nde extremo a extremo", "EEGNet, ShallowConvNet,\nDeepConvNet", "#fdf0d5")
x, w = C[5]; box(x, Y0, w, H, "5. Decodificación\nde la intención", "Clase de\nmovimiento\nde mano", "#f6dede")
x, w = C[6]; box(x, Y0, w, H, "6. Comando a la\nórtesis de mano", "Control en línea\ndel actuador", "#eeeeee", ec=OUT_EC, ls=(0, (3, 2)), tcol="#444444")

# flechas principales
for a, b in [(1, 2), (2, 3)]:
    arrow(C[a][0] + C[a][1] + 0.4, ymid, C[b][0] - 0.4, ymid)
xs = C[3][0] + C[3][1] + 0.4; xb = C[4][0] - 0.4
y4a = Y0 + H - hh/2; y4b = Y0 + hh/2
ax.plot([xs, xs + 0.8], [ymid, ymid], color=INK, lw=0.8)
arrow(xs + 0.8, ymid, xb, y4a); arrow(xs + 0.8, ymid, xb, y4b)
xe = C[4][0] + C[4][1] + 0.4; x5 = C[5][0] - 0.4
arrow(xe, y4a, x5, ymid + 0.6); arrow(xe, y4b, x5, ymid - 0.6)
arrow(C[5][0] + C[5][1] + 0.4, ymid, C[6][0] - 0.4, ymid, color=OUT_EC, ls=(0, (3, 2)))

# lazo cerrado: retroalimentación al usuario
xr = C[6][0] + C[6][1]/2; xl = C[1][0] + C[1][1]/2; yb = 1.6
ax.plot([xr, xr], [Y0 - 0.4, yb], color=OUT_EC, lw=0.8, ls=(0, (3, 2)))
ax.plot([xr, xl], [yb, yb], color=OUT_EC, lw=0.8, ls=(0, (3, 2)))
arrow(xl, yb, xl, Y0 - 0.4, color=OUT_EC, ls=(0, (3, 2)))
ax.text((xr + xl)/2, yb, "  Retroalimentación al usuario (lazo cerrado, en línea)  ", ha="center", va="center",
        fontsize=BF, color="#555555", style="italic", bbox=dict(fc="white", ec="none", pad=0.3))

plt.subplots_adjust(0.005, 0.005, 0.995, 0.995)
for e in ("png", "svg"):
    fig.savefig(os.path.join(OUT, f"Figura_pipeline_BCI.{e}"), dpi=300)
print("ok", os.path.join(OUT, "Figura_pipeline_BCI.png"))
