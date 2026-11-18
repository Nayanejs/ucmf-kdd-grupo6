"""Gera as figuras do artigo a partir de results/results.json."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = sys.argv[2] if len(sys.argv) > 2 else "paper"
r = json.load(open(f"{RES}/results.json"))

BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85", "#1f1f1e", "#6b6b66"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": INK, "pdf.fonttype": 42})
PREV = r["prev_anormal"] * 100


def hbars(ax, labels, vals, ns, title):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=BLUE, height=0.62, edgecolor="white", linewidth=1)
    for yi, v, n in zip(y, vals, ns):
        ax.text(v + 2, yi, f"{v:.0f}%", va="center", fontsize=6.5, color=INK)
    ax.axvline(PREV, color=GRAY, ls="--", lw=0.8)
    ax.set_yticks(y, [f"{l} (n={n:,})".replace(",", ".") for l, n in zip(labels, ns)], fontsize=6.5)
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("% de anormais")
    ax.set_title(title, loc="left", color=INK)
    ax.tick_params(axis="y", length=0)


# Figura 1: vazamento (SOPRO) e prevalência por faixa etária
fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 1.55), gridspec_kw={"wspace": 1.25})
s = r["eda"]["sopro"]
lab = [("Ausente", "Ausente"), ("Sistolico", "Sistólico"), ("Outro", "Outros")]
hbars(a, [l for _, l in lab], [s[k][1] * 100 for k, _ in lab], [s[k][0] for k, _ in lab], "(a) SOPRO")
f = r["eda"]["faixa"]
ks = ["<1", "1-4", "5-9", "10-14", "15-19"]
hbars(b, [f"{k} a" for k in ks], [f[k][1] * 100 for k in ks], [f[k][0] for k in ks], "(b) Faixa etária")
fig.savefig(f"{OUT}/fig_eda.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig_eda.png", dpi=300, bbox_inches="tight")

# Figura 2: curvas ROC + matriz de confusão do melhor modelo
fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 1.75), gridspec_kw={"wspace": 0.55, "width_ratios": [1.15, 1]})
names = [("Regressao Logistica", "Reg. Logística", BLUE), ("Random Forest", "Random Forest", ORANGE),
         ("XGBoost", "XGBoost", AQUA)]
for k, lbl, c in names:
    fpr, tpr = r["roc"][k]
    auc = r["modelos"][k]["teste"]["ROC_AUC"]
    a.plot(fpr, tpr, color=c, lw=1.4, label=f"{lbl} ({auc:.3f})".replace(".", ","))
a.plot([0, 1], [0, 1], color=GRAY, lw=0.8, ls="--")
a.set_xlabel("1 − Especificidade")
a.set_ylabel("Sensibilidade")
a.set_title("(a) Curvas ROC (teste)", loc="left")
a.legend(frameon=False, loc="lower right", handlelength=1.0, fontsize=5.6, borderaxespad=0.1)
a.set_xlim(0, 1); a.set_ylim(0, 1.01)
m = r["modelos"][r["melhor"]]["teste"]
cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
rowpct = cm / cm.sum(1, keepdims=True)
b.imshow(rowpct, cmap="Blues", vmin=0, vmax=1.15)
for i in range(2):
    for j in range(2):
        b.text(j, i, f"{cm[i, j]}\n({rowpct[i, j] * 100:.0f}%)", ha="center", va="center", fontsize=7,
               color="white" if rowpct[i, j] > 0.6 else INK)
b.set_xticks([0, 1], ["Normal", "Anormal"]); b.set_yticks([0, 1], ["Normal", "Anormal"])
b.set_xlabel("Predito"); b.set_ylabel("Real")
b.set_title("(b) XGBoost, limiar 0,5", loc="left")
for sp in b.spines.values():
    sp.set_visible(False)
b.tick_params(length=0)
fig.savefig(f"{OUT}/fig_roc_cm.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig_roc_cm.png", dpi=300, bbox_inches="tight")

# Figura 3: importância por permutação
NICE = {"MOTIVO2": "MOTIVO2", "B2": "B2", "MOTIVO1": "MOTIVO1", "FC": "FC", "HDA_INFORMADO": "HDA informada",
        "IDADE": "IDADE", "Convenio": "Convênio", "LOG_PESO": "Peso (log)", "HDA_ASSINTOMATICO": "HDA: assintomático",
        "SEXO": "SEXO", "PAS": "PA sistólica", "HDA_PALPITACAO": "HDA: palpitação", "LOG_IMC": "IMC (log)",
        "PULSOS_ALTERADO": "Pulsos alterados"}
imp = list(r["importancia"].items())[:10]
fig, ax = plt.subplots(figsize=(3.5, 1.75))
y = np.arange(len(imp))[::-1]
vals = [v[0] * 100 for _, v in imp]
errs = [v[1] * 100 for _, v in imp]
ax.barh(y, vals, xerr=errs, color=BLUE, height=0.62, edgecolor="white", linewidth=1,
        error_kw={"ecolor": MUTED, "elinewidth": 0.7, "capsize": 1.5})
ax.set_yticks(y, [NICE.get(k, k) for k, _ in imp])
ax.set_xlabel("Queda da AUC ao permutar o atributo (×100)")
ax.tick_params(axis="y", length=0)
ax.xaxis.grid(True, color="#e6e6e3", lw=0.6); ax.set_axisbelow(True)
fig.savefig(f"{OUT}/fig_imp.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig_imp.png", dpi=300, bbox_inches="tight")
print("ok")
