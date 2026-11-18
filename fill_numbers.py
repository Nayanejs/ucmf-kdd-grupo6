"""Gera paper/numbers.tex (tabelas e números do artigo) a partir de results/results.json."""
import json
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, ".")
import ucmf_pipeline as up  # noqa: E402

r = json.load(open("results/results.json"))
br = lambda x, d=3: f"{x:.{d}f}".replace(".", "{,}")
pc = lambda x: f"{100 * x:.1f}".replace(".", "{,}") + r"\%"

rows = []
best = r["melhor"]
NAME = {"Dummy (classe majoritaria)": "Classe majoritária", "Regressao Logistica": "Regressão Logística",
        "Random Forest": "Random Forest", "XGBoost": "XGBoost"}
for k, m in r["modelos"].items():
    t, ci = m["teste"], r["auc_ic95"][k]
    cells = [f"{br(m['cv_auc'][0])} $\\pm$ {br(m['cv_auc'][1])}",
             f"{br(t['ROC_AUC'])} [{br(ci[0])}; {br(ci[1])}]", br(t["PR_AUC"]), pc(t["Acuracia"]),
             pc(t["Acur_bal"]), pc(t["Sensib"]), pc(t["Especif"]), br(t["F1"])]
    nm = f"\\textbf{{{NAME[k]}}}" if k == best else NAME[k]
    rows.append(nm + " & " + " & ".join(cells) + r" \\")

def lrow(name, t):
    return f"{name} & {br(t['ROC_AUC'])} & {pc(t['Acuracia'])} & {pc(t['Sensib'])} & {pc(t['Especif'])} \\\\"

leak = [lrow("Regra: só SOPRO", r["regra_so_sopro"]),
        lrow("XGBoost + SOPRO", r["com_sopro"]),
        r"\midrule",
        lrow("XGBoost sem SOPRO, limiar 0{,}5", r["modelos"][best]["teste"]),
        lrow(f"XGBoost sem SOPRO, limiar {br(r['limiar_sens90'], 2)}", r["teste_limiar_sens90"])]

# Número de colunas após a codificação (pré-processador ajustado no treino)
df, _ = up.clean(pd.read_csv("data/UCMF_raw.csv", dtype=str))
X, y = df.drop(columns="y"), df["y"].values
Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=up.RANDOM_STATE)
ncols = up.make_preprocessor(up.NUM).fit(Xtr, ytr).transform(Xtr).shape[1]

d = r["dif_auc_xgb_rl"]
with open("paper/numbers.tex", "w") as fh:
    fh.write("\\newcommand{\\TABROWS}{" + "\n".join(rows) + "}\n")
    fh.write("\\newcommand{\\LEAKROWS}{" + "\n".join(leak) + "}\n")
    fh.write(f"\\newcommand{{\\NCOLS}}{{{ncols}}}\n")
    fh.write(f"\\newcommand{{\\DIFF}}{{{br(d[0])}}}\n")
    fh.write(f"\\newcommand{{\\DIFFCI}}{{{br(d[1])} a {br(d[2])}}}\n")
print(open("paper/numbers.tex").read())
