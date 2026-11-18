"""UCMF - Classificação Normal x Anormal (Mineração de Dados, UFG - Grupo 6).

Pipeline completo e reprodutível (RANDOM_STATE = 42):
  1. Limpeza determinística linha a linha (não usa estatísticas do conjunto).
  2. Divisão estratificada treino/teste (80/20).
  3. Transformações ajustadas SOMENTE no treino (dentro do Pipeline do scikit-learn):
     detecção de outliers por faixa etária, imputação, codificação e padronização.
  4. Ajuste de hiperparâmetros por validação cruzada estratificada (5 folds) no treino.
  5. Avaliação final no conjunto de teste, análise de erros e importância dos atributos.

O atributo SOPRO é EXCLUÍDO dos modelos (data leakage); ele só é usado no
experimento de controle que mede o efeito do vazamento.
"""
import json
import re
import sys
import unicodedata
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
DATA = sys.argv[1] if len(sys.argv) > 1 else "data/UCMF_raw.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"

# ---------------------------------------------------------------------------
# 1. Limpeza determinística (regras fixas, sem estatísticas do conjunto)
# ---------------------------------------------------------------------------

def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def parse_date(x):
    """dd/mm/aa ou dd/mm/aaaa; número de série do Excel; demais -> NaT.
    Ano com 2 dígitos: <= 10 -> 20xx, senão 19xx."""
    if pd.isna(x):
        return pd.NaT
    x = str(x).strip()
    if re.fullmatch(r"\d{5}", x):  # serial do Excel
        return pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(x))
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})", x)
    if not m:
        return pd.NaT
    d, mo, y = map(int, m.groups())
    if y < 100:
        y = 2000 + y if y <= 10 else 1900 + y
    try:
        return pd.Timestamp(year=y, month=mo, day=d)
    except ValueError:
        return pd.NaT


def parse_fc(x):
    """Texto -> número; faixas ('80-90', '156 a 168') -> ponto médio."""
    if pd.isna(x):
        return np.nan
    x = str(x).strip().replace(",", ".")
    try:
        return float(x)
    except ValueError:
        nums = re.findall(r"\d+(?:\.\d+)?", x)
        return np.mean([float(n) for n in nums[:2]]) if nums else np.nan


HDA_MAP = {"dispneia": "DISPNEIA", "dor precordial": "DOR_PRECORDIAL", "palpitacao": "PALPITACAO",
           "cianose": "CIANOSE", "desmaio/tontura": "DESMAIO", "ganho de peso": "GANHO_PESO",
           "outro": "OUTRO", "assintomatico": "ASSINTOMATICO"}


def clean(raw: pd.DataFrame, log=None):
    log = {} if log is None else log
    df = raw.copy()
    log["bruto"] = len(df)

    # Alvo: unifica grafias e remove linhas sem rótulo
    t = df["NORMAL X ANORMAL"].fillna("").astype(str).str.strip().str.lower()
    df["y"] = np.where(t.str.startswith("anorm"), 1, np.where(t.str.startswith("norm"), 0, -1))
    log["sem_rotulo"] = int((df["y"] == -1).sum())
    df = df[df["y"] != -1]

    # Duplicatas que diferem só no ID
    cols = [c for c in df.columns if c != "ID"]
    log["duplicatas"] = int(df.duplicated(subset=cols).sum())
    df = df.drop_duplicates(subset=cols).drop(columns="ID")

    # Datas -> apenas para validar/reconstruir a IDADE
    at, dn = df["Atendimento"].map(parse_date), df["DN"].map(parse_date)
    idade_datas = (at - dn).dt.days / 365.25
    idade_datas = idade_datas.where((idade_datas >= 0) & (idade_datas < 20))
    idade = pd.to_numeric(df["IDADE"], errors="coerce")
    seculo = idade + 100  # erro de século (ex.: -95,2 anos)
    idade = np.where(idade >= 0, idade,
                     np.where(idade_datas.notna(), idade_datas,
                              np.where((seculo >= 0) & (seculo < 20), seculo, np.nan)))
    idade = pd.Series(idade, index=df.index)
    idade = idade.fillna(idade_datas)
    df["IDADE"] = idade
    log["acima_19"] = int((idade >= 20).sum())
    df = df[~(idade >= 20)]

    # Antropometria: limites físicos (zeros e negativos viram ausentes)
    peso = pd.to_numeric(df["Peso"], errors="coerce")
    alt = pd.to_numeric(df["Altura"], errors="coerce")
    peso = peso.where((peso >= 0.3) & (peso <= 200))
    alt = alt.where((alt >= 30) & (alt <= 220))
    imc = peso / (alt / 100) ** 2  # IMC recalculado (o original tem erros de digitação)
    imc = imc.where((imc >= 7) & (imc <= 70))
    df["LOG_PESO"] = np.log1p(peso)
    df["ALTURA"] = alt
    df["LOG_IMC"] = np.log1p(imc)

    # Pressão arterial: limites físicos e PAD < PAS
    pas = pd.to_numeric(df["PA SISTOLICA"], errors="coerce").where(lambda s: s.between(50, 220))
    pad = pd.to_numeric(df["PA DIASTOLICA"], errors="coerce").where(lambda s: s.between(20, 150))
    inval = pas.notna() & pad.notna() & (pad >= pas)
    df["PAS"], df["PAD"] = pas.mask(inval), pad.mask(inval)
    df["PA_MEDIDA"] = (df["PAS"].notna() & df["PAD"].notna()).astype(int)

    # Frequência cardíaca
    df["FC"] = df["FC"].map(parse_fc).where(lambda s: s.between(40, 250))

    # Categóricas
    sexo = df["SEXO"].fillna("").astype(str).str.strip().str.upper().str[0]
    df["SEXO"] = sexo.map({"M": "M", "F": "F"}).fillna("Nao informado")
    pul = df["PULSOS"].fillna("").astype(str).str.strip().str.lower()
    df["PULSOS_ALTERADO"] = np.where(df["PULSOS"].isna(), np.nan, (~pul.str.startswith("norma")).astype(float))
    b2 = df["B2"].fillna("").astype(str).str.strip()
    df["B2"] = b2.where(df["B2"].notna(), np.nan)
    ppa = df["PPA"].fillna("").astype(str).str.lower()
    df["PPA"] = np.select([ppa.str.startswith("normal"), ppa.str.startswith("pre"), ppa.str.startswith("has")],
                          ["Normal", "Pre-hipertensao", "Hipertensao"], "Nao calculado")

    for c in ["MOTIVO1", "MOTIVO2"]:
        m = df[c].fillna("").astype(str).str.replace(r"^\s*\d+\s*-\s*", "", regex=True).str.strip()
        m = m.str.replace("congenica", "congênita")
        df[c] = m.where(df[c].notna(), "Nao informado")

    conv = df["Convenio"].fillna("").astype(str).map(lambda s: strip_accents(s).strip().upper())
    conv = conv.where(df["Convenio"].notna() & ~conv.str.fullmatch(r"\d+(\.\d+)?"), "NAO INFORMADO")
    df["Convenio"] = conv

    # HDA 1 + HDA2 -> indicadores multi-hot (HDA2 é 97% vazia)
    h1 = df["HDA 1"].fillna("").astype(str).map(lambda s: strip_accents(s).strip().lower())
    h2 = df["HDA2"].fillna("").astype(str).map(lambda s: strip_accents(s).strip().lower())
    for k, v in HDA_MAP.items():
        df[f"HDA_{v}"] = ((h1 == k) | (h2 == k)).astype(int)
    df["HDA_INFORMADO"] = df["HDA 1"].notna().astype(int)

    # SOPRO: mantido à parte apenas para o experimento de vazamento
    sop = df["SOPRO"].fillna("").astype(str).str.strip().str.lower()
    df["SOPRO"] = np.select([sop.eq("ausente"), sop.str.startswith("sist")], ["Ausente", "Sistolico"], "Outro")
    df.loc[raw.loc[df.index, "SOPRO"].isna(), "SOPRO"] = np.nan

    log["final"] = len(df)
    return df.reset_index(drop=True), log


NUM = ["IDADE", "LOG_PESO", "ALTURA", "LOG_IMC", "PAS", "PAD", "FC"]
NUM_LINEAR = ["IDADE", "LOG_IMC", "PAS", "PAD", "FC"]  # Peso e Altura: rho >= 0,98 com IDADE
CAT = ["SEXO", "B2", "PPA", "MOTIVO1", "MOTIVO2", "Convenio"]
BIN = ["PULSOS_ALTERADO", "PA_MEDIDA", "HDA_INFORMADO"] + [f"HDA_{v}" for v in HDA_MAP.values()]


# ---------------------------------------------------------------------------
# 2. Transformador ajustado no treino: outliers por faixa etária (escore-z robusto)
# ---------------------------------------------------------------------------

def age_band(age):
    a = np.asarray(age, dtype=float)
    band = np.where(a < 1, np.floor(a * 4), np.where(a < 2, 4 + np.floor((a - 1) * 2), 4 + a))
    return np.floor(band)


class AgeBandOutlierMasker(BaseEstimator, TransformerMixin):
    """Marca como ausente valores com |z robusto| > k dentro da faixa etária (medianas e MAD do treino)."""

    def __init__(self, cols=("LOG_PESO", "ALTURA", "LOG_IMC"), k=5.0, min_n=30):
        self.cols, self.k, self.min_n = cols, k, min_n

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        X["_b"] = age_band(X["IDADE"])
        self.stats_ = {}
        for c in self.cols:
            g = X.groupby("_b")[c]
            med, mad = g.median(), g.apply(lambda s: (s - s.median()).abs().median())
            n = g.count()
            self.stats_[c] = pd.DataFrame({"med": med, "mad": mad * 1.4826, "n": n})
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        b = pd.Series(age_band(X["IDADE"]), index=X.index)
        for c in self.cols:
            st = self.stats_[c]
            st = st[(st["n"] >= self.min_n) & (st["mad"] > 0)]
            med, mad = b.map(st["med"]), b.map(st["mad"])
            z = (X[c] - med) / mad
            X.loc[z.abs() > self.k, c] = np.nan
        return X


def make_preprocessor(num_cols, with_sopro=False):
    num = Pipeline([
        ("imp", IterativeImputer(estimator=BayesianRidge(), max_iter=15, random_state=RANDOM_STATE,
                                 add_indicator=True)),
        ("sc", StandardScaler()),
    ])
    cat_cols = CAT + (["SOPRO"] if with_sopro else [])
    cat = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01,
                              sparse_output=False)),
    ])
    bins = SimpleImputer(strategy="most_frequent")
    ct = ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols), ("bin", bins, BIN)],
                           verbose_feature_names_out=False)
    return Pipeline([("out", AgeBandOutlierMasker()), ("ct", ct)])


def make_pipe(model, num_cols=NUM, with_sopro=False):
    return Pipeline([("prep", make_preprocessor(num_cols, with_sopro)), ("clf", model)])


def metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    return {"Acuracia": accuracy_score(y, yhat), "Acur_bal": balanced_accuracy_score(y, yhat),
            "Precisao": precision_score(y, yhat, zero_division=0), "Sensib": recall_score(y, yhat),
            "Especif": tn / (tn + fp), "F1": f1_score(y, yhat), "ROC_AUC": roc_auc_score(y, p),
            "PR_AUC": average_precision_score(y, p), "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    raw = pd.read_csv(DATA, dtype=str)
    df, log = clean(raw)
    y = df["y"].values
    X = df.drop(columns=["y"])
    res = {"limpeza": log, "n": len(df), "prev_anormal": float(y.mean())}

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    res["n_treino"], res["n_teste"] = len(Xtr), len(Xte)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    spos = (ytr == 0).sum() / (ytr == 1).sum()
    specs = {
        "Dummy (classe majoritaria)": (make_pipe(DummyClassifier(strategy="most_frequent")), {}),
        "Regressao Logistica": (make_pipe(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE), NUM_LINEAR),
                                {"clf__C": [0.01, 0.1, 1, 10], "clf__class_weight": [None, "balanced"]}),
        "Random Forest": (make_pipe(RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE)),
                          {"clf__max_depth": [None, 10, 20], "clf__min_samples_leaf": [1, 5, 10],
                           "clf__max_features": ["sqrt", 0.3], "clf__class_weight": [None, "balanced"]}),
        "XGBoost": (make_pipe(XGBClassifier(subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                                            n_jobs=-1, random_state=RANDOM_STATE)),
                    {"clf__n_estimators": [200, 400], "clf__max_depth": [3, 5],
                     "clf__learning_rate": [0.03, 0.1], "clf__scale_pos_weight": [1, float(spos)]}),
    }

    res["modelos"], probs, best = {}, {}, {}
    for name, (pipe, grid) in specs.items():
        gs = GridSearchCV(pipe, grid or {"clf__strategy": ["most_frequent"]}, cv=cv, n_jobs=1,
                          scoring={"auc": "roc_auc", "f1": "f1", "bal": "balanced_accuracy"},
                          refit="auc")
        gs.fit(Xtr, ytr)
        i = gs.best_index_
        cvr = gs.cv_results_
        p = gs.predict_proba(Xte)[:, 1]
        probs[name], best[name] = p, gs.best_estimator_
        res["modelos"][name] = {
            "best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
            "cv_auc": [float(cvr["mean_test_auc"][i]), float(cvr["std_test_auc"][i])],
            "cv_f1": [float(cvr["mean_test_f1"][i]), float(cvr["std_test_f1"][i])],
            "cv_bal": [float(cvr["mean_test_bal"][i]), float(cvr["std_test_bal"][i])],
            "teste": metrics(yte, p)}
        print(name, res["modelos"][name]["best_params"], round(cvr["mean_test_auc"][i], 4),
              {k: round(v, 3) for k, v in res["modelos"][name]["teste"].items()}, flush=True)

    pd.DataFrame({"y": yte, **probs}).to_csv(f"{OUT}/probs_teste.csv", index=False)
    # IC 95% bootstrap (1000 reamostragens do teste) para a AUC de cada modelo e a diferença XGB - RL
    rng = np.random.RandomState(RANDOM_STATE)
    boots = {n: [] for n in probs}
    for _ in range(1000):
        ix = rng.randint(0, len(yte), len(yte))
        if len(np.unique(yte[ix])) < 2:
            continue
        for n, p in probs.items():
            boots[n].append(roc_auc_score(yte[ix], p[ix]))
    res["auc_ic95"] = {n: [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] for n, b in boots.items()}
    diff = np.array(boots["XGBoost"]) - np.array(boots["Regressao Logistica"])
    res["dif_auc_xgb_rl"] = [float(diff.mean()), float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]

    # Escolhe o melhor modelo pela AUC na validação cruzada (não pelo teste)
    best_name = max([n for n in specs if not n.startswith("Dummy")], key=lambda n: res["modelos"][n]["cv_auc"][0])
    res["melhor"] = best_name
    bm = best[best_name]

    # Limiar para triagem: sensibilidade >= 0,90 nas predições out-of-fold do treino
    oof = cross_val_predict(bm, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
    fpr, tpr, thr = roc_curve(ytr, oof)
    thr90 = float(thr[np.argmax(tpr >= 0.90)])
    res["limiar_sens90"] = thr90
    res["teste_limiar_sens90"] = metrics(yte, probs[best_name], thr90)

    # Experimento de controle: o mesmo modelo COM o SOPRO (vazamento)
    from sklearn.base import clone
    leak = make_pipe(clone(bm.named_steps["clf"]), with_sopro=True)
    leak.fit(Xtr, ytr)
    res["com_sopro"] = metrics(yte, leak.predict_proba(Xte)[:, 1])
    regra = (Xte["SOPRO"].fillna("Ausente") != "Ausente").astype(float).values
    res["regra_so_sopro"] = metrics(yte, regra)

    # Análise de erros do melhor modelo (limiar 0,5)
    pb = probs[best_name]
    yhat = (pb >= 0.5).astype(int)
    E = Xte.copy()
    E["y"], E["yhat"] = yte, yhat
    E["tipo"] = np.select([(E.y == 1) & (E.yhat == 0), (E.y == 0) & (E.yhat == 1), (E.y == 1) & (E.yhat == 1)],
                          ["FN", "FP", "VP"], "VN")
    E["faixa"] = pd.cut(E["IDADE"], [-0.01, 1, 5, 10, 15, 20], labels=["<1", "1-4", "5-9", "10-14", "15-19"])
    err = {}
    for t in ["FN", "FP", "VP", "VN"]:
        s = E[E.tipo == t]
        err[t] = {"n": len(s), "idade_mediana": float(s["IDADE"].median()),
                  "pct_sopro_presente": float((s["SOPRO"].fillna("Ausente") != "Ausente").mean()),
                  "pct_motivo2_sopro": float((s["MOTIVO2"] == "Sopro").mean()),
                  "pct_b2_alterada": float((s["B2"].fillna("Normal") != "Normal").mean()),
                  "pct_motivo1_parecer": float((s["MOTIVO1"] == "Parecer cardiológico").mean()),
                  "pct_cardiopatia_estab": float((s["MOTIVO1"] == "Cardiopatia já estabelecida").mean())}
    res["erros"] = err
    res["sens_por_faixa"] = {str(k): float(g.loc[g.y == 1, "yhat"].mean()) for k, g in E.groupby("faixa")}
    res["espec_por_faixa"] = {str(k): float(1 - g.loc[g.y == 0, "yhat"].mean()) for k, g in E.groupby("faixa")}

    # Importância por permutação (colunas originais, no teste)
    feats = NUM + CAT + BIN
    pi = permutation_importance(bm, Xte, yte, scoring="roc_auc", n_repeats=10,
                                random_state=RANDOM_STATE, n_jobs=1)
    cols = list(Xte.columns)
    imp = {c: (float(pi.importances_mean[cols.index(c)]), float(pi.importances_std[cols.index(c)])) for c in feats}
    res["importancia"] = dict(sorted(imp.items(), key=lambda kv: -kv[1][0]))

    # Coeficientes da regressão logística (atributos padronizados / one-hot)
    lr = best["Regressao Logistica"]
    names = lr.named_steps["prep"].named_steps["ct"].get_feature_names_out()
    coef = lr.named_steps["clf"].coef_[0]
    res["coef_lr"] = dict(sorted(zip(map(str, names), map(float, coef)), key=lambda kv: -abs(kv[1]))[:15])

    # Dados para figuras
    res["roc"] = {}
    for name, p in probs.items():
        f, t, _ = roc_curve(yte, p)
        idx = np.linspace(0, len(f) - 1, min(200, len(f))).astype(int)
        res["roc"][name] = [f[idx].tolist(), t[idx].tolist()]
    tot = df.copy()
    tot["faixa"] = pd.cut(tot["IDADE"], [-0.01, 1, 5, 10, 15, 20], labels=["<1", "1-4", "5-9", "10-14", "15-19"])
    res["eda"] = {
        "sopro": {k: [int(len(g)), float(g.y.mean())] for k, g in tot.groupby(tot["SOPRO"].fillna("NA"))},
        "faixa": {str(k): [int(len(g)), float(g.y.mean())] for k, g in tot.groupby("faixa")},
        "motivo2": {k: [int(len(g)), float(g.y.mean())] for k, g in tot.groupby("MOTIVO2")},
        "motivo1": {k: [int(len(g)), float(g.y.mean())] for k, g in tot.groupby("MOTIVO1")},
        "b2": {k: [int(len(g)), float(g.y.mean())] for k, g in tot.groupby(tot["B2"].fillna("NA"))},
        "faltantes": {c: float(tot[c].isna().mean()) for c in NUM},
        "n_atributos_modelo": len(feats),
    }
    with open(f"{OUT}/results.json", "w") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False, default=float)
    print(json.dumps({k: res[k] for k in ["limpeza", "melhor", "limiar_sens90", "teste_limiar_sens90",
                                          "com_sopro", "regra_so_sopro", "erros", "sens_por_faixa"]},
                     indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
