"""Análises avançadas (PCA, HDBSCAN, classificação, SHAP, justiça) para o dashboard.

Mesma lógica de scripts/auditoria_complementar_artigo.py, mas devolvendo tabelas
e gráficos para a aba "Análises Avançadas". O cálculo roda uma vez e fica em cache.
"""
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    precision_score, r2_score, recall_score, roc_auc_score, silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from analysis.clusters import FEATURES_KMEANS

LIMIAR_GPA = 3.5
SEMENTE = 42
# Só hábitos e idade. final_exam_score e assignment_score ficam de fora de propósito:
# são resultados acadêmicos e vazariam o GPA para o modelo.
FEATURES_MODELO = FEATURES_KMEANS + ["age"]


def _pca_e_outliers(df):
    X = StandardScaler().fit_transform(df[FEATURES_KMEANS])
    pca = PCA(n_components=2, random_state=SEMENTE)
    X_pca = pca.fit_transform(X)

    sil_orig = silhouette_score(X, KMeans(3, random_state=SEMENTE, n_init=10).fit_predict(X))
    sil_pca = silhouette_score(X_pca, KMeans(3, random_state=SEMENTE, n_init=10).fit_predict(X_pca))

    rotulos = KMeans(3, random_state=SEMENTE, n_init=10).fit_predict(X)
    out = df.copy()
    out["cluster_bruto"] = rotulos
    ordem = out.groupby("cluster_bruto")["GPA"].mean().sort_values(ascending=False).index
    out["Perfil"] = out["cluster_bruto"].map({c: f"Perfil {i + 1}" for i, c in enumerate(ordem)})
    out["PC1"], out["PC2"] = X_pca[:, 0], X_pca[:, 1]
    out["Atípico"] = np.where(
        HDBSCAN(min_cluster_size=30, min_samples=10, copy=True).fit_predict(X_pca) == -1, "Atípico", "Típico"
    )

    cargas = pd.DataFrame(pca.components_.T, index=FEATURES_KMEANS, columns=["PC1", "PC2"]).round(3)
    cargas.insert(0, "Hábito", cargas.index)
    return {
        "variancia": [float(v) for v in pca.explained_variance_ratio_],
        "sil_orig": float(sil_orig),
        "sil_pca": float(sil_pca),
        "cargas": cargas.reset_index(drop=True),
        "dados": out,
    }


def _classificacao(df):
    y = (df["GPA"] >= LIMIAR_GPA).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(
        df[FEATURES_MODELO], y, test_size=0.2, random_state=SEMENTE, stratify=y
    )
    linhas, matrizes = [], {}
    for nome, peso in [("Sem balanceamento", None), ("Balanceado (class_weight)", "balanced")]:
        clf = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.03, class_weight=peso, random_state=SEMENTE
        ).fit(Xtr, ytr)
        pred = clf.predict(Xte)
        linhas.append({
            "Modelo": nome,
            "Acurácia": round(accuracy_score(yte, pred), 3),
            "Precisão": round(precision_score(yte, pred), 3),
            "Recall": round(recall_score(yte, pred), 3),
            "F1": round(f1_score(yte, pred), 3),
            "AUC-ROC": round(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]), 3),
        })
        matrizes[nome] = confusion_matrix(yte, pred)
    return {
        "metricas": pd.DataFrame(linhas),
        "matrizes": matrizes,
        "positivos": float(y.mean() * 100),
    }


def _shap_e_justica(df):
    Xtr, Xte, ytr, yte, _, idx = train_test_split(
        df[FEATURES_MODELO], df["GPA"], df.index, test_size=0.2, random_state=SEMENTE
    )
    reg = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.03, random_state=SEMENTE).fit(Xtr, ytr)
    pred = np.clip(reg.predict(Xte), 0.0, 4.0)

    shap_df, aviso = None, None
    try:
        import shap
        valores = shap.TreeExplainer(reg).shap_values(Xte)
        shap_df = (
            pd.DataFrame({"Variável": FEATURES_MODELO, "|SHAP| médio": np.abs(valores).mean(axis=0).round(4)})
            .sort_values("|SHAP| médio")
        )
    except ImportError:
        aviso = "Biblioteca 'shap' não instalada (pip install shap)."

    aval = df.loc[idx].copy()
    aval["previsto"] = pred
    justica = {}
    for atributo, rotulo in [("gender", "Gênero"), ("country", "País")]:
        linhas = []
        for grupo, sub in aval.groupby(atributo):
            linhas.append({
                rotulo: grupo,
                "Alunos (teste)": len(sub),
                "GPA real": round(sub["GPA"].mean(), 3),
                "GPA previsto": round(sub["previsto"].mean(), 3),
                "MAE": round(mean_absolute_error(sub["GPA"], sub["previsto"]), 4),
            })
        justica[rotulo] = pd.DataFrame(linhas)
    return {"shap": shap_df, "aviso": aviso, "justica": justica}


def _vazamento(df):
    """Mostra quanto do GPA cada grupo de variáveis explica (teste 80/20)."""
    grupos = {
        "Só hábitos + idade (usado nos modelos)": FEATURES_MODELO,
        "Só notas (final_exam_score + assignment_score)": ["final_exam_score", "assignment_score"],
        "Hábitos + idade + notas": FEATURES_MODELO + ["final_exam_score", "assignment_score"],
    }
    linhas = []
    for nome, cols in grupos.items():
        Xtr, Xte, ytr, yte = train_test_split(df[cols], df["GPA"], test_size=0.2, random_state=SEMENTE)
        reg = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.03, random_state=SEMENTE).fit(Xtr, ytr)
        linhas.append({
            "Variáveis usadas para prever o GPA": nome,
            "R² (teste)": round(float(r2_score(yte, np.clip(reg.predict(Xte), 0, 4))), 3),
        })
    corr = df[["GPA", "final_exam_score", "assignment_score"]].corr().loc["GPA"]
    return {
        "tabela": pd.DataFrame(linhas),
        "corr_prova": float(corr["final_exam_score"]),
        "corr_trabalhos": float(corr["assignment_score"]),
    }


def _composicao_atipicos(dados):
    """Entre os melhores alunos (Perfil 1), como atípicos e típicos se distribuem nas categorias."""
    p1 = dados[dados["Perfil"] == "Perfil 1"]
    linhas = []
    for col, rotulo in [("gender", "Gênero"), ("family_income_level", "Renda familiar"),
                        ("part_time_job", "Trabalha"), ("major", "Curso")]:
        pct = pd.crosstab(p1[col], p1["Atípico"], normalize="columns") * 100
        for cat in pct.index:
            tip, ati = pct.loc[cat].get("Típico", 0.0), pct.loc[cat].get("Atípico", 0.0)
            linhas.append({
                "Característica": rotulo, "Categoria": cat,
                "% dos típicos": round(tip, 1), "% dos atípicos": round(ati, 1),
                "Diferença (p.p.)": round(ati - tip, 1),
            })
    tabela = pd.DataFrame(linhas)
    tabela = tabela.reindex(tabela["Diferença (p.p.)"].abs().sort_values(ascending=False).index).head(8)
    medias = p1.groupby("Atípico")[["mental_stress_level", "sleep_hours", "study_hours_per_day"]].mean().round(2)
    return {"diferencas": tabela.reset_index(drop=True), "n_tipicos": int((p1["Atípico"] == "Típico").sum()),
            "n_atipicos": int((p1["Atípico"] == "Atípico").sum()), "medias": medias}


def _limiares(df):
    linhas = []
    for lim in (3.0, 3.5, 3.8):
        y = (df["GPA"] >= lim).astype(int)
        Xtr, Xte, ytr, yte = train_test_split(
            df[FEATURES_MODELO], y, test_size=0.2, random_state=SEMENTE, stratify=y
        )
        clf = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.03, class_weight="balanced", random_state=SEMENTE
        ).fit(Xtr, ytr)
        linhas.append({
            "Limiar (GPA ≥)": lim, "% de positivos": round(float(y.mean() * 100), 1),
            "F1": round(f1_score(yte, clf.predict(Xte)), 3),
            "AUC-ROC": round(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]), 3),
        })
    return pd.DataFrame(linhas)


@lru_cache(maxsize=1)
def _calcular(chave):
    # chave só serve para o cache; o df vem de _DF_ATUAL
    df = _DF_ATUAL
    pca = _pca_e_outliers(df)
    return {
        "vazamento": _vazamento(df),
        "composicao": _composicao_atipicos(pca["dados"]),
        "limiares": _limiares(df),
        "pca": pca,
        "classificacao": _classificacao(df),
        "explicabilidade": _shap_e_justica(df),
    }


_DF_ATUAL = None


def calcular_analises_avancadas(df):
    """Roda tudo sobre a base completa e guarda o resultado em cache."""
    global _DF_ATUAL
    _DF_ATUAL = df.reset_index(drop=True)
    return _calcular((len(df), round(float(df["GPA"].sum()), 4)))


def grafico_pca_outliers(dados):
    fig = px.scatter(
        dados.sample(min(len(dados), 4000), random_state=SEMENTE),
        x="PC1", y="PC2", color="Perfil", symbol="Atípico",
        symbol_map={"Típico": "circle", "Atípico": "x"},
        opacity=0.65,
    )
    return fig


def grafico_shap(shap_df):
    return px.bar(shap_df, x="|SHAP| médio", y="Variável", orientation="h")


def grafico_confusao(matriz, titulo):
    return px.imshow(
        matriz, text_auto=True, x=["Previsto abaixo", "Previsto ≥ 3,5"],
        y=["Real abaixo", "Real ≥ 3,5"], aspect="auto", title=titulo,
    )
