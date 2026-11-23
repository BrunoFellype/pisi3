"""
Auditoria complementar de métricas estatísticas e modelos de Machine Learning
para validação dos resultados reportados no artigo científico do projeto ASTRA.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HABITOS_SELECIONADOS = [
    "study_hours_per_day",
    "sleep_hours",
    "mental_stress_level",
    "class_attendance_percent",
    "social_media_hours",
]


def localizar_dataset() -> Path:
    """Localiza o arquivo CSV oficial a partir da raiz do repositório."""
    raiz_repo = Path(__file__).resolve().parent.parent
    caminhos_possiveis = [
        raiz_repo / "Data" / "global_university_students_performance_habits_10000.csv",
        raiz_repo / "ASTRA_Dashboard" / "data" / "global_university_students_performance_habits_10000.csv",
    ]
    for caminho in caminhos_possiveis:
        if caminho.exists():
            return caminho
    raise FileNotFoundError(f"Dataset não encontrado nos caminhos esperados: {caminhos_possiveis}")


def carregar_dados() -> tuple[pd.DataFrame, str]:
    """Carrega o dataset padronizando nomes de colunas e identifica a coluna alvo."""
    caminho_csv = localizar_dataset()
    df = pd.read_csv(caminho_csv)
    df.columns = [col.strip().lower() for col in df.columns]
    coluna_gpa = next(col for col in df.columns if "gpa" in col)
    return df, coluna_gpa


def analisar_distribuicao_e_categorias(df: pd.DataFrame, coluna_gpa: str) -> None:
    """Verifica saturação no teto do GPA e frequência das categorias de anotação."""
    print("\n" + "=" * 75)
    print("[1] EFEITO TETO DO GPA E CATEGORIAS DE ANOTAÇÃO")
    print("=" * 75)

    qtd_teto = int((df[coluna_gpa] == 4.0).sum())
    print(f"Registros com GPA == 4.00: {qtd_teto} ({qtd_teto / len(df) * 100:.2f}%)")

    if "note_taking_method" in df.columns:
        print("\nDistribuição de note_taking_method:")
        print(df["note_taking_method"].value_counts(dropna=False).to_string())


def avaliar_clusterizacao_e_pca(df: pd.DataFrame, coluna_gpa: str) -> pd.DataFrame:
    """Executa PCA, K-Means (k=2..8), estabilidade ARI e detecção de outliers com HDBSCAN."""
    print("\n" + "=" * 75)
    print("[2] REDUÇÃO DE DIMENSIONALIDADE (PCA), K-MEANS E OUTLIERS (HDBSCAN)")
    print("=" * 75)

    scaler = StandardScaler()
    X_padronizado = scaler.fit_transform(df[HABITOS_SELECIONADOS])

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_padronizado)
    var_total = pca.explained_variance_ratio_.sum() * 100
    print(f"PCA (2 componentes) - Variância explicada: {var_total:.2f}% {pca.explained_variance_ratio_.round(4)}")

    print("\nComparativo de Inércia e Silhueta (Espaço Original vs. Pós-PCA):")
    for k in range(2, 9):
        km_orig = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_padronizado)
        km_pca = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_pca)
        sil_orig = silhouette_score(X_padronizado, km_orig.labels_)
        sil_pca = silhouette_score(X_pca, km_pca.labels_)
        print(
            f"  k={k} | Original -> Inércia: {km_orig.inertia_:,.2f}, Silhueta: {sil_orig:.4f} "
            f"| Pós-PCA -> Silhueta: {sil_pca:.4f}"
        )

    rotulos_base = KMeans(n_clusters=3, random_state=0, n_init=10).fit_predict(X_padronizado)
    scores_ari = [
        adjusted_rand_score(
            rotulos_base,
            KMeans(n_clusters=3, random_state=seed, n_init=10).fit_predict(X_padronizado),
        )
        for seed in range(1, 31)
    ]
    print(f"\nEstabilidade ARI (30 sementes, k=3): Mínimo = {min(scores_ari):.4f} | Médio = {np.mean(scores_ari):.4f}")

    # Ordena os 3 clusters pelo GPA médio decrescente (Perfil 1 = maior GPA)
    df_resultado = df.copy()
    rotulos_k3 = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_padronizado)
    df_resultado["cluster_bruto"] = rotulos_k3
    ordem_gpa = (
        df_resultado.groupby("cluster_bruto")[coluna_gpa]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    mapa_perfis = {cid: idx + 1 for idx, cid in enumerate(ordem_gpa)}
    df_resultado["perfil"] = df_resultado["cluster_bruto"].map(mapa_perfis)

    print("\nDiscriminação dos Perfis K-Means (Média ± Desvio-Padrão):")
    for perfil_id, grupo in df_resultado.groupby("perfil"):
        print(f"\n  Perfil {perfil_id} (n={len(grupo)}, {len(grupo) / len(df_resultado) * 100:.2f}%):")
        for col in [coluna_gpa] + HABITOS_SELECIONADOS:
            print(f"    {col}: {grupo[col].mean():.2f} ± {grupo[col].std():.2f}")

    # Detecção de outliers com HDBSCAN sobre a projeção PCA
    hdb = HDBSCAN(min_cluster_size=30, min_samples=10, copy=True)
    df_resultado["hdbscan_label"] = hdb.fit_predict(X_pca)
    outliers_globais = df_resultado[df_resultado["hdbscan_label"] == -1]
    outliers_perfil_1 = df_resultado[(df_resultado["perfil"] == 1) & (df_resultado["hdbscan_label"] == -1)]

    print(
        f"\nOutliers HDBSCAN (PCA-2D): {len(outliers_globais)} registros "
        f"({len(outliers_globais) / len(df_resultado) * 100:.2f}% da base)"
    )
    print(
        f"Outliers pertencentes ao Perfil 1: {len(outliers_perfil_1)} "
        f"({len(outliers_perfil_1) / (df_resultado['perfil'] == 1).sum() * 100:.2f}% do Perfil 1)"
    )
    for col in [coluna_gpa] + HABITOS_SELECIONADOS:
        print(f"    {col}: {outliers_perfil_1[col].mean():.2f} ± {outliers_perfil_1[col].std():.2f}")

    return df_resultado


def avaliar_classificacao_binaria(df: pd.DataFrame, coluna_gpa: str) -> None:
    """Avalia classificação binária de alto rendimento (GPA >= 3.5) com e sem balanceamento."""
    print("\n" + "=" * 75)
    print("[3] CLASSIFICAÇÃO BINÁRIA DE ALTO RENDIMENTO (GPA >= 3.5)")
    print("=" * 75)

    y_binario = (df[coluna_gpa] >= 3.5).astype(int)
    features = HABITOS_SELECIONADOS + ["age"]
    print(
        f"Distribuição de classes: Positivos = {y_binario.sum()} ({y_binario.mean() * 100:.2f}%) | "
        f"Negativos = {(1 - y_binario).sum()} ({(1 - y_binario).mean() * 100:.2f}%)"
    )

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        df[features], y_binario, test_size=0.2, random_state=42, stratify=y_binario
    )

    for peso in [None, "balanced"]:
        clf = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.03, class_weight=peso, random_state=42
        )
        clf.fit(X_treino, y_treino)
        predicoes = clf.predict(X_teste)
        probabilidades = clf.predict_proba(X_teste)[:, 1]

        print(f"\nHistGradientBoostingClassifier (class_weight={peso}):")
        print(
            f"  Acurácia: {accuracy_score(y_teste, predicoes):.4f} | "
            f"Precisão: {precision_score(y_teste, predicoes):.4f} | "
            f"Recall: {recall_score(y_teste, predicoes):.4f} | "
            f"F1-Score: {f1_score(y_teste, predicoes):.4f} | "
            f"AUC-ROC: {roc_auc_score(y_teste, probabilidades):.4f}"
        )
        print(f"  Matriz de Confusão:\n{confusion_matrix(y_teste, predicoes)}")


def avaliar_explicabilidade_e_justica(df: pd.DataFrame, coluna_gpa: str) -> None:
    """Calcula valores SHAP e avalia consistência de erro (MAE) entre grupos demográficos."""
    print("\n" + "=" * 75)
    print("[4] EXPLICABILIDADE (SHAP) E AUDITORIA DE JUSTIÇA (FAIRNESS)")
    print("=" * 75)

    features = HABITOS_SELECIONADOS + ["age"]
    X_treino, X_teste, y_treino, y_teste, _, idx_teste = train_test_split(
        df[features], df[coluna_gpa], df.index, test_size=0.2, random_state=42
    )

    regressor = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.03, random_state=42)
    regressor.fit(X_treino, y_treino)
    predicoes = np.clip(regressor.predict(X_teste), 0.0, 4.0)

    try:
        import shap

        explainer = shap.TreeExplainer(regressor)
        valores_shap = explainer.shap_values(X_teste)
        importancia_media = np.abs(valores_shap).mean(axis=0)
        print("Importância média absoluta (|SHAP|):")
        for atributo, valor in sorted(zip(features, importancia_media), key=lambda item: -item[1]):
            print(f"  {atributo}: {valor:.4f}")
    except ImportError:
        print("Aviso: biblioteca 'shap' não instalada no ambiente virtual.")

    df_avaliacao = df.loc[idx_teste].copy()
    df_avaliacao["gpa_previsto"] = predicoes

    for atributo_demo in ["gender", "country"]:
        if atributo_demo in df_avaliacao.columns:
            print(f"\nAuditoria de Justiça por '{atributo_demo}' (Top 5 grupos):")
            for categoria, subgrupo in list(df_avaliacao.groupby(atributo_demo))[:5]:
                mae_grupo = mean_absolute_error(subgrupo[coluna_gpa], subgrupo["gpa_previsto"])
                print(
                    f"  {categoria} (n={len(subgrupo)}): "
                    f"GPA Real = {subgrupo[coluna_gpa].mean():.3f} | "
                    f"GPA Previsto = {subgrupo['gpa_previsto'].mean():.3f} | "
                    f"MAE = {mae_grupo:.4f}"
                )


def main() -> None:
    df, coluna_gpa = carregar_dados()
    analisar_distribuicao_e_categorias(df, coluna_gpa)
    avaliar_clusterizacao_e_pca(df, coluna_gpa)
    avaliar_classificacao_binaria(df, coluna_gpa)
    avaliar_explicabilidade_e_justica(df, coluna_gpa)


if __name__ == "__main__":
    main()