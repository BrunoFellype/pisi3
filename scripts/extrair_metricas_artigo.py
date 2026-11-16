"""
Script de Extração e Auditoria de Métricas do Artigo ASTRA.

Gera os valores consolidados para:
- Seção 5: Análise Exploratória (Efeito Teto e Distribuição do GPA)
- Seção 7.1: Clusterização K-Means (k=3, Silhueta e Perfis Comportamentais)
- Seção 7.2: Torneio de Regressão (Tabela 1: Treino / Tabela 2: Teste / Importâncias)
"""

import sys
from pathlib import Path

# Adiciona a raiz do projeto (um nível acima de scripts/) ao sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from ASTRA_Dashboard.analysis.hipoteses import comparar_desempenho_por_cluster
from ASTRA_Dashboard.analysis.predicao import treinar_motor_preditivo
from ASTRA_Dashboard.data.load import get_Dataset


def main():
    print("=" * 75)
    print("      ASTRA - AUDITORIA E EXTRAÇÃO DE MÉTRICAS PARA O ARTIGO")
    print("=" * 75)

    # Carregamento oficial da base via load.py
    df = get_Dataset()

    # -------------------------------------------------------------
    # 1. SEÇÃO 5: ANÁLISE EXPLORATÓRIA & AUDITORIA DA BASE
    # -------------------------------------------------------------
    print("\n[1] SEÇÃO 5 - ANÁLISE EXPLORATÓRIA (DISTRIBUIÇÕES & EFEITO TETO)")
    total_alunos = len(df)
    teto_exame = int((df["final_exam_score"] == 100).sum())
    pct_teto = (teto_exame / total_alunos) * 100

    print(f"Total de registros na base: {total_alunos}")
    print(f"Alunos com nota 100 no exame final: {teto_exame} ({pct_teto:.2f}%) -> EFEITO TETO")
    print(
        f"GPA - Mínimo: {df['GPA'].min():.2f} | Máximo: {df['GPA'].max():.2f} | "
        f"Média: {df['GPA'].mean():.2f} | Mediana: {df['GPA'].median():.2f}"
    )

    # -------------------------------------------------------------
    # 2. SEÇÃO 7.1: CLUSTERIZAÇÃO K-MEANS (k=3)
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("[2] SEÇÃO 7.1 - CLUSTERIZAÇÃO K-MEANS (PERFIS DE ESTUDANTES)")
    print("-" * 75)

    habitos_cluster = [
        "study_hours_per_day",
        "sleep_hours",
        "mental_stress_level",
        "class_attendance_percent",
        "social_media_hours",
    ]
    sub_cluster = df[habitos_cluster].dropna()
    scaler_cluster = StandardScaler()
    X_scaled = scaler_cluster.fit_transform(sub_cluster)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    df_clustered = df.copy()
    df_clustered["cluster_id"] = labels

    silhueta = silhouette_score(X_scaled, labels)
    print(f"Coeficiente de Silhueta médio (k=3): {silhueta:.4f}")

    res_clusters = comparar_desempenho_por_cluster(
        df_clustered, cluster_coluna="cluster_id"
    )
    print("\nPerfil Médio dos Clusters:")
    print(pd.DataFrame(res_clusters["tabela"]).to_string(index=False))

    # -------------------------------------------------------------
    # 3. SEÇÃO 7.2: TABELAS 1 E 2 (TREINO VS TESTE VIA MOTOR PREDITIVO)
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("[3] SEÇÃO 7.2 - REGRESSÃO PREDITIVA (TREINO VS TESTE)")
    print("-" * 75)

    retorno_motor = treinar_motor_preditivo(df)

    modelos_treinados = retorno_motor["modelos"]
    X_test = retorno_motor["X_test"]
    y_test = retorno_motor["y_test"]

    # Tabelas lidas direto do motor (mesmos números do código de produção)
    linhas_treino = []
    linhas_teste = []
    for _, r in retorno_motor["resultados"].iterrows():
        linhas_treino.append({
            "Modelo": r["Modelo"],
            "R²": round(float(r["R² (treino)"]), 4),
            "MAE": round(float(r["MAE (treino)"]), 4),
            "RMSE": round(float(r["RMSE (treino)"]), 4),
        })
        linhas_teste.append({
            "Modelo": r["Modelo"],
            "R²": round(float(r["R²"]), 4),
            "MAE": round(float(r["MAE"]), 4),
            "RMSE": round(float(r["RMSE"]), 4),
        })

    print("\n>>> TABELA 1: DESEMPENHO NO TREINO (80% - 8.000 amostras)")
    print(pd.DataFrame(linhas_treino).to_string(index=False))

    print("\n>>> TABELA 2: DESEMPENHO NO TESTE (20% - 2.000 amostras)")
    print(pd.DataFrame(linhas_teste).to_string(index=False))

    # -------------------------------------------------------------
    # 4. IMPORTÂNCIA DAS VARIÁVEIS NO MODELO VENCEDOR
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print(f"[4] SEÇÃO 7.2 - IMPORTÂNCIA DAS VARIÁVEIS ({retorno_motor['nome']})")
    print("-" * 75)

    melhor_pipe = retorno_motor["modelo"]
    r_perm = permutation_importance(
        melhor_pipe, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )

    df_imp = (
        pd.DataFrame({
            "Variável": retorno_motor["features"],
            "Importância Média": r_perm.importances_mean,
            "Desvio Padrão": r_perm.importances_std,
        })
        .sort_values(by="Importância Média", ascending=False)
        .reset_index(drop=True)
    )

    print(df_imp.to_string(index=False))
    print("\n" + "=" * 75)
    print("Extração concluída com sucesso. Todos os dados estão prontos!")
    print("=" * 75)


if __name__ == "__main__":
    main()