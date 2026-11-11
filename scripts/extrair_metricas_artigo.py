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

    # Reconstitui X_train e y_train com base nas amostras que sobraram do split
    df_trab = df.copy()
    features_habitos = [col for col in habitos_cluster if col in df_trab.columns]
    
    try:
        sub_temp = df_trab[features_habitos].dropna()
        if not sub_temp.empty:
            sc_temp = StandardScaler()
            labels_temp = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(
                sc_temp.fit_transform(sub_temp)
            )
            mapa = {0: "Perfil Comportamental A", 1: "Perfil Comportamental B", 2: "Perfil Comportamental C"}
            df_trab.loc[sub_temp.index, "cluster_perfil"] = pd.Series(labels_temp, index=sub_temp.index).map(mapa)
    except Exception:
        pass
    
    if "cluster_perfil" not in df_trab.columns:
        df_trab["cluster_perfil"] = "Perfil Geral"
    else:
        df_trab["cluster_perfil"] = df_trab["cluster_perfil"].fillna("Perfil Geral")

    X_completo = df_trab[retorno_motor["features"]]
    y_completo = df_trab["GPA"]
    
    X_train = X_completo.drop(index=X_test.index)
    y_train = y_completo.drop(index=y_test.index)

    linhas_treino = []
    linhas_teste = []

    for nome_modelo, pipeline in modelos_treinados.items():
        # Avaliação em Treino
        y_pred_train = np.clip(pipeline.predict(X_train), 0.0, 4.0)
        linhas_treino.append({
            "Modelo": nome_modelo,
            "R²": round(r2_score(y_train, y_pred_train), 4),
            "MAE": round(mean_absolute_error(y_train, y_pred_train), 4),
            "RMSE": round(np.sqrt(mean_squared_error(y_train, y_pred_train)), 4),
        })

        # Avaliação em Teste
        y_pred_test = np.clip(pipeline.predict(X_test), 0.0, 4.0)
        linhas_teste.append({
            "Modelo": nome_modelo,
            "R²": round(r2_score(y_test, y_pred_test), 4),
            "MAE": round(mean_absolute_error(y_test, y_pred_test), 4),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 4),
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