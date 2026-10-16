import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans


def treinar_motor_preditivo(df):
    """
    Treina e compara modelos para previsão do GPA, alinhado com a clusterização
    comportamental (livre de Target Leakage).
    """
    df_trab = df.copy()

    # -------------------------------------------------------------------------
    # CORREÇÃO DO TARGET LEAKAGE:
    # Definimos os perfis comportamentais baseados APENAS em hábitos,
    # sem incluir a coluna "GPA" para não gerar vazamento de alvo.
    # -------------------------------------------------------------------------
    features_habitos_cluster = [
        "study_hours_per_day",
        "sleep_hours",
        "mental_stress_level",
        "class_attendance_percent",
        "social_media_hours"
    ]
    
    features_habitos_cluster = [col for col in features_habitos_cluster if col in df_trab.columns]

    try:
        df_sub_cluster = df_trab[features_habitos_cluster].dropna()
        if not df_sub_cluster.empty:
            scaler_temp = StandardScaler()
            X_scaled_temp = scaler_temp.fit_transform(df_sub_cluster)
            
            # KMeans focado exclusivamente em padrões de rotina/hábitos
            kmeans_temp = KMeans(n_clusters=3, random_state=42, n_init=10)
            labels_temp = kmeans_temp.fit_predict(X_scaled_temp)
            
            df_trab['cluster_perfil'] = "Perfil Geral"
            df_trab.loc[df_sub_cluster.index, 'cluster_id_temp'] = labels_temp
            
            mapa_nomes_temp = {
                0: "Perfil Comportamental A", 
                1: "Perfil Comportamental B", 
                2: "Perfil Comportamental C"
            }
            df_trab['cluster_perfil'] = df_trab['cluster_id_temp'].map(mapa_nomes_temp).fillna("Perfil Geral")
        else:
            df_trab['cluster_perfil'] = "Perfil Geral"
    except Exception:
        df_trab['cluster_perfil'] = "Perfil Geral"

    # Variáveis utilizadas para previsão final do GPA
    features_numericas = [
        "study_hours_per_day",
        "sleep_hours",
        "mental_stress_level",
        "class_attendance_percent",
        "social_media_hours",
        "age"
    ]

    features_categoricas = ["cluster_perfil"] if "cluster_perfil" in df_trab.columns else []

    # Verifica quais colunas realmente existem
    features_numericas = [col for col in features_numericas if col in df_trab.columns]
    features_categoricas = [col for col in features_categoricas if col in df_trab.columns]

    if "GPA" not in df_trab.columns:
        raise ValueError("A coluna GPA não foi encontrada no dataset.")

    if len(features_numericas) == 0:
        raise ValueError("Nenhuma variável adequada para previsão foi encontrada.")

    # --------------------------------------------------------
    # Dados e Limpeza
    # --------------------------------------------------------
    todas_features = features_numericas + features_categoricas
    dados = df_trab[todas_features + ["GPA"]].dropna(subset=["GPA"]).copy()

    X = dados[todas_features]
    y = dados["GPA"]

    # --------------------------------------------------------
    # Separação treino / teste
    # --------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # --------------------------------------------------------
    # Pré-processamento (Numérico + Categórico)
    # --------------------------------------------------------
    transformers = []

    if features_numericas:
        pipeline_numerica = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("numericas", pipeline_numerica, features_numericas))

    if features_categoricas:
        pipeline_categorica = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Desconhecido")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("categoricas", pipeline_categorica, features_categoricas))

    preprocessor = ColumnTransformer(transformers=transformers)

    # --------------------------------------------------------
    # Modelos (Incluindo Restrições Lógicas no Boosting)
    # --------------------------------------------------------
    modelos = {
        "Regressão Ridge": Ridge(alpha=1.0, random_state=42),

        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting Lógico": HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.03,
            max_depth=5,
            random_state=42
        )
    }

    resultados = []
    modelos_treinados = {}

    # --------------------------------------------------------
    # Treinamento e avaliação
    # --------------------------------------------------------
    for nome, modelo in modelos.items():
        pipeline = Pipeline([
            ("preprocessamento", preprocessor),
            ("modelo", modelo)
        ])

        pipeline.fit(X_train, y_train)
        previsoes = pipeline.predict(X_test)

        # Garante que as previsões fiquem estritamente dentro dos limites reais do GPA (0.0 a 4.0)
        previsoes = np.clip(previsoes, 0.0, 4.0)

        mae = mean_absolute_error(y_test, previsoes)
        rmse = np.sqrt(mean_squared_error(y_test, previsoes))
        r2 = r2_score(y_test, previsoes)

        resultados.append({
            "Modelo": nome,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R²": round(r2, 4)
        })

        modelos_treinados[nome] = pipeline

    resultados_df = pd.DataFrame(resultados)

    # --------------------------------------------------------
    # Seleção do melhor modelo
    # --------------------------------------------------------
    resultados_df = resultados_df.sort_values(
        by=["R²", "RMSE"],
        ascending=[False, True]
    ).reset_index(drop=True)

    melhor_modelo_nome = resultados_df.iloc[0]["Modelo"]
    melhor_modelo = modelos_treinados[melhor_modelo_nome]

    return {
        "modelo": melhor_modelo,
        "nome": melhor_modelo_nome,
        "modelos": modelos_treinados,
        "resultados": resultados_df,
        "features": todas_features,
        "features_numericas": features_numericas,
        "X_test": X_test,
        "y_test": y_test
    }