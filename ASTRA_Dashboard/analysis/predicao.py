import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
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


RANDOM_STATE = 42

# Hábitos usados para definir o perfil comportamental (sem GPA: evita vazamento do alvo)
FEATURES_HABITOS_CLUSTER = [
    "study_hours_per_day",
    "sleep_hours",
    "mental_stress_level",
    "class_attendance_percent",
    "social_media_hours"
]

NOMES_PERFIS = ["Perfil Comportamental A", "Perfil Comportamental B", "Perfil Comportamental C"]


def _avaliar(y_real, y_previsto):
    return {
        "MAE": round(mean_absolute_error(y_real, y_previsto), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_real, y_previsto))), 4),
        "R²": round(r2_score(y_real, y_previsto), 4),
    }


def atribuir_perfil(motor, habitos):
    """
    Classifica novas rotinas em um dos perfis comportamentais aprendidos no treino.
    `habitos` é um DataFrame com as colunas de FEATURES_HABITOS_CLUSTER.
    """
    if motor.get("kmeans_perfil") is None:
        return pd.Series(["Perfil Geral"] * len(habitos), index=habitos.index)

    colunas = motor["features_habitos_cluster"]
    dados = habitos[colunas].fillna(motor["medias_habitos_treino"])
    rotulos = motor["kmeans_perfil"].predict(motor["scaler_perfil"].transform(dados))
    return pd.Series(rotulos, index=habitos.index).map(motor["mapa_perfis"])


def treinar_motor_preditivo(df):
    """
    Treina e compara modelos para previsão do GPA, alinhado com a clusterização
    comportamental (livre de Target Leakage).
    """
    df_trab = df.copy()

    if "GPA" not in df_trab.columns:
        raise ValueError("A coluna GPA não foi encontrada no dataset.")

    features_habitos_cluster = [col for col in FEATURES_HABITOS_CLUSTER if col in df_trab.columns]

    # Variáveis utilizadas para previsão final do GPA
    features_numericas = [
        "study_hours_per_day",
        "sleep_hours",
        "mental_stress_level",
        "class_attendance_percent",
        "social_media_hours",
        "age"
    ]
    features_numericas = [col for col in features_numericas if col in df_trab.columns]

    if len(features_numericas) == 0:
        raise ValueError("Nenhuma variável adequada para previsão foi encontrada.")

    # --------------------------------------------------------
    # Separação treino / teste (ANTES de qualquer ajuste)
    # --------------------------------------------------------
    dados = df_trab.dropna(subset=["GPA"]).copy()
    treino, teste = train_test_split(dados, test_size=0.20, random_state=RANDOM_STATE)
    treino = treino.copy()
    teste = teste.copy()

    # --------------------------------------------------------
    # Perfil comportamental (K-Means, k = 3) ajustado SÓ no treino.
    # Usa apenas hábitos, sem o GPA, para não vazar o alvo.
    # Os estudantes de teste são apenas classificados nos perfis já aprendidos.
    # --------------------------------------------------------
    motor_perfil = {
        "kmeans_perfil": None,
        "scaler_perfil": None,
        "mapa_perfis": {},
        "features_habitos_cluster": features_habitos_cluster,
        "medias_habitos_treino": None,
    }
    features_categoricas = []

    if len(features_habitos_cluster) == len(FEATURES_HABITOS_CLUSTER) and len(treino) >= 3:
        medias_treino = treino[features_habitos_cluster].mean()
        scaler_perfil = StandardScaler()
        X_hab_treino = scaler_perfil.fit_transform(treino[features_habitos_cluster].fillna(medias_treino))

        kmeans_perfil = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
        ids_treino = kmeans_perfil.fit_predict(X_hab_treino)

        # Nomeia os perfis pela média de GPA do TREINO (A = maior GPA médio)
        ordem = (
            pd.Series(treino["GPA"].values, index=ids_treino)
            .groupby(level=0).mean()
            .sort_values(ascending=False)
            .index.tolist()
        )
        mapa_perfis = {cid: NOMES_PERFIS[pos] for pos, cid in enumerate(ordem)}

        motor_perfil.update({
            "kmeans_perfil": kmeans_perfil,
            "scaler_perfil": scaler_perfil,
            "mapa_perfis": mapa_perfis,
            "medias_habitos_treino": medias_treino,
        })
        treino["cluster_perfil"] = pd.Series(ids_treino, index=treino.index).map(mapa_perfis)
        teste["cluster_perfil"] = atribuir_perfil(motor_perfil, teste)
        features_categoricas = ["cluster_perfil"]

    todas_features = features_numericas + features_categoricas

    X_train, y_train = treino[todas_features], treino["GPA"]
    X_test, y_test = teste[todas_features], teste["GPA"]

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
    # Modelos
    # --------------------------------------------------------
    modelos = {
        "Regressão Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),

        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.03,
            max_depth=5,
            random_state=RANDOM_STATE
        )
    }

    resultados = []
    modelos_treinados = {}

    # --------------------------------------------------------
    # Treinamento e avaliação (teste) + comparação com o treino (overfitting)
    # --------------------------------------------------------
    for nome, modelo in modelos.items():
        pipeline = Pipeline([
            ("preprocessamento", preprocessor),
            ("modelo", modelo)
        ])

        pipeline.fit(X_train, y_train)

        # Garante que as previsões fiquem dentro dos limites reais do GPA (0.0 a 4.0)
        previsoes_teste = np.clip(pipeline.predict(X_test), 0.0, 4.0)
        previsoes_treino = np.clip(pipeline.predict(X_train), 0.0, 4.0)

        metricas_teste = _avaliar(y_test, previsoes_teste)
        metricas_treino = _avaliar(y_train, previsoes_treino)

        resultados.append({
            "Modelo": nome,
            **metricas_teste,
            "R² (treino)": metricas_treino["R²"],
            "RMSE (treino)": metricas_treino["RMSE"],
        })

        modelos_treinados[nome] = pipeline

    resultados_df = pd.DataFrame(resultados)

    # --------------------------------------------------------
    # Seleção do melhor modelo (maior R² no TESTE; desempate pelo menor RMSE)
    # --------------------------------------------------------
    resultados_df = resultados_df.sort_values(
        by=["R²", "RMSE"],
        ascending=[False, True]
    ).reset_index(drop=True)

    melhor_modelo_nome = resultados_df.iloc[0]["Modelo"]
    melhor_modelo = modelos_treinados[melhor_modelo_nome]

    # --------------------------------------------------------
    # Importância das variáveis (por permutação, no conjunto de teste).
    # Funciona para qualquer modelo, inclusive o HistGradientBoosting,
    # que não possui o atributo feature_importances_.
    # --------------------------------------------------------
    perm = permutation_importance(
        melhor_modelo, X_test, y_test,
        n_repeats=10, random_state=RANDOM_STATE, scoring="r2"
    )
    importancia_df = pd.DataFrame({
        "Variável": todas_features,
        "Queda no R²": perm.importances_mean,
        "Desvio": perm.importances_std,
    }).sort_values("Queda no R²", ascending=False).reset_index(drop=True)

    return {
        "modelo": melhor_modelo,
        "nome": melhor_modelo_nome,
        "modelos": modelos_treinados,
        "resultados": resultados_df,
        "importancia": importancia_df,
        "features": todas_features,
        "features_numericas": features_numericas,
        "X_test": X_test,
        "y_test": y_test,
        **motor_perfil,
    }
