import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def treinar_motor_preditivo(df):
    """
    Treina e compara modelos para previsão do GPA.
    """

    # Variáveis utilizadas para previsão
    features_numericas = [
        "study_hours_per_day",
        "sleep_hours",
        "mental_stress_level",
        "class_attendance_percent",
        "social_media_hours",
        "age"
    ]

    # Verifica quais colunas realmente existem
    features_numericas = [
        col for col in features_numericas
        if col in df.columns
    ]

    features = features_numericas

    if "GPA" not in df.columns:
        raise ValueError("A coluna GPA não foi encontrada no dataset.")

    if len(features) == 0:
        raise ValueError(
            "Nenhuma variável adequada para previsão foi encontrada."
        )

    # --------------------------------------------------------
    # Dados
    # --------------------------------------------------------

    dados = df[features + ["GPA"]].copy()

    # Remove registros sem GPA
    dados = dados.dropna(subset=["GPA"])

    X = dados[features]
    y = dados["GPA"]

    # --------------------------------------------------------
    # Separação treino / teste
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # --------------------------------------------------------
    # Pré-processamento
    # --------------------------------------------------------

    transformers = []

    if features_numericas:
        pipeline_numerica = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        transformers.append(
            ("numericas", pipeline_numerica, features_numericas)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    # --------------------------------------------------------
    # Modelos
    # --------------------------------------------------------

    modelos = {
        "Regressão Linear": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
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

        mae = mean_absolute_error(y_test, previsoes)

        rmse = np.sqrt(
            mean_squared_error(y_test, previsoes)
        )

        r2 = r2_score(y_test, previsoes)

        resultados.append({
            "Modelo": nome,
            "MAE": mae,
            "RMSE": rmse,
            "R²": r2
        })

        modelos_treinados[nome] = pipeline

    resultados_df = pd.DataFrame(resultados)

    # --------------------------------------------------------
    # Seleção do melhor modelo
    # --------------------------------------------------------

    # Maior R² = melhor capacidade explicativa.
    # Em caso de empate, menor RMSE.

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
        "features": features,
        "features_numericas": features_numericas,
        "X_test": X_test,
        "y_test": y_test
    }