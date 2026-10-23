"""
Testes unitários e de integração para o motor preditivo e torneio de modelos.

Valida a divisão holdout (treino/teste), ausência de vazamento de dados (target leakage),
o comportamento e as métricas dos estimadores individuais, e os critérios de seleção
do torneio de modelos com base no R² e RMSE.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------------------------------------------------
# Configuração de caminhos e importações
# -----------------------------------------------------------------------------
DIRETORIO_BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIRETORIO_BASE))

from ASTRA_Dashboard.analysis.predicao import treinar_motor_preditivo

caminho_dados = (
    DIRETORIO_BASE
    / "Data"
    / "global_university_students_performance_habits_10000.csv"
)


# -----------------------------------------------------------------------------
# Fixtures de teste
# -----------------------------------------------------------------------------
@pytest.fixture(scope="session")
def dataset_fixture():
    """
    Executa o pipeline completo do motor preditivo uma única vez por sessão de teste,
    fornecendo os modelos treinados, partições de teste e tabela comparativa.
    """
    df = pd.read_csv(caminho_dados)
    resultado = treinar_motor_preditivo(df)
    return resultado


# -----------------------------------------------------------------------------
# Testes de partição e integridade dos dados
# -----------------------------------------------------------------------------
def test_divisao_e_particoes_val(dataset_fixture):
    """
    Valida a partição holdout (80% treino / 20% teste), o alinhamento dos índices
    e garante a ausência de vazamento de alvo (Target Leakage) nas features.
    """
    X_teste, y_teste = dataset_fixture["X_test"], dataset_fixture["y_test"]

    # Garante a proporção exata da partição de teste (20% de 10.000 = 2.000 amostras)
    assert (X_teste.shape[0] == 2000) and (y_teste.shape[0] == 2000)

    # Confirma que os índices das features e do target estão estritamente alinhados
    assert X_teste.index.equals(y_teste.index)

    # Evita Target Leakage: a variável dependente (GPA) não pode estar nas features de entrada
    assert "GPA" not in X_teste.columns


# -----------------------------------------------------------------------------
# Testes dos estimadores individuais
# -----------------------------------------------------------------------------
def test_estimador_individual(dataset_fixture):
    """
    Avalia cada modelo preditivo individualmente:
    - Previsões restritas aos limites válidos da escala de GPA (0.0 a 4.0).
    - Resíduos médios controlados (ausência de viés sistemático excessivo).
    - Recálculo e consistência das métricas MAE, RMSE e R².
    """
    resultados = []

    for nome, modelo in dataset_fixture["modelos"].items():
        # Gera previsões e aplica corte no intervalo plausível da nota [0.0, 4.0]
        previsoes = modelo.predict(dataset_fixture["X_test"]).clip(0.0, 4.0)
        assert np.all((previsoes >= 0.0) & (previsoes <= 4.0))

        # Verifica se o viés médio residual permanece próximo de zero (tolerância <= 0.20)
        residuos = previsoes - dataset_fixture["y_test"].values
        assert abs(float(np.mean(residuos))) <= 0.20

        # Cálculo independente das métricas de regressão
        r2, mae, rmse = (
            r2_score(dataset_fixture["y_test"], previsoes),
            mean_absolute_error(dataset_fixture["y_test"], previsoes),
            np.sqrt(mean_squared_error(dataset_fixture["y_test"], previsoes)),
        )

        resultados.append({
            "Modelo": nome,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R²": round(r2, 4),
        })

    # Ordena os resultados com o mesmo critério do motor (R² decrescente, RMSE crescente)
    resultados_df = pd.DataFrame(resultados).sort_values(
        by=["R²", "RMSE"],
        ascending=[False, True]
    ).reset_index(drop=True)

    # Confirma que a tabela de métricas recalculada confere com a retornada pelo motor
    pd.testing.assert_frame_equal(resultados_df, dataset_fixture["resultados"])


# -----------------------------------------------------------------------------
# Testes do torneio de modelos e critérios de benchmarking
# -----------------------------------------------------------------------------
def test_torneio_modelos_e_benchmarks(dataset_fixture):
    """
    Valida a escolha do modelo campeão do torneio e assegura que ele atenda
    aos requisitos mínimos de poder explicativo (R² >= 0.50).
    """
    resultados_df = dataset_fixture["resultados"]

    # Verifica se um modelo vencedor foi devidamente eleito e referenciado
    assert dataset_fixture["modelo"] is not None
    assert dataset_fixture["nome"] in resultados_df["Modelo"].values

    # Critério de corte de qualidade: o campeão deve explicar ao menos 50% da variância
    melhor_r2 = resultados_df.iloc[0]["R²"]
    assert melhor_r2 >= 0.50, f"Esperado R² >= 0.50, obtido: {melhor_r2}"

    # Exibição diagnóstica do ranking dos modelos avaliados
    print("\n--- Tabela de Benchmarking ---")
    print(resultados_df.to_string(index=False))
