import pandas as pd
import pytest

from ASTRA_Dashboard.analysis.correlacoes import (
    HABIT_VARIABLES,
    PERFORMANCE_VARIABLES,
    calcular_correlacoes,
    calcular_matriz_correlacoes,
)


def _dados_completos():
    valores = {}
    for indice, coluna in enumerate(HABIT_VARIABLES + PERFORMANCE_VARIABLES):
        valores[coluna] = [indice + 1, indice + 2, indice + 3, indice + 4]
    return pd.DataFrame(valores)


def test_processa_as_36_combinacoes_sem_duplicidade():
    resultado = calcular_correlacoes(_dados_completos())
    combinacoes = [
        (item["variavel"], item["desempenho"])
        for item in resultado["habito_desempenho"]
    ]

    assert len(combinacoes) == 36
    assert len(set(combinacoes)) == 36
    assert resultado["combinacoes_esperadas"] == 36


def test_preserva_correlacoes_positiva_e_negativa():
    dados = pd.DataFrame(
        {
            "habito_positivo": [1, 2, 3, 4],
            "habito_negativo": [4, 3, 2, 1],
            "desempenho": [1, 2, 3, 4],
        }
    )
    resultado = calcular_correlacoes(
        dados,
        habitos=["habito_positivo", "habito_negativo"],
        desempenhos=["desempenho"],
    )["habito_desempenho"]

    assert resultado[0]["correlacao"] == pytest.approx(1)
    assert resultado[0]["sinal"] == "positiva"
    assert resultado[0]["valor_p"] == pytest.approx(0)
    assert resultado[0]["intensidade"] == "muito forte"
    assert resultado[1]["correlacao"] == pytest.approx(-1)
    assert resultado[1]["sinal"] == "negativa"


def test_trata_constante_ausentes_inexistentes_e_pares_insuficientes():
    dados = pd.DataFrame(
        {
            "constante": [1, 1, 1],
            "desempenho": [1, 2, 3],
            "com_nulos": [1, None, 3],
            "com_um_par": [1, None, None],
        }
    )
    resultado = calcular_correlacoes(
        dados,
        habitos=["constante", "com_nulos", "com_um_par", "inexistente"],
        desempenhos=["desempenho"],
    )["habito_desempenho"]

    assert resultado[0]["correlacao"] is None
    assert resultado[0]["valida"] is False
    assert resultado[1]["correlacao"] == pytest.approx(1)
    assert resultado[1]["pares_validos"] == 2
    assert resultado[2]["correlacao"] is None
    assert resultado[2]["pares_validos"] == 1
    assert resultado[3]["correlacao"] is None
    assert resultado[3]["pares_validos"] == 0


def test_ignora_valores_nao_finitos_e_preserva_formato_da_matriz():
    dados = pd.DataFrame(
        {
            "habito": [1, 2, float("inf"), 4],
            "desempenho": [1, 2, 3, 4],
        }
    )
    matriz = calcular_matriz_correlacoes(
        dados, habitos=["habito"], desempenhos=["desempenho"]
    )

    assert list(matriz.index) == ["habito", "desempenho"]
    assert matriz.loc["habito", "desempenho"] == pytest.approx(1)
