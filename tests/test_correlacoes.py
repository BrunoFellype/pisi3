import numpy as np
import pandas as pd
import pytest

from ASTRA_Dashboard.analysis.correlacoes import (
    HABIT_VARIABLES,
    PERFORMANCE_VARIABLES,
    calcular_correlacoes,
    calcular_matriz_correlacoes,
)
from ASTRA_Dashboard.visualization.graphs import criar_heatmap_correlacao


def _dados_completos():
    valores = {}
    for indice, coluna in enumerate(HABIT_VARIABLES + PERFORMANCE_VARIABLES):
        valores[coluna] = [indice + 1, indice + 2, indice + 3, indice + 4]
    return pd.DataFrame(valores)


def test_calculo_correlacoes_estrutura():
    df_dummy = pd.DataFrame({
        "study_hours_per_day": [1.0, 2.0, 3.0, 4.0, 5.0],
        "class_attendance_percent": [60, 70, 80, 90, 100],
        "GPA": [2.0, 2.5, 3.0, 3.5, 4.0],
        "final_exam_score": [50, 60, 70, 80, 90]
    })

    res = calcular_correlacoes(df_dummy, habitos=["study_hours_per_day"], desempenhos=["GPA"])
    assert "habito_desempenho" in res
    assert len(res["habito_desempenho"]) == 1
    assert res["habito_desempenho"][0]["correlacao"] == pytest.approx(1.0, 0.01)


def test_matriz_correlacao():
    df_dummy = pd.DataFrame({
        "study_hours_per_day": [1.0, 2.0, 3.0],
        "GPA": [2.0, 3.0, 4.0]
    })
    matriz = calcular_matriz_correlacoes(df_dummy, habitos=["study_hours_per_day"], desempenhos=["GPA"])
    assert matriz.shape == (2, 2)
    assert matriz.loc["study_hours_per_day", "GPA"] == pytest.approx(1.0, 0.01)


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


def test_heatmap_correlacao_traduz_rotulos_e_formata_sem_alterar_coeficientes():
    colunas = HABIT_VARIABLES + PERFORMANCE_VARIABLES
    valores = np.eye(len(colunas))
    valores[0, 1] = valores[1, 0] = 0.7171817
    valores[2, 12] = valores[12, 2] = -0.427295
    matriz = pd.DataFrame(valores, index=colunas, columns=colunas)

    fig = criar_heatmap_correlacao(matriz, "Matriz de correlação")
    trace = fig.data[0]
    rotulos_esperados = [
        "Horas de estudo por dia",
        "Frequência nas aulas (%)",
        "Horas de sono",
        "Tempo de tela",
        "Uso de redes sociais",
        "Horas de jogos",
        "Exercícios por semana",
        "Nível de estresse mental",
        "Uso de ferramentas de IA",
        "Dias de preparação para provas",
        "Consumo diário de café",
        "Atividades extracurriculares",
        "Desempenho acadêmico (GPA)",
        "Nota da prova final",
        "Nota das atividades",
    ]

    assert list(trace.x) == rotulos_esperados
    assert list(trace.y) == rotulos_esperados
    assert trace.text[0][0] == "1,00"
    assert trace.text[0][1] == "0,72"
    assert trace.text[2][12] == "-0,43"
    assert np.array_equal(np.asarray(trace.z), valores)
    assert fig.layout.coloraxis.cmin == -1
    assert fig.layout.coloraxis.cmax == 1
    assert trace.xgap == 2
    assert trace.ygap == 2