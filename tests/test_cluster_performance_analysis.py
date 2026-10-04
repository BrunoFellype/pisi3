import pandas as pd

from ASTRA_Dashboard.analysis.hipoteses import comparar_desempenho_por_cluster
from ASTRA_Dashboard.visualization.graphs import criar_grafico_desempenho_por_cluster


def test_comparar_desempenho_por_cluster_gera_tabela_e_graficos():
    df = pd.DataFrame(
        {
            "GPA": [3.8, 3.4, 2.2, 2.0, 3.1, 2.7],
            "final_exam_score": [95, 90, 75, 70, 82, 78],
            "assignment_score": [90, 88, 65, 60, 77, 74],
            "class_attendance_percent": [98, 95, 70, 64, 80, 72],
            "study_hours_per_day": [8, 7, 4, 3, 5, 4],
            "sleep_hours": [7.5, 7.2, 6.3, 5.8, 6.7, 6.4],
            "mental_stress_level": [3, 4, 7, 8, 5, 6],
        }
    )
    labels = [0, 0, 1, 1, 2, 2]

    resultado = comparar_desempenho_por_cluster(df, cluster_labels=labels, cluster_coluna="cluster_id")

    assert "tabela" in resultado
    assert list(resultado["tabela"]["cluster_id"]) == [0, 1, 2]
    assert resultado["tabela"]["n"].sum() == len(df)
    assert set(resultado["tabela"].columns) >= {"cluster_id", "n", "GPA_media", "final_exam_score_media", "assignment_score_media"}

    fig = criar_grafico_desempenho_por_cluster(resultado["tabela"])
    assert fig is not None
    assert "data" in fig
