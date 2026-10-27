import numpy as np
import pandas as pd
import pytest

from ASTRA_Dashboard.analysis.clusters import executar_kmeans, FEATURES_KMEANS, K_AVALIADOS
from ASTRA_Dashboard.analysis.predicao import treinar_motor_preditivo, atribuir_perfil, FEATURES_HABITOS_CLUSTER
from ASTRA_Dashboard.data.load import get_Dataset


@pytest.fixture(scope="module")
def amostra():
    # Amostra fixa para os testes rodarem rápido
    return get_Dataset().sample(n=1500, random_state=0).reset_index(drop=True)


def test_gpa_nao_entra_no_kmeans(amostra):
    assert "GPA" not in FEATURES_KMEANS

    original = executar_kmeans(amostra, 3)[0]
    alterado = amostra.copy()
    alterado["GPA"] = np.random.default_rng(1).uniform(0, 4, len(alterado))
    novo = executar_kmeans(alterado, 3)[0]

    # Mudar o GPA não pode mudar os grupos
    assert (original["cluster_id"].values == novo["cluster_id"].values).all()


def test_kmeans_reprodutivel_e_completo(amostra):
    a = executar_kmeans(amostra, 3)
    b = executar_kmeans(amostra, 3)
    df_cluster, medias, ordem, mapa, metricas, silhueta = a

    assert (a[0]["cluster_id"].values == b[0]["cluster_id"].values).all()
    assert list(metricas["k"]) == K_AVALIADOS
    assert -1 <= silhueta <= 1
    assert set(df_cluster["cluster_id"].unique()) == {0, 1, 2}
    assert {"PC1", "PC2", "perfil_cluster"} <= set(df_cluster.columns)

    # Perfis ordenados do maior para o menor GPA médio
    gpas = [medias.loc[cid, "GPA"] for cid in ordem]
    assert gpas == sorted(gpas, reverse=True)

    variancia = df_cluster.attrs["pca_variancia_explicada"]
    assert len(variancia) == 2 and variancia[0] >= variancia[1] > 0


def test_inercia_diminui_com_k(amostra):
    metricas = executar_kmeans(amostra, 3)[4]
    inercias = list(metricas["Inércia"])
    assert inercias == sorted(inercias, reverse=True)


def test_motor_preditivo(amostra):
    motor = treinar_motor_preditivo(amostra)
    resultados = motor["resultados"]

    assert set(resultados["Modelo"]) == {"Regressão Ridge", "Random Forest", "Gradient Boosting"}
    assert {"MAE", "RMSE", "R²", "R² (treino)", "RMSE (treino)"} <= set(resultados.columns)
    assert motor["nome"] == resultados.iloc[0]["Modelo"]
    assert len(motor["X_test"]) == 300  # 20% de 1500

    # Previsões sempre dentro da escala do GPA
    previsoes = motor["modelo"].predict(motor["X_test"])
    assert np.clip(previsoes, 0, 4).shape == previsoes.shape

    # Importância por permutação para todas as variáveis de entrada
    assert list(motor["importancia"]["Variável"].sort_values()) == sorted(motor["features"])


def test_atribuir_perfil_usa_o_kmeans_do_treino(amostra):
    motor = treinar_motor_preditivo(amostra)
    perfis = atribuir_perfil(motor, motor["X_test"])

    # Mesmo perfil que o motor usou ao avaliar o teste
    assert (perfis.values == motor["X_test"]["cluster_perfil"].values).all()

    rotina = pd.DataFrame([{coluna: amostra[coluna].mean() for coluna in FEATURES_HABITOS_CLUSTER}])
    assert atribuir_perfil(motor, rotina).iloc[0] in set(motor["mapa_perfis"].values())
