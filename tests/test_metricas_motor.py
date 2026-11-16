"""Trava as métricas do motor preditivo numa faixa, para o artigo não divergir do código."""
from ASTRA_Dashboard.analysis.predicao import treinar_motor_preditivo
from ASTRA_Dashboard.data.load import get_Dataset


def test_metricas_do_motor_na_faixa_do_artigo():
    res = treinar_motor_preditivo(get_Dataset())["resultados"]
    assert len(res) == 3
    for _, r in res.iterrows():
        assert 0.70 <= r["R²"] <= 0.76
        assert 0.14 <= r["MAE"] <= 0.17
        assert 0.70 <= r["R² (treino)"] <= 0.80
        assert 0.13 <= r["MAE (treino)"] <= 0.16
