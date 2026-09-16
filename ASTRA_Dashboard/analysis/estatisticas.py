import pandas as pd

try:
    from data.load import get_Dataset
except ImportError:
    from ..data.load import get_Dataset


FINAL_EXAM_COLUMN = 'final_exam_score'


def _obter_serie(df, coluna):
    if df is None:
        df = get_Dataset()
    if coluna not in df.columns:
        return None
    return pd.to_numeric(df[coluna], errors='coerce').dropna()


def calcular_media(df=None, coluna=FINAL_EXAM_COLUMN):
    serie = _obter_serie(df, coluna)
    return None if serie is None or serie.empty else float(serie.mean())


def calcular_mediana(df=None, coluna=FINAL_EXAM_COLUMN):
    serie = _obter_serie(df, coluna)
    return None if serie is None or serie.empty else float(serie.median())


def calcular_intervalo_interquartil(df=None, coluna=FINAL_EXAM_COLUMN):
    serie = _obter_serie(df, coluna)
    if serie is None or serie.empty:
        return None
    return float(serie.quantile(0.75) - serie.quantile(0.25))


def calcular_assimetria(df=None, coluna=FINAL_EXAM_COLUMN):
    serie = _obter_serie(df, coluna)
    if serie is None or len(serie) < 3:
        return None
    assimetria = serie.skew()
    return None if pd.isna(assimetria) else float(assimetria)


def diagnosticar_efeito_teto(df=None, coluna=FINAL_EXAM_COLUMN):
    serie = _obter_serie(df, coluna)
    if serie is None or serie.empty:
        return {
            'coluna': coluna,
            'disponivel': False,
            'efeito_teto': None,
            'limite_superior': None,
            'quantidade': 0,
            'quantidade_no_limite': 0,
            'proporcao_no_limite': None,
            'q1': None,
            'mediana': None,
            'q3': None
        }

    limite_superior = serie.max()
    q1 = serie.quantile(0.25)
    mediana = serie.median()
    q3 = serie.quantile(0.75)
    quantidade_no_limite = int((serie == limite_superior).sum())
    proporcao_no_limite = quantidade_no_limite / len(serie)

    return {
        'coluna': coluna,
        'disponivel': True,
        'efeito_teto': bool(q3 == limite_superior),
        'limite_superior': float(limite_superior),
        'quantidade': int(len(serie)),
        'quantidade_no_limite': quantidade_no_limite,
        'proporcao_no_limite': float(proporcao_no_limite),
        'q1': float(q1),
        'mediana': float(mediana),
        'q3': float(q3)
    }


def calcular_estatisticas(df=None, coluna=FINAL_EXAM_COLUMN):
    diagnostico = diagnosticar_efeito_teto(df, coluna)
    return {
        'coluna': coluna,
        'media': calcular_media(df, coluna),
        'mediana': calcular_mediana(df, coluna),
        'intervalo_interquartil': calcular_intervalo_interquartil(df, coluna),
        'assimetria': calcular_assimetria(df, coluna),
        'efeito_teto': diagnostico
    }