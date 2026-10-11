import pandas as pd
import numpy as np
from scipy.stats import pearsonr

try:
    from data.load import get_Dataset
except ImportError:
    from ..data.load import get_Dataset


HABIT_VARIABLES = [
    'study_hours_per_day',
    'class_attendance_percent',
    'sleep_hours',
    'screen_time_hours',
    'social_media_hours',
    'gaming_hours',
    'exercise_hours_per_week',
    'mental_stress_level',
    'AI_tool_usage_hours',
    'exam_preparation_days',
    'coffee_consumption_per_day',
    'extracurricular_hours_per_week'
]

PERFORMANCE_VARIABLES = ['GPA', 'final_exam_score', 'assignment_score']

HABIT_RELATIONSHIPS = [
    ('sleep_hours', 'mental_stress_level'),
    ('exercise_hours_per_week', 'mental_stress_level'),
    ('screen_time_hours', 'social_media_hours')
]


def _correlacao_pearson(df, coluna_x, coluna_y):
    detalhes = _detalhes_correlacao(df, coluna_x, coluna_y)
    return detalhes['correlacao'], detalhes['pares_validos']


def _detalhes_correlacao(df, coluna_x, coluna_y):
    if coluna_x not in df.columns or coluna_y not in df.columns:
        return {
            'correlacao': None,
            'valor_p': None,
            'pares_validos': 0,
            'sinal': 'indisponível',
            'intensidade': 'não calculável',
            'valida': False
        }

    pares = df[[coluna_x, coluna_y]].apply(pd.to_numeric, errors='coerce').dropna()
    pares = pares.replace([float('inf'), float('-inf')], pd.NA).dropna()
    if len(pares) < 2 or pares[coluna_x].nunique() < 2 or pares[coluna_y].nunique() < 2:
        return {
            'correlacao': None,
            'valor_p': None,
            'pares_validos': len(pares),
            'sinal': 'indisponível',
            'intensidade': 'não calculável',
            'valida': False
        }

    resultado = pearsonr(pares[coluna_x], pares[coluna_y])
    correlacao = float(resultado.statistic)
    valor_p = float(resultado.pvalue)
    if not np.isfinite(correlacao) or not np.isfinite(valor_p):
        return {
            'correlacao': None,
            'valor_p': None,
            'pares_validos': len(pares),
            'sinal': 'indisponível',
            'intensidade': 'não calculável',
            'valida': False
        }

    return {
        'correlacao': correlacao,
        'valor_p': valor_p,
        'pares_validos': len(pares),
        'sinal': 'positiva' if correlacao > 0 else 'negativa',
        'intensidade': _classificar_intensidade(correlacao),
        'valida': True
    }


def _classificar_intensidade(correlacao):
    intensidade = abs(correlacao)
    if intensidade < 0.2:
        return 'muito fraca'
    if intensidade < 0.4:
        return 'fraca'
    if intensidade < 0.6:
        return 'moderada'
    if intensidade < 0.8:
        return 'forte'
    return 'muito forte'


def calcular_correlacoes(df=None, habitos=None, desempenhos=None):
    if df is None:
        df = get_Dataset()
    habitos = list(dict.fromkeys(HABIT_VARIABLES if habitos is None else habitos))
    desempenhos = list(dict.fromkeys(PERFORMANCE_VARIABLES if desempenhos is None else desempenhos))

    habit_to_performance = []
    for habito in habitos:
        for desempenho in desempenhos:
            detalhes = _detalhes_correlacao(df, habito, desempenho)
            habit_to_performance.append({
                'variavel': habito,
                'desempenho': desempenho,
                **detalhes
            })

    habit_to_habit = []
    for variavel_x, variavel_y in HABIT_RELATIONSHIPS:
        detalhes = _detalhes_correlacao(df, variavel_x, variavel_y)
        habit_to_habit.append({
            'variavel_x': variavel_x,
            'variavel_y': variavel_y,
            **detalhes
        })

    return {
        'habito_desempenho': habit_to_performance,
        'habito_habito': habit_to_habit,
        'habitos_disponiveis': [coluna for coluna in habitos if coluna in df.columns],
        'desempenhos_disponiveis': [coluna for coluna in desempenhos if coluna in df.columns],
        'combinacoes_esperadas': len(habitos) * len(desempenhos)
    }


def calcular_matriz_correlacoes(df=None, habitos=None, desempenhos=None):
    if df is None:
        df = get_Dataset()

    habitos = list(dict.fromkeys(HABIT_VARIABLES if habitos is None else habitos))
    desempenhos = list(dict.fromkeys(PERFORMANCE_VARIABLES if desempenhos is None else desempenhos))

    variaveis = [
        coluna
        for coluna in habitos + desempenhos
        if coluna in df.columns
    ]

    return (
        df[variaveis]
        .apply(pd.to_numeric, errors="coerce")
        .replace([float("inf"), float("-inf")], float("nan"))
        .corr(method="pearson")
    )
