import pandas as pd

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
    if coluna_x not in df.columns or coluna_y not in df.columns:
        return None, 0

    pares = df[[coluna_x, coluna_y]].apply(pd.to_numeric, errors='coerce').dropna()
    if len(pares) < 2 or pares[coluna_x].nunique() < 2 or pares[coluna_y].nunique() < 2:
        return None, len(pares)

    correlacao = pares[coluna_x].corr(pares[coluna_y], method='pearson')
    return (None if pd.isna(correlacao) else float(correlacao)), len(pares)


def calcular_correlacoes(df=None, habitos=None, desempenhos=None):
    if df is None:
        df = get_Dataset()
    habitos = HABIT_VARIABLES if habitos is None else list(habitos)
    desempenhos = PERFORMANCE_VARIABLES if desempenhos is None else list(desempenhos)

    habit_to_performance = []
    for habito in habitos:
        for desempenho in desempenhos:
            correlacao, quantidade = _correlacao_pearson(df, habito, desempenho)
            habit_to_performance.append({
                'variavel': habito,
                'desempenho': desempenho,
                'correlacao': correlacao,
                'pares_validos': quantidade
            })

    habit_to_habit = []
    for variavel_x, variavel_y in HABIT_RELATIONSHIPS:
        correlacao, quantidade = _correlacao_pearson(df, variavel_x, variavel_y)
        habit_to_habit.append({
            'variavel_x': variavel_x,
            'variavel_y': variavel_y,
            'correlacao': correlacao,
            'pares_validos': quantidade
        })

    return {
        'habito_desempenho': habit_to_performance,
        'habito_habito': habit_to_habit,
        'habitos_disponiveis': [coluna for coluna in habitos if coluna in df.columns],
        'desempenhos_disponiveis': [coluna for coluna in desempenhos if coluna in df.columns],
        'combinacoes_esperadas': len(habitos) * len(desempenhos)
    }


def calcular_matriz_correlacoes(df=None, habitos=None, desempenhos=None):
    return calcular_correlacoes(df, habitos, desempenhos)