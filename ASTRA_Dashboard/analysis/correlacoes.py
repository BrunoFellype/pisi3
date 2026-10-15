import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

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
            'spearman': None,
            'spearman_valor_p': None,
            'pares_validos': 0,
            'sinal': 'indisponível',
            'intensidade': 'não calculável',
            'valida': False
        }

    pares = df[[coluna_x, coluna_y]].apply(pd.to_numeric, errors='coerce')
    pares = pares.replace([float('inf'), float('-inf')], np.nan).dropna()
    if len(pares) < 2 or pares[coluna_x].nunique() < 2 or pares[coluna_y].nunique() < 2:
        return {
            'correlacao': None,
            'valor_p': None,
            'spearman': None,
            'spearman_valor_p': None,
            'pares_validos': len(pares),
            'sinal': 'indisponível',
            'intensidade': 'não calculável',
            'valida': False
        }

    resultado = pearsonr(pares[coluna_x], pares[coluna_y])
    resultado_spearman = spearmanr(pares[coluna_x], pares[coluna_y])
    correlacao = float(resultado.statistic)
    valor_p = float(resultado.pvalue)
    spearman = float(resultado_spearman.statistic)
    spearman_valor_p = float(resultado_spearman.pvalue)
    if not all(np.isfinite(valor) for valor in (correlacao, valor_p, spearman, spearman_valor_p)):
        return {
            'correlacao': None,
            'valor_p': None,
            'spearman': None,
            'spearman_valor_p': None,
            'pares_validos': len(pares),
            'sinal': 'indisponível',
            'intensidade': 'não calculável',
            'valida': False
        }

    return {
        'correlacao': correlacao,
        'valor_p': valor_p,
        'spearman': spearman,
        'spearman_valor_p': spearman_valor_p,
        'pares_validos': len(pares),
        'sinal': 'positiva' if correlacao > 0 else 'negativa' if correlacao < 0 else 'nula',
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


def _ajustar_valores_p_fdr(resultados):
    validos = [resultado for resultado in resultados if resultado['valor_p'] is not None]
    valores_p = np.array([resultado['valor_p'] for resultado in validos], dtype=float)
    ordem = np.argsort(valores_p)
    ajustes = np.empty(len(valores_p), dtype=float)
    menor_ajuste = 1.0
    for posicao in range(len(ordem) - 1, -1, -1):
        indice = ordem[posicao]
        menor_ajuste = min(menor_ajuste, valores_p[indice] * len(valores_p) / (posicao + 1))
        ajustes[indice] = menor_ajuste
    for resultado, ajuste in zip(validos, ajustes):
        resultado['valor_p_ajustado_fdr'] = float(np.clip(ajuste, 0, 1))
    for resultado in resultados:
        resultado.setdefault('valor_p_ajustado_fdr', None)
    return len(validos)


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

    comparacoes_habito_desempenho = _ajustar_valores_p_fdr(habit_to_performance)
    comparacoes_habito_habito = _ajustar_valores_p_fdr(habit_to_habit)

    return {
        'habito_desempenho': habit_to_performance,
        'habito_habito': habit_to_habit,
        'habitos_disponiveis': [coluna for coluna in habitos if coluna in df.columns],
        'desempenhos_disponiveis': [coluna for coluna in desempenhos if coluna in df.columns],
        'combinacoes_esperadas': len(habitos) * len(desempenhos),
        'comparacoes_validas': comparacoes_habito_desempenho + comparacoes_habito_habito,
        'comparacoes_validas_por_familia': {
            'habito_desempenho': comparacoes_habito_desempenho,
            'habito_habito': comparacoes_habito_habito
        },
        'ajuste_valores_p': 'Benjamini-Hochberg (FDR), aplicado separadamente por família'
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
