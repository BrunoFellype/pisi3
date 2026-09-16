import pandas as pd
import os
from data.load import get_Dataset

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

def calcular_correlacoes(df, alvo='GPA', variaveis=None):
    if variaveis is None:
        variaveis = HABIT_VARIABLES

    resultados = []

    for variavel in variaveis:
        correlacao = df[variavel].corr(df[alvo])
        resultados.append({
            'variavel': variavel,
            'desempenho': alvo,
            'correlacao': correlacao
        })

    return resultados