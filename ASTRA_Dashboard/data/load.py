import os
import pandas as pd

def get_Dataset():
    dir_current = os.path.dirname(os.path.abspath(__file__))
    caminho = os.path.join(dir_current, 'global_university_students_performance_habits_10000.csv')

    if os.path.exists(caminho):
        return pd.read_csv(caminho)
    raise FileNotFoundError("Arquivo de dados CSV não encontrado.")