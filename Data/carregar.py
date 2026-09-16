import os
import pandas as pd

def obter_caminho_dataset() -> str:
    """
    Localiza o caminho absoluto do dataset canônico na pasta Data/.
    """
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    raiz_projeto = os.path.abspath(os.path.join(diretorio_atual, ".."))
    
    candidatos = [
        os.path.join(raiz_projeto, "Data", "global_university_students_performance_habits_10000.csv"),
        os.path.join("Data", "global_university_students_performance_habits_10000.csv"),
        os.path.join("..", "Data", "global_university_students_performance_habits_10000.csv"),
    ]
    
    for caminho in candidatos:
        if os.path.exists(caminho):
            return os.path.abspath(caminho)
            
    raise FileNotFoundError("Arquivo 'global_university_students_performance_habits_10000.csv' não encontrado na pasta Data/.")


def carregar_dados_brutos() -> pd.DataFrame:
    """
    Carrega o dataset em formato Pandas DataFrame a partir do arquivo CSV canônico.
    """
    caminho = obter_caminho_dataset()
    return pd.read_csv(caminho)
