import pandas as pd
import numpy as np

def tratar_dados_ia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trata os valores ausentes de favorite_AI_tool sem imputações enviesadas:
    - 1.276 estudantes não utilizam IA (AI_tool_usage_hours == 0)
    - 444 estudantes utilizam IA (AI_tool_usage_hours > 0), mas não possuem ferramenta predileta declarada.
    """
    df = df.copy()
    
    condicoes = [
        (df['favorite_AI_tool'].isna()) & (df['AI_tool_usage_hours'] == 0.0),
        (df['favorite_AI_tool'].isna()) & (df['AI_tool_usage_hours'] > 0.0)
    ]
    escolhas = ['Não Utiliza IA', 'Usuário sem Favorita']
    
    df['status_ia'] = np.select(condicoes, escolhas, default=df['favorite_AI_tool'])
    df['usa_ia'] = df['AI_tool_usage_hours'] > 0
    return df


def adicionar_faixas_comportamentais(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona colunas categóricas derivadas para enriquecer as análises estatísticas e visualizações:
    - Faixa de sono (Crítico, Alerta, Adequado, Alto)
    - Nível categórico de estresse (Baixo, Moderado, Elevado)
    - Carga de estudo (Baixa, Média, Alta)
    """
    df = df.copy()
    
    # Faixas de Sono
    df['faixa_sono'] = pd.cut(
        df['sleep_hours'],
        bins=[0, 5, 7, 9, 24],
        labels=["< 5h (Crítico)", "5h-7h (Alerta)", "7h-9h (Adequado)", "> 9h (Alto)"]
    )
    
    # Nível de Estresse
    df['categoria_estresse'] = pd.cut(
        df['mental_stress_level'],
        bins=[0, 3.5, 6.5, 10],
        labels=["Baixo (1.0 - 3.5)", "Moderado (3.6 - 6.5)", "Elevado (6.6 - 10.0)"]
    )
    
    # Nível de Dedicação aos Estudos
    df['faixa_estudo'] = pd.cut(
        df['study_hours_per_day'],
        bins=[0, 2.5, 4.5, 24],
        labels=["Baixo (< 2.5h)", "Médio (2.5h - 4.5h)", "Alto (> 4.5h)"]
    )
    
    return df


def preparar_dataset_completo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executa o pipeline completo de pré-processamento e engenharia de atributos.
    """
    df = tratar_dados_ia(df)
    df = adicionar_faixas_comportamentais(df)
    return df
