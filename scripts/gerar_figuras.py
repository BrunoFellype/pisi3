"""
Geração das figuras científicas da Análise Exploratória (Seção 5) do artigo ASTRA.

Gera em alta resolução (300 DPI) as Figuras 1, 2 e 3 a partir da base de dados de
estudantes universitários (N = 10.000), seguindo os padrões editoriais para publicação.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# -----------------------------------------------------------------------------
# Configuração de Caminhos e Resolução do Repositório
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

# Localização do dataset primário (10.000 registros, 27 variáveis comportamentais)
caminho_dados = (
    BASE_DIR
    / "Data"
    / "global_university_students_performance_habits_10000.csv"
)
if not caminho_dados.exists():
    # Caminho alternativo caso executado em subsistema de dashboard
    caminho_dados = (
        BASE_DIR
        / "ASTRA_Dashboard"
        / "data"
        / "global_university_students_performance_habits_10000.csv"
    )

# Diretório de destino padronizado para as imagens exportadas
PASTA_SAIDA = BASE_DIR / "figuras_artigo"
PASTA_SAIDA.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# Configuração Estética Global (Padrão SBC / ABNT para Artigos Científicos)
# -----------------------------------------------------------------------------
def configurar_estilo_visual():
    """
    Define os parâmetros tipográficos e estéticos globais das figuras.
    
    Adota fonte serifada (padrão LaTeX / SBC), paleta de cores sóbria ('deep'),
    estilo de grid suave ('whitegrid') e dimensionamento de rótulos otimizado
    para legibilidade em colunas duplas de publicações científicas.
    """
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({
        "font.family": "serif",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "figure.titlesize": 13,
    })


# -----------------------------------------------------------------------------
# FIGURA 1: Comparativo Efeito Teto (Exame Final) vs. GPA Contínuo
# -----------------------------------------------------------------------------
def gerar_figura_1_efeito_teto_vs_gpa(df: pd.DataFrame, pasta_saida: Path):
    """
    Gera a Figura 1 do Artigo:
    'Comparativo de distribuição de frequência entre a nota do exame final e o GPA contínuo'.
    """
    print("Gerando Figura 1: Comparativo Efeito Teto vs. GPA Contínuo...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)

    # Subgráfico 1.1: Histograma com KDE da nota do Exame Final (Efeito Teto)
    sns.histplot(
        df["final_exam_score"],
        bins=30,
        kde=True,
        color="#d9534f",
        edgecolor="black",
        ax=axes[0],
    )
    axes[0].set_title("Exame Final (Efeito Teto em 100)")
    axes[0].set_xlabel("Nota do Exame Final (0 a 100)")
    axes[0].set_ylabel("Frequência de Alunos")
    axes[0].axvline(
        100,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Ponto de Saturação (100)",
    )
    axes[0].legend(loc="upper left", fontsize=8)

    # Subgráfico 1.2: Histograma com KDE do GPA contínuo (Alvo selecionado)
    sns.histplot(
        df["GPA"],
        bins=30,
        kde=True,
        color="#2a4b7c",
        edgecolor="black",
        ax=axes[1],
    )
    axes[1].set_title("Distribuição do GPA (Métrica Contínua)")
    axes[1].set_xlabel("Grade Point Average (GPA: 0.0 a 4.0)")
    axes[1].set_ylabel("Frequência de Alunos")

    plt.tight_layout()
    caminho_fig1 = pasta_saida / "figura1_efeito_teto_vs_gpa.png"
    fig.savefig(caminho_fig1, dpi=300)
    plt.close(fig)
    print(f" -> Salvo com sucesso em: {caminho_fig1.name}")


# -----------------------------------------------------------------------------
# FIGURA 2: Matriz de Correlação de Pearson dos Hábitos Comportamentais e GPA
# -----------------------------------------------------------------------------
def gerar_figura_2_heatmap_correlacao(df: pd.DataFrame, pasta_saida: Path):
    """
    Gera a Figura 2 do Artigo:
    'Matriz de correlação de Pearson entre hábitos comportamentais e o coeficiente de rendimento discente'.
    """
    print("Gerando Figura 2: Heatmap de Correlação de Pearson (12 Hábitos + GPA)...")

    # Relação exata das 12 variáveis de hábitos numéricos oficiais e o target GPA
    variaveis_correlacao = [
        "study_hours_per_day",
        "class_attendance_percent",
        "sleep_hours",
        "mental_stress_level",
        "social_media_hours",
        "gaming_hours",
        "screen_time_hours",
        "exercise_hours_per_week",
        "AI_tool_usage_hours",
        "coffee_consumption_per_day",
        "extracurricular_hours_per_week",
        "exam_preparation_days",
        "GPA",
    ]

    # Rótulos em português para publicação científica
    rotulos_legiveis = [
        "Horas Estudo/Dia",
        "Assiduidade (%)",
        "Horas Sono/Dia",
        "Nível Estresse",
        "Redes Sociais (h)",
        "Videogame (h)",
        "Tempo Tela (h)",
        "Exercício/Sem (h)",
        "Uso de IA (h)",
        "Consumo Café/Dia",
        "Extracurricular/Sem (h)",
        "Dias Prep. Exame",
        "GPA (Rendimento)",
    ]

    # Extração da matriz de correlação de Pearson
    matriz_corr = df[variaveis_correlacao].corr(method="pearson")
    matriz_corr.columns = rotulos_legiveis
    matriz_corr.index = rotulos_legiveis

    fig, ax = plt.subplots(figsize=(11, 9), dpi=300)

    # Renderização do Heatmap com gradiente divergente simétrico (-0.8 a +0.8)
    sns.heatmap(
        matriz_corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.8,
        vmax=0.8,
        square=True,
        annot_kws={"size": 7.5},
        cbar_kws={"shrink": 0.8, "label": "Coeficiente de Pearson (r)"},
        linewidths=0.5,
        ax=ax,
    )

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    caminho_fig2 = pasta_saida / "figura2_heatmap_correlacao.png"
    fig.savefig(caminho_fig2, dpi=300)
    plt.close(fig)
    print(f" -> Salvo com sucesso em: {caminho_fig2.name}")


# -----------------------------------------------------------------------------
# FIGURA 3: Dispersão Estudo vs. GPA estratificado por Estresse Mental
# -----------------------------------------------------------------------------
def gerar_figura_3_estudo_vs_gpa_estresse(df: pd.DataFrame, pasta_saida: Path):
    """
    Gera a Figura 3 do Artigo:
    'Relação entre horas de estudo e GPA sob diferentes regimes de estresse mental e qualidade de sono'.
    """
    print("Gerando Figura 3: Dispersão Estudo vs. GPA por Estresse Mental...")
    df_plot = df.copy()

    # Estratificação em 2 faixas psicoemocionais via quantis (mediana)
    df_plot["Faixa_Estresse"] = pd.qcut(
        df_plot["mental_stress_level"],
        q=2,
        labels=["Baixo/Médio Estresse", "Alto Estresse"],
    )

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)

    # Amostragem representativa aleatória fixada para evitar sobreposição excessiva de pontos
    sns.scatterplot(
        data=df_plot.sample(1500, random_state=42),
        x="study_hours_per_day",
        y="GPA",
        hue="Faixa_Estresse",
        palette={"Baixo/Médio Estresse": "#2a4b7c", "Alto Estresse": "#d9534f"},
        alpha=0.35,
        s=25,
        ax=ax,
    )

    # Reta de tendência e regressão linear para Baixo/Médio Estresse
    sns.regplot(
        data=df_plot[df_plot["Faixa_Estresse"] == "Baixo/Médio Estresse"],
        x="study_hours_per_day",
        y="GPA",
        scatter=False,
        color="#1f3b64",
        label="Tendência: Baixo/Médio Estresse",
        ax=ax,
    )

    # Reta de tendência e regressão linear para Alto Estresse
    sns.regplot(
        data=df_plot[df_plot["Faixa_Estresse"] == "Alto Estresse"],
        x="study_hours_per_day",
        y="GPA",
        scatter=False,
        color="#a72828",
        label="Tendência: Alto Estresse",
        ax=ax,
    )

    ax.set_title(
        "Impacto das Horas de Estudo no GPA por Nível de Estresse Mental", pad=12
    )
    ax.set_xlabel("Horas de Estudo por Dia")
    ax.set_ylabel("Grade Point Average (GPA)")
    ax.legend(title="Condição Psicoemocional", loc="upper left", frameon=True)

    plt.tight_layout()
    caminho_fig3 = pasta_saida / "figura3_estudo_vs_gpa_estresse.png"
    fig.savefig(caminho_fig3, dpi=300)
    plt.close(fig)
    print(f" -> Salvo com sucesso em: {caminho_fig3.name}")


# -----------------------------------------------------------------------------
# Orquestrador Principal de Execução
# -----------------------------------------------------------------------------
def main():
    """
    Função principal que orquestra o carregamento dos dados, validação de integridade,
    configuração de estilo editorial e geração sequencial das figuras do artigo.
    """
    print("=" * 78)
    print("PROJETO ASTRA - Geração de Figuras Científicas da Análise Exploratória")
    print("Autores: Bruno Vieira, Felipe Acioli, Heitor Quental, Lucas Santos, Renato Barros")
    print("=" * 78)

    # 1. Carregamento e validação inicial da base de dados
    if not caminho_dados.exists():
        raise FileNotFoundError(
            f"Arquivo de dados não localizado nos caminhos esperados:\n -> {caminho_dados}"
        )

    print(f"Carregando dataset de: {caminho_dados.resolve()}")
    df = pd.read_csv(caminho_dados)
    print(f"Dataset carregado com sucesso: {df.shape[0]} registros e {df.shape[1]} atributos.")

    # Validação do tamanho amostral conforme citado no artigo (10.000 discentes)
    if df.shape[0] != 10000:
        print(f"Aviso: Esperavam-se 10.000 registros, encontrados {df.shape[0]}.")

    # 2. Configurações estéticas globais
    configurar_estilo_visual()

    # 3. Geração sequencial das Figuras 1, 2 e 3 do Artigo
    gerar_figura_1_efeito_teto_vs_gpa(df, PASTA_SAIDA)
    gerar_figura_2_heatmap_correlacao(df, PASTA_SAIDA)
    gerar_figura_3_estudo_vs_gpa_estresse(df, PASTA_SAIDA)

    print("=" * 78)
    print(f"Concluído! Todas as 3 figuras foram exportadas em: {PASTA_SAIDA.resolve()}/")
    print("=" * 78)


if __name__ == "__main__":
    main()
