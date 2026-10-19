import os
import sys
from functools import lru_cache
from pathlib import Path

DASHBOARD_ROOT = Path(__file__).resolve().parent
if str(DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_ROOT))

from dash import Dash, Input, Output, State, dcc, html, dash_table
import pandas as pd
import plotly.express as px
import numpy as np

from analysis.correlacoes import HABIT_VARIABLES, PERFORMANCE_VARIABLES
from analysis.clusters import executar_kmeans
from analysis.predicao import treinar_motor_preditivo
from analysis.correlacoes import calcular_correlacoes, calcular_matriz_correlacoes
from data.load import get_Dataset
from visualization.graphs import (
    criar_grafico_clusters, criar_grafico_gpa_major, criar_grafico_genero,
    criar_grafico_internet, criar_grafico_perfil_clusters, criar_grafico_renda,
    criar_grafico_distribuicao_gpa, criar_grafico_distribuicao_notas,
    criar_grafico_comparacao_modelos,
    criar_grafico_trabalho, criar_heatmap_correlacao,
    criar_grafico_real_vs_previsto, criar_grafico_dispersao_correlacao
)
from visualization.style import ASTRA_COLORS, estilizar_grafico

TAB_STYLE = {
    "backgroundColor": "#0F172A",
    "color": "#94A3B8",
    "borderBottom": "1px solid #1E293B",
    "padding": "12px 18px",
    "fontWeight": "600"
}

TAB_SELECTED_STYLE = {
    "backgroundColor": "#1E293B",
    "color": "#38BDF8",
    "borderTop": "3px solid #38BDF8",
    "borderBottom": "1px solid #1E293B",
    "padding": "12px 18px",
    "fontWeight": "700"
}

try:
    import sklearn
    SKLEARN_DISPONIVEL = True
except ImportError:
    SKLEARN_DISPONIVEL = False

app = Dash(__name__, title="ASTRA - Inteligência Acadêmica", suppress_callback_exceptions=True)

# Template HTML base com regras de hover de alto contraste
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* 1. Opções da Sidebar (Radio e Checkbox) */
            #trabalho label,
            .dash-radioitems label,
            #ativar-kmeans label,
            .dash-checklist label {
                color: #c084fc !important;
                cursor: pointer;
                font-size: 13px;
                transition: all 0.2s ease-in-out !important;
            }

            /* Efeito Hover nas opções de seleção */
            #trabalho label:hover,
            .dash-radioitems label:hover,
            #ativar-kmeans label:hover,
            .dash-checklist label:hover {
                color: #ffffff !important;
                text-shadow: 0 0 10px #ede4ff, 0 0 20px #c084fc !important;
            }

            /* 2. Destaque dos números dos sliders ao passar o mouse ou quando ativos */
            .rc-slider-mark-text:hover,
            .rc-slider-mark-text-active,
            div.rc-slider-mark span:hover {
                color: #ffffff !important;
                font-weight: 800 !important;
                text-shadow: 0 0 10px #ffffff !important;
            }

            /* 3. Trilho e marcador do Slider */
            .rc-slider-track {
                background-color: #8b5cf6 !important;
            }

            .rc-slider-handle {
                border-color: #c084fc !important;
                background-color: #201d2a !important;
            }

            .rc-slider-handle:hover,
            .rc-slider-handle-dragging.rc-slider-handle-dragging {
                border-color: #ffffff !important;
                box-shadow: 0 0 12px #c084fc !important;
            }

            #parametros-simulador .rc-slider-tooltip-inner {
                background-color: #F8FAFC !important;
                color: #1F2937 !important;
                border: 1px solid #CBD5E1 !important;
                font-weight: 700 !important;
            }

            #parametros-simulador .rc-slider-tooltip-arrow {
                border-top-color: #F8FAFC !important;
            }

            #parametros-simulador input[type="number"] {
                background-color: #F8FAFC !important;
                color: #1F2937 !important;
                border-color: #CBD5E1 !important;
            }

            #parametros-simulador input[type="number"]:focus,
            #parametros-simulador input[type="number"]:focus-visible {
                color: #111827 !important;
                border-color: #8B5CF6 !important;
                outline: 2px solid rgba(139, 92, 246, 0.35) !important;
            }

            #parametros-simulador input[type="number"]:disabled,
            #parametros-simulador input[type="number"][readonly] {
                background-color: #E5E7EB !important;
                color: #374151 !important;
                opacity: 1 !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

server = app.server
df_raw = get_Dataset()

CSS = {"backgroundColor": ASTRA_COLORS["bg_dark"], "color": ASTRA_COLORS["text_primary"], "fontFamily": "Plus Jakarta Sans, sans-serif", "minHeight": "100vh", "padding": "24px 32px"}
SIDEBAR = {"backgroundColor": "#0F172A", "borderRight": "1px solid #1E293B", "padding": "20px", "width": "285px", "flexShrink": 0}
CONTENT = {"flex": 1, "padding": "0 0 0 26px", "minWidth": 0}
CARD = {"background": "linear-gradient(135deg, #1E293B 0%, #0F172A 100%)", "border": "1px solid #334155", "borderTop": "3px solid #38BDF8", "borderRadius": "12px", "padding": "16px 20px"}
GRID_2 = {"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))", "gap": "20px", "alignIitems": "stretch"}
GRID_3 = {"display": "grid", "gridTemplateColumns": "repeat(3, minmax(0, 1fr))", "gap": "16px"}
GRID_4 = {"display": "grid", "gridTemplateColumns": "repeat(4, minmax(0, 1fr))", "gap": "12px"}


# Função auxiliar para garantir a cor lavanda visível diretamente nos números do Slider
def mark_style(text):
    return {
        "label": str(text),
        "style": {
            "color": "#c084fc",
            "fontSize": "11px",
            "fontWeight": "600"
        }
    }


def control_label(text):
    return html.Label(text, style={"color": "#ede4ff", "fontWeight": "600", "display": "block", "marginBottom": "8px"})


def multi_control(identifier, label, options):
    return html.Div([
        control_label(label),
        dcc.Dropdown(
            id=identifier,
            options=[{"label": str(item), "value": item} for item in options],
            value=options,
            multi=True,
            style={"color": "#0F172A"}
        )
    ], style={"marginBottom": "18px"})


def metric(label, value, delta=None):
    children = [
        html.Div(
            label,
            style={
                "color": "#c084fc",
                "fontSize": "0.82rem",
                "fontWeight": "600",
                "textTransform": "uppercase"
            }
        ),
        html.Div(
            value,
            style={
                "color": "#ede4ff",
                "fontSize": "1.85rem",
                "fontWeight": "800",
                "marginTop": "5px"
            }
        )
    ]
    if delta is not None:
        children.append(html.Div(delta, style={"color": "#34D399", "fontSize": "0.85rem"}))
    return html.Div(children, style=CARD)

def metric_featured(label, value, description=None):

    return html.Div(
        [
            # Identificação do indicador
            html.P(
                label.upper(),
                style={
                    "fontSize": "0.78rem",
                    "fontWeight": "700",
                    "letterSpacing": "1.2px",
                    "color": "#94A3B8",
                    "margin": "0 0 12px 0"
                }
            ),

            # Valor
            html.H2(
                value,
                style={
                    "fontSize": "2.2rem",
                    "fontWeight": "800",
                    "color": ASTRA_COLORS["accent_cyan"],
                    "margin": "0 0 8px 0"
                }
            ),

            # Descrição complementar
            html.P(
                description,
                style={
                    "fontSize": "0.88rem",
                    "color": "#94A3B8",
                    "margin": "0"
                }
            ) if description else None
        ],

        style={
            **CARD,
            "padding": "26px 30px",
            "borderLeft": f"4px solid {ASTRA_COLORS['accent_cyan']}",
            "marginBottom": "28px"
        }
    )

def graph(figure):
    return dcc.Graph(figure=figure, config={"displayModeBar": False}, style={"backgroundColor": ASTRA_COLORS["card_dark"]})

def dashboard_section(titulo, descricao, children):

    return html.Section(
        [
            html.Div(
                [
                    html.H2(
                        titulo,
                        style={
                            "fontSize": "1.3rem",
                            "fontWeight": "700",
                            "color": ASTRA_COLORS["text_primary"],
                            "marginBottom": "6px"
                        }
                    ),

                    html.P(
                        descricao,
                        style={
                            "color": "#94A3B8",
                            "fontSize": "0.9rem",
                            "marginBottom": "20px"
                        }
                    )
                ]
            ),

            *children
        ],
        style={
            "marginBottom": "36px"
        }
    )

def gpa_para_nota(gpa):
    return (gpa / 4) * 10


def filtered_data(cursos, anos, generos, idade, trabalho, estresse, frequencia, rendas, metodos):
    data = df_raw.copy()
    if cursos:
        data = data[data["major"].isin(cursos)]
    if anos:
        data = data[data["university_year"].isin(anos)]
    if generos:
        data = data[data["gender"].isin(generos)]
    data = data[(data["age"] >= idade[0]) & (data["age"] <= idade[1])]
    if trabalho == "Sim (Trabalha)":
        data = data[data["part_time_job"] == "Yes"]
    elif trabalho == "Não":
        data = data[data["part_time_job"] == "No"]
    data = data[(data["mental_stress_level"] >= estresse[0]) & (data["mental_stress_level"] <= estresse[1])]
    data = data[(data["class_attendance_percent"] >= frequencia[0]) & (data["class_attendance_percent"] <= frequencia[1])]
    if rendas:
        data = data[data["family_income_level"].isin(rendas)]
    if metodos:
        data = data[data["note_taking_method"].isin(metodos)]
    return data


def preparar_clusterizacao(data, k):
    modelo, labels = executar_kmeans(data.copy(), k)
    clustered = data.copy()
    clustered["cluster_id"] = labels
    medias_cluster = clustered.groupby("cluster_id")["GPA"].mean().sort_values()
    order = list(medias_cluster.index)
    names = {cluster_id: f"Perfil {i + 1}" for i, cluster_id in enumerate(order)}
    clustered["perfil_cluster"] = clustered["cluster_id"].map(names)
    means = clustered.groupby("cluster_id")[["GPA", "final_exam_score", "class_attendance_percent", "study_hours_per_day", "sleep_hours", "mental_stress_level", "social_media_hours"]].mean()
    return clustered, means, order, names


def project_section():
    return html.Section([
        html.H2(
            "Sobre o Projeto",
            style={
                "color": "#F8FAFC",
                "marginBottom": "12px"
            }
        ),

        html.P(
            "O ASTRA apresenta uma análise exploratória e preditiva "
            "de dados sobre hábitos e desempenho acadêmico universitário.",
            style={
                "color": "#CBD5E1",
                "fontSize": "1rem",
                "lineHeight": "1.7",
                "marginBottom": "24px"
            }
        ),

        # Objetivo principal
        html.Div([
            html.H3(
                "Objetivo principal",
                style={"color": "#C0A7EA", "marginTop": "0"}
            ),

            html.P(
                "Desenvolver uma plataforma analítica e preditiva que "
                "centraliza a rotina acadêmica universitária, "
                "correlacionando hábitos de estudo, sono, bem-estar e "
                "uso de ferramentas tecnológicas para antecipar o "
                "desempenho acadêmico (GPA) e mitigar o risco de "
                "sobrecarga estudantil.",
                style={
                    "color": "#CBD5E1",
                    "lineHeight": "1.7",
                    "marginBottom": "0"
                }
            )
        ], style={
            "backgroundColor": "#1E293B",
            "padding": "20px",
            "borderRadius": "10px",
            "border": "1px solid #334155",
            "marginBottom": "20px"
        }),

        # Objetivos secundários
        html.Details([
            html.Summary(
                "🎯 Objetivos secundários",
                style={
                    "cursor": "pointer",
                    "fontSize": "1.05rem",
                    "fontWeight": "700",
                    "color": "#F8FAFC"
                }
            ),

            dcc.Markdown("""
- **Investigação exploratória:** realizar a limpeza, transformação e análise exploratória de dados multivariados sobre hábitos de 10.000 estudantes universitários.
- **Identificação de fatores críticos:** mapear correlações estatísticas entre indicadores comportamentais e o rendimento acadêmico final.
- **Modelagem preditiva:** estruturar modelos de regressão capazes de estimar a pontuação acadêmica com base em registros diários.
- **Interface de suporte à decisão:** disponibilizar um painel interativo intuitivo que transforme métricas estatísticas em recomendações práticas.
            """, style={
                "color": "#CBD5E1",
                "lineHeight": "1.8",
                "marginTop": "14px"
            })
        ], style={
            "backgroundColor": "#0F172A",
            "padding": "18px",
            "borderRadius": "10px",
            "border": "1px solid #334155",
            "marginBottom": "20px"
        }),

        # Perguntas norteadoras
        html.Details([
            html.Summary(
                "❓ Perguntas norteadoras da pesquisa",
                style={
                    "cursor": "pointer",
                    "fontSize": "1.05rem",
                    "fontWeight": "700",
                    "color": "#F8FAFC"
                }
            ),

            html.Div([
                html.P(
                    "1. De que maneira o equilíbrio entre horas de sono "
                    "e nível de estresse modula o impacto das horas de "
                    "estudo no desempenho acadêmico (GPA)?",
                    style={
                        "color": "#CBD5E1",
                        "lineHeight": "1.7"
                    }
                ),

                html.P(
                    "2. Qual é a correlação do uso de ferramentas de "
                    "Inteligência Artificial no desempenho dos estudantes "
                    "quando contrastado com o tempo gasto em redes sociais "
                    "e a frequência às aulas?",
                    style={
                        "color": "#CBD5E1",
                        "lineHeight": "1.7",
                        "marginBottom": "0"
                    }
                )
            ], style={"marginTop": "14px"})
        ], style={
            "backgroundColor": "#0F172A",
            "padding": "18px",
            "borderRadius": "10px",
            "border": "1px solid #334155",
            "marginBottom": "20px"
        })
    ])

def methodology_section():
    return html.Section([
        html.H2(
            "Metodologia da Análise",
            style={
                "color": "#F8FAFC",
                "marginBottom": "12px"
            }
        ),

        html.P(
            "Esta seção apresenta os procedimentos utilizados para "
            "explorar os dados, investigar relações entre hábitos e "
            "desempenho acadêmico e identificar perfis de estudantes.",
            style={
                "color": "#CBD5E1",
                "fontSize": "1rem",
                "lineHeight": "1.7",
                "marginBottom": "24px"
            }
        ),

        # 1. Dados e preparação
        html.Details([
            html.Summary(
                "📂 1. Dados e preparação",
                style={
                    "cursor": "pointer",
                    "fontSize": "1.05rem",
                    "fontWeight": "700",
                    "color": "#C0A7EA"
                }
            ),

            dcc.Markdown("""
O estudo utiliza um conjunto de dados sobre estudantes universitários, disponibilizado no Kaggle, composto por **10.000 registros e 27 variáveis**.

Os dados abrangem características dos estudantes, hábitos cotidianos, utilização de ferramentas tecnológicas e indicadores de desempenho acadêmico.

A preparação dos dados é importante para verificar sua consistência e adequação às análises estatísticas e aos modelos utilizados no dashboard. 

Portanto, anterior a qualquer análise, foi feita uma investigação minunciosa buscando outliers, valores nulos, a interpretação do dataset e a definição

dos objetivos das análises dentro dos limites impostos pelas 27 variáveis.
            """, style={
                "color": "#CBD5E1",
                "lineHeight": "1.8",
                "marginTop": "14px"
            })
        ], style={
            "backgroundColor": "#0F172A",
            "padding": "18px",
            "borderRadius": "10px",
            "border": "1px solid #334155",
            "marginBottom": "16px"
        }),

        # 2. Análise exploratória e correlações
        html.Details([
            html.Summary(
                "📊 2. Análise exploratória e correlações",
                style={
                    "cursor": "pointer",
                    "fontSize": "1.05rem",
                    "fontWeight": "700",
                    "color": "#C0A7EA"
                }
            ),

            dcc.Markdown("""
A análise exploratória permite compreender as características do conjunto de dados e investigar possíveis relações entre os hábitos dos estudantes e seu desempenho acadêmico.

São utilizadas medidas estatísticas e visualizações para examinar a distribuição das variáveis e identificar padrões relevantes.

Para investigar as relações entre as variáveis, o dashboard utiliza:

- **Correlação de Pearson:** avalia a intensidade e a direção da relação linear entre duas variáveis.
- **Correlação de Spearman:** avalia a intensidade e a direção de relações monotônicas com base nos postos dos valores.

A análise também considera a significância estatística e a correção para múltiplas comparações, auxiliando na interpretação dos resultados encontrados. 

Sendo relevante destacar que dentro das 27 variáveis, 3 são consideradas essenciais: gpa, final_exam_score e assignment_score. Com o gpa sendo a variável target.
            """, style={
                "color": "#CBD5E1",
                "lineHeight": "1.8",
                "marginTop": "14px"
            })
        ], style={
            "backgroundColor": "#0F172A",
            "padding": "18px",
            "borderRadius": "10px",
            "border": "1px solid #334155",
            "marginBottom": "16px"
        }),

        # 3. Agrupamento de estudantes
        html.Details([
            html.Summary(
                "🧩 3. Agrupamento de estudantes — K-Means",
                style={
                    "cursor": "pointer",
                    "fontSize": "1.05rem",
                    "fontWeight": "700",
                    "color": "#C0A7EA"
                }
            ),

            dcc.Markdown("""
O algoritmo **K-Means** é utilizado para agrupar estudantes com características semelhantes, permitindo explorar diferentes perfis presentes no conjunto de dados.

O procedimento é executado para diferentes quantidades de grupos, com valores de **K entre 2 e 5**.

A avaliação dos agrupamentos considera duas métricas principais:

- **Inércia:** mede a soma dos quadrados das distâncias entre as observações e os respectivos centroides. Valores menores indicam grupos mais compactos, embora a métrica tenda a diminuir à medida que o número de grupos aumenta.
- **Silhouette Score:** avalia a proximidade das observações em relação ao próprio grupo em comparação com os demais. Valores mais elevados geralmente indicam melhor separação entre os agrupamentos.

Os resultados permitem comparar as configurações testadas e examinar as características médias dos grupos identificados.
            """, style={
                "color": "#CBD5E1",
                "lineHeight": "1.8",
                "marginTop": "14px"
            })
        ], style={
            "backgroundColor": "#0F172A",
            "padding": "18px",
            "borderRadius": "10px",
            "border": "1px solid #334155",
            "marginBottom": "20px"
        }),

        # 4. Modelagem preditiva e avaliação
        html.Details([
            html.Summary(
                "🧠 4. Modelagem preditiva e avaliação",
                style={
                    "cursor": "pointer",
                    "fontSize": "1.05rem",
                    "fontWeight": "700",
                    "color": "#C0A7EA"
                }
            ),

            dcc.Markdown("""
        A etapa de modelagem preditiva busca estimar o **GPA (Grade Point Average)** dos estudantes a partir das características utilizadas pelo motor preditivo.

        Para isso, o dashboard compara três algoritmos de regressão:

        - **Regressão Linear:** modelo linear que utiliza regularização para reduzir a influência de coeficientes excessivamente elevados e melhorar a estabilidade das estimativas.
        - **Random Forest Regressor:** combina diferentes árvores de decisão, permitindo representar relações não lineares entre as variáveis.
        - **Gradient Boosting Regressor:** constrói modelos sequencialmente, buscando reduzir os erros produzidos nas etapas anteriores.

        ### Treinamento e validação

        Os dados são divididos em dois conjuntos:

        - **80% para treinamento:** utilizados no ajuste dos modelos.
        - **20% para teste:** reservados para avaliar o desempenho preditivo em observações não utilizadas durante o treinamento.

        A comparação dos algoritmos permite identificar o modelo com melhor desempenho segundo os critérios definidos no motor preditivo.

        ### Métricas de avaliação

        A qualidade das previsões é examinada por meio de três métricas:

        | Métrica | Interpretação |
        |---|---|
        | **R²** | Indica a proporção da variabilidade do GPA explicada pelo modelo no conjunto avaliado. |
        | **MAE** | Representa a média dos erros absolutos entre os valores reais e previstos. |
        | **RMSE** | Mede a raiz do erro quadrático médio, atribuindo maior peso aos erros de maior magnitude. |

        Para MAE e RMSE, valores menores indicam previsões mais próximas dos valores observados. Para R², valores maiores geralmente indicam melhor ajuste.

        ### Validação visual

        O dashboard também apresenta um gráfico de dispersão comparando os valores reais e previstos do GPA no conjunto de teste.

        A linha de referência representa a situação ideal, na qual o valor previsto é igual ao valor observado.

        **Importante:** as estimativas representam associações aprendidas a partir dos dados. Elas não demonstram que alterações isoladas nos hábitos de um estudante causarão necessariamente mudanças em seu desempenho acadêmico.
            """, style={
                "color": "#CBD5E1",
                "lineHeight": "1.8",
                "marginTop": "14px"
            })

        ], style={
            "backgroundColor": "#0F172A",
            "padding": "18px",
            "borderRadius": "10px",
            "border": "1px solid #334155",
            "marginBottom": "20px"
        })
    ])

def dictionary_section():
    dictionary_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "Data",
        "dicionario_de_dados.md"
    )

    dictionary_text = (
        open(dictionary_path, encoding="utf-8").read()
        if os.path.exists(dictionary_path)
        else "Consulte o arquivo Data/dicionario_de_dados.md no repositório."
    )

    return html.Section([
        html.H2(
            "Dicionário de Dados",
            style={
                "color": "#F8FAFC",
                "marginBottom": "12px"
            }
        ),

        html.P(
            "Consulte a descrição das variáveis utilizadas no conjunto "
            "de dados, incluindo seus significados e características.",
            style={
                "color": "#CBD5E1",
                "lineHeight": "1.7",
                "marginBottom": "20px"
            }
        ),

        html.Details([
            html.Summary(
                "📖 Visualizar dicionário de dados",
                style={
                    "cursor": "pointer",
                    "fontSize": "1.05rem",
                    "fontWeight": "700",
                    "color": "#C0A7EA"
                }
            ),

            dcc.Markdown(
                dictionary_text,
                style={
                    "color": "#CBD5E1",
                    "marginTop": "18px",
                    "lineHeight": "1.8"
                }
            )

        ], style={
            "backgroundColor": "#0F172A",
            "padding": "18px",
            "borderRadius": "10px",
            "border": "1px solid #334155"
        })
    ])


def layout_controls():
    total = len(df_raw)
    return html.Aside([
        html.H2("⚡ Filtros ASTRA", style={"color": "#38BDF8"}),
        html.P("Personalize sua análise:", style={"color": "#c084fc", "fontSize": "13px"}),
        
        multi_control("cursos", "Cursos (Graduação):", sorted(df_raw["major"].dropna().unique())),
        multi_control("anos", "Período / Ano Académico:", sorted(df_raw["university_year"].dropna().unique())),
        multi_control("generos", "Género:", sorted(df_raw["gender"].dropna().unique())),
        
        control_label("Faixa Etária (Idade):"),
        dcc.RangeSlider(
            id="idade",
            min=int(df_raw["age"].min()),
            max=int(df_raw["age"].max()),
            value=[int(df_raw["age"].min()), int(df_raw["age"].max())],
            step=1,
            marks={i: mark_style(str(i)) for i in range(18, 30, 2)}
        ),
        html.Div(style={"height": "20px"}),
        
        control_label("Trabalho em Meio Período:"),
        dcc.RadioItems(
            id="trabalho",
            options=[{"label": x, "value": x} for x in ["Todos", "Sim (Trabalha)", "Não"]],
            value="Todos",
            labelStyle={"display": "block", "marginBottom": "6px", "cursor": "pointer"}
        ),
        html.Div(style={"height": "18px"}),
        
        control_label("Faixa de Estresse Mental (0 a 10):"),
        dcc.RangeSlider(
            id="estresse",
            min=0,
            max=10,
            value=[0, 10],
            step=0.5,
            marks={i: mark_style(str(i)) for i in range(0, 11, 2)}
        ),
        html.Div(style={"height": "20px"}),
        
        control_label("Frequência às Aulas (%):"),
        dcc.RangeSlider(
            id="frequencia",
            min=0,
            max=100,
            value=[0, 100],
            step=5,
            marks={i: mark_style(f"{i}%") for i in range(0, 101, 25)}
        ),
        html.Div(style={"height": "20px"}),
        
        multi_control("rendas", "Nível de Renda Familiar:", sorted(df_raw["family_income_level"].dropna().unique())),
        multi_control("metodos", "Método de Anotação:", sorted(df_raw["note_taking_method"].dropna().unique())),
        html.Hr(style={"borderColor": "#1E293B", "margin": "24px 0"}),
        
        html.H3("🤖 Machine Learning (K-Means)", style={"color": "#38BDF8"}),
        dcc.Checklist(
            id="ativar-kmeans",
            options=[{"label": " Ativar Clusterização K-Means", "value": "ativo"}],
            value=["ativo"],
            labelStyle={"cursor": "pointer"}
        ),
        html.Div(style={"height": "12px"}),
        control_label("Quantidade de Perfis (K):"),
        dcc.Slider(
            id="k-clusters",
            min=2,
            max=5,
            value=3,
            step=1,
            marks={i: mark_style(str(i)) for i in range(2, 6)}
        ),
    ], style=SIDEBAR)


app.layout = html.Div([
    layout_controls(),
    html.Main([
        html.Div([
            html.H1("ASTRA Analytics - Painel de Hábitos & Desempenho Académico",
                    style={"color": "#F8FAFC", "marginBottom": "4px"}),
            html.P("Plataforma analítica e preditiva da rotina e rendimento universitário.",
                    style={"color": "#c084fc", "marginBottom": "20px"}),
        ]),
        dcc.Tabs(
            id="main-tabs",
            value="tab-simulador",
            children=[
                dcc.Tab(
                    label="Visão Geral", 
                    value="tab-geral", 
                    style=TAB_STYLE, 
                    selected_style=TAB_SELECTED_STYLE
                    ),

                dcc.Tab(
                    label="Hábitos e Correlações", 
                    value="tab-correlacoes", 
                    style=TAB_STYLE, 
                    selected_style=TAB_SELECTED_STYLE
                    ),
                    
                dcc.Tab(
                    label="Perfis e Predição", 
                    value="tab-simulador", 
                    style=TAB_STYLE, 
                    selected_style=TAB_SELECTED_STYLE
                    ),

                dcc.Tab(
                    label="Metodologia", 
                    value="tab-metodologia", 
                    style=TAB_STYLE, 
                    selected_style=TAB_SELECTED_STYLE
                    ),
            ],
            style={"marginBottom": "24px"}
        ),
        html.Div(id="dashboard-content")
    ], style=CONTENT)
], style={**CSS, "display": "flex"})


@lru_cache(maxsize=1)
def obter_motor_preditivo():
    return treinar_motor_preditivo(df_raw)


def layout_simulador_desempenho():
    if not SKLEARN_DISPONIVEL:
        return html.Section([
            html.H2("⚡ Motor Preditivo de Desempenho Académico"),
            html.P("O pacote scikit-learn não está disponível.", style={"color": "#FBBF24"})
        ])

    return html.Div([
        html.H2("⚡ Simulador de Desempenho (Motor Preditivo)", style={"color": "#38BDF8"}),
        html.P("Simule cenários da rotina de um estudante e estime o impacto no GPA através dos modelos de regressão.",
                style={"color": "#c084fc", "marginBottom": "20px"}),
        
        html.H3("Parâmetros da Rotina a Simular", style={"color": "#F8FAFC", "marginBottom": "12px"}),
        html.Div([
            html.Div([
                control_label("Horas de estudo diário:"),
                dcc.Slider(
                    id="sim-estudo",
                    min=0.0,
                    max=10.0,
                    step=0.5,
                    value=4.0,
                    marks={i: mark_style(f"{i}h") for i in range(11)}
                )
            ]),
            html.Div([
                control_label("Frequência presencial às aulas (%):"),
                dcc.Slider(
                    id="sim-frequencia",
                    min=50,
                    max=100,
                    step=1,
                    value=85,
                    marks={50: mark_style("50%"), 75: mark_style("75%"), 100: mark_style("100%")}
                )
            ]),
            html.Div([
                control_label("Horas de sono por noite:"),
                dcc.Slider(
                    id="sim-sono",
                    min=3.0,
                    max=10.0,
                    step=0.5,
                    value=7.0,
                    marks={i: mark_style(f"{i}h") for i in range(3, 11)}
                )
            ]),
            html.Div([
                control_label("Nível de stress mental (0 a 10):"),
                dcc.Slider(
                    id="sim-estresse",
                    min=0,
                    max=10,
                    step=0.5,
                    value=4.0,
                    marks={0: mark_style("Baixo"), 5: mark_style("Médio"), 10: mark_style("Alto")}
                )
            ]),
            html.Div([
                control_label("Tempo em redes sociais (h/dia):"),
                dcc.Slider(
                    id="sim-redes",
                    min=0.0,
                    max=8.0,
                    step=0.5,
                    value=2.0,
                    marks={i: mark_style(f"{i}h") for i in range(9)}
                )
            ]),
            html.Div([
                control_label("Idade do estudante:"),
                dcc.Slider(
                    id="sim-idade",
                    min=17,
                    max=35,
                    step=1,
                    value=21,
                    marks={17: mark_style("17"), 25: mark_style("25"), 35: mark_style("35")}
                )
            ]),
        ], id="parametros-simulador", style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))", "gap": "20px", "marginBottom": "24px"}),

        html.Button("Calcular Previsão de Desempenho", id="gerar-previsao", n_clicks=1,
                    style={"padding": "12px 24px", "backgroundColor": "#38BDF8", "color": "#0F172A",
                           "border": 0, "borderRadius": "8px", "fontWeight": "800", "cursor": "pointer"}),

        html.Div(id="resultado-simulador-container", style={"marginTop": "24px"})
    ])


@app.callback(
    Output("resultado-simulador-container", "children"),
    Input("gerar-previsao", "n_clicks"),
    State("sim-estudo", "value"),
    State("sim-sono", "value"),
    State("sim-estresse", "value"),
    State("sim-frequencia", "value"),
    State("sim-redes", "value"),
    State("sim-idade", "value"),
    prevent_initial_call=False
)
def executar_simulacao(n_clicks, estudo, sono, estresse, frequencia, redes, idade):
    if not SKLEARN_DISPONIVEL:
        return html.Div()
    
    motor = obter_motor_preditivo()
    
    # Dicionário com os valores vindos dos sliders da interface
    values = {
        "study_hours_per_day": estudo,
        "sleep_hours": sono,
        "mental_stress_level": estresse,
        "class_attendance_percent": frequencia,
        "social_media_hours": redes,
        "age": idade
    }
    
    # -------------------------------------------------------------------------
    # ATRIBUIÇÃO AUTOMÁTICA DE PERFIL COMPORTAMENTAL PARA A SIMULAÇÃO:
    # Descobrimos a qual dos 3 perfis comportamentais a nova rotina simulada pertence
    # -------------------------------------------------------------------------
    perfil_simulado = "Perfil Comportamental A" # Padrão de segurança
    try:
        # Pega os dados simulados e classifica no cluster comportamental correspondente
        vetor_habitos = np.array([[estudo, sono, estresse, frequencia, redes]])
        # Como o pipeline armazena o pré-processador e o modelo, podemos estimar o cluster 
        # avaliando qual perfil se encaixa melhor, ou definimos uma regra simples baseada nos centroides.
        # Para simplificar e garantir estabilidade sem quebrar, atribuímos o perfil com base no estresse/estudo:
        if estresse >= 7.0:
            perfil_simulado = "Perfil Comportamental C"
        elif estudo >= 5.0:
            perfil_simulado = "Perfil Comportamental A"
        else:
            perfil_simulado = "Perfil Comportamental B"
    except Exception:
        perfil_simulado = "Perfil Geral"

    values["cluster_perfil"] = perfil_simulado

    # Monta o DataFrame de entrada contendo todas as features exigidas pelo motor (numéricas + categóricas)
    entrada = pd.DataFrame(
        [{feature: values.get(feature, 0) for feature in motor["features"]}], 
        columns=motor["features"]
    )
    
    gpa = max(0.0, min(4.0, float(motor["modelo"].predict(entrada)[0])))
    
    if sono < 5.5 and estresse >= 6.0:
        classificacao = "Indício de sobrecarga académica"
        cor_status = "#F43F5E"
        obs = "Atenção: A combinação de sono reduzido e stress elevado prejudica fortemente o rendimento contínuo."
    elif gpa >= 3.5:
        classificacao = "Alto Desempenho Académico"
        cor_status = "#34D399"
        obs = "Padrão de rotina consistente e favorável para manter o GPA próximo da faixa máxima."
    elif gpa >= 2.5:
        classificacao = "Desempenho Intermédio"
        cor_status = "#FBBF24"
        obs = "Rendimento médio. Aumentar a frequência presencial e o sono são os fatores de maior ganho potencial."
    else:
        classificacao = "Atenção ao Desempenho"
        cor_status = "#F43F5E"
        obs = "Projeção abaixo do coeficiente mínimo de retenção discente."

    children = [
        html.Div([
            metric("GPA Estimado", f"{gpa:.2f}"),
            html.Div([
                metric("Classificação Projetada", classificacao),
                html.P(f"Equivalente a {gpa_para_nota(gpa):.2f}/10 na escala decimal. {obs}", 
                       style={"color": cor_status, "marginTop": "8px", "fontSize": "0.9rem"})
            ])
        ], style=GRID_2),
        html.P(f"Perfil comportamental atribuído automaticamente à simulação: {perfil_simulado}",
               style={"color": "#38BDF8", "fontSize": "0.85rem", "marginTop": "8px"}),
        html.P("Nota: Estimativa puramente estatística treinada sobre a base de 10.000 estudantes universitários.",
               style={"color": "#c084fc", "fontSize": "0.85rem", "marginTop": "4px"})
    ]

    try:
        modelo = motor["modelo"].named_steps["modelo"]
        preprocessor = motor["modelo"].named_steps["preprocessamento"]
        if hasattr(modelo, "feature_importances_"):
            rotulos_variaveis = {
                "age": "Idade",
                "social_media_hours": "Horas diárias em redes sociais",
                "sleep_hours": "Horas de sono por noite",
                "mental_stress_level": "Nível de estresse",
                "study_hours_per_day": "Horas de estudo por dia",
                "class_attendance_percent": "Frequência às aulas (%)",
                "cluster_perfil": "Perfil Comportamental"
            }
            nomes_variaveis = [
                rotulos_variaveis.get(nome.split("__")[-1], nome.split("__")[-1])
                for nome in preprocessor.get_feature_names_out()
            ]
            importancia = pd.DataFrame({
                "Variável": nomes_variaveis,
                "Importância (%)": modelo.feature_importances_ * 100
            }).sort_values("Importância (%)", ascending=False).head(8)
            fig = px.bar(importancia, x="Importância (%)", y="Variável", orientation="h",
                         labels={
                             "Importância (%)": "Importância da variável",
                             "Variável": "Variável"
                         })
            estilizar_grafico(fig, "Importância das Variáveis no Modelo Preditivo")
            children.append(graph(fig))
    except Exception:
        pass

    return html.Div(children)

################################################ LAYOUTS DOS PAINEIS DO DASHBORAD ################################################
def layout_visao_geral(data):

    total_estudantes = len(df_raw)
    estudantes_analisados = len(data)

    percentual_amostra = (
        estudantes_analisados / total_estudantes * 100
        if total_estudantes > 0
        else 0
    )

    # Indicadores da amostra filtrada
    gpa_medio = data["GPA"].mean()
    frequencia_media = data["class_attendance_percent"].mean()
    estudo_medio = data["study_hours_per_day"].mean()
    sono_medio = data["sleep_hours"].mean()
    estresse_medio = data["mental_stress_level"].mean()


    return html.Div([
        html.H1(
            "Visão Geral",
            style={
                "color": "#F8FAFC",
                "marginBottom": "8px"
            }
        ),

        # Quantificação da amostra selecionada
        metric_featured(
            label="Amostra analisada",
            value=f"{estudantes_analisados:,}".replace(",", ".") + " estudantes",
            description=(
                f"{percentual_amostra:.1f}% do conjunto de dados original"
            )
        ),

        # Indicadores
        dashboard_section(
            "Indicadores Gerais",
            "Resumo dos principais indicadores da amostra selecionada.",
            [
                html.Div(
                    [
                        metric("GPA Médio", f"{gpa_medio:.2f}"),
                        metric("Frequência Média", f"{frequencia_media:.1f}%"),
                        metric("Estudo Diário", f"{estudo_medio:.1f} h"),
                        metric("Sono Diário", f"{sono_medio:.1f} h"),
                        metric("Estresse Médio", f"{estresse_medio:.1f}")
                    ],
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fit, minmax(185px, 1fr))",
                        "gap": "14px"
                    }
                )
            ]
        ),

        # Desempenho
        dashboard_section(
            "Desempenho Acadêmico",
            "Distribuição dos indicadores de rendimento dos estudantes.",
            [
                html.Div(
                    [
                        graph(criar_grafico_distribuicao_gpa(data)),
                        graph(criar_grafico_distribuicao_notas(data))
                    ],
                    style=GRID_2
                )
            ]
        ),

        # Caracterização da amostra
        html.Details(
            [
                html.Summary(
                    "Características da Amostra",
                    style={
                        "cursor": "pointer",
                        "fontWeight": "700",
                        "fontSize": "1.1rem",
                        "marginBottom": "20px"
                    }
                ),

                html.Div(
                    [
                        graph(criar_grafico_genero(data)),
                        graph(criar_grafico_renda(data)),
                        graph(criar_grafico_trabalho(data)),
                        graph(criar_grafico_internet(data))
                    ],
                    style=GRID_2
                )
            ],
            style={
                **CARD,
                "marginBottom": "28px"
            }
        ),

        # Análises complementares
        dashboard_section(
            "Comparações Complementares",
            "Exploração do desempenho acadêmico entre diferentes cursos.",
            [
                graph(criar_grafico_gpa_major(data))
            ]
        )

    ])

def layout_habitos_correlacoes(data):
    matriz_correlacao = calcular_matriz_correlacoes(df=data)

    heatmap = criar_heatmap_correlacao(
        matriz_correlacao,
        "Hábitos vs. Desempenho Acadêmico"
    )

    resultados_correlacao = calcular_correlacoes(df=data)

    tabela_correlacao = pd.DataFrame(
        resultados_correlacao["habito_desempenho"]
    )[
        [
            "variavel",
            "desempenho",
            "correlacao",
            "spearman",
            "valor_p",
            "valor_p_ajustado_fdr",
            "pares_validos"
        ]
    ].rename(columns={
        "variavel": "Hábito",
        "desempenho": "Desempenho",
        "correlacao": "Pearson (r)",
        "spearman": "Spearman (rho)",
        "valor_p": "p-valor",
        "valor_p_ajustado_fdr": "p-valor ajustado (FDR)",
        "pares_validos": "Pares válidos"
    }).round({
        "Pearson (r)": 3,
        "Spearman (rho)": 3,
        "p-valor": 4,
        "p-valor ajustado (FDR)": 4
    })

    rotulos_correlacao = {
        "study_hours_per_day": "Horas de estudo por dia",
        "class_attendance_percent": "Frequência das aulas (%)",
        "sleep_hours": "Horas de sono por noite",
        "screen_time_hours": "Tempo de tela por dia (horas)",
        "social_media_hours": "Tempo em redes sociais por dia (horas)",
        "mental_stress_level": "Nível de estresse",
        "GPA": "Desempenho acadêmico (GPA)",
        "final_exam_score": "Nota da prova final",
        "assignment_score": "Nota das atividades"
    }

    tabela_correlacao["Hábito"] = tabela_correlacao["Hábito"].map(
        lambda valor: rotulos_correlacao.get(valor, valor)
    )

    tabela_correlacao["Desempenho"] = tabela_correlacao["Desempenho"].map(
        lambda valor: rotulos_correlacao.get(valor, valor)
    )

    return html.Div([
        html.H2(
            "Matriz de Correlações & Fatores Críticos",
            style={
                "color": "#F8FAFC",
                "marginTop": "20px"
            }
        ),

        html.P(
            "A matriz mostra associações lineares (Pearson) entre "
            "hábitos e indicadores de desempenho. Associação não implica causalidade.",
            style={"color": "#c084fc"}
        ),

        graph(heatmap),

        html.H2("Exploração de correlações"),
        html.P(
            "Selecione duas variáveis para explorar a relação entre elas."
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.Label("Variável do eixo X"),
                        dcc.Dropdown(
                            id="correlacao-x",
                            options=[
                                {"label": coluna, "value": coluna}
                                for coluna in HABIT_VARIABLES + PERFORMANCE_VARIABLES
                                if coluna in data.columns
                            ],
                            value="study_hours_per_day",
                            clearable=False,
                            style={
                                "color": "#1E293B",
                                "backgroundColor": "#FFFFFF"
                            }
                        ),
                    ],
                    style={"flex": "1"},
                ),
                html.Div(
                    [
                        html.Label("Variável do eixo Y"),
                        dcc.Dropdown(
                            id="correlacao-y",
                            options=[
                                {"label": coluna, "value": coluna}
                                for coluna in HABIT_VARIABLES + PERFORMANCE_VARIABLES
                                if coluna in data.columns
                            ],
                            value="GPA",
                            clearable=False,
                            style={
                                "color": "#1E293B",
                                "backgroundColor": "#FFFFFF"
                            }
                        ),
                    ],
                    style={"flex": "1"},
                ),
            ],
            style={
                "display": "flex",
                "gap": "20px",
                "marginBottom": "20px",
            },
        ),

        dcc.Graph(
            id="grafico-dispersao-correlacao",
            figure={},
        ),
        
        html.H3(
            "Correlações com os indicadores de desempenho",
            style={
                "color": "#F8FAFC",
                "marginTop": "20px"
            }
        ),

        html.P(
            "Spearman é apresentado como diagnóstico de robustez diante "
            "do efeito teto. Os valores-p da família hábito-desempenho "
            "foram ajustados pelo método Benjamini-Hochberg (FDR).",
            style={"color": "#94A3B8"}
        ),

        dash_table.DataTable(
            data=tabela_correlacao.to_dict("records"),
            columns=[
                {"name": coluna, "id": coluna}
                for coluna in tabela_correlacao.columns
            ],
            sort_action="native",
            page_size=12,
            style_table={"overflowX": "auto"},
            style_header={
                "backgroundColor": ASTRA_COLORS["border_subtle"],
                "color": ASTRA_COLORS["text_primary"],
                "fontWeight": "700"
            },
            style_cell={
                "backgroundColor": ASTRA_COLORS["card_dark"],
                "color": ASTRA_COLORS["text_primary"],
                "textAlign": "left",
                "padding": "8px"
            }
        )
    ])

def layout_perfis_estudantes(
    df_cluster,
    ordem_perfis,
    mapa_cores_perfis,
    silhueta_atual,
    k_clusters
):
    # Gráfico de distribuição dos clusters
    grafico_clusters = criar_grafico_clusters(
        df_cluster,
        ordem_perfis,
        mapa_cores_perfis
    )

    # Colunas utilizadas na comparação dos perfis
    colunas_habitos = {
        "study_hours_per_day": ("Horas de Estudo", 0, 10),
        "sleep_hours": ("Horas de Sono", 0, 10),
        "mental_stress_level": ("Nível de Estresse", 0, 10),
        "class_attendance_percent": ("Frequência", 0, 100),
        "GPA": ("GPA (Proporcional)", 0, 4)
    }

    dados_perfil = []

    # Calcula as médias de cada hábito para cada cluster
    for cluster_id in sorted(df_cluster["cluster_id"].unique()):
        mascara_cluster = df_cluster["cluster_id"] == cluster_id
        dados_cluster = df_cluster.loc[mascara_cluster]

        perfil = dados_cluster["perfil_cluster"].iloc[0]

        for coluna, (nome_habito, min_val, max_val) in colunas_habitos.items():
            media_real = dados_cluster[coluna].mean()

            # Normaliza a média para uma escala de 0 a 100
            media_normalizada = (
                (media_real - min_val) / (max_val - min_val)
            ) * 100

            media_normalizada = max(
                0,
                min(100, media_normalizada)
            )

            dados_perfil.append({
                "Hábito": nome_habito,
                "Média": round(media_normalizada, 1),
                "Média Real": round(media_real, 2),
                "perfil_cluster": perfil,
                "cluster_id": cluster_id
            })

    df_perfil = pd.DataFrame(dados_perfil)

    # Gráfico comparativo dos perfis
    grafico_perfil = criar_grafico_perfil_clusters(
        df_perfil,
        ordem_perfis,
        mapa_cores_perfis
    )

    # Classificação descritiva da silhueta
    qualidade_silhueta = (
        "Boa separação"
        if silhueta_atual >= 0.35
        else "Separação moderada/fraca"
    )

    return html.Section([
        html.H2(
            f"Perfis de estudantes — Agrupamento K-Means (k = {k_clusters})",
            style={"marginTop": "30px"}
        ),

        html.P(
            "Segmentação dos estudantes com base em hábitos padronizados "
            "e desempenho acadêmico.",
            style={"color": "#94A3B8"}
        ),

        # Métrica de qualidade do agrupamento
        html.Div([
            metric(
                f"Silhouette Score (k={k_clusters})",
                f"{silhueta_atual:.3f}",
                qualidade_silhueta
            )
        ], style={
            "marginBottom": "20px",
            "maxWidth": "350px"
        }),

        # Gráficos dos perfis
        html.Div([
            graph(grafico_clusters),
            graph(grafico_perfil)
        ], style=GRID_2)

    ])

def layout_avaliacao_kmeans(df_metricas_k):
    return html.Section([
        html.H2(
            "Avaliação do agrupamento — K-Means",
            style={"color": "#F8FAFC", "marginBottom": "10px"}
        ),

        html.P(
            "Comparação entre diferentes valores de K utilizando "
            "métricas de qualidade e compactação dos agrupamentos.",
            style={"color": "#94A3B8", "marginBottom": "16px"}
        ),

        dash_table.DataTable(
            data=df_metricas_k.to_dict("records"),
            columns=[
                {"name": col, "id": col}
                for col in df_metricas_k.columns
            ],
            style_table={
                "overflowX": "auto",
                "marginBottom": "28px"
            },
            style_header={
                "backgroundColor": "#1E293B",
                "color": "#F8FAFC",
                "fontWeight": "bold"
            },
            style_cell={
                "backgroundColor": "#0F172A",
                "color": "#CBD5E1",
                "padding": "10px",
                "border": "1px solid #334155"
            }
        )
    ])

def layout_avaliacao_motor_preditivo():
    if not SKLEARN_DISPONIVEL:
        return html.Section([
            html.H2("Avaliação do Motor Preditivo"),
            html.P(
                "O pacote scikit-learn não está disponível.",
                style={"color": "#FBBF24"}
            )
        ])

    motor = obter_motor_preditivo()
    resultados = motor["resultados"].copy().round(3)

    # Previsões sobre o conjunto de teste
    X_test = motor["X_test"]
    y_test = motor["y_test"]

    y_pred_teste = motor["modelo"].predict(X_test)
    y_pred_teste = np.clip(y_pred_teste, 0.0, 4.0)

    # Gráfico de valores reais versus previstos
    fig_validacao = criar_grafico_real_vs_previsto(
        y_test,
        y_pred_teste
    )

    return html.Section([
        html.H2(
            "Avaliação do Motor Preditivo",
            style={"color": "#F8FAFC", "marginBottom": "10px"}
        ),

        html.P(
            "Comparação e validação dos modelos de regressão utilizados "
            "para estimar o desempenho acadêmico.",
            style={"color": "#94A3B8", "marginBottom": "20px"}
        ),

        # Métricas do modelo selecionado
        html.Div([
            metric("Modelo Ativo", motor["nome"]),
            metric("Erro Quadrado (R²)", f"{resultados.iloc[0]['R²']:.3f}"),
            metric("Erro Médio (MAE)", f"{resultados.iloc[0]['MAE']:.3f}")
        ], style={**GRID_3, "marginBottom": "24px"}),

        # Comparação entre os modelos
        html.H3(
            "Torneio de Modelos Preditivos",
            style={"color": "#F8FAFC", "marginBottom": "12px"}
        ),

        dash_table.DataTable(
            data=resultados.to_dict("records"),
            columns=[
                {"name": col, "id": col}
                for col in resultados.columns
            ],
            style_table={
                "overflowX": "auto",
                "marginBottom": "28px"
            },
            style_header={
                "backgroundColor": "#1E293B",
                "color": "#F8FAFC",
                "fontWeight": "bold"
            },
            style_cell={
                "backgroundColor": "#0F172A",
                "color": "#CBD5E1",
                "padding": "10px",
                "border": "1px solid #334155"
            }
        ),

        # Explicação dos algoritmos
        html.Details([
            html.Summary(
                "🔍 Por que comparamos estes três modelos?",
                style={
                    "cursor": "pointer",
                    "fontWeight": "600",
                    "color": "#38BDF8"
                }
            ),

            dcc.Markdown("""
                O motor preditivo do ASTRA utiliza uma abordagem de **torneio de algoritmos**:

                - **Regressão Ridge:** modelo linear com regularização.
                - **Random Forest:** combina múltiplas árvores de decisão para capturar relações não lineares.
                - **Gradient Boosting:** constrói árvores sequencialmente, buscando corrigir erros das etapas anteriores.
            """, style={
                "color": "#CBD5E1",
                "marginTop": "10px",
                "fontSize": "0.9rem"
            })
        ], style={
            "marginBottom": "20px",
            "padding": "12px",
            "backgroundColor": "#1E293B",
            "borderRadius": "8px",
            "border": "1px solid #334155"
        }),

        graph(criar_grafico_comparacao_modelos(resultados)),

        # Validação no conjunto de teste
        html.H3(
            "Validação no Conjunto de Teste",
            style={
                "color": "#F8FAFC",
                "marginTop": "24px",
                "marginBottom": "12px"
            }
        ),

        html.P(
            "O gráfico compara os valores reais do GPA com as previsões "
            "do modelo no conjunto de teste. A linha tracejada representa "
            "a previsão ideal.",
            style={
                "color": "#CBD5E1",
                "fontSize": "0.9rem",
                "marginBottom": "16px"
            }
        ),

        graph(fig_validacao)
    ])

@app.callback(
    Output("grafico-dispersao-correlacao", "figure"),
    Input("correlacao-x", "value"),
    Input("correlacao-y", "value"),
    Input("cursos", "value"),
    Input("anos", "value"),
    Input("generos", "value"),
    Input("idade", "value"),
    Input("trabalho", "value"),
    Input("estresse", "value"),
    Input("frequencia", "value"),
    Input("rendas", "value"),
    Input("metodos", "value"),
)
def atualizar_grafico_dispersao(
    coluna_x,
    coluna_y,
    qtd,
    cursos,
    anos,
    generos,
    idade,
    trabalho,
    estresse,
    frequencia,
    rendas,
    metodos,
):
    data = filtered_data(
        qtd,
        cursos,
        anos,
        generos,
        idade,
        trabalho,
        estresse,
        frequencia,
        rendas,
        metodos,
    )

    if data.empty or coluna_x not in data.columns or coluna_y not in data.columns:
        return {}

    return criar_grafico_dispersao_correlacao(data, coluna_x, coluna_y,)

@app.callback(
    Output("dashboard-content", "children"),
    Input("main-tabs", "value"),
    Input("cursos", "value"),
    Input("anos", "value"),
    Input("generos", "value"),
    Input("idade", "value"),
    Input("trabalho", "value"),
    Input("estresse", "value"),
    Input("frequencia", "value"),
    Input("rendas", "value"),
    Input("metodos", "value"),
    Input("k-clusters", "value")
)
def renderizar_conteudo_abas(aba_ativa, cursos, anos, generos, idade, trabalho, estresse, frequencia, rendas, metodos, k_clusters):

    data = filtered_data(cursos, anos, generos, idade, trabalho, estresse, frequencia, rendas, metodos)

    if data.empty:
        return html.Div("⚠️ Nenhum registro encontrado para a combinação atual de filtros. Tente flexibilizar os parâmetros na barra lateral.", style={"color": "#FBBF24", "padding": "24px"})
    gpa_medio = data["GPA"].mean()
    frequencia_media = data["class_attendance_percent"].mean()
    estudo_medio = data["study_hours_per_day"].mean()
    sono_medio = data["sleep_hours"].mean()
    estresse_medio = data["mental_stress_level"].mean()
    adocao_ia = data["favorite_AI_tool"].fillna("None").ne("None").mean() * 100

    kpis_header = [
        html.P(f"Amostra ativa: {len(data):,} estudantes filtrados (de {len(df_raw):,} totais).",
                style={"color": "#c084fc", "marginBottom": "14px"}),
        html.Div([
            metric("GPA Médio (0 - 4.0)", f"{gpa_medio:.2f}", f"{gpa_medio - 3.0:+.2f} vs Alvo"),
            metric("Frequência Média", f"{frequencia_media:.1f}%"),
            metric("Estudo Diário", f"{estudo_medio:.1f} h/dia"),
            metric("Sono Diário", f"{sono_medio:.1f} h/noite"),
            metric("Stress Mental", f"{estresse_medio:.1f} / 10"),
            metric("Adoção de IA", f"{adocao_ia:.1f}%"),
        ], style={"display": "grid", "gridTemplateColumns": "repeat(6, minmax(0, 1fr))", "gap": "12px", "marginBottom": "24px"})
    ]

    if aba_ativa == "tab-simulador":

        # Executa o K-Means usando o 'k' selecionado no slider da barra lateral
        df_cluster, medias_cluster, ordem_gpa, mapa_nomes, df_metricas_k, silhueta_atual = executar_kmeans(
            data.copy(),
            k_clusters
        )

        ordem_perfis = [
            mapa_nomes[cid]
            for cid in ordem_gpa
        ]

        cores_perfis = [
            ASTRA_COLORS["accent_emerald"],
            ASTRA_COLORS["accent_amber"],
            ASTRA_COLORS["accent_rose"],
            ASTRA_COLORS["accent_cyan"],
            ASTRA_COLORS["accent_indigo"]
        ]

        mapa_cores_perfis = dict(zip(ordem_perfis, cores_perfis))

        secao_perfis = layout_perfis_estudantes(
            df_cluster=df_cluster,
            ordem_perfis=ordem_perfis,
            mapa_cores_perfis=mapa_cores_perfis,
            silhueta_atual=silhueta_atual,
            k_clusters=k_clusters
        )

        return html.Div(
            kpis_header + [

                secao_perfis,

                # Seção Inferior: Relatório Geral Quantitativo dos diferentes 'k'
                html.Hr(style={"borderColor": "#334155", "margin": "40px 0"}),
                
                layout_avaliacao_kmeans(df_metricas_k),

                html.Hr(
                    style={
                        "borderColor": "#334155",
                        "margin": "40px 0"
                    }
                ),

                # Avaliação do motor preditivo
                layout_avaliacao_motor_preditivo(),

                html.Hr(
                    style={"borderColor": "#334155", "margin": "40px 0"}
                ),

                # Simulador de desempenho
                layout_simulador_desempenho()
            ]
        )

    if aba_ativa == "tab-metodologia":
        return html.Div([
            project_section(),
            methodology_section(),
            dictionary_section()
        ])

    if aba_ativa == "tab-correlacoes":
        return html.Div(
            kpis_header + [
                layout_habitos_correlacoes(data)
            ]
        )

    if aba_ativa == "tab-geral":
        return layout_visao_geral(data)

    return html.Div("Selecione um separador válido.")


if __name__ == "__main__":
    app.run(debug=True, port=8050)