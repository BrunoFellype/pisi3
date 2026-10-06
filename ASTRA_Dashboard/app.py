import os
from functools import lru_cache
from dash import Dash, Input, Output, State, dcc, html, dash_table
import pandas as pd
import plotly.express as px

from analysis.clusters import executar_kmeans
from analysis.predicao import treinar_motor_preditivo
from analysis.correlacoes import calcular_matriz_correlacoes
from data.load import get_Dataset
from visualization.graphs import (
    criar_grafico_cafe_sono, criar_grafico_clusters, criar_grafico_gpa_major,
    criar_grafico_genero, criar_grafico_ia_tools, criar_grafico_internet,
    criar_grafico_metodos_anotacao, criar_grafico_perfil_clusters,
    criar_grafico_presenca_gpa, criar_grafico_renda, criar_grafico_sono_estresse,
    criar_grafico_study_gpa, criar_grafico_trabalho, criar_heatmap_correlacao,
    criar_scatter_com_tendencia
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
GRID_2 = {"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))", "gap": "16px"}
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


def graph(figure):
    return dcc.Graph(figure=figure, config={"displayModeBar": False}, style={"backgroundColor": ASTRA_COLORS["card_dark"]})


def gpa_para_nota(gpa):
    return (gpa / 4) * 10


def filtered_data(qtd_analise, cursos, anos, generos, idade, trabalho, estresse, frequencia, rendas, metodos):
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
    return data.iloc[:qtd_analise]


def project_section():
    return html.Details([
        html.Summary("📌 Sobre o Projeto ASTRA, Objetivos e Questões Norteadoras", style={"cursor": "pointer", "fontSize": "1.1rem", "fontWeight": "700"}),
        dcc.Markdown("""
### 💡 Objetivo Principal
Desenvolver uma plataforma analítica e preditiva que centraliza a rotina acadêmica universitária, correlacionando hábitos de estudo, sono, bem-estar e uso de ferramentas tecnológicas para antecipar o desempenho acadêmico (GPA) e mitigar o risco de sobrecarga estudantil.

### 🎯 Objetivos Secundários
* **Investigação Exploratória:** Realizar a limpeza, transformação e análise exploratória de dados multivariados sobre hábitos de 10.000 estudantes universitários.
* **Identificação de Fatores Críticos:** Mapear correlações estatísticas entre indicadores comportamentais e o rendimento acadêmico final.
* **Modelagem Preditiva:** Estruturar modelos de regressão capazes de estimar a pontuação acadêmica com base em registros diários.
* **Interface de Suporte à Decisão:** Disponibilizar um painel interativo intuitivo que transforme métricas estatísticas em recomendações práticas.

### ❓ Perguntas Norteadoras da Pesquisa
**1. De que maneira o equilíbrio entre horas de sono e nível de estresse modula o impacto das horas de estudo no desempenho acadêmico (GPA)?**

**2. Qual é a correlação do uso de ferramentas de Inteligência Artificial no desempenho dos estudantes quando contrastado com o tempo gasto em redes sociais e a frequência às aulas?**
""", style={"color": "#CBD5E1", "marginTop": "18px"}),
    ], style={"marginBottom": "20px"})


def dictionary_section():
    dictionary_path = os.path.join(os.path.dirname(__file__), "..", "Data", "dicionario_de_dados.md")
    dictionary_text = open(dictionary_path, encoding="utf-8").read() if os.path.exists(dictionary_path) else "Consulte o arquivo Data/dicionario_de_dados.md no repositório."
    return html.Details([
        html.Summary("📖 Dicionário de Dados do Dataset", style={"cursor": "pointer", "fontSize": "1.1rem", "fontWeight": "700"}),
        dcc.Markdown(dictionary_text, style={"color": "#CBD5E1", "marginTop": "18px"}),
    ], style={"marginBottom": "20px"})


def layout_controls():
    total = len(df_raw)
    return html.Aside([
        html.H2("⚡ Filtros ASTRA", style={"color": "#38BDF8"}),
        html.P("Personalize sua análise:", style={"color": "#c084fc", "fontSize": "13px"}),
        
        control_label("Volume de Dados Analisado:"),
        dcc.Slider(
            id="qtd-analise",
            min=500,
            max=total,
            value=total,
            step=500,
            marks={
                500: mark_style("500"),
                3000: mark_style("3k"),
                5500: mark_style("5.5k"),
                8000: mark_style("8k"),
                total: mark_style("10k")
            }
        ),
        html.Hr(style={"borderColor": "#1E293B", "margin": "24px 0"}),
        
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
                dcc.Tab(label="📊 Visão Geral & KPIs", value="tab-geral", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                dcc.Tab(label="🔬 Hábitos & Correlações", value="tab-correlacoes", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                dcc.Tab(label="🤖 Perfis & Simulador", value="tab-simulador", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                dcc.Tab(label="📖 Metodologia & Dicionário", value="tab-metodologia", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
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
    
    motor = obter_motor_preditivo()
    resultados = motor["resultados"].copy().round(3)

    return html.Div([
        html.H2("⚡ Simulador de Desempenho (Motor Preditivo)", style={"color": "#38BDF8"}),
        html.P("Simule cenários da rotina de um estudante e estime o impacto no GPA através dos modelos de regressão.",
               style={"color": "#c084fc", "marginBottom": "20px"}),
        
        html.Div([
            metric("Modelo Ativo", motor["nome"]),
            metric("Aderência (R²)", f"{resultados.iloc[0]['R²']:.3f}"),
            metric("Erro Médio (MAE)", f"{resultados.iloc[0]['MAE']:.3f}")
        ], style={**GRID_3, "marginBottom": "24px"}),

        html.H3("Torneio de Modelos Preditivos", style={"color": "#F8FAFC", "marginBottom": "12px"}),
        dash_table.DataTable(
            data=resultados.to_dict("records"),
            columns=[{"name": col, "id": col} for col in resultados.columns],
            style_table={"overflowX": "auto", "marginBottom": "28px"},
            style_header={"backgroundColor": "#1E293B", "color": "#F8FAFC", "fontWeight": "bold"},
            style_cell={"backgroundColor": "#0F172A", "color": "#CBD5E1", "padding": "10px", "border": "1px solid #334155"}
        ),

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
        ], style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))", "gap": "20px", "marginBottom": "24px"}),

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
    values = {
        "study_hours_per_day": estudo,
        "sleep_hours": sono,
        "mental_stress_level": estresse,
        "class_attendance_percent": frequencia,
        "social_media_hours": redes,
        "age": idade
    }
    
    entrada = pd.DataFrame([{feature: values.get(feature, 0) for feature in motor["features"]}], columns=motor["features"])
    gpa = max(0.0, min(4.0, float(motor["modelo"].predict(entrada)[0])))
    
    if sono < 5.5 and estresse >= 6.0:
        classificacao = "Risco de Burnout Detectado"
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
        html.P("Nota: Estimativa puramente estatística treinada sobre a base de 10.000 estudantes universitários.",
               style={"color": "#c084fc", "fontSize": "0.85rem", "marginTop": "12px"})
    ]

    try:
        modelo = motor["modelo"].named_steps["modelo"]
        preprocessor = motor["modelo"].named_steps["preprocessamento"]
        if hasattr(modelo, "feature_importances_"):
            importancia = pd.DataFrame({
                "Variável": preprocessor.get_feature_names_out(),
                "Importância (%)": modelo.feature_importances_ * 100
            }).sort_values("Importância (%)", ascending=False).head(8)
            fig = px.bar(importancia, x="Importância (%)", y="Variável", orientation="h",
                         title="Fatores com Maior Peso no Modelo")
            estilizar_grafico(fig, "Fatores com Maior Peso no Modelo")
            children.append(graph(fig))
    except Exception:
        pass

    return html.Div(children)


@app.callback(
    Output("dashboard-content", "children"),
    Input("main-tabs", "value"),
    Input("qtd-analise", "value"),
    Input("cursos", "value"),
    Input("anos", "value"),
    Input("generos", "value"),
    Input("idade", "value"),
    Input("trabalho", "value"),
    Input("estresse", "value"),
    Input("frequencia", "value"),
    Input("rendas", "value"),
    Input("metodos", "value")
)
def renderizar_conteudo_abas(aba_ativa, qtd, cursos, anos, generos, idade, trabalho, estresse, frequencia, rendas, metodos):
    if aba_ativa == "tab-simulador":
        return layout_simulador_desempenho()

    if aba_ativa == "tab-metodologia":
        return html.Div([
            project_section(),
            dictionary_section()
        ])

    data = filtered_data(qtd, cursos, anos, generos, idade, trabalho, estresse, frequencia, rendas, metodos)

    if data.empty:
        return html.Div("⚠️ Nenhum registo encontrado para a combinação atual de filtros.",
                        style={"color": "#FBBF24", "padding": "24px"})

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

    if aba_ativa == "tab-correlacoes":
        matriz_correlacao = calcular_matriz_correlacoes(df=data)
        heatmap = criar_heatmap_correlacao(matriz_correlacao, "Hábitos vs. Desempenho Académico")
        return html.Div(kpis_header + [
            html.H2("🗺️ Matriz de Correlações & Fatores Críticos", style={"color": "#F8FAFC", "marginTop": "20px"}),
            html.P("Associação linear entre hábitos discentes e notas académicas.", style={"color": "#c084fc"}),
            graph(heatmap)
        ])

    if aba_ativa == "tab-geral":

        data["faixa_sono"] = pd.cut(
            data["sleep_hours"],
            bins=[0, 5, 7, 9, float("inf")],
            labels=[
                "< 5h (Crítico)",
                "5h-7h (Alerta)",
                "7h-9h (Adequado)",
                "> 9h (Alto)"
            ],
            right=False
        )

        return html.Div(
            kpis_header + [

                html.H2(
                    "📊 Visão Geral do Corpo Discente",
                    style={"marginTop": "30px"}
                ),

                html.P(
                    "Análise dos principais hábitos e características dos estudantes.",
                    style={"color": "#94A3B8"}
                ),

                graph(
                    criar_scatter_com_tendencia(
                        data,
                        "study_hours_per_day",
                        "GPA",
                        "Horas de Estudo por Dia",
                        "GPA",
                        "Horas de Estudo x GPA",
                        ASTRA_COLORS["accent_cyan"],
                        ASTRA_COLORS["accent_rose"]
                    )
                ),

                html.Div(
                    [
                        graph(criar_grafico_genero(data)),
                        graph(criar_grafico_renda(data)),
                    ],
                    style=GRID_2
                ),

                html.Div(
                    [
                        graph(criar_grafico_trabalho(data)),
                        graph(criar_grafico_internet(data)),
                    ],
                    style=GRID_2
                ),

                html.Div(
                    [
                        graph(criar_grafico_study_gpa(data)),
                        graph(criar_grafico_presenca_gpa(data)),
                    ],
                    style=GRID_2
                ),

                html.Div(
                    [
                        graph(criar_grafico_sono_estresse(data)),
                        graph(criar_grafico_cafe_sono(data)),
                    ],
                    style=GRID_2
                ),

                html.Div(
                    [
                        graph(criar_grafico_gpa_major(data)),
                        graph(criar_grafico_metodos_anotacao(data)),
                    ],
                    style=GRID_2
                ),

                html.Div(
                    [
                        graph(criar_grafico_ia_tools(data)),
                    ],
                    style={"marginTop": "20px"}
                )
            ]
        )

    return html.Div("Selecione um separador válido.")


if __name__ == "__main__":
    app.run(debug=True, port=8050)