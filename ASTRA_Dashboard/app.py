import os
from functools import lru_cache

from dash import Dash, Input, Output, State, dcc, html, dash_table
import pandas as pd
import plotly.express as px

from analysis.clusters import executar_kmeans
from analysis.predicao import treinar_motor_preditivo
from data.load import get_Dataset
from visualization.graphs import (
    criar_grafico_cafe_sono, criar_grafico_clusters, criar_grafico_gpa_major,
    criar_grafico_genero, criar_grafico_ia_tools, criar_grafico_internet,
    criar_grafico_metodos_anotacao, criar_grafico_perfil_clusters,
    criar_grafico_presenca_gpa, criar_grafico_renda, criar_grafico_sono_estresse,
    criar_grafico_study_gpa, criar_grafico_trabalho, criar_heatmap_correlacao,
    criar_scatter_com_tendencia,
)
from visualization.style import ASTRA_COLORS, estilizar_grafico

try:
    import sklearn
    SKLEARN_DISPONIVEL = True
except ImportError:
    SKLEARN_DISPONIVEL = False

app = Dash(__name__, title="ASTRA - Inteligência Acadêmica", suppress_callback_exceptions=True)
server = app.server
df_raw = get_Dataset()

CSS = {"backgroundColor": ASTRA_COLORS["bg_dark"], "color": ASTRA_COLORS["text_primary"], "fontFamily": "Plus Jakarta Sans, sans-serif", "minHeight": "100vh", "padding": "24px 32px"}
SIDEBAR = {"backgroundColor": "#0F172A", "borderRight": "1px solid #1E293B", "padding": "20px", "width": "285px", "flexShrink": 0}
CONTENT = {"flex": 1, "padding": "0 0 0 26px", "minWidth": 0}
CARD = {"background": "linear-gradient(135deg, #1E293B 0%, #0F172A 100%)", "border": "1px solid #334155", "borderTop": "3px solid #38BDF8", "borderRadius": "12px", "padding": "16px 20px"}
GRID_2 = {"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))", "gap": "16px"}
GRID_3 = {"display": "grid", "gridTemplateColumns": "repeat(3, minmax(0, 1fr))", "gap": "16px"}
GRID_4 = {"display": "grid", "gridTemplateColumns": "repeat(4, minmax(0, 1fr))", "gap": "12px"}


def control_label(text):
    return html.Label(text, style={"color": "#CBD5E1", "fontWeight": "600", "display": "block", "marginBottom": "8px"})


def multi_control(identifier, label, options):
    return html.Div([control_label(label), dcc.Dropdown(id=identifier, options=[{"label": str(item), "value": item} for item in options], value=options, multi=True, style={"color": "#0F172A"})], style={"marginBottom": "18px"})


def metric(label, value, delta=None):
    children = [html.Div(label, style={"color": "#94A3B8", "fontSize": "0.82rem", "fontWeight": "600", "textTransform": "uppercase"}), html.Div(value, style={"fontSize": "1.85rem", "fontWeight": "800", "marginTop": "5px"})]
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
        html.H2("🎯 Filtros ASTRA", style={"color": "#38BDF8"}),
        html.P("Personalize sua análise:", style={"color": "#94A3B8"}),
        control_label("Volume de Dados Analisado:"),
        dcc.Slider(id="qtd-analise", min=500, max=total, value=total, step=500),
        html.Hr(style={"borderColor": "#1E293B", "margin": "24px 0"}),
        multi_control("cursos", "Cursos (Graduação):", sorted(df_raw["major"].dropna().unique())),
        multi_control("anos", "Período / Ano Acadêmico:", sorted(df_raw["university_year"].dropna().unique())),
        multi_control("generos", "Gênero:", sorted(df_raw["gender"].dropna().unique())),
        control_label("Faixa Etária (Idade):"),
        dcc.RangeSlider(id="idade", min=int(df_raw["age"].min()), max=int(df_raw["age"].max()), value=[int(df_raw["age"].min()), int(df_raw["age"].max())], step=1),
        html.Div(style={"height": "20px"}),
        control_label("Trabalho em Meio Período:"),
        dcc.RadioItems(id="trabalho", options=[{"label": x, "value": x} for x in ["Todos", "Sim (Trabalha)", "Não"]], value="Todos", labelStyle={"display": "block", "marginBottom": "6px"}),
        html.Div(style={"height": "18px"}),
        control_label("Faixa de Estresse Mental (0 a 10):"),
        dcc.RangeSlider(id="estresse", min=0, max=10, value=[0, 10], step=0.5),
        html.Div(style={"height": "20px"}),
        control_label("Frequência às Aulas (%):"),
        dcc.RangeSlider(id="frequencia", min=0, max=100, value=[0, 100], step=5),
        html.Div(style={"height": "20px"}),
        multi_control("rendas", "Nível de Renda Familiar:", sorted(df_raw["family_income_level"].dropna().unique())),
        multi_control("metodos", "Método de Anotação:", sorted(df_raw["note_taking_method"].dropna().unique())),
        html.Hr(style={"borderColor": "#1E293B", "margin": "24px 0"}),
        html.H3("🤖 Machine Learning (K-Means)", style={"color": "#38BDF8"}),
        dcc.Checklist(id="ativar-kmeans", options=[{"label": "Ativar Clusterização K-Means", "value": "ativo"}], value=["ativo"]),
        html.Div(style={"height": "12px"}),
        control_label("Quantidade de Perfis (K):"),
        dcc.Slider(id="k-clusters", min=2, max=5, value=3, step=1, marks={i: str(i) for i in range(2, 6)}),
    ], style=SIDEBAR)


app.layout = html.Div([layout_controls(), html.Main([html.H1("ASTRA Analytics — Painel de Hábitos & Desempenho Acadêmico"), project_section(), dictionary_section(), html.Div(id="dashboard-content")], style=CONTENT)], style={**CSS, "display": "flex"})


@lru_cache(maxsize=1)
def obter_motor_preditivo():
    return treinar_motor_preditivo(df_raw)


def prediction_section(n_clicks, estudo, sono, estresse, frequencia, redes, idade):
    if not SKLEARN_DISPONIVEL:
        return html.Section([html.H2("🔮 Motor Preditivo de Desempenho Acadêmico"), html.P("O pacote scikit-learn não está disponível.", style={"color": "#FBBF24"})])
    try:
        motor = obter_motor_preditivo()
        resultados = motor["resultados"].copy().round(3)
        children = [html.H2("🔮 Motor Preditivo de Desempenho Acadêmico"), html.P("O ASTRA utiliza Machine Learning para estimar o GPA a partir de hábitos e características acadêmicas.", style={"color": "#CBD5E1"}), html.Div([metric("Modelo selecionado", motor["nome"]), metric("R²", f"{resultados.iloc[0]['R²']:.3f}"), metric("Erro médio (MAE)", f"{resultados.iloc[0]['MAE']:.3f}")], style=GRID_3), html.H3("📊 Comparação dos modelos"), dash_table.DataTable(data=resultados.to_dict("records"), columns=[{"name": col, "id": col} for col in resultados.columns], style_table={"overflowX": "auto"}, style_header={"backgroundColor": "#1E293B", "color": "#F8FAFC", "fontWeight": "bold"}, style_cell={"backgroundColor": "#0F172A", "color": "#CBD5E1", "padding": "10px"}), html.H3("🎯 Simular desempenho de um estudante"), html.P("Informe os hábitos do estudante para gerar uma estimativa.", style={"color": "#94A3B8"})]
        form_fields = [
            ("previsao-estudo", "Horas de estudo por dia", estudo, 0, 24, 0.5),
            ("previsao-sono", "Horas de sono por noite", sono, 0, 24, 0.5),
            ("previsao-estresse", "Nível de estresse", estresse, 0, 10, 0.5),
            ("previsao-frequencia", "Frequência às aulas (%)", frequencia, 0, 100, 1),
            ("previsao-redes", "Horas de redes sociais por dia", redes, 0, 24, 0.5),
            ("previsao-idade", "Idade", idade, 10, 100, 1),
        ]
        children.append(html.Div([html.Div([control_label(label), dcc.Input(id=identifier, type="number", min=minimum, max=maximum, step=step, value=value, style={"width": "100%"})]) for identifier, label, value, minimum, maximum, step in form_fields] + [html.Button("🔮 Gerar previsão", id="gerar-previsao", n_clicks=0, style={"padding": "10px 18px", "backgroundColor": "#38BDF8", "border": 0, "borderRadius": "6px", "fontWeight": "700"})], style={"display": "grid", "gridTemplateColumns": "repeat(3, minmax(0, 1fr))", "gap": "16px", "alignItems": "end"}))
        if n_clicks:
            values = {"study_hours_per_day": estudo, "sleep_hours": sono, "mental_stress_level": estresse, "class_attendance_percent": frequencia, "social_media_hours": redes, "age": idade}
            entrada = pd.DataFrame([{feature: values[feature] for feature in motor["features"]}], columns=motor["features"])
            gpa = max(0, min(4, motor["modelo"].predict(entrada)[0]))
            classificacao = "Alto desempenho" if gpa >= 3.5 else "Desempenho intermediário" if gpa >= 2.5 else "Atenção ao desempenho"
            children.append(html.Div([metric("GPA estimado", f"{gpa:.2f}"), html.Div([metric("Classificação", classificacao), html.P(f"Equivalente a uma nota de {gpa_para_nota(gpa):.2f} na escala de 0–10.", style={"color": "#CBD5E1"})])], style={**GRID_2, "marginTop": "18px"}))
            children.append(html.P("Esta é uma estimativa estatística baseada nos padrões encontrados no dataset. Ela não representa uma garantia sobre o desempenho futuro de um estudante.", style={"color": "#38BDF8"}))
        modelo = motor["modelo"].named_steps["modelo"]
        preprocessor = motor["modelo"].named_steps["preprocessamento"]
        if hasattr(modelo, "feature_importances_"):
            importancia = pd.DataFrame({"Variável": preprocessor.get_feature_names_out(), "Importância (%)": modelo.feature_importances_ * 100}).sort_values("Importância (%)", ascending=False).head(10)
            fig = px.bar(importancia, x="Importância (%)", y="Variável", orientation="h", title="Importância das variáveis no modelo")
            estilizar_grafico(fig, "Importância das Variáveis no Modelo")
            children.append(graph(fig))
        return html.Section(children, style={"marginTop": "28px"})
    except Exception as error:
        return html.Section([html.H2("🔮 Motor Preditivo de Desempenho Acadêmico"), html.P(f"Não foi possível executar o motor preditivo: {error}", style={"color": "#F43F5E"})])


@app.callback(
    Output("dashboard-content", "children"),
    Input("qtd-analise", "value"), Input("cursos", "value"), Input("anos", "value"), Input("generos", "value"), Input("idade", "value"), Input("trabalho", "value"), Input("estresse", "value"), Input("frequencia", "value"), Input("rendas", "value"), Input("metodos", "value"), Input("ativar-kmeans", "value"), Input("k-clusters", "value"), Input("gerar-previsao", "n_clicks"),
    State("previsao-estudo", "value"), State("previsao-sono", "value"), State("previsao-estresse", "value"), State("previsao-frequencia", "value"), State("previsao-redes", "value"), State("previsao-idade", "value"),
)
def atualizar_dashboard(qtd, cursos, anos, generos, idade, trabalho, estresse, frequencia, rendas, metodos, kmeans_ativo, k, n_clicks, estudo, sono, previsao_estresse, previsao_frequencia, redes, previsao_idade):
    data = filtered_data(qtd, cursos, anos, generos, idade, trabalho, estresse, frequencia, rendas, metodos)
    if data.empty:
        return html.Div("⚠️ Nenhum registro encontrado para a combinação atual de filtros. Tente flexibilizar os parâmetros na barra lateral.", style={"color": "#FBBF24", "padding": "24px"})
    gpa = data["GPA"].mean()
    content = [html.P(f"Exibindo dados de {len(data):,} estudantes analisados (de um total de {len(df_raw):,} disponíveis na base).", style={"color": "#94A3B8"}), html.Div([metric("GPA Médio (0 - 4.0)", f"{gpa:.2f}", f"{gpa - 3.0:+.2f} vs Alvo"), metric("Frequência Média", f"{data['class_attendance_percent'].mean():.1f}%"), metric("Estudo Diário", f"{data['study_hours_per_day'].mean():.1f} h/dia"), metric("Sono Diário", f"{data['sleep_hours'].mean():.1f} h/noite"), metric("Nível de Estresse", f"{data['mental_stress_level'].mean():.1f} / 10"), metric("Adoção de IA", f"{(data['favorite_AI_tool'].fillna('None') != 'None').mean() * 100:.1f}%")], style={"display": "grid", "gridTemplateColumns": "repeat(6, minmax(0, 1fr))", "gap": "12px"})]
    corr_cols = ["GPA", "final_exam_score", "class_attendance_percent", "study_hours_per_day", "sleep_hours", "mental_stress_level", "social_media_hours", "screen_time_hours", "gaming_hours", "AI_tool_usage_hours"]
    corr_names = {"GPA": "GPA", "final_exam_score": "Nota Exame", "class_attendance_percent": "Frequência %", "study_hours_per_day": "Estudo (h)", "sleep_hours": "Sono (h)", "mental_stress_level": "Estresse", "social_media_hours": "Redes Sociais", "screen_time_hours": "Tempo de Tela", "gaming_hours": "Games (h)", "AI_tool_usage_hours": "Uso IA (h)"}
    heatmap = criar_heatmap_correlacao(data[corr_cols].rename(columns=corr_names).corr().round(2), "Matriz de Correlação dos Hábitos vs. Performance")
    insight = html.Div([html.H4("🏆 Os 2 Maiores Impulsionadores do GPA:", style={"color": "#38BDF8"}), html.P("+0.74 • Frequência às Aulas: É o preditor número 1 de sucesso."), html.P("+0.72 • Horas de Estudo Diário: Estudo consistente gera impacto positivo."), html.H4("⚠️ Os 3 Maiores Detratores do GPA:", style={"color": "#F43F5E"}), html.P("-0.27 • Estresse Mental; -0.23 • Redes Sociais; -0.18 • Tempo de Tela Total."), html.H4("🔍 Mitos Desmistificados pelos Dados:", style={"color": "#818CF8"}), html.P("Videogames têm quase nenhum impacto no GPA. Uso de IA não eleva o GPA sozinho sem estudo prévio.")], style={**CARD, "height": "100%"})
    content += [html.H2("🔬 Mapa de Correlações & Principais Fatores de Performance"), html.P("Visão comparativa de como cada hábito influencia diretamente o GPA e o Desempenho no Exame:", style={"color": "#94A3B8"}), html.Div([graph(heatmap), insight], style=GRID_2)]
    scatters = [("class_attendance_percent", "GPA", "Frequência às Aulas (%)", "GPA (0.0 a 4.0)", "Frequência às Aulas vs. GPA (r = +0.74)", "accent_emerald"), ("social_media_hours", "GPA", "Horas em Redes Sociais / Dia", "GPA (0.0 a 4.0)", "Redes Sociais vs. GPA (r = -0.23)", "accent_amber"), ("sleep_hours", "GPA", "Horas de Sono por Noite", "GPA (0.0 a 4.0)", "Sono Diário vs. GPA (r = +0.23)", "accent_cyan"), ("screen_time_hours", "final_exam_score", "Tempo de Tela Diário (Horas)", "Nota no Exame Final (0 a 100)", "Tempo de Tela vs. Exame Final (r = -0.08)", "accent_rose"), ("AI_tool_usage_hours", "GPA", "Horas de Uso de IA / Dia", "GPA (0.0 a 4.0)", "Uso de IA vs. GPA (r = +0.03)", "accent_indigo"), ("exercise_hours_per_week", "mental_stress_level", "Exercício Físico (Horas/Semana)", "Nível de Estresse (0 a 10)", "Exercício Físico vs. Estresse (r = -0.15)", "accent_teal")]
    content += [html.H2("📈 Investigação Visual de Hábitos Críticos (Scatter Plots)"), html.P("Dispersão detalhada de cada aluno com linha de tendência estimada para os fatores essenciais:", style={"color": "#94A3B8"}), html.Div([graph(criar_scatter_com_tendencia(data, x, y, lx, ly, title, ASTRA_COLORS[color], "#FFFFFF")) for x, y, lx, ly, title, color in scatters], style=GRID_3)]
    content += [html.H2("🥧 Análise Demográfica & Controle de Vieses"), html.P("Gráficos de proporção para verificar a representatividade da amostra e evitar conclusões enviesadas:", style={"color": "#94A3B8"}), html.Div([graph(fig) for fig in [criar_grafico_genero(data), criar_grafico_renda(data), criar_grafico_trabalho(data), criar_grafico_internet(data)]], style=GRID_4)]
    display_data = data.copy()
    display_data["favorite_AI_tool"] = display_data["favorite_AI_tool"].fillna("Nenhuma")
    display_data["faixa_sono"] = pd.cut(display_data["sleep_hours"], bins=[0, 5, 7, 9, 24], labels=["< 5h (Crítico)", "5h-7h (Alerta)", "7h-9h (Adequado)", "> 9h (Alto)"])
    content += [html.H2("📊 Hábitos de Estudo, Métodos & Rendimento por Curso"), html.Div([graph(criar_grafico_study_gpa(display_data.sample(min(len(display_data), 2500), random_state=42))), graph(criar_grafico_sono_estresse(display_data.dropna(subset=["faixa_sono"])))], style=GRID_2), html.Div([graph(criar_grafico_gpa_major(data)), graph(criar_grafico_metodos_anotacao(data))], style=GRID_2), html.Div([graph(criar_grafico_ia_tools(data)), graph(criar_grafico_cafe_sono(data)), graph(criar_grafico_presenca_gpa(data))], style=GRID_3)]
    content.append(prediction_section(n_clicks, estudo or 4, sono or 7, previsao_estresse or 5, previsao_frequencia or 80, redes or 2, previsao_idade or 20))
    if kmeans_ativo and SKLEARN_DISPONIVEL and len(data) >= k:
        clustered, means, order, names = executar_kmeans(data.copy(), k)
        colors = [ASTRA_COLORS[key] for key in ["accent_emerald", "accent_cyan", "accent_indigo", "accent_amber", "accent_rose"]]
        cards = [html.Div([html.Div(names[cid], style={"fontWeight": "700"}), html.Div(f"{(clustered['cluster_id'] == cid).sum():,} ({(clustered['cluster_id'] == cid).mean() * 100:.1f}%)", style={"color": colors[i], "fontSize": "1.3rem", "fontWeight": "800"}), html.Div(f"GPA: {means.loc[cid, 'GPA']:.2f} • Presença: {means.loc[cid, 'class_attendance_percent']:.0f}% • Estresse: {means.loc[cid, 'mental_stress_level']:.1f}", style={"color": "#CBD5E1", "fontSize": "0.8rem"})], style={**CARD, "borderTopColor": colors[i]}) for i, cid in enumerate(order)]

        bars = clustered.groupby("perfil_cluster")[["study_hours_per_day", "sleep_hours", "mental_stress_level", "social_media_hours"]].mean().reset_index().melt(id_vars=["perfil_cluster"], var_name="Hábito", value_name="Média").replace({"Hábito": {"study_hours_per_day": "Estudo (h/dia)", "sleep_hours": "Sono (h/noite)", "mental_stress_level": "Estresse (0-10)", "social_media_hours": "Redes Sociais (h)"}})
        resumo = clustered.groupby("perfil_cluster")[["GPA", "final_exam_score", "class_attendance_percent", "study_hours_per_day", "sleep_hours", "mental_stress_level", "social_media_hours"]].mean().round(2).reset_index()
        
        content += [html.H2("🤖 Segmentação Inteligente de Perfis (K-Means Clustering)"), html.P(f"O algoritmo analisa hábitos e rendimento para agrupar os {len(clustered):,} estudantes em {k} perfis comportamentais:", style={"color": "#94A3B8"}), html.Div(cards, style={"display": "grid", "gridTemplateColumns": f"repeat({k}, minmax(0, 1fr))", "gap": "12px"}), html.Div([graph(criar_grafico_clusters(clustered, colors, order)), graph(criar_grafico_perfil_clusters(bars))], style=GRID_2), html.Details([html.Summary("📊 Ver Tabela Comparativa Detalhada dos Perfis K-Means"), dash_table.DataTable(data=resumo.to_dict("records"), columns=[{"name": col, "id": col} for col in resumo.columns], style_table={"overflowX": "auto"}, style_header={"backgroundColor": "#1E293B", "color": "#F8FAFC"}, style_cell={"backgroundColor": "#0F172A", "color": "#CBD5E1"})])]
    elif kmeans_ativo and SKLEARN_DISPONIVEL:
        content.append(html.P("ℹ️ Dados insuficientes para o número de clusters selecionado.", style={"color": "#38BDF8"}))
    elif kmeans_ativo:
        content.append(html.P("⚠️ O pacote scikit-learn não foi encontrado. Instale scikit-learn para habilitar esta seção.", style={"color": "#FBBF24"}))
    content.append(html.Details([html.Summary(f"🔍 Visualizar Amostra dos Dados Analisados (Primeiras 100 de {len(data):,} linhas)"), dash_table.DataTable(data=data.head(100).to_dict("records"), columns=[{"name": col, "id": col} for col in data.columns], page_size=15, style_table={"overflowX": "auto"}, style_header={"backgroundColor": "#1E293B", "color": "#F8FAFC"}, style_cell={"backgroundColor": "#0F172A", "color": "#CBD5E1", "minWidth": "120px"})]))
    return content


if __name__ == "__main__":
    app.run(debug=False)