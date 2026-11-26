import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import statsmodels

try:
    from visualization.style import estilizar_grafico, ASTRA_COLORS, CATEGORICAL_PALETTE
except ImportError:
    from ASTRA_Dashboard.visualization.style import estilizar_grafico, ASTRA_COLORS, CATEGORICAL_PALETTE


def _histograma_densidade(dados, col_x, col_y, bins=40):
    contagens, bordas_x, bordas_y = np.histogram2d(
        dados[col_x].to_numpy(dtype=float),
        dados[col_y].to_numpy(dtype=float),
        bins=bins
    )
    centros_x = (bordas_x[:-1] + bordas_x[1:]) / 2
    centros_y = (bordas_y[:-1] + bordas_y[1:]) / 2
    contagens = contagens.T.astype(int)
    return contagens, centros_x, centros_y


def _escala_densidade(cor, max_contagem):
    vermelho = int(cor[1:3], 16)
    verde = int(cor[3:5], 16)
    azul = int(cor[5:7], 16)
    transparente = f"rgba({vermelho}, {verde}, {azul}, 0)"
    baixa = f"rgba({vermelho}, {verde}, {azul}, 0.18)"
    intensa = f"rgba({vermelho}, {verde}, {azul}, 0.95)"
    limiar = min(1 / max_contagem, 0.99)
    return [[0, transparente], [limiar, baixa], [1, intensa]]


def criar_grafico_densidade_por_grupo(
    df,
    col_x,
    col_y,
    col_grupo,
    ordem_grupos,
    mapa_cores,
    label_x,
    label_y,
    titulo,
    destaques=None,
    label_destaques="Densidade de estudantes atípicos"
):
    dados = df[[col_x, col_y, col_grupo]].copy()
    dados[col_x] = pd.to_numeric(dados[col_x], errors="coerce")
    dados[col_y] = pd.to_numeric(dados[col_y], errors="coerce")
    dados = dados.replace([np.inf, -np.inf], np.nan).dropna()
    if dados.empty:
        bins = 40
    else:
        bordas_x = np.histogram_bin_edges(dados[col_x], bins=40)
        bordas_y = np.histogram_bin_edges(dados[col_y], bins=40)
        bins = (bordas_x, bordas_y)

    camadas = []
    maior_contagem = 1
    for grupo in ordem_grupos:
        dados_grupo = dados.loc[dados[col_grupo] == grupo]
        if dados_grupo.empty:
            continue
        histograma = _histograma_densidade(
            dados_grupo,
            col_x,
            col_y,
            bins=bins
        )
        maior_contagem = max(maior_contagem, int(histograma[0].max()))
        camadas.append((grupo, mapa_cores[grupo], histograma))

    fig = go.Figure()
    for grupo, cor, (contagens, centros_x, centros_y) in camadas:
        fig.add_trace(go.Heatmap(
            x=centros_x,
            y=centros_y,
            z=contagens,
            colorscale=_escala_densidade(cor, maior_contagem),
            zmin=0,
            zmax=maior_contagem,
            showscale=False,
            showlegend=True,
            hoverinfo="skip",
            name=grupo,
            legendgroup=grupo
        ))

    if destaques is not None and not destaques.empty:
        dados_destaques = destaques[[col_x, col_y]].copy()
        dados_destaques[col_x] = pd.to_numeric(dados_destaques[col_x], errors="coerce")
        dados_destaques[col_y] = pd.to_numeric(dados_destaques[col_y], errors="coerce")
        dados_destaques = dados_destaques.replace([np.inf, -np.inf], np.nan).dropna()
        if not dados_destaques.empty:
            contagens, centros_x, centros_y = _histograma_densidade(
                dados_destaques,
                col_x,
                col_y,
                bins=bins
            )
            max_destaques = max(1, int(contagens.max()))
            fig.add_trace(go.Heatmap(
                x=centros_x,
                y=centros_y,
                z=contagens,
                colorscale=_escala_densidade(
                    ASTRA_COLORS["accent_amber"],
                    max_destaques
                ),
                zmin=0,
                zmax=max_destaques,
                showscale=False,
                showlegend=True,
                hoverinfo="skip",
                name=label_destaques,
                legendgroup=label_destaques
            ))

    estilizar_grafico(fig, titulo)
    fig.update_layout(
        xaxis_title=label_x,
        yaxis_title=label_y,
        legend_title_text="Perfil do estudante",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        ),
        hovermode=False,
        margin=dict(t=55, b=100)
    )
    return fig


def _criar_grafico_densidade(
    dados,
    col_x,
    col_y,
    label_x,
    label_y,
    titulo,
    cor,
):
    dados = dados[[col_x, col_y]].copy()
    for coluna in (col_x, col_y):
        dados[coluna] = pd.to_numeric(dados[coluna], errors="coerce")
    dados = dados.replace([np.inf, -np.inf], np.nan).dropna()

    fig = go.Figure()
    if not dados.empty:
        contagens, centros_x, centros_y = _histograma_densidade(
            dados,
            col_x,
            col_y
        )
        max_contagem = max(1, int(contagens.max()))
        fig.add_trace(go.Heatmap(
            x=centros_x,
            y=centros_y,
            z=contagens,
            colorscale=_escala_densidade(cor, max_contagem),
            zmin=0,
            zmax=max_contagem,
            showscale=False,
            hoverinfo="skip",
            name="Densidade de estudantes"
        ))
    estilizar_grafico(fig, titulo)
    fig.update_layout(
        xaxis_title=label_x,
        yaxis_title=label_y,
        hovermode=False
    )
    return fig


def _adicionar_tendencia_linear(fig, dados, col_x, col_y, cor, nome="Tendência estatística"):
    if len(dados) > 1 and dados[col_x].nunique() > 1:
        x_values = dados[col_x].to_numpy(dtype=float)
        y_values = dados[col_y].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x_values, y_values, 1)
        x_line = np.linspace(x_values.min(), x_values.max(), 100)
        fig.add_trace(go.Scatter(
            x=x_line,
            y=slope * x_line + intercept,
            mode="lines",
            name=nome,
            line=dict(color=cor, width=2, dash="dash"),
            hoverinfo="skip"
        ))


def criar_scatter_com_tendencia(df_plot, col_x, col_y, label_x, label_y, titulo, cor_pontos, cor_linha):
    dados = df_plot[[col_x, col_y]].apply(pd.to_numeric, errors="coerce").dropna()
    fig = _criar_grafico_densidade(
        dados,
        col_x,
        col_y,
        label_x,
        label_y,
        titulo,
        cor_pontos
    )
    _adicionar_tendencia_linear(fig, dados, col_x, col_y, cor_linha, "Tendência")
    fig.update_yaxes(
    range=[0, 4.5],
    title_text=label_y)

    return fig

def criar_heatmap_correlacao(corr_matrix, titulo):
    rotulos = {
        "study_hours_per_day": "Horas de estudo por dia",
        "class_attendance_percent": "Frequência nas aulas (%)",
        "sleep_hours": "Horas de sono",
        "screen_time_hours": "Tempo de tela",
        "social_media_hours": "Uso de redes sociais",
        "gaming_hours": "Horas de jogos",
        "exercise_hours_per_week": "Exercícios por semana",
        "mental_stress_level": "Nível de estresse mental",
        "AI_tool_usage_hours": "Uso de ferramentas de IA",
        "exam_preparation_days": "Dias de preparação para provas",
        "coffee_consumption_per_day": "Consumo diário de café",
        "extracurricular_hours_per_week": "Atividades extracurriculares",
        "GPA": "Desempenho acadêmico (GPA)",
        "final_exam_score": "Nota da prova final",
        "assignment_score": "Nota das atividades"
    }
    rotulos_x = [rotulos.get(str(coluna), str(coluna)) for coluna in corr_matrix.columns]
    rotulos_y = [rotulos.get(str(indice), str(indice)) for indice in corr_matrix.index]
    valores = corr_matrix.to_numpy(dtype=float)
    texto = [
        [
            f"{valor:.2f}".replace(".", ",") if np.isfinite(valor) else ""
            for valor in linha
        ]
        for linha in valores
    ]
    fig = px.imshow(
        corr_matrix,
        x=rotulos_x,
        y=rotulos_y,
        text_auto=False,
        aspect="auto",
        color_continuous_scale=[ASTRA_COLORS["accent_rose"], "#1E293B", ASTRA_COLORS["accent_cyan"]],
        zmin=-1, zmax=1,
        title= titulo
    )
    fig.update_traces(
        text=texto,
        texttemplate="%{text}",
        textfont=dict(size=10),
        xgap=2,
        ygap=2,
        hovertemplate="%{y} × %{x}<br>Correlação de Pearson: %{text}<extra></extra>"
    )

    estilizar_grafico(fig, titulo)
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=10), automargin=True)
    fig.update_yaxes(tickfont=dict(size=10), automargin=True)
    fig.update_layout(
        height=620,
        margin=dict(l=185, r=35, t=70, b=185),
        coloraxis_colorbar=dict(
            title="Correlação",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["-1", "-0,5", "0", "0,5", "1"]
        )
    )

    return fig

def criar_grafico_genero(df):
    df_gen = (
        df["gender"]
        .replace({
            "Female": "Feminino",
            "Male": "Masculino"
        })
        .value_counts()
        .rename_axis("Gênero")
        .reset_index(name="Total")
    )

    df_gen["Percentual"] = df_gen["Total"] / df_gen["Total"].sum() * 100

    fig = px.bar(
        df_gen,
        x="Percentual",
        y="Gênero",
        orientation="h",
        text="Percentual",
        custom_data=["Total"],
        color="Gênero",
        color_discrete_sequence=[
            ASTRA_COLORS["accent_cyan"],
            ASTRA_COLORS["accent_rose"]
        ]
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Estudantes: %{customdata[0]}<br>"
            "Percentual: %{x:.1f}%"
            "<extra></extra>"
        )
    )

    estilizar_grafico(fig, "Distribuição por Gênero")

    fig.update_layout(
        showlegend=False,
        xaxis_title="Percentual de estudantes (%)",
        yaxis_title=None,
        xaxis_range=[0, 110],
        margin=dict(l=30, r=45, t=65, b=45)
    )

    return fig

def criar_grafico_distribuicao_gpa(df):
    fig = px.histogram(
        df,
        x="GPA",
        nbins=20,
        range_x=[0, 4],
        labels={
            "GPA": "Desempenho acadêmico (GPA)",
            "count": "Quantidade de estudantes"
        },
        color_discrete_sequence=[
            ASTRA_COLORS["accent_cyan"]
        ]
    )

    fig.update_traces(
        marker_line_width=0,
        hovertemplate=(
            "Faixa de GPA: %{x:.2f}<br>"
            "Quantidade de estudantes: %{y}"
            "<extra></extra>"
        )
    )

    estilizar_grafico(
        fig,
        "Distribuição do GPA"
    )

    fig.update_layout(
        bargap=0.08,
        height=360,
        xaxis=dict(
            title="Desempenho acadêmico (GPA, escala de 0 a 4)",
            range=[0, 4]
        ),
        yaxis_title="Quantidade de estudantes",
        margin=dict(l=55, r=25, t=65, b=55)
    )

    return fig

def criar_grafico_renda(df):
    ordem_renda = ["Baixa", "Média", "Alta"]

    df_renda = (
        df["family_income_level"]
        .replace({
            "Low": "Baixa",
            "Middle": "Média",
            "High": "Alta"
        })
        .value_counts()
        .reindex(ordem_renda, fill_value=0)
        .rename_axis("Renda")
        .reset_index(name="Total")
    )

    df_renda["Percentual"] = (
        df_renda["Total"] / df_renda["Total"].sum() * 100
    )

    fig = px.bar(
        df_renda,
        x="Percentual",
        y="Renda",
        orientation="h",
        text="Percentual",
        custom_data=["Total"],
        color="Renda",
        category_orders={"Renda": ordem_renda},
        color_discrete_map={
            "Baixa": ASTRA_COLORS["accent_rose"],
            "Média": ASTRA_COLORS["accent_amber"],
            "Alta": ASTRA_COLORS["accent_emerald"]
        }
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        hovertemplate=(
            "<b>Renda %{y}</b><br>"
            "Estudantes: %{customdata[0]}<br>"
            "Percentual: %{x:.1f}%"
            "<extra></extra>"
        )
    )

    estilizar_grafico(fig, "Distribuição da Renda Familiar")

    fig.update_layout(
        showlegend=False,
        xaxis_title="Percentual de estudantes (%)",
        yaxis_title=None,
        xaxis_range=[0, 110],
        yaxis=dict(autorange="reversed"),
        margin=dict(l=30, r=45, t=65, b=45)
    )

    return fig

def criar_grafico_trabalho(df):
    df_job = (
        df["part_time_job"]
        .replace({
            "Yes": "Trabalha",
            "No": "Não trabalha"
        })
        .value_counts()
        .reindex(["Trabalha", "Não trabalha"], fill_value=0)
        .rename_axis("Trabalho")
        .reset_index(name="Total")
    )

    df_job["Percentual"] = (
        df_job["Total"] / df_job["Total"].sum() * 100
    )

    df_job["Categoria"] = "Estudantes"

    fig = px.bar(
        df_job,
        x="Percentual",
        y="Categoria",
        color="Trabalho",
        orientation="h",
        text="Percentual",
        custom_data=["Total"],
        color_discrete_map={
            "Trabalha": ASTRA_COLORS["accent_indigo"],
            "Não trabalha": ASTRA_COLORS["accent_teal"]
        }
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="inside",
        insidetextanchor="middle",
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            "Estudantes: %{customdata[0]}<br>"
            "Percentual: %{x:.1f}%"
            "<extra></extra>"
        )
    )

    estilizar_grafico(fig, "Trabalho em Meio Período")

    fig.update_layout(
        barmode="stack",
        xaxis_title="Percentual de estudantes (%)",
        yaxis_title=None,
        xaxis_range=[0, 100],
        showlegend=True,
        legend=dict(
            orientation="h",
            y=-0.35,
            x=0.5,
            xanchor="center"
        ),
        margin=dict(l=30, r=30, t=65, b=75)
    )

    return fig

def criar_grafico_internet(df):
    ordem_internet = ["Ruim", "Média", "Boa"]

    df_net = (
        df["internet_quality"]
        .replace({
            "Poor": "Ruim",
            "Average": "Média",
            "Good": "Boa"
        })
        .value_counts()
        .reindex(ordem_internet, fill_value=0)
        .rename_axis("Internet")
        .reset_index(name="Total")
    )

    df_net["Percentual"] = (
        df_net["Total"] / df_net["Total"].sum() * 100
    )

    fig = px.bar(
        df_net,
        x="Percentual",
        y="Internet",
        orientation="h",
        text="Percentual",
        custom_data=["Total"],
        color="Internet",
        category_orders={"Internet": ordem_internet},
        color_discrete_map={
            "Ruim": ASTRA_COLORS["accent_rose"],
            "Média": ASTRA_COLORS["accent_amber"],
            "Boa": ASTRA_COLORS["accent_cyan"]
        }
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        hovertemplate=(
            "<b>Internet %{y}</b><br>"
            "Estudantes: %{customdata[0]}<br>"
            "Percentual: %{x:.1f}%"
            "<extra></extra>"
        )
    )

    estilizar_grafico(fig, "Qualidade da Internet")

    fig.update_layout(
        showlegend=False,
        xaxis_title="Percentual de estudantes (%)",
        yaxis_title=None,
        xaxis_range=[0, 110],
        yaxis=dict(autorange="reversed"),
        margin=dict(l=30, r=45, t=65, b=45)
    )

    return fig

def criar_grafico_study_gpa(df_disp):
    fig_disp = _criar_grafico_densidade(
        df_disp,
        "study_hours_per_day",
        "GPA",
        "Horas de estudo por dia",
        "Desempenho acadêmico (GPA)",
        "Relação entre Horas de Estudo e Desempenho Acadêmico (GPA)",
        ASTRA_COLORS["accent_cyan"],
    )
    return fig_disp

def criar_grafico_estudo_prova(df):
    fig = _criar_grafico_densidade(
        df,
        "study_hours_per_day",
        "final_exam_score",
        "Horas de estudo por dia",
        "Nota da prova final",
        "Relação entre Horas de Estudo e Nota da Prova Final",
        ASTRA_COLORS["accent_violet"]
    )
    fig.update_yaxes(range=[0, 105])
    return fig

def _criar_grafico_relacao(df, col_x, col_y, label_x, label_y, titulo, cor, y_range=None):
    dados = df[[col_x, col_y]].apply(pd.to_numeric, errors="coerce").dropna()
    fig = _criar_grafico_densidade(
        dados,
        col_x,
        col_y,
        label_x,
        label_y,
        titulo,
        cor
    )
    _adicionar_tendencia_linear(
        fig,
        dados,
        col_x,
        col_y,
        ASTRA_COLORS["accent_amber"]
    )
    if y_range is not None:
        fig.update_yaxes(range=y_range)
    return fig

def criar_grafico_tela_gpa(df):
    return _criar_grafico_relacao(
        df,
        "screen_time_hours",
        "GPA",
        "Tempo de tela por dia (horas)",
        "Desempenho acadêmico (GPA, escala de 0 a 4)",
        "Relação entre Tempo de Tela e Desempenho Acadêmico",
        ASTRA_COLORS["accent_violet"],
        [0, 4]
    )

def criar_grafico_atividade_estresse(df):
    return _criar_grafico_relacao(
        df,
        "exercise_hours_per_week",
        "mental_stress_level",
        "Horas de atividade física por semana",
        "Nível de estresse (escala de 1 a 10)",
        "Atividade Física Semanal e Nível de Estresse",
        ASTRA_COLORS["accent_emerald"],
        [0, 10]
    )

def criar_grafico_sono_gpa(df):
    fig = _criar_grafico_densidade(
        df,
        "sleep_hours",
        "GPA",
        "Horas de sono por noite",
        "Desempenho acadêmico (GPA, escala de 0 a 4)",
        "Horas de Sono e Desempenho Acadêmico",
        ASTRA_COLORS["accent_cyan"]
    )
    fig.update_yaxes(range=[0, 4])
    return fig

def criar_grafico_distribuicao_notas(df):
    colunas_notas = {
        "Prova final": "final_exam_score",
        "Atividades": "assignment_score"
    }

    dados = pd.DataFrame({
        rotulo: df[coluna]
        for rotulo, coluna in colunas_notas.items()
    }).melt(
        var_name="Avaliação",
        value_name="Nota"
    ).dropna()

    fig = px.box(
        dados,
        x="Avaliação",
        y="Nota",
        color="Avaliação",
        color_discrete_map={
            "Prova final": ASTRA_COLORS["accent_cyan"],
            "Atividades": ASTRA_COLORS["accent_emerald"]
        },
        labels={
            "Avaliação": "Tipo de avaliação",
            "Nota": "Nota obtida"
        },
        points="outliers"
    )

    fig.update_traces(
        line_width=2,
        hovertemplate=(
            "Avaliação: %{x}<br>"
            "Nota: %{y:.2f}"
            "<extra></extra>"
        )
    )

    estilizar_grafico(
        fig,
        "Distribuição das Notas Acadêmicas"
    )

    fig.update_layout(
        height=420,
        showlegend=False,
        xaxis_title=None,
        yaxis=dict(
            title="Nota",
            range=[0, 100]
        ),
        margin=dict(l=55, r=25, t=65, b=55)
    )

    return fig

def criar_grafico_comparacao_modelos(resultados):
    metricas = resultados[["Modelo", "MAE", "RMSE", "R²"]].copy()
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Erros de previsão (menor é melhor)",
            "Ajuste relativo (maior é melhor)"
        ],
        horizontal_spacing=0.14
    )
    for metrica, cor in [
        ("MAE", ASTRA_COLORS["accent_amber"]),
        ("RMSE", ASTRA_COLORS["accent_rose"])
    ]:
        fig.add_trace(
            go.Bar(
                x=metricas["Modelo"],
                y=metricas[metrica],
                name=metrica,
                marker_color=cor,
                hovertemplate=(
                    "Modelo: %{x}<br>"
                    f"{metrica}: %{{y:.3f}}<extra></extra>"
                )
            ),
            row=1,
            col=1
        )
    fig.add_trace(
        go.Bar(
            x=metricas["Modelo"],
            y=metricas["R²"],
            name="R²",
            marker_color=ASTRA_COLORS["accent_cyan"],
            hovertemplate="Modelo: %{x}<br>R²: %{y:.3f}<extra></extra>"
        ),
        row=1,
        col=2
    )
    fig.update_xaxes(title_text="Modelo preditivo", row=1, col=1)
    fig.update_xaxes(title_text="Modelo preditivo", row=1, col=2)
    fig.update_yaxes(title_text="Valor da métrica", row=1, col=1)
    fig.update_yaxes(title_text="R²", row=1, col=2)
    fig.update_layout(
        barmode="group",
        legend_title_text="Métrica",
        margin=dict(l=55, r=35, t=80, b=70)
    )
    estilizar_grafico(fig, "Comparação de Desempenho dos Modelos Preditivos")
    return fig

def criar_grafico_sono_estresse(df_sono):
    fig_box = px.box(
        df_sono,
        x="faixa_sono",
        y="mental_stress_level",
        color="faixa_sono",
        category_orders={'faixa_sono': ["Menos de 5 h", "5 a 7 h", "7 a 9 h", "Mais de 9 h"]},
        color_discrete_sequence=[
            ASTRA_COLORS["accent_rose"],
            ASTRA_COLORS["accent_amber"],
            ASTRA_COLORS["accent_emerald"],
            ASTRA_COLORS["accent_cyan"]
        ],
        labels={"faixa_sono": "Faixa de sono", "mental_stress_level": "Nível de estresse mental (escala de 1 a 10)"}
    )
    estilizar_grafico(fig_box, "Distribuição do Estresse por Faixa de Sono")
    fig_box.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Mínimo: %{lowerfence:.1f}<br>"
            "1º quartil (Q1): %{q1:.1f}<br>"
            "Mediana: %{median:.1f}<br>"
            "3º quartil (Q3): %{q3:.1f}<br>"
            "Máximo: %{upperfence:.1f}<extra></extra>"
        )
    )
    fig_box.update_layout(showlegend=False)
    return fig_box

def criar_grafico_gpa_major(df):
    dados = df.copy()
    dados["major"] = dados["major"].replace({
        "Biology": "Biologia",
        "Business": "Negócios",
        "Computer Science": "Ciência da Computação",
        "Economics": "Economia",
        "Engineering": "Engenharia",
        "Law": "Direito",
        "Mathematics": "Matemática",
        "Medicine": "Medicina",
        "Physics": "Física",
        "Psychology": "Psicologia"
    })
    fig_box_major = px.box(
        dados,
        x="major",
        y="GPA",
        color="major",
        color_discrete_sequence=CATEGORICAL_PALETTE,
        labels={"major": "Curso", "GPA": "Desempenho acadêmico (GPA)"}
    )
    estilizar_grafico(fig_box_major, "Distribuição do GPA por Curso")
    fig_box_major.update_layout(showlegend=False)
    return fig_box_major

def criar_grafico_metodos_anotacao(df):
    df_metodo = df.copy()
    df_metodo["note_taking_method"] = df_metodo["note_taking_method"].replace({
        "Digital": "Digital",
        "Handwritten": "Manuscrito",
        "Mixed": "Misto"
    })
    df_metodo = df_metodo.groupby("note_taking_method", as_index=False)["final_exam_score"].mean().round(1)
    fig_metodo = px.bar(
        df_metodo,
        x="note_taking_method",
        y="final_exam_score",
        color="note_taking_method",
        color_discrete_sequence=[
            ASTRA_COLORS["accent_cyan"],
            ASTRA_COLORS["accent_indigo"],
            ASTRA_COLORS["accent_emerald"]
        ],
        text_auto=".1f",
        labels={"note_taking_method": "Método de anotação", "final_exam_score": "Nota média na prova final"}
    )
    estilizar_grafico(fig_metodo, "Nota média na prova final por método de anotação")
    fig_metodo.update_layout(showlegend=False, yaxis_range=[0, 100])
    return fig_metodo

def criar_grafico_ia_tools(df):
    df_ia = df[df['favorite_AI_tool'].fillna('None') != 'None']
    fig_ia = px.histogram(
        df_ia,
        x="favorite_AI_tool",
        color="favorite_AI_tool",
        color_discrete_sequence=CATEGORICAL_PALETTE,
        labels={"favorite_AI_tool": "Ferramenta de IA preferida", "count": "Quantidade de estudantes"}
    )
    estilizar_grafico(fig_ia, "Distribuição das Ferramentas de IA Preferidas")
    fig_ia.update_layout(showlegend=False)
    return fig_ia

def criar_grafico_cafe_sono(df):
    df_cafe = df.groupby("coffee_consumption_per_day", as_index=False)["sleep_hours"].mean().round(2)
    fig_cafe = px.bar(
        df_cafe,
        x="coffee_consumption_per_day",
        y="sleep_hours",
        color="coffee_consumption_per_day",
        color_continuous_scale=["#38BDF8", "#FBBF24", "#F43F5E"],
        text_auto=".2f",
        labels={"coffee_consumption_per_day": "Consumo diário de café (xícaras)", "sleep_hours": "Média de sono (horas)"}
    )
    estilizar_grafico(fig_cafe, "Consumo de café e média de horas de sono")
    fig_cafe.update_layout(coloraxis_showscale=False)
    return fig_cafe

def criar_grafico_clusters(df, ordem_perfis, mapa_cores_perfis):
    variancia = df.attrs.get('pca_variancia_explicada', [])
    sufixo_pc1 = f" — {variancia[0] * 100:.1f}% da variância" if len(variancia) > 0 else ""
    sufixo_pc2 = f" — {variancia[1] * 100:.1f}% da variância" if len(variancia) > 1 else ""
    return criar_grafico_densidade_por_grupo(
        df,
        "PC1",
        "PC2",
        "perfil_cluster",
        ordem_perfis,
        mapa_cores_perfis,
        f"PC1: Engajamento acadêmico{sufixo_pc1}",
        f"PC2: Sobrecarga e distração{sufixo_pc2}",
        "Espaço Latente dos Estudantes (PCA) — Agrupamento K-Means"
    )

def criar_grafico_cotovelo_silhueta(df_metricas_k, k_escolhido=None):
    """
    Método do cotovelo (inércia) e coeficiente de silhueta para cada valor de k.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Método do Cotovelo (Inércia)", "Coeficiente de Silhueta")
    )
    fig.add_trace(go.Scatter(
        x=df_metricas_k["k"], y=df_metricas_k["Inércia"],
        mode="lines+markers", name="Inércia",
        line=dict(color=ASTRA_COLORS["accent_cyan"], width=2)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_metricas_k["k"], y=df_metricas_k["Silhouette Score"],
        mode="lines+markers", name="Silhueta",
        line=dict(color=ASTRA_COLORS["accent_amber"], width=2)
    ), row=1, col=2)

    if k_escolhido is not None:
        for coluna in (1, 2):
            fig.add_vline(x=k_escolhido, line_dash="dash",
                          line_color=ASTRA_COLORS["accent_rose"], row=1, col=coluna)

    estilizar_grafico(fig, "Escolha do número de perfis (k)")
    fig.update_xaxes(title_text="Número de grupos (k)", dtick=1,
                     gridcolor=ASTRA_COLORS["border_subtle"])
    fig.update_yaxes(gridcolor=ASTRA_COLORS["border_subtle"])
    fig.update_yaxes(title_text="Inércia", row=1, col=1)
    fig.update_yaxes(title_text="Silhueta", row=1, col=2)
    fig.update_layout(showlegend=False)
    return fig

def criar_grafico_perfil_clusters(df, ordem_perfis, cores_cluster_map):
    fig_km_bars = px.bar(
        df,
        x="Hábito",
        y="Média",
        error_y="Desvio",
        color="perfil_cluster",
        category_orders={
            "perfil_cluster": ordem_perfis
        },
        color_discrete_map= cores_cluster_map,
        barmode="group",
        custom_data=["Média Real", "Desvio Real"],
        labels={"Média": "Média observada", "perfil_cluster": "Perfil"}
    )
    fig_km_bars.update_traces(
        hovertemplate="%{x}<br>Média: %{customdata[0]} ± %{customdata[1]} (desvio-padrão)<extra>%{fullData.name}</extra>"
    )
    estilizar_grafico(fig_km_bars, "Média ± desvio-padrão dos perfis de estudantes")
    fig_km_bars.update_layout(
    legend=dict(
        orientation="h",       # Coloca os itens da legenda na horizontal (um do lado do outro)
        yanchor="top",
        y=-0.2,                # Joga a legenda para baixo do eixo X
        xanchor="center",
        x=0.5,
        font=dict(size=10)     # Diminui um pouco o tamanho da fonte
    ),
    margin=dict(t=40, b=80)    # Aumenta a margem inferior para a legenda caber sem cortar
    )
    return fig_km_bars

def criar_grafico_presenca_gpa(df):
    return _criar_grafico_densidade(
        df,
        "class_attendance_percent",
        "GPA",
        "Frequência às aulas (%)",
        "Desempenho acadêmico (GPA)",
        "Frequência às Aulas × Desempenho Acadêmico (GPA)",
        ASTRA_COLORS["accent_cyan"]
    )


def criar_grafico_desempenho_por_cluster(df_cluster):
    if df_cluster is None or df_cluster.empty:
        return px.bar(title="Desempenho por grupo")

    metric_columns = [coluna for coluna in df_cluster.columns if coluna.endswith('_media')]
    if not metric_columns:
        return px.bar(title="Desempenho por grupo")

    df_long = df_cluster[["cluster_id"] + metric_columns].copy()
    df_long.columns = ["cluster_id" if coluna == "cluster_id" else coluna.replace('_media', '').replace('_', ' ') for coluna in df_long.columns]
    df_long = df_long.melt(id_vars=["cluster_id"], var_name="Indicador", value_name="Média")
    df_long["Indicador"] = df_long["Indicador"].replace({
        "final exam score": "Nota da prova final",
        "assignment score": "Nota das atividades",
        "GPA": "Desempenho acadêmico (GPA)"
    })

    fig = px.bar(
        df_long,
        x="cluster_id",
        y="Média",
        color="Indicador",
        barmode="group",
        labels={"cluster_id": "Grupo", "Média": "Média do indicador", "Indicador": "Indicador"},
        title="Desempenho médio por grupo"
    )
    estilizar_grafico(fig, "Desempenho médio por grupo")
    return fig

def criar_grafico_real_vs_previsto(y_test, y_pred):
    """
    Gera um gráfico comparando os valores reais vs previstos no conjunto de teste.
    """
    df_avaliacao = pd.DataFrame({
        "GPA Real": y_test,
        "GPA Previsto": y_pred
    })
    
    fig = _criar_grafico_densidade(
        df_avaliacao,
        "GPA Real",
        "GPA Previsto",
        "Desempenho acadêmico (GPA) observado",
        "Desempenho acadêmico (GPA) previsto",
        "Validação do Conjunto de Teste: GPA Observado e Previsto",
        ASTRA_COLORS["accent_cyan"]
    )
    
    # Adiciona uma linha de referência ideal (45 graus, onde Real == Previsto)
    min_val = min(df_avaliacao["GPA Real"].min(), df_avaliacao["GPA Previsto"].min())
    max_val = max(df_avaliacao["GPA Real"].max(), df_avaliacao["GPA Previsto"].max())
    
    fig.add_shape(
        type="line",
        x0=min_val, y0=min_val,
        x1=max_val, y1=max_val,
        line=dict(color="#38BDF8", dash="dash", width=2)
    )
    
    return fig

def criar_grafico_dispersao_correlacao(df, coluna_x, coluna_y):
    rotulos = {
        "study_hours_per_day": "Horas de estudo por dia",
        "class_attendance_percent": "Frequência nas aulas (%)",
        "sleep_hours": "Horas de sono por noite",
        "screen_time_hours": "Tempo de tela por dia (horas)",
        "social_media_hours": "Tempo em redes sociais (horas)",
        "gaming_hours": "Tempo com jogos (horas)",
        "exercise_hours_per_week": "Atividade física (horas por semana)",
        "mental_stress_level": "Nível de estresse mental",
        "AI_tool_usage_hours": "Horas de uso de ferramentas de IA",
        "exam_preparation_days": "Dias de preparação para provas",
        "coffee_consumption_per_day": "Consumo diário de café",
        "extracurricular_hours_per_week": "Horas semanais em atividades extracurriculares",
        "GPA": "Desempenho acadêmico (GPA)",
        "final_exam_score": "Nota da prova final",
        "assignment_score": "Nota das atividades"
    }

    dados = df[[coluna_x, coluna_y]].copy()
    dados[coluna_x] = pd.to_numeric(dados[coluna_x], errors="coerce")
    dados[coluna_y] = pd.to_numeric(dados[coluna_y], errors="coerce")

    dados = dados.replace([np.inf, -np.inf], np.nan).dropna()

    nome_x = rotulos.get(coluna_x, coluna_x)
    nome_y = rotulos.get(coluna_y, coluna_y)

    fig = _criar_grafico_densidade(
        dados,
        coluna_x,
        coluna_y,
        nome_x,
        nome_y,
        f"{nome_x} × {nome_y}",
        ASTRA_COLORS["accent_cyan"]
    )
    _adicionar_tendencia_linear(
        fig,
        dados,
        coluna_x,
        coluna_y,
        ASTRA_COLORS["accent_rose"],
        "Tendência linear"
    )

    fig.update_layout(
        height=450,
        xaxis_title=nome_x,
        yaxis_title=nome_y,
        margin=dict(l=60, r=30, t=70, b=60)
    )

    return fig