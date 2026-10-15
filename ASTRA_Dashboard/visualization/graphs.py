import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

try:
    from visualization.style import estilizar_grafico, ASTRA_COLORS, CATEGORICAL_PALETTE
except ImportError:
    from ASTRA_Dashboard.visualization.style import estilizar_grafico, ASTRA_COLORS, CATEGORICAL_PALETTE

def criar_scatter_com_tendencia(df_plot, col_x, col_y, label_x, label_y, titulo, cor_pontos, cor_linha):
    tamanho_amostra = min(len(df_plot), 2500)
    df_amostra = df_plot.sample(tamanho_amostra, random_state=42) if len(df_plot) > tamanho_amostra else df_plot
    
    fig = px.scatter(
        df_amostra,
        x=col_x,
        y=col_y,
        labels={col_x: label_x, col_y: label_y}
    )
    fig.update_traces(
        marker=dict(
            color=cor_pontos,
            size=6,
            opacity=0.6,
            line=dict(width=0.5, color="#0B1120")
        )
    )
    if len(df_plot) > 1 and df_plot[col_x].nunique() > 1:
        x_vals = df_plot[col_x].astype(float).values
        y_vals = df_plot[col_y].astype(float).values
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        y_line = slope * x_line + intercept
        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            name=f'Tendência ({slope:+.3f})',
            line=dict(color=cor_linha, width=2.5, dash='dash'),
            opacity = 0.5
        )
        )
    estilizar_grafico(fig, titulo)

    fig.update_yaxes(
    range=[0, 4.5],
    title_text=label_y)

    return fig

def criar_heatmap_correlacao(corr_matrix, titulo):
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=[ASTRA_COLORS["accent_rose"], "#1E293B", ASTRA_COLORS["accent_cyan"]],
        zmin=-1, zmax=1,
        title= titulo
    )

    estilizar_grafico(fig, titulo)
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Correlação",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["-1", "-0,5", "0", "0,5", "1"]
        )
    )

    return fig

def criar_grafico_genero(df):
    df_gen = df['gender'].replace({
        "Female": "Feminino",
        "Male": "Masculino"
    }).value_counts().reset_index()
    df_gen.columns = ['Gênero', 'Total']
    fig_pz_gen = px.pie(
        df_gen,
        names='Gênero',
        values='Total',
        hole=0.45,
        color_discrete_sequence=[ASTRA_COLORS["accent_cyan"], ASTRA_COLORS["accent_rose"], ASTRA_COLORS["accent_amber"]]
    )
    fig_pz_gen.update_traces(textposition='inside', textinfo='percent+label')
    estilizar_grafico(fig_pz_gen, "Distribuição por Gênero")
    fig_pz_gen.update_layout(showlegend=False)
    return fig_pz_gen

def criar_grafico_distribuicao_gpa(df):
    fig = px.histogram(
        df,
        x="GPA",
        nbins=20,
        range_x=[0, 4],
        labels={
            "GPA": "Desempenho acadêmico (GPA, escala de 0 a 4)",
            "count": "Quantidade de estudantes"
        },
        color_discrete_sequence=[ASTRA_COLORS["accent_cyan"]]
    )
    fig.update_traces(
        hovertemplate=(
            "Faixa de GPA: %{x}<br>"
            "Quantidade de estudantes: %{y}<extra></extra>"
        )
    )
    estilizar_grafico(fig, "Distribuição do Desempenho Acadêmico (GPA)")
    fig.update_layout(bargap=0.08)
    return fig

def criar_grafico_renda(df):
    df_renda = df['family_income_level'].replace({
        "Low": "Baixa",
        "Middle": "Média",
        "High": "Alta"
    }).value_counts().reset_index()
    df_renda.columns = ['Renda', 'Total']
    fig_pz_renda = px.pie(
        df_renda,
        names='Renda',
        values='Total',
        hole=0.45,
        color_discrete_sequence=[ASTRA_COLORS["accent_emerald"], ASTRA_COLORS["accent_amber"], ASTRA_COLORS["accent_rose"]]
    )
    fig_pz_renda.update_traces(textposition='inside', textinfo='percent+label')
    estilizar_grafico(fig_pz_renda, "Renda Familiar")
    fig_pz_renda.update_layout(showlegend=False)
    return fig_pz_renda

def criar_grafico_trabalho(df):
    df_job = df['part_time_job'].replace({'Yes': 'Trabalha', 'No': 'Não Trabalha'}).value_counts().reset_index()
    df_job.columns = ['Trabalho', 'Total']
    fig_pz_job = px.pie(
        df_job,
        names='Trabalho',
        values='Total',
        hole=0.45,
        color_discrete_sequence=[ASTRA_COLORS["accent_indigo"], ASTRA_COLORS["accent_teal"]]
    )
    fig_pz_job.update_traces(textposition='inside', textinfo='percent+label')
    estilizar_grafico(fig_pz_job, "Trabalho Meio Período")
    fig_pz_job.update_layout(showlegend=False)
    return fig_pz_job

def criar_grafico_internet(df):
    df_net = df['internet_quality'].replace({
        "Good": "Boa",
        "Average": "Média",
        "Poor": "Ruim"
    }).value_counts().reset_index()
    df_net.columns = ['Internet', 'Total']
    fig_pz_net = px.pie(
        df_net,
        names='Internet',
        values='Total',
        hole=0.45,
        color_discrete_sequence=[ASTRA_COLORS["accent_cyan"], ASTRA_COLORS["accent_indigo"], ASTRA_COLORS["accent_rose"]]
    )
    fig_pz_net.update_traces(textposition='inside', textinfo='percent+label')
    estilizar_grafico(fig_pz_net, "Qualidade da Internet")
    fig_pz_net.update_layout(showlegend=False)
    return fig_pz_net

def criar_grafico_study_gpa(df_disp):
    fig_disp = px.scatter(
        df_disp,
        x="study_hours_per_day",
        y="GPA",
        color="mental_stress_level",
        hover_data=["major", "university_year", "favorite_AI_tool"],
        labels={
            "study_hours_per_day": "Horas de Estudo / Dia",
            "GPA": "Desempenho acadêmico (GPA)",
            "mental_stress_level": "Nível de Estresse",
            "major": "Curso",
            "university_year": "Ano acadêmico",
            "favorite_AI_tool": "Ferramenta de IA preferida"
        },
        color_continuous_scale=["#38BDF8", "#818CF8", "#F43F5E"],
        opacity=0.75
    )
    fig_disp.update_traces(marker=dict(size=6))
    estilizar_grafico(fig_disp, "Relação entre Estudo Diário e Desempenho (GPA)")
    return fig_disp

def criar_grafico_estudo_prova(df):
    fig = px.scatter(
        df,
        x="study_hours_per_day",
        y="final_exam_score",
        labels={
            "study_hours_per_day": "Horas de estudo por dia",
            "final_exam_score": "Nota da prova final"
        },
        hover_data={
            "study_hours_per_day": ":.1f",
            "final_exam_score": ":.1f"
        },
        opacity=0.65,
        color_discrete_sequence=[ASTRA_COLORS["accent_violet"]]
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color="#0B1120")))
    estilizar_grafico(fig, "Relação entre Horas de Estudo e Nota da Prova Final")
    fig.update_yaxes(range=[0, 105])
    return fig

def _criar_grafico_relacao(df, col_x, col_y, label_x, label_y, titulo, cor, y_range=None):
    dados = df[[col_x, col_y]].apply(pd.to_numeric, errors="coerce").dropna()
    fig = px.scatter(
        dados,
        x=col_x,
        y=col_y,
        labels={col_x: label_x, col_y: label_y},
        opacity=0.65,
        color_discrete_sequence=[cor]
    )
    fig.update_traces(
        marker=dict(size=6, line=dict(width=0.5, color="#0B1120")),
        hovertemplate=(
            f"{label_x}: %{{x:.2f}}<br>"
            f"{label_y}: %{{y:.2f}}<extra></extra>"
        )
    )
    if len(dados) > 1 and dados[col_x].nunique() > 1:
        x_values = dados[col_x].to_numpy(dtype=float)
        y_values = dados[col_y].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x_values, y_values, 1)
        x_line = np.linspace(x_values.min(), x_values.max(), 100)
        fig.add_trace(go.Scatter(
            x=x_line,
            y=slope * x_line + intercept,
            mode="lines",
            name="Tendência estatística",
            line=dict(color=ASTRA_COLORS["accent_amber"], width=2, dash="dash"),
            hovertemplate="Tendência estatística<extra></extra>"
        ))
    estilizar_grafico(fig, titulo)
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
    fig = px.scatter(
        df,
        x="sleep_hours",
        y="GPA",
        labels={
            "sleep_hours": "Horas de sono por noite",
            "GPA": "Desempenho acadêmico (GPA, escala de 0 a 4)"
        },
        hover_data={
            "sleep_hours": ":.1f",
            "GPA": ":.2f"
        },
        opacity=0.65,
        color_discrete_sequence=[ASTRA_COLORS["accent_cyan"]]
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color="#0B1120")))
    estilizar_grafico(fig, "Horas de Sono e Desempenho Acadêmico")
    fig.update_yaxes(range=[0, 4])
    return fig

def criar_grafico_distribuicao_notas(df):
    colunas_notas = {
        "Nota do exame final": "final_exam_score",
        "Nota das atividades avaliativas": "assignment_score"
    }
    dados = pd.DataFrame({
        rotulo: df[coluna]
        for rotulo, coluna in colunas_notas.items()
    }).melt(var_name="Avaliação", value_name="Nota").dropna()
    fig = px.box(
        dados,
        x="Avaliação",
        y="Nota",
        color="Avaliação",
        color_discrete_sequence=[
            ASTRA_COLORS["accent_cyan"],
            ASTRA_COLORS["accent_emerald"]
        ],
        labels={"Avaliação": "Tipo de avaliação", "Nota": "Nota"},
        points="outliers"
    )
    estilizar_grafico(fig, "Distribuição das Notas Acadêmicas")
    fig.update_layout(showlegend=False)
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
        labels={"faixa_sono": "Faixa de sono", "mental_stress_level": "Nível de estresse (escala de 1 a 10)"}
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
    fig_box_major = px.box(
        df,
        x="major",
        y="GPA",
        color="major",
        color_discrete_sequence=CATEGORICAL_PALETTE,
        labels={"major": "Curso", "GPA": "GPA Acadêmico"}
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
        labels={"note_taking_method": "Método de Anotação", "final_exam_score": "Nota Média no Exame Final"}
    )
    estilizar_grafico(fig_metodo, "Nota média no exame final por método de anotação")
    fig_metodo.update_layout(showlegend=False, yaxis_range=[0, 100])
    return fig_metodo

def criar_grafico_ia_tools(df):
    df_ia = df[df['favorite_AI_tool'].fillna('None') != 'None']
    fig_ia = px.histogram(
        df_ia,
        x="favorite_AI_tool",
        color="favorite_AI_tool",
        color_discrete_sequence=CATEGORICAL_PALETTE,
        labels={"favorite_AI_tool": "Ferramenta de IA", "count": "Quantidade de Estudantes"}
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
        labels={"coffee_consumption_per_day": "Xícaras de Café / Dia", "sleep_hours": "Média de Sono (Horas)"}
    )
    estilizar_grafico(fig_cafe, "Consumo de Café vs. Horas de Sono Médio")
    fig_cafe.update_layout(coloraxis_showscale=False)
    return fig_cafe

def criar_grafico_clusters(df_cluster, ordem_perfis, mapa_cores_perfis):
    dados = df_cluster.copy()
    dados["frequencia_exibicao"] = dados["class_attendance_percent"].map(
        lambda valor: f"{valor:.0f}%"
    )
    dados["gpa_exibicao"] = dados["GPA"].map(lambda valor: f"{valor:.2f}".replace(".", ","))
    dados["estudo_exibicao"] = dados["study_hours_per_day"].map(
        lambda valor: f"{valor:.1f}".replace(".", ",")
    )
    dados["estresse_exibicao"] = dados["mental_stress_level"].map(
        lambda valor: f"{valor:.1f}".replace(".", ",")
    )
    dados["sono_exibicao"] = dados["sleep_hours"].map(
        lambda valor: f"{valor:.1f}".replace(".", ",")
    )
    fig_km_scatter = px.scatter(
        dados,
        x="class_attendance_percent",
        y="GPA",
        color="perfil_cluster",

        category_orders={
            "perfil_cluster": ordem_perfis
        },

        color_discrete_map=mapa_cores_perfis,

        labels={
            "class_attendance_percent": "Frequência das aulas (%)",
            "GPA": "Desempenho acadêmico (GPA, escala de 0 a 4)",
            "perfil_cluster": "Perfil"
        },
        custom_data=[
            "frequencia_exibicao",
            "gpa_exibicao",
            "estudo_exibicao",
            "estresse_exibicao",
            "sono_exibicao"
        ]
    )

    fig_km_scatter.update_traces(
        marker=dict(
            size=8,
            opacity=0.8,
            line=dict(
                width=0.8,
                color="#0B1120"
            )
        )
    )

    estilizar_grafico(
        fig_km_scatter,
        "Perfis de Estudantes: Frequência às Aulas e Desempenho Acadêmico"
    )
    fig_km_scatter.update_traces(
        hovertemplate=(
            "<b>Perfil: %{fullData.name}</b><br>"
            "Frequência das aulas: %{customdata[0]}<br>"
            "Desempenho acadêmico (GPA): %{customdata[1]} de 4,00<br>"
            "Horas de estudo por dia: %{customdata[2]}<br>"
            "Nível de estresse: %{customdata[3]}<br>"
            "Horas de sono por noite: %{customdata[4]}"
            "<extra></extra>"
        )
    )
    fig_km_scatter.update_yaxes(
        range=[0, 4]
    )

    return fig_km_scatter

def criar_grafico_perfil_clusters(df, ordem_perfis, cores_cluster_map):
    fig_km_bars = px.bar(
        df,
        x="Hábito",
        y="Média",
        color="perfil_cluster",
        category_orders={
            "perfil_cluster": ordem_perfis
        },
        color_discrete_map= cores_cluster_map,
        barmode="group",
        labels={"Média": "Média observada", "perfil_cluster": "Perfil"}
    )
    estilizar_grafico(fig_km_bars, "Médias observadas dos perfis de estudantes")
    return fig_km_bars

def criar_grafico_presenca_gpa(df):
    fig = px.scatter(
        df,
        x="class_attendance_percent",
        y="GPA",
        labels={
            "class_attendance_percent": "Presença nas Aulas (%)",
            "GPA": "GPA"
        })

    estilizar_grafico(fig,"Presença nas Aulas × GPA")
    return fig


def criar_grafico_desempenho_por_cluster(df_cluster):
    if df_cluster is None or df_cluster.empty:
        return px.bar(title="Desempenho por cluster")

    metric_columns = [coluna for coluna in df_cluster.columns if coluna.endswith('_media')]
    if not metric_columns:
        return px.bar(title="Desempenho por cluster")

    df_long = df_cluster[["cluster_id"] + metric_columns].copy()
    df_long.columns = ["cluster_id" if coluna == "cluster_id" else coluna.replace('_media', '').replace('_', ' ') for coluna in df_long.columns]
    df_long = df_long.melt(id_vars=["cluster_id"], var_name="Indicador", value_name="Média")
    df_long["Indicador"] = df_long["Indicador"].str.replace("final exam score", "Nota do Exame Final").str.replace("assignment score", "Nota de Atividades").str.replace("GPA", "GPA")

    fig = px.bar(
        df_long,
        x="cluster_id",
        y="Média",
        color="Indicador",
        barmode="group",
        labels={"cluster_id": "Cluster", "Média": "Média do indicador", "Indicador": "Indicador"},
        title="Desempenho médio por cluster"
    )
    estilizar_grafico(fig, "Desempenho médio por cluster")
    return fig