import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from visualization.style import estilizar_grafico, ASTRA_COLORS, CATEGORICAL_PALETTE

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
            line=dict(color=cor_linha, width=2.5, dash='dash')
        ))
    estilizar_grafico(fig, titulo)
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
    fig.update_layout(coloraxis_showscale=False)

    return fig

def criar_grafico_genero(df):
    df_gen = df['gender'].value_counts().reset_index()
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

def criar_grafico_renda(df):
    df_renda = df['family_income_level'].value_counts().reset_index()
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
    df_net = df['internet_quality'].value_counts().reset_index()
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
            "GPA": "Média Acadêmica (GPA)",
            "mental_stress_level": "Nível de Estresse"
        },
        color_continuous_scale=["#38BDF8", "#818CF8", "#F43F5E"],
        opacity=0.75
    )
    fig_disp.update_traces(marker=dict(size=6))
    estilizar_grafico(fig_disp, "Impacto do Estudo Diário no Desempenho (GPA)")
    return fig_disp

def criar_grafico_sono_estresse(df_sono):
    fig_box = px.box(
        df_sono,
        x="faixa_sono",
        y="mental_stress_level",
        color="faixa_sono",
        category_orders={'faixa_sono': ["< 5h (Crítico)", "5h-7h (Alerta)", "7h-9h (Adequado)", "> 9h (Alto)"]},
        color_discrete_sequence=[
            ASTRA_COLORS["accent_rose"],
            ASTRA_COLORS["accent_amber"],
            ASTRA_COLORS["accent_emerald"],
            ASTRA_COLORS["accent_cyan"]
        ],
        labels={"faixa_sono": "Faixa de Sono", "mental_stress_level": "Nível de Estresse (1 a 10)"}
    )
    estilizar_grafico(fig_box, "Estresse Reportado por Faixa de Descanso")
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
    df_metodo = df.groupby("note_taking_method", as_index=False)["final_exam_score"].mean().round(1)
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
    estilizar_grafico(fig_metodo, "Eficácia Média do Método de Anotação no Exame")
    fig_metodo.update_layout(showlegend=False, yaxis_range=[80, 105])
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

def criar_grafico_clusters(df_cluster, cores_cluster_map, ordem_gpa):
    fig_km_scatter = px.scatter(
        df_cluster,
        x="class_attendance_percent",
        y="GPA",
        color="perfil_cluster",
        color_discrete_sequence=[cores_cluster_map[ordem_gpa.index(cid) % len(cores_cluster_map)] for cid in sorted(df_cluster['cluster_id'].unique())],
        labels={"class_attendance_percent": "Frequência às Aulas (%)", "GPA": "GPA Acadêmico (0 a 4.0)", "perfil_cluster": "Perfil K-Means"},
        hover_data=["study_hours_per_day", "mental_stress_level", "sleep_hours"]
    )
    fig_km_scatter.update_traces(marker=dict(size=6, opacity=0.75))
    estilizar_grafico(fig_km_scatter, "Dispersão dos Perfis: Frequência vs. GPA")
    return fig_km_scatter

def criar_grafico_perfil_clusters(df):
    fig_km_bars = px.bar(
        df,
        x="Hábito",
        y="Média",
        color="perfil_cluster",
        barmode="group",
        color_discrete_sequence=CATEGORICAL_PALETTE,
        labels={"Média": "Média do Hábito", "perfil_cluster": "Perfil"}
    )
    estilizar_grafico(fig_km_bars, "Comparativo dos Hábitos Médios entre os Perfis")
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