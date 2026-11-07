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
            "GPA": "GPA",
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
            title="GPA (escala de 0 a 4)",
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

def criar_grafico_clusters(df, ordem_perfis, mapa_cores_perfis):
    # Variância explicada por cada componente (calculada em executar_kmeans)
    variancia = df.attrs.get('pca_variancia_explicada', [])
    sufixo_pc1 = f" — {variancia[0] * 100:.1f}% da variância" if len(variancia) > 0 else ""
    sufixo_pc2 = f" — {variancia[1] * 100:.1f}% da variância" if len(variancia) > 1 else ""

    fig = px.scatter(
        df,
        x='PC1',                    # <--- Usa o Componente Principal 1
        y='PC2',                    # <--- Usa o Componente Principal 2
        color='perfil_cluster',
        category_orders={"perfil_cluster": ordem_perfis},
        color_discrete_map=mapa_cores_perfis,
        opacity=0.6,                # <--- Transparência para ver onde há mais concentração
        labels={
            # PC1: frequência e estudo (+), estresse e redes sociais (−)
            'PC1': f'PC1: Engajamento acadêmico{sufixo_pc1}',
            # PC2: estresse, redes sociais e estudo (+), sono (−)
            'PC2': f'PC2: Sobrecarga e distração{sufixo_pc2}',
            'perfil_cluster': 'Perfil do Estudante'
        }
    )
    
    # Deixa os pontos menores e com uma bordinha sutil para definição limpa
    fig.update_traces(
        marker=dict(size=6, line=dict(width=0.3, color='#0F172A'))
    )
    fig.update_layout(
        legend=dict(
            orientation="h",       # Coloca os itens da legenda na horizontal
            yanchor="top",
            y=-0.25,               # Joga a legenda um pouco mais para baixo do eixo X
            xanchor="center",
            x=0.5,
            font=dict(size=11)     # Tamanho de fonte ideal para leitura
        ),
        margin=dict(t=40, b=100)   # Aumenta a margem inferior para acomodar as linhas da legenda
    )
    
    estilizar_grafico(fig, "Espaço Latente dos Estudantes (PCA) — Agrupamento K-Means")
    return fig

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
        color="perfil_cluster",
        category_orders={
            "perfil_cluster": ordem_perfis
        },
        color_discrete_map= cores_cluster_map,
        barmode="group",
        labels={"Média": "Média observada", "perfil_cluster": "Perfil"}
    )
    estilizar_grafico(fig_km_bars, "Médias observadas dos perfis de estudantes")
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

def criar_grafico_real_vs_previsto(y_test, y_pred):
    """
    Gera um gráfico comparando os valores reais vs previstos no conjunto de teste.
    """
    df_avaliacao = pd.DataFrame({
        "GPA Real": y_test,
        "GPA Previsto": y_pred
    })
    
    fig = px.scatter(
        df_avaliacao,
        x="GPA Real",
        y="GPA Previsto",
        opacity=0.5,
        labels={"GPA Real": "GPA Real (Observado)", "GPA Previsto": "GPA Previsto (Modelo)"}
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
    
    estilizar_grafico(fig, "Validação do Grupo de Teste: GPA Real vs. Previsto")
    return fig

def criar_grafico_dispersao_correlacao(df, coluna_x, coluna_y):
    rotulos = {
        "study_hours_per_day": "Horas de estudo por dia",
        "class_attendance_percent": "Frequência às aulas (%)",
        "sleep_hours": "Horas de sono por dia",
        "screen_time_hours": "Tempo de tela por dia (horas)",
        "social_media_hours": "Tempo em redes sociais (horas)",
        "gaming_hours": "Tempo com jogos (horas)",
        "exercise_hours_per_week": "Atividade física (horas por semana)",
        "mental_stress_level": "Nível de estresse",
        "AI_tool_usage_hours": "Uso de ferramentas de IA (horas)",
        "exam_preparation_days": "Dias de preparação para provas",
        "coffee_consumption_per_day": "Consumo diário de café",
        "extracurricular_hours_per_week": "Atividades extracurriculares (horas por semana)",
        "GPA": "GPA",
        "final_exam_score": "Nota da prova final",
        "assignment_score": "Nota das atividades"
    }

    dados = df[[coluna_x, coluna_y]].copy()
    dados[coluna_x] = pd.to_numeric(dados[coluna_x], errors="coerce")
    dados[coluna_y] = pd.to_numeric(dados[coluna_y], errors="coerce")

    dados = dados.replace([np.inf, -np.inf], np.nan).dropna()

    nome_x = rotulos.get(coluna_x, coluna_x)
    nome_y = rotulos.get(coluna_y, coluna_y)

    fig = px.scatter(
        dados,
        x=coluna_x,
        y=coluna_y,
        labels={
            coluna_x: nome_x,
            coluna_y: nome_y
        },
        opacity=0.6,
        color_discrete_sequence=[ASTRA_COLORS["accent_cyan"]],
        trendline="ols",
        hover_data={
            coluna_x: ":.2f",
            coluna_y: ":.2f"
        }
    )

    estilizar_grafico(
        fig,
        f"{nome_x} × {nome_y}"
    )

    fig.update_layout(
        height=450,
        xaxis_title=nome_x,
        yaxis_title=nome_y,
        margin=dict(l=60, r=30, t=70, b=60)
    )

    fig.update_traces(
        line=dict(
            color=ASTRA_COLORS["accent_rose"],
            width=3,
        ),
        selector=dict(mode="lines")
    )

    return fig