import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 1. Configuração da Página (Profissional, sem ícones genéricos)
st.set_page_config(
    page_title="ASTRA - Inteligência Acadêmica",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Paleta de Cores de Alto Contraste & Design Tokens ASTRA
ASTRA_COLORS = {
    "bg_dark": "#0B1120",
    "card_dark": "#1E293B",
    "border_subtle": "#334155",
    "text_primary": "#F8FAFC",
    "text_muted": "#94A3B8",
    "accent_cyan": "#38BDF8",
    "accent_indigo": "#818CF8",
    "accent_emerald": "#34D399",
    "accent_amber": "#FBBF24",
    "accent_rose": "#F43F5E",
}

CATEGORICAL_PALETTE = [
    ASTRA_COLORS["accent_cyan"],
    ASTRA_COLORS["accent_indigo"],
    ASTRA_COLORS["accent_emerald"],
    ASTRA_COLORS["accent_amber"],
    ASTRA_COLORS["accent_rose"],
    "#A78BFA",
    "#2DD4BF",
]

# 3. Estilização CSS Moderna
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        
        * {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        .stApp {
            background-color: #0B1120;
            color: #F8FAFC;
        }
        
        h1, h2, h3, h4 {
            color: #F8FAFC !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
        }
        
        p, span, label {
            color: #CBD5E1;
        }
        
        section[data-testid="stSidebar"] {
            background-color: #0F172A !important;
            border-right: 1px solid #1E293B;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #38BDF8 !important;
        }
        
        /* Cards de Métricas (KPIs) */
        div[data-testid="metric-container"] {
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            border: 1px solid #334155;
            border-top: 3px solid #38BDF8;
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
            transition: all 0.25s ease;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            border-color: #38BDF8;
            box-shadow: 0 8px 24px rgba(56, 189, 248, 0.15);
        }
        div[data-testid="metric-container"] label {
            color: #94A3B8 !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
            color: #F8FAFC !important;
            font-size: 1.85rem !important;
            font-weight: 800 !important;
        }
        
        /* Moldura de Gráficos */
        .stPlotlyChart {
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 10px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Função auxiliar para padronizar o estilo dos gráficos Plotly
def estilizar_grafico(fig, titulo=""):
    fig.update_layout(
        template="plotly_dark",
        title={
            "text": f"<b>{titulo}</b>",
            "font": {"size": 15, "color": "#F8FAFC", "family": "Plus Jakarta Sans"}
        },
        paper_bgcolor=ASTRA_COLORS["card_dark"],
        plot_bgcolor=ASTRA_COLORS["card_dark"],
        font=dict(color="#E2E8F0", family="Plus Jakarta Sans"),
        xaxis=dict(
            gridcolor=ASTRA_COLORS["border_subtle"],
            zerolinecolor=ASTRA_COLORS["border_subtle"],
            tickfont=dict(color="#94A3B8", size=11),
            title_font=dict(color="#CBD5E1", size=12)
        ),
        yaxis=dict(
            gridcolor=ASTRA_COLORS["border_subtle"],
            zerolinecolor=ASTRA_COLORS["border_subtle"],
            tickfont=dict(color="#94A3B8", size=11),
            title_font=dict(color="#CBD5E1", size=12)
        ),
        legend=dict(
            font=dict(color="#CBD5E1", size=11),
            bgcolor="rgba(15, 23, 42, 0.7)",
            bordercolor=ASTRA_COLORS["border_subtle"],
            borderwidth=1
        ),
        margin=dict(l=45, r=35, t=55, b=45)
    )
    return fig

# 4. Carregamento dos Dados com Cache
@st.cache_data
def carregar_dados():
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    caminhos = [
        os.path.join(diretorio_atual, "global_university_students_performance_habits_10000.csv"),
        os.path.join(diretorio_atual, "..", "Data", "global_university_students_performance_habits_10000.csv"),
        "global_university_students_performance_habits_10000.csv",
        os.path.join("Data", "global_university_students_performance_habits_10000.csv"),
        os.path.join("ASTRA_Dashboard", "global_university_students_performance_habits_10000.csv"),
    ]
    for caminho in caminhos:
        if os.path.exists(caminho):
            return pd.read_csv(caminho)
    raise FileNotFoundError("Arquivo de dados CSV não encontrado.")

try:
    df_raw = carregar_dados()
except Exception as erro:
    st.error(f"Erro ao carregar o arquivo CSV: {erro}")
    st.stop()

# 5. Barra Lateral: Filtros Detalhados
st.sidebar.title("🎯 Filtros ASTRA")
st.sidebar.caption("Personalize sua análise:")

# Filtro: Quantidade de Dados a Analisar (Controle de Amostra)
total_registros = len(df_raw)
qtd_analise = st.sidebar.slider(
    "Volume de Dados Analisado:",
    min_value=500,
    max_value=total_registros,
    value=total_registros,
    step=500,
    help="Define quantos registros da base serão considerados na análise."
)

st.sidebar.markdown("---")

# Filtro: Cursos (Padrão: TODOS selecionados para contemplar os 10.000)
cursos_unicos = sorted(df_raw['major'].dropna().unique())
cursos_sel = st.sidebar.multiselect(
    "Cursos (Graduação):",
    options=cursos_unicos,
    default=cursos_unicos
)

# Filtro: Ano Acadêmico
anos_unicos = sorted(df_raw['university_year'].dropna().unique())
anos_sel = st.sidebar.multiselect(
    "Período / Ano Acadêmico:",
    options=anos_unicos,
    default=anos_unicos
)

# Filtro: Gênero
generos_unicos = sorted(df_raw['gender'].dropna().unique())
genero_sel = st.sidebar.multiselect(
    "Gênero:",
    options=generos_unicos,
    default=generos_unicos
)

# Filtro: Faixa Etária
min_idade = int(df_raw['age'].min())
max_idade = int(df_raw['age'].max())
idade_sel = st.sidebar.slider(
    "Faixa Etária (Idade):",
    min_value=min_idade,
    max_value=max_idade,
    value=(min_idade, max_idade),
    step=1
)

# Filtro: Trabalho em Meio Período
trabalho_sel = st.sidebar.radio(
    "Trabalho em Meio Período:",
    options=["Todos", "Sim (Trabalha)", "Não"],
    index=0
)

# Filtro: Nível de Estresse (0 a 10)
estresse_sel = st.sidebar.slider(
    "Faixa de Estresse Mental (0 a 10):",
    min_value=0.0,
    max_value=10.0,
    value=(0.0, 10.0),
    step=0.5
)

# Filtro: Frequência às Aulas (%)
freq_sel = st.sidebar.slider(
    "Frequência às Aulas (%):",
    min_value=0,
    max_value=100,
    value=(0, 100),
    step=5
)

# Filtro: Nível de Renda Familiar
rendas_unicas = sorted(df_raw['family_income_level'].dropna().unique())
renda_sel = st.sidebar.multiselect(
    "Nível de Renda Familiar:",
    options=rendas_unicas,
    default=rendas_unicas
)

# Filtro: Método de Anotação
metodos_unicos = sorted(df_raw['note_taking_method'].dropna().unique())
metodos_sel = st.sidebar.multiselect(
    "Método de Anotação:",
    options=metodos_unicos,
    default=metodos_unicos
)

# 6. Aplicação da Filtragem e Amostragem
df_filtrado = df_raw.copy()

if cursos_sel:
    df_filtrado = df_filtrado[df_filtrado['major'].isin(cursos_sel)]

if anos_sel:
    df_filtrado = df_filtrado[df_filtrado['university_year'].isin(anos_sel)]

if genero_sel:
    df_filtrado = df_filtrado[df_filtrado['gender'].isin(genero_sel)]

df_filtrado = df_filtrado[
    (df_filtrado['age'] >= idade_sel[0]) &
    (df_filtrado['age'] <= idade_sel[1])
]

if trabalho_sel == "Sim (Trabalha)":
    df_filtrado = df_filtrado[df_filtrado['part_time_job'] == 'Yes']
elif trabalho_sel == "Não":
    df_filtrado = df_filtrado[df_filtrado['part_time_job'] == 'No']

df_filtrado = df_filtrado[
    (df_filtrado['mental_stress_level'] >= estresse_sel[0]) &
    (df_filtrado['mental_stress_level'] <= estresse_sel[1])
]

df_filtrado = df_filtrado[
    (df_filtrado['class_attendance_percent'] >= freq_sel[0]) &
    (df_filtrado['class_attendance_percent'] <= freq_sel[1])
]

if renda_sel:
    df_filtrado = df_filtrado[df_filtrado['family_income_level'].isin(renda_sel)]

if metodos_sel:
    df_filtrado = df_filtrado[df_filtrado['note_taking_method'].isin(metodos_sel)]

# Aplicação do limite de registros configurado pelo usuário
if len(df_filtrado) > qtd_analise:
    df_filtrado = df_filtrado.iloc[:qtd_analise]

# 7. Cabeçalho Principal
st.title("ASTRA Analytics — Painel de Hábitos & Desempenho Acadêmico")
st.caption(
    f"Exibindo dados de **{len(df_filtrado):,}** estudantes analisados "
    f"(de um total de **{total_registros:,}** disponíveis na base)."
)

if df_filtrado.empty:
    st.warning("⚠️ Nenhum registro encontrado para a combinação atual de filtros. Tente flexibilizar os parâmetros na barra lateral.")
    st.stop()

# 8. Linha de KPIs de Alto Impacto
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    gpa_medio = df_filtrado['GPA'].mean()
    st.metric("GPA Médio (0 - 4.0)", f"{gpa_medio:.2f}", delta=f"{gpa_medio - 3.0:+.2f} vs Alvo")

with col2:
    freq_media = df_filtrado['class_attendance_percent'].mean()
    st.metric("Frequência Média", f"{freq_media:.1f}%")

with col3:
    estudo_medio = df_filtrado['study_hours_per_day'].mean()
    st.metric("Estudo Diário", f"{estudo_medio:.1f} h/dia")

with col4:
    sono_medio = df_filtrado['sleep_hours'].mean()
    st.metric("Sono Diário", f"{sono_medio:.1f} h/noite")

with col5:
    estresse_medio = df_filtrado['mental_stress_level'].mean()
    st.metric("Nível de Estresse", f"{estresse_medio:.1f} / 10")

with col6:
    uso_ia = (df_filtrado['favorite_AI_tool'] != 'None').mean() * 100
    st.metric("Adoção de IA", f"{uso_ia:.1f}%")

st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)

# 9. SEÇÃO: MATRIZ DE CORRELAÇÕES & DESCOBERTAS CHAVE
st.subheader("🔬 Mapa de Correlações & Principais Fatores de Performance")
st.caption("Visão comparativa de como cada hábito influencia diretamente o GPA e o Desempenho no Exame:")

col_hm1, col_hm2 = st.columns([1.3, 1.0])

with col_hm1:
    # Heatmap das correlações dos hábitos com GPA e Notas
    cols_corr = [
        'GPA', 'final_exam_score', 'class_attendance_percent',
        'study_hours_per_day', 'sleep_hours', 'mental_stress_level',
        'social_media_hours', 'screen_time_hours', 'gaming_hours', 'AI_tool_usage_hours'
    ]
    nomes_amigaveis = {
        'GPA': 'GPA',
        'final_exam_score': 'Nota Exame',
        'class_attendance_percent': 'Frequência %',
        'study_hours_per_day': 'Estudo (h)',
        'sleep_hours': 'Sono (h)',
        'mental_stress_level': 'Estresse',
        'social_media_hours': 'Redes Sociais',
        'screen_time_hours': 'Tempo de Tela',
        'gaming_hours': 'Games (h)',
        'AI_tool_usage_hours': 'Uso IA (h)'
    }
    df_corr_sub = df_filtrado[cols_corr].rename(columns=nomes_amigaveis)
    corr_matrix = df_corr_sub.corr().round(2)
    
    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=[ASTRA_COLORS["accent_rose"], "#1E293B", ASTRA_COLORS["accent_cyan"]],
        zmin=-1, zmax=1,
        title="Matriz de Correlação Linear (Pearson)"
    )
    estilizar_grafico(fig_heatmap, "Matriz de Correlação dos Hábitos vs. Performance")
    fig_heatmap.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_heatmap, width="stretch")

with col_hm2:
    # Destaque dos Pilares de Alto e Baixo Impacto
    st.markdown("""
        <div style="background:#1E293B; border:1px solid #334155; border-radius:12px; padding:18px 20px; height:100%;">
            <h4 style="color:#38BDF8 !important; margin-top:0; font-size:1.05rem;">🏆 Os 2 Maiores Impulsionadores do GPA:</h4>
            <div style="margin-bottom:12px;">
                <span style="color:#34D399; font-weight:700; font-size:1.1rem;">+0.74</span> • <b>Frequência às Aulas:</b> É o preditor número 1 de sucesso. Alunos assíduos têm notas substancialmente superiores.<br>
                <span style="color:#34D399; font-weight:700; font-size:1.1rem;">+0.72</span> • <b>Horas de Estudo Diário:</b> Estudo consistente dia a dia gera o segundo maior impacto positivo.
            </div>
            <h4 style="color:#F43F5E !important; margin-top:16px; font-size:1.05rem;">⚠️ Os 3 Maiores Detratores do GPA:</h4>
            <div style="margin-bottom:12px;">
                <span style="color:#F43F5E; font-weight:700; font-size:1.1rem;">-0.27</span> • <b>Estresse Mental:</b> Alunos sob tensão elevada sofrem queda significativa de rendimento.<br>
                <span style="color:#F43F5E; font-weight:700; font-size:1.1rem;">-0.23</span> • <b>Redes Sociais:</b> Cada hora adicional reduz o GPA em ritmo contínuo.<br>
                <span style="color:#FBBF24; font-weight:700; font-size:1.1rem;">-0.18</span> • <b>Tempo de Tela Total:</b> Excesso de telas tem impacto negativo no foco e descanso.
            </div>
            <h4 style="color:#818CF8 !important; margin-top:16px; font-size:1.05rem;">🔍 Mitos Desmistificados pelos Dados:</h4>
            <div style="font-size:0.88rem; color:#CBD5E1;">
                • <b>Videogames (r = -0.002):</b> Quase nenhum impacto no GPA, ao contrário das Redes Sociais.<br>
                • <b>Uso de IA (r = +0.032):</b> Tempo de uso de IA não eleva o GPA sozinho sem estudo prévio.
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

# 10. SEÇÃO: GRÁFICOS DE DISPERSÃO COM LINHAS DE TENDÊNCIA
st.subheader("📈 Investigação Visual de Hábitos Críticos")
st.caption("Dispersão detalhada com linha de tendência estimada para os principais fatores:")

def criar_scatter_com_tendencia(df_plot, col_x, col_y, label_x, label_y, titulo, cor_pontos, cor_linha):
    fig = px.scatter(
        df_plot,
        x=col_x,
        y=col_y,
        opacity=0.6,
        color_discrete_sequence=[cor_pontos],
        labels={col_x: label_x, col_y: label_y}
    )
    if len(df_plot) > 1 and df_plot[col_x].nunique() > 1:
        x_vals = df_plot[col_x].values
        y_vals = df_plot[col_y].values
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        y_line = slope * x_line + intercept
        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            name=f'Tendência (Inclin: {slope:.3f})',
            line=dict(color=cor_linha, width=2.5, dash='dash')
        ))
    estilizar_grafico(fig, titulo)
    return fig

col_c1, col_c2, col_c3 = st.columns(3)

with col_c1:
    fig_freq_gpa = criar_scatter_com_tendencia(
        df_filtrado,
        "class_attendance_percent",
        "GPA",
        "Frequência às Aulas (%)",
        "GPA (0.0 a 4.0)",
        "Frequência às Aulas vs. GPA (r = +0.74)",
        "#34D399",
        "#FFFFFF"
    )
    st.plotly_chart(fig_freq_gpa, width="stretch")

with col_c2:
    fig_redes_gpa = criar_scatter_com_tendencia(
        df_filtrado,
        "social_media_hours",
        "GPA",
        "Horas em Redes Sociais / Dia",
        "GPA (0.0 a 4.0)",
        "Redes Sociais vs. GPA (r = -0.23)",
        "#FBBF24",
        "#FFFFFF"
    )
    st.plotly_chart(fig_redes_gpa, width="stretch")

with col_c3:
    fig_sono_gpa = criar_scatter_com_tendencia(
        df_filtrado,
        "sleep_hours",
        "GPA",
        "Horas de Sono por Noite",
        "GPA (0.0 a 4.0)",
        "Sono Diário vs. GPA (r = +0.23)",
        "#38BDF8",
        "#FFFFFF"
    )
    st.plotly_chart(fig_sono_gpa, width="stretch")

# Comparativo: Redes Sociais vs. Jogos (Gaming)
col_sub1, col_sub2 = st.columns(2)

with col_sub1:
    fig_tela_nota = criar_scatter_com_tendencia(
        df_filtrado,
        "screen_time_hours",
        "final_exam_score",
        "Tempo de Tela Diário (Horas)",
        "Nota no Exame Final (0 a 100)",
        "Tempo de Tela vs. Exame Final (r = -0.08)",
        "#F43F5E",
        "#FFFFFF"
    )
    st.plotly_chart(fig_tela_nota, width="stretch")

with col_sub2:
    fig_ia_gpa = criar_scatter_com_tendencia(
        df_filtrado,
        "AI_tool_usage_hours",
        "GPA",
        "Horas de Uso de IA / Dia",
        "GPA (0.0 a 4.0)",
        "Uso de IA vs. GPA (r = +0.03)",
        "#818CF8",
        "#FFFFFF"
    )
    st.plotly_chart(fig_ia_gpa, width="stretch")

st.markdown("---")

# 11. Linha de Gráficos de Hábitos
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.subheader("Relação: Horas de Estudo vs. GPA")
    fig_disp = px.scatter(
        df_filtrado,
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
        opacity=0.85
    )
    estilizar_grafico(fig_disp, "Impacto do Estudo Diário no Desempenho (GPA)")
    st.plotly_chart(fig_disp, width="stretch")

with col_g2:
    st.subheader("Faixas de Sono vs. Nível de Estresse")
    df_filtrado['faixa_sono'] = pd.cut(
        df_filtrado['sleep_hours'],
        bins=[0, 5, 7, 9, 24],
        labels=["< 5h (Crítico)", "5h-7h (Alerta)", "7h-9h (Adequado)", "> 9h (Alto)"]
    )
    df_sono = df_filtrado.dropna(subset=['faixa_sono'])
    fig_box = px.box(
        df_sono,
        x="faixa_sono",
        y="mental_stress_level",
        color="faixa_sono",
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
    st.plotly_chart(fig_box, width="stretch")

# 12. Linha de Gráficos Categóricos
col_g3, col_g4 = st.columns(2)

with col_g3:
    st.subheader("Ferramentas de IA mais Usadas")
    df_ia = df_filtrado[df_filtrado['favorite_AI_tool'] != 'None']
    fig_ia = px.histogram(
        df_ia,
        x="favorite_AI_tool",
        color="favorite_AI_tool",
        color_discrete_sequence=CATEGORICAL_PALETTE,
        labels={"favorite_AI_tool": "Ferramenta de IA", "count": "Quantidade de Estudantes"}
    )
    estilizar_grafico(fig_ia, "Distribuição das Ferramentas de IA Preferidas")
    fig_ia.update_layout(showlegend=False)
    st.plotly_chart(fig_ia, width="stretch")

with col_g4:
    st.subheader("Média no Exame Final por Método de Estudo")
    df_metodo = df_filtrado.groupby("note_taking_method", as_index=False)["final_exam_score"].mean().round(1)
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
    st.plotly_chart(fig_metodo, width="stretch")

# 13. Tabela de Amostra dos Dados
with st.expander(f"🔍 Visualizar Amostra dos Dados Analisados (Primeiras 100 de {len(df_filtrado):,} linhas)"):
    st.dataframe(df_filtrado.head(100), width="stretch")
