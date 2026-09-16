import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from data.load import get_Dataset
from analysis.clusters import executar_kmeans
from visualization.style import estilizar_grafico, ASTRA_COLORS, CATEGORICAL_PALETTE
from visualization.graphs import criar_scatter_com_tendencia, criar_heatmap_correlacao, criar_grafico_genero, criar_grafico_renda, criar_grafico_trabalho, criar_grafico_internet,criar_grafico_study_gpa, criar_grafico_sono_estresse, criar_grafico_gpa_major, criar_grafico_metodos_anotacao, criar_grafico_ia_tools, criar_grafico_cafe_sono, criar_grafico_clusters, criar_grafico_perfil_clusters

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_DISPONIVEL = True
except ImportError:
    SKLEARN_DISPONIVEL = False

# 1. Configuração da Página
st.set_page_config(
    page_title="ASTRA - Inteligência Acadêmica",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# 4. Carregamento dos Dados com Cache
# @st.cache_data
try:
    df_raw = get_Dataset()
except Exception as erro:
    st.error(f"Erro ao carregar o arquivo CSV: {erro}")
    st.stop()

# 5. Barra Lateral: Filtros Detalhados
st.sidebar.title("🎯 Filtros ASTRA")
st.sidebar.caption("Personalize sua análise:")

# Filtro: Quantidade de Dados a Analisar
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

# Filtro: Cursos
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

# Configurações de Machine Learning (K-Means) na Barra Lateral
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Machine Learning (K-Means)")
ativar_kmeans = st.sidebar.checkbox(
    "Ativar Clusterização K-Means",
    value=True,
    help="Agrupa os estudantes em perfis de comportamento automaticamente com Scikit-Learn."
)
if ativar_kmeans:
    k_clusters = st.sidebar.slider(
        "Quantidade de Perfis (K):",
        min_value=2,
        max_value=5,
        value=3,
        step=1,
        help="Número de grupos (perfis de alunos) a serem descobertos pelo K-Means."
    )
else:
    k_clusters = 3

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
# SEÇÃO: VISÃO GERAL, OBJETIVOS E HIPÓTESES
with st.expander("📌 Sobre o Projeto ASTRA, Objetivos e Questões Norteadoras", expanded=False):
    st.markdown("### 💡 Objetivo Principal")
    st.write(
        "Desenvolver uma plataforma analítica e preditiva que centraliza a rotina "
        "acadêmica universitária, correlacionando hábitos de estudo, sono, bem-estar e "
        "uso de ferramentas tecnológicas para antecipar o desempenho acadêmico (GPA) "
        "e mitigar o risco de sobrecarga estudantil."
    )

    st.markdown("### 🎯 Objetivos Secundários")
    st.markdown("""
    * **Investigação Exploratória:** Realizar a limpeza, transformação e análise exploratória de dados multivariados sobre hábitos de 10.000 estudantes universitários via Streamlit.
    * **Identificação de Fatores Críticos:** Mapear correlações estatísticas entre indicadores comportamentais (horas de estudo, qualidade de sono, estresse, frequência e redes sociais) e o rendimento acadêmico final.
    * **Modelagem Preditiva:** Estruturar modelos de regressão capazes de estimar a pontuação acadêmica com base em registros diários de baixo atrito.
    * **Interface de Suporte à Decisão:** Disponibilizar um painel interativo intuitivo que transforme métricas estatísticas em recomendações práticas para a rotina do estudante.
    """)

    st.markdown("---")
    st.markdown("### ❓ Perguntas Norteadoras da Pesquisa")
    
    st.markdown("**1. De que maneira o equilíbrio entre horas de sono e nível de estresse modula o impacto das horas de estudo no desempenho acadêmico (GPA)?**")
    st.info(
        "**Justificativa e Importância:** Na cultura universitária predomina a crença de que quanto mais horas de estudo diárias, "
        "melhor o rendimento. No entanto, dados empíricos indicam que noites de sono reduzidas e níveis elevados de estresse geram "
        "saturação cognitiva, diminuindo a retenção de conteúdo. Investigar essa relação permite identificar o 'ponto de retorno decrescente' "
        "do esforço e validar a tese central do ASTRA: orientar o aluno a dormir melhor e gerenciar o estresse pode produzir um GPA superior "
        "ao de simplesmente aumentar as horas de estudo sem descanso."
    )

    st.markdown("**2. Qual é a correlação do uso de ferramentas de Inteligência Artificial no desempenho dos estudantes quando contrastado com o tempo gasto em redes sociais e a frequência às aulas?**")
    st.info(
        "**Justificativa e Importância:** A rápida adoção de ferramentas de IA generativa no meio acadêmico cria um cenário ainda pouco mapeado: "
        "a IA funciona como potencializadora de aprendizado ou como atalho superficial? Ao cruzar o tempo de uso de IA com a presença em sala "
        "de aula e o tempo em redes sociais, o projeto extrai padrões comportamentais do estudante moderno, permitindo ao ASTRA calibrar se "
        "o uso da tecnologia está associado a ganho real de produtividade ou a dispersão."
    )

# SEÇÃO: DICIONÁRIO DE DADOS
with st.expander("📖 Dicionário de Dados do Dataset", expanded=False):
    dict_path = os.path.join(os.path.dirname(__file__), "..", "Data", "dicionario_de_dados.md")
    if os.path.exists(dict_path):
        with open(dict_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.write("Consulte o arquivo Data/dicionario_de_dados.md no repositório.")

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
    uso_ia = (df_filtrado['favorite_AI_tool'].fillna('None') != 'None').mean() * 100
    st.metric("Adoção de IA", f"{uso_ia:.1f}%")

st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)

# 9. SEÇÃO: MATRIZ DE CORRELAÇÕES & DESCOBERTAS CHAVE
st.subheader("🔬 Mapa de Correlações & Principais Fatores de Performance")
st.caption("Visão comparativa de como cada hábito influencia diretamente o GPA e o Desempenho no Exame:")

col_hm1, col_hm2 = st.columns([1.3, 1.0])

with col_hm1:
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
    
    fig_heatmap = criar_heatmap_correlacao(corr_matrix=corr_matrix, titulo="Matriz de Correlação dos Hábitos vs. Performance")
    st.plotly_chart(fig_heatmap, use_container_width=True)

with col_hm2:
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

st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)

# 10. SEÇÃO: GRÁFICOS DE DISPERSÃO COM LINHAS DE TENDÊNCIA (SCATTER PLOTS)
st.subheader("📈 Investigação Visual de Hábitos Críticos (Scatter Plots)")
st.caption("Dispersão detalhada de cada aluno com linha de tendência estimada para os fatores essenciais:")


# Linha 1 de Scatters (3 colunas)
col_c1, col_c2, col_c3 = st.columns(3)

with col_c1:
    fig_freq_gpa = criar_scatter_com_tendencia(
        df_filtrado,
        "class_attendance_percent",
        "GPA",
        "Frequência às Aulas (%)",
        "GPA (0.0 a 4.0)",
        "Frequência às Aulas vs. GPA (r = +0.74)",
        ASTRA_COLORS["accent_emerald"],
        "#FFFFFF"
    )
    st.plotly_chart(fig_freq_gpa, use_container_width=True)

with col_c2:
    fig_redes_gpa = criar_scatter_com_tendencia(
        df_filtrado,
        "social_media_hours",
        "GPA",
        "Horas em Redes Sociais / Dia",
        "GPA (0.0 a 4.0)",
        "Redes Sociais vs. GPA (r = -0.23)",
        ASTRA_COLORS["accent_amber"],
        "#FFFFFF"
    )
    st.plotly_chart(fig_redes_gpa, use_container_width=True)

with col_c3:
    fig_sono_gpa = criar_scatter_com_tendencia(
        df_filtrado,
        "sleep_hours",
        "GPA",
        "Horas de Sono por Noite",
        "GPA (0.0 a 4.0)",
        "Sono Diário vs. GPA (r = +0.23)",
        ASTRA_COLORS["accent_cyan"],
        "#FFFFFF"
    )
    st.plotly_chart(fig_sono_gpa, use_container_width=True)

# Linha 2 de Scatters (3 colunas)
col_c4, col_c5, col_c6 = st.columns(3)

with col_c4:
    fig_tela_nota = criar_scatter_com_tendencia(
        df_filtrado,
        "screen_time_hours",
        "final_exam_score",
        "Tempo de Tela Diário (Horas)",
        "Nota no Exame Final (0 a 100)",
        "Tempo de Tela vs. Exame Final (r = -0.08)",
        ASTRA_COLORS["accent_rose"],
        "#FFFFFF"
    )
    st.plotly_chart(fig_tela_nota, use_container_width=True)

with col_c5:
    fig_ia_gpa = criar_scatter_com_tendencia(
        df_filtrado,
        "AI_tool_usage_hours",
        "GPA",
        "Horas de Uso de IA / Dia",
        "GPA (0.0 a 4.0)",
        "Uso de IA vs. GPA (r = +0.03)",
        ASTRA_COLORS["accent_indigo"],
        "#FFFFFF"
    )
    st.plotly_chart(fig_ia_gpa, use_container_width=True)

with col_c6:
    fig_exercicio_estresse = criar_scatter_com_tendencia(
        df_filtrado,
        "exercise_hours_per_week",
        "mental_stress_level",
        "Exercício Físico (Horas/Semana)",
        "Nível de Estresse (0 a 10)",
        "Exercício Físico vs. Estresse (r = -0.15)",
        ASTRA_COLORS["accent_teal"],
        "#FFFFFF"
    )
    st.plotly_chart(fig_exercicio_estresse, use_container_width=True)

st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)

# 11. SEÇÃO: CONTROLE DE VIESES & DISTRIBUIÇÃO DEMOGRÁFICA (GRÁFICOS DE PIZZA / DONUT)
st.subheader("🥧 Análise Demográfica & Controle de Vieses")
st.caption("Gráficos de proporção para verificar a representatividade da amostra e evitar conclusões enviesadas:")

col_pz1, col_pz2, col_pz3, col_pz4 = st.columns(4)

with col_pz1:
    fig_pz_gen = criar_grafico_genero(df_filtrado)
    st.plotly_chart(fig_pz_gen, use_container_width=True)

with col_pz2:
    fig_pz_renda = criar_grafico_renda(df_filtrado)
    st.plotly_chart(fig_pz_renda, use_container_width=True)

with col_pz3:
    fig_pz_job = criar_grafico_trabalho(df_filtrado)
    st.plotly_chart(fig_pz_job, use_container_width=True)

with col_pz4:
    fig_pz_net = criar_grafico_internet(df_filtrado)
    st.plotly_chart(fig_pz_net, use_container_width=True)

st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)

# 12. SEÇÃO: HÁBITOS, METODOLOGIAS & DISTRIBUIÇÃO ACADÊMICA
st.subheader("📊 Hábitos de Estudo, Métodos & Rendimento por Curso")

col_g1, col_g2 = st.columns(2)

with col_g1:
    st.caption("Relação: Horas de Estudo vs. GPA com dispersão contínua por Nível de Estresse:")
    df_disp = df_filtrado.copy()
    df_disp['favorite_AI_tool'] = df_disp['favorite_AI_tool'].fillna('Nenhuma')
    df_disp_amostra = df_disp.sample(min(len(df_disp), 2500), random_state=42) if len(df_disp) > 2500 else df_disp
    
    fig_disp = criar_grafico_study_gpa(df_disp_amostra)
    st.plotly_chart(fig_disp, use_container_width=True)

with col_g2:
    st.caption("Distribuição do Nível de Estresse por Faixas de Sono Diário:")
    df_filtrado['faixa_sono'] = pd.cut(
        df_filtrado['sleep_hours'],
        bins=[0, 5, 7, 9, 24],
        labels=["< 5h (Crítico)", "5h-7h (Alerta)", "7h-9h (Adequado)", "> 9h (Alto)"]
    )
    df_sono = df_filtrado.dropna(subset=['faixa_sono'])

    fig_box = criar_grafico_sono_estresse(df_sono)
    st.plotly_chart(fig_box, use_container_width=True)

col_g3, col_g4 = st.columns(2)

with col_g3:
    st.caption("Distribuição do GPA Acadêmico entre os diferentes Cursos de Graduação:")

    fig_box_major = criar_grafico_gpa_major(df_filtrado)
    st.plotly_chart(fig_box_major, use_container_width=True)

with col_g4:
    st.caption("Eficácia do Método de Anotação no Exame Final:")

    fig_metodo = criar_grafico_metodos_anotacao(df_filtrado)
    st.plotly_chart(fig_metodo, use_container_width=True)

col_g5, col_g6 = st.columns(2)

with col_g5:
    st.caption("Ferramentas de Inteligência Artificial mais utilizadas pelos estudantes:")

    fig_ia = criar_grafico_ia_tools(df_filtrado)
    st.plotly_chart(fig_ia, use_container_width=True)

with col_g6:
    st.caption("Relação entre Consumo Diário de Café e Horas Médias de Sono:")
    fig_cafe = criar_grafico_cafe_sono(df_filtrado)
    st.plotly_chart(fig_cafe, use_container_width=True)

st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)

# 13. SEÇÃO: CLUSTERIZAÇÃO K-MEANS & PERFIS AUTOMÁTICOS
if ativar_kmeans:
    st.markdown("---")
    st.subheader("🤖 Segmentação Inteligente de Perfis (K-Means Clustering)")
    
    if not SKLEARN_DISPONIVEL:
        st.warning("⚠️ O pacote `scikit-learn` está sendo instalado ou não foi encontrado. Para habilitar esta seção, execute: `pip install scikit-learn`")
    elif len(df_filtrado) < k_clusters:
        st.info("ℹ️ Dados insuficientes para o número de clusters selecionado.")
    else:
        st.caption(
            "O algoritmo de **Machine Learning (Scikit-Learn)** analisa simultaneamente hábitos de estudo, sono, estresse, presença, redes sociais e rendimento "
            f"para agrupar os **{len(df_filtrado):,}** estudantes em **{k_clusters} perfis comportamentais**:"
        )
        features_kmeans = [
                'study_hours_per_day', 'sleep_hours', 'mental_stress_level',
                'class_attendance_percent', 'social_media_hours', 'GPA'
            ]
        df_filtrado, medias_cluster, ordem_gpa, mapa_nomes = executar_kmeans(df_filtrado, k_clusters)

        cores_cluster_map = [
            ASTRA_COLORS["accent_emerald"],  # Top 1
            ASTRA_COLORS["accent_cyan"],     # Top 2
            ASTRA_COLORS["accent_indigo"],   # Top 3
            ASTRA_COLORS["accent_amber"],    # Top 4
            ASTRA_COLORS["accent_rose"]      # Top 5
        ]
        
        # Cards com Métricas dos Perfis
        col_k_cards = st.columns(k_clusters)
        for i, cid in enumerate(ordem_gpa):
            qtd_c = (df_filtrado['cluster_id'] == cid).sum()
            pct_c = (qtd_c / len(df_filtrado)) * 100
            gpa_c = medias_cluster.loc[cid, 'GPA']
            estresse_c = medias_cluster.loc[cid, 'mental_stress_level']
            freq_c = medias_cluster.loc[cid, 'class_attendance_percent']
            with col_k_cards[i]:
                cor_card = cores_cluster_map[i % len(cores_cluster_map)]
                st.markdown(f"""
                    <div style="background:#1E293B; border:1px solid #334155; border-top:3px solid {cor_card}; border-radius:10px; padding:12px 14px; margin-bottom:15px;">
                        <div style="font-size:0.83rem; font-weight:700; color:#F8FAFC;">{mapa_nomes[cid]}</div>
                        <div style="font-size:1.45rem; font-weight:800; color:{cor_card}; margin:3px 0;">{qtd_c:,} <span style="font-size:0.82rem; color:#94A3B8;">({pct_c:.1f}%)</span></div>
                        <div style="font-size:0.77rem; color:#CBD5E1;">GPA: <b>{gpa_c:.2f}</b> • Presença: <b>{freq_c:.0f}%</b> • Estresse: <b>{estresse_c:.1f}</b></div>
                    </div>
                """, unsafe_allow_html=True)
        
        # Gráficos da Clusterização
        col_km1, col_km2 = st.columns(2)
        
        with col_km1:
            df_cluster_sample = df_filtrado.sample(min(len(df_filtrado), 2500), random_state=42) if len(df_filtrado) > 2500 else df_filtrado
            fig_km_scatter = criar_grafico_clusters(df_filtrado, cores_cluster_map, ordem_gpa)
            st.plotly_chart(fig_km_scatter, use_container_width=True)
            
        with col_km2:
            nomes_labels = {
                'study_hours_per_day': 'Estudo (h/dia)',
                'sleep_hours': 'Sono (h/noite)',
                'mental_stress_level': 'Estresse (0-10)',
                'social_media_hours': 'Redes Sociais (h)'
            }
            df_plot_bars = df_filtrado.groupby('perfil_cluster')[['study_hours_per_day', 'sleep_hours', 'mental_stress_level', 'social_media_hours']].mean().reset_index()
            df_plot_melt = df_plot_bars.melt(id_vars=['perfil_cluster'], var_name='Hábito', value_name='Média')
            df_plot_melt['Hábito'] = df_plot_melt['Hábito'].map(nomes_labels)
            
            fig_km_bars = criar_grafico_perfil_clusters(df_plot_melt)
            st.plotly_chart(fig_km_bars, use_container_width=True)
        
        # Tabela Resumo dos Perfis K-Means
        with st.expander("📊 Ver Tabela Comparativa Detalhada dos Perfis K-Means"):
            df_resumo_tabela = df_filtrado.groupby('perfil_cluster')[features_kmeans + ['final_exam_score']].agg({
                'GPA': 'mean',
                'final_exam_score': 'mean',
                'class_attendance_percent': 'mean',
                'study_hours_per_day': 'mean',
                'sleep_hours': 'mean',
                'mental_stress_level': 'mean',
                'social_media_hours': 'mean'
            }).round(2).rename(columns={
                'GPA': 'GPA Médio',
                'final_exam_score': 'Nota Exame',
                'class_attendance_percent': 'Frequência (%)',
                'study_hours_per_day': 'Estudo (h)',
                'sleep_hours': 'Sono (h)',
                'mental_stress_level': 'Estresse (0-10)',
                'social_media_hours': 'Redes Sociais (h)'
            })
            st.dataframe(df_resumo_tabela, use_container_width=True)

st.markdown("---")

# 14. Tabela de Amostra dos Dados
with st.expander(f"🔍 Visualizar Amostra dos Dados Analisados (Primeiras 100 de {len(df_filtrado):,} linhas)"):
    st.dataframe(df_filtrado.head(100), use_container_width=True)
