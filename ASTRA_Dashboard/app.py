import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    "accent_violet": "#A78BFA",
    "accent_teal": "#2DD4BF"
}

CATEGORICAL_PALETTE = [
    ASTRA_COLORS["accent_cyan"],
    ASTRA_COLORS["accent_indigo"],
    ASTRA_COLORS["accent_emerald"],
    ASTRA_COLORS["accent_amber"],
    ASTRA_COLORS["accent_rose"],
    ASTRA_COLORS["accent_violet"],
    ASTRA_COLORS["accent_teal"]
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

# ============================================================
# MOTOR PREDITIVO - ASTRA
# ============================================================

def gpa_para_nota(gpa):
    """
    Converte GPA da escala 0–4 para uma escala de 0–10.
    """
    return (gpa / 4) * 10

@st.cache_resource
def treinar_motor_preditivo(df):
    """
    Treina e compara modelos para previsão do GPA.
    """

    # Variáveis utilizadas para previsão
    features_numericas = [
        "study_hours_per_day",
        "sleep_hours",
        "mental_stress_level",
        "class_attendance_percent",
        "social_media_hours",
        "age"
    ]

    # Verifica quais colunas realmente existem
    features_numericas = [
        col for col in features_numericas
        if col in df.columns
    ]

    features = features_numericas

    if "GPA" not in df.columns:
        raise ValueError("A coluna GPA não foi encontrada no dataset.")

    if len(features) == 0:
        raise ValueError(
            "Nenhuma variável adequada para previsão foi encontrada."
        )

    # --------------------------------------------------------
    # Dados
    # --------------------------------------------------------

    dados = df[features + ["GPA"]].copy()

    # Remove registros sem GPA
    dados = dados.dropna(subset=["GPA"])

    X = dados[features]
    y = dados["GPA"]

    # --------------------------------------------------------
    # Separação treino / teste
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # --------------------------------------------------------
    # Pré-processamento
    # --------------------------------------------------------

    transformers = []

    if features_numericas:
        pipeline_numerica = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        transformers.append(
            ("numericas", pipeline_numerica, features_numericas)
        )

    
    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    # --------------------------------------------------------
    # Modelos
    # --------------------------------------------------------

    modelos = {
        "Regressão Linear": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
    }

    resultados = []
    modelos_treinados = {}

    # --------------------------------------------------------
    # Treinamento e avaliação
    # --------------------------------------------------------

    for nome, modelo in modelos.items():

        pipeline = Pipeline([
            ("preprocessamento", preprocessor),
            ("modelo", modelo)
        ])

        pipeline.fit(X_train, y_train)

        previsoes = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, previsoes)
        rmse = np.sqrt(mean_squared_error(y_test, previsoes))
        r2 = r2_score(y_test, previsoes)

        resultados.append({
            "Modelo": nome,
            "MAE": mae,
            "RMSE": rmse,
            "R²": r2
        })

        modelos_treinados[nome] = pipeline

    resultados_df = pd.DataFrame(resultados)

    # --------------------------------------------------------
    # Seleção do melhor modelo
    # --------------------------------------------------------
    # Maior R² = melhor capacidade explicativa
    # Em caso de empate, menor RMSE

    resultados_df = resultados_df.sort_values(
        by=["R²", "RMSE"],
        ascending=[False, True]
    ).reset_index(drop=True)

    melhor_modelo_nome = resultados_df.iloc[0]["Modelo"]
    melhor_modelo = modelos_treinados[melhor_modelo_nome]

    return {
        "modelo": melhor_modelo,
        "nome": melhor_modelo_nome,
        "modelos": modelos_treinados,
        "resultados": resultados_df,
        "features": features,
        "features_numericas": features_numericas,
        "X_test": X_test,
        "y_test": y_test
    }

try:
    df_raw = carregar_dados()
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
    df_gen = df_filtrado['gender'].value_counts().reset_index()
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
    st.plotly_chart(fig_pz_gen, use_container_width=True)

with col_pz2:
    df_renda = df_filtrado['family_income_level'].value_counts().reset_index()
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
    st.plotly_chart(fig_pz_renda, use_container_width=True)

with col_pz3:
    df_job = df_filtrado['part_time_job'].replace({'Yes': 'Trabalha', 'No': 'Não Trabalha'}).value_counts().reset_index()
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
    st.plotly_chart(fig_pz_job, use_container_width=True)

with col_pz4:
    df_net = df_filtrado['internet_quality'].value_counts().reset_index()
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
    
    fig_disp = px.scatter(
        df_disp_amostra,
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
    st.plotly_chart(fig_disp, use_container_width=True)

with col_g2:
    st.caption("Distribuição do Nível de Estresse por Faixas de Sono Diário:")
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
    st.plotly_chart(fig_box, use_container_width=True)

col_g3, col_g4 = st.columns(2)

with col_g3:
    st.caption("Distribuição do GPA Acadêmico entre os diferentes Cursos de Graduação:")
    fig_box_major = px.box(
        df_filtrado,
        x="major",
        y="GPA",
        color="major",
        color_discrete_sequence=CATEGORICAL_PALETTE,
        labels={"major": "Curso", "GPA": "GPA Acadêmico"}
    )
    estilizar_grafico(fig_box_major, "Distribuição do GPA por Curso")
    fig_box_major.update_layout(showlegend=False)
    st.plotly_chart(fig_box_major, use_container_width=True)

with col_g4:
    st.caption("Eficácia do Método de Anotação no Exame Final:")
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
    st.plotly_chart(fig_metodo, use_container_width=True)

col_g5, col_g6 = st.columns(2)

with col_g5:
    st.caption("Ferramentas de Inteligência Artificial mais utilizadas pelos estudantes:")
    df_ia = df_filtrado[df_filtrado['favorite_AI_tool'].fillna('None') != 'None']
    fig_ia = px.histogram(
        df_ia,
        x="favorite_AI_tool",
        color="favorite_AI_tool",
        color_discrete_sequence=CATEGORICAL_PALETTE,
        labels={"favorite_AI_tool": "Ferramenta de IA", "count": "Quantidade de Estudantes"}
    )
    estilizar_grafico(fig_ia, "Distribuição das Ferramentas de IA Preferidas")
    fig_ia.update_layout(showlegend=False)
    st.plotly_chart(fig_ia, use_container_width=True)

with col_g6:
    st.caption("Relação entre Consumo Diário de Café e Horas Médias de Sono:")
    df_cafe = df_filtrado.groupby("coffee_consumption_per_day", as_index=False)["sleep_hours"].mean().round(2)
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
    st.plotly_chart(fig_cafe, use_container_width=True)

st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)

# ============================================================
# 13. MOTOR PREDITIVO
# ============================================================

st.markdown("---")
st.subheader("🔮 Motor Preditivo de Desempenho Acadêmico")

st.caption(
    "O ASTRA utiliza Machine Learning para estimar o GPA a partir "
    "de hábitos e características acadêmicas do estudante."
)

if not SKLEARN_DISPONIVEL:

    st.warning(
        "⚠️ O pacote scikit-learn não está disponível. "
        "Execute: pip install scikit-learn"
    )

else:

    try:

        # ----------------------------------------------------
        # Treinamento
        # ----------------------------------------------------

        motor = treinar_motor_preditivo(df_raw)

        modelo_escolhido = motor["modelo"]
        nome_modelo = motor["nome"]
        resultados_modelos = motor["resultados"]

        # ----------------------------------------------------
        # Indicadores do motor
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)

        melhor_r2 = resultados_modelos.iloc[0]["R²"]
        melhor_mae = resultados_modelos.iloc[0]["MAE"]
        melhor_rmse = resultados_modelos.iloc[0]["RMSE"]

        with col1:
            st.metric(
                "Modelo selecionado",
                nome_modelo
            )

        with col2:
            st.metric(
                "R²",
                f"{melhor_r2:.3f}"
            )

        with col3:
            st.metric(
                "Erro médio (MAE)",
                f"{melhor_mae:.3f}"
            )

        st.markdown("### 📊 Comparação dos modelos")

        tabela_modelos = resultados_modelos.copy()

        tabela_modelos["MAE"] = tabela_modelos["MAE"].round(3)
        tabela_modelos["RMSE"] = tabela_modelos["RMSE"].round(3)
        tabela_modelos["R²"] = tabela_modelos["R²"].round(3)

        st.dataframe(
            tabela_modelos,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # Formulário de previsão
        # ----------------------------------------------------

        st.markdown("### 🎯 Simular desempenho de um estudante")

        st.caption(
            "Informe os hábitos do estudante para gerar uma estimativa."
        )

        col_a, col_b, col_c = st.columns(3)

        with col_a:

            estudo = st.number_input(
                "Horas de estudo por dia",
                min_value=0.0,
                max_value=24.0,
                value=4.0,
                step=0.5
            )

            sono = st.number_input(
                "Horas de sono por noite",
                min_value=0.0,
                max_value=24.0,
                value=7.0,
                step=0.5
            )

        with col_b:

            estresse = st.slider(
                "Nível de estresse",
                min_value=0.0,
                max_value=10.0,
                value=5.0,
                step=0.5
            )

            frequencia = st.slider(
                "Frequência às aulas (%)",
                min_value=0.0,
                max_value=100.0,
                value=80.0,
                step=1.0
            )

        with col_c:

            redes = st.number_input(
                "Horas de redes sociais por dia",
                min_value=0.0,
                max_value=24.0,
                value=2.0,
                step=0.5
            )

            idade = st.number_input(
                "Idade",
                min_value=10,
                max_value=100,
                value=20,
                step=1
            )


        # ----------------------------------------------------
        # Botão de previsão
        # ----------------------------------------------------

        if st.button(
            "🔮 Gerar previsão",
            use_container_width=True
        ):

            dados_estudante = {}

            if "study_hours_per_day" in motor["features"]:
                dados_estudante[
                    "study_hours_per_day"
                ] = estudo

            if "sleep_hours" in motor["features"]:
                dados_estudante[
                    "sleep_hours"
                ] = sono

            if "mental_stress_level" in motor["features"]:
                dados_estudante[
                    "mental_stress_level"
                ] = estresse

            if "class_attendance_percent" in motor["features"]:
                dados_estudante[
                    "class_attendance_percent"
                ] = frequencia

            if "social_media_hours" in motor["features"]:
                dados_estudante[
                    "social_media_hours"
                ] = redes

            if "age" in motor["features"]:
                dados_estudante[
                    "age"
                ] = idade

            entrada = pd.DataFrame(
                [dados_estudante],
                columns=motor["features"]
            )

            # ------------------------------------------------
            # Previsão
            # ------------------------------------------------

            gpa_previsto = modelo_escolhido.predict(entrada)[0]

            # Limita ao intervalo esperado do GPA
            gpa_previsto = max(0, min(4, gpa_previsto))

            # Converte para escala de 0 a 10
            nota_prevista = gpa_para_nota(gpa_previsto)
            # ------------------------------------------------
            # Classificação
            # ------------------------------------------------

            if gpa_previsto >= 3.5:

                classificacao = "Alto desempenho"
                icone = "🟢"

            elif gpa_previsto >= 2.5:

                classificacao = "Desempenho intermediário"
                icone = "🟡"

            else:

                classificacao = "Atenção ao desempenho"
                icone = "🔴"

            st.markdown("---")

            resultado_col1, resultado_col2 = st.columns(2)

            with resultado_col1:

                st.metric(
                    "GPA estimado",
                    f"{gpa_previsto:.2f}"
                )
                st.caption(
                    f"Equivalente a uma nota de {nota_prevista:.2f} na escala de 0–10."
                )

            with resultado_col2:

                st.metric(
                    "Classificação",
                    f"{icone} {classificacao}"
                )

            st.info(
                "Esta é uma estimativa estatística baseada nos padrões "
                "encontrados no dataset. Ela não representa uma garantia "
                "sobre o desempenho futuro de um estudante."
            )

        # ------------------------------------------------
        # Importância das variáveis
        # ------------------------------------------------

        st.markdown("### 🧩 Importância das variáveis")

        modelo_final = modelo_escolhido.named_steps["modelo"]

        preprocessor_final = modelo_escolhido.named_steps[
            "preprocessamento"
        ]

        if hasattr(modelo_final, "feature_importances_"):

            importancias = modelo_final.feature_importances_

            nomes_features = (
                preprocessor_final
                .get_feature_names_out()
            )

            nomes_amigaveis = {
                "numericas__study_hours_per_day": "Horas de estudo por dia",
                "numericas__sleep_hours": "Horas de sono por noite",
                "numericas__mental_stress_level": "Nível de estresse",
                "numericas__class_attendance_percent": "Frequência às aulas",
                "numericas__social_media_hours": "Horas em redes sociais",
                "numericas__age": "Idade"
            }

            df_importancia = pd.DataFrame({
                "Variável": [
                    nomes_amigaveis.get(nome, nome)
                    for nome in nomes_features
                ],
                "Importância": importancias
            })

            # Converte a importância para percentual
            df_importancia["Importância (%)"] = (
                df_importancia["Importância"] * 100
            )

            df_importancia = (
                df_importancia
                .sort_values(
                    "Importância (%)",
                    ascending=False
                )
                .head(10)
            )

            fig_importancia = px.bar(
                df_importancia,
                x="Importância (%)",
                y="Variável",
                orientation="h",
                title="Importância das variáveis no modelo"
            )

            fig_importancia.update_layout(
                yaxis={"categoryorder": "total ascending"}
            )

            estilizar_grafico(
                fig_importancia,
                "Importância das Variáveis no Modelo"
            )

            st.plotly_chart(
                fig_importancia,
                use_container_width=True
            )

        else:

            st.info(
                "A importância das variáveis é exibida para modelos "
                "baseados em árvores, como Random Forest e Gradient Boosting."
            )

    except Exception as erro:

        st.error(
            f"❌ Não foi possível executar o motor preditivo: {erro}"
        )

# 14. SEÇÃO: CLUSTERIZAÇÃO K-MEANS & PERFIS AUTOMÁTICOS
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
        
        # Normalização com StandardScaler e treinamento do KMeans
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_filtrado[features_kmeans].fillna(df_filtrado[features_kmeans].mean()))
        
        kmeans_model = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        df_filtrado['cluster_id'] = kmeans_model.fit_predict(X_scaled)
        
        # Rótulos automáticos ordenados por desempenho acadêmico (GPA)
        medias_cluster = df_filtrado.groupby('cluster_id')[features_kmeans].mean()
        ordem_gpa = medias_cluster['GPA'].sort_values(ascending=False).index.tolist()
        
        cores_cluster_map = [
            ASTRA_COLORS["accent_emerald"],  # Top 1
            ASTRA_COLORS["accent_cyan"],     # Top 2
            ASTRA_COLORS["accent_indigo"],   # Top 3
            ASTRA_COLORS["accent_amber"],    # Top 4
            ASTRA_COLORS["accent_rose"]      # Top 5
        ]
        
        rotulos_sugeridos = [
            "🎓 Perfil 1: Alto Desempenho & Foco",
            "⚖️ Perfil 2: Rendimento Equilibrado",
            "⚡ Perfil 3: Tensão & Risco de Burnout",
            "⚠️ Perfil 4: Desengajamento / Baixa Assiduidade",
            "🔄 Perfil 5: Hábitos Mistos"
        ]
        
        mapa_nomes = {}
        for rank, cid in enumerate(ordem_gpa):
            mapa_nomes[cid] = rotulos_sugeridos[rank] if rank < len(rotulos_sugeridos) else f"Perfil {rank+1}"
            
        df_filtrado['perfil_cluster'] = df_filtrado['cluster_id'].map(mapa_nomes)
        
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
            fig_km_scatter = px.scatter(
                df_cluster_sample,
                x="class_attendance_percent",
                y="GPA",
                color="perfil_cluster",
                color_discrete_sequence=[cores_cluster_map[ordem_gpa.index(cid) % len(cores_cluster_map)] for cid in sorted(df_filtrado['cluster_id'].unique())],
                labels={"class_attendance_percent": "Frequência às Aulas (%)", "GPA": "GPA Acadêmico (0 a 4.0)", "perfil_cluster": "Perfil K-Means"},
                hover_data=["study_hours_per_day", "mental_stress_level", "sleep_hours"]
            )
            fig_km_scatter.update_traces(marker=dict(size=6, opacity=0.75))
            estilizar_grafico(fig_km_scatter, "Dispersão dos Perfis: Frequência vs. GPA")
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
            
            fig_km_bars = px.bar(
                df_plot_melt,
                x="Hábito",
                y="Média",
                color="perfil_cluster",
                barmode="group",
                color_discrete_sequence=CATEGORICAL_PALETTE,
                labels={"Média": "Média do Hábito", "perfil_cluster": "Perfil"}
            )
            estilizar_grafico(fig_km_bars, "Comparativo dos Hábitos Médios entre os Perfis")
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

# 15. Tabela de Amostra dos Dados
with st.expander(f"🔍 Visualizar Amostra dos Dados Analisados (Primeiras 100 de {len(df_filtrado):,} linhas)"):
    st.dataframe(df_filtrado.head(100), use_container_width=True)
