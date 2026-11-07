import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

try:
    from data.load import get_Dataset
except ImportError:
    try:
        from ASTRA_Dashboard.data.load import get_Dataset
    except ImportError:
        from ..data.load import get_Dataset


# -------------------------------------------------------------------------
# DEFINIÇÃO DO ESCOPO DE VARIÁVEIS COMPORTAMENTAIS E DE DESEMPENHO
# -------------------------------------------------------------------------
HABIT_VARIABLES = [
    "study_hours_per_day",
    "class_attendance_percent",
    "sleep_hours",
    "screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "exercise_hours_per_week",
    "mental_stress_level",
    "AI_tool_usage_hours",
    "exam_preparation_days",
    "coffee_consumption_per_day",
    "extracurricular_hours_per_week"
]

PERFORMANCE_VARIABLES = [
    "GPA",
    "final_exam_score",
    "assignment_score"
]

HABIT_RELATIONSHIPS = [
    ("sleep_hours", "mental_stress_level"),
    ("exercise_hours_per_week", "mental_stress_level"),
    ("screen_time_hours", "social_media_hours")
]


# -------------------------------------------------------------------------
# FUNÇÕES AUXILIARES DE CLASSIFICAÇÃO DE MAGNITUDE ESTATÍSTICA
# -------------------------------------------------------------------------
def _classificar_intensidade(correlacao):
    """
    Classifica qualitativamente a magnitude absoluta do coeficiente de correlação
    linear segundo faixas convencionais de interpretação estatística.
    """
    intensidade = abs(correlacao)
    if intensidade < 0.2:
        return "muito fraca"
    if intensidade < 0.4:
        return "fraca"
    if intensidade < 0.6:
        return "moderada"
    if intensidade < 0.8:
        return "forte"
    return "muito forte"


# -------------------------------------------------------------------------
# MOTOR DE CORRELAÇÃO BIVARIADA (PEARSON, SPEARMAN E CORREÇÃO FDR)
# -------------------------------------------------------------------------
def calcular_correlacoes(df=None, dados=None, habitos=None, desempenhos=None):
    """
    Calcula as correlações bivariadas entre variáveis de hábito e indicadores de
    desempenho acadêmico, combinando:
    - Coeficiente de Pearson (r): mede a associação linear paramétrica.
    - Coeficiente de Spearman (rho): mede a associação monotônica baseada em postos,
      atuando como diagnóstico de robustez frente ao efeito teto do GPA e das provas.
    - Correção de Benjamini-Hochberg (FDR): controla a Taxa de Falsas Descobertas
      em testes múltiplos simultâneos na família hábito-desempenho.
    """
    if df is None and dados is not None:
        df = dados
    if df is None:
        df = get_Dataset()

    habitos = list(dict.fromkeys(HABIT_VARIABLES if habitos is None else habitos))
    desempenhos = list(dict.fromkeys(PERFORMANCE_VARIABLES if desempenhos is None else desempenhos))

    df_limpo = df.replace([np.inf, -np.inf], np.nan)
    habito_desempenho = []
    p_valores_lista = []
    indices_validos = []

    # ---------------------------------------------------------------------
    # ETAPA 1: CÁLCULO PAR A PAR (PEARSON r + SPEARMAN rho)
    # ---------------------------------------------------------------------
    # Avalia simultaneamente a relação linear (Pearson) e por postos (Spearman)
    # para diagnosticar distorções de não linearidade ou saturação (efeito teto).
    for h in habitos:
        for d in desempenhos:
            item = {
                "variavel": h,
                "desempenho": d,
                "correlacao": None,
                "spearman": None,
                "spearman_valor_p": None,
                "valor_p": None,
                "valor_p_ajustado_fdr": None,
                "pares_validos": 0,
                "sinal": "indisponível",
                "intensidade": "não calculável",
                "valida": False,
            }

            if h in df_limpo.columns and d in df_limpo.columns:
                sub = (
                    df_limpo[[h, d]]
                    .apply(pd.to_numeric, errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
                item["pares_validos"] = int(len(sub))

                if len(sub) >= 2:
                    x = sub[h].to_numpy(dtype=float)
                    y = sub[d].to_numpy(dtype=float)

                    # Verifica variância não nula em ambos os vetores antes da inferência
                    if not (np.all(x == x[0]) or np.all(y == y[0])):
                        r, p = stats.pearsonr(x, y)
                        rho, p_rho = stats.spearmanr(x, y)

                        if all(np.isfinite(val) for val in (r, p, rho)):
                            r_float = float(r)
                            p_float = float(p)
                            rho_float = float(rho)

                            item["correlacao"] = r_float
                            item["spearman"] = rho_float
                            item["spearman_valor_p"] = float(p_rho) if np.isfinite(p_rho) else None
                            item["valor_p"] = p_float
                            item["sinal"] = (
                                "positiva" if r_float > 0
                                else "negativa" if r_float < 0
                                else "nula"
                            )
                            item["intensidade"] = _classificar_intensidade(r_float)
                            item["valida"] = True

                            p_valores_lista.append(p_float)
                            indices_validos.append(len(habito_desempenho))

            habito_desempenho.append(item)

    # ---------------------------------------------------------------------
    # ETAPA 2: CONTROLE DA TAXA DE FALSAS DESCOBERTAS (BENJAMINI-HOCHBERG)
    # ---------------------------------------------------------------------
    # Ajusta os p-valores brutos pelo procedimento step-up de Benjamini-Hochberg (fdr_bh)
    # para mitigar a inflação do Erro Tipo I decorrente de múltiplas comparações.
    if p_valores_lista:
        _, p_adj, _, _ = multipletests(p_valores_lista, method="fdr_bh")
        for idx_item, valor_fdr in zip(indices_validos, p_adj):
            habito_desempenho[idx_item]["valor_p_ajustado_fdr"] = float(valor_fdr)

    return {
        "habito_desempenho": habito_desempenho,
        "combinacoes_esperadas": len(habitos) * len(desempenhos),
        "habitos_disponiveis": [c for c in habitos if c in df_limpo.columns],
        "desempenhos_disponiveis": [c for c in desempenhos if c in df_limpo.columns],
        "comparacoes_validas": len(indices_validos),
        "ajuste_valores_p": "Benjamini-Hochberg (FDR)",
    }


# -------------------------------------------------------------------------
# GERAÇÃO DA MATRIZ DE CORRELAÇÃO DE PEARSON PARA O HEATMAP
# -------------------------------------------------------------------------
def calcular_matriz_correlacoes(df=None, dados=None, habitos=None, desempenhos=None):
    """
    Constrói a matriz simétrica de correlação linear de Pearson entre variáveis de
    hábito e desempenho acadêmico, tratando valores infinitos e não numéricos.
    """
    if df is None and dados is not None:
        df = dados
    if df is None:
        df = get_Dataset()

    habitos = list(dict.fromkeys(HABIT_VARIABLES if habitos is None else habitos))
    desempenhos = list(dict.fromkeys(PERFORMANCE_VARIABLES if desempenhos is None else desempenhos))

    colunas = [c for c in habitos + desempenhos if c in df.columns]
    return (
        df[colunas]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .corr(method="pearson")
    )
