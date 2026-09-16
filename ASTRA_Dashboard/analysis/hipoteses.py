import pandas as pd

try:
    from scipy import stats
except ImportError:
    stats = None

try:
    from data.load import get_Dataset
except ImportError:
    from ..data.load import get_Dataset


NIVEL_SIGNIFICANCIA = 0.05
GPA_COLUMN = 'GPA'
NOTE_TAKING_METHOD_COLUMN = 'note_taking_method'
MAJOR_COLUMN = 'major'


def _resultado_base(agrupamento, grupos=None, tamanhos=None, quantidade=0):
    return {
        'variavel_agrupamento': agrupamento,
        'variavel_analisada': GPA_COLUMN,
        'teste_utilizado': None,
        'estatistica': None,
        'p_valor': None,
        'quantidade_grupos': len(grupos) if grupos is not None else 0,
        'grupos_utilizados': grupos or [],
        'tamanhos_amostras': tamanhos or {},
        'tamanho_amostra': quantidade,
        'nivel_significancia': NIVEL_SIGNIFICANCIA,
        'conclusao_estatistica': None,
        'diagnosticos': {}
    }


def _conclusao(p_valor):
    if p_valor < NIVEL_SIGNIFICANCIA:
        return 'Rejeita-se a hipótese nula: há diferença estatisticamente significativa entre pelo menos dois grupos.'
    return 'Não se rejeita a hipótese nula: não foi identificada diferença estatisticamente significativa entre os grupos.'


def _dados_para_teste(df, agrupamento):
    if df is None:
        df = get_Dataset()

    resultado = _resultado_base(agrupamento)
    colunas_necessarias = [agrupamento, GPA_COLUMN]
    if any(coluna not in df.columns for coluna in colunas_necessarias):
        resultado['conclusao_estatistica'] = 'Dados insuficientes: uma ou mais colunas necessárias não estão disponíveis.'
        return resultado, None

    dados = df[colunas_necessarias].copy()
    dados[GPA_COLUMN] = pd.to_numeric(dados[GPA_COLUMN], errors='coerce')
    dados = dados.dropna(subset=colunas_necessarias)
    dados = dados[dados[GPA_COLUMN].apply(lambda valor: pd.notna(valor) and pd.api.types.is_number(valor))]
    if dados.empty:
        resultado['conclusao_estatistica'] = 'Dados insuficientes: não há registros válidos para a análise.'
        return resultado, None

    if pd.api.types.is_object_dtype(dados[agrupamento]) or pd.api.types.is_string_dtype(dados[agrupamento]):
        dados[agrupamento] = dados[agrupamento].astype(str).str.strip()
        dados = dados[dados[agrupamento] != '']

    grupos = list(dados[agrupamento].drop_duplicates())
    amostras = {
        grupo: dados.loc[dados[agrupamento] == grupo, GPA_COLUMN].tolist()
        for grupo in grupos
    }
    tamanhos = {grupo: len(amostra) for grupo, amostra in amostras.items()}
    resultado.update({
        'quantidade_grupos': len(grupos),
        'grupos_utilizados': grupos,
        'tamanhos_amostras': tamanhos,
        'tamanho_amostra': int(len(dados))
    })
    return resultado, amostras


def _executar_teste(df, agrupamento):
    resultado, amostras = _dados_para_teste(df, agrupamento)
    if amostras is None:
        return resultado

    if stats is None:
        resultado['conclusao_estatistica'] = 'Teste não executado: scipy não está disponível no ambiente.'
        return resultado

    if len(amostras) < 2:
        resultado['conclusao_estatistica'] = 'Dados insuficientes: são necessários pelo menos dois grupos válidos.'
        return resultado

    if any(len(amostra) < 2 for amostra in amostras.values()):
        resultado['conclusao_estatistica'] = 'Dados insuficientes: cada grupo precisa conter pelo menos dois registros válidos.'
        return resultado

    normalidade = {}
    normalidade_verificavel = True
    for grupo, amostra in amostras.items():
        if len(amostra) < 3:
            normalidade_verificavel = False
            normalidade[grupo] = None
            continue
        _, p_valor = stats.shapiro(amostra)
        normalidade[grupo] = None if pd.isna(p_valor) else float(p_valor)
        if normalidade[grupo] is None or normalidade[grupo] <= NIVEL_SIGNIFICANCIA:
            normalidade_verificavel = False

    _, p_levene = stats.levene(*amostras.values(), center='median')
    variancias_homogeneas = not pd.isna(p_levene) and p_levene > NIVEL_SIGNIFICANCIA
    resultado['diagnosticos'] = {
        'normalidade_shapiro_p_valor': normalidade,
        'homogeneidade_levene_p_valor': None if pd.isna(p_levene) else float(p_levene),
        'normalidade_atendida': normalidade_verificavel,
        'homogeneidade_variancias_atendida': variancias_homogeneas
    }

    if normalidade_verificavel and variancias_homogeneas:
        teste = stats.f_oneway(*amostras.values())
        nome_teste = 'ANOVA'
    else:
        teste = stats.kruskal(*amostras.values())
        nome_teste = 'Kruskal-Wallis'

    if pd.isna(teste.statistic) or pd.isna(teste.pvalue):
        resultado['conclusao_estatistica'] = 'Teste não executado: os dados não permitem calcular uma estatística válida.'
        return resultado

    resultado.update({
        'teste_utilizado': nome_teste,
        'estatistica': float(teste.statistic),
        'p_valor': float(teste.pvalue),
        'conclusao_estatistica': _conclusao(float(teste.pvalue))
    })
    return resultado


def testar_metodo_anotacao_gpa(df=None):
    return _executar_teste(df, NOTE_TAKING_METHOD_COLUMN)


def testar_curso_gpa(df=None):
    return _executar_teste(df, MAJOR_COLUMN)
