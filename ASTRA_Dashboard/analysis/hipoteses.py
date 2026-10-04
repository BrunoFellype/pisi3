import pandas as pd

try:
    from scipy import stats
except ImportError:
    stats = None

try:
    from data.load import get_Dataset
except ImportError:
    try:
        from ASTRA_Dashboard.data.load import get_Dataset
    except ImportError:
        from ..data.load import get_Dataset


NIVEL_SIGNIFICANCIA = 0.05
GPA_COLUMN = 'GPA'
NOTE_TAKING_METHOD_COLUMN = 'note_taking_method'
MAJOR_COLUMN = 'major'


def _resultado_base(agrupamento, variavel=GPA_COLUMN, grupos=None, tamanhos=None, quantidade=0):
    return {
        'variavel_agrupamento': agrupamento,
        'variavel_analisada': variavel,
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


def _dados_para_teste(df, agrupamento, variavel=GPA_COLUMN):
    if df is None:
        df = get_Dataset()

    resultado = _resultado_base(agrupamento, variavel=variavel)
    colunas_necessarias = [agrupamento, variavel]
    if any(coluna not in df.columns for coluna in colunas_necessarias):
        resultado['conclusao_estatistica'] = 'Dados insuficientes: uma ou mais colunas necessárias não estão disponíveis.'
        return resultado, None

    dados = df[colunas_necessarias].copy()
    dados[variavel] = pd.to_numeric(dados[variavel], errors='coerce')
    dados = dados.dropna(subset=colunas_necessarias)
    dados = dados[dados[variavel].apply(lambda valor: pd.notna(valor) and pd.api.types.is_number(valor))]
    if dados.empty:
        resultado['conclusao_estatistica'] = 'Dados insuficientes: não há registros válidos para a análise.'
        return resultado, None

    if pd.api.types.is_object_dtype(dados[agrupamento]) or pd.api.types.is_string_dtype(dados[agrupamento]):
        dados[agrupamento] = dados[agrupamento].astype(str).str.strip()
        dados = dados[dados[agrupamento] != '']

    grupos = list(dados[agrupamento].drop_duplicates())
    amostras = {
        grupo: dados.loc[dados[agrupamento] == grupo, variavel].tolist()
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


def _executar_teste(df, agrupamento, variavel=GPA_COLUMN):
    resultado, amostras = _dados_para_teste(df, agrupamento, variavel)
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


def comparar_desempenho_por_cluster(df=None, cluster_labels=None, cluster_coluna='cluster_id', variaveis_desempenho=None):
    if df is None:
        df = get_Dataset()

    dados = df.copy()
    if cluster_labels is not None:
        if len(cluster_labels) != len(dados):
            raise ValueError('O número de rótulos de cluster deve coincidir com o número de linhas do DataFrame.')
        dados[cluster_coluna] = list(cluster_labels)

    if cluster_coluna not in dados.columns:
        raise ValueError(f'A coluna de cluster {cluster_coluna} não foi encontrada no DataFrame.')

    variaveis = ['GPA', 'final_exam_score', 'assignment_score'] if variaveis_desempenho is None else list(variaveis_desempenho)
    variaveis = [variavel for variavel in variaveis if variavel in dados.columns]
    if not variaveis:
        return {'tabela': pd.DataFrame(columns=[cluster_coluna, 'n']), 'teste_por_variavel': {}}

    tabela = dados.groupby(cluster_coluna, dropna=False).agg(
        n=(cluster_coluna, 'size'),
        **{f'{variavel}_media': (variavel, 'mean') for variavel in variaveis},
        **{f'{variavel}_mediana': (variavel, 'median') for variavel in variaveis}
    ).reset_index()

    testes = {}
    for variavel in variaveis:
        teste = _executar_teste(dados, cluster_coluna, variavel=variavel)
        testes[variavel] = {
            'teste_utilizado': teste.get('teste_utilizado'),
            'estatistica': teste.get('estatistica'),
            'p_valor': teste.get('p_valor'),
            'conclusao_estatistica': teste.get('conclusao_estatistica')
        }
        tabela[f'{variavel}_teste'] = teste.get('teste_utilizado')
        tabela[f'{variavel}_estatistica'] = teste.get('estatistica')
        tabela[f'{variavel}_p_valor'] = teste.get('p_valor')
        tabela[f'{variavel}_significativo'] = None if teste.get('p_valor') is None else teste.get('p_valor') < NIVEL_SIGNIFICANCIA

    tabela = tabela.sort_values(by=[cluster_coluna], kind='mergesort').reset_index(drop=True)
    return {
        'tabela': tabela,
        'teste_por_variavel': testes,
        'variaveis_analisadas': variaveis,
        'cluster_coluna': cluster_coluna
    }


def testar_metodo_anotacao_gpa(df=None):
    return _executar_teste(df, NOTE_TAKING_METHOD_COLUMN)


def testar_curso_gpa(df=None):
    return _executar_teste(df, MAJOR_COLUMN)
