import hashlib

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import pandas as pd

# Variáveis de hábitos usadas no agrupamento.
# O GPA NÃO entra no K-Means: ele é a variável que queremos explicar/prever.
# Ele é usado apenas depois, para ordenar e nomear os perfis encontrados.
FEATURES_KMEANS = [
    'study_hours_per_day', 'sleep_hours', 'mental_stress_level',
    'class_attendance_percent', 'social_media_hours'
]

# Valores de k avaliados pelo método do cotovelo (inércia) e pela silhueta.
K_AVALIADOS = list(range(2, 9))

RANDOM_STATE = 42

# Guarda a tabela de k já calculada para os mesmos dados (a silhueta é cara:
# compara todos os pares de estudantes). Mudar só o k no painel não recalcula a tabela.
_CACHE_METRICAS_K = {}


def _avaliar_valores_de_k(X_scaled):
    chave = hashlib.sha1(X_scaled.tobytes()).hexdigest()
    if chave in _CACHE_METRICAS_K:
        return _CACHE_METRICAS_K[chave].copy()

    metricas_k = []
    for k in K_AVALIADOS:
        if k >= len(X_scaled):
            break
        modelo_temp = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels_temp = modelo_temp.fit_predict(X_scaled)
        sil_score = silhouette_score(X_scaled, labels_temp)
        metricas_k.append({"k": k, "Silhouette Score": round(sil_score, 3), "Inércia": round(modelo_temp.inertia_, 2)})

    df_metricas_k = pd.DataFrame(metricas_k)
    if len(_CACHE_METRICAS_K) >= 16:
        _CACHE_METRICAS_K.clear()
    _CACHE_METRICAS_K[chave] = df_metricas_k
    return df_metricas_k.copy()


def executar_kmeans(df, k_clusters):
    features_kmeans = FEATURES_KMEANS

    df_cluster = df.copy()
    X_data = df_cluster[features_kmeans].fillna(df_cluster[features_kmeans].mean())

    # Normalização com StandardScaler (média 0 e desvio padrão 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)

    # Testar múltiplos valores de k para o relatório de cotovelo e silhueta
    df_metricas_k = _avaliar_valores_de_k(X_scaled)

    # Execução oficial com o 'k_clusters' escolhido
    kmeans_model = KMeans(n_clusters=k_clusters, random_state=RANDOM_STATE, n_init=10)
    df_cluster['cluster_id'] = kmeans_model.fit_predict(X_scaled)
    silhueta_atual = silhouette_score(X_scaled, df_cluster['cluster_id'])

    # PCA: redução para 2 dimensões, usada apenas para VISUALIZAR os perfis.
    # O K-Means roda no espaço completo das 5 variáveis padronizadas.
    pca = PCA(n_components=2)
    componentes_pca = pca.fit_transform(X_scaled)
    df_cluster['PC1'] = componentes_pca[:, 0]
    df_cluster['PC2'] = componentes_pca[:, 1]
    df_cluster.attrs['pca_variancia_explicada'] = [float(v) for v in pca.explained_variance_ratio_]

    # Rótulos automáticos ordenados pelo GPA médio de cada perfil (do maior para o menor)
    colunas_medias = features_kmeans + (['GPA'] if 'GPA' in df_cluster.columns else [])
    medias_cluster = df_cluster.groupby('cluster_id')[colunas_medias].mean()
    if 'GPA' in medias_cluster.columns:
        ordem_gpa = medias_cluster['GPA'].sort_values(ascending=False).index.tolist()
    else:
        ordem_gpa = sorted(medias_cluster.index.tolist())

    total_perfis = len(ordem_gpa)
    emojis_perfis = ["🎓", "⚖️", "⚡", "⚠️", "🔄", "📘", "📗", "📙"]
    mapa_nomes = {}

    for rank, cid in enumerate(ordem_gpa):
        posicao = rank + 1
        if posicao == 1:
            descricao = "Maior desempenho acadêmico"
        elif posicao == total_perfis:
            descricao = "Menor desempenho acadêmico"
        else:
            descricao = "Desempenho acadêmico intermediário"

        emoji = emojis_perfis[rank] if rank < len(emojis_perfis) else ""
        mapa_nomes[cid] = f"{emoji} Perfil {posicao} — {descricao}".strip()

    df_cluster['perfil_cluster'] = df_cluster['cluster_id'].map(mapa_nomes)

    return df_cluster, medias_cluster, ordem_gpa, mapa_nomes, df_metricas_k, round(silhueta_atual, 3)
