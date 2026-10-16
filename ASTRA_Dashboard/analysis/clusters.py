from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # <--- 1. Importe o PCA
from sklearn.metrics import silhouette_score
import pandas as pd

def executar_kmeans(df, k_clusters):
    features_kmeans = [
        'study_hours_per_day', 'sleep_hours', 'mental_stress_level',
        'class_attendance_percent', 'social_media_hours', 'GPA'
    ]
    
    df_cluster = df.copy()
    X_data = df_cluster[features_kmeans].fillna(df_cluster[features_kmeans].mean())
    
    # Normalização com StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)
    
    # Testar múltiplos valores de k (2 a 5) para o relatório de silhueta
    metricas_k = []
    for k in [2, 3, 4, 5]:
        modelo_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_temp = modelo_temp.fit_predict(X_scaled)
        sil_score = silhouette_score(X_scaled, labels_temp)
        metricas_k.append({"k": k, "Silhouette Score": round(sil_score, 3), "Inércia": round(modelo_temp.inertia_, 2)})
    
    df_metricas_k = pd.DataFrame(metricas_k)

    # Execução oficial com o 'k_clusters' escolhido
    kmeans_model = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    df_cluster['cluster_id'] = kmeans_model.fit_predict(X_scaled)
    silhueta_atual = silhouette_score(X_scaled, df_cluster['cluster_id'])

    # --- 2. APLICAR O PCA (Redução de dimensionalidade para 2D) ---
    pca = PCA(n_components=2)
    componentes_pca = pca.fit_transform(X_scaled)
    df_cluster['PC1'] = componentes_pca[:, 0]
    df_cluster['PC2'] = componentes_pca[:, 1]
    # -------------------------------------------------------------

    # Rótulos automáticos ordenados por GPA
    # Rótulos automáticos ordenados por desempenho acadêmico (GPA) com descrição completa
    medias_cluster = df_cluster.groupby('cluster_id')[features_kmeans].mean()
    ordem_gpa = medias_cluster['GPA'].sort_values(ascending=False).index.tolist()

    total_perfis = len(ordem_gpa)
    emojis_perfis = ["🎓", "⚖️", "⚡", "⚠️", "🔄"]
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
        # Restaura o nome completo com emoji, posição e descrição
        mapa_nomes[cid] = f"{emoji} Perfil {posicao} — {descricao}".strip()
        
    df_cluster['perfil_cluster'] = df_cluster['cluster_id'].map(mapa_nomes)

    return df_cluster, medias_cluster, ordem_gpa, mapa_nomes, df_metricas_k, round(silhueta_atual, 3)