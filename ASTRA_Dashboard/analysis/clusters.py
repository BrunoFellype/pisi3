from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def executar_kmeans(df, k_clusters):
    features_kmeans = [
        'study_hours_per_day', 'sleep_hours', 'mental_stress_level',
        'class_attendance_percent', 'social_media_hours', 'GPA'
    ]
    
    # Normalização com StandardScaler e treinamento do KMeans
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features_kmeans].fillna(df[features_kmeans].mean()))
    
    kmeans_model = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    df['cluster_id'] = kmeans_model.fit_predict(X_scaled)
    
    # Rótulos automáticos ordenados por desempenho acadêmico (GPA)
    medias_cluster = df.groupby('cluster_id')[features_kmeans].mean()
    ordem_gpa = medias_cluster['GPA'].sort_values(ascending=False).index.tolist()

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
        
    df['perfil_cluster'] = df['cluster_id'].map(mapa_nomes)

    return df, medias_cluster, ordem_gpa, mapa_nomes