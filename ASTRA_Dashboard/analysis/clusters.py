from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

def selecionar_features():
    return [
        'study_hours_per_day',
        'class_attendance_percent',
        'sleep_hours',
        'screen_time_hours',
        'social_media_hours',
        'gaming_hours',
        'exercise_hours_per_week',
        'mental_stress_level',
        'AI_tool_usage_hours',
        'exam_preparation_days',
        'coffee_consumption_per_day',
        'extracurricular_hours_per_week'
    ]

def preprocess_cluster(df):
    features = selecionar_features()
    dados = df[features].copy()
    dados = dados.fillna(dados.mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(dados)
    return X_scaled

def executar_dbscan(df, eps, min_samples):
    X_scaled = preprocess_cluster(df)
    
    modelo = DBSCAN(eps=eps, min_samples=min_samples)
    labels = modelo.fit_predict(X_scaled)

    return modelo, labels

def executar_pca(df, n_components=None):
    X_scaled = preprocess_cluster(df)
    pca = PCA(n_components=n_components)
    
    X_pca = pca.fit_transform(X_scaled)
    return pca, X_pca

def executar_kmeans(df, k_clusters):
    X_scaled = preprocess_cluster(df)

    modelo = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    labels = modelo.fit_predict(X_scaled)

    return modelo, labels