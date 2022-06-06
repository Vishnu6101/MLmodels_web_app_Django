import base64
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def Kmeans():
    path = 'clustering/wine-clustering.csv'
    data = pd.read_csv(path)

    Scale = StandardScaler().fit_transform(data)
    scaled_data = pd.DataFrame(Scale, columns = data.columns)

    pca = PCA(n_components=2).fit_transform(scaled_data)

    kmeans = KMeans(n_clusters = 3, init = "k-means++",random_state=None, tol=0.0001)
    kmeans.fit(pca)

    silhoutteScore = round(silhouette_score(pca, kmeans.labels_, metric = "euclidean"), 4)

    cluster_polt = clusters(pca, kmeans)

    SSE = []
    for cluster in range(1,20):
        kmeans = KMeans(n_clusters = cluster, init='k-means++')
        kmeans.fit(pca)
        SSE.append(kmeans.inertia_)

    # converting the results into a dataframe and plotting them
    cluster_frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
    elbow = elbow_graph(cluster_frame)

    return dict({'File' : path.split('/')[1], 'Data Columns' : data.columns.tolist(), 'Silhoutte Score' : silhoutteScore}), cluster_polt, elbow

def clusters(data, model):
    plt.figure()
    plt.scatter(data[:,0], data[:,1], s=50, c=model.labels_)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    graph = get_graph()
    return graph

def elbow_graph(frame):
    plt.figure(figsize=(12,6))
    plt.plot(frame['Cluster'], frame['SSE'], marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    graph = get_graph()
    return graph


def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer, formt='png')
    buffer.seek(0)
    img = buffer.getvalue()
    graph = base64.b64encode(img)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph
