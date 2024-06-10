import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import time
import random
from sklearn.metrics import pairwise_distances_argmin

# Загрузка данных
digits = load_digits()
data = scale(digits.data)
labels = digits.target

# Вывод информации о данных
print(f"Shape of data: {data.shape}")
print(f"Shape of labels: {labels.shape}")

# Собственная реализация KMeans
def my_kmeans(data, n_clusters, max_iters=300, tol=1e-4):
    # Выбор случайных точек как начальные центры кластеров
    initial_centers = random.sample(list(data), n_clusters)
    centers = np.array(initial_centers)
    for i in range(max_iters):
        labels = pairwise_distances_argmin(data, centers)
        new_centers = np.array([data[labels == j].mean(axis=0) for j in range(n_clusters)])
        if np.all(np.linalg.norm(new_centers - centers, axis=1) < tol):
            break
        centers = new_centers
    return centers, labels

# Использование my_kmeans для кластеризации данных
n_clusters = len(np.unique(labels))
start_time = time.time()
my_centers, my_labels = my_kmeans(data, n_clusters)
my_kmeans_time = time.time() - start_time

# Использование KMeans с 'k-means++'
kmeans_plus = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
start_time = time.time()
kmeans_plus.fit(data)
kmeans_plus_time = time.time() - start_time

# Использование KMeans с 'random'
kmeans_random = KMeans(init='random', n_clusters=n_clusters, n_init=10)
start_time = time.time()
kmeans_random.fit(data)
kmeans_random_time = time.time() - start_time

# Метрики для my_kmeans
ari_my_kmeans = adjusted_rand_score(labels, my_labels)
ami_my_kmeans = adjusted_mutual_info_score(labels, my_labels)

# Метрики для KMeans с 'k-means++'
ari_kmeans_plus = adjusted_rand_score(labels, kmeans_plus.labels_)
ami_kmeans_plus = adjusted_mutual_info_score(labels, kmeans_plus.labels_)

# Метрики для KMeans с 'random'
ari_kmeans_random = adjusted_rand_score(labels, kmeans_random.labels_)
ami_kmeans_random = adjusted_mutual_info_score(labels, kmeans_random.labels_)

# Вывод результатов
print(f"My KMeans: ARI={ari_my_kmeans}, AMI={ami_my_kmeans}, Time={my_kmeans_time}")
print(f"KMeans++: ARI={ari_kmeans_plus}, AMI={ami_kmeans_plus}, Time={kmeans_plus_time}")
print(f"Random KMeans: ARI={ari_kmeans_random}, AMI={ami_kmeans_random}, Time={kmeans_random_time}")

# PCA для уменьшения размерности до 2D
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)

# Визуализация результатов
def plot_clusters(data, labels, centers, title):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x')
    plt.title(title)
    plt.show()

# Визуализация my_kmeans
plot_clusters(reduced_data, my_labels, pca.transform(my_centers), 'My KMeans')

# Визуализация KMeans с 'k-means++'
plot_clusters(reduced_data, kmeans_plus.labels_, pca.transform(kmeans_plus.cluster_centers_), 'KMeans++')

# Визуализация KMeans с 'random'
plot_clusters(reduced_data, kmeans_random.labels_, pca.transform(kmeans_random.cluster_centers_), 'Random KMeans')
