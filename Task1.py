import cupy as cp
import numpy as np
import time
from test import testdata_kmeans, testdata_knn, testdata_ann
from tqdm import tqdm

# ----------------------------------------------------
# Setting seed for reproducibility 
# ----------------------------------------------------
'''please uncomment if you wish to set seed for reproducibility'''

# seed_value = 42
# np.random.seed(seed_value)  # Set seed for NumPy
# cp.random.seed(seed_value)  # Set seed for CuPy

# ----------------------------------------------------
# Distance Functions for GPU (CUPY)
# ----------------------------------------------------
def distance_cosine(X, Y):
    X_cp = cp.asarray(X, dtype=cp.float32)
    Y_cp = cp.asarray(Y, dtype=cp.float32)
    if X_cp.ndim == 1:
        X_cp = X_cp[None, :]
    if Y_cp.ndim == 1:
        Y_cp = Y_cp[None, :]
    dot_product = cp.sum(X_cp * Y_cp, axis=1)
    norm_X = cp.linalg.norm(X_cp)
    norm_Y = cp.linalg.norm(Y_cp, axis=1)
    cosine_similarity = dot_product / (norm_X * norm_Y)
    return 1 - cosine_similarity

def distance_l2(X, Y):
    X_cp = cp.asarray(X, dtype=cp.float32)
    Y_cp = cp.asarray(Y, dtype=cp.float32)
    if X_cp.ndim == 1:
        X_cp = X_cp[None, :]
    if Y_cp.ndim == 1:
        Y_cp = Y_cp[None, :]
    diff = X_cp - Y_cp
    return cp.sum(diff ** 2, axis=1)

def distance_dot(X, Y):
    X_cp = cp.asarray(X, dtype=cp.float32)
    Y_cp = cp.asarray(Y, dtype=cp.float32)
    if X_cp.ndim == 1:
        X_cp = X_cp[None, :]
    return -cp.sum(X_cp * Y_cp, axis=1)

def distance_manhattan(X, Y):
    X_cp = cp.asarray(X, dtype=cp.float32)
    Y_cp = cp.asarray(Y, dtype=cp.float32)
    if X_cp.ndim == 1:
        X_cp = X_cp[None, :]
    if Y_cp.ndim == 1:
        Y_cp = Y_cp[None, :]
    diff = X_cp - Y_cp
    return cp.abs(diff).sum(axis=-1)

def compute_distance(X, Y, metric='l2'):
    if metric == 'l2':
        return distance_l2(X, Y)
    elif metric == 'cosine':
        return distance_cosine(X, Y)
    elif metric == 'dot':
        return distance_dot(X, Y)
    elif metric == 'manhattan':
        return distance_manhattan(X, Y)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")


# ----------------------------------------------------
# CPU-based Distance Functions for CPU (NUMPY)
# ----------------------------------------------------
def compute_distance_cpu(X, Y, metric='l2'):
    X_np = np.asarray(X, dtype=np.float32)
    Y_np = np.asarray(Y, dtype=np.float32)
    if X_np.ndim == 1:
        X_np = X_np[None, :]
    if Y_np.ndim == 1:
        Y_np = Y_np[None, :]

    if metric == 'l2':
        return np.sum((X_np - Y_np) ** 2, axis=1)
    elif metric == 'cosine':
        dot = np.sum(X_np * Y_np, axis=1)
        norm_x = np.linalg.norm(X_np)
        norm_y = np.linalg.norm(Y_np, axis=1)
        return 1 - (dot / (norm_x * norm_y))
    elif metric == 'dot':
        return -np.sum(X_np * Y_np, axis=1)
    elif metric == 'manhattan':
        return np.sum(np.abs(X_np - Y_np), axis=1)
    else:
        raise ValueError(f"Unsupported metric {metric}")


# ----------------------------------------------------
# Utility: Estimate batch size based on available GPU memory
# ----------------------------------------------------
def estimate_max_batch_size(D, memory_limit_bytes=2 * 1024**3):
    bytes_per_vector = D * 4  # float32
    return max(1, memory_limit_bytes // bytes_per_vector)

# ----------------------------------------------------
# KNN - GPU
# ----------------------------------------------------
def our_knn(N, D, A, X, K, distance_metric='l2'):
    A_cp = cp.asarray(A, dtype=cp.float32)
    X_cp = cp.asarray(X, dtype=cp.float32)
    distances = compute_distance(X_cp, A_cp, distance_metric)
    indices = cp.argsort(distances)[:K]
    return indices.get()


# ----------------------------------------------------
# KNN - CPU
# ----------------------------------------------------
def our_knn_cpu(N, D, A, X, K, distance_metric='l2'):
    A_np = np.asarray(A, dtype=np.float32)
    X_np = np.asarray(X, dtype=np.float32)
    if X_np.ndim == 1:
        X_np = X_np[None, :]
    
    distances = compute_distance_cpu(X_np, A_np, distance_metric)
    indices = np.argsort(distances)[:K]
    return indices


# ----------------------------------------------------
# K-Means - GPU
# ----------------------------------------------------
def our_kmeans(N, D, A, K, max_iters=100, tol=1e-4, distance_metric='l2', convergence='centroid'):
    A_cp = cp.asarray(A, dtype=cp.float32)
    indices = cp.random.permutation(N)[:K]
    centroids = A_cp[indices].copy()
    cluster_ids = cp.zeros(N, dtype=cp.int32)
    prev_cluster_ids = cp.full(N, -1, dtype=cp.int32)

    for _ in range(max_iters):
        for i in range(N):
            distances = compute_distance(A_cp[i], centroids, distance_metric)
            cluster_ids[i] = cp.argmin(distances)

        new_centroids = cp.zeros_like(centroids)
        for k in range(K):
            mask = cluster_ids == k
            if cp.any(mask):
                new_centroids[k] = A_cp[mask].mean(axis=0)
            else:
                new_centroids[k] = centroids[k]

        if convergence == 'centroid':
            if cp.all(cp.linalg.norm(new_centroids - centroids, axis=1) < tol):
                break
        elif convergence == 'cluster':
            if cp.all(cluster_ids == prev_cluster_ids):
                break
            prev_cluster_ids = cluster_ids.copy()

        centroids = new_centroids

    return cluster_ids


# ----------------------------------------------------
# K-Means - CPU
# ----------------------------------------------------
def our_kmeans_cpu(N, D, A, K, max_iters=100, tol=1e-4, distance_metric='l2', convergence='centroid'):
    A_np = np.asarray(A, dtype=np.float32)
    indices = np.random.permutation(N)[:K]
    centroids = A_np[indices].copy()
    cluster_ids = np.zeros(N, dtype=np.int32)
    prev_cluster_ids = np.full(N, -1, dtype=np.int32)

    for _ in range(max_iters):
        for i in range(N):
            distances = compute_distance_cpu(A_np[i], centroids, distance_metric)
            cluster_ids[i] = np.argmin(distances)

        new_centroids = np.zeros_like(centroids)
        for k in range(K):
            mask = cluster_ids == k
            if np.any(mask):
                new_centroids[k] = A_np[mask].mean(axis=0)
            else:
                new_centroids[k] = centroids[k]

        if convergence == 'centroid':
            if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
                break
        elif convergence == 'cluster':
            if np.all(cluster_ids == prev_cluster_ids):
                break
            prev_cluster_ids = cluster_ids.copy()

        centroids = new_centroids

    return cluster_ids


# ----------------------------------------------------
# OUR ANN - GPU
# ----------------------------------------------------
def our_ann(N, D, A, X, K, distance_metric='l2'):
    A_cp = cp.asarray(A, dtype=cp.float32)
    X_cp = cp.asarray(X, dtype=cp.float32)
    if X_cp.ndim == 1:
        X_cp = X_cp[None, :]

    cluster_num = 50
    K1 = 10
    K2 = 25

    index_start = time.time()
    clusters = our_kmeans(N, D, A_cp, cluster_num, distance_metric=distance_metric)
    centroids = cp.zeros((cluster_num, D), dtype=A_cp.dtype)
    for i in range(cluster_num):
        indices = cp.where(clusters == i)[0]
        if indices.size > 0:
            centroids[i] = cp.mean(A_cp[indices], axis=0)
    index_time = time.time() - index_start

    max_batch_size = estimate_max_batch_size(D)
    results = []
    query_start = time.time()

    for i in tqdm(range(0, X_cp.shape[0], max_batch_size), desc="ANN Query Batching"):
        X_batch = X_cp[i:i + max_batch_size]
        for query in X_batch:
            centroid_dists = compute_distance(query, centroids, distance_metric)
            nearest_cluster_indices = cp.argpartition(centroid_dists, K1)[:K1]

            candidate_indices_list = []
            for c in nearest_cluster_indices:
                indices = cp.where(clusters == c)[0]
                if indices.size == 0:
                    continue
                points_in_cluster = A_cp[indices]
                dists_in_cluster = compute_distance(query, points_in_cluster, distance_metric)
                k2 = int(min(K2, indices.size))
                top_k_in_cluster = cp.argpartition(dists_in_cluster, k2 - 1)[:k2]
                candidate_indices_list.append(indices[top_k_in_cluster])

            if len(candidate_indices_list) == 0:
                results.append(cp.array([], dtype=cp.int32))
                continue

            candidate_indices = cp.concatenate(candidate_indices_list)
            candidate_points = A_cp[candidate_indices]
            candidate_dists = compute_distance(query, candidate_points, distance_metric)
            k_final = int(min(K, candidate_dists.size))
            final_top_k = cp.argpartition(candidate_dists, k_final - 1)[:k_final]
            ann_indices = candidate_indices[final_top_k]
            results.append(ann_indices)

    query_time = time.time() - query_start
    print(f"Indexing Time: {index_time:.4f}s | Query Time: {query_time:.4f}s")
    return results[0] if len(results) == 1 else cp.stack(results)


# ----------------------------------------------------
# OUR ANN - CPU
# ----------------------------------------------------
def our_ann_cpu(N, D, A, X, K, distance_metric='l2'):
    A_np = np.asarray(A, dtype=np.float32)
    X_np = np.asarray(X, dtype=np.float32)
    if X_np.ndim == 1:
        X_np = X_np[None, :]

    cluster_num = 50
    K1 = 10
    K2 = 25

    # KMeans (on CPU)
    clusters = our_kmeans_cpu(N, D, A_np, cluster_num, distance_metric=distance_metric)
    centroids = np.zeros((cluster_num, D), dtype=np.float32)
    for i in range(cluster_num):
        indices = np.where(clusters == i)[0]
        if indices.size > 0:
            centroids[i] = A_np[indices].mean(axis=0)

    results = []
    for query in X_np:
        centroid_dists = compute_distance_cpu(query, centroids, distance_metric)
        nearest_cluster_indices = np.argpartition(centroid_dists, K1)[:K1]

        candidate_indices_list = []
        for c in nearest_cluster_indices:
            indices = np.where(clusters == c)[0]
            if indices.size == 0:
                continue
            points_in_cluster = A_np[indices]
            dists_in_cluster = compute_distance_cpu(query, points_in_cluster, distance_metric)
            k2 = int(min(K2, indices.size))
            top_k_in_cluster = np.argpartition(dists_in_cluster, k2 - 1)[:k2]
            candidate_indices_list.append(indices[top_k_in_cluster])

        if not candidate_indices_list:
            results.append(np.array([], dtype=np.int32))
            continue

        candidate_indices = np.concatenate(candidate_indices_list)
        candidate_points = A_np[candidate_indices]
        candidate_dists = compute_distance_cpu(query, candidate_points, distance_metric)
        k_final = int(min(K, candidate_dists.size))
        final_top_k = np.argpartition(candidate_dists, k_final - 1)[:k_final]
        ann_indices = candidate_indices[final_top_k]
        results.append(ann_indices)

    return results[0] if len(results) == 1 else np.stack(results)


# ----------------------------------------------------
# ANN - HNSW 
# ----------------------------------------------------
def our_ann_hnsw(N, D, A, X, K, ef=50, M=5, distance_metric='l2'):
    A_cp = cp.asarray(A, dtype=cp.float32)
    X_cp = cp.asarray(X, dtype=cp.float32)
    if X_cp.ndim == 1:
        X_cp = X_cp[None, :]

    graph = [[] for _ in range(N)]
    for i in range(N):
        distances = compute_distance(A_cp[i], A_cp[:i], distance_metric)
        if len(distances) > 0:
            neighbors = cp.argsort(distances)[:M].tolist()
            graph[i].extend(neighbors)
            for n in neighbors:
                graph[n].append(i)

    results = []
    for query in X_cp:
        entry_point = 0
        visited = set([entry_point])
        candidates = [(compute_distance(query, A_cp[entry_point][None, :], distance_metric)[0], entry_point)]

        for _ in range(ef):
            new_candidates = []
            for _, node in candidates:
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        dist = compute_distance(query, A_cp[neighbor][None, :], distance_metric)[0]
                        new_candidates.append((dist, neighbor))
            candidates.extend(new_candidates)
            candidates = sorted(candidates)[:ef]

        top_k = [idx for (_, idx) in sorted(candidates)[:K]]
        results.append(cp.array(top_k, dtype=cp.int32))

    return results[0] if len(results) == 1 else cp.stack(results)


# ----------------------------------------------------
# ANN - IVFPQ 
# ----------------------------------------------------
def our_ann_ivfpq(N, D, A, X, K, num_clusters=32, distance_metric='l2'):
    A_cp = cp.asarray(A, dtype=cp.float32)
    X_cp = cp.asarray(X, dtype=cp.float32)
    if X_cp.ndim == 1:
        X_cp = X_cp[None, :]

    clusters = our_kmeans(N, D, A_cp, num_clusters, distance_metric=distance_metric)
    centroids = cp.zeros((num_clusters, D), dtype=A_cp.dtype)
    for i in range(num_clusters):
        indices = cp.where(clusters == i)[0]
        if indices.size > 0:
            centroids[i] = cp.mean(A_cp[indices], axis=0)

    results = []
    for query in X_cp:
        query_cluster = cp.argmin(compute_distance(query, centroids, distance_metric))
        candidate_indices = cp.where(clusters == query_cluster)[0]
        if candidate_indices.size == 0:
            results.append(cp.array([], dtype=cp.int32))
            continue

        candidate_points = A_cp[candidate_indices]
        candidate_dists = compute_distance(query, candidate_points, distance_metric)
        k_final = int(min(K, candidate_dists.size))
        final_top_k = cp.argpartition(candidate_dists, k_final - 1)[:k_final]
        ann_indices = candidate_indices[final_top_k]
        results.append(ann_indices)

    return results[0] if len(results) == 1 else cp.stack(results)


# ----------------------------------------------------
# Utils
# ----------------------------------------------------
def recall_rate(list1, list2):
    list1=list1.tolist()
    list2=list2.tolist()
    
    return len(set(list1) & set(list2)) / len(set(list1))

def test_knn(test_file="", distance_metric='l2'):
    N, D, A, X, K = testdata_knn(test_file)
    return our_knn(N, D, A, X, K, distance_metric=distance_metric)

def test_kmeans(test_file="", distance_metric='l2'):
    N, D, A, K = testdata_kmeans(test_file)
    return our_kmeans(N, D, A, K, distance_metric=distance_metric)

def test_ann(test_file="", distance_metric='l2', method='our_ann'):
    N, D, A, X, K = testdata_ann(test_file)
    if method == 'our_ann':
        return our_ann(N, D, A, X, K, distance_metric=distance_metric)
    elif method == 'our_ann_hnsw':
        return our_ann_hnsw(N, D, A, X, K, distance_metric=distance_metric)
    elif method == 'our_ann_ivfpq':
        return our_ann_ivfpq(N, D, A, X, K, distance_metric=distance_metric)
    else:
        raise ValueError("Unknown ANN method")


def final_testing(test_file="", distance_metric='l2', method='our_ann'):
    N, D, A, X, K = testdata_ann(test_file)
    
    print(f"REQUESTED RESULTS FOR [{method.upper()}] \n")
    
    knn_val = our_knn(N, D, A, X, K, distance_metric=distance_metric)
    print(f"Result for KNN : {knn_val}")
    benchmark_ann(N, D, A, X, K, distance_metric=distance_metric, method='knn')
    
    
    if method == 'our_ann':
        ann_val = our_ann(N, D, A, X, K, distance_metric=distance_metric)
        print(f"Result for [{method.upper()}] : {ann_val}")
        recall = recall_rate(knn_val, ann_val)
        print(f"recall rate for [{method.upper()}] : {recall:.4f}")
        benchmark_ann(N, D, A, X, K, distance_metric=distance_metric, method=method)
        
    elif method == 'our_ann_hnsw':
        ann_val = our_ann_hnsw(N, D, A, X, K, distance_metric=distance_metric)
        print(f"Result for [{method.upper()}] : {ann_val}")
        recall = recall_rate(knn_val, ann_val)
        print(f"recall rate for [{method.upper()}] : {recall:.4f}")
        benchmark_ann(N, D, A, X, K, distance_metric=distance_metric, method=method)
        
    elif method == 'our_ann_ivfpq':
        ann_val = our_ann_ivfpq(N, D, A, X, K, distance_metric=distance_metric)
        print(f"Result for [{method.upper()}] : {ann_val}")
        recall = recall_rate(knn_val, ann_val)
        print(f"recall rate for [{method.upper()}] : {recall:.4f}")
        benchmark_ann(N, D, A, X, K, distance_metric=distance_metric, method=method)        
        
    else:
        raise ValueError("Unknown ANN method")


# ----------------------------------------------------
# Benchmarking ANN on CPU and GPU
# ----------------------------------------------------
def benchmark_ann(N, D, A, X, K, distance_metric='l2', method='our_ann'):
    # GPU benchmark
    start_gpu = time.time()
    if method == 'knn':
        indices_gpu = our_knn(N, D, A, X, K, distance_metric=distance_metric)
    elif method == 'our_ann':
        indices_gpu = our_ann(N, D, A, X, K, distance_metric=distance_metric)
    elif method == 'our_ann_hnsw':
        indices_gpu = our_ann_hnsw(N, D, A, X, K, distance_metric=distance_metric)
    elif method == 'our_ann_ivfpq':
        indices_gpu = our_ann_ivfpq(N, D, A, X, K, distance_metric=distance_metric)
    else:
        raise ValueError("Unknown method")
    end_gpu = time.time()
    gpu_time = end_gpu - start_gpu

    # CPU benchmark

    start_cpu = time.time()
    if method == 'knn':
        indices_cpu = our_knn_cpu(N, D, A, X, K, distance_metric=distance_metric)
    elif method == 'our_ann':
        indices_cpu = our_ann_cpu(N, D, A, X, K, distance_metric=distance_metric)

    end_cpu = time.time()
    cpu_time = end_cpu - start_cpu
    
    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
    if method == 'our_ann_hnsw' or method == 'our_ann_ivfpq' :
        print(f"[{method.upper()}] GPU Time: {gpu_time:.4f}s")     
    else :
        print(f"[{method.upper()}] GPU Time: {gpu_time:.4f}s | CPU Time: {cpu_time:.4f}s | Speedup: {speedup:.2f}x")


def recall_knn(N, D, A, X, K, distance_metric='l2'):
    return our_knn(N, D, A, X, K, distance_metric=distance_metric)

def recall_ann(N, D, A, X, K, distance_metric='l2', method='our_ann'):
    if method == 'our_ann':
        indices = our_ann(N, D, A, X, K, distance_metric=distance_metric)
    elif method == 'our_ann_hnsw':
        indices = our_ann_hnsw(N, D, A, X, K, distance_metric=distance_metric)
    elif method == 'our_ann_ivfpq':
        indices = our_ann_ivfpq(N, D, A, X, K, distance_metric=distance_metric)
    else:
        raise ValueError("Unknown method")
    
    return indices

 
# ----------------------------------------------------
# MAIN FUNCTION
# ----------------------------------------------------   
if __name__ == "__main__":
    
    # PLEASE USE THE BELOW MENTIONED FUNCTION : final_testing FOR GENERALIZED USAGE TO GET KNN AND ANN VALUES ALONG WITH RECALL AND BENCHMARK TIME VALUES FOR ANY SPECIFIC FILE. 
    '''
    USAGE AND FUNCTION PARAMETERS DESCRIPTION : 
    test_file : pass any file fitting the requirements (eg. test_file.json)
    distance_metric : any of the 4 distance metrics can be set - 'l2', 'cosine', 'dot', 'manhattan'
    method : any of the 3 ann implementations can be used - 'our_ann', 'our_ann_hnsw', 'our_ann_ivfpq'
    '''
    
    final_testing(test_file="", distance_metric='l2', method='our_ann')

    
    
    # THE BELOW TESTS ARE PURELY FOR TESTING AND COMPLETION OF QUESTIONS MENTIONED IN REPORT
    # YOU CAN COMMENT THEM OUT IF YOU DO NOT WISH TO TEST EVERY SINGLE FUNCTION SEPARATELY
    
    print("Testing to check every single implementation separately with default test.py values for a thorough analysis............")
    
    print(f"K-Means with l2 distance : {test_kmeans(test_file='', distance_metric='l2')}")
    print(f"K-Means with cosine distance : {test_kmeans(test_file='', distance_metric='cosine')}")
    print(f"K-Means with dot product : {test_kmeans(test_file='', distance_metric='dot')}")
    print(f"K-Means with manhattan distance : {test_kmeans(test_file='', distance_metric='manhattan')}")
    
    print(f"KNN with l2 distance : {test_knn(test_file='', distance_metric='l2')}")
    print(f"KNN with cosine distance : {test_knn(test_file='', distance_metric='cosine')}")
    print(f"KNN with dot product : {test_knn(test_file='', distance_metric='dot')}")
    print(f"KNN with manhattan distance : {test_knn(test_file='', distance_metric='manhattan')}")

    print(f"Our ANN with l2 distance : {test_ann(test_file='', distance_metric='l2', method='our_ann')}")
    print(f"Our ANN with cosine distance : {test_ann(test_file='', distance_metric='cosine', method='our_ann')}")
    print(f"Our ANN with dot product : {test_ann(test_file='', distance_metric='dot', method='our_ann')}")
    print(f"Our ANN with manhattan distance : {test_ann(test_file='', distance_metric='manhattan', method='our_ann')}")

    print(f"Our ANN with l2 distance : {test_ann(test_file='', distance_metric='l2', method='our_ann_hnsw')}")
    print(f"Our ANN with cosine distance : {test_ann(test_file='', distance_metric='cosine', method='our_ann_hnsw')}")
    print(f"Our ANN with dot product : {test_ann(test_file='', distance_metric='dot', method='our_ann_hnsw')}")
    print(f"Our ANN with manhattan distance : {test_ann(test_file='', distance_metric='manhattan', method='our_ann_hnsw')}")

    print(f"Our ANN with l2 distance : {test_ann(test_file='', distance_metric='l2', method='our_ann_ivfpq')}")
    print(f"Our ANN with cosine distance : {test_ann(test_file='', distance_metric='cosine', method='our_ann_ivfpq')}")
    print(f"Our ANN with dot product : {test_ann(test_file='', distance_metric='dot', method='our_ann_ivfpq')}")
    print(f"Our ANN with manhattan distance : {test_ann(test_file='', distance_metric='manhattan', method='our_ann_ivfpq')}")


    print("All preliminary tests completed with default test.py values.")
    
    
    print("\nTesting for benchmarking time taken and recall rate with values mentioned in README.md for task-1 and task-2 report completion. \n")
    
    # DIFFERENT VALUES TO BE USED FOR ANSWERING THE QUESTIONS MENTIONED IN THE README.md FILE FOR THE REPORT.
    
    N = 1000 
    # N = 4000
    # N = 4000000
    
    D = 100 
    # D = 2
    # D = 2**15
    # D = 1024
    
    A = np.random.randn(N, D)
    X = np.random.randn(D)
    K = 10
    
    
    # TIME BENCHMARKING TEST
    
    
    print("\n--- Benchmarking on l2 distance ---")
    
    print("our knn :")
    benchmark_ann(N, D, A, X, K, distance_metric='l2', method='knn')
    print("our ann :")
    benchmark_ann(N, D, A, X, K, distance_metric='l2', method='our_ann')
    print("hnsw :")
    benchmark_ann(N, D, A, X, K, distance_metric='l2', method='our_ann_hnsw')
    print("ivfpq :")
    benchmark_ann(N, D, A, X, K, distance_metric='l2', method='our_ann_ivfpq')


    print("\n--- Benchmarking on cosine distance ---")
    
    print("our knn :")
    benchmark_ann(N, D, A, X, K, distance_metric='cosine', method='knn')
    print("our ann :")
    benchmark_ann(N, D, A, X, K, distance_metric='cosine', method='our_ann')
    print("hnsw :")
    benchmark_ann(N, D, A, X, K, distance_metric='cosine', method='our_ann_hnsw')
    print("ivfpq :")
    benchmark_ann(N, D, A, X, K, distance_metric='cosine', method='our_ann_ivfpq')


    print("\n--- Benchmarking on dot product ---")
    
    print("our knn :")
    benchmark_ann(N, D, A, X, K, distance_metric='dot', method='knn')
    print("our ann :")
    benchmark_ann(N, D, A, X, K, distance_metric='dot', method='our_ann')
    print("hnsw :")
    benchmark_ann(N, D, A, X, K, distance_metric='dot', method='our_ann_hnsw')
    print("ivfpq :")
    benchmark_ann(N, D, A, X, K, distance_metric='dot', method='our_ann_ivfpq')


    print("\n--- Benchmarking on manhattan distance ---")
    
    print("our knn :")
    benchmark_ann(N, D, A, X, K, distance_metric='manhattan', method='knn')
    print("our ann :")
    benchmark_ann(N, D, A, X, K, distance_metric='manhattan', method='our_ann')
    print("hnsw :")
    benchmark_ann(N, D, A, X, K, distance_metric='manhattan', method='our_ann_hnsw')
    print("ivfpq :")
    benchmark_ann(N, D, A, X, K, distance_metric='manhattan', method='our_ann_ivfpq')


    # RECALL TEST


    print("Recall testing for Our ANN")

    print("Testing Recall on Our ANN for l2 distance .....")
    l_knn = recall_knn(N, D, A, X, K, distance_metric='l2')
    l_ann = recall_ann(N, D, A, X, K, distance_metric='l2', method='our_ann')
    print(f'Recall Rate: {recall_rate(l_knn, l_ann)}')
    
    print("Testing Recall on Our ANN for cosine distance .....")
    l_knn = recall_knn(N, D, A, X, K, distance_metric='cosine')
    l_ann = recall_ann(N, D, A, X, K, distance_metric='cosine', method='our_ann')
    print(f'Recall Rate: {recall_rate(l_knn, l_ann)}')
    
    print("Testing Recall on Our ANN for dot product .....")
    l_knn = recall_knn(N, D, A, X, K, distance_metric='dot')
    l_ann = recall_ann(N, D, A, X, K, distance_metric='dot', method='our_ann')
    print(f'Recall Rate: {recall_rate(l_knn, l_ann)}')

    print("Testing Recall on Our ANN for manhattan distance .....")
    l_knn = recall_knn(N, D, A, X, K, distance_metric='manhattan')
    l_ann = recall_ann(N, D, A, X, K, distance_metric='manhattan', method='our_ann')
    print(f'Recall Rate: {recall_rate(l_knn, l_ann)}')


    print("Recall testing for HNSW")

    print("Testing Recall on HNSW for l2 distance .....")
    l_knn = recall_knn(N, D, A, X, K, distance_metric='l2')
    l_ann = recall_ann(N, D, A, X, K, distance_metric='l2', method='our_ann_hnsw')
    print(f'Recall Rate: {recall_rate(l_knn, l_ann)}')
    
    print("Testing Recall on HNSW for cosine distance .....")
    l_knn = recall_knn(N, D, A, X, K, distance_metric='cosine')
    l_ann = recall_ann(N, D, A, X, K, distance_metric='cosine', method='our_ann_hnsw')
    print(f'Recall Rate: {recall_rate(l_knn, l_ann)}')
    
    print("Testing Recall on HNSW for dot product .....")
    l_knn = recall_knn(N, D, A, X, K, distance_metric='dot')
    l_ann = recall_ann(N, D, A, X, K, distance_metric='dot', method='our_ann_hnsw')
    print(f'Recall Rate: {recall_rate(l_knn, l_ann)}')

    print("Testing Recall on HNSW for manhattan distance .....")
    l_knn = recall_knn(N, D, A, X, K, distance_metric='manhattan')
    l_ann = recall_ann(N, D, A, X, K, distance_metric='manhattan', method='our_ann_hnsw')
    print(f'Recall Rate: {recall_rate(l_knn, l_ann)}')


    print("Recall testing for IVFPQ")

    print("Testing Recall on IVFPQ for l2 distance .....")
    l_knn = recall_knn(N, D, A, X, K, distance_metric='l2')
    l_ann = recall_ann(N, D, A, X, K, distance_metric='l2', method='our_ann_ivfpq')
    print(f'Recall Rate: {recall_rate(l_knn, l_ann)}')
    
    print("Testing Recall on IVFPQ for cosine distance .....")
    l_knn = recall_knn(N, D, A, X, K, distance_metric='cosine')
    l_ann = recall_ann(N, D, A, X, K, distance_metric='cosine', method='our_ann_ivfpq')
    print(f'Recall Rate: {recall_rate(l_knn, l_ann)}')
    
    print("Testing Recall on IVFPQ for dot product .....")
    l_knn = recall_knn(N, D, A, X, K, distance_metric='dot')
    l_ann = recall_ann(N, D, A, X, K, distance_metric='dot', method='our_ann_ivfpq')
    print(f'Recall Rate: {recall_rate(l_knn, l_ann)}')

    print("Testing Recall on IVFPQ for manhattan distance .....")
    l_knn = recall_knn(N, D, A, X, K, distance_metric='manhattan')
    l_ann = recall_ann(N, D, A, X, K, distance_metric='manhattan', method='our_ann_ivfpq')
    print(f'Recall Rate: {recall_rate(l_knn, l_ann)}')


    