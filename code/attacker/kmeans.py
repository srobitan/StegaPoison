"""
kmeans utility for GapStatistics in agg.py
Reconstructed to match the signature used by the UNION defense.
"""
import torch
import numpy as np
from sklearn.cluster import KMeans as SklearnKMeans


def kmeans(X, num_clusters, init='kmeans++', tol=1e-7, verbose=False, seed=None):
    """
    Wrapper around sklearn KMeans that returns PyTorch tensors.
    
    Args:
        X: torch.Tensor of shape (N, D)
        num_clusters: int
        init: str, initialization method
        tol: float, tolerance
        verbose: bool
        seed: int, random seed
        
    Returns:
        centroids: torch.Tensor of shape (num_clusters, D)
        labels: torch.Tensor of shape (N,)
    """
    device = X.device
    X_np = X.cpu().numpy()
    
    kmeans_model = SklearnKMeans(
        n_clusters=num_clusters,
        init=init if init in ['k-means++', 'random'] else 'k-means++',
        tol=tol,
        verbose=verbose,
        random_state=seed,
        n_init='auto'
    )
    kmeans_model.fit(X_np)
    
    centroids = torch.from_numpy(kmeans_model.cluster_centers_).float().to(device)
    labels = torch.from_numpy(kmeans_model.labels_).long().to(device)
    
    return centroids, labels
