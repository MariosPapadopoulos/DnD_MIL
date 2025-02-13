import torch
import numpy as np
from sklearn.cluster import KMeans

def construct_fairlets(labels, p, q):
    """
    Constructs fairlets based on the class labels.

    Args:
        labels (np.array): Array of class labels.
        p (int): Number of data points from class 0 in a fairlet.
        q (int): Number of data points from class 1 in a fairlet.

    Returns:
        List of fairlets, each fairlet is a list of indices.
    """
    class0_indices = np.where(labels == 0)[0].tolist()
    class1_indices = np.where(labels == 1)[0].tolist()

    fairlets = []
    while class0_indices or class1_indices:
        fairlet = []
        for _ in range(p):
            if class0_indices:
                fairlet.append(class0_indices.pop())
        for _ in range(q):
            if class1_indices:
                fairlet.append(class1_indices.pop())
        fairlets.append(fairlet)

    return fairlets


def construct_fairlets_multiclass(labels, class_distribution):
    """
    Constructs fairlets for multiclass data.

    Args:
        labels (np.array): Array of class labels.
        class_distribution (dict): Desired class distribution in each fairlet.

    Returns:
        List of fairlets, each fairlet is a list of indices.
    """
    class_indices = {cls: np.where(labels == cls)[0].tolist() for cls in np.unique(labels)}
    max_fairlet_size = sum(class_distribution.values())
    fairlets = []

    while any(class_indices.values()):
        fairlet = []
        for cls, count in class_distribution.items():
            for _ in range(count):
                if class_indices[cls]:
                    fairlet.append(class_indices[cls].pop())
        fairlets.append(fairlet)

    return fairlets


def cluster_fairlets(logits, fairlets, n_clusters):
    """
    Clusters fairlets based on their centroids.

    Args:
        logits (np.array): Feature representations of data points.
        fairlets (list): List of fairlets.
        n_clusters (int): Number of clusters.

    Returns:
        cluster_labels (np.array): Cluster labels for each data point.
    """
    fairlet_centroids = []
    for fairlet in fairlets:
        centroid = np.mean(logits[fairlet], axis=0)
        fairlet_centroids.append(centroid)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    fairlet_labels = kmeans.fit_predict(fairlet_centroids)

    cluster_labels = np.zeros(logits.shape[0], dtype=int)
    for fairlet, label in zip(fairlets, fairlet_labels):
        cluster_labels[fairlet] = label

    return cluster_labels



