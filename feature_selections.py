from sklearn.decomposition import PCA

def apply_pca(features, n_components):
    """Reduce features to top n_components using PCA."""
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    return reduced_features
