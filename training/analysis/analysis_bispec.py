import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import umap
import pickle
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import os
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import confusion_matrix, classification_report


class BispectrumClusteringAnalyzer:
    """Comprehensive clustering analysis of bispectrum targets"""
    
    def __init__(self):
        # Create output directories
        os.makedirs('plots/bispectrum_clustering', exist_ok=True)
        os.makedirs('plots/bispectrum_clustering/volume_analysis', exist_ok=True)
        os.makedirs('plots/bispectrum_clustering/bravais_analysis', exist_ok=True)
        os.makedirs('plots/bispectrum_clustering/comparison', exist_ok=True)
        
    def load_and_prepare_data(self):
        """Load bispectrum data and corresponding crystal structures"""
        print("Loading data...")
        
        with open('pickles/mp20_bispec_lat_val.pkl', 'rb') as f:
            bispec_list_val = pickle.load(f)
        
        with open('pickles/val_results.pkl', 'rb') as f:
            ordered_result = pickle.load(f)
        
        bispectrum_array = np.array(bispec_list_val)
        
        print(f"Bispectrum shape: {bispectrum_array.shape}")
        print(f"Number of structures: {len(ordered_result)}")
        
        
        return bispectrum_array, ordered_result
    
    def extract_volume_labels_10_bins(self, ordered_result):
        """Extract cell volumes and create 10 quantile-based bins"""
        volumes = []
        
        for result in ordered_result:
            crystal = result['crystal']
            volume = crystal.lattice.volume
            volumes.append(volume)
        
        volumes = np.array(volumes)
        
        # Create 10 quantile-based bins
        volume_percentiles = np.linspace(0, 100, 11)
        volume_bins = np.percentile(volumes, volume_percentiles)
        volume_labels = np.digitize(volumes, volume_bins[1:-1])
        
        # Create descriptive labels
        volume_bin_labels = []
        for i, vol in enumerate(volumes):
            bin_idx = volume_labels[i]
            if bin_idx == 0:
                label = f"Bin 1 (< {volume_bins[1]:.0f})"
            elif bin_idx == 9:
                label = f"Bin 10 (> {volume_bins[9]:.0f})"
            else:
                label = f"Bin {bin_idx+1} ({volume_bins[bin_idx]:.0f}-{volume_bins[bin_idx+1]:.0f})"
            volume_bin_labels.append(label)
        
        return np.array(volume_bin_labels), volumes, volume_bins
    
    def extract_bravais_lattice_labels(self, ordered_result):
        """Extract Bravais lattice type labels"""
        bravais_labels = []
        
        # Space group to Bravais lattice mapping
        space_group_to_bravais = {
            **{i: "Triclinic" for i in range(1, 3)},
            **{i: "Monoclinic" for i in range(3, 16)},
            **{i: "Orthorhombic" for i in range(16, 75)},
            **{i: "Tetragonal" for i in range(75, 143)},
            **{i: "Hexagonal" for i in range(143, 195)},
            **{i: "Cubic" for i in range(195, 231)}
        }
        
        for result in ordered_result:
            crystal = result['crystal']
            try:
                sga = SpacegroupAnalyzer(crystal)
                space_group_number = sga.get_space_group_number()
                bravais_type = space_group_to_bravais.get(space_group_number, "Unknown")
                bravais_labels.append(bravais_type)
            except Exception as e:
                print(f"Warning: Could not determine Bravais lattice: {e}")
                bravais_labels.append("Unknown")
        
        return np.array(bravais_labels)
    
    def preprocess_bispectrum(self, bispectrum_array, method='standard'):
        """Preprocess bispectrum data"""
        print(f"Preprocessing bispectrum data using {method} scaling...")
        print(f"Original bispectrum shape: {bispectrum_array.shape}")
        
        # Handle multidimensional bispectrum data
        original_shape = bispectrum_array.shape
        
        # Reshape to 2D if necessary (flatten all dimensions except the first)
        if len(original_shape) > 2:
            n_samples = original_shape[0]
            n_features = np.prod(original_shape[1:])
            bispectrum_reshaped = bispectrum_array.reshape(n_samples, n_features)
            print(f"Reshaped bispectrum to: {bispectrum_reshaped.shape}")
        else:
            bispectrum_reshaped = bispectrum_array
        
        # Apply scaling
        if method == 'standard':
            scaler = StandardScaler()
            bispectrum_scaled = scaler.fit_transform(bispectrum_reshaped)
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            bispectrum_scaled = scaler.fit_transform(bispectrum_reshaped)
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            bispectrum_scaled = scaler.fit_transform(bispectrum_reshaped)
        else:
            bispectrum_scaled = bispectrum_reshaped
            scaler = None
        
        print(f"Final processed shape: {bispectrum_scaled.shape}")
        return bispectrum_scaled, scaler
    
    def perform_dimensionality_reduction(self, bispectrum_data, methods=['pca', 'tsne', 'umap']):
        """Perform multiple dimensionality reduction techniques"""
        results = {}
        
        print("Performing dimensionality reduction...")
        print(f"Input data shape: {bispectrum_data.shape}")
        
        # Ensure we have enough samples for t-SNE
        n_samples = bispectrum_data.shape[0]
        perplexity = min(30, (n_samples - 1) // 3)  # Ensure perplexity < n_samples/3
        
        for method in methods:
            print(f"  Computing {method.upper()}...")
            
            try:
                if method == 'pca':
                    # Ensure we don't request more components than available
                    n_components = min(2, bispectrum_data.shape[1])
                    reducer = PCA(n_components=n_components, random_state=42)
                    embedding = reducer.fit_transform(bispectrum_data)
                    
                    # If only 1 component, pad with zeros for visualization
                    if n_components == 1:
                        embedding = np.column_stack([embedding, np.zeros(len(embedding))])
                    
                    results[method] = embedding
                    results[f'{method}_explained_variance'] = reducer.explained_variance_ratio_
                    results[f'{method}_reducer'] = reducer
                    
                elif method == 'tsne':
                    if n_samples < 4:
                        print(f"    Skipping t-SNE: not enough samples ({n_samples})")
                        continue
                    
                    reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
                    embedding = reducer.fit_transform(bispectrum_data)
                    results[method] = embedding
                    
                elif method == 'umap':
                    if n_samples < 4:
                        print(f"    Skipping UMAP: not enough samples ({n_samples})")
                        continue
                    
                    n_neighbors = min(15, n_samples - 1)
                    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.1)
                    embedding = reducer.fit_transform(bispectrum_data)
                    results[method] = embedding
                    results[f'{method}_reducer'] = reducer
                
                # Verify the output shape
                if method in results:
                    print(f"    {method.upper()} output shape: {results[method].shape}")
                    
            except Exception as e:
                print(f"    Error computing {method.upper()}: {str(e)}")
                continue
        
        return results
    
    def perform_clustering_analysis(self, bispectrum_data, true_labels, clustering_methods=['kmeans', 'hierarchical', 'dbscan']):
        """Perform multiple clustering analyses"""
        le = LabelEncoder()
        numeric_true_labels = le.fit_transform(true_labels)
        n_true_clusters = len(np.unique(true_labels))
        
        clustering_results = {}
        
        print(f"Performing clustering analysis with {n_true_clusters} expected clusters...")
        
        for method in clustering_methods:
            print(f"  Running {method}...")
            
            if method == 'kmeans':
                clusterer = KMeans(n_clusters=n_true_clusters, random_state=42, n_init=10)
                cluster_labels = clusterer.fit_predict(bispectrum_data)
                
            elif method == 'hierarchical':
                clusterer = AgglomerativeClustering(n_clusters=n_true_clusters)
                cluster_labels = clusterer.fit_predict(bispectrum_data)
                
            elif method == 'dbscan':
                # Use multiple eps values and select best
                eps_values = np.linspace(0.1, 2.0, 20)
                best_eps = 0.5
                best_score = -1
                
                for eps in eps_values:
                    clusterer_temp = DBSCAN(eps=eps, min_samples=5)
                    temp_labels = clusterer_temp.fit_predict(bispectrum_data)
                    
                    if len(np.unique(temp_labels)) > 1:  # Avoid single cluster
                        try:
                            score = silhouette_score(bispectrum_data, temp_labels)
                            if score > best_score:
                                best_score = score
                                best_eps = eps
                        except:
                            continue
                
                clusterer = DBSCAN(eps=best_eps, min_samples=5)
                cluster_labels = clusterer.fit_predict(bispectrum_data)
            
            
            clustering_results[method] = {
                'cluster_labels': cluster_labels,
                'clusterer': clusterer,
                'n_clusters': len(np.unique(cluster_labels[cluster_labels >= 0]))  # Exclude noise for DBSCAN
            }
        
        return clustering_results
    
    
    def create_clustering_visualization(self, embeddings, true_labels, cluster_results, 
                                      grouping_type, save_prefix=''):
        """Create comprehensive clustering visualization"""
        
        # Filter out failed embeddings and non-embedding objects
        valid_embeddings = {}
        embedding_methods = ['pca', 'tsne', 'umap']  # Only these are actual embeddings
        
        for method in embedding_methods:
            if method in embeddings:
                emb = embeddings[method]
                if (emb is not None and 
                    hasattr(emb, 'shape') and 
                    isinstance(emb, np.ndarray) and 
                    len(emb.shape) == 2 and 
                    emb.shape[1] >= 2):
                    valid_embeddings[method] = emb
                else:
                    print(f"Warning: Skipping {method} embedding due to invalid shape: {emb.shape if hasattr(emb, 'shape') else 'No shape attribute'}")
        
        if not valid_embeddings:
            print("Error: No valid embeddings found!")
            return None
        
        n_methods = len(valid_embeddings)
        n_clustering = len(cluster_results)
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_clustering + 1, n_methods, figsize=(6*n_methods, 6*(n_clustering + 1)))
        
        # Handle different subplot configurations
        if n_methods == 1 and n_clustering == 0:
            axes = np.array([[axes]])
        elif n_methods == 1:
            axes = axes.reshape(-1, 1)
        elif n_clustering == 0:
            axes = axes.reshape(1, -1)
        
        embedding_methods = list(valid_embeddings.keys())
        clustering_methods = list(cluster_results.keys())
        
        # Plot true labels (first row)
        for j, emb_method in enumerate(embedding_methods):
            ax = axes[0, j] if n_clustering > 0 or n_methods > 1 else axes[0, 0] if n_methods == 1 else axes[0]
            
            unique_labels = np.unique(true_labels)
            if grouping_type == 'volume':
                colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
            else:
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = true_labels == label
                if np.sum(mask) > 0:  # Only plot if there are points
                    ax.scatter(valid_embeddings[emb_method][mask, 0], valid_embeddings[emb_method][mask, 1],
                             c=[colors[i]], label=f'{label}', alpha=0.6, s=10)
            
            ax.set_title(f'True {grouping_type.title()} Labels - {emb_method.upper()}')
            ax.set_xlabel(f'{emb_method.upper()} Component 1')
            ax.set_ylabel(f'{emb_method.upper()} Component 2')
            ax.grid(True, alpha=0.3)
            
            if j == n_methods - 1:  # Add legend to last plot
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Plot clustering results
        for i, clust_method in enumerate(clustering_methods):
            for j, emb_method in enumerate(embedding_methods):
                if emb_method in valid_embeddings and clust_method in cluster_results:
                    # Handle subplot indexing
                    if n_clustering > 0 and n_methods > 1:
                        ax = axes[i + 1, j]
                    elif n_methods == 1:
                        ax = axes[i + 1] if n_clustering > 0 else axes[0]
                    else:
                        ax = axes[i + 1, j] if n_clustering > 0 else axes[j]
                    
                    cluster_labels = cluster_results[clust_method]['cluster_labels']
                    unique_clusters = np.unique(cluster_labels)
                    
                    # Use different colors for clusters
                    cluster_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
                    
                    for k, cluster in enumerate(unique_clusters):
                        mask = cluster_labels == cluster
                        if np.sum(mask) > 0:  # Only plot if there are points
                            if cluster == -1:  # Noise points for DBSCAN
                                ax.scatter(valid_embeddings[emb_method][mask, 0], valid_embeddings[emb_method][mask, 1],
                                         c='black', label='Noise', alpha=0.3, s=5, marker='x')
                            else:
                                ax.scatter(valid_embeddings[emb_method][mask, 0], valid_embeddings[emb_method][mask, 1],
                                         c=[cluster_colors[k]], label=f'Cluster {cluster}', alpha=0.6, s=10)
                    
                    title = f'{clust_method.title()} - {emb_method.upper()}\n'
                    
                    ax.set_title(title)
                    ax.set_xlabel(f'{emb_method.upper()} Component 1')
                    ax.set_ylabel(f'{emb_method.upper()} Component 2')
                    ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Bispectrum Clustering Analysis - {grouping_type.title()} Grouping', fontsize=16)
        plt.tight_layout()
        
        if save_prefix:
            filename = f'plots/bispectrum_clustering/{grouping_type}_analysis/{save_prefix}_clustering_analysis.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_confusion_matrices(self, true_labels, cluster_results, grouping_type, save_prefix=''):
        """Create confusion matrices for clustering results"""
        
        le = LabelEncoder()
        numeric_true_labels = le.fit_transform(true_labels)
        
        n_methods = len(cluster_results)
        fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 5))
        
        if n_methods == 1:
            axes = [axes]
        
        for i, (method, results) in enumerate(cluster_results.items()):
            cluster_labels = results['cluster_labels']
            
            # Filter out noise points for DBSCAN
            valid_mask = cluster_labels >= 0
            valid_true = numeric_true_labels[valid_mask]
            valid_cluster = cluster_labels[valid_mask]
            
            if len(valid_cluster) > 0:
                cm = confusion_matrix(valid_true, valid_cluster)
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
                axes[i].set_xlabel('Predicted Clusters')
                axes[i].set_ylabel('True Labels')
                axes[i].set_title(f'{method.title()} Clustering\n'
                                f'ARI: {results["metrics"]["adjusted_rand_score"]:.3f}')
        
        plt.suptitle(f'Clustering Confusion Matrices - {grouping_type.title()} Grouping')
        plt.tight_layout()
        
        if save_prefix:
            filename = f'plots/bispectrum_clustering/{grouping_type}_analysis/{save_prefix}_confusion_matrices.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_pca_analysis(self, bispectrum_data, true_labels, grouping_type, save_prefix=''):
        """Create detailed PCA analysis"""
        
        print(f"Creating PCA analysis for {grouping_type}...")
        
        try:
            # Ensure we have enough components for analysis
            n_features = bispectrum_data.shape[1]
            n_components = min(50, n_features, bispectrum_data.shape[0])
            
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(bispectrum_data)
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Explained variance
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            n_plot_components = min(50, len(cumvar))
            ax1.plot(range(1, n_plot_components + 1), cumvar[:n_plot_components], 'b-', linewidth=2)
            ax1.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
            ax1.axhline(y=0.90, color='orange', linestyle='--', label='90% variance')
            ax1.set_xlabel('Number of Components')
            ax1.set_ylabel('Cumulative Explained Variance')
            ax1.set_title('PCA Explained Variance')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Individual component variance
            n_bar_components = min(20, len(pca.explained_variance_ratio_))
            ax2.bar(range(1, n_bar_components + 1), pca.explained_variance_ratio_[:n_bar_components])
            ax2.set_xlabel('Principal Component')
            ax2.set_ylabel('Explained Variance Ratio')
            ax2.set_title('Individual Component Variance')
            ax2.grid(True, alpha=0.3)
            
            # PC1 vs PC2 scatter (if we have at least 2 components)
            if pca_result.shape[1] >= 2:
                unique_labels = np.unique(true_labels)
                if grouping_type == 'volume':
                    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
                else:
                    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
                
                for i, label in enumerate(unique_labels):
                    mask = true_labels == label
                    if np.sum(mask) > 0:
                        ax3.scatter(pca_result[mask, 0], pca_result[mask, 1],
                                   c=[colors[i]], label=f'{label}', alpha=0.6, s=10)
                
                ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
                ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
                ax3.set_title(f'PCA: PC1 vs PC2 - {grouping_type.title()}')
                ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'Insufficient components\nfor PC1 vs PC2 plot', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('PC1 vs PC2 - Not Available')
            
            # PC2 vs PC3 scatter (if we have at least 3 components)
            if pca_result.shape[1] >= 3:
                for i, label in enumerate(unique_labels):
                    mask = true_labels == label
                    if np.sum(mask) > 0:
                        ax4.scatter(pca_result[mask, 1], pca_result[mask, 2],
                                   c=[colors[i]], label=f'{label}', alpha=0.6, s=10)
                
                ax4.set_xlabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
                ax4.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)')
                ax4.set_title(f'PCA: PC2 vs PC3 - {grouping_type.title()}')
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'Insufficient components\nfor PC2 vs PC3 plot', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('PC2 vs PC3 - Not Available')
            
            plt.suptitle(f'PCA Analysis of Bispectrum Data - {grouping_type.title()}')
            plt.tight_layout()
            
            if save_prefix:
                filename = f'plots/bispectrum_clustering/{grouping_type}_analysis/{save_prefix}_pca_detailed.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            print(f"Error creating PCA analysis: {str(e)}")
            return None
    
    def create_summary_comparison(self, volume_results, bravais_results):
        """Create summary comparison between volume and Bravais analysis"""
        
        # Extract metrics for comparison
        methods = ['kmeans', 'hierarchical', 'dbscan']
        metrics = ['silhouette_score', 'adjusted_rand_score', 'calinski_harabasz_score']
        
        # Prepare data
        comparison_data = []
        
        for method in methods:
            for metric in metrics:
                if method in volume_results and method in bravais_results:
                    vol_value = volume_results[method]['eetrics'].get(metric, 0)
                    brav_value = bravais_results[method]['metrics'].get(metric, 0)
                    
                    comparison_data.append({
                        'Method': method.title(),
                        'Metric': metric.replace('_', ' ').title(),
                        'Volume (10 bins)': vol_value,
                        'Bravais Lattice': brav_value
                    })
        
        df = pd.DataFrame(comparison_data)
        
        # Create comparison plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, metric in enumerate(metrics):
            metric_data = df[df['Metric'] == metric.replace('_', ' ').title()]
            
            x_pos = np.arange(len(methods))
            width = 0.35
            
            vol_values = metric_data['Volume (10 bins)'].values
            brav_values = metric_data['Bravais Lattice'].values
            
            axes[i].bar(x_pos - width/2, vol_values, width, label='Volume (10 bins)', alpha=0.7)
            axes[i].bar(x_pos + width/2, brav_values, width, label='Bravais Lattice', alpha=0.7)
            
            axes[i].set_xlabel('Clustering Method')
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
            axes[i].set_xticks(x_pos)
            axes[i].set_xticklabels([m.title() for m in methods])
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Bispectrum Clustering Performance Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig('plots/bispectrum_clustering/comparison/summary_comparison.png', dpi=300, bbox_inches='tight')
        
        return fig, df

def main():
    """Main analysis pipeline for bispectrum clustering"""
    
    print("="*80)
    print("BISPECTRUM TARGET CLUSTERING ANALYSIS")
    print("="*80)
    
    # Initialize analyzer
    analyzer = BispectrumClusteringAnalyzer()
    
    # Load and prepare data
    bispectrum_array, ordered_result = analyzer.load_and_prepare_data()
    
    # Extract grouping labels
    print("\nExtracting grouping labels...")
    volume_labels, volumes, volume_bins = analyzer.extract_volume_labels_10_bins(ordered_result)
    bravais_labels = analyzer.extract_bravais_lattice_labels(ordered_result)
    
    # Print distributions
    print("\nVolume Distribution (10 bins):")
    unique_vol, vol_counts = np.unique(volume_labels, return_counts=True)
    for vol, count in zip(unique_vol, vol_counts):
        percentage = (count / len(volume_labels)) * 100
        print(f"  {vol}: {count} samples ({percentage:.1f}%)")
    
    print("\nBravais Lattice Distribution:")
    unique_brav, brav_counts = np.unique(bravais_labels, return_counts=True)
    for brav, count in zip(unique_brav, brav_counts):
        percentage = (count / len(bravais_labels)) * 100
        print(f"  {brav}: {count} samples ({percentage:.1f}%)")
    
    # Preprocess bispectrum data
    bispectrum_scaled, scaler = analyzer.preprocess_bispectrum(bispectrum_array, method='standard')
    
    print(f"\nBispectrum statistics:")
    print(f"  Original shape: {bispectrum_array.shape}")
    print(f"  Mean: {np.mean(bispectrum_array):.3f}")
    print(f"  Std: {np.std(bispectrum_array):.3f}")
    print(f"  After scaling - Mean: {np.mean(bispectrum_scaled):.3f}, Std: {np.std(bispectrum_scaled):.3f}")
    
    # ========== VOLUME ANALYSIS ==========
    print("\n" + "="*60)
    print("VOLUME ANALYSIS (10 bins)")
    print("="*60)
    
    # Dimensionality reduction for volume analysis
    volume_embeddings = analyzer.perform_dimensionality_reduction(bispectrum_scaled)
    
    # Clustering analysis for volume
    volume_clustering = analyzer.perform_clustering_analysis(bispectrum_scaled, volume_labels)
    
    # Print volume clustering results
    print("\nVolume Clustering Results:")
    for method, results in volume_clustering.items():
        print(f"\n{method.title()}:")
        print(f"  Number of clusters found: {results['n_clusters']}")
    
    # Create volume visualizations
    print("\nCreating volume analysis visualizations...")
    
    try:
        vol_clustering_fig = analyzer.create_clustering_visualization(
            volume_embeddings, volume_labels, volume_clustering, 'volume', 'bispectrum_volume'
        )
        if vol_clustering_fig:
            plt.close(vol_clustering_fig)
    except Exception as e:
        print(f"Error creating volume clustering visualization: {str(e)}")
    
    try:
        vol_pca_fig = analyzer.create_pca_analysis(
            bispectrum_scaled, volume_labels, 'volume', 'bispectrum_volume'
        )
        if vol_pca_fig:
            plt.close(vol_pca_fig)
    except Exception as e:
        print(f"Error creating volume PCA analysis: {str(e)}")
    
    # ========== BRAVAIS LATTICE ANALYSIS ==========
    print("\n" + "="*60)
    print("BRAVAIS LATTICE ANALYSIS")
    print("="*60)
    
    # Dimensionality reduction for Bravais analysis
    bravais_embeddings = analyzer.perform_dimensionality_reduction(bispectrum_scaled)
    
    # Clustering analysis for Bravais lattice
    bravais_clustering = analyzer.perform_clustering_analysis(bispectrum_scaled, bravais_labels)
    
    # Print Bravais clustering results
    print("\nBravais Lattice Clustering Results:")
    for method, results in bravais_clustering.items():
        print(f"\n{method.title()}:")
        print(f"  Number of clusters found: {results['n_clusters']}")
    
    # Create Bravais visualizations
    print("\nCreating Bravais lattice analysis visualizations...")
    
    try:
        brav_clustering_fig = analyzer.create_clustering_visualization(
            bravais_embeddings, bravais_labels, bravais_clustering, 'bravais', 'bispectrum_bravais'
        )
        if brav_clustering_fig:
            plt.close(brav_clustering_fig)
    except Exception as e:
        print(f"Error creating Bravais clustering visualization: {str(e)}") 
    
    
    try:
        brav_pca_fig = analyzer.create_pca_analysis(
            bispectrum_scaled, bravais_labels, 'bravais', 'bispectrum_bravais'
        )
        if brav_pca_fig:
            plt.close(brav_pca_fig)
    except Exception as e:
        print(f"Error creating Bravais PCA analysis: {str(e)}")
    
    # ========== COMPARISON ANALYSIS ==========
    print("\n" + "="*60)
    print("COMPARISON ANALYSIS")
    print("="*60)
    
    # Create summary comparison
    try:
        summary_fig, summary_df = analyzer.create_summary_comparison(volume_clustering, bravais_clustering)
        if summary_fig:
            plt.close(summary_fig)
        
        print("\nSummary of Results:")
        print(summary_df.to_string(index=False))
    except Exception as e:
        print(f"Error creating summary comparison: {str(e)}")
        summary_df = pd.DataFrame()  # Empty dataframe as fallback
    
    # Save all results
    results = {
        'volume_labels': volume_labels,
        'bravais_labels': bravais_labels,
        'volumes': volumes,
        'volume_bins': volume_bins,
        'bispectrum_original': bispectrum_array,
        'bispectrum_scaled': bispectrum_scaled,
        'volume_embeddings': volume_embeddings,
        'bravais_embeddings': bravais_embeddings,
        'volume_clustering': volume_clustering,
        'bravais_clustering': bravais_clustering,
        'summary_comparison': summary_df
    }
    
    print("\nSaving analysis results...")
    with open('pickles/bispectrum_clustering_analysis_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Print final summary
    print("\n" + "="*80)
    print("BISPECTRUM CLUSTERING ANALYSIS COMPLETE")
    print("="*80)
    
    print(f"\nDataset Summary:")
    print(f"  Total samples: {len(bispectrum_array)}")
    print(f"  Bispectrum features: {bispectrum_scaled.shape[1]}")
    print(f"  Volume bins: {len(np.unique(volume_labels))}")
    print(f"  Bravais lattice types: {len(np.unique(bravais_labels))}")
    
    print(f"\nClustering Methods Analyzed:")
    print(f"  - K-Means: Standard centroid-based clustering")
    print(f"  - Hierarchical: Agglomerative clustering with Ward linkage")
    print(f"  - DBSCAN: Density-based clustering with noise detection")
    
    print(f"\nDimensionality Reduction Methods:")
    print(f"  - PCA: Linear dimensionality reduction")
    print(f"  - t-SNE: Non-linear, local structure preservation")
    print(f"  - UMAP: Non-linear, global structure preservation")
     
    print(f"\nAll plots saved to 'plots/bispectrum_clustering/' directory")
    print(f"Analysis results saved to 'pickles/bispectrum_clustering_analysis_results.pkl'")


if __name__ == "__main__":
    main()
