import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import umap
import pickle
from tqdm import tqdm
from model import XRDTransformerEncoder
from train import prepare_data
import pandas as pd
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from collections import defaultdict
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


class LatentSpaceAnalyzer:
    """Analyze the latent representations learned by the XRD Transformer"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
        # Storage for activations
        self.activations = {}
        self.hooks = []
        
    def register_hooks(self, layer_names=None):
        """Register forward hooks to capture activations from specific layers"""
        if layer_names is None:
            # default layers to analyze
            layer_names = [
                'tokenizer',
                'self_attn_blocks',
                'ln'
            ]
        
        def get_activation(name):
            def hook(model, input, output):
                # handle different output types
                if isinstance(output, tuple):
                    # For layers that return multiple outputs (like MultiheadAttention)
                    self.activations[name] = output[0].detach()
                elif isinstance(output, torch.Tensor):
                    self.activations[name] = output.detach()
                else:
                    # For other types, try to convert to tensor
                    try:
                        self.activations[name] = torch.tensor(output).detach()
                    except:
                        print(f"Could not capture activation for {name}")
            return hook
        
        # Register hooks
        for name, module in self.model.named_modules():
            if any(layer_name in name for layer_name in layer_names):
                self.hooks.append(module.register_forward_hook(get_activation(name)))
   
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def extract_features_with_structures(self, data_loader, layer_name='ln'):
        """Extract features from a specific layer for all data, including structure information"""
        features = []
        targets = []
        structures = []
        
        self.register_hooks([layer_name])
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Extracting {layer_name} features"):
                if len(batch) == 4:  # Assuming batch includes structures
                    data, target, padding_mask, struct_batch = batch
                    structures.extend(struct_batch)
                elif len(batch) == 3:
                    data, target, padding_mask = batch
                    structures.extend([None] * len(data))  # Placeholder if no structures
                else:
                    data, target = batch
                    padding_mask = None
                    structures.extend([None] * len(data))
                
                data = data.to(self.device).float()
                
                # forward pass - handle both dense and sparse inputs
                if hasattr(self.model, 'input_type') and self.model.input_type == 'sparse': # 'sparse' is under construction; for now use dense only
                    if padding_mask is not None:
                        _ = self.model(data, padding_mask=padding_mask)
                    else:
                        _ = self.model(data)
                else:
                    # Dense input
                    _ = self.model(data)
                
                # get the activation from the specified layer
                if layer_name in self.activations:
                    feat = self.activations[layer_name]
                    if len(feat.shape) > 2: # mean pooling of multiple tokens 
                        # if sequence output, take mean pooling
                        feat = feat.mean(dim=1)
                    features.append(feat.cpu())
                
                targets.append(target.cpu())
        
        self.remove_hooks()
        
        features = torch.cat(features, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()
        
        return features, targets, structures

    def extract_bravais_labels_from_sorted_data(self, sorted_data):
        all_labels = []
        
        for bravais_type, data in sorted_data.items():
            n_samples = len(data['structure'])
            all_labels.extend([bravais_type] * n_samples)
        
        return np.array(all_labels)
    
    def extract_features(self, data_loader, layer_name='ln'):
        """Extract features from a specific layer for all data"""
        features = []
        targets = []
        
        self.register_hooks([layer_name])
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Extracting {layer_name} features"):
                if len(batch) == 3:
                    data, target, padding_mask = batch
                else:
                    data, target = batch
                    padding_mask = None
                
                data = data.to(self.device).float()
                
                # forward pass - handle both dense and sparse inputs
                if hasattr(self.model, 'input_type') and self.model.input_type == 'sparse':
                    if padding_mask is not None:
                        _ = self.model(data, padding_mask=padding_mask)
                    else:
                        _ = self.model(data)
                else:
                    # Dense input
                    _ = self.model(data)
                
                # Get the activation from the specified layer
                if layer_name in self.activations:
                    feat = self.activations[layer_name]
                    if len(feat.shape) > 2:
                        # If sequence output, take mean pooling
                        feat = feat.mean(dim=1)
                    features.append(feat.cpu())
                
                targets.append(target.cpu())
        
        self.remove_hooks()
        
        features = torch.cat(features, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()
        
        return features, targets


def extract_cell_volume_labels(ordered_result, n_bins=10):
    """Extract cell volumes from ordered_result and create volume bins"""
    volumes = []
    
    for result in ordered_result:
        crystal = result['crystal']
        # Get unit cell volume
        volume = crystal.lattice.volume
        volumes.append(volume)
    
    volumes = np.array(volumes)
    
    # Create volume bins for analysis
    # Using quantile-based binning for more balanced groups
    volume_percentiles = np.linspace(0, 100, n_bins + 1)
    volume_bins = np.percentile(volumes, volume_percentiles)
    volume_labels = np.digitize(volumes, volume_bins[1:-1])  # Remove first and last bin edges
    
    # Create descriptive labels with zero-padded numbers for proper sorting
    volume_bin_labels = []
    for i, vol in enumerate(volumes):
        bin_idx = volume_labels[i]
        if bin_idx == 0:
            label = f"Bin {1:02d} (< {volume_bins[1]:.0f} Å³)"
        elif bin_idx == n_bins - 1:
            label = f"Bin {n_bins:02d} (> {volume_bins[-2]:.0f} Å³)"
        else:
            label = f"Bin {bin_idx + 1:02d} ({volume_bins[bin_idx]:.0f}-{volume_bins[bin_idx + 1]:.0f} Å³)"
        volume_bin_labels.append(label)
    
    return np.array(volume_bin_labels), volumes, volume_bins

def extract_bravais_lattice_labels_from_ordered_result(ordered_result):
    """Extract crystal system labels from ordered_result based on space group numbers"""
    crystal_system_labels = []
    
    for result in ordered_result:
        crystal = result['crystal']
        # Get space group and convert to crystal system
        try:
            sga = SpacegroupAnalyzer(crystal)
            space_group_number = sga.get_space_group_number()
            
            # Map space group number to crystal system
            if space_group_number <= 2:  # Space groups 1-2
                crystal_system = "Triclinic"
            elif space_group_number <= 15:  # Space groups 3-15
                crystal_system = "Monoclinic"
            elif space_group_number <= 74:  # Space groups 16-74
                crystal_system = "Orthorhombic"
            elif space_group_number <= 142:  # Space groups 75-142
                crystal_system = "Tetragonal"
            elif space_group_number <= 167:  # Space groups 143-167
                crystal_system = "Trigonal"
            elif space_group_number <= 194:  # Space groups 168-194
                crystal_system = "Hexagonal"
            else:  # Space groups 195-230
                crystal_system = "Cubic"
                
            crystal_system_labels.append(crystal_system)
            
        except Exception as e:
            print(f"Warning: Could not determine crystal system for structure: {e}")
            crystal_system_labels.append("Unknown")
    
    return np.array(crystal_system_labels)


def compute_embeddings(features, method='tsne'):
    """Compute 2D embeddings using specified method"""
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4))
        return reducer.fit_transform(features)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(features)//4), min_dist=0.1)
        return reducer.fit_transform(features)
    elif method == 'pca':
        reducer = PCA(n_components=2)
        return reducer.fit_transform(features)
    else:
        raise ValueError(f"Unknown method: {method}")


def create_comprehensive_plot(features_dict, labels, title_prefix, filename_prefix, colors=None):
    """Create a comprehensive 5x3 plot with different layers and embedding methods"""
    
    # Define the layers and methods
    layer_names = list(features_dict.keys())
    methods = ['tsne', 'umap', 'pca']
    method_titles = ['t-SNE', 'UMAP', 'PCA']
    
    # Create figure
    fig, axes = plt.subplots(5, 3, figsize=(18, 25))
    fig.suptitle(f'{title_prefix} - Latent Space Analysis Across Layers', fontsize=20, y=0.98)
    
    # Get unique labels and colors
    unique_labels = np.unique(labels)
    if colors is None:
        if len(unique_labels) <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    # Process each layer
    for row, layer_name in enumerate(layer_names):
        features = features_dict[layer_name]
        
        # Get layer display name
        if 'tokenizer' in layer_name:
            layer_display = 'Tokenization'
        elif 'self_attn_blocks.0' in layer_name:
            layer_display = 'Transformer First'
        elif 'self_attn_blocks' in layer_name and 'self_attn_blocks.0' not in layer_name:
            # Check if this is middle or last based on the reordered list
            if row == 2:  # Third position after reordering
                layer_display = 'Transformer Middle'
            elif row == 3:  # Fourth position after reordering  
                layer_display = 'Transformer Last'
            else:
                layer_display = 'Transformer Layer'
        elif 'ln' in layer_name:
            layer_display = 'Layer Norm Final'
        else:
            layer_display = layer_name.replace('_', ' ').title()
        
        # Process each embedding method
        for col, (method, method_title) in enumerate(zip(methods, method_titles)):
            ax = axes[row, col]
            
            try:
                # Compute embedding
                print(f"Computing {method_title} for {layer_display}...")
                embedded_features = compute_embeddings(features, method)
                
                # Plot each label group
                for i, label in enumerate(unique_labels):
                    mask = labels == label
                    ax.scatter(embedded_features[mask, 0], embedded_features[mask, 1], 
                             c=[colors[i]], label=label, alpha=0.6, s=1)
                
                # Formatting
                ax.set_xlabel(f'{method_title} Component 1', fontsize=10)
                ax.set_ylabel(f'{method_title} Component 2', fontsize=10)
                ax.set_title(f'{layer_display} - {method_title}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Add legend only to the rightmost column
                if col == 2:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                
            except Exception as e:
                print(f"Error processing {layer_display} with {method_title}: {str(e)}")
                ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=10)
                ax.set_title(f'{layer_display} - {method_title} (Error)', fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    plt.savefig(f'plots/latent/{filename_prefix}_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_knn_accuracy_comparison(features_dict, volume_labels, crystal_system_labels):
    """Create KNN accuracy comparison plot across layers and embedding methods"""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder
    
    # Prepare labels
    le_volume = LabelEncoder()
    le_crystal = LabelEncoder()
    numeric_volume = le_volume.fit_transform(volume_labels)
    numeric_crystal = le_crystal.fit_transform(crystal_system_labels)
    
    layers = list(features_dict.keys())
    methods = ['tsne', 'umap', 'pca']
    method_titles = ['t-SNE: Knn Accuracy', 'UMAP: Knn Accuracy', 'PCA: Knn Accuracy']
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Layer display names for x-axis
    layer_display_names = []
    for layer_name in layers:
        if 'tokenizer' in layer_name:
            layer_display_names.append('tokenizer')
        elif 'self_attn_blocks.' in layer_name:
            # Extract block number
            block_num = layer_name.split('self_attn_blocks.')[1].split('.')[0]
            layer_display_names.append(f'Block_{block_num}')
        elif 'ln' in layer_name:
            layer_display_names.append('ln')
        else:
            layer_display_names.append(layer_name.replace('.', '_'))
    
    # Compute accuracies for each method
    for method_idx, (method, method_title) in enumerate(zip(methods, method_titles)):
        ax = axes[method_idx]
        
        volume_accuracies = []
        crystal_accuracies = []
        
        for layer_name in layers:
            features = features_dict[layer_name]
            
            # Compute embeddings
            try:
                embedded_features = compute_embeddings(features, method)
                
                # KNN classifier
                knn = KNeighborsClassifier(n_neighbors=5)
                
                # Volume accuracy
                vol_acc = cross_val_score(knn, embedded_features, numeric_volume, cv=5).mean()
                volume_accuracies.append(vol_acc)
                
                # Crystal system accuracy
                crystal_acc = cross_val_score(knn, embedded_features, numeric_crystal, cv=5).mean()
                crystal_accuracies.append(crystal_acc)
                
            except Exception as e:
                print(f"Error computing {method} for {layer_name}: {e}")
                volume_accuracies.append(0)
                crystal_accuracies.append(0)
        
        # Plot bars
        x = np.arange(len(layers))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, volume_accuracies, width, label='Volume (10 bins)', 
                      color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, crystal_accuracies, width, label='Bravais Lattice', 
                      color='coral', alpha=0.8)
        
        # Formatting
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Knn Accuracy', fontsize=12)
        ax.set_title(method_title, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(layer_display_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 0.85)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('plots/latent/knn_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_volume_crystal_system_correlation(crystal_system_labels, volumes):
    """Analyze correlation between crystal systems and cell volumes"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Box plot of volumes by crystal system
    unique_crystal_systems = np.unique(crystal_system_labels)
    volume_data = [volumes[crystal_system_labels == system] for system in unique_crystal_systems]
    
    ax1.boxplot(volume_data, labels=unique_crystal_systems)
    ax1.set_xlabel('Crystal System', fontsize=12)
    ax1.set_ylabel('Cell Volume (ų)', fontsize=12)
    ax1.set_title('Cell Volume Distribution by Crystal System', fontsize=14)
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_yscale('log')  # Log scale for better visualization
    
    # Violin plot for better distribution visualization
    parts = ax2.violinplot(volume_data, positions=range(1, len(unique_crystal_systems) + 1), 
                           showmeans=True, showmedians=True)
    ax2.set_xticks(range(1, len(unique_crystal_systems) + 1))
    ax2.set_xticklabels(unique_crystal_systems, rotation=45)
    ax2.set_xlabel('Crystal System', fontsize=12)
    ax2.set_ylabel('Cell Volume (ų)', fontsize=12)
    ax2.set_title('Cell Volume Distribution by Crystal System (Violin)', fontsize=14)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('plots/latent/volume_crystal_system_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistical analysis
    print("\nVolume Statistics by Crystal System:")
    for system in unique_crystal_systems:
        mask = crystal_system_labels == system
        vols = volumes[mask]
        print(f"{system:12s}: Mean={np.mean(vols):8.1f}, Median={np.median(vols):8.1f}, "
              f"Std={np.std(vols):8.1f}, Count={len(vols):4d}")


def main():
    """Enhanced main analysis pipeline focusing on comprehensive layer-wise analysis"""
    
    # Load model
    with open('model_data/transformer_sixteen_32.pt', 'rb') as f:
        model = pickle.load(f)

    # Load validation results (from preprocess function)
    with open('pickles/val_results.pkl', 'rb') as f:
        ordered_result = pickle.load(f)
    
    print(f"Loaded {len(ordered_result)} validation samples")

    # Load data for model inference
    with open('pickles/mp20_sim_xrd.pkl', 'rb') as f:
        sim_xrd = pickle.load(f)

    with open('pickles/mp20_bispec_lat.pkl', 'rb') as f:
        bispec_list = pickle.load(f)
    
    with open('pickles/mp20_sim_xrd_val.pkl', 'rb') as f:
        sim_xrd_val = pickle.load(f)
    
    with open('pickles/mp20_bispec_lat_val.pkl', 'rb') as f:
        bispec_list_val = pickle.load(f) 
     
    print("\nExtracting crystal system labels...")
    bravais_labels = extract_bravais_lattice_labels_from_ordered_result(ordered_result)
    
    print("\nExtracting cell volume labels...")
    volume_bin_labels, volumes, volume_bins = extract_cell_volume_labels(ordered_result, n_bins=10)
    
    print("\nCrystal System Distribution:")
    unique_bravais, bravais_counts = np.unique(bravais_labels, return_counts=True)
    for btype, count in zip(unique_bravais, bravais_counts):
        percentage = (count / len(bravais_labels)) * 100
        print(f"  {btype}: {count} samples ({percentage:.1f}%)")
    
    print("\nCell Volume Bin Distribution:")
    unique_volumes, volume_counts = np.unique(volume_bin_labels, return_counts=True)
    for vtype, count in zip(unique_volumes, volume_counts):
        percentage = (count / len(volume_bin_labels)) * 100
        print(f"  {vtype}: {count} samples ({percentage:.1f}%)")
    
    # Prepare data loaders
    train_loader, val_loader = prepare_data(
        sim_xrd, bispec_list,  # original training data
        sim_xrd_val, bispec_list_val,  # full validation set
        batch_size=32, 
        is_normalized=False, 
        indices=False,
        d_model=384  # Adjust based on your model
    ) 
    
    # Initialize analyzer
    analyzer = LatentSpaceAnalyzer(model)
    
    # Define layers to analyze (5 layers as requested)
    layers_to_analyze = []
    
    # Add tokenizer output if it exists
    if hasattr(model, 'tokenizer'):
        layers_to_analyze.append('tokenizer')
    elif hasattr(model, 'lin_xrd'):
        layers_to_analyze.append('lin_xrd')
    
    # Add specific self-attention blocks in the correct order
    if hasattr(model, 'self_attn_blocks'):
        num_blocks = len(model.self_attn_blocks)
        if num_blocks > 0:
            layers_to_analyze.append('self_attn_blocks.0')  # First block
        if num_blocks > 1:
            layers_to_analyze.append(f'self_attn_blocks.{num_blocks-1}')  # Last block
        if num_blocks > 2:
            layers_to_analyze.append(f'self_attn_blocks.{(num_blocks-1)//2}')  # Middle block

    # Add final layer norm
    layers_to_analyze.append('ln')
    
    # Reorder to: tokenization, first, middle, last, final
    # Current order is: tokenization, first, last, middle, final - need to swap middle and last
    if len(layers_to_analyze) >= 4:
        # Swap positions 2 and 3 (last and middle transformer layers)
        layers_to_analyze[2], layers_to_analyze[3] = layers_to_analyze[3], layers_to_analyze[2]
    
    # Ensure we have exactly 5 layers
    layers_to_analyze = layers_to_analyze[:5]
    
    print(f"\nAnalyzing {len(layers_to_analyze)} layers: {layers_to_analyze}")
    
    # Extract features from all layers
    features_dict = {}
    for layer in layers_to_analyze:
        print(f"\nExtracting features from layer: {layer}")
        try:
            features, targets = analyzer.extract_features(val_loader, layer)
            features_dict[layer] = features
            print(f"Successfully extracted features from {layer}: shape {features.shape}")
        except Exception as e:
            print(f"Warning: Could not extract features from layer {layer}: {str(e)}")
            continue
    
    print(f"\nSuccessfully extracted features from {len(features_dict)} layers")
    
    # Create comprehensive plots
    print("\nCreating comprehensive crystal system analysis plot...")
    bravais_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_bravais)))
    create_comprehensive_plot(
        features_dict, bravais_labels, 
        'Crystal System Types', 'crystal_system',
        colors=bravais_colors
    )
    
    print("\nCreating comprehensive volume analysis plot...")
    volume_colors = plt.cm.viridis(np.linspace(0, 1, len(unique_volumes)))
    create_comprehensive_plot(
        features_dict, volume_bin_labels, 
        'Cell Volume Analysis', 'cell_volume',
        colors=volume_colors
    )
    
    # Generate the volume-crystal system correlation plot
    print("\nAnalyzing volume-crystal system correlation...")
    crystal_system_labels = []
    for result in ordered_result:
        crystal = result['crystal']
        try:
            sga = SpacegroupAnalyzer(crystal)
            space_group_number = sga.get_space_group_number()
            
            # Map space group number to crystal system
            if space_group_number <= 2:
                crystal_system = "Triclinic"
            elif space_group_number <= 15:
                crystal_system = "Monoclinic"
            elif space_group_number <= 74:
                crystal_system = "Orthorhombic"
            elif space_group_number <= 142:
                crystal_system = "Tetragonal"
            elif space_group_number <= 167:
                crystal_system = "Trigonal"
            elif space_group_number <= 194:
                crystal_system = "Hexagonal"
            else:
                crystal_system = "Cubic"
                
            crystal_system_labels.append(crystal_system)
            
        except Exception as e:
            print(f"Warning: Could not determine crystal system for structure: {e}")
            crystal_system_labels.append("Unknown")
    
    crystal_system_labels = np.array(crystal_system_labels)
    analyze_volume_crystal_system_correlation(crystal_system_labels, volumes)
    
    # Generate KNN accuracy comparison plot
    print("\nCreating KNN accuracy comparison plot...")
    create_knn_accuracy_comparison(features_dict, volume_bin_labels, bravais_labels)
    
    # Save analysis results
    analysis_results = {
        'crystal_system_labels': bravais_labels,
        'volume_bin_labels': volume_bin_labels,
        'crystal_system_labels_original': crystal_system_labels,
        'volumes': volumes,
        'volume_bins': volume_bins,
        'features_dict': features_dict,
        'layers_analyzed': layers_to_analyze
    }
    
    with open('pickles/comprehensive_latent_analysis_results.pkl', 'wb') as f:
        pickle.dump(analysis_results, f)
    
    print("\nAnalysis complete! Generated plots:")
    print("  - plots/latent/crystal_system_comprehensive_analysis.png")
    print("  - plots/latent/cell_volume_comprehensive_analysis.png")
    print("  - plots/latent/volume_crystal_system_correlation.png")
    print("  - plots/latent/knn_accuracy_comparison.png")
    print(f"Analysis results saved to 'pickles/comprehensive_latent_analysis_results.pkl'")
    print(f"Total samples analyzed: {len(list(features_dict.values())[0])}")
    print(f"Layers analyzed: {len(features_dict)}")
    print(f"Number of crystal system types: {len(unique_bravais)}")
    print(f"Number of volume bins: {len(unique_volumes)}")


if __name__ == "__main__": 
    main()
