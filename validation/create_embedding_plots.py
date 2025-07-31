from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import logging

def create_embedding_plots(embeddings, labels, selected_indices, method_name, output_dir):
    """
    Create embedding plots showing clustering, selected compounds, and cluster size distribution.
    
    Args:
        embeddings: 2D embedding coordinates
        labels: Cluster labels (optional)
        selected_indices: Indices of selected compounds
        method_name: Name of the method
        output_dir: Directory to save plots
    """
    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set up the plot style
    plt.style.use('default')
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Clustered embeddings (if labels available)
    if labels is not None:
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels != -1])
        scatter = axes[0].scatter(embeddings[:, 0], embeddings[:, 1], 
                                c=labels, alpha=0.6, s=20, cmap='tab10')
        axes[0].set_title(f"After Clustering ({n_clusters} clusters)")
        if n_clusters <= 10:
            plt.colorbar(scatter, ax=axes[0], label='Cluster')
    else:
        axes[0].scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.6, s=20, c='blue')
        axes[0].set_title("No Clustering Info Available")
    axes[0].set_xlabel("Component 1")
    axes[0].set_ylabel("Component 2")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Selected compounds highlighted
    axes[1].scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.3, s=20, 
                   color='lightgray', label='Unselected')
    if len(selected_indices) > 0:
        axes[1].scatter(embeddings[selected_indices, 0], embeddings[selected_indices, 1], 
                       alpha=0.8, s=50, color='red', label='Selected')
    axes[1].set_title(f"Selected Compounds ({len(selected_indices)} selected)")
    axes[1].set_xlabel("Component 1")
    axes[1].set_ylabel("Component 2")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Cluster size distribution (if labels available)
    if labels is not None:
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        if len(unique_labels) > 0:
            n_clusters = len(unique_labels)
            
            # Use KDE for many clusters (>50), otherwise bar plot
            if n_clusters > 50:
                # Create KDE plot for cluster sizes
                from scipy.stats import gaussian_kde
                if len(counts) > 1:  # Need at least 2 points for KDE
                    kde = gaussian_kde(counts)
                    x_range = np.linspace(counts.min(), counts.max(), 200)
                    kde_values = kde(x_range)
                    axes[2].fill_between(x_range, kde_values, alpha=0.7, color='skyblue')
                    axes[2].plot(x_range, kde_values, color='darkblue', linewidth=2)
                    axes[2].set_xlabel("Cluster Size")
                    axes[2].set_ylabel("Density")
                else:
                    # Single cluster size - show as vertical line
                    axes[2].axvline(counts[0], color='darkblue', linewidth=3)
                    axes[2].set_xlabel("Cluster Size")
                    axes[2].set_ylabel("Density")
                axes[2].set_title(f"Cluster Size Distribution (KDE, {n_clusters} clusters)")
            else:
                # Use bar plot for fewer clusters
                axes[2].bar(range(len(unique_labels)), counts, color='skyblue', edgecolor='darkblue')
                axes[2].set_xlabel("Cluster ID")
                axes[2].set_ylabel("Number of Compounds")
                axes[2].set_title(f"Cluster Size Distribution ({n_clusters} clusters)")
                
                # Set reasonable x-axis ticks for readability
                if n_clusters <= 20:
                    axes[2].set_xticks(range(0, n_clusters, max(1, n_clusters // 10)))
                else:
                    axes[2].set_xticks(range(0, n_clusters, n_clusters // 10))
            
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, "No valid clusters", ha='center', va='center', 
                        transform=axes[2].transAxes)
            axes[2].set_title("Cluster Size Distribution")
    else:
        axes[2].text(0.5, 0.5, "No clustering information", ha='center', va='center', 
                    transform=axes[2].transAxes)
        axes[2].set_title("Cluster Size Distribution")
    
    plt.suptitle(f"{method_name} - Analysis ({len(embeddings)} compounds)", fontsize=16)
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / f"{method_name.lower()}_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {plot_file}")
    
    plt.show()

print("✅ Visualization function defined")