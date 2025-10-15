import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import plotly.graph_objects as go
import networkx as nx
from networkx.algorithms import community as nx_community

from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
from matplotlib.animation import FuncAnimation, PillowWriter


# =====================================================================================
# 4. Visualization and Analysis Functions
# =====================================================================================


def plot_sankey_evolution(epoch_snapshots, num_clusters, seed, max_nodes_to_plot=1000):
    if len(epoch_snapshots) < 2: return
    print(f"\nGenerating Sankey plot for seed {seed}...")
    print("  - Processing node assignment data...")
    epochs = sorted(epoch_snapshots.keys())
    all_nodes = set().union(*(data['node_ids'].numpy() for data in epoch_snapshots.values()))
    df = pd.DataFrame(index=sorted(list(all_nodes)))
    for epoch, data in epoch_snapshots.items():
        epoch_map = dict(zip(data['node_ids'].numpy(), data['assignments_l1'].numpy()))
        df[f'epoch_{epoch}'] = df.index.map(epoch_map)
    df = df.fillna(-1).astype(int)
    if len(df) > max_nodes_to_plot:
        print(f"  - Visualizing a random sample of {max_nodes_to_plot} out of {len(df)} nodes for speed.")
        df_sampled = df.sample(n=max_nodes_to_plot, random_state=seed)
    else:
        df_sampled = df
    labels, source, target, value = [], [], [], []
    label_map, current_label_index = {}, 0
    for col in df_sampled.columns:
        epoch_num = col.split('_')[-1]
        for i in range(num_clusters):
            label_name = f"E{epoch_num} C{i}"
            labels.append(label_name)
            label_map[(col, i)] = current_label_index
            current_label_index += 1
    for i in range(len(epochs) - 1):
        epoch_from, epoch_to = f'epoch_{epochs[i]}', f'epoch_{epochs[i+1]}'
        flows = df_sampled.groupby([epoch_from, epoch_to]).size().reset_index(name='count')
        for _, row in flows.iterrows():
            from_cluster, to_cluster = int(row[epoch_from]), int(row[epoch_to])
            if from_cluster == -1 or to_cluster == -1: continue
            source.append(label_map[(epoch_from, from_cluster)])
            target.append(label_map[(epoch_to, to_cluster)])
            value.append(row['count'])
    print("  - Data processing complete. Generating plot file...")
    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels, color="blue"),
        link=dict(source=source, target=target, value=value))])
    fig.update_layout(title_text=f"Cluster Assignment Evolution (Sample of {len(df_sampled)} Nodes) (Seed {seed})", font_size=10)
    plot_filename = f"icews18_sankey_evolution_seed_{seed}.html"
    fig.write_html(plot_filename)
    print(f"Sankey plot saved to {plot_filename}")

def plot_hierarchical_snapshots(epoch_snapshots, seed, perplexity=30):
    if not epoch_snapshots: return
    print(f"\nGenerating hierarchical t-SNE snapshot plot for seed {seed}...")
    
    epochs = sorted(epoch_snapshots.keys())
    final_trained_epoch = max([e for e in epochs if e > 0], default=None)
    if not final_trained_epoch: 
        print("  - No trained epochs found to create a stable layout. Skipping."); return

    # --- Create a stable t-SNE layout based on the FINAL node embeddings ---
    final_epoch_data = epoch_snapshots[final_trained_epoch]
    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(final_epoch_data['embeddings_final']) - 1), 
                max_iter=1000, random_state=seed, init='pca', learning_rate='auto')
    stable_layout = tsne.fit_transform(final_epoch_data['embeddings_final'].numpy())
    node_to_pos = {node_id.item(): pos for node_id, pos in zip(final_epoch_data['node_ids'], stable_layout)}

    # --- Determine the structure of our plot grid ---
    has_l1 = 'assignments_l1' in final_epoch_data
    has_l2 = 'assignments_l2' in final_epoch_data
    num_cols = 1 + int(has_l1) + int(has_l2)
    num_rows = len(epochs)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(7 * num_cols, 6 * num_rows), squeeze=False)
    
    for i, epoch in enumerate(epochs):
        epoch_data = epoch_snapshots[epoch]
        positions = np.array([node_to_pos.get(nid.item(), [np.nan, np.nan]) for nid in epoch_data['node_ids']])
        valid_mask = ~np.isnan(positions).any(axis=1)

        # COLUMN 1: Raw Embeddings (z_tgn), colored by Level 1 clusters
        ax = axes[i, 0]
        if has_l1:
            num_clusters_l1 = epoch_data['assignments_l1'].max().item() + 1
            colors_l1 = plt.cm.nipy_spectral(np.linspace(0, 1, num_clusters_l1))
            node_colors = colors_l1[epoch_data['assignments_l1'].numpy()]
            ax.scatter(positions[valid_mask, 0], positions[valid_mask, 1], c=node_colors[valid_mask], s=10, alpha=0.7)
        else:
            ax.scatter(positions[valid_mask, 0], positions[valid_mask, 1], s=10, alpha=0.7)
        
        title = f"Epoch {epoch}\nRaw Embeddings (Colored by L1)"
        ax.set_title(title, fontsize=14)
        ax.set_xticks([]); ax.set_yticks([])

        # COLUMN 2: Raw Embeddings, colored by Level 2 clusters
        if has_l1:
            ax = axes[i, 1]
            if has_l2:
                num_clusters_l2 = epoch_data['assignments_l2'].max().item() + 1
                colors_l2 = plt.cm.tab20(np.linspace(0, 1, num_clusters_l2))
                node_colors = colors_l2[epoch_data['assignments_l2'].numpy()]
                ax.scatter(positions[valid_mask, 0], positions[valid_mask, 1], c=node_colors[valid_mask], s=10, alpha=0.7)
                title = f"Epoch {epoch}\nRaw Embeddings (Colored by L2)"
            else:
                # If only one level, just show the L1 coloring again
                ax.scatter(positions[valid_mask, 0], positions[valid_mask, 1], c=colors_l1[epoch_data['assignments_l1'].numpy()][valid_mask], s=10, alpha=0.7)
                title = f"Epoch {epoch}\nLevel 1 Assignments"
            ax.set_title(title, fontsize=14)
            ax.set_xticks([]); ax.set_yticks([])

        # COLUMN 3: Level 2 assignments (if they exist)
        if has_l2:
             ax = axes[i, 2]
             ax.scatter(positions[valid_mask, 0], positions[valid_mask, 1], c=colors_l2[epoch_data['assignments_l2'].numpy()][valid_mask], s=10, alpha=0.7)
             ax.set_title(f"Epoch {epoch}\nLevel 2 Assignments", fontsize=14)
             ax.set_xticks([]); ax.set_yticks([])


    fig.suptitle(f'Hierarchical Temporal Clustering Evolution (Seed {seed})', fontsize=22, y=1.0)
    plt.tight_layout()
    plot_filename = f"icews18_hierarchical_snapshots_seed_{seed}.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"Hierarchical snapshot plot saved to {plot_filename}")
    plt.close(fig)


# Add these two new functions to your script's visualization section

def plot_utilization_heatmap(snapshots, num_clusters, seed):
    """
    Creates a heatmap showing the utilization of each cluster over epochs.
    """
    print(f"\nGenerating cluster utilization heatmap for seed {seed}...")
    epochs = sorted(snapshots.keys())
    
    # Collect the average utilization vectors from each snapshot
    utilization_data = []
    for epoch in epochs:
        if 'avg_utilization' in snapshots[epoch]:
            utilization_data.append(snapshots[epoch]['avg_utilization'].numpy())
    
    if not utilization_data:
        print("  - No utilization data captured. Skipping heatmap.")
        return

    # Stack into a (num_epochs, num_clusters) matrix
    utilization_matrix = np.stack(utilization_data, axis=0)

    # To make it readable, sort clusters by their utilization in the final epoch
    final_utilization = utilization_matrix[-1, :]
    sorted_indices = np.argsort(final_utilization)[::-1] # Sort descending
    
    # Only plot the top N most utilized clusters for clarity
    top_k = min(100, num_clusters) 
    
    sorted_matrix = utilization_matrix[:, sorted_indices[:top_k]]

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(sorted_matrix.T, cmap='viridis', aspect='auto', interpolation='nearest')

    ax.set_yticks(np.arange(top_k))
    ax.set_yticklabels([f'C{i}' for i in sorted_indices[:top_k]])
    ax.set_xticks(np.arange(len(epochs)))
    ax.set_xticklabels([f"E{e}" for e in epochs])
    
    plt.colorbar(im, ax=ax, label='Average Cluster Utilization')
    ax.set_title(f'Top {top_k} Cluster Utilization Heatmap (Seed {seed})', fontsize=16)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cluster ID (Sorted by Final Utilization)')
    
    plot_filename = f"icews18_utilization_heatmap_seed_{seed}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Utilization heatmap saved to {plot_filename}")
    plt.close(fig)


def plot_assignment_switch_rate(snapshots, seed):
    """
    Calculates and plots the percentage of nodes that change cluster assignment
    between consecutive captured epochs.
    """
    print(f"\nGenerating assignment switch-rate curve for seed {seed}...")
    epochs = sorted(snapshots.keys())
    switch_rates = []
    switch_epochs = []

    for i in range(len(epochs) - 1):
        epoch1, epoch2 = epochs[i], epochs[i+1]
        
        # Skip comparing the random initial state if it's too different
        if epoch1 == 0: continue

        snap1, snap2 = snapshots[epoch1], snapshots[epoch2]
        
        index1 = snap1['node_ids'].cpu().numpy()
        data1 = snap1['assignments_l1'].cpu().numpy() 
        
        index2 = snap2['node_ids'].cpu().numpy()
        data2 = snap2['assignments_l1'].cpu().numpy()
    
        
        df1 = pd.DataFrame({'assignment1': data1}, index=index1)
        df2 = pd.DataFrame({'assignment2': data2}, index=index2)
        
        # Find nodes common to both snapshots
        merged_df = df1.join(df2, how='inner')
        
        if merged_df.empty:
            continue

        # Calculate the number of nodes that switched clusters
        switches = (merged_df['assignment1'] != merged_df['assignment2']).sum()
        total_common_nodes = len(merged_df)
        
        switch_rate = (switches / total_common_nodes) * 100
        switch_rates.append(switch_rate)
        switch_epochs.append(epoch2)

    if not switch_rates:
        print("  - Not enough data to calculate switch rates. Skipping plot.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    plt.plot(switch_epochs, switch_rates, marker='o', linestyle='-')
    
    plt.title(f'Node Assignment Switch Rate (Seed {seed})', fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel('Nodes Changing Cluster (%)')
    plt.xticks(switch_epochs)
    plt.ylim(bottom=0)
    plt.grid(True, which='both', linestyle='--')
    
    plot_filename = f"icews18_switch_rate_seed_{seed}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Switch-rate curve saved to {plot_filename}")
    plt.close()


# +++ NEW GEOPOLITICAL ANALYSIS FUNCTIONS +++
def plot_animated_trajectories(epoch_snapshots, entity_mapping, seed, countries_to_track=None, filename="icews18_animated_trajectories.gif"):
    """
    Generates an ANIMATED plot showing the true movement of countries over time.
    It works by:
    1. Generating a t-SNE layout for each epoch.
    2. Using Procrustes analysis to align each layout to a common reference frame.
    3. Rendering the aligned layouts as frames in an animation (GIF).
    """
    print(f"\nGenerating animated trajectory plot for seed {seed}...")
    
    # --- 1. Basic Setup ---
    if countries_to_track is None:
        countries_to_track = ["United States", "China", "Russia", "India", "Iran", "United Kingdom"]
    
    name_to_id = {v: k for k, v in entity_mapping.items()}
    track_ids = {name: name_to_id.get(name) for name in countries_to_track if name_to_id.get(name) is not None}
    
    if not track_ids:
        print("  - Could not find any specified countries. Skipping animation.")
        return

    epochs = sorted([e for e in epoch_snapshots.keys() if e > 0])
    
    # --- 2. Generate an independent t-SNE layout for each epoch ---
    print("  - Calculating t-SNE layout for each epoch...")
    layouts = {}
    node_maps = {}
    for epoch in epochs:
        snapshot = epoch_snapshots[epoch]
        embeddings = snapshot['embeddings_final'].numpy()
        
        perplexity = min(30, len(embeddings) - 1)
        if perplexity <= 0: 
            continue
            
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, init='pca', learning_rate='auto')
        layouts[epoch] = tsne.fit_transform(embeddings)
        node_maps[epoch] = {node_id.item(): i for i, node_id in enumerate(snapshot['node_ids'])}

    # --- 3. Align layouts sequentially using Procrustes Analysis ---
    print("  - Aligning epoch layouts using Procrustes analysis...")
    aligned_layouts = {}
    
    final_epoch = epochs[-1]
    aligned_layouts[final_epoch] = layouts[final_epoch]
    
    # Iterate backwards, aligning each epoch to the (already aligned) next one
    for i in range(len(epochs) - 2, -1, -1):
        epoch_curr = epochs[i]
        epoch_next = epochs[i+1]
        
        map_curr = node_maps[epoch_curr]
        map_next = node_maps[epoch_next]
        common_nodes = list(set(map_curr.keys()) & set(map_next.keys()))
        
        if len(common_nodes) < 3:
            print(f"  - Warning: Only {len(common_nodes)} common nodes between epochs {epoch_curr} and {epoch_next}. Using original layout.")
            aligned_layouts[epoch_curr] = layouts[epoch_curr]
            continue
        
        # Get positions of common nodes in both layouts
        curr_indices = [map_curr[nid] for nid in common_nodes]
        next_indices = [map_next[nid] for nid in common_nodes]
        
        pts_curr = layouts[epoch_curr][curr_indices]
        pts_next = aligned_layouts[epoch_next][next_indices]
        
        # Center both point sets
        pts_curr_mean = pts_curr.mean(axis=0)
        pts_next_mean = pts_next.mean(axis=0)
        
        pts_curr_centered = pts_curr - pts_curr_mean
        pts_next_centered = pts_next - pts_next_mean
        
        # Find optimal rotation
        R, _ = orthogonal_procrustes(pts_curr_centered, pts_next_centered)
        
        # Find scale factor
        scale = np.sqrt((pts_next_centered**2).sum() / (pts_curr_centered**2).sum())
        
        # Apply transformation to ALL points in current epoch
        layout_centered = layouts[epoch_curr] - pts_curr_mean
        layout_transformed = scale * (layout_centered @ R) + pts_next_mean
        
        aligned_layouts[epoch_curr] = layout_transformed

    # Safety check
    if not aligned_layouts:
        print("  - Warning: No layouts could be aligned. Skipping animation.")
        return

    # --- 4. Set up the animation plot ---
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Find global bounds for all aligned layouts to keep the axes stable
    min_x, max_x, min_y, max_y = (np.inf, -np.inf, np.inf, -np.inf)
    for layout in aligned_layouts.values():
        min_x = min(min_x, layout[:, 0].min())
        max_x = max(max_x, layout[:, 0].max())
        min_y = min(min_y, layout[:, 1].min())
        max_y = max(max_y, layout[:, 1].max())
    
    ax.set_xlim(min_x - 5, max_x + 5)
    ax.set_ylim(min_y - 5, max_y + 5)
    
    # Initial plot objects that will be updated each frame
    background_scatter = ax.scatter([], [], c='lightgray', s=10, alpha=0.3)
    country_scatters = {name: ax.scatter([], [], s=200, label=name, edgecolors='black') for name in track_ids.keys()}
    epoch_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=16, fontweight='bold')
    
    # --- 5. Define the animation update function ---
    def update(frame):
        epoch = epochs[frame]
        
        # Safety check
        if epoch not in aligned_layouts:
            return list(country_scatters.values()) + [background_scatter, epoch_text]
        
        layout = aligned_layouts[epoch]
        node_map = node_maps[epoch]
        
        # Update background points
        background_scatter.set_offsets(layout)
        
        # Update country points
        for name, country_id in track_ids.items():
            if country_id in node_map:
                idx = node_map[country_id]
                country_scatters[name].set_offsets(layout[idx])
        
        epoch_text.set_text(f'Epoch: {epoch}')
        return list(country_scatters.values()) + [background_scatter, epoch_text]

    # --- 6. Create and save the animation ---
    print(f"  - Rendering animation frames... This may take a few minutes.")
    ani = FuncAnimation(fig, update, frames=len(epochs), blit=True, interval=500)
    ax.legend()
    ax.set_title("Animated Geopolitical Trajectories", fontsize=18)
    
    # Use PillowWriter to save as a GIF
    writer = PillowWriter(fps=2)
    ani.save(filename, writer=writer)
    
    print(f"Animation saved to {filename}")
    plt.close(fig)

def plot_country_trajectories(epoch_snapshots, node_to_pos, entity_mapping, seed, countries_to_track=None):
    """
    Shows the final position of key countries in the stable t-SNE embedding space.
    This plot shows the final "context" of these entities relative to each other.
    """
    if countries_to_track is None:
        countries_to_track = ["United States", "China", "Russia", "United Kingdom", "India", "Iran"]

    print(f"\nGenerating geopolitical context plot for seed {seed}...")

    name_to_id = {v: k for k, v in entity_mapping.items()}
    track_ids = [name_to_id.get(name) for name in countries_to_track if name_to_id.get(name) is not None]

    if not track_ids:
        print("  - Could not find any of the specified countries in the entity mapping. Skipping plot.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 14))

    # Plot all other entities as a faint background
    all_positions = np.array(list(node_to_pos.values()))
    ax.scatter(all_positions[:, 0], all_positions[:, 1], c='lightgray', s=10, alpha=0.3, label='Other Entities')

    # Get the final cluster assignments to color the countries
    final_epoch = max([e for e in epoch_snapshots.keys() if e > 0])
    final_snapshot = epoch_snapshots[final_epoch]
    node_to_cluster_map = dict(zip(final_snapshot['node_ids'].numpy(), final_snapshot['assignments_l1'].numpy()))
    
    # Use a consistent color map
    num_clusters = final_snapshot['assignments_l1'].max().item() + 1
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, num_clusters))

    # Plot each tracked country as a single, prominent point
    for country_id in track_ids:
        country_name = entity_mapping[country_id]
        pos = node_to_pos.get(country_id)
        cluster_id = node_to_cluster_map.get(country_id)
        
        if pos is not None and cluster_id is not None:
            ax.scatter(pos[0], pos[1], color=colors[cluster_id], s=250, 
                       edgecolors='black', linewidth=1.5, label=country_name)
            ax.text(pos[0] + 0.5, pos[1] + 0.5, country_name, fontsize=12, fontweight='bold')

    # Create a clean legend without duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Countries", fontsize=12)
    
    # Update title to be more accurate
    ax.set_title(f'Geopolitical Context in Embedding Space (Seed {seed})', fontsize=20)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=14)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=14)

    plt.tight_layout()
    plot_filename = f"icews18_geopolitical_context_seed_{seed}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Geopolitical context plot saved to {plot_filename}")
    plt.close(fig)



def generate_hierarchical_analysis_report(final_snapshot, data_all, entity_mapping, seed, top_k=5):
    """
    Generates a comprehensive report detailing:
    1. Graph coarsening sizes (nodes/edges) at each level.
    2. Cluster composition for both Level 1 and Level 2.
    """
    print(f"\n\n{'='*25} HIERARCHICAL ANALYSIS REPORT (Seed: {seed}) {'='*25}")

    if 'assignments_l1' not in final_snapshot or not entity_mapping:
        print("  - Missing L1 assignments or entity mapping. Cannot generate report.")
        return

    # --- Basic Data Prep ---
    node_ids = final_snapshot['node_ids'].numpy()
    assignments_l1 = final_snapshot['assignments_l1'].numpy()
    has_l2 = 'assignments_l2' in final_snapshot
    
    df = pd.DataFrame({'node_id': node_ids, 'l1_cluster': assignments_l1})
    if has_l2:
        assignments_l2 = final_snapshot['assignments_l2'].numpy()
        df['l2_cluster'] = assignments_l2

    # --- 1. CALCULATE COARSENED GRAPH SIZES ---
    
    # Level 0: Original Graph
    num_l0_nodes = data_all.num_nodes
    num_l0_edges = data_all.num_events

    # Level 1: Coarsened Graph
    active_l1_clusters = np.unique(assignments_l1)
    num_l1_nodes = len(active_l1_clusters)
    node_to_l1_map = dict(zip(node_ids, assignments_l1))
    
    inferred_l1_edges = set()
    original_src, original_dst = data_all.src.numpy(), data_all.dst.numpy()
    for i in range(len(original_src)):
        c_src = node_to_l1_map.get(original_src[i])
        c_dst = node_to_l1_map.get(original_dst[i])
        if c_src is not None and c_dst is not None and c_src != c_dst:
            inferred_l1_edges.add(tuple(sorted((c_src, c_dst))))
    num_l1_edges = len(inferred_l1_edges)

    # Level 2: Coarsened Graph (if it exists)
    num_l2_nodes, num_l2_edges = 'N/A', 'N/A'
    if has_l2:
        active_l2_clusters = np.unique(assignments_l2)
        num_l2_nodes = len(active_l2_clusters)
        
        # Map L1 clusters to their dominant L2 parent cluster
        l1_to_l2_map = df.groupby('l1_cluster')['l2_cluster'].apply(lambda x: x.mode()[0]).to_dict()

        inferred_l2_edges = set()
        for l1_u, l1_v in inferred_l1_edges:
            l2_u = l1_to_l2_map.get(l1_u)
            l2_v = l1_to_l2_map.get(l1_v)
            if l2_u is not None and l2_v is not None and l2_u != l2_v:
                inferred_l2_edges.add(tuple(sorted((l2_u, l2_v))))
        num_l2_edges = len(inferred_l2_edges)
        
    print("\n--- Graph Coarsening Summary ---")
    print(f"| Level | Graph Type        | Num Nodes (Clusters) | Num Edges (Inferred) |")
    print(f"|-------|-------------------|----------------------|----------------------|")
    print(f"|   0   | Original Graph    | {num_l0_nodes:<20} | {num_l0_edges:<20} |")
    print(f"|   1   | L1 Meta-Graph     | {num_l1_nodes:<20} | {num_l1_edges:<20} |")
    print(f"|   2   | L2 Meta-Graph     | {str(num_l2_nodes):<20} | {str(num_l2_edges):<20} |")
    
    
    # --- 2. CLUSTER COMPOSITION REPORT ---
    print("\n--- Cluster Composition Report ---")

    # LEVEL 1 REPORT
    print(f"\n--- LEVEL 1: Top {top_k} Largest Clusters ---")
    l1_cluster_sizes = df['l1_cluster'].value_counts()
    
    # Generate summary names for L1 clusters for the L2 report
    l1_summary_names = {}
    for i in range(min(len(l1_cluster_sizes), 100)): # Generate names for top 100 L1 clusters
        cid = l1_cluster_sizes.index[i]
        member_ids = df[df['l1_cluster'] == cid]['node_id'].head(5)
        member_names = [entity_mapping.get(nid, f"ID:{nid}") for nid in member_ids]
        # A simple summary: the first two entities.
        summary = f"C{cid} ({', '.join(member_names[:2])}, ...)"
        l1_summary_names[cid] = summary

    for i in range(min(top_k, len(l1_cluster_sizes))):
        cid = l1_cluster_sizes.index[i]
        csize = l1_cluster_sizes.iloc[i]
        print(f"\n========================================")
        print(f"Cluster {l1_summary_names.get(cid, f'C{cid}')} | Size: {csize} nodes")
        print(f"----------------------------------------")
        member_ids = df[df['l1_cluster'] == cid]['node_id'].sample(n=min(10, csize), random_state=1)
        for node_id in member_ids:
            name = entity_mapping.get(node_id, f"ID:{node_id} (Name not found)")
            print(f"  - {name}")

    # LEVEL 2 REPORT
    if has_l2:
        print(f"\n\n--- LEVEL 2: Top {top_k} Largest Meta-Clusters ---")
        l2_cluster_sizes = df['l2_cluster'].value_counts()
        
        # Create a mapping from L2 cluster ID to a list of its L1 cluster members
        l2_to_l1_members = df.groupby('l2_cluster')['l1_cluster'].unique().to_dict()

        for i in range(min(top_k, len(l2_cluster_sizes))):
            cid = l2_cluster_sizes.index[i]
            csize = l2_cluster_sizes.iloc[i]
            l1_members = l2_to_l1_members.get(cid, [])
            
            print(f"\n========================================")
            print(f"Meta-Cluster M{cid} | Size: {csize} nodes ({len(l1_members)} L1 Clusters)")
            print(f"----------------------------------------")
            print(f"  Contains L1 Clusters such as:")
            
            # Show a sample of the contained L1 clusters using their summary names
            for l1_cid in list(l1_members)[:min(10, len(l1_members))]:
                summary_name = l1_summary_names.get(l1_cid, f"Cluster C{l1_cid}")
                print(f"  - {summary_name}")
    
    print(f"\n{'='*75}\n")

def plot_coarsening_comparison(final_snapshot, data_all, seed, num_clusters_to_sample=5, max_edges=200):
    """
    Visualizes graph coarsening by showing a 'before' and 'after' view.
    Now guaranteed to show inter-cluster edges.
    """
    print(f"\nGenerating graph coarsening comparison plot for seed {seed}...")
    
    if 'assignments_l1' not in final_snapshot:
        print("  - Level 1 assignments not found. Skipping coarsening plot.")
        return

    node_ids = final_snapshot['node_ids'].numpy()
    assignments = final_snapshot['assignments_l1'].numpy()
    node_to_cluster_map = dict(zip(node_ids, assignments))

    # --- 1. Find edges between DIFFERENT clusters first ---
    original_src, original_dst = data_all.src.numpy(), data_all.dst.numpy()
    
    inter_cluster_edges = []
    for i in range(len(original_src)):
        src, dst = original_src[i], original_dst[i]
        c_src = node_to_cluster_map.get(src)
        c_dst = node_to_cluster_map.get(dst)
        
        # Only keep edges between DIFFERENT clusters
        if c_src is not None and c_dst is not None and c_src != c_dst:
            inter_cluster_edges.append((src, dst, c_src, c_dst))
    
    if not inter_cluster_edges:
        print("  - No inter-cluster edges found. Cannot visualize coarsening.")
        return
    
    # Sample edges to keep visualization manageable
    if len(inter_cluster_edges) > max_edges:
        sampled_edge_indices = np.random.choice(len(inter_cluster_edges), max_edges, replace=False)
        inter_cluster_edges = [inter_cluster_edges[i] for i in sampled_edge_indices]
    
    print(f"  - Using {len(inter_cluster_edges)} inter-cluster edges for visualization.")
    
    # --- 2. Build the 'Before' Graph from these edges ---
    G_before = nx.Graph()
    cluster_edge_counts = {}
    
    for src, dst, c_src, c_dst in inter_cluster_edges:
        G_before.add_edge(src, dst)
        edge_key = tuple(sorted((c_src, c_dst)))
        cluster_edge_counts[edge_key] = cluster_edge_counts.get(edge_key, 0) + 1
    
    sampled_nodes = list(G_before.nodes())
    print(f"  - Sampled {len(sampled_nodes)} nodes from {len(set(node_to_cluster_map.get(n) for n in sampled_nodes))} clusters.")

    # --- 3. Build the 'After' Graph (Coarsened Meta-Graph) ---
    G_after = nx.Graph()
    for (c_u, c_v), weight in cluster_edge_counts.items():
        G_after.add_edge(f"C{c_u}", f"C{c_v}", weight=weight)

    # --- 4. Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot Before
    ax = axes[0]
    node_colors = [node_to_cluster_map.get(n, -1) for n in G_before.nodes()]
    pos_before = nx.spring_layout(G_before, seed=seed, k=0.5, iterations=50)
    nx.draw(G_before, pos=pos_before, ax=ax, with_labels=False, node_size=150,
            node_color=node_colors, cmap=plt.cm.nipy_spectral, width=0.8, alpha=0.9, edge_color='gray')
    ax.set_title(f'Before: Original Graph (Sample of {len(sampled_nodes)} Nodes)', fontsize=16)

    # Plot After
    ax = axes[1]
    pos_after = nx.spring_layout(G_after, seed=seed, k=1.0, iterations=50)
    edge_widths = [d['weight'] * 0.3 for u, v, d in G_after.edges(data=True)]
    cluster_id_colors = [int(node_label[1:]) for node_label in G_after.nodes()]

    nx.draw(G_after, pos=pos_after, ax=ax, with_labels=True, node_size=2000,
            node_color=cluster_id_colors, cmap=plt.cm.nipy_spectral,
            width=edge_widths, font_size=12, font_color='white', font_weight='bold', 
            edge_color='gray', alpha=0.9)
    ax.set_title(f'After: Coarsened Meta-Graph ({G_after.number_of_nodes()} Clusters)', fontsize=16)
    
    fig.suptitle(f'Visualizing Graph Coarsening (Seed {seed})', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_filename = f"icews18_coarsening_comparison_seed_{seed}.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"Coarsening comparison plot saved to {plot_filename}")
    plt.close(fig)

def plot_alluvial_evolution(epoch_snapshots, seed, max_clusters_to_show=15):
    """
    Generates an Alluvial diagram to show the evolution of communities over time.
    Focuses on the largest communities to maintain readability.
    """
    if len(epoch_snapshots) < 2: return
    print(f"\nGenerating community evolution Alluvial plot for seed {seed}...")

    epochs = sorted([e for e in epoch_snapshots.keys() if e > 0]) # Exclude initial random state
    if len(epochs) < 2: 
        print("  - Not enough trained epochs to show evolution. Skipping.")
        return

    # --- 1. Consolidate all node assignments into a single DataFrame ---
    df_list = []
    for epoch in epochs:
        snap = epoch_snapshots[epoch]
        df_epoch = pd.DataFrame({
            'node_id': snap['node_ids'].numpy(),
            f'E{epoch}': snap['assignments_l1'].numpy()
        }).set_index('node_id')
        df_list.append(df_epoch)
    
    df_assignments = pd.concat(df_list, axis=1).fillna(-1).astype(int)

    # --- 2. Identify the most significant clusters to track ---
    final_epoch_col = f'E{epochs[-1]}'
    top_clusters = df_assignments[final_epoch_col].value_counts().head(max_clusters_to_show).index
    
    # Filter the DataFrame to only include nodes that end up in these top clusters
    df_filtered = df_assignments[df_assignments[final_epoch_col].isin(top_clusters)]
    print(f"  - Tracking evolution for {len(df_filtered)} nodes ending in the top {max_clusters_to_show} clusters.")

    # --- 3. Prepare data for Plotly Sankey/Alluvial ---
    all_labels = []
    label_map = {}
    
    # Create unique labels for each cluster at each epoch (e.g., "E5_C123")
    for epoch in epochs:
        col = f'E{epoch}'
        unique_clusters = df_filtered[col].unique()
        for cluster in unique_clusters:
            if cluster != -1:
                label = f'E{epoch}_C{cluster}'
                if label not in label_map:
                    label_map[label] = len(all_labels)
                    all_labels.append(label)

    source, target, value = [], [], []
    
    # Calculate flows between consecutive epochs
    for i in range(len(epochs) - 1):
        col_from, col_to = f'E{epochs[i]}', f'E{epochs[i+1]}'
        flows = df_filtered.groupby([col_from, col_to]).size().reset_index(name='count')
        
        for _, row in flows.iterrows():
            c_from, c_to, count = row[col_from], row[col_to], row['count']
            if c_from != -1 and c_to != -1:
                source_label = f'E{epochs[i]}_C{c_from}'
                target_label = f'E{epochs[i+1]}_C{c_to}'
                
                if source_label in label_map and target_label in label_map:
                    source.append(label_map[source_label])
                    target.append(label_map[target_label])
                    value.append(count)

    # --- 4. Create the Plotly Figure ---
    # Assign positions to nodes to force chronological layout
    node_x = [int(label.split('_')[0][1:])/max(epochs) for label in all_labels]
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=[label.split('_')[1] for label in all_labels], # Show only "C123" on node
            x=node_x, # Horizontal position based on epoch
            customdata=[label.split('_')[0] for label in all_labels],
            hovertemplate='Epoch: %{customdata}, %{label}<extra></extra>'
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        )
    )])

    fig.update_layout(
        title_text=f"Community Evolution Alluvial Diagram (Seed {seed})",
        font_size=12,
        xaxis={'showgrid': False, 'zeroline': False, 'visible': False},
        yaxis={'showgrid': False, 'zeroline': False, 'visible': False},
        plot_bgcolor='white'
    )
    
    # Add epoch labels at the top
    for epoch in epochs:
        fig.add_annotation(
            x=epoch/max(epochs), y=1.05, text=f"<b>Epoch {epoch}</b>",
            showarrow=False, font=dict(size=14)
        )
    
    plot_filename = f"icews18_alluvial_evolution_seed_{seed}.html"
    fig.write_html(plot_filename)
    print(f"Alluvial evolution plot saved to {plot_filename}")





# STRATEGY 1: Filter by Final Cluster Size (The "Top K" Plot)
def plot_top_k_alluvial(epoch_snapshots, num_top_clusters, seed):
    """
    Generates a simplified Alluvial diagram showing the evolution of only the
    nodes that end up in the 'num_top_clusters' largest communities.
    """
    print(f"--- Generating Top {num_top_clusters} Cluster Alluvial plot for seed {seed} ---")
    
    # Prepare the data in a pandas DataFrame for easier manipulation
    all_data = []
    sorted_epochs = sorted(epoch_snapshots.keys())
    for epoch in sorted_epochs:
        snapshot = epoch_snapshots[epoch]
        if 'assignments_l1' in snapshot:
            for node_id, cluster_id in zip(snapshot['node_ids'].numpy(), snapshot['assignments_l1'].numpy()):
                all_data.append({'epoch': epoch, 'node_id': node_id, 'cluster_id': cluster_id})
    df = pd.DataFrame(all_data)

    # --- Filtering Logic ---
    final_epoch = max(df['epoch'])
    final_clusters = df[df['epoch'] == final_epoch]
    top_k_clusters = final_clusters['cluster_id'].value_counts().nlargest(num_top_clusters).index
    
    # Get the nodes that end up in these top K clusters
    nodes_to_track = final_clusters[final_clusters['cluster_id'].isin(top_k_clusters)]['node_id'].unique()
    
    # Filter the entire dataframe to only these nodes
    df_filtered = df[df['node_id'].isin(nodes_to_track)]
    # --- End Filtering ---

    # --- Data preparation for Sankey/Alluvial ---
    all_nodes = []
    for epoch in sorted_epochs:
        if epoch in df_filtered['epoch'].unique():
            clusters = df_filtered[df_filtered['epoch'] == epoch]['cluster_id'].unique()
            all_nodes.extend([f"E{epoch}-C{c}" for c in clusters])
    
    label_to_id = {label: i for i, label in enumerate(all_nodes)}
    
    source, target, value = [], [], []
    for i, epoch in enumerate(sorted_epochs[:-1]):
        next_epoch = sorted_epochs[i+1]
        
        if epoch not in df_filtered['epoch'].unique() or next_epoch not in df_filtered['epoch'].unique():
            continue

        current_df = df_filtered[df_filtered['epoch'] == epoch]
        next_df = df_filtered[df_filtered['epoch'] == next_epoch]
        
        merged_df = pd.merge(current_df, next_df, on='node_id', suffixes=('_curr', '_next'))
        transitions = merged_df.groupby(['cluster_id_curr', 'cluster_id_next']).size().reset_index(name='count')

        for _, row in transitions.iterrows():
            source_label = f"E{epoch}-C{int(row['cluster_id_curr'])}"
            target_label = f"E{next_epoch}-C{int(row['cluster_id_next'])}"
            if source_label in label_to_id and target_label in label_to_id:
                source.append(label_to_id[source_label])
                target.append(label_to_id[target_label])
                value.append(row['count'])

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        ))])

    fig.update_layout(title_text=f"Top {num_top_clusters} Community Evolution (Seed {seed})", font_size=10)
    fig.write_html(f"icews18_alluvial_top_{num_top_clusters}_seed_{seed}.html")
    print(f"Top {num_top_clusters} Alluvial plot saved to icews18_alluvial_top_{num_top_clusters}_seed_{seed}.html")


# STRATEGY 2: Focus on a Specific Event (The "Highlight" Plot)
def plot_event_highlight_alluvial(epoch_snapshots, parent_epoch, parent_cluster_id, child_epoch, seed, highlight_color='rgba(220, 50, 50, 0.8)', ghost_color='rgba(200, 200, 200, 0.2)'):
    """
    Generates a full Alluvial diagram but highlights the flows from a specific
    parent cluster to its children, ghosting out all other flows.
    """
    print(f"--- Generating Event Highlight Alluvial plot for seed {seed} ---")

    # Prepare the full data just like the original complex plot
    all_data = []
    sorted_epochs = sorted(epoch_snapshots.keys())
    for epoch in sorted_epochs:
        snapshot = epoch_snapshots[epoch]
        if 'assignments_l1' in snapshot:
            for node_id, cluster_id in zip(snapshot['node_ids'].numpy(), snapshot['assignments_l1'].numpy()):
                all_data.append({'epoch': epoch, 'node_id': node_id, 'cluster_id': cluster_id})
    df = pd.DataFrame(all_data)

    all_nodes = []
    for epoch in sorted_epochs:
        clusters = df[df['epoch'] == epoch]['cluster_id'].unique()
        all_nodes.extend([f"E{epoch}-C{c}" for c in clusters])
    
    label_to_id = {label: i for i, label in enumerate(all_nodes)}
    
    source, target, value = [], [], []
    links_data = []
    
    for i, epoch in enumerate(sorted_epochs[:-1]):
        next_epoch = sorted_epochs[i+1]
        
        current_df = df[df['epoch'] == epoch]
        next_df = df[df['epoch'] == next_epoch]
        
        merged_df = pd.merge(current_df, next_df, on='node_id', suffixes=('_curr', '_next'))
        transitions = merged_df.groupby(['cluster_id_curr', 'cluster_id_next']).size().reset_index(name='count')

        for _, row in transitions.iterrows():
            curr_c = int(row['cluster_id_curr'])
            next_c = int(row['cluster_id_next'])
            source_label = f"E{epoch}-C{curr_c}"
            target_label = f"E{next_epoch}-C{next_c}"
            
            source.append(label_to_id[source_label])
            target.append(label_to_id[target_label])
            value.append(row['count'])
            links_data.append({'source_epoch': epoch, 'source_cluster': curr_c, 'target_epoch': next_epoch})

    # --- Highlighting Logic ---
    # Find the children of the parent cluster
    parent_nodes = df[(df['epoch'] == parent_epoch) & (df['cluster_id'] == parent_cluster_id)]['node_id']
    child_clusters = df[(df['epoch'] == child_epoch) & (df['node_id'].isin(parent_nodes))]['cluster_id'].unique()

    link_colors = []
    for link in links_data:
        is_highlighted = (link['source_epoch'] == parent_epoch and 
                          link['source_cluster'] == parent_cluster_id)
        if is_highlighted:
            link_colors.append(highlight_color)
        else:
            link_colors.append(ghost_color)
    # --- End Highlighting ---

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors # Apply the custom colors
        ))])

    fig.update_layout(title_text=f"Highlighting Split from E{parent_epoch}-C{parent_cluster_id} (Seed {seed})", font_size=10)
    fig.write_html(f"icews18_alluvial_highlight_E{parent_epoch}C{parent_cluster_id}_seed_{seed}.html")
    print(f"Event Highlight Alluvial plot saved to icews18_alluvial_highlight_E{parent_epoch}C{parent_cluster_id}_seed_{seed}.html")


# STRATEGY 3: "Ego" Alluvial for a Specific Cluster
def plot_cluster_lineage_alluvial(epoch_snapshots, final_cluster_id, seed, color='rgba(70, 70, 200, 0.8)'):
    """
    Generates a highly simplified Alluvial diagram showing only the history
    of nodes that end up in a single, specified final cluster.
    """
    print(f"--- Generating Lineage Alluvial plot for C{final_cluster_id} (seed {seed}) ---")
    
    # This function is a special case of plot_top_k_alluvial where k=1
    # We can reuse the same filtering logic
    
    all_data = []
    sorted_epochs = sorted(epoch_snapshots.keys())
    for epoch in sorted_epochs:
        snapshot = epoch_snapshots[epoch]
        if 'assignments_l1' in snapshot:
            for node_id, cluster_id in zip(snapshot['node_ids'].numpy(), snapshot['assignments_l1'].numpy()):
                all_data.append({'epoch': epoch, 'node_id': node_id, 'cluster_id': cluster_id})
    df = pd.DataFrame(all_data)

    final_epoch = max(df['epoch'])
    nodes_to_track = df[(df['epoch'] == final_epoch) & (df['cluster_id'] == final_cluster_id)]['node_id'].unique()
    
    if len(nodes_to_track) == 0:
        print(f"Warning: Cluster C{final_cluster_id} has no members in the final epoch. Skipping plot.")
        return

    df_filtered = df[df['node_id'].isin(nodes_to_track)]

    # --- Data preparation for Sankey/Alluvial ---
    all_nodes = []
    for epoch in sorted_epochs:
        if epoch in df_filtered['epoch'].unique():
            clusters = sorted(df_filtered[df_filtered['epoch'] == epoch]['cluster_id'].unique())
            all_nodes.extend([f"E{epoch}-C{c}" for c in clusters])
    
    label_to_id = {label: i for i, label in enumerate(all_nodes)}
    
    source, target, value = [], [], []
    for i, epoch in enumerate(sorted_epochs[:-1]):
        next_epoch = sorted_epochs[i+1]
        
        if epoch not in df_filtered['epoch'].unique() or next_epoch not in df_filtered['epoch'].unique():
            continue

        current_df = df_filtered[df_filtered['epoch'] == epoch]
        next_df = df_filtered[df_filtered['epoch'] == next_epoch]
        
        merged_df = pd.merge(current_df, next_df, on='node_id', suffixes=('_curr', '_next'))
        transitions = merged_df.groupby(['cluster_id_curr', 'cluster_id_next']).size().reset_index(name='count')

        for _, row in transitions.iterrows():
            source_label = f"E{epoch}-C{int(row['cluster_id_curr'])}"
            target_label = f"E{next_epoch}-C{int(row['cluster_id_next'])}"
            if source_label in label_to_id and target_label in label_to_id:
                source.append(label_to_id[source_label])
                target.append(label_to_id[target_label])
                value.append(row['count'])

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color=color # Color all nodes in the lineage
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=color # Color all links in the lineage
        ))])

    fig.update_layout(title_text=f"Lineage of Final Cluster C{final_cluster_id} (Seed {seed})", font_size=10)
    fig.write_html(f"icews18_alluvial_lineage_C{final_cluster_id}_seed_{seed}.html")
    print(f"Lineage Alluvial plot saved to icews18_alluvial_lineage_C{final_cluster_id}_seed_{seed}.html")
    
    

def plot_simplified_alluvial(epoch_snapshots, final_top_k, intermediate_top_k, seed):
    """
    Generates a simplified Alluvial diagram where, for intermediate epochs, only the
    'intermediate_top_k' largest clusters are shown, and the rest are aggregated
    into a single "Other" node.
    """
    print(f"--- Generating SIMPLIFIED Alluvial plot for seed {seed} ---")

    all_data = []
    sorted_epochs = sorted(epoch_snapshots.keys())
    for epoch in sorted_epochs:
        snapshot = epoch_snapshots[epoch]
        if 'assignments_l1' in snapshot:
            for node_id, cluster_id in zip(snapshot['node_ids'].numpy(), snapshot['assignments_l1'].numpy()):
                all_data.append({'epoch': epoch, 'node_id': node_id, 'cluster_id': cluster_id})
    df = pd.DataFrame(all_data)

    # --- AGGREGATION LOGIC ---
    df_simplified = df.copy()
    final_epoch = max(sorted_epochs)

    # Determine which final clusters to keep
    final_clusters = df[df['epoch'] == final_epoch]
    top_k_final_clusters = final_clusters['cluster_id'].value_counts().nlargest(final_top_k).index
    nodes_to_track = final_clusters[final_clusters['cluster_id'].isin(top_k_final_clusters)]['node_id'].unique()
    
    # Filter the entire dataframe to only these nodes
    df_simplified = df_simplified[df_simplified['node_id'].isin(nodes_to_track)]

    # For intermediate epochs, aggregate small clusters into "Other" (ID -1)
    for epoch in sorted_epochs:
        if epoch == 0 or epoch == final_epoch: # Keep all clusters for first and last epoch
            continue
        
        epoch_df = df_simplified[df_simplified['epoch'] == epoch]
        top_clusters_in_epoch = epoch_df['cluster_id'].value_counts().nlargest(intermediate_top_k).index
        
        # Replace non-top clusters with -1
        df_simplified.loc[
            (df_simplified['epoch'] == epoch) & (~df_simplified['cluster_id'].isin(top_clusters_in_epoch)),
            'cluster_id'
        ] = -1
    # --- END AGGREGATION LOGIC ---


    # --- Data preparation for Sankey/Alluvial (same as before but on simplified data) ---
    all_nodes = []
    node_colors = []
    
    # Define a color for the 'Other' node
    other_color = 'rgba(200, 200, 200, 0.8)' # Light Gray

    for epoch in sorted_epochs:
        if epoch in df_simplified['epoch'].unique():
            clusters = sorted(df_simplified[df_simplified['epoch'] == epoch]['cluster_id'].unique())
            for c in clusters:
                label = f"E{epoch}-C{c}" if c != -1 else f"E{epoch}-Other"
                all_nodes.append(label)
                # Assign colors
                if c == -1:
                    node_colors.append(other_color)
                else:
                    # You can use a more sophisticated coloring scheme here if you want
                    node_colors.append(go.colors.qualitative.Plotly[c % len(go.colors.qualitative.Plotly)])

    label_to_id = {label: i for i, label in enumerate(all_nodes)}
    
    source, target, value = [], [], []
    link_colors = []

    for i, epoch in enumerate(sorted_epochs[:-1]):
        next_epoch = sorted_epochs[i+1]
        
        if epoch not in df_simplified['epoch'].unique() or next_epoch not in df_simplified['epoch'].unique():
            continue

        current_df = df_simplified[df_simplified['epoch'] == epoch]
        next_df = df_simplified[df_simplified['epoch'] == next_epoch]
        
        merged_df = pd.merge(current_df, next_df, on='node_id', suffixes=('_curr', '_next'))
        transitions = merged_df.groupby(['cluster_id_curr', 'cluster_id_next']).size().reset_index(name='count')

        for _, row in transitions.iterrows():
            c_curr = int(row['cluster_id_curr'])
            c_next = int(row['cluster_id_next'])
            
            source_label = f"E{epoch}-C{c_curr}" if c_curr != -1 else f"E{epoch}-Other"
            target_label = f"E{next_epoch}-C{c_next}" if c_next != -1 else f"E{next_epoch}-Other"
            
            if source_label in label_to_id and target_label in label_to_id:
                source.append(label_to_id[source_label])
                target.append(label_to_id[target_label])
                value.append(row['count'])
                # Make links involving "Other" gray
                if c_curr == -1 or c_next == -1:
                    link_colors.append('rgba(200, 200, 200, 0.5)')
                else:
                    link_colors.append('rgba(150, 150, 150, 0.5)')


    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color=node_colors # Apply custom node colors
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors # Apply custom link colors
        ))])

    fig.update_layout(title_text=f"Simplified Top {final_top_k} Community Evolution (Seed {seed})", font_size=12)
    fig.write_html(f"icews18_alluvial_simplified_top{final_top_k}_seed_{seed}.html")
    print(f"Simplified Alluvial plot saved to icews18_alluvial_simplified_top{final_top_k}_seed_{seed}.html")