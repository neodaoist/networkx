import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Create a new directed graph
G = nx.MultiDiGraph()

# Add nodes with specific types
apps = ['A₁', 'A₂']
builders = ['B₁', 'B₂', 'B₃', 'B₄']

# Add nodes with their types
for app in apps:
    G.add_node(app, node_type='app')
for builder in builders:
    G.add_node(builder, node_type='builder')

# Add directed edges with labels, using explicit keys for multiple edges
edge_info = [
    ('B₁', 'A₁', 0, "GH PR to Repo A1-FE"),
    ('B₂', 'A₁', 0, "GH PR to Repo A1-FE"),
    ('B₃', 'A₁', 0, "GH PR to Repo A1-SC"),  # First edge with key 0
    ('B₃', 'A₁', 1, "SC Deploy"),            # Second edge with key 1
    ('B₄', 'A₁', 0, "GH PR to Repo A1-SC"),
    ('B₄', 'A₂', 0, "GH PR to Repo A2-Only")
]

# Add edges with their labels using explicit keys
for source, target, key, label in edge_info:
    G.add_edge(source, target, key=key, label=label)

# Set up the plot
plt.figure(figsize=(12, 8))

# Create positions dictionary
pos = {}

# Position builders exactly equidistant at the top
builder_xs = np.linspace(-0.75, 0.75, 4)
for i, builder in enumerate(builders):
    pos[builder] = np.array([builder_xs[i], 0.5])

# Position apps at the bottom
pos['A₁'] = np.array([-0.25, -0.5])
pos['A₂'] = np.array([0.25, -0.5])

# Define colors
color_map = {
    'app': '#0000FF',     # Blue
    'builder': '#00CED1'  # Bright teal
}

# Draw nodes
node_colors = [color_map[G.nodes[node]['node_type']] for node in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000)

# Draw edges with different curvatures for multiple edges
for source, target, key, data in G.edges(data=True, keys=True):
    rad = 0.2
    
    # Special handling for B3->A1 edges
    if source == 'B₃' and target == 'A₁':
        rad = 0.1 if key == 0 else 0.3  # Different curves for each key

    edge_path = nx.draw_networkx_edges(G, pos,
                                     edgelist=[(source, target)],
                                     edge_color='#404040',
                                     width=2,
                                     connectionstyle=f'arc3,rad={rad}',
                                     arrows=True,
                                     arrowstyle='-|>',
                                     arrowsize=50,
                                     min_source_margin=30,
                                     min_target_margin=30)

# Create edge labels dictionary
edge_labels = {}
for source, target, key, data in G.edges(data=True, keys=True):
    edge_labels[(source, target, key)] = data['label']

# Print edge labels for debugging
print("Edge labels dictionary:")
for (s, t, k), l in edge_labels.items():
    print(f"Edge ({s}, {t}, {k}): {l}")

# Draw edge labels with different positions based on keys
for (source, target, key), label in edge_labels.items():
    label_pos = 0.5  # Default position
    bbox_props = dict(facecolor='white', edgecolor='none', alpha=0.6)
    
    if source == 'B₃' and target == 'A₁':
        if key == 0:  # First edge (GH PR to Repo A1-SC)
            label_pos = 0.25
            bbox_props['pad'] = 0
        else:  # Second edge (SC Deploy)
            label_pos = 0.6
            bbox_props['pad'] = 0

    try:
        nx.draw_networkx_edge_labels(G, pos,
                                    {(source, target): label},  # Remove key from the dictionary key
                                    font_size=8,
                                    label_pos=label_pos,
                                    bbox=bbox_props,
                                    bbox_ref_point='e' if key == 0 else 'w',
                                    rotate=True)
    except TypeError:
        # If that fails, fall back to standard drawing
        nx.draw_networkx_edge_labels(G, pos,
                                    {(source, target, key): label},  # Use the full key tuple
                                    font_size=8,
                                    label_pos=label_pos,
                                    bbox=bbox_props,
                                    rotate=True)

# Create a dictionary for node label colors
label_colors = {node: 'white' if node in apps else 'black' for node in G.nodes()}

# Draw all labels with same properties
nx.draw_networkx_labels(G, pos,
                       font_size=14,
                       font_family='DejaVu Sans',
                       font_weight='bold',
                       font_color=label_colors)

plt.axis('off')
plt.margins(0.2)
plt.tight_layout()
plt.show()