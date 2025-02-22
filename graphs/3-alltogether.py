import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Create directed graph
G = nx.DiGraph()
G_undir = nx.Graph()  # For undirected edges

# Define node groups
P_nodes = ['P₁', 'P₂', 'P₃']
U_nodes = ['U₁', 'U₂']
E_nodes = ['E₁', 'E₂']
A_top_nodes = ['A₁']
A_middle_nodes = ['A₂', 'A₃']
S_nodes = ['S₁']  # Changed A4 to S1
C_nodes = ['C₁', 'C₂', 'C₃', 'C₄', 'C₅', 'C₆']  # Removed C7
App_nodes = ['App₁', 'App₂', 'App₃', 'App₄']
B_nodes = ['B₁', 'B₂', 'B₃', 'B₄', 'B₅']

# Add nodes with types
node_types = {}
for p in P_nodes:
    node_types[p] = 'person'
for u in U_nodes:
    node_types[u] = 'user'
for e in E_nodes:
    node_types[e] = 'entity'
for a in A_top_nodes + A_middle_nodes:
    node_types[a] = 'auth'
for s in S_nodes:
    node_types[s] = 'system'
for c in C_nodes:
    node_types[c] = 'contract'
for app in App_nodes:
    node_types[app] = 'app'
for b in B_nodes:
    node_types[b] = 'builder'

# Add all nodes to both graphs
for node, type in node_types.items():
    G.add_node(node, type=type)
    G_undir.add_node(node, type=type)

# Add directed edges
directed_edges = [
    ('P₁', 'U₁'),
    ('U₁', 'E₁'),
    ('U₁', 'A₁'),
    ('A₁', 'C₁'),
    ('B₁', 'App₁'),
    ('B₂', 'App₁'),
    ('C₁', 'C₃'),         # New directed edge
    ('P₂', 'E₂'),
    ('P₂', 'A₂'),
    ('P₂', 'A₃'),
    ('P₂', 'S₁'),
    ('E₁', 'C₃'),
    ('E₂', 'C₃'),
    ('E₂', 'C₄'),
    ('E₂', 'C₅'),
    ('A₂', 'C₃'),
    ('A₃', 'C₆'),
    ('S₁', 'C₅'),
    ('App₁', 'C₁'),
    ('App₂', 'C₂'),
    ('App₃', 'C₆'),
    ('App₄', 'C₅'),       # Reversed direction
    ('B₃', 'App₂'),
    ('P₃', 'U₂'),
    ('U₂', 'C₅'),
    ('U₂', 'C₆'),
    ('B₂', 'App₂'),
    ('B₄', 'App₂'),
    ('B₄', 'App₃'),
    ('B₅', 'App₄')
]

# Add undirected edges
undirected_edges = [
    ('C₃', 'C₂'),         # Made undirected
    ('C₄', 'C₂'),         # Made undirected
    ('C₅', 'C₂')          # Made undirected
]

# Add edges to respective graphs
G.add_edges_from(directed_edges)
G_undir.add_edges_from(undirected_edges)

# Set up the plot
plt.figure(figsize=(15, 12))

# Define positions manually
pos = {
    # Top row - P nodes
    'P₁': (-6, 8), 'P₂': (0, 8), 'P₃': (6, 8),
    
    # Second row - U, E, A nodes
    'U₁': (-7, 6), 'E₁': (-5, 6), 'A₁': (-6, 6),
    'E₂': (-1, 6), 'A₂': (0, 6), 'A₃': (1, 6), 'S₁': (2, 6),
    'U₂': (6, 6),
    
    # Middle row - C nodes
    'C₁': (-6, 4), 'C₃': (-1, 4), 'C₄': (0, 4), 'C₅': (1, 4), 'C₆': (5, 4),
    'C₂': (0, 3),  # Slightly lower for hierarchy
    
    # Fourth row - App nodes
    'App₁': (-6, 2), 'App₂': (0, 2), 'App₃': (5, 2), 'App₄': (7, 2),
    
    # Bottom row - B nodes
    'B₁': (-7, 0), 'B₂': (-5, 0), 'B₃': (0, 0), 'B₄': (5, 0), 'B₅': (7, 0)
}

# Define colors with updated scheme
color_map = {
    'person': '#00FF00',    # Green
    'user': '#FFA500',      # Orange (changed from teal)
    'entity': '#FF7F7F',    # Light red
    'auth': '#E6E6FA',      # Light purple
    'system': '#9370DB',    # Darker purple for S1
    'contract': '#FFD700',  # Gold
    'app': '#0000FF',       # Blue
    'builder': '#00CED1'    # Teal (changed from orange)
}

# Draw the dotted boxes first
plt.gca().add_patch(patches.Rectangle((-6.5, 3.5), 1, 1,
                                    fill=False, linestyle='--',
                                    edgecolor='gray'))
plt.gca().add_patch(patches.Rectangle((-1.5, 3), 3, 1.5,
                                    fill=False, linestyle='--',
                                    edgecolor='gray'))
plt.gca().add_patch(patches.Rectangle((4.5, 3.5), 1, 1,
                                    fill=False, linestyle='--',
                                    edgecolor='gray'))

# Draw nodes
node_colors = [color_map[G.nodes[node]['type']] for node in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000)

# Draw directed edges with enhanced arrows
for edge in G.edges():
    nx.draw_networkx_edges(G, pos,
                          edgelist=[edge],
                          edge_color='#404040',
                          width=2,
                          connectionstyle='arc3,rad=0.2',
                          arrows=True,
                          arrowstyle='-|>',
                          arrowsize=50,
                          min_source_margin=30,
                          min_target_margin=30)

# Draw undirected edges
for edge in G_undir.edges():
    nx.draw_networkx_edges(G_undir, pos,
                          edgelist=[edge],
                          edge_color='#404040',
                          width=2,
                          connectionstyle='arc3,rad=0.2')

# Create label colors
label_colors = {node: 'white' if G.nodes[node]['type'] == 'app' else 'black' 
                for node in G.nodes()}

# Draw labels
nx.draw_networkx_labels(G, pos,
                       font_size=12,
                       font_family='DejaVu Sans',
                       font_weight='bold',
                       font_color=label_colors)

plt.axis('off')
plt.tight_layout()
plt.show()