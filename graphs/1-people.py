import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Create a new directed graph
G = nx.DiGraph()

# Define nodes by type
people = ['P₁', 'P₂', 'P₃', 'P₄']
users = ['U₁']
entities = ['E₁', 'E₂', 'E₃']
systems = ['S₁']  # Changed E4 to S1
apps = ['A₁', 'A₂', 'A₃']

# Add nodes with types
for p in people:
    G.add_node(p, node_type='person')
for u in users:
    G.add_node(u, node_type='user')
for e in entities:
    G.add_node(e, node_type='entity')
G.add_node('S₁', node_type='system')  # Add S1 with new type
for a in apps:
    G.add_node(a, node_type='app')

# Add edges with labels
G.add_edge('P₁', 'U₁')
G.add_edge('U₁', 'E₁', style='dotted')
G.add_edge('U₁', 'E₂', style='dotted')
G.add_edge('U₁', 'S₁', style='dotted')  # New dotted edge to S1
G.add_edge('U₁', 'A₁', style='solid', label='spendPermission')
G.add_edge('U₁', 'A₂', style='solid', label='spendPermission')
G.add_edge('U₁', 'A₃', style='solid', label='spendPermission')

plt.figure(figsize=(12, 10))

# Create positions dictionary
pos = {}

# Position nodes
people_xs = np.linspace(-0.75, 0.75, len(people))
for i, p in enumerate(people):
    pos[p] = np.array([people_xs[i], 0.75])

pos['U₁'] = np.array([0, 0.25])

# Position entities and S1 in same row
all_middle_nodes = entities + ['S₁']
entity_xs = np.linspace(-0.75, 0.75, len(all_middle_nodes))
for i, node in enumerate(all_middle_nodes):
    pos[node] = np.array([entity_xs[i], -0.25])

app_xs = np.linspace(-0.5, 0.5, len(apps))
for i, a in enumerate(apps):
    pos[a] = np.array([app_xs[i], -0.75])

# Define colors
color_map = {
    'person': '#00FF00',    # Green
    'user': '#00CED1',      # Bright teal
    'entity': '#FF7F7F',    # Light red
    'system': '#E6E6FA',    # Light purple
    'app': '#0000FF'        # Blue
}

# Draw nodes
node_colors = [color_map[G.nodes[node]['node_type']] for node in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000)

# Function to get curve direction based on target position
def get_curve_direction(source_pos, target_pos):
    if target_pos[0] < source_pos[0]:
        return -0.2
    elif target_pos[0] > source_pos[0]:
        return 0.2
    return 0.1

# Draw edges with curves
solid_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d.get('style', 'solid') == 'solid']
dotted_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d.get('style') == 'dotted']

# Draw solid edges with curves
for edge in solid_edges:
    rad = get_curve_direction(pos[edge[0]], pos[edge[1]])
    nx.draw_networkx_edges(G, pos,
                          edgelist=[edge],
                          edge_color='#404040',
                          width=2,
                          connectionstyle=f'arc3,rad={rad}',
                          arrows=True,
                          arrowstyle='-|>',
                          arrowsize=50,
                          min_source_margin=30,
                          min_target_margin=30)

# Draw dotted edges with curves
for edge in dotted_edges:
    rad = get_curve_direction(pos[edge[0]], pos[edge[1]])
    nx.draw_networkx_edges(G, pos,
                          edgelist=[edge],
                          edge_color='#404040',
                          width=2,
                          style='dotted',
                          connectionstyle=f'arc3,rad={rad}',
                          arrows=True,
                          arrowstyle='-|>',
                          arrowsize=50,
                          min_source_margin=30,
                          min_target_margin=30)

# Add edge labels
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos,
                            edge_labels=edge_labels,
                            font_size=8,
                            label_pos=0.7,
                            bbox=dict(facecolor='white', 
                                    edgecolor='none', 
                                    alpha=0.6))

# Create node label colors
label_colors = {node: 'white' if G.nodes[node]['node_type'] == 'app' else 'black' for node in G.nodes()}

# Draw node labels
nx.draw_networkx_labels(G, pos,
                       font_size=14,
                       font_family='DejaVu Sans',
                       font_weight='bold',
                       font_color=label_colors)

plt.axis('off')
plt.margins(0.2)
plt.tight_layout()
plt.show()