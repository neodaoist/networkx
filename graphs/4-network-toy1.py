import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import yaml
import re

def parse_yaml_data(file_path):
    """
    Parse the YAML data from the file.
    This is a simplified parser for the provided structure.
    """
    with open(file_path, 'r') as file:
        data = file.read()
    
    # Since the YAML might have some non-standard formatting, 
    # we'll extract the necessary information manually
    
    # Initialize data structures
    network_data = {
        'people': [],
        'builder_teams': [],
        'apps': [],
        'contracts': [],
        'connections': []
    }
    
    # Extract people
    people_pattern = re.compile(r'- id: (P\d+)')
    network_data['people'] = people_pattern.findall(data)
    
    # Extract builder teams
    builder_pattern = re.compile(r'- id: (B\d+)')
    network_data['builder_teams'] = builder_pattern.findall(data)
    
    # Extract apps
    app_pattern = re.compile(r'- id: (A\d+)')
    network_data['apps'] = app_pattern.findall(data)
    
    # Extract contracts (including tokens)
    contract_pattern = re.compile(r'- id: ([A-Za-z0-9]+)\n\s+type:')
    all_contracts = contract_pattern.findall(data)
    
    # Additional LP contract from transactions
    lp4_pattern = re.compile(r'id: (LP4)')
    lp4_matches = lp4_pattern.findall(data)
    if lp4_matches:
        all_contracts += lp4_matches
    
    # Token pattern
    token_pattern = re.compile(r'- id: (T\d+)')
    tokens = token_pattern.findall(data)
    
    network_data['contracts'] = list(set(all_contracts + tokens))
    
    # Extract connections
    
    # People holding tokens
    for person in network_data['people']:
        for token in tokens:
            network_data['connections'].append({
                'source': person,
                'target': token,
                'type': 'holds'
            })
    
    # Builder teams deploying apps
    for i, builder in enumerate(network_data['builder_teams']):
        if i < len(network_data['apps']):
            network_data['connections'].append({
                'source': builder,
                'target': network_data['apps'][i],
                'type': 'deploys'
            })
    
    # Apps containing contracts
    app_contracts = {
        'A1': ['UniV2Factory', 'UniV2Router', 'LP1', 'LP2', 'LP3'],
        'A2': ['ClankerFactory', 'ClankerLocker'],
        'A3': ['BasePaintBrush721', 'BasePaintCanvas1155']
    }
    
    for app, contracts in app_contracts.items():
        for contract in contracts:
            network_data['connections'].append({
                'source': app,
                'target': contract,
                'type': 'contains'
            })
    
    # Liquidity pool pairs
    lp_pairs = {
        'LP1': ['T1', 'T2'],
        'LP2': ['T1', 'T3'],
        'LP3': ['T2', 'T3'],
        'LP4': ['T1', 'T4'] if 'LP4' in all_contracts else []
    }
    
    for lp, tokens in lp_pairs.items():
        for token in tokens:
            if token in network_data['contracts']:
                network_data['connections'].append({
                    'source': lp,
                    'target': token,
                    'type': 'pairs'
                })
    
    # Transaction connections from blocks
    transaction_pattern = re.compile(r'initiator: (P\d+).*?target: ([A-Za-z0-9]+)', re.DOTALL)
    transactions = transaction_pattern.findall(data)
    
    for initiator, target in transactions:
        network_data['connections'].append({
            'source': initiator,
            'target': target,
            'type': 'calls'
        })
    
    return network_data

def create_network_graph(network_data):
    """
    Create a NetworkX graph from the parsed data.
    """
    G = nx.Graph()
    
    # Add nodes with attributes
    for person in network_data['people']:
        G.add_node(person, type='person')
    
    for builder in network_data['builder_teams']:
        G.add_node(builder, type='builder')
    
    for app in network_data['apps']:
        G.add_node(app, type='app')
    
    for contract in network_data['contracts']:
        # Determine contract type
        if contract.startswith('T'):
            node_type = 'token'
        elif contract.startswith('LP'):
            node_type = 'liquidityPool'
        else:
            node_type = 'contract'
        
        G.add_node(contract, type=node_type)
    
    # Add edges with attributes
    for conn in network_data['connections']:
        # Skip invalid connections (e.g., with empty targets)
        if not conn['source'] or not conn['target']:
            continue
        
        G.add_edge(conn['source'], conn['target'], type=conn['type'])
    
    return G

def visualize_network(G):
    """
    Visualize the network using matplotlib with nodes of the same type
    aligned horizontally in separate rows.
    """
    plt.figure(figsize=(16, 12))
    
    # Define node colors based on type
    node_colors = {
        'person': 'skyblue',
        'builder': 'orange',
        'app': 'green',
        'token': 'red',
        'contract': 'purple',
        'liquidityPool': 'brown'
    }
    
    # Create a custom position dictionary with nodes arranged by type
    pos = {}
    
    # Group nodes by type
    type_to_nodes = {
        'person': [],
        'token': [],
        'contract': [],
        'app': [],
        'builder': [],
        'liquidityPool': []
    }
    
    for node, attrs in G.nodes(data=True):
        node_type = attrs['type']
        type_to_nodes[node_type].append(node)
    
    # Define vertical positions for each type (in the requested order)
    type_y_positions = {
        'person': 5,      # P nodes at the top
        'token': 4,       # T nodes second
        'contract': 3,    # C nodes third
        'liquidityPool': 3,  # LP nodes on same row as contracts
        'app': 2,         # A nodes fourth
        'builder': 1      # B nodes at the bottom
    }
    
    # Position each node
    for node_type, nodes in type_to_nodes.items():
        if not nodes:
            continue
            
        # Get y position for this type
        y_pos = type_y_positions[node_type]
        
        # Special ordering for contracts and liquidity pools
        if node_type == 'contract' or node_type == 'liquidityPool':
            # Define specific order for contracts with grouping
            contract_order = [
                'UniV2Factory', 'UniV2Router', 
                'LP1', 'LP2', 'LP3', 'LP4',
                'ClankerFactory', 'ClankerLocker', 
                'BasePaintBrush721', 'BasePaintCanvas1155'
            ]
            
            # Filter to only include nodes that exist in the graph
            ordered_nodes = [n for n in contract_order if n in nodes]
            
            # Add any remaining nodes that weren't in the predefined order
            other_nodes = [n for n in nodes if n not in contract_order]
            ordered_nodes.extend(other_nodes)
            
            # Create custom positions with grouping
            x_positions = {}
            y_positions = {}  # Add custom y positions
            
            # Group 1: Uniswap contracts - stack vertically
            # Router at top, LPs in middle, Factory at bottom
            uniswap_base_x = 0.15  # Base x position for Uniswap group
            
            # UniV2Router (top)
            if 'UniV2Router' in ordered_nodes:
                x_positions['UniV2Router'] = uniswap_base_x
                y_positions['UniV2Router'] = y_pos + 0.4  # Double vertical spacing (from 0.1 to 0.2)
            
            # LP nodes (middle row)
            lp_nodes = [n for n in ordered_nodes if n.startswith('LP')]
            for i, node in enumerate(lp_nodes):
                x_positions[node] = uniswap_base_x + (i * 0.08)  # More spacing between LPs
                y_positions[node] = y_pos  # Keep at normal y position
            
            # UniV2Factory (bottom)
            if 'UniV2Factory' in ordered_nodes:
                x_positions['UniV2Factory'] = uniswap_base_x
                y_positions['UniV2Factory'] = y_pos - 0.4  # Double vertical spacing (from -0.1 to -0.2)
            
            # Group 2: Clanker contracts - increase spacing
            clanker_group = ['ClankerFactory', 'ClankerLocker']
            clanker_nodes = [n for n in clanker_group if n in ordered_nodes]
            for i, node in enumerate(clanker_nodes):
                x_positions[node] = 0.5 + (i * 0.15)  # Increased spacing (was 0.06)
                y_positions[node] = y_pos  # Keep at normal y position
            
            # Group 3: BasePaint contracts - increase spacing
            basepaint_group = ['BasePaintBrush721', 'BasePaintCanvas1155']
            basepaint_nodes = [n for n in basepaint_group if n in ordered_nodes]
            for i, node in enumerate(basepaint_nodes):
                x_positions[node] = 0.8 + (i * 0.1875)  # Increased spacing by 50% (from 0.15 to 0.225)
                y_positions[node] = y_pos  # Keep at normal y position
            
            # Add other nodes at the end if any
            current_max_x = max(x_positions.values()) if x_positions else 0.1
            for i, node in enumerate([n for n in ordered_nodes if n not in x_positions]):
                x_positions[node] = current_max_x + 0.08 * (i + 1)
                y_positions[node] = y_pos  # Default y position
            
            # Apply positions with custom y-coordinates where specified
            for node in ordered_nodes:
                node_y = y_positions.get(node, y_pos)  # Use custom y if available, else default
                pos[node] = (x_positions[node], node_y)
        
        # Special ordering for tokens
        elif node_type == 'token':
            # Define specific order for tokens
            token_order = ['T1', 'T2', 'T3', 'T4']
            
            # Filter to only include tokens that exist in the graph
            ordered_tokens = [t for t in token_order if t in nodes]
            
            # Add any remaining tokens that weren't in the predefined order
            other_tokens = [t for t in nodes if t not in token_order]
            ordered_tokens.extend(other_tokens)
            
            # Distribute tokens evenly along x-axis in the specified order
            x_positions = np.linspace(0, 1, len(ordered_tokens))
            for i, token in enumerate(ordered_tokens):
                pos[token] = (x_positions[i], y_pos)
        
        else:
            # Distribute nodes evenly along x-axis for other types
            x_positions = np.linspace(0, 1, len(nodes))
            for i, node in enumerate(nodes):
                pos[node] = (x_positions[i], y_pos)
    
    # Get colors for each node
    colors = [node_colors[G.nodes[node]['type']] for node in G.nodes]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=colors, alpha=0.8)
    
    # Draw edges with different styles based on type
    edge_styles = {
        'holds': 'solid',
        'deploys': 'dashed',
        'contains': 'dotted',
        'pairs': 'dashdot',
        'calls': (0, (3, 1, 1, 1))  # More complex dash pattern
    }
    
    for edge_type, style in edge_styles.items():
        edges = [(u, v) for u, v, data in G.edges(data=True) if data['type'] == edge_type]
        nx.draw_networkx_edges(G, pos, edgelist=edges, style=style, alpha=0.6, width=1.5)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Add legend for node types with title and better positioning
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                       markersize=10, label=node_type) 
                       for node_type, color in node_colors.items()]
    
    # Position legend halfway between its previous position and P10's right edge
    plt.legend(handles=legend_elements, loc='upper right', title='Legend', 
              bbox_to_anchor=(1.07, 1), borderaxespad=1)
    
    # Add title and remove axis
    plt.title('Onchain Economic Network', fontsize=20)
    plt.axis('off')
    
    # Save the figure
    plt.savefig('network_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_matrices(G):
    """
    Calculate adjacency matrix, eigenvector and Laplacian matrix eigenvalues.
    """
    # Convert to numpy array (adjacency matrix)
    A = nx.to_numpy_array(G)
    
    # Calculate eigenvector centrality using NumPy
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # The eigenvector corresponding to the largest eigenvalue
    idx = np.argmax(eigenvalues)
    eigenvector_centrality = eigenvectors[:, idx]
    
    # Normalize eigenvector centrality to sum to 1 (instead of L2 norm)
    eigenvector_centrality = np.abs(eigenvector_centrality)  # Take absolute values
    eigenvector_centrality = eigenvector_centrality / np.sum(eigenvector_centrality)
    
    # Create Laplacian matrix
    L = nx.laplacian_matrix(G).toarray()
    
    # Calculate eigenvalues of the Laplacian matrix
    laplacian_eigenvalues = np.linalg.eigvals(L)
    
    # Sort eigenvalues
    laplacian_eigenvalues = np.sort(laplacian_eigenvalues)
    
    return {
        'adjacency_matrix': A,
        'eigenvalues': eigenvalues,
        'eigenvector_centrality': eigenvector_centrality,
        'laplacian_matrix': L,
        'laplacian_eigenvalues': laplacian_eigenvalues
    }

def analyze_and_visualize_network(file_path):
    """
    Main function to analyze and visualize the network.
    """
    # Parse data
    network_data = parse_yaml_data(file_path)
    
    # Create network graph
    G = create_network_graph(network_data)
    
    # Print basic network statistics
    print("Network Statistics:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Network density: {nx.density(G):.4f}")
    print(f"Average clustering coefficient: {nx.average_clustering(G):.4f}")
        
    # Calculate matrices and eigenvalues
    results = calculate_matrices(G)
    
    # Print results
    print("\nFull Adjacency Matrix:")
    print(results['adjacency_matrix'])
    
    print("\nFull Laplacian Matrix:")
    print(results['laplacian_matrix'])
    
    print("\nEigenvector Centrality (all nodes):")
    node_centrality = dict(zip(G.nodes(), results['eigenvector_centrality']))
    sorted_centrality = sorted(node_centrality.items(), key=lambda x: abs(x[1]), reverse=True)
    for node, centrality in sorted_centrality:
        print(f"{node}: {centrality:.6f}")
    
    # Print the sum to verify it equals 1
    centrality_sum = sum(abs(c) for _, c in node_centrality.items())
    print(f"\nSum of centrality values: {centrality_sum:.6f}")
    
    print("\nLaplacian Eigenvalues (first 10):")
    print(results['laplacian_eigenvalues'][:10])
    
    # Visualize eigenvalues
    # plt.figure(figsize=(12, 6))
    
    # plt.subplot(1, 2, 1)
    # plt.plot(np.real(results['eigenvalues']), np.imag(results['eigenvalues']), 'bo')
    # plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    # plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
    # plt.title('Eigenvalues of Adjacency Matrix')
    # plt.xlabel('Real')
    # plt.ylabel('Imaginary')
    # plt.grid(True)
    
    # plt.subplot(1, 2, 2)
    # plt.plot(range(len(results['laplacian_eigenvalues'])), results['laplacian_eigenvalues'], 'go-')
    # plt.title('Eigenvalues of Laplacian Matrix')
    # plt.xlabel('Index')
    # plt.ylabel('Eigenvalue')
    # plt.grid(True)
    
    # plt.tight_layout()
    # plt.savefig('eigenvalues_plot.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # Visualize network
    visualize_network(G)
    
    return G, results

if __name__ == "__main__":
    # Replace with the actual file path
    file_path = "data/network-toy1.yaml"
    G, results = analyze_and_visualize_network(file_path)