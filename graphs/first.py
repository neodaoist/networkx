import networkx as nx
import matplotlib.pyplot as plt

def main():
    G = nx.Graph()
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)
    G.add_edge(1, 2)
    G.add_edge(1, 3)

    print("hello world")

    nx.draw(G)
    plt.show()


if __name__ == "__main__":
    main()
