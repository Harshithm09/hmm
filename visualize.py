import matplotlib.pyplot as plt
import networkx as nx


def draw_states(A):

    G = nx.DiGraph()

    n = len(A)

    for i in range(n):
        G.add_node("State " + str(i))

    for i in range(n):
        for j in range(n):

            value = round(A[i][j], 2)

            G.add_edge(
                "State " + str(i),
                "State " + str(j),
                weight=value
            )

    pos = nx.circular_layout(G)

    nx.draw(G, pos,
            with_labels=True,
            node_size=2500)

    labels = nx.get_edge_attributes(G, 'weight')

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=labels
    )

    plt.title("HMM State Transition Diagram")

    plt.show()
