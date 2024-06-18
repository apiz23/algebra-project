import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Button

def main():
    root = Tk()
    root.title("Choose Option")

    label = Label(root, text="Choose an option:")
    label.pack()

    button1 = Button(root, text="Option 1: Show Network Traffic Flow", command=prob1)
    button1.pack()
    button2 = Button(root, text="Option 1: Show Network Traffic Flow", command=prob2)
    button2.pack()

    root.mainloop()

def prob1():

    matrix_A = np.array([
        [0, 22, 28, 0, 0],
        [22, 0, 0, 44, 0],
        [28, 0, 0, 72, 92],
        [0, 44, 72, 0, 53],
        [0, 0, 92, 53, 0]
    ])

    eigenvalues, eigenvectors = np.linalg.eig(matrix_A)

    print(f"Eigen Values: {eigenvalues}\nEigenvectors: {eigenvectors}")

    Gph = nx.Graph()
    nodes = ['Jasin', 'Merlimau', 'Batu Berendam', 'Pagoh', 'Segamat']
    edges = [
        ('Jasin', 'Merlimau', 22),
        ('Jasin', 'Batu Berendam', 28),
        ('Merlimau', 'Pagoh', 44),
        ('Batu Berendam', 'Pagoh', 72),
        ('Batu Berendam', 'Segamat', 92),
        ('Pagoh', 'Segamat', 53)
    ]

    Gph.add_nodes_from(nodes)
    Gph.add_weighted_edges_from(edges)

    pos = nx.spring_layout(Gph)
    nx.draw(Gph, pos, with_labels=True, node_color='lightblue', node_size=1500, font_size=12, font_weight='bold')
    edge_labels = nx.get_edge_attributes(Gph, 'weight')
    nx.draw_networkx_edge_labels(Gph, pos, edge_labels=edge_labels)
    plt.title('Network Traffic Flow')
    plt.show()

    smallest_nonzero_eigenvalue_index = np.argsort(eigenvalues)[1]
    critical_eigenvector = eigenvectors[:, smallest_nonzero_eigenvalue_index]


    critical_links = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i + 1, len(nodes)) if abs(critical_eigenvector[i] - critical_eigenvector[j]) > 0.1]

    nx.draw(Gph, pos, with_labels=True, node_color='lightblue', node_size=1500, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(Gph, pos, edgelist=critical_links, edge_color='r', width=2)

    print("Critical Eigenvector:", critical_eigenvector)

    plt.title('Critical Links in the Network')
    plt.show()

def prob2():
    A = np.array([
        [0, 15, 20, 0, 0],
        [15, 0, 0, 25, 0],
        [20, 0, 0, 30, 35],
        [0, 25, 30, 0, 40],
        [0, 0, 35, 40, 0]
    ])

    B = np.array([100, 80, 120, 90, 110])

    A_inv = np.linalg.inv(A)

    optimal_flow = np.dot(A_inv, B)

    print("Optimal Flow using Inverse Matrix Method:")
    for i, city in enumerate(['A', 'B', 'C', 'D', 'E']):
        print(f"{city}: {optimal_flow[i]} units")

main()
