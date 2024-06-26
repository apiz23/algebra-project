import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pulp
import pyfiglet
import os

def main():            
    os.system("clear || cls")
    while True:
        ascii_title = pyfiglet.figlet_format("Algebra Project")
        print(ascii_title)

        print("Choose an option:")
        print("1. Show Network Traffic Flow")
        print("2. Optimize Manufacturing Production")
        print("3. Exit")

        choice = input("Enter your choice (1/2/3): ")

        if choice == '1':
            prob1()
            press_key_to_continue()
        elif choice == '2':
            prob2()
            press_key_to_continue()
        elif choice == '3':
            print("Exiting...")
            break 
        else:
            os.system("clear || cls")
            print("Invalid choice. Please enter 1, 2, or 3.")

def prob1():
    A = np.array([
        [0, 22, 28, 0, 0, 90],       # Jasin
        [22, 0, 0, 44, 0, 0],        # Merlimau
        [28, 0, 0, 72, 92, 0],       # Batu Berendam
        [0, 44, 72, 0, 53, 94],      # Pagoh
        [0, 0, 92, 53, 0, 0],        # Segamat
        [90, 0, 0, 94, 0, 0]         # Kluang
    ])

    eigenvalues, eigenvectors = np.linalg.eig(A)

    print(f"Eigen Values: {eigenvalues}\nEigenvectors: {eigenvectors}")

    Gph = nx.Graph()
    nodes = ['Jasin', 'Merlimau', 'Batu Berendam', 'Pagoh', 'Segamat', 'Kluang']
    edges = [
        ('Jasin', 'Merlimau', 22),
        ('Jasin', 'Batu Berendam', 28),
        ('Kluang', 'Pagoh', 90),
        ('Merlimau', 'Pagoh', 44),
        ('Batu Berendam', 'Pagoh', 72),
        ('Batu Berendam', 'Segamat', 92),
        ('Pagoh', 'Segamat', 53),
        ('Segamat', 'Kluang', 94)
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

    critical_links = [
        (nodes[i], nodes[j]) for i in range(len(nodes)) 
        for j in range(i + 1, len(nodes)) 
        if abs(critical_eigenvector[i] - critical_eigenvector[j]) > 0.1
    ]

    nx.draw(Gph, pos, with_labels=True, node_color='lightblue', node_size=1500, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(Gph, pos, edgelist=critical_links, edge_color='r', width=2)

    print("\nCritical Eigenvector:", critical_eigenvector)

    plt.title('Critical Links in the Network')
    plt.show()

    B = np.array([100, 80, 120, 90, 110, 100])
    
    A_inv = np.linalg.inv(A)

    optimal_flow = np.dot(A_inv, B)

    print("\nOptimal Flow using Inverse Matrix Method:")
    for i, city in enumerate(['Jasin', 'Merlimau', 'Batu Berendam', 'Pagoh', 'Segamat', 'Kluang']):
        print(f"{city}: {optimal_flow[i]:.4f}")

    #adjust to perform the Gauss Jordan Elimination
    A = np.array([
        [1, 22, 28, 0, 0, 90],       # Jasin
        [22, 1, 0, 44, 0, 0],        # Merlimau
        [28, 0, 1, 72, 92, 0],       # Batu Berendam
        [0, 44, 72, 1, 53, 94],      # Pagoh
        [0, 0, 92, 53, 1, 0],        # Segamat
        [90, 0, 0, 94, 0, 1]         # Kluang
    ])

    B = np.array([100, 80, 120, 90, 110, 100])

    n = len(B)
    augmented_matrix = np.hstack([A, B.reshape(-1, 1)])

    for i in range(n):
        if augmented_matrix[i, i] == 0:
            for k in range(i + 1, n):
                if augmented_matrix[k, i] != 0:
                    augmented_matrix[[i, k]] = augmented_matrix[[k, i]]
                    break
        diag_element = augmented_matrix[i, i]
        augmented_matrix[i] = augmented_matrix[i] / diag_element
        
        for j in range(n):
            if i != j:
                row_factor = augmented_matrix[j, i]
                augmented_matrix[j] -= row_factor * augmented_matrix[i]

    print("\nOptimal Flow using Gauss-Jordan Elimination:")
    for i, city in enumerate(['Jasin', 'Merlimau', 'Batu Berendam', 'Pagoh', 'Segamat', 'Kluang']):
        print(f"{city}: {augmented_matrix[i, -1]:.4f}")

def prob2():
    problem = pulp.LpProblem("Maximize Profit", pulp.LpMaximize)

    regular_mix = pulp.LpVariable("Regular Mix", lowBound=0, cat='Integer')
    deluxe_mix = pulp.LpVariable("Deluxe Mix", lowBound=0, cat='Integer')

    problem += 3 * regular_mix + 4 * deluxe_mix, "Total Profit"

    problem += 14 * regular_mix + 12 * deluxe_mix <= 840, "Peanuts Constraint"
    problem += 4 * regular_mix + 6 * deluxe_mix <= 360, "Cashews Constraint"

    problem.solve()

    print(f"Status: {pulp.LpStatus[problem.status]}\nOptimal number of Regular Mixes to produce: {pulp.value(regular_mix)}", )
    print(f"Optimal number of Deluxe Mixes to produce: {pulp.value(deluxe_mix)}\nMaximum Profit:{pulp.value(problem.objective)}", )

    fig, ax = plt.subplots()

    x1 = np.linspace(0, 60, 100)
    y1 = (840 - 14 * x1) / 12.0

    x2 = np.linspace(0, 100, 100)
    y2 = (360 - 4 * x2) / 6.0

    ax.plot(x1, y1, label='Peanuts Constraint')
    ax.plot(x2, y2, label='Cashews Constraint')

    x, y = np.meshgrid(x1, x2)
    ax.imshow(((14 * x + 12 * y <= 840) & (4 * x + 6 * y <= 360)).astype(int), extent=(x.min(), x.max(), y.min(), y.max()), origin="lower", cmap="Greys", alpha=0.3)

    ax.plot(pulp.value(regular_mix), pulp.value(deluxe_mix), 'ro', label='Optimal Solution')

    ax.set_xlabel('Regular Mix')
    ax.set_ylabel('Deluxe Mix')
    ax.set_title('Linear Programming: Feasible Region and Optimal Solution')
    ax.legend()

    plt.show()

def press_key_to_continue():
    input("Press Enter to continue...")
    os.system("clear || cls")

main()
