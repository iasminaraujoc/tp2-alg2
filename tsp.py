import sys
import os
import time
import networkx as nx
import heapq

from memory_profiler import memory_usage
from functools import partial

def dist_euclidiana(grafo, no1, no2):
    return ((grafo.nodes[no1]['coord'][0] - grafo.nodes[no2]['coord'][0]) ** 2 + (grafo.nodes[no1]['coord'][1] - grafo.nodes[no2]['coord'][1]) ** 2) ** 0.5

def criaGrafo(caminho):
    grafo = nx.Graph()
    with open(caminho, 'r') as file:
        linhas = file.readlines()

        for linha in linhas[6:-1]:
            if(linha != "EOF\n"):
                numero, x, y = linha.split()
                grafo.add_node(numero, coord=(float(x), float(y)))

    for u in grafo.nodes():
        for v in grafo.nodes():
            if u != v:
                grafo.add_edge(u, v, weight=dist_euclidiana(grafo, u, v))

    return grafo

def print_grafo(grafo):
    for u, v, data in grafo.edges(data=True):
        print(f"Nó {u} - Nó {v}: Distância Euclidiana = {data['weight']}")

def print_caminho(caminho):
    print("Caminho:")
    for node in caminho:
        print(f"Nó {node}")

# Algoritmo aproximativo para o TSP com fator de aproximação 2
def twice_around_the_tree(G):
    # Árvore geradora mínima
    agm = nx.minimum_spanning_tree(G)

    # Cria hipergrafo com AGM do grafo e adiciona arestas de "volta" (ida-volta)
    hipergrafo = nx.MultiGraph(agm)
    for u, v in agm.edges():
        hipergrafo.add_edge(v, u, weight=dist_euclidiana(G, v, u))

    # Circuito euleriano a partir do hipergrafo
    circuito_euleriano = list(nx.eulerian_circuit(hipergrafo))

    # Encontra o circuito hamiltoniano a partir do euleriano (remove repetições)
    visitados = set()
    circuito_hamiltoniano = []
    for u, _ in circuito_euleriano:
        if not u in visitados:
            circuito_hamiltoniano.append(u)
            visitados.add(u)

    # Calcula o custo total do cicuito hamiltoniano
    custo_total = sum(grafo[u][v]['weight'] for u, v in zip(circuito_hamiltoniano, circuito_hamiltoniano[1:]))

    return custo_total

# Algoritmo aproximativo para o TSP com fator de aproximação 1,5
def christofides(G):
    # Cria árvore geradora mínima
    agm = nx.minimum_spanning_tree(G)

    # Encontra vértices de grau ímpar e cria um subgrafo induzido com eles
    subg_impar = [v for v, grau in agm.degree() if grau % 2 == 1]
    subg_ind = nx.Graph(G.subgraph(subg_impar))

    # Encontra o match mínimo perfeito
    match = nx.min_weight_matching(subg_ind)

    # Cria hipergrafo e adiciona arestas do matching à árvore
    hipergrafo = nx.MultiGraph(agm)
    for u, v in match:
        hipergrafo.add_edge(u, v, weight=subg_ind[u][v]['weight'])

    # Circuito euleriano a partir do hipergrafo
    circuito_euleriano = list(nx.eulerian_circuit(hipergrafo))

    # Encontra o circuito hamiltoniano a partir do euleriano (remove repetições)
    visitados = set()
    circuito_hamiltoniano = []
    for u, _ in circuito_euleriano:
        if not u in visitados:
            circuito_hamiltoniano.append(u)
            visitados.add(u)

    # Calcula o custo total do cicuito hamiltoniano
    custo_total = sum(grafo[u][v]['weight'] for u, v in zip(circuito_hamiltoniano, circuito_hamiltoniano[1:]))

    return custo_total

# Algoritmo baseado no pseudocódigo disponibilizado em aula
def bound(s, A, n):
    bound_value = 0

    for i in range(n):
        # Encontra as duas arestas de menor peso incidentes em cada vértice
        min1, min2 = float('inf'), float('inf')

        for j in range(n):
            if j != i and A[i][j] < min1:
                min2 = min1
                min1 = A[i][j]
            elif j != i and A[i][j] < min2:
                min2 = A[i][j]

        bound_value += min1 + min2

    # Divide por 2, pois cada aresta seria contada duas vezes
    return bound_value / 2

# Solução inicial computada para ter um bound mais rápido
def solucao_inicial(grafo):
    sol = [0]
    while len(sol) < len(grafo.nodes()):
        proximo_vertice = None
        menor_peso = float('inf')
        for v in grafo.nodes():
            if v not in sol:
                arestas_v = grafo.edges(v, data=True)
                for _, vizinho, data in arestas_v:
                    if vizinho not in sol and data['weight'] < menor_peso:
                        menor_peso = data['weight']
                        proximo_vertice = vizinho
        sol.append(proximo_vertice)
    return sol

def branch_and_bound(A, n):
    root = (bound([0], A, n), 0, 0, [0])
    queue = [root]
    best = float('inf')
    sol = sol = solucao_inicial(grafo)

    while queue:
        node = heapq.heappop(queue)

        if node[1] > n:
            if best > node[2]:
                best = node[2]
                sol = node[3]
        elif node[0] < best:
            if node[1] < n:
                for k in range(1, n):
                    if k not in node[3] and A[node[3][0]][k] != 0 and bound(node[3] + [k], A, n) < best:
                        new_node = (bound(node[3] + [k], A, n), node[1] + 1, node[2] + A[node[3][0]][k], node[3] + [k])
                        heapq.heappush(queue, new_node)
            elif A[node[3][0]][0] != 0 and bound(node[3] + [0], A, n) < best:
                new_node = (bound(node[3] + [0], A, n), node[1] + 1, node[2] + A[node[3][0]][0], node[3] + [0])
                heapq.heappush(queue, new_node)

    return sol


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python script.py nome_do_arquivo.tsp")
        sys.exit(1)

    caminho = sys.argv[1]
    grafo = criaGrafo(caminho)

    #momento inicial para medir o tempo
    inicio = time.time()

    mem, custo_total = memory_usage(partial(twice_around_the_tree, grafo), interval = 0.1, max_usage=True, retval=True)

    fim = time.time()
    tempo = fim - inicio

    # Pega o nome do arquivo para coletar informações
    nome_arquivo = os.path.basename(caminho)
    nome_sem_extensao = os.path.splitext(nome_arquivo)[0]

    with open('tp2_datasets.txt', 'r') as file:
        linhas = file.readlines() 

    limiar = 0
    for linha in linhas[1:-1]:
        nome, tam, lim = linha.split()
        if(nome == nome_sem_extensao):
            print("Dataset: ", nome)
            limiar = float(lim)
            break

    # # Imprime as métricas obtidas
    print(f"custo obtido: {custo_total}")

    print("limiar: ", limiar)

    print("qualidade: ", custo_total/limiar)

    print("tempo total gasto: ", tempo)

    print("memória utilizada: ", mem, "MB")

    print("\n\n")


