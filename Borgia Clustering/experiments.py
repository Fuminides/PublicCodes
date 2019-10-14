# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:53:11 2019

@author: javi-
"""

import temporal_analysis as ta
import speech_to_adjacency as sa
import borgia_clustering as bc
import numpy as np
import pandas as pd
import networkx as nx

# =============================================================================
# ~ AUXILIARS
# =============================================================================
def top_k_terms(semantics_terms, k = 5):
    '''
    Returns the K most important terms globally.
    '''
    import operator
    
    sum_semantics = [(x, np.sum(semantics_terms[x])) for x in semantics_terms.keys()]
    sorted_x = sorted(sum_semantics, key=operator.itemgetter(1), reverse=True)
    
    return sorted_x[0:k]

# =============================================================================
# ~ POST PROCESS RESULTS (GRAPHICS) 
# =============================================================================

def measure_time(adjacency_list, control, N = 1, sr = 1):
    '''
    Measures the time for an adjacency list, subsampling the sr% edges. It does it N times. 
    For the control parameter check the Borgia algorithm code.
    '''
    import time
    from random import sample
    import networkx as nx
    tiempos = [0]*N
    edges_len = adjacency_list.shape[0]

    for i in range(N):
        muestra = sample(range(edges_len), int(sr * edges_len))
        array_affinity, array_connection, df_affinity, diccionario_masas, _, _, _ = bc.full_process_csv(adjacency_list.iloc[muestra,:])
        masses = np.sum(array_connection,1).reshape((array_connection.shape[0],1))
        G = nx.from_pandas_adjacency(df_affinity)

        while not nx.is_connected(G):
            print('Fail!')
            muestra = sample(range(edges_len), int(sr * edges_len))
            sarray_affinity, array_connection, df_affinity, diccionario_masas, _, _, _ = bc.full_process_csv(adjacency_list.iloc[muestra,:])
            masses = np.sum(array_connection,1).reshape((array_connection.shape[0],1))
            G = nx.from_pandas_adjacency(df_affinity)

        start = time.time()
        try:
            bc.borgia_clustering(array_affinity, masses, bc.control)
        except ZeroDivisionError:
            pass
        end = time.time()
        tiempos[i] = (end - start)
        print(tiempos[i])

    return tiempos

def complete_measure(data_set, tiempos = np.arange(0.1, 1.01, 0.1), N = 30, resultado = 'result_big.csv'):
    '''
    Returns the execution times for a range of subsamples and N pop. Check the paper to see the result.
    '''
    tiempos_df = pd.DataFrame(np.zeros((N, len(tiempos))))

    for i in range(len(tiempos)):
        print(str(i) + ' out of ' + str(len(tiempos)))
        ans = measure_time(data_set, bc.control, N=N, sr=tiempos[i])
        tiempos_df.iloc[:,i] = ans

    tiempos_df.to_csv(resultado)

    return tiempos_df


def calc_affinities(array_connection, df_affinity, masas):
    '''
    Calculates The 3 affinities in the paper.

    :param red:
    :param semantics:
    :param array_connection:
    :param dic_masas:
    :return:
    '''
    import affinity as af

    df_connection = df_affinity.copy()
    df_connection[:] = array_connection

    best_friend = df_affinity

    df_important_ally = df_affinity.copy()
    important_ally = af.connexion2affinity(array_connection, af.affinity_ally)
    df_important_ally[:] = important_ally

    df_maquiavelo = df_affinity.copy()
    maquiavelo = af.connexion2affinity(array_connection, lambda x,y: af.affinity_maquiavelo(x, y, masas))
    df_maquiavelo[:] = maquiavelo

    return best_friend, df_important_ally, df_maquiavelo

def plot_results_figure(data, methods=('Greedy Modularity (GM)', 'Girvan-Newman (GN)', 'Label Propagation (LP)', 'Label Eigenvector (LE)', 'Lovaine (LO)', 'Borgia Clustering (BC)'),
                        datasets=('Zachary', 'Dolphin', 'Football', 'Polbooks'),
                        measures=('ARI', 'NMI')):
    '''
    An ad-hoc method to produce the same output as in fig 2 from the original paper.
    '''
    results_methods = [(methods[int(x/2)], measures[x%2]) for x in range(len(methods)*2)]
    datasets_names = []
    for i in range(len(datasets)* len(results_methods)):
        datasets_names.append(datasets[int(i / len(results_methods))])

    ans = list(zip(datasets_names, results_methods*len(datasets_names)))
    results_full_index = [(ans[x][0], ans[x][1][0], ans[x][1][1]) for x in range(len(ans))]
    index = pd.MultiIndex.from_tuples(results_full_index, names=['Dataset', 'Methods', 'Measure'])
    df = pd.DataFrame(data, index=index)
    df.columns = ['Value']

    return df