# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:59:20 2018

@author: javi-
"""
import numpy as np
import pandas as pd
from numpy import matlib
from sklearn.preprocessing import normalize

import numba as nb

scores=[]; configuraciones = []; linkeos = []

def gravitational_clustering_numG(afinidades, delta, epsilon, p, num_clusters, h, unitary, masses=None, cl_test=False, normalization=False,
                                  connect=False, correccion_dim = False, conexiones = None, adaptativo = True, penalty=0, verbose = False,
                                  sparse_flag = False, diccionario=None, diccionario_reverso=None):
    '''
        %%SUMMARY:
            %Applies the modified gravitational clustering algorithm to dataset 
            %"data" in order to get a set of clusters.
    
        %%INPUT:
            %data = double[num_examples, num_vars]
                %data(i, j) -> j-th attribute of i-th example
            %delta = double -> Value of parameter delta (delta > 0)
            %epsilon = double -> Value of parameter epsilon (epsilon > 0)
                %Recommended value for epsilon = 2*delta
            %num_clusters = int -> Number of expected clusters (num_clusters > 0)
            %overlap = int
                %h = 1 -> Unit markovian model.
                %h = 2 -> O(1, 1/m) = (1/m)^p
                %h = 3 -> O(1, 1/m) = sqrt(1/m) / (sqrt(1/m) + max(1 - 1/m, 0))
                %h = 4 -> O(1, 1/m) = sin(pi/2 * (1/m)^p)
        %%OUTPUT:
            %clustering = int[num_examples]
                %clustering(i) -> Cluster the i-th example belongs to
            %real_time = double -> Real time the algorithm has been in execution (debug).
            %simulated_time = double -> Simulated time the clustering has
            %lasted.
    '''
    global linkeos
    _sparse_flag = sparse_flag

    num_particles = afinidades.shape[0]
    vivos = np.ones(num_particles, dtype = bool)
    
    if conexiones is None:
        conexiones = afinidades.copy()!=0
    if masses is None:
        masses = np.ones(num_particles, np.float64)

    masses = masses.reshape((len(masses),))
    masas_originales  = masses
    masses = masses / np.sum(masses)

    positions = afinidades.copy()
    
    clustering = np.arange(0, num_particles)
    t = 0
    num_iter = 1 

    afinidades_originales = afinidades.copy()
    delta_original = delta
    [_, clustering, positions, afinidades, masses, conexiones, num_particles] = \
        check_collisions(positions, clustering, masses, epsilon, conexiones, afinidades_originales=afinidades_originales,
                         afinidades = afinidades, connect = connect, verbose=verbose, correccion_dim=correccion_dim,
                         diccionario_reverso=diccionario_reverso, diccionario=diccionario, vivos=vivos, masas_originales=masas_originales,
                         linkeos=linkeos)
        
    best_conf = clustering
    best_conf_count = 0
    actual_conf_count = 0

    cmp_max = len(np.unique(clustering))
    primera = adaptativo
    num_particles = len(masses)

    while num_particles > 1:
        if afinidades.shape[0] > 1 and (np.sum(afinidades) - np.sum(np.diag(afinidades)) <= 10E-5):
            #print('Help me to name it!')
            return best_conf

        [velocities, dt, distancia_influencia] = compute_G(positions, masses, delta, p, np.int64(h), afinidades,
                                                           primera=np.bool(primera), simmetry=np.bool(False), penalty=np.int64(penalty))
        if primera:
            primera = False
            delta = delta_original

        t = t + dt
        positions = positions + velocities
        actual_conf_count = actual_conf_count + dt


        [shock, clustering_new, positions, afinidades, masses, conexiones, num_particles] = \
            check_collisions(positions, clustering, masses, epsilon, afinidades, connect, afinidades_originales=afinidades_originales,
                             afinidades = afinidades, correccion_dim=correccion_dim, verbose=verbose,
                             diccionario_reverso=diccionario_reverso, diccionario=diccionario, vivos = vivos,
                             masas_originales=masas_originales, target = num_clusters, linkeos=linkeos)



        if not cl_test and shock:
            if num_clusters > 1 and num_particles <= num_clusters:
                return clustering_new
            elif (not unitary) and  num_clusters <= 1 and (actual_conf_count*np.log2(num_particles) >= best_conf_count):
                best_conf = clustering
                best_conf_count = actual_conf_count * np.log2(num_particles)

                actual_conf_count = 0
            elif unitary and  num_clusters <= 1 and (actual_conf_count >= best_conf_count):
                best_conf = clustering
                best_conf_count = actual_conf_count

                actual_conf_count = 0


            clustering = clustering_new
            num_iter = num_iter + 1
            num_particles = len(masses)

        else:
           #current_score = np.sum
           pass

    return best_conf

def check_collisions(positions, clustering_0, masses, epsilon, conexiones, connect, afinidades, afinidades_originales,
                     diccionario, diccionario_reverso, masas_originales, linkeos, verbose=False, correccion_dim = False, vivos=None, target=0):
    '''
    %%SUMMARY:
        %Checks wether two particles of the system have gotten into epsilon
        %units of each other, fusing them in that case.

    %%INPUT:
        %particles = bool[num_examples];
            %particles(i) -> i-th particle (hasn't been fused)
        %positions = double[num_examples, num_attributes] 
            %positions(i, :) -> Position of particle i
        %clustering = double[num_examples];
            %clustering(i) -> Cluster the i-th particle belongs to.
        %masses = double[num_examples];
            %masses(i) -> Mass of the i-th particle
        %epsilon = double -> Value of parameter epsilon (epsilon > 0)
        
    %%OUTPUT:
        %any_collision = bool -> Indicates if there has been any collision
    '''
    any_collision = False
    num_particles = positions.shape[0]
    cols = positions.shape[1]
    clustering = clustering_0.copy()
    if not correccion_dim:
        correccion_dimensional = 1
    else:
        correccion_dimensional = np.sqrt(positions.shape[1])
    #verbose:
    #    score = [modularity_density(conexiones_originales, clustering), clustering]
    candidatos = np.nonzero(np.tril(afinidades, -1))
    num_candidatos = len(candidatos[0])

    ix = 0

    while ix < num_candidatos:
        i = candidatos[0][ix]
        j = candidatos[1][ix]
        ix += 1

        if  (positions[i,j]  >= positions[j,j]) or (positions[j,i]  >= positions[i,i]):
            any_collision = True

            afinidades, conexiones, lider, no_lider = fuse(afinidades, conexiones, masses, i, j, clustering, linkeos=linkeos,
                                                           afinidades_originales=afinidades_originales, verbose=verbose, vivos=vivos, masas_originales=masas_originales,
                                                           diccionario=diccionario, diccionario_reverso=diccionario_reverso)

            positions[lider, :] = (masses[i] * positions[i, :] + masses[j] * positions[j, :]) / (masses[i] + masses[j])
            positions[:, lider] = (masses[i] * positions[:, i] + masses[j] * positions[:, j]) / (masses[i] + masses[j])

            prev_cluster = clustering[diccionario[list(diccionario_reverso.values())[no_lider]]]
            clustering[clustering == prev_cluster] = clustering[diccionario[list(diccionario_reverso.values())[lider]]]
            masses[lider] = masses[lider] + masses[no_lider]
            configuraciones.append(clustering)


            del diccionario_reverso[list(diccionario_reverso.keys())[no_lider]]
            masses = np.transpose(np.delete(masses, no_lider))
            masses = np.array(masses).reshape([masses.shape[0],])
            positions = np.delete(positions, no_lider, axis=0)
            positions = np.delete(positions, no_lider, axis=1)

            num_particles = num_particles - 1

            candidatos[0][candidatos[0] > no_lider] += -1
            candidatos[1][candidatos[1] > no_lider] += -1
            conservar = np.logical_not(np.logical_or(candidatos[0] == no_lider, candidatos[1] == no_lider))
            candidatos = (candidatos[0][conservar], candidatos[1][conservar])
            num_candidatos = np.sum(conservar)

            #if verbose:
            #    print('Score: ' + str(score[0]))
            #    scores.append(score[0])
            #if score2[0] > score[0]:
            #    score = score2
            if num_particles == target:
                return any_collision, clustering, positions, afinidades, masses, conexiones, num_particles


    return any_collision, clustering, positions, afinidades, masses, conexiones, num_particles

def fuse(afinidades, conexiones, masses, i, j, clustering, afinidades_originales,  diccionario, diccionario_reverso, vivos, masas_originales,linkeos=(), verbose = False):
    if masses[i] > masses[j]:
        #if verbose:
        #   print('Masas', masses[i], masses[j])
        lider =  i; no_lider = j
    elif masses[i] < masses[j]:
        #if verbose:
        #    print('Masas',masses[j], masses[i])
        lider = j; no_lider = i
    elif np.sum(conexiones[i,:]) >= np.sum(conexiones[j,:]):
        #if verbose:
        #    print('Masas',masses[i], masses[j])
        lider = i; no_lider = j
    else:
        #if verbose:
        #    print('Masas',masses[j], masses[i])
        lider = j; no_lider = i
    
    #if verbose:
    #    print('Conexion entre ambos: ', conexiones[lider, no_lider])

    indexado = np.ones(conexiones.shape[0], dtype=bool)
    indexado[no_lider] = False
    
    cluster_lider = clustering == clustering[diccionario[list(diccionario_reverso.values())[lider]]]
    cluster_no_lider = clustering == clustering[diccionario[list(diccionario_reverso.values())[no_lider]]]
    cluster_total = cluster_lider + cluster_no_lider
    
    masas_usar = masas_originales[cluster_total]
    masas_usar = masas_usar / np.sum(masas_usar)
    masas_usar = masas_usar.reshape((len(masas_usar), 1))

    raf = np.squeeze(np.array(np.sum(np.multiply(afinidades_originales[cluster_total, :], matlib.repmat(masas_usar, 1, afinidades_originales.shape[1])), axis = 0)))
    caf = np.squeeze(np.array(np.sum(np.multiply(afinidades_originales[:, cluster_total], matlib.repmat(np.transpose(masas_usar), afinidades_originales.shape[1], 1)), axis = 1)))
    afinidades[lider, :] = compute_inter_cluster_affinities(raf, clustering, diccionario, diccionario_reverso, len(diccionario_reverso))
    afinidades[:, lider] = compute_inter_cluster_affinities(caf, clustering, diccionario, diccionario_reverso, len(diccionario_reverso))

    vivos[diccionario[list(diccionario_reverso.values())[no_lider]]] = False

    conexiones_select = conexiones[indexado, :]
    conexiones_select = conexiones_select[:, indexado]
    
    afinidades_select = afinidades[indexado, :]
    afinidades_select = afinidades_select[:, indexado]

    if verbose:
        linkeos.append([diccionario[list(diccionario_reverso.values())[lider]]*1.0,
                        diccionario[list(diccionario_reverso.values())[no_lider]]*1.0, 1.0,
                        np.sum(cluster_total)*1.0])
        #print("Fusiono, lider: ", list(diccionario_reverso.values())[lider], " no lider: ", list(diccionario_reverso.values())[no_lider])
    
    return afinidades_select, conexiones_select, lider, no_lider

def compute_inter_cluster_affinities(vAffinity, clustering, diccionario, diccionario_reverso, vtam):
    '''

    :param afinity_row:
    :param clustering:
    :return:
    '''
    res = np.zeros((vtam,))
    comunidades = np.unique(clustering)

    for c in comunidades:
        miembros_c = np.equal(clustering, c)
        af_c = np.sum(vAffinity[miembros_c])

        res[list(diccionario_reverso.values()).index(list(diccionario.keys())[c])] = af_c


    return res

@nb.jit(nb.types.Tuple((nb.float64[:,:], nb.float64, nb.float64))\
                    (nb.float64[:,:], nb.float64[:], nb.float64, nb.int64,
                                        nb.int64, nb.float64[:,:], nb.boolean, nb.boolean, nb.int64), nopython= True, cache=True, fastmath=True)
def compute_G(positions, masses, delta, p, h, afinidades, primera = True, simmetry = True, penalty = 1):
    '''
    Computes the gravitational attraction for all particles.
    '''
    num_particles = positions.shape[0]
    
    velocities = np.zeros(positions.shape, np.float64)
    velocity_modulus = np.zeros(positions.shape[0], np.float64)
    min_shock_distance = 1

    for i in range(num_particles):
        utiles = afinidades[i,:] > 0
        utiles[i] = False

        saux = np.sum(utiles)
        acum = np.zeros((1, saux), dtype=np.float64)
        h_masses = np.zeros(acum.shape, dtype=np.float64)

        dis_aux = np.min(np.abs(positions[i,:] -np.diag(positions))[np.arange(len(positions[i,:]))!=i])
        if dis_aux < min_shock_distance:
            min_shock_distance = dis_aux

        x = masses[i]# / np.sum(masses)
        y = masses#[np.arange(0,positions.shape[0]) != i]# / np.sum(masses)

        h_masses += np.power(x * y[utiles], p)
        h_masses = np.multiply(h_masses, afinidades[utiles, i].reshape(h_masses.shape)) / (x**(penalty+1))
        position_other = positions[utiles, :]
            
        distances = np.zeros(position_other.shape)
        for j in range(position_other.shape[0]):
            distances[j] = -(positions[i] - position_other[j]) #Velocidad signo contrario de la distancia.
                
        modules = np.sum(np.multiply(distances, distances), axis=1)
        acum += np.sum(np.multiply(h_masses, 1/(modules+10E-6)), axis=0)

        #velocities[i,utiles] = acum

        slice_veloc = velocities[i,:]
        slice_veloc[utiles] = acum
        velocities[i, :] = slice_veloc
   
    velocity_modulus = np.sum(np.abs(velocities), axis=1)

    if primera:
        delta = min_shock_distance

    dt_squared = delta / np.max(velocity_modulus+10E-3)
    velocities = velocities * dt_squared
    dt = np.sqrt(dt_squared)

    return velocities, dt, min_shock_distance
        
#######################################################
#######################################################
def linkage2mat(df):
    rows, cols = df.shape
    uns = list(pd.unique(df.iloc[:,0]))
    uns.extend(list(pd.unique(df.iloc[:,1])))
    uns = np.unique(uns)
    size1 = len(uns)
    res = np.zeros([size1, size1])
    contador = 1
    
    for i in range(rows):
        source = df.iloc[i,0]
        target = df.iloc[i,1]
        res[source, target] = contador
        res[target, source] = contador
        contador += 1

    return res
    
    
def process_basic_dendrogram(dendro, clustering):
    contador = np.array(dendro).shape[0]+1; contador_fila = 0
    traduccion = {}
    res = dendro.copy()
    anterior = 0
    colores = list(clustering)
    
    for row in dendro:
        lider = row[0]
        no_lider = row[1]
        try:
            lider_traducido = traduccion[lider]
            res[contador_fila][0] = lider_traducido
            traduccion[lider] = contador
        except KeyError:
            traduccion[lider] = contador
            
        try:
            res[contador_fila][1] = traduccion[no_lider]
        except KeyError:
            pass
    
        colores.append(clustering[int(lider)])        
        res[contador_fila][2] = row[2] + anterior
        anterior = res[contador_fila][2]
        contador += 1        
        contador_fila += 1
    
    res = np.array(res)

    return res, colores
        
def delete_row(mat, i):
    return np.delete(mat, (i), axis=0)
    
def delete_col_lil(mat, i):
    columnas = list(np.arange(0,mat.shape[1]))
    columnas.remove(i)
    
    return mat[:,columnas]

# =============================================================================
# ~ Quality network measures
# =============================================================================
    
def social_cohesion(nodes, clustering):
    '''
    This has never worked well.
    '''
    res = 0
    for cluster in np.unique(clustering):
        select = clustering==cluster
        internas = np.sum(nodes[select][:, select]) - np.sum(select)
        externas = np.sum(nodes[select][:, np.logical_not(select)])
        
        if externas == 0:
            res += 0
        else:
            res += internas / externas / np.sum(nodes[select, select] != 0)
    
    return res - res  / len(np.unique(clustering))

def modularity_density(conexions0, clustering):
    '''
    Check google scholar.
    '''
    comunidades = np.unique(clustering)
    res = 0
    conexiones = conexions0# - np.identity(conexions0.shape[0])
    
    for ix, com in enumerate(comunidades):
        nodos_com = clustering == com
        nodos_outro = clustering != com
        
        otras = np.unique(clustering)
        otras = np.delete(otras, ix)
        
        inter_links = conexiones[nodos_com, nodos_com]
        #outro_links = conexiones[nodos_com, nodos_outro]
        
        E = np.sum(conexiones)
        E_ci_in = np.sum(inter_links)
        
        density_i = 2*np.sum(inter_links>0)
        termino_1 = E_ci_in / E * density_i
        termino_2 = 0
        
        for jx, otra in enumerate(otras):
            otra_nodos = clustering == otra
            
            E_ci_cj = np.sum(conexiones[nodos_com][:, otra_nodos])
            d_ci_cj = np.sum(conexiones[nodos_com][:, otra_nodos] > 0) / (np.sum(otra_nodos) + np.sum(nodos_com))
            
            E_ci_out = np.sum(conexiones[nodos_com][:, nodos_outro])
            E_out_ci = np.sum(conexiones[nodos_outro][:, nodos_com])
            
            termino_cj =  E_ci_cj / E * d_ci_cj - (E_ci_in + E_out_ci) * (E_ci_in + E_ci_out)/(E*E)*density_i*density_i
            
            termino_2 += termino_cj
            
        res += termino_1 + termino_2
        
    return res
    
    
def truncate(x, d):
    return int(x*(10.0**d))/(10.0**d)  