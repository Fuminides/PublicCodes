# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 16:12:50 2019

@author: javi-
"""

# =============================================================================
# ~Affinity calculation
# =============================================================================
import numpy as np
from scipy import stats
from scipy import sparse


def affinity_most_important_ally(list1, list2):
    '''
    Calculates the affinity based on the % of common elements between to entities.
    '''
    def common_elements(list1, list2):
        result = []

        for i in range(len(list1)):
            result.append(min(list1[i], list2[i]))

        return result


    res = common_elements(list1, list2)

    if len(res) == 0:
        return 0,0
    else:
        return np.max(res)/np.max(list1), np.max(res)/np.max(list2)

def connexion2affinity(conex, af_func=affinity_most_important_ally):
    '''
    Given a connectivity matrix calculates a predefined affinity.
    '''
    rows, cols = conex.shape
    res = np.zeros(conex.shape)
    
    for i in range(rows):
        suj = conex[i,:]
        for j in range(cols):
            if i != j:
                suj2 = conex[j,:]
                afinidad1, afinidad2 = af_func(suj, suj2) #- 2 * conex[i,j] / cols
                res[i, j] = afinidad1
                res[j, i] = afinidad2
                
    return res

def connexion2affinity_network(conex):
    '''
    Given a connectivity matrix calculates the conex affinity.
    '''
    afinidad_base = connexion2affinity_important_friend(conex)
    rows, cols = conex.shape
    res = np.zeros(conex.shape)
    
    for i in range(rows):
        suj = afinidad_base[i,:]
        for j in range(cols):
            if i != j:
                suj2 = conex[j,:]
                afinidad1, afinidad2 = affinity_networking(suj, suj2, afinidad_base, i, j) #- 2 * conex[i,j] / cols
                res[i, j] = afinidad1
                res[j, i] = afinidad2
    
    return res

def connexion2affinity_important_friend(conex, csr = False):
    '''
    Given a connectivity matrix calculates the important friend affinity.
    '''
    from itertools import compress

    rows, cols = conex.shape
    res = np.zeros(conex.shape, np.float64)
    if csr:
        rows_copy = sparse.csr_matrix(conex)
        columns_copy = sparse.csc_matrix(conex)

    for i in range(rows):
        if not csr:
            suj = conex[i,:]
            candidatos = np.where(suj>0)[0]
        else:
            suj = rows_copy[i, :]
            candidatos = (suj>0).nonzero()[1]

        for jx, j in enumerate(candidatos):

            if i != j and (j >= i):
                if not csr:
                    suj2 = conex[:, j]
                else:
                    suj2 = columns_copy[:, j]


                afinidad1, afinidad2 = affinity_important_friend(suj, suj2, i, j)

                res[i, j] = afinidad1 #+ aux[0]
                res[j, i] = afinidad2 #+ aux[1]

    return res

def affinity_important_friend_ordinal(conex):
    '''
    Given a connectivity matrix calculates the important friend affinity. Then
    it ranks them according to the biggest values.
    '''
    res = connexion2affinity_important_friend(conex)
    for i in range(res.shape[0]):
        res[i,:] = stats.rankdata(res[i,:], 'max')
    
    return res

def affinity_friends(row, row2, i , j):
    '''
    Given to entities, calculates the affinity based on the most important difference
    between these two.
    '''        
    a1 = (np.abs(row - row2))
    index = np.maximum(row, row2)!=0
    a1 = 1 - a1[index] / np.maximum(row, row2)[index]
    a1 = np.mean(a1)
    return a1, a1

def affinity_basic(row, row2):
    '''
    Euclidean distances between to entities.
    '''        
    a1 = np.sum(np.abs(row - row2))
    
    return a1, a1

def affinity_enemy(row, row2):
    '''
    Return the most different affinity.
    '''        
    a1 = np.max(np.abs(row - row2))
    
    return a1, a1

def affinity_ally(list1, list2):
    '''
    Calculates the affinity based on the % of common elements between to entities.
    '''
    def common_elements(list1, list2):
        result = []
        
        for i in range(len(list1)):
            result.append(min(list1[i], list2[i]))
                
        return result


    res = common_elements(list1, list2)

    if len(res) == 0:
        return 0,0
    else:
        return np.sum(res) / np.sum(list1) , np.sum(res) / np.sum(list2)

        
def affinity_important_friend(row, row2, i, j):
    '''
    Calculates the affinity between two entities based on the % that on entity 
    represents over the other.
    '''
    try:
        a1 = row[j] / np.sum(row)
        a2 = row2[i] / np.sum(row2)

    except IndexError:
        a1 = row[0,j] / np.sum(row)
        a2 = row2[i,0] / np.sum(row2)

    return a1, a2

def affinity_maquiavelo(row, row2, grades):
    '''
    Calculates the afifnity between two entities based on the grades of their 
    connected nodes.
    '''
    x_prim = row > 0
    y_prim = row2 > 0
    
    ix = np.sum(grades[x_prim])
    iy = np.sum(grades[y_prim])
    
    res = 1 - abs(ix - iy) / max(ix, iy)

    return res, res

def affinity_networking(row, row2, affinities, x, y):
    '''
    Calculates the affinity between two particles based on a preexisting 
    affinity between their two social groups.
    '''
    x_friends = row > 0
    y_friends = row2 > 0
    
    fx = np.mean(affinities[x_friends, y])
    fy = np.mean(affinities[y_friends, x])
    
    #res = 1 - abs(ix - iy) / max(ix, iy)
    return fx, fy