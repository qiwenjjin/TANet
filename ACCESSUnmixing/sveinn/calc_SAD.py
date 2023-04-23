'''
Calculate Spectral Angle Distance (SAD) between a set of spectral vectors,
input as N x r arrays where N is the number of channels and r is the 
number of vectors.

Calculates the SAD r times, each time starting with a new column.
The best reslult is returned.
    
Based on the work of Jakob Sigurðsson et al.

Inputs:
A = Array of size (N x r) containing column vectors,
B = Another array of the same size (N x r),

Outputs:
sad = The best mean SAD for all columns,
idx_org, idx_hat = the indexes of the columns that are paired together,
sad_k = per column SAD for the chosen pairings
s0 = array of all SADs calculated

Example run:

A = np.array([[0,3,6], [1,4,7], [2,5,8]])
B = np.array([[0,60,5], [1,70,4], [2,80,3]])

(sad,idx_org,idx_hat,sad_k,s0) = calc_SAD_2(A,B)

sad = 0.13423861386022068,
idx_org = array([0, 1, 2]),
idx_hat = array([0, 2, 1]),
sad_k = array([0.        , 0.40271584, 0.        ]),
s0 = array([[0.        , 0.48336128, 0.56860155],
            [0.56860155, 0.08524027, 0.        ],
            [0.88607712, 0.40271584, 0.31747558]]))

Python code by: Sveinn E. Ármannsson

5. júní 2019, Reykjavík
'''

import numpy as np


#%%
# Function that calculates the spectral angle distance between two
# column vectors, ai and aj
def cs(ai,aj):

    if np.sum(np.abs(aj-ai)) == 0:
        s = 0
        return s
    s = np.divide(np.transpose(aj)@ai,np.linalg.norm(ai)*np.linalg.norm(aj))
    s = np.arccos(s)
    # if 0 < np.abs(s.imag) < 1e-7:
    #     s = s.real
    return s

#%%
# Function that calculates the mean spectral angle distance between 
# the columns of Aorg and Ahat.
# idxOrg and idxHat are the indexes of the columns that are paired
# together.
#                                                                   
def calc_SAD(Aorg,Ahat,start_col=-1):

    r1 = Aorg.shape[1]
    r2 = Ahat.shape[1]
    s = np.zeros((r2,r1))
    sad = np.zeros(r1)
    idx_hat = np.zeros(r1,dtype=int)
    idx_org = np.zeros(r1,dtype=int)
    
    for i in range(r1):
        ao = Aorg[:,i]
        for j in range(r2):
            ah = Ahat[:,j]
            s[j][i] = np.minimum(cs(ao,ah),cs(ao,-1*ah))
    s0 = np.copy(s)

    for p in range(r1):
        if start_col > -1 and p == 0:
            sad[p], b = np.min(s[:,start_col]), np.argmin(s[:,start_col])
            idx_hat[p] = b
            idx_org[p] = start_col
        else:
            sad[p], b = np.min(s), np.argmin(s)
            (idx_hat[p],idx_org[p]) = np.unravel_index(b,s.shape)
        
        s[:, idx_org[p]] = np.inf
        s[idx_hat[p]][:] = np.inf
        
        if np.isinf(sad[p]):
            idx_hat[p] = np.inf
            idx_org[p] = np.inf
    
    sad_k = sad
    
    idx = idx_org.argsort(kind='stable')
    idx_org = idx_org[idx]
    idx_hat = idx_hat[idx]
    sad_k = sad_k[idx]
    
    sad = np.mean(sad)
    
    return sad, idx_org, idx_hat, sad_k, s0

#%%
# Function that finds the minimum SAD available through different 
# pairings of vectors in the given arrays.
# Calculates the SAD r times, each time starting with a new column.
# The best reslult is returned.
#
def calc_SAD_2(Aorg,Ahat):
    
    
    r = Aorg.shape[1]
    sad = np.zeros(r)
    idx_hat = np.zeros((r,r),dtype=int)
    idx_org = np.zeros((r,r),dtype=int)
    sad_k = np.zeros((r,r))

    for i in range(r):
        sad[i], idx_org[i,:], idx_hat[i,:], sad_k[i,:], s0 = calc_SAD(Aorg, Ahat, i)
    
    sad_m, idx = min(sad), np.argmin(sad)
    
    idx_org_m = idx_org[idx] 
    idx_hat_m = idx_hat[idx] 
    sad_k_m = sad_k[idx]

    return sad_m, idx_org_m, idx_hat_m, sad_k_m, s0
