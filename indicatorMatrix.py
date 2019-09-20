import numpy as np 

def indicatorMatrix(what,c):
    '''
    Makes a indicator Matrix for the categories in x
    INPUT:
        what   : gives the type of indicator matrix that is needed
          'identity'       : a regressor for each category 
          'identity_p'     : a regressor for each category, except for c==0 
          'reduced'        : GLM-reduced coding (last category is -1 on all indicators)
          'pairs'          : Codes K category as K-1 pairs
          'allpairs'       : Codes all K*(K-1) pairs in the same sequence as pdist
          'allpairs_p'     : Same a allpairs, but ignoring 0 
          'interaction_reduced' 
          'hierarchical'   : Codes for a fixed variable underneath a normal factor 
          'hierarchicalI'  : Codes for a random variable unerneath a normal factor
        c      : np-vector of categories
    OUTPUT:
        design Matrix
    Joern Diedrichsen 2019
    '''
    cc = np.unique(c)
    K = cc.size
    rows = np.size(c)
    
    if (what=="identity"):
        X=np.zeros((rows,K))
        for i in range(K):
            X[c==cc[i],i]=1
    elif (what=="identity_p"):
        X=np.zeros((rows,K))    
    else: 
        print("invalid option")
    return X

