import numpy as np 

def indicatorMatrix(what,c):
    '''
    Makes a indicator Matrix for the categories in x
    INPUT:
        what   : gives the type of indicator matrix that is needed
          'identity'       : a regressor for each category 
          'identity_p'     : a regressor for each category, except for c==0 
          'allpairs'       : Codes all K*(K-1)/2 pairs in the same sequence as pdist
          'allpairs_p'     : Same a allpairs, but ignoring 0 
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
        cc = cc[cc>0]
        K = cc.size
        X=np.zeros((rows,K))   
        for i in range(K):
            X[c==cc[i],i]=1
    elif (what=="allpairs"):
        X=np.zeros((rows,int(K*(K-1)/2)));
        k=0
        for i in range(K):
            for j in np.arange(i+1,K):
                X[c==cc[i],k] = 1./sum(c==i)
                X[c==cc[j],k] = -1./sum(c==j)
                k         = k+1
    else: 
        print("invalid option")
    return X

