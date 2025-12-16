""" our lime implementation """

import numpy as np
from sklearn.linear_model import Lasso


class LIME:

    def __init__(self, model, n , prob, rand):

        self.model = model
        self.n = n
        self.prob = prob
        self.rand = rand

    
    def perturbations(self, x, n):

        masks = np.random.binomial(1, self.prob, size=(self.n, n)).astype(np.float32)
        masks[0] = 1.0
        pert = x * masks
        return masks, pert
    
    def cosine_distance(self, x, xpert):
        return np.clip(1.0 - np.dot(xpert / (np.linalg.norm(xpert, axis=1, keepdims=True) + 0.0000000001), x / (np.linalg.norm(x) + 0.0000000001)), 0.0, 2.0)
    
    def sim_weights(self, dists, n):
        
        return np.exp(-(dists ** 2) / (0.75 * np.sqrt(n) ** 2))


    def explain(self, x, names, alpha):

        x = np.asarray(x).flatten()
        n = len(x)

        masks, pert = self.perturbations(x, n)
        proba = self.model.predict_proba(pert)
        if proba.shape[1] == 2:
            y = proba[:, 1]
        else:
            y = proba[:, 0]

        opred = self.model.predict_proba(x.reshape(1, -1))
        if opred.shape[1] == 2:
            oprob = opred[0, 1]
        else:
            oprob = opred[0, 0]

        w = self.sim_weights(self.cosine_distance(x, pert), n)
        lasso = Lasso(alpha= (( alpha/np.sqrt(n)) * max(np.std(y), 0.01)), fit_intercept=True, max_iter=3000, tol=0.00001, selection='random')
        lasso.fit(masks, y, sample_weight=w)
        coefs = lasso.coef_

        fcoefs = list(zip(names, coefs))
        fcoefs.sort(key=lambda x: x[1], reverse=True)
        
        pfeats = []
        for i, j in fcoefs:
            if j > 0:
                pfeats.append((i, j))
        
        nfeats = []
        for i, j in fcoefs:
            if j < 0:
                nfeats.append((i, j))

        return {'positive_features': pfeats,'negative_features': nfeats,'negative_abs': sorted(nfeats, key=lambda x: abs(x[1]), reverse=True),
                'intercept': lasso.intercept_,'prediction': oprob, 'features_used': np.sum(np.abs(coefs) > 0.000001)}   
    

