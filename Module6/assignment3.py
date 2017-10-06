#
# This code is intentionally missing!
# Read the directions on the course lab page!
#

import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC

# load data file
X = pd.read_csv('Datasets\parkinsons.data')

X.drop('name',axis=1,inplace=True)
# Extract status and save it in y
y = X['status']

#Remove status from X
X.drop('status',axis=1,inplace=True)



print X.dtypes
print '############'
print y.dtypes

from sklearn import preprocessing

X_train , X_test , y_train , y_test = train_test_split (X,y,test_size=0.30 , random_state=7)

from sklearn.preprocessing import Normalizer, MaxAbsScaler, MinMaxScaler, KernelCenterer, StandardScaler
processor = StandardScaler()
#processor = KernelCenterer()
#processor = MinMaxScaler()
#processor = MaxAbsScaler()
#processor = Normalizer()
#T = preprocessing.KernelCenterer().fit(X_train)
#T = preprocessing.MinMaxScaler().fit(X_train)
#T = preprocessing.MaxAbsScaler().fit(X_train)
#T = preprocessing.Normalizer().fit(X_train)

processor.fit(X_train)
X_train = processor.transform(X_train)
X_test = processor.transform(X_test)
#svc_model = SVC(C=1.6,gamma=0.1)

#svc_model.fit(X_train,y_train)
#score = svc_model.score(X_test,y_test)


#==============================================================================
# from sklearn.decomposition import PCA
# pca = PCA(n_components = 14)
# pca.fit(X_train)
# X_train = pca.transform(X_train)
# X_test = pca.transform(X_test)
#==============================================================================

from sklearn.manifold import Isomap

iso = Isomap(n_neighbors=5, n_components=6)

iso.fit(X_train)
X_train = iso.transform(X_train)
X_test = iso.transform(X_test)


best_score = 0
for c in np.arange (start =0.05 ,  stop = 2.05 ,step =0.05):
    for gamma in np.arange (start =0.001 ,  stop = 0.101 ,step =0.001):
        model = SVC(C=c,gamma=gamma)
        model.fit(X_train,y_train)
        score =model.score(X_test,y_test)
        if best_score < score:
            best_score = score
            print "Best Score :", best_score
            print "C value :",c
            print "Gamma Value :",gamma
print "Best Score :", best_score
print "C value :",c
print "Gamma Value :",gamma
            

        





