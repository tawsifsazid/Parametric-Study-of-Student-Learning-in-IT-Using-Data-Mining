# -*- coding: utf-8 -*-
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

import matplotlib.pyplot as plt
# machine learning
#from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

from sklearn.svm import SVC, LinearSVC
from sklearn import grid_search
from sklearn.grid_search import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split

#import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

from pandas import ExcelWriter
from pandas import ExcelFile
from sklearn import preprocessing

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold

df = pd.read_excel('C:\\Users\\Tawsif Sazid\\Desktop\\A.xls')


#print(df.head(5))




df["A2"][df["A2"] == "E"] = 3
df["A2"][df["A2"] == "E "] = 3
#df["A2"][df["A2"] == "AA"] = 2
df["A2"][df["A2"] == "G"] = 2
df["A2"][df["A2"] == "G "] = 2
df["A2"][df["A2"] == " G"] = 2
df["A2"][df["A2"] == "A"] = 1
df["A2"][df["A2"] == "A "] = 2
df["A2"][df["A2"] == "nil"] = 2

target = df["A2"].values

# Error checking

    
df["A3"][df["A3"] == "M"] = 1    
df["A3"][df["A3"] == "F"] = 0
df["A3"][df["A3"] == "nil"] = 1   

df["E4"][df["E4"] == "c"] = 1    
df["E4"][df["E4"] == "d"] = 2
df["E4"][df["E4"] == "b"] = 3   
df["E4"][df["E4"] == "nil"] = 3
df["E4"][df["E4"] == "a"] = 4
df["E4"][df["E4"] == "e"] = 5

df["D12"][df["D12"] == "a,b"] = 5
df["D12"][df["D12"] == "a"] = 1
df["D12"][df["D12"] == "b"] = 2
df["D12"][df["D12"] == "c"] = 3
df["D12"][df["D12"] == "d"] = 4

df["E1"][df["E1"] == "a"] = 1
df["E1"][df["E1"] == "b"] = 2
#df["D12"][df["D12"] == "b"] = 2
df["E1"][df["E1"] == "c"] = 3
df["E1"][df["E1"] == "d"] = 4
df["E1"][df["E1"] == "e"] = 5
df["E1"][df["E1"] == "nil"] = 3

df["E2"][df["E2"] == "a"] = 1
df["E2"][df["E2"] == "b"] = 2
#df["D12"][df["D12"] == "b"] = 2
df["E2"][df["E2"] == "c"] = 3
df["E2"][df["E2"] == "d"] = 4
df["E2"][df["E2"] == "e"] = 5
df["E2"][df["E2"] == "nil"] = 3

df["E3"][df["E3"] == "a"] = 1
df["E3"][df["E3"] == "b"] = 2
#df["D12"][df["D12"] == "b"] = 2
df["E3"][df["E3"] == "c"] = 3
df["E3"][df["E3"] == "d"] = 4
df["E3"][df["E3"] == "e"] = 5
df["E3"][df["E3"] == "nil"] = 3

df["E5"][df["E5"] == "a"] = 1
df["E5"][df["E5"] == "b"] = 2
#df["D12"][df["D12"] == "b"] = 2
df["E5"][df["E5"] == "c"] = 3
df["E5"][df["E5"] == "d"] = 4
df["E5"][df["E5"] == "e"] = 5
df["E5"][df["E5"] == "nil"] = 3
'''
with open('B.xls', 'rb') as csvfile_temp1:
    reader1 = excell.reader(csvfile_temp1)
    for row in reader1: 
        x = float(row[W_temp])
'''

'''
with pd.option_context('display.max_rows', None, 'display.max_columns', 53):
    print(df["E5"])
'''

#print(df["A2"].value_counts())
#print(df.groupby('A2').mean())
########################################################################################## Data VISUALIZATION #####################################################################
'''
excellent = df["A2"][df["A2"] == 3]
good = df["A2"][df["A2"] == 2]
avg = df["A2"][df["A2"] == 1]
plt.scatter(excellent)
plt.scatter( good)
plt.scatter( avg)
#plt.scatter(grades_range, boys_grades, color='g')
plt.xlabel('Grades Range')
plt.ylabel('Grades Scored')
plt.show()
'''

'''
table=pd.crosstab(df.C1,df.A2)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Nothing')
plt.xlabel('feature')
plt.ylabel('Excellent = Red, Good = Green, Average = Blue')
plt.savefig('edu_vs_pur_stack')
plt.show()


pd.crosstab(df["C1"],df["A2"]).plot(kind='bar')
plt.title('Bar Plot')
plt.xlabel('feautre')
plt.ylabel('Excellent = Red, Good = Green, Average = Blue')
plt.savefig('purchase_fre_job')
plt.show()
'''
first = df[["E1","E2","E3","E5","C14","A3","B1","B2","B3","B4","B5","B6","B7","C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","C11","C12","C16a","C16b","C16c","C16d","C16e","C16f","C16g","C16h","C17"
,"D1","D2","D3","D4","D5","D6","D7","D8","D9","D10","D11","E4"]].values
y = df["A2"].astype(int)  ############ int a convert korte hobe #################################################################
####################################################################################################################################################################################

print(df["A2"].value_counts())

##################################################################################   Feature Selection  #############################################################################
'''

"E1","E2","E3","E5","C14","A3","B1","B2","B3","B4","B5","B6","B7","C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","C11","C12","C16a","C16b","C16c","C16d","C16e","C16f","C16g","C16h","C17"
,"D1","D2","D3","D4","D5","D6","D7","D8","D9","D10","D11","E4"

"A3","C2","C7","C11","C16c","C16h","D1","D4","D7","D9"
'''
'''
import numpy as np
from sklearn.decomposition import PCA

pca = PCA(n_components=30)
pca.fit(first)
yoo1 = pca.transform(first)
print(yoo1.shape)
print(pca.explained_variance_ratio_)  
'''


#selector = SelectKBest(f_classif, k=18)
#print(selector.get_support)
S = SVC(kernel='linear')

# create the RFE model for the svm classifier 
# and select attributes
rfe = RFE(S,30)
rfe = rfe.fit(first, y)
# print summaries for the selection of attributes
print(rfe.support_)
print(rfe.ranking_)
print(rfe.n_features_)
yoo = rfe.transform(first)
#yoo = ["A3","B3","B4","B6","C2","C6","C7","C8","C11","C16c","C17","D1","D2","D3","D5","D6","D7","D8","D9","D10","D11","E4","E3","E5"]

#print(rfe.score(first,y))


'''
rfecv = RFECV(estimator=S, step=1, cv=StratifiedKFold(2))
rfecv.fit(yoo, y)
print(rfecv.grid_scores_) 
'''
'''

clf = DecisionTreeClassifier()
clf.fit(first, y)
sfm = SelectFromModel(clf,threshold=0.022)
sfm.fit(first, y)

X_important_train = sfm.transform(first)
print(X_important_train.shape)

p = pd.DataFrame(X_important_train)
p.to_pickle('C:\\Users\\Tawsif Sazid\\Desktop\\lala.xls') 
print(list(p.columns.values))
'''
'''
clf = KNeighborsClassifier()
clf.fit(first,y)
sfm = SelectFromModel(clf,threshold=0.022)
sfm.fit(first,y)
'''
'''
clf = RandomForestClassifier()
clf.fit(first, y)
sfm = SelectFromModel(clf)
sfm.fit(first, y)

X_important_train = sfm.transform(first)
print(X_important_train.shape)
#print(pca.singular_values_)  
'''


L = LogisticRegression()

# create the RFE model for the svm classifier 
# and select attributes
rfe = RFE(L,24)
rfe = rfe.fit(first, y)
# print summaries for the selection of attributes
print(rfe.support_)
print(rfe.ranking_)
print(rfe.n_features_)
yoo1 = rfe.transform(first)

'''
R = RandomForestClassifier()

# create the RFE model for the svm classifier 
# and select attributes
rfe = RFE(R,20)
rfe = rfe.fit(first, y)
# print summaries for the selection of attributes
print(rfe.support_)
print(rfe.ranking_)
print(rfe.n_features_)
yoo2 = rfe.transform(first)
'''
####################################################################################################################################################################################


#df =  df[["A3","C1","C2","C3","C4","C7","C9","B5","B3","B4","B6","B7","C16a","C16b","C16c","C16h","D1","D4","D8","D11","E4","B1","B2"]].astype(int)





############################################################## int a convert na korle ERRRRRRRRROR DEY!!!!!!!!!!!!!!!!!! ###################################
# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(first, y, test_size=0.4)
#print(X_train.shape, y_train.shape)
#print(X_test.shape, y_test.shape)
#print(rfe.score(X_test,y_test))
###########################################################      Logistic Regression  with grid search  ##############################################################################


#p = pd.read_pickle('C:\\Users\\Tawsif Sazid\\Desktop\\lala.xls')
#print(first.dtype)
Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 2000, .0001, 1E6, .00000001]
#solver = ['newton-cg','sag']
##gammas = [0.001, 0.01, 0.1, 1, 10 , 100, 200, 1000, 2000, .0001]   gamma lagbe naa
max_iter = [100,1000,10000]
param_grid = {"C": Cs, 'max_iter': max_iter}

mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
grid_search = GridSearchCV(estimator=mul_lr,param_grid=param_grid,cv=6,refit=True)
grid_search.fit(yoo1,y)
print(grid_search.best_score_)


####import statsmodels.api as sm
'''
mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(X_train, y_train)
print("Logistic Regression")
print(mul_lr.predict(X_test))
print(mul_lr.score(X_test,y_test))
'''
######################################################### Perform 6-fold cross validation #########################################################################
'''
# Necessary imports: 
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
scores = cross_val_score(mul_lr, first, y, cv=10)
print("Cross Validated score: ", scores)
'''

##########################################################        Random Forest with grid_search    #################################################################################
'''
#p = pd.read_pickle('C:\\Users\\Tawsif Sazid\\Desktop\\lala.xls')
param_grid = { 
    'n_estimators': [10,20,50],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [5,10,20],
    'min_samples_split': [2,4,6] 
    #'warm_start': [True, False],
    #'class_weight': ['balanced']
}
#forest = RandomForestClassifier(max_depth = 10,min_samples_split=2, n_estimators = 500,warm_start = True)
forest = RandomForestClassifier()
grid_search = GridSearchCV(estimator=forest,param_grid=param_grid,cv=10,refit=True)
grid_search.fit(yoo2,y)
print(grid_search.best_score_)
'''

'''
forest = RandomForestClassifier(max_depth = 100,min_samples_split=2, n_estimators = 500,warm_start = True)
my_forest = forest.fit(X_train,y_train)
print("Random Forest")
print(forest.predict(X_test))
print(forest.score(X_test,y_test))

from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
scores = cross_val_score(my_forest, first, y, cv=10)
print("Cross Validated score: ", scores)

'''


'''
predictions = cross_val_predict(my_forest, first, y, cv=6)

print(predictions)
accuracy = metrics.r2_score(y, predictions)
print("Cross-Predicted Accuracy : ", accuracy)

'''

##########################################################  Decision Tree with grid_search #######################################################################################
'''
param_grid = { 
    'max_features': [1,10,'auto', 'sqrt', 'log2'],
    'max_depth': [10, 20, 50, 100, 1000],
    'min_samples_split': [2,3,5,7,100], 
    #'warm_start': [True, False],
    #'class_weight': ['balanced']
}
p = pd.read_pickle('C:\\Users\\Tawsif Sazid\\Desktop\\lala.xls')

my_tree = tree.DecisionTreeClassifier()
grid_search = GridSearchCV(estimator=my_tree,param_grid=param_grid,cv=12,refit='false')
grid_search.fit(p,y)
print(grid_search.best_score_)
'''

'''
from sklearn.metrics import accuracy_score
clf_gini = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
            max_features=30, max_leaf_nodes=None, min_samples_leaf=5,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=100, splitter='best')

#clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)


y_pred = clf_gini.predict(X_test)
print("Accuracy is ", accuracy_score(y_test,y_pred)*100)
'''
'''
my_tree = tree.DecisionTreeClassifier(max_depth = 500, min_samples_split = 2, random_state = 3)
#my_tree = tree.DecisionTreeClassifier()
my_tree = my_tree.fit(X_train, y_train)
print(my_tree.score(X_test,y_test))

from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
scores = cross_val_score(my_tree, first, y, cv=12)
print("Cross Validated score: ", scores)
'''
######################################################## SVM ######################################################################################################
#################################################### SVM with grid_search #########################################################################################

Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 2000, .0001]
gammas = [0.001, 0.01, 0.1, 1, 10 , 100, 200, 1000, 2000, .0001]
kernel = ['rbf','polynomial','linear','sigmoid']
param_grid = {"C": Cs,"gamma" : gammas}
S = SVC()
grid_search = GridSearchCV(estimator=S,param_grid=param_grid,cv=12,refit=True)
grid_search.fit(yoo,y)
print(grid_search.best_score_)

######################################################## Below cheking how accurate ##############################################################################
#print(y_test)
#print(grid_search.predict(X_test))
#print(grid_search.cv_results_)
#print(grid_search.grid_scores_)
#print(grid_search.cv_results_)
 
                
'''
S = SVC(kernel='rbf')
S.fit(X_train, y_train)
print(S.score(X_test,y_test))

from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
scores = cross_val_score(grid_search, first, y, cv=10)
print("Cross Validated score: ", scores)
'''
#########################################################  Cross Validation  #########################################################################################
'''
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=8) 
mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(X_train, y_train) ## 54
forest = RandomForestClassifier(max_depth = 100,min_samples_split=2, n_estimators = 500,warm_start = True) 
S = SVC(kernel='rbf', C=1E6,shrinking = True)
my_tree = tree.DecisionTreeClassifier(max_depth = 500, min_samples_split = 2, random_state = 3)
scoring = 'accuracy'
results = model_selection.cross_val_score(mul_lr, first, y, cv=kfold, scoring=scoring)
print("n-fold cross validation avg accuracy: %.3f" % (results.mean()))
'''
#################################################################################################################################################################
#plt.scatter(df["C8"],y)
#plt.show()

####################################################################### Neural Network #####################################################################################
'''
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

X = first
y = df["A2"].astype(int)
'''
###############################################################################################################################################

################################################################################# K-NN ###########################################################################

'''
algorithm  = ['ball_tree','kd_tree','brute','auto']
weights  = ['uniform','distance']
leaf_size  = [30,10,100,200,5,1000]


param_grid = {'algorithm' : algorithm, 'weights': weights, 'leaf_size' : leaf_size}
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(p,y)
#my_tree = tree.DecisionTreeClassifier()
grid_search = GridSearchCV(estimator=knn,param_grid=param_grid,cv=12,refit='false')
grid_search.fit(p,y)
print(grid_search.best_score_)
'''
##################################################################################################################################################################

################################# New Try ########################################

