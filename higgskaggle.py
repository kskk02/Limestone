from pprint import pprint
from time import time
import logging

from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn import cross_validation
from sklearn.preprocessing import scale
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import json

print(__doc__)


# if __name__ == "__main__":

# load the data in 
df = pd.read_csv("training.csv")
df.index = df["EventId"]
df = df.drop("EventId", axis=1)
df = df.drop("Weight", axis=1)

# create our Y
Y = df["Label"]
Y[Y == "b"] = -1
Y[Y == "s"] = 1 
Y = Y.ravel() 
Y = Y.astype(int) 

Ylog = Y.copy()
Ylog[Ylog == -1] = False 
Ylog[Ylog == 1] = True 
Ylog = Ylog.astype(int) 

# create and scale our X
X = df.drop("Label", axis=1)

# transforming data

columns_with_bad_data = []
good_columns = []

X2 = X.copy()
X3 = X.copy() 
# print X2.head() 
    
for col in X.columns: 
    percent_bad = np.mean(X2[col] == -999.00)
    if percent_bad > .01:
        columns_with_bad_data.append(col)
        X2[col] = (X2[col] != -999.00) 
    else:
        good_columns.append(col) 

# print X2.head() 

# sanity check 
print columns_with_bad_data

X = scale(X)
X2[good_columns] = scale(X2[good_columns])


# now we're going to bin the problematic columns
for col in columns_with_bad_data:

    col_max = np.max(X3[col])
    col_min = np.min(X3[col][X3[col] > -999.00])

    if col_min < 0:
        X3[col][X3[col] == -999.00] = 1.1 * col_min
    else:
        X3[col][X3[col] == -999.00] = -1.1 * col_min 

    X3[col] = X2[col]*pd.cut(X3[col], 10, labels=(0., 1., 2., 3., 4., 5., 6., 7., 8., 9.)) 


X3[good_columns] = scale(X3[good_columns])
print X3.head() 



    # Now try running two basic logistic regressions, one 
    # where we "binarize" the missi+-ng data away and one where we
    # simply leave it in

    # print "---------------"
    # print "LOGISTIC REGRESSION: " 

    # log1 = LogisticRegression()
    # log1.fit(X, Ylog)
    # log1_score = np.mean(cross_validation.cross_val_score(log1, X, Ylog, cv=3, scoring="accuracy"))
    # print "Leaving data as is: " + str(log1_score)

    # log2 = LogisticRegression()
    # log2.fit(X2, Ylog)
    # log2_score = np.mean(cross_validation.cross_val_score(log2, X2, Ylog, cv=3, scoring="accuracy"))
    # print "Transforming: " + str(log2_score)

    # log3 = LogisticRegression() 
    # log3.fit(X2[good_columns], Ylog) 
    # log3_score = np.mean(cross_validation.cross_val_score(log3, X2[good_columns], Ylog, cv=3, scoring="accuracy"))
    # print "Dropping columns with bad data: " + str(log3_score)

    # log4 = LogisticRegression()
    # log4.fit(X3, Ylog)
    # log4_score = np.mean(cross_validation.cross_val_score(log4, X3, Ylog, cv=3, scoring="accuracy"))
    # print "Binning columns: " + str(log4_score) 


    # print "---------------"
    # print "SVM: "

    # svm1 = svm.SVC(class_weight="auto")
    # svm1.fit(X, Y)
    # svm1_score = np.mean(cross_validation.cross_val_score(svm1, X, Y, cv=3, scoring="accuracy")) 
    # print "Leaving data as is: " + str(svm1_score)

    # svm2 = svm.SVC(class_weight="auto")
    # svm2.fit(X2, Y)
    # svm2_score = np.mean(cross_validation.cross_val_score(svm2, X2, Y, cv=3, scoring="accuracy"))
    # print "Transforming: " + str(svm2_score)

# X3.to_csv("X_train.csv")
# pd.DataFrame(Ylog).to_csv("Ylog.csv")
# pd.DataFrame(Ylog).to_csv("Y.csv")


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# data = pd.read_table('training.csv')
# label = data['Label']
# data.pop('Label')
# print "adfs"
# #extract, label = extract_text(data)
# ###############################################################################
# define a pipeline combining a text feature extractor with a simple
# classifier
# npdata = np.array(data)
pipeline = Pipeline([
    ('clf', LogisticRegression())
])

pipeline = Pipeline([
    ('clf', SVC())
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
# parameters = {
#     'clf__class_weight': (None,'auto'),
#     'clf__penalty': ('l1','l2'),
#     'clf__C':(1.0,2.0)
#     #'clf__n_iter': (10, 50, 80),
# }

parameters = {
    'clf__class_weight': (None,'auto'),
    'clf__kernel': ('rbf', 'poly'),
    'clf__degree': [5],
    'clf__C':[0.5,5.0,50.0]
    #'clf__n_iter': (10, 50, 80),
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # # classifier
    # xtrain, xtest, ytrain, ytest = cross_validation.train_test_split(X3, Ylog, test_size=0.10, random_state=42)

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring="accuracy")

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(X3, Ylog)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))