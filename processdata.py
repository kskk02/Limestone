import pandas as pd 
import numpy as np 
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn import svm 
from sklearn import cross_validation
from sklearn import metrics 




if __name__ == "__main__":

	# load the data in 
	df = pd.read_csv("training.csv")
	df.index = df["EventId"]
	df = df.drop("EventId", axis=1)

	# create our Y
	Y = df["Label"]
	Y[Y == "b"] = -1
	Y[Y == "s"] = 1 

	# create and scale our X
	X = df.drop("Label", axis=1)
	X = scale(X)

	# Now try running two basic logistic regressions, one 
	# where we "binarize" the missing data away and one where we
	# simply leave it in

	log1 = LogisticRegression()
	log1.fit(X, Y)
	log1_score = np.mean(cross_validation.cross_val_score(log1, X, Y, cv=3))
	print "Leaving data as is: " + str(log1_score)

	# transforming data









