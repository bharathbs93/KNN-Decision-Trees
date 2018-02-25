# Importing necessary Libraries
import pandas as pd
from sklearn.datasets import load_files
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Reading the csv file of term document frequency generated in R(TF-IDF used here)
X = pd.read_csv("full_matrix.csv",index_col=0, header =0 )

# Loading the raw input

load_input = load_files('20news-18828',decode_error='ignore')

# target is the label with a range from 1-20, associated with each article category

y = load_input['target']

# Initializing the classifier( here n = 3 by default)

knn = KNeighborsClassifier(n_jobs=-1)

# using 5 fold cross validation here
accuracy_scores = cross_val_score(knn,X,y,cv=5, scoring='accuracy')

print "the average accuracy score percentage of the KNN after cross validation is %f " % (accuracy_scores.mean() * 100)

# using 5 fold cross validation here once again
f1_score = cross_val_score(knn, X, y, cv=5,scoring='f1_weighted')


print "the average f1_score of the KNN after cross validation is %f" % (f1_score.mean() * 100)

# Initializing the Decision Tree Classifier with all default arguements
dt = DecisionTreeClassifier()

# 5 fold cross validation
accuracy_scores = cross_val_score(dt,X,y,cv=5, scoring='accuracy')

print "the average accuracy score percentage of the Decision Tree after cross validation is %f " % (accuracy_scores.mean() * 100)

# 5 fold cross validation
f1_score = cross_val_score(dt, X, y, cv=5,scoring='f1_weighted')

print "the average f1_score of the Decision Tree after cross validation is %f" % (f1_score.mean() * 100)

