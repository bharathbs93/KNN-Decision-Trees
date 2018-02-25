
# Using the top 100 features
import pandas as pd
from sklearn.datasets import load_files
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as score


# This is the x for the KNN,since there is cross validation I will not be doing test-train split
# Applied Tf-idf here, not normalization
X = pd.read_csv("finamatrix.csv",index_col=0, header =0 )

# Loading the raw input
load_input = load_files('20news-18828',decode_error='ignore')

# target is the label with a range from 1-20, associated with each article category

y = load_input['target']

# Initializing classifier
knn = KNeighborsClassifier(n_jobs=-1)

# 5 fold cross validation
accuracy_scores = cross_val_score(knn,X,y,cv=5, scoring='accuracy')

print "the average accuracy score percentage of the KNN after cross validation and feature selection is %f " % (accuracy_scores.mean() * 100)

# 5 fold cross validation

f1_score = cross_val_score(knn, X, y, cv=5,scoring='f1_weighted')

print "the average f1_score of the KNN after cross validation and feature selection is %f" % (f1_score.mean() * 100)

# Initializing decision tree
dt = DecisionTreeClassifier()


# computing accuracy
accuracy_scores = cross_val_score(dt,X,y,cv=5, scoring='accuracy')

print "the average accuracy score percentage of the Decision Tree after cross validation and feature selection is %f " % (accuracy_scores.mean() * 100)


# computing f-measure
f1_score = cross_val_score(dt, X, y, cv=5,scoring='f1_weighted')

print "the average f1_score of the Decision Tree after cross validation and feature selection is %f" % (f1_score.mean() * 100)


# Trying to find the best K(since 100 features trying 1- 100)
# creating odd list of K for KNN
myList = list(range(1,101))

# sub setting just the even ones
neighbors = filter(lambda x: x % 2 == 0, myList)

# empty list that will hold cv scores
cv_scores = []

# perform 5-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k,n_jobs=-1)
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())


# finding the mis-classification error
MSE = [1 - x for x in cv_scores]

# determining the best k
optimal_k = neighbors[MSE.index(min(MSE))]
print "The optimal number of neighbors is %d" % optimal_k

# plot mis-classification error vs k
plt.plot(neighbors, MSE)
plt.title('K-accuracy-score vs Mis-Classification Error')
plt.xlabel('Number of Neighbors K')
plt.ylabel('Mis-classification Error')
plt.show()

# Initializing with the best K
knn = KNeighborsClassifier(n_neighbors=78,n_jobs=-1)


# Calculations using the best K
accuracy_scores = cross_val_score(knn,X,y,cv=5, scoring='accuracy')

print "the average accuracy score percentage of the best K-KNN after cross validation is %f " % (accuracy_scores.mean() * 100)


# finding the best K for the f-measure score

# creating odd list of K for KNN
myList = list(range(1,101))

# sub-setting just the even ones
neighbors = filter(lambda x: x % 2 == 0, myList)

# empty list that will hold f1 scores
f1_scores = []

# perform 5-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    scores = cross_val_score(knn,X, y,scoring='f1_weighted')
    f1_scores.append(scores.mean())

# calculating mis-classification error
MSE = [1 - x for x in f1_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print "The optimal number of neighbors is %d" % optimal_k

# plot mis-classification error vs k
plt.plot(neighbors, MSE)
plt.title('K-f-measure vs Mis-Classification Error')
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()

# F-measure with best K
knn = KNeighborsClassifier(n_neighbors=32,n_jobs=-1)

f1_score = cross_val_score(knn, X, y, cv=5,scoring='f1_weighted')
print "the average f1_score of the best K-KNN after cross validation is %f" % (f1_score.mean() * 100)

# Converting the accuracy and f-measure score into numpy array's for the plot
cv_scores = np.asarray(cv_scores)
f1_scores = np.asarray(f1_scores)
# Plotting Scores vs K
plt.plot(neighbors,cv_scores, label = 'Accuracy_Score')


plt.plot(neighbors,f1_scores, label = 'f-measure')


plt.xlabel('K')
plt.ylabel('Score')

plt.title("Scores vs K")

plt.legend()

plt.show()

# Plotting best-k vs Decision Tree accuracies
objects = ('Best-K-Accuracy', 'Best-K-f-measure', 'DT-Accuracy-Score', 'DT-f-measure')

y_pos = np.arange(len(objects))
scores = [0.05120027924347279,0.04606845137659447,0.0497127097191066,0.04995423544075248]
plt.bar(y_pos, scores,align='center', alpha=1)
plt.xticks(y_pos, objects)
plt.ylabel('Scores')
plt.title('Best K vs Decision Tree Scores')

plt.show()


# Calculating Precision for the best K
knn = KNeighborsClassifier(n_neighbors=78,n_jobs=-1)

predicted = cross_val_predict(knn,X,y,cv=5)
# precision, recall, f-score and support
precision, recall, fscore, support = score(y, predicted)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

# Converting to percentages
precision = precision * 100

# Decision Tree
dt = DecisionTreeClassifier()
predicted = cross_val_predict(dt,X,y,cv=5)


dt_precision, recall, fscore, support = score(y, predicted)

dt_precision = dt_precision * 100

# Category ID for each article category
categories = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
categories = np.asarray(categories)

# Plotting Precision for Best-KNN vs DT
fig, ax = plt.subplots()
n_groups = 20
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, precision, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Best-K-Accuracy')
 
rects2 = plt.bar(index + bar_width, dt_precision, bar_width,
                 alpha=opacity,
                 color='g',
                 label='DT-Accuracy')
 
plt.xlabel('Category-ID')
plt.ylabel('Accuracy(%)')
plt.title('KNN vs DT accuracy for each category')
plt.xticks(index + bar_width, ('1', '2', '3', '4','5','6', '7', '8', '9','10','11', '12', '13', '14','15','16', '17', '18', '19','20'))
plt.legend()
 
plt.tight_layout()
plt.show()

