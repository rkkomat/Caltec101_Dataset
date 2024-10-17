#%%
import numpy as np
from sklearn import metrics
import cv2 
import pandas as pd
import matplotlib.pyplot as plt
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.preprocessing import StandardScaler

#%%
df_edge_hist = pd.read_csv('./EdgeHistogram.csv', delimiter=';', header = None, names=['ID']+[F'feature{i}' for i in range(1, 81)])
df_image = pd.read_csv('./Images.csv', delimiter=';', header=None, names = ['ID', 'Class'])
df_final = pd.merge(df_image, df_edge_hist, left_on='ID', right_on = 'ID')
#%%

X= df_final.drop(['ID', 'Class'], axis=1).copy()
y= df_final['Class'].copy()
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

#%%
# try K=1 through K=25 and record testing accuracy
k_range = range(2, 20)

# We can create Python dictionary using [] or dict()
scores = []
score_compare=0
k_max=2
# We use a loop through the range 1 to 26
# We append the scores in the dictionary
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    if(score_compare<metrics.accuracy_score(y_test, y_pred)):
        score_compare=metrics.accuracy_score(y_test, y_pred)
        k_max=k   
        #print(k_max)
    scores.append(metrics.accuracy_score(y_test, y_pred))


# plot the relationship between K and testing accuracy
# plt.plot(x_axis, y_axis)
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')

kNN_Classifier = KNeighborsClassifier(n_neighbors=k_max)
kNN_Classifier.fit(x_train,y_train)
kNN_pred = kNN_Classifier.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, kNN_pred))

#%%

#Model2 RandomForest
from sklearn.ensemble import RandomForestClassifier

random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

random_forest_classifier.fit(x_train, y_train)

RFC_pred = random_forest_classifier.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, RFC_pred))


# %%
 #Model 3 SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)

# Create an SVM classifier with a linear kernel
SVMclassifier = svm.SVC(kernel='linear', C=1.0)

SVMclassifier.fit(X_train_scaled, y_train)
SVM_pred = SVMclassifier.predict(X_test_scaled)

print("Accuracy:",metrics.accuracy_score(y_test, SVM_pred))

# %%
from sklearn.metrics import accuracy_score, confusion_matrix
def ConfusionMatrix(y_true, pred, filename):
    cm = confusion_matrix(y_true, pred)
    labels = np.unique(y_true)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    df_cm.to_csv(filename, sep=",", index_label="")

ConfusionMatrix(y_test, kNN_pred, "group004_result1.csv")
ConfusionMatrix(y_test, RFC_pred, "group004_result2.csv")
ConfusionMatrix(y_test, SVM_pred, "group004_result3.csv")


# %%
# Hyperparameters
def HyperParameters(classifier, params, filename):
    params_df = pd.DataFrame(params.items(), columns=["name", "value"])
    params_df.to_csv(filename, sep=",", index=False)

knn_hyperparams = {"n_neighbors": kNN_Classifier.n_neighbors, "weights": kNN_Classifier.weights}
rf_hyperparams = {"n_estimators": random_forest_classifier.n_estimators, "max_depth": random_forest_classifier.max_depth}
svm_hyperparams = {"C": SVMclassifier.C, "kernel": SVMclassifier.kernel}

HyperParameters(kNN_Classifier, knn_hyperparams, "group001_parameters4.csv")
HyperParameters(random_forest_classifier, rf_hyperparams, "group001_parameters3.csv")
HyperParameters(SVMclassifier, svm_hyperparams, "group001_parameters1.csv")
# %%
