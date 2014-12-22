
#Learning Pandas and skikit-learn: Titanic Kaggle competition
import os
import csv as csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

train = pd.read_csv("C:\Users\Greg\Desktop\Kaggle\\titanic\\titanic_train.csv")

test = pd.read_csv("C:\Users\Greg\Desktop\Kaggle\\titanic\\titanic_test.csv")

train2 = train[[2,4,5,6,7,9]]
labels = train[[1]]

gender1 = train2["Sex"]
gender1 = [1 if x == "male" else 2 for x in gender1]
train2["Sex"] = gender1
train2 =train2.fillna(0)


test2 = test[[1,3,4,5,6,8]]
gender2 = test["Sex"]
gender2 = [1 if x == "male" else 2 for x in gender2]
test2["Sex"] = gender2
test2 = test2.fillna(0)

rf = RandomForestClassifier(n_estimators=2000,max_depth=6)

fit = rf.fit(train2, np.ravel(labels))

rf_pred = rf.predict(test2)

output = pd.DataFrame( data={"PassengerId":test["PassengerId"], "Survived":rf_pred} )

output.to_csv( "scikitRF.csv", index=False)


#Try support vector machines
from sklearn.svm import SVC

normalized_train = (train2 - train2.mean()) / (train2.max() - train2.min())

svm_classifier = SVC(C=4, kernel="rbf", gamma=5, random_state=12)

#Validation. Tuning paramters
print "10 Fold CV Score: ", np.mean(cross_validation.cross_val_score(svm_classifier, \
                        normalized_train, np.ravel(labels), cv=10, scoring='accuracy'))

svm_fit = svm_classifier.fit(normalized_train, np.ravel(labels))

normalized_test= (test2 - test2.mean()) / (test2.max() - test2.min())

svm_pred = svm_fit.predict(normalized_test)

output = pd.DataFrame( data={"PassengerId":test["PassengerId"], "Survived":svm_pred} )

output.to_csv( "scikitSVM.csv", index=False)


#Try a polynomial kernel
svm_classifier = SVC(C=50, kernel="poly", degree=2, gamma=4, random_state=12)

#Validation. Tuning paramters
print "10 Fold CV Score: ", np.mean(cross_validation.cross_val_score(svm_classifier, \
                        normalized_train, np.ravel(labels), cv=10, scoring='accuracy'))

svm_fit = svm_classifier.fit(normalized_train, np.ravel(labels))

normalized_test= (test2 - test2.mean()) / (test2.max() - test2.min())

svm_pred_poly = svm_fit.predict(normalized_test)

output = pd.DataFrame( data={"PassengerId":test["PassengerId"], "Survived":svm_pred_poly} )

output.to_csv("scikitSVM2.csv", index=False)