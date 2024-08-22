from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

"""
Apply three classification algorithms to the same ckd_data.zip dataset as in Problem 2 (40 points)
a. Support Vector Machine with the linear kernel and default parameters (sklearn.svm.SVC).
b. Support Vector Machine with the RBF kernel and default parameters.
c. Random forest with default parameters (sklearn.ensemble.RandomForestClassifier).

Assess all three classification algorithms using the following protocol:
i. Use 80% of each class data to train your classifier and the remaining 20% to test it.

ii. Report the f-measure of the algorithm’s performance on the training and test sets
f-measure = (2*Pre *rec)/*pre+rec)
pre=(tp)/(tp+fp)
rec=(tp)/(tp_fn)

TP is the number of true positives (class 1 members predicted as class 1),
TN is the number of true negatives (class 2 members predicted as class 2),
FP is the number of false positives (class 2 members predicted as class 1),
and FN is the number of false negatives (class 1 members predicted as class 2).

"""

#data cleaning is based on these pages:
# https://stackoverflow.com/questions/62653514/open-an-arff-file-with-scipy-io
# https://stackoverflow.com/questions/40389764/how-to-translate-bytes-objects-into-literal-strings-in-pandas-dataframe-pytho
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

#read in and clean the data
data = arff.loadarff('chronic_kidney_disease_full.arff') #read in the arff file
data = pd.DataFrame(data[0]) #make it into a pandas dataframe


data.replace('?', np.nan, inplace=True) #replace '?' to nan (which we can then change to be a numerical value)

#function to replace missing values in the dataset (ie values that were '?' and are now nan
def impute_mode(feature):
    mode = data[feature].mode()[0] #find the mode of the column
    data[feature] = data[feature].fillna(mode) #replace the nan in the column with the mode

obj_cols = [col for col in data.columns if data[col].dtype == 'object'] #identify which columns are objects and therefore do not contain numerical values
numerical_cols = [col for col in data.columns if data[col].dtype != 'object'] #identify which columns are not objects and therefore contain numerical values

for col in numerical_cols: #replace nan in numerical columns with the mode of that column
    impute_mode(col)

#decode object columns to change them from bytes to strings (which we can then change into numerical values)
for col in obj_cols:
    data[col] = data[col].str.decode('utf-8').fillna(data[col])   # decode to string or return nan if missing value

for col in obj_cols: #replace the categorical info with numerical values
    data[col] = LabelEncoder().fit_transform(data[col]) #use the LabelEncoder to encode the object data into numerical data

#get the data, the cdk class is the dependent variable we will be predicting
features = []
for col in data.columns:
    if col != 'class':
        features.append(col)

x = data[features]
y = data['class'].values

#split the data into training and test data using 80% for training and 20% for test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)


#a. Support Vector Machine with the linear kernel and default parameters (sklearn.svm.SVC).
linear_svm_model = SVC(kernel='linear') #create the model with default parameters
linear_svm_model.fit(x_train, y_train) #fit the model to the training data

pred_train_linear = linear_svm_model.predict(x_train) #predict based on training
pred_test_linear = linear_svm_model.predict(x_test) #predict based on test

# calculate f-measure for training data using the sklearn functions to calculate precision score and recall score
precision_train_linear = precision_score(y_train, pred_train_linear)
recall_train_linear = recall_score(y_train, pred_train_linear)
f_measure_train_linear = (2 * precision_train_linear * recall_train_linear) / (precision_train_linear + recall_train_linear)

# calculate f-measure for test data using the sklearn functions to calculate precision score and recall score
precision_test_linear = precision_score(y_test, pred_test_linear)
recall_test_linear = recall_score(y_test, pred_test_linear)
f_measure_test_linear = (2 * precision_test_linear * recall_test_linear) / (precision_test_linear + recall_test_linear)

# print f-measure results
print("SVM with linear kernel f-measure:")
print("Training data f-measure: ", f_measure_train_linear)
print("Test data f-measure: ", f_measure_test_linear)

"""
SVM with linear kernel f-measure:
Training data f-measure:  0.9864253393665158
Test data f-measure:  0.9873417721518987
"""

#b. Support Vector Machine with the RBF kernel and default parameters.
rbf_svm_model = SVC(kernel='rbf') #create the model with default parameters
rbf_svm_model.fit(x_train, y_train) #fit the model to the training data

pred_train_rbf = rbf_svm_model.predict(x_train) #predict based on training
pred_test_rbf = rbf_svm_model.predict(x_test) #predict based on test

# calculate f-measure for training data using the sklearn functions to calculate precision score and recall score
precision_train_rbf = precision_score(y_train, pred_train_rbf)
recall_train_rbf = recall_score(y_train, pred_train_rbf)
f_measure_train_rbf = (2 * precision_train_rbf * recall_train_rbf) / (precision_train_rbf + recall_train_rbf)

# calculate f-measure for test data using the sklearn functions to calculate precision score and recall score
precision_test_rbf = precision_score(y_test, pred_test_rbf)
recall_test_rbf = recall_score(y_test, pred_test_rbf)
f_measure_test_rbf = (2 * precision_test_rbf * recall_test_rbf) / (precision_test_rbf + recall_test_rbf)

# print f-measure results
print("\nSVM with RBF kernel f-measure:")
print("Training data f-measure: ", f_measure_train_rbf)
print("Test data f-measure: ", f_measure_test_rbf)
"""
SVM with RBF kernel f-measure:
Training data f-measure:  0.5462555066079295
Test data f-measure:  0.5396825396825397
"""

#c. Random forest with default parameters (sklearn.ensemble.RandomForestClassifier).
rf_model = RandomForestClassifier() #create the model with default parameters
rf_model.fit(x_train, y_train) #fit the model to the training data

pred_train_rf = rf_model.predict(x_train) #predict based on training
pred_test_rf = rf_model.predict(x_test) #predict based on test

# calculate f-measure for training data using the sklearn functions to calculate precision score and recall score
precision_train_rf = precision_score(y_train, pred_train_rf)
recall_train_rf = recall_score(y_train, pred_train_rf)
f_measure_train_rf = (2 * precision_train_rf * recall_train_rf) / (precision_train_rf + recall_train_rf)

# calculate f-measure for test data using the sklearn functions to calculate precision score and recall score
precision_test_rf = precision_score(y_test, pred_test_rf)
recall_test_rf = recall_score(y_test, pred_test_rf)
f_measure_test_rf = (2 * precision_test_rf * recall_test_rf) / (precision_test_rf + recall_test_rf)

# Print results for Random Forest
print("\nRandom Forest f-measure:")
print("Training data f-measure: ", f_measure_train_rf)
print("Test data f-measure: ", f_measure_test_rf)
"""
Random Forest f-measure:
Training data f-measure:  1.0
Test data f-measure:  1.0
"""