from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler

"""
Implement your own logistic regression with regularization algorithm from the lecture slides
using Python. (20 points)

b. Using the implemented algorithm, train and test the data from the attached file ckd_data.zip.
(20 points)
• Use 80% of each class data to train your classifier and the remaining 20% to test it.
• Run different values of logistic regression regularization parameter (λ). The range of λ is
from -2 to 4 and the step is 0.2
• Plot the f-measure of the algorithm’s performance on the training and test sets as a
function of λ:
c. Repeat the procedure in (b) but now using the features normalized with the standardization
protocol discussed in the class. (10 points)
"""


#data cleaning is based on these pages:
# https://stackoverflow.com/questions/62653514/open-an-arff-file-with-scipy-io
# https://stackoverflow.com/questions/40389764/how-to-translate-bytes-objects-into-literal-strings-in-pandas-dataframe-pytho
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

#read in and clean the data
data = arff.loadarff('chronic_kidney_disease_full.arff') #read in the arff file
data = pd.DataFrame(data[0]) #make it into a pandas dataframe

obj_cols = [col for col in data.columns if data[col].dtype == 'object'] #identify which columns are objects and therefore do not contain numerical values
numerical_cols = [col for col in data.columns if data[col].dtype != 'object']#identify which columns are not objects and therefore contain numerical values

data.replace('?', np.nan, inplace=True) #replace '?' to nan (which we can then change to be a numerical value)

#function to replace missing values in the dataset (ie values that were '?' and are now nan
def impute_mode(feature):
    mode = data[feature].mode()[0] #find the mode of the column
    data[feature] = data[feature].fillna(mode) #replace the nan in the column with the mode

for col in numerical_cols: #replace nan in numerical columns with the mode of that column
    impute_mode(col)

#decode object columns to change them from bytes to strings (which we can then change into numerical values)
for col in obj_cols:
    data[col] = data[col].str.decode('utf-8').fillna(data[col])   # decode to string or return nan if missing value

for col in obj_cols: #replace the categorical info with numerical values
    data[col] = LabelEncoder().fit_transform(data[col]) #use the LabelEncoder to encode the object data into numerical data

#extracting the features from the dataset
features = []
for col in data.columns:
    if col != 'class':
        features.append(col)

#logistic regression with regularization
#equations based on lecture slides

def sigmoid_func(z): #create a sigmoid function g(z), g(w^T x)
    return 1/(1 + np.exp(-z))

def cost_with_regularization(x, y, w, lam): # cost function, takes in x, y, weights, and lambda values
    h = sigmoid_func(x.dot(w))
    m = len(y)
    nonzero = 1e-10  # I was getting divide by 0 errors so this is a small value that will prevent those errors
    cost = ((-1 / m) * (y.T.dot(np.log(h + nonzero)) + (1 - y).T.dot(np.log(1 - h + nonzero)))) + ((lam / (2 * m)) * np.sum(w[1:] ** 2))  #cost function + regularization term
    return cost

def gradient_descent_with_regularization(x, y, w, alpha, lam): # gradient descent function, takes in x, y, weights, alpha and lambda values
    #reshaping the data
    m, n = x.shape
    w = w.reshape((n, 1))
    y = y.reshape((m, 1))
    j_list = [] #holds previous values for j, so we can check for convergence
    num_iterations = 0

    while num_iterations < 5000: #running until a convergence is reached or until its been run 5000 times (arbitrary number)
        h = sigmoid_func(x.dot(w))
        gradient = ((1 / m) * x.T.dot(h - y)) + ((lam / m) * np.vstack([0, w[1:]])) #gradient equation + the regularization term, using vstack so that the weight values line up
        w_new = w - alpha * gradient
        j = cost_with_regularization(x, y, w_new, lam)
        j_list.append(j)

        if len(j_list) > 1 and abs(j_list[-2] - j_list[-1]) < 0.001:  # check for convergence, set a difference of 0.001 as the threshold for convergence
            break

        w = w_new
        num_iterations += 1
    return w

#get the data, the cdk class is the dependent variable we will be predicting
x = data[features].values
y = data['class'].values


#split the data into training and test data using 80% for training and 20% for test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

initial_weights = np.zeros(x.shape[1]) #initialize weights as 0
alpha = 0.1 #set the learning rate


# store f measure values for training and testing
f_measure_training_regularization = []
f_measure_testing_regularization = []

# run logistic regression with regularization for lambda values from -2 to 4 with step 0.2
for lam in np.arange(-2, 4.2, 0.2):
    #run gradient descent using the training data
    weights = gradient_descent_with_regularization(x_train, y_train, initial_weights, alpha, lam)

    #predict labels for the training data
    train_predictions = sigmoid_func(x_train.dot(weights))
    train_predictions = (train_predictions >= 0.5).astype(int) #decide if the label is a 1 or a 0

    # calculate precision, recall, and the f-measure
    precision_train_regularization = precision_score(y_train, train_predictions)
    recall_train_regularization = recall_score(y_train, train_predictions)

    if (precision_train_regularization + recall_train_regularization) != 0: #handle divide by 0 error
        f_measure_train_regularization = (2 * precision_train_regularization * recall_train_regularization) / (precision_train_regularization + recall_train_regularization)
    else:
        f_measure_train_regularization = 0

    #predict labels for the testing data
    test_predictions = sigmoid_func(x_test.dot(weights))
    test_predictions = (test_predictions >= 0.5).astype(int) #decide if the label is a 1 or a 0

    #calculate precision, recall, and the f-measure
    precision_test_regularization = precision_score(y_test, test_predictions)
    recall_test_regularization = recall_score(y_test, test_predictions)

    if (precision_test_regularization + recall_test_regularization) != 0:  #handle divide by 0
        f_measure_test_regularization = (2 * precision_test_regularization * recall_test_regularization) / (precision_test_regularization + recall_test_regularization)
    else:
        f_measure_test_regularization = 0

    #add f-measure results to the list
    f_measure_training_regularization.append(f_measure_train_regularization)
    f_measure_testing_regularization.append(f_measure_test_regularization)


# plotting the f-measure for training and testing against the lambda values
plt.plot(np.arange(-2, 4.2, 0.2), f_measure_training_regularization, label='Training f-measures')
plt.plot(np.arange(-2, 4.2, 0.2), f_measure_testing_regularization, label='Testing f-measures')
plt.xlabel('lambda')
plt.ylabel('f-measure')
plt.title('f-measure vs. lambda with regularization')
plt.legend()
plt.show()

#part c- using the features normalized with the standardization

#use StandardScaler from sklearn to standardize the features
#works by calculating the z score of the data --> z = (x - mu) / sigma,  mu = mean of data, sigma = standard deviation
scaler = StandardScaler()
x_train_standardize = scaler.fit_transform(x_train)
x_test_standardize = scaler.transform(x_test)

# store f measure values for training and testing
f_measure_training_standardization = []
f_measure_testing_standardization = []

# run logistic regression with standardization for lambda values from -2 to 4 with step 0.2
for lam in np.arange(-2, 4.2, 0.2):
    #run gradient descent using the training data
    weights = gradient_descent_with_regularization(x_train_standardize, y_train, initial_weights, alpha, lam)

    #predict labels for the training data
    train_predictions_standardization = sigmoid_func(x_train_standardize.dot(weights))
    train_predictions_standardization = (train_predictions_standardization >= 0.5).astype(int) #decide if the label is a 1 or a 0

    # calculate precision, recall, and the f-measure
    precision_train_standardization = precision_score(y_train, train_predictions_standardization)
    recall_train_standardization = recall_score(y_train, train_predictions_standardization)

    if (precision_train_standardization + recall_train_standardization) != 0: #handle divide by 0 error
        f_measure_train_standardization = (2 * precision_train_standardization * recall_train_standardization) / (precision_train_standardization + recall_train_standardization)
    else:
        f_measure_train_standardization = 0

    #predict labels for the testing data
    test_predictions_standardization = sigmoid_func(x_test_standardize.dot(weights))
    test_predictions_standardization = (test_predictions_standardization >= 0.5).astype(int) #decide if the label is a 1 or a 0

    #calculate precision, recall, and the f-measure
    precision_test_standardization = precision_score(y_test, test_predictions_standardization)
    recall_test_standardization = recall_score(y_test, test_predictions_standardization)

    if (precision_test_standardization + recall_test_standardization) != 0:  #handle divide by 0
        f_measure_test_standardization = (2 * precision_test_standardization * recall_test_standardization) / (precision_test_standardization + recall_test_standardization)
    else:
        f_measure_test_standardization = 0

    #add f-measure results to the list
    f_measure_training_standardization.append(f_measure_train_standardization)
    f_measure_testing_standardization.append(f_measure_test_standardization)

# plotting the f-measure for training and testing against the lambda values
plt.plot(np.arange(-2, 4.2, 0.2), f_measure_training_standardization, label='Training f-measures')
plt.plot(np.arange(-2, 4.2, 0.2), f_measure_testing_standardization, label='Testing f-measures')
plt.xlabel('lambda')
plt.ylabel('f-measure')
plt.title('f-measure vs. lambda with standardization')
plt.legend()
plt.show()