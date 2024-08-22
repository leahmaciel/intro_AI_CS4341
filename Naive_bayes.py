import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score

"""
A. You should implement from scratch a Naïve Bayes classifier (using the spam filter example discussed in class).
Also implement Laplacian smoothing to handle words not in the dictionary. (50 points)

B. Using the implemented algorithm, train and test the model for each of the 2 datasets.
Use 80% of each class data to train your classifier and the remaining 20% to test it.
Which dataset provides better classification i.e. email body or email subject?

C. Compare your classifier with the scikit-learn implementation (sklearn.naive_bayes.MultinomialNB).
Repeat the analysis from (b)
"""

#read in data
subjects = pd.read_csv('dbworld_subjects_stemmed.csv')
bodies = pd.read_csv('dbworld_bodies_stemmed.csv')

# remove the id column
subjects = subjects.iloc[:, 1:]
bodies = bodies.iloc[:, 1:]


#A. You should implement from scratch a Naïve Bayes classifier (using the spam filter example discussed in class).
#Also implement Laplacian smoothing to handle words not in the dictionary. (50 points)

"""
using the training data which is a dataframe where the columns are words and the rows are if the word occured or not
I want to get the probability of each class occurring
I want to calculate the probability of each word being in spam and in ham using laplacian smoothing
I want to store the probabilities as a dictionary
"""
#create a dictionary with the probability of words from the training data
#calculate the probability of each word being in spam and in ham using laplacian smoothing
def get_word_probs(training_data):
    k = 1  #Laplacian smoothing tunable parameter
    Nw = 2 #number of classes (2 because spam or ham)
    N = len(training_data.columns) - 1  # total number of words, -1 because the last column is the class

    class_probabilities = {} #create a dictionary to hold the class probabilities
    num_messages = len(training_data) #get the total number of messages

    word_probs_dict = {} #create a dictionary to hold the word probabilities
    spam_words = training_data[training_data.iloc[:, -1] == 0] #get the words that are spam
    ham_words = training_data[training_data.iloc[:, -1] == 1] #get the words that are ham

    for word in training_data.columns[:-1]:
        count_x_spam = 0 #initialize counts for each word
        count_x_ham = 0

        for index, message in spam_words.iterrows(): #loop through the spam words and increase the count if the word is in the message
            if message[word] == 1:
                count_x_spam += 1

        for index, message in ham_words.iterrows(): #loop through the ham words and increase the count if the word is in the message
            if message[word] == 1:
                count_x_ham += 1

        spam_prob = (count_x_spam + k) / (N + k * Nw) #calculate probability with Laplacian smoothing
        ham_prob = (count_x_ham + k) / (N + k * Nw) #calculate probability with Laplacian smoothing

        word_probs_dict[word] = {0: spam_prob, 1: ham_prob} #add the word probabilites to the dictionary


    for class_label in training_data.iloc[:, -1]: #loop through the class column
        class_count = len(training_data[training_data.iloc[:, -1] == class_label]) #get the counts of each class
        class_prob = class_count / num_messages #get the probability of each class occuring
        class_probabilities[class_label] = class_prob #add the calculated probability to the dictionary

    return class_probabilities, word_probs_dict

#function to predict whether the messages are spam or ham
def predict(messages, class_probabilities, word_probabilities):
    predictions =[] #list to store the predictions

    for index, row in messages.iterrows(): #loop through each message
        spam_probability = np.log(class_probabilities[0])  # get the probability of each class, using log
        ham_probability = np.log(class_probabilities[1])

        for word, word_value in row.items():  # loop through each word in the message
            if word in word_probabilities:  # check if the word is in the dictionary of training words
                if word_value == 1:  # if the word is in the word training dictionary
                    spam_probability += np.log(word_probabilities[word][0]) # update the probability of the message with the probability in the dictionary
                    ham_probability += np.log(word_probabilities[word][1])
                else:  # if the word is not in the word training dictionary
                    spam_probability += np.log(1 - word_probabilities[word][0]) # update the probability by taking the complement (prob that the word is not in the message)
                    ham_probability += np.log(1 - word_probabilities[word][1])

        if spam_probability > ham_probability: #the prediction is whichever class has the greater probability
            prediction = 0
        else:
            prediction = 1
        predictions.append(prediction) #add the prediction value to the list

    return predictions

#B. Using the implemented algorithm, train and test the model for each of the 2 datasets.
#Use 80% of each class data to train your classifier and the remaining 20% to test it.

subjects_train, subjects_test = train_test_split(subjects, test_size=0.2) # split data into training and testing (80:20)
bodies_train, bodies_test = train_test_split(bodies, test_size=0.2)

#run model on subject dataset
subject_class_probs, subject_word_probs = get_word_probs(subjects_train) #calculate word probabilities for training data
subject_train_predictions = predict(subjects_train, subject_class_probs, subject_word_probs) #predict on training data
subject_test_predictions = predict(subjects_test, subject_class_probs, subject_word_probs) #predict on testing data

subject_train_true_labels = subjects_train.iloc[:, -1].tolist() #get the true class labels for training and testing
subject_test_true_labels = subjects_test.iloc[:, -1].tolist()

# calculate f-measure for training data using the sklearn functions to calculate precision score and recall score
precision_train_subjects = precision_score(subject_train_true_labels, subject_train_predictions)
recall_train_subjects = recall_score(subject_train_true_labels, subject_train_predictions)
f_measure_train_subjects = (2 * precision_train_subjects * recall_train_subjects) / (precision_train_subjects + recall_train_subjects)

# calculate f-measure for testing data using the sklearn functions to calculate precision score and recall score
precision_test_subjects = precision_score(subject_test_true_labels, subject_test_predictions)
recall_test_subjects = recall_score(subject_test_true_labels, subject_test_predictions)
f_measure_test_subjects = (2 * precision_test_subjects * recall_test_subjects) / (precision_test_subjects + recall_test_subjects)

#print out the f-measure for subjects for training and testing
print("\nNaive Bayes using my method on subject dataset:")
print("Training data f-measure: ", f_measure_train_subjects)
print("Testing data f-measure: ", f_measure_test_subjects)

"""
Naive Bayes using my method on subject dataset:
Training data f-measure:  0.9811320754716981
Testing data f-measure:  0.8571428571428571
"""

#run model on bodies dataset
bodies_class_probs, bodies_word_probs = get_word_probs(bodies_train)  #calculate word probabilities for training data
bodies_train_predictions = predict(bodies_train, bodies_class_probs, bodies_word_probs) #predict on training data
bodies_test_predictions = predict(bodies_test, bodies_class_probs, bodies_word_probs) #predict on testing data

bodies_train_true_labels = bodies_train.iloc[:, -1].tolist() #get the true class labels for training and testing
bodies_test_true_labels = bodies_test.iloc[:, -1].tolist()

# calculate f-measure for training data using the sklearn functions to calculate precision score and recall score
precision_train_bodies = precision_score(bodies_train_true_labels, bodies_train_predictions)
recall_train_bodies = recall_score(bodies_train_true_labels, bodies_train_predictions)
f_measure_train_bodies = (2 * precision_train_bodies * recall_train_bodies) / (precision_train_bodies + recall_train_bodies)

# calculate f-measure for testing data using the sklearn functions to calculate precision score and recall score
precision_test_bodies = precision_score(bodies_test_true_labels, bodies_test_predictions)
recall_test_bodies = recall_score(bodies_test_true_labels, bodies_test_predictions)
f_measure_test_bodies = (2 * precision_test_bodies * recall_test_bodies) / (precision_test_bodies + recall_test_bodies)

#print out the f-measure for subjects for training and testing
print("\nNaive Bayes using my method on bodies dataset:")
print("Training data f-measure: ", f_measure_train_bodies)
print("Testing data f-measure: ", f_measure_test_bodies)

"""
Naive Bayes using my method on bodies dataset:
Training data f-measure:  0.9777777777777777
Testing data f-measure:  0.9333333333333333
"""

#c.  Compare your classifier with the scikit-learn implementation (sklearn.naive_bayes.MultinomialNB). Repeat the analysis from (b)
# splitting the datasets into x which is the words and y which is the class (spam or ham)
x_subjects = subjects.iloc[:, :-1]
x_bodies = bodies.iloc[:, :-1]

y_subjects = subjects.iloc[:, -1]
y_bodies = bodies.iloc[:, -1]

# split the data into training and testing (80:20)
x_subjects_train, x_subjects_test, y_subjects_train, y_subjects_test = train_test_split(x_subjects, y_subjects, test_size=0.2)
x_bodies_train, x_bodies_test, y_bodies_train, y_bodies_test = train_test_split(x_bodies, y_bodies, test_size=0.2)

#naive bayes on subjects dataset
naive_bayes_sklearn_subjects = MultinomialNB() #make the model
naive_bayes_sklearn_subjects.fit(x_subjects_train, y_subjects_train) #train the model

pred_train_sklearn_subjects = naive_bayes_sklearn_subjects.predict(x_subjects_train) #predict based on training dataset
pred_test_sklearn_subjects = naive_bayes_sklearn_subjects.predict(x_subjects_test) #predict based on testing dataset

# calculate f-measure for training data using the sklearn functions to calculate precision score and recall score
precision_train_sklearn_subjects = precision_score(y_subjects_train, pred_train_sklearn_subjects)
recall_train_sklearn_subjects = recall_score(y_subjects_train, pred_train_sklearn_subjects)
f_measure_train_sklearn_subjects = (2 * precision_train_sklearn_subjects * recall_train_sklearn_subjects) / (precision_train_sklearn_subjects + recall_train_sklearn_subjects)

# calculate f-measure for testing data using the sklearn functions to calculate precision score and recall score
precision_test_sklearn_subjects = precision_score(y_subjects_test, pred_test_sklearn_subjects)
recall_test_sklearn_subjects = recall_score(y_subjects_test, pred_test_sklearn_subjects)
f_measure_test_sklearn_subjects = (2 * precision_test_sklearn_subjects * recall_test_sklearn_subjects) / (precision_test_sklearn_subjects + recall_test_sklearn_subjects)

#print out the f-measure for subjects for training and testing
print("\nNaive Bayes using sklearn method on subjects dataset:")
print("Training data f-measure: ", f_measure_train_sklearn_subjects)
print("Testing data f-measure: ", f_measure_test_sklearn_subjects)
"""
Naive Bayes using sklearn method on subjects dataset:
Training data f-measure:  0.9824561403508771
Testing data f-measure:  0.6666666666666666
"""

#naive bayes on bodies dataset
naive_bayes_sklearn_bodies = MultinomialNB() #make the model
naive_bayes_sklearn_bodies.fit(x_bodies_train, y_bodies_train) #train the model

pred_train_sklearn_bodies = naive_bayes_sklearn_bodies.predict(x_bodies_train) #predict based on training dataset
pred_test_sklearn_bodies = naive_bayes_sklearn_bodies.predict(x_bodies_test) #predict based on testing dataset

# calculate f-measure for training data using the sklearn functions to calculate precision score and recall score
precision_train_sklearn_bodies = precision_score(y_bodies_train, pred_train_sklearn_bodies)
recall_train_sklearn_bodies = recall_score(y_bodies_train, pred_train_sklearn_bodies)
f_measure_train_sklearn_bodies = (2 * precision_train_sklearn_bodies * recall_train_sklearn_bodies) / (precision_train_sklearn_bodies + recall_train_sklearn_bodies)

# calculate f-measure for testing data using the sklearn functions to calculate precision score and recall score
precision_test_sklearn_bodies = precision_score(y_bodies_test, pred_test_sklearn_bodies)
recall_test_sklearn_bodies = recall_score(y_bodies_test, pred_test_sklearn_bodies)
f_measure_test_sklearn_bodies = (2 * precision_test_sklearn_bodies * recall_test_sklearn_bodies) / (precision_test_sklearn_bodies + recall_test_sklearn_bodies)

#print out the f-measure for bodies for training and testing
print("\nNaive Bayes using sklearn method on bodies dataset:")
print("Training data f-measure: ", f_measure_train_sklearn_bodies)
print("Testing data f-measure: ", f_measure_test_sklearn_bodies)
"""
Naive Bayes using sklearn method on bodies dataset:
Training data f-measure:  0.9787234042553191
Testing data f-measure:  0.923076923076923
"""