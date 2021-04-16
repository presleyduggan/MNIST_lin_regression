#Calculate linear regression of MNIST data
#Uses MNIST_15_15.csv and MNIST_LABEL.csv
import pandas as pd
import numpy as np
from prettytable import PrettyTable

#function to solve linear regression - uses psuedo inverse as inverse causes an error with 0 values
def SolverLinearRegression(X, y):
    return np.dot(np.dot(np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), y)

# grab the data
text_data = pd.read_csv('MNIST_15_15.csv', header=None)
label_data = pd.read_csv('MNIST_LABEL.csv', header=None)

# make blank lists to help with 10-fold cross-validation - will have 10 items of 'equal' (34 for 5 items, and 33 for the other 5 since total is 335) length. This will be the 10 data sets for the model
data_collection = []
label_collection = []
# these variables are used to split the data evenly 
start = 0 
end = 0

all_sum = 0 # for total accuracy

for i in range(1, 11):
    if(i < 6): # for first 5 sets have data of 34
        end = end + 34
    else: # for last 5 sets have data of 33
        end = end + 33
    data_collection.append((text_data.iloc[start:end])/255)
    label_collection.append(label_data.iloc[start:end])
    start = end # next set should start from the end of the previous

t = PrettyTable(['Experiment #', 'TPR', 'FPR', 'Accuracy']) # To make the table

for i in range(0, 10): # 10 experiments ---> 1 set for test and the rest for data
    # create blank lists to copy from original data from. This way we can remove the test set each time without actually deleting it
    data_list = []
    label_list = []

    data_list = data_collection.copy()
    label_list = label_collection.copy()

    # create blank data frames for test and training sets. Label = whether it is 5 or 6
    train = pd.DataFrame()
    train_label = pd.DataFrame()
    test = pd.DataFrame()
    test_label = pd.DataFrame()

    # grab test set and remove from list --> test set will be different each iteration
    test = data_list[i]
    test_label = label_list[i]

    data_list.pop(i)
    label_list.pop(i)

    #put rest into training
    for j in range(0, len(data_list)):
        train = train.append(data_list[j], ignore_index=True)
        train_label = train_label.append(label_list[j], ignore_index=True)

    # start working with data
    n, p = train.shape # number of samples and features

    # if y is greater than 5 (checking for 5 or 6), class will be 1 otherwise 0
    y = np.zeros(n)
    y[train_label.iloc[:, -1] > 5] = 1

    # data preparation for training data
    X = train
    X = pd.DataFrame(np.c_[np.ones(n), X])

    # data preparation for test data
    y_groundtruth = np.zeros(test_label.shape[0])
    y_groundtruth[test_label.iloc[:, -1] > 5] = 1

    X_test = test
    X_test = pd.DataFrame(np.c_[np.ones(len(X_test)), X_test])

    # do linear regression calculations
    b_opt = SolverLinearRegression(X, y)

    # Compute accuracy
    acc = sum(np.array(np.dot(X_test, b_opt) > 0.5) == y_groundtruth) / len(y_groundtruth)

    fp = 0 #false positive
    fn = 0 #false negative
    tp = 0 #true positive
    p = 0 #total positives

    for ii in range(0, len(y_groundtruth)): #calculate TPR and FPR
        if (np.array(np.dot(X_test, b_opt) > 0.5))[ii] and y_groundtruth[ii]:
            tp = tp + 1
            p = p + 1

        if (np.array(np.dot(X_test, b_opt) > 0.5))[ii] != y_groundtruth[ii]:
            if (np.array(np.dot(X_test, b_opt) > 0.5))[ii] == True:
                fp = fp + 1
                p = p + 1
            else:
                fn = fn + 1

    t.add_row([str(i), str(round((tp/p)*100,2)), str(round((fp/p)*100,2)), str(round(acc,2))]) # add experiment number, TPR, FPR, and Accuracy to table
    all_sum = acc + all_sum # save accuracy for average calc

#print total accuracy
print(t)
print("Average Accuracy = "+str(round((all_sum/10)*100,2))+"%")
