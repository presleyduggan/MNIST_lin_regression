import numpy as np
import pandas as pd
from prettytable import PrettyTable as pt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# grab the data
data = pd.read_csv('MNIST_DATA.csv')

# randomize and split the label and pixel data
data = data.sample(frac=1).reset_index(drop=True)
X = data.drop('label', axis = 1)
X = X/255 # normalize
Y = data['label']

# split the data into 5-fold
x_folds = np.array_split(X, 5)
y_folds = np.array_split(Y, 5)

# initialize accuracy lists
lin_accuracy = []
poly_accuracy = []
rbf_accuracy = []

# make SVC models
lin_model = SVC(kernel= 'linear')
poly_model = SVC(kernel= 'poly')
rbf_model = SVC(kernel= 'rbf')
t = pt(['Experiment #', 'Linear Accuracy', 'Poly Accuracy', 'RBF Accuracy']) # To make the table

for i in range(0,5): # 5 fold 
    x_train = pd.DataFrame()
    x_test = pd.DataFrame()
    y_train = pd.DataFrame(columns= ['label'])
    y_test = pd.DataFrame()
    # split into test and training --> current i value will be the test set
    for j in range(0,5): # make training sets 
        if j != i:
            x_train = x_train.append(x_folds[j])
            y_train = pd.concat([y_train, y_folds[j]])
        else: # if i value then we save as test
            x_test = x_test.append(x_folds[j])
            y_test = y_test.append(y_folds[j], ignore_index=True)

    y_train = y_train.drop('label', axis=1) #remove extra blank column
    
    # train linear model and predict
    lin_model.fit(x_train, y_train.values.ravel()) # ravel to just get values
    y_predict = lin_model.predict(x_test)
    # save accuracy 
    lin_accuracy.append(accuracy_score(y_test.values.ravel(),y_predict))

    # train poly model and predict
    poly_model.fit(x_train, y_train.values.ravel()) # ravel to just get values
    y_predict = poly_model.predict(x_test)
    # save accuracy
    poly_accuracy.append(accuracy_score(y_test.values.ravel(),y_predict))

    # train rbf model and predict
    rbf_model.fit(x_train, y_train.values.ravel()) # ravel to just get values
    y_predict = rbf_model.predict(x_test)
    # save accuracy
    rbf_accuracy.append(accuracy_score(y_test.values.ravel(),y_predict))

    # add values to table
    t.add_row([str(i), str(round(lin_accuracy[i]*100,2)), str(round(poly_accuracy[i]*100,2)), str(round(rbf_accuracy[i]*100,2))])

t.add_row(["Average: ", str(round(np.mean(lin_accuracy)*100,2)), str(round(np.mean(poly_accuracy)*100,2)), str(round(np.mean(rbf_accuracy)*100,2))]) # add the average across all experiments row
print(t) # print the beautiful table