import pandas as pd
import numpy as np

# k was determined by testing 1-15 and seeing which k had the highest accuracy
# 4,5,6,7,8 and 9 had the highest accuracy
k_knn = 7

# grab data
test_data = pd.read_csv('MNIST_test.csv')
training_data = pd.read_csv('MNIST_training.csv')

# total accurately predicted
total_correct = 0

# loop through all 50 test sets
for i in range(0,test_data.shape[0]):
	test = test_data.iloc[i]
	test = test.to_numpy() # faster
	save_test_label = test[0] # save label to test prediction
	test = np.delete(test, 0) # delete label because it is not needed in sum
	best_fit = [] # knn nearest neighbors
	best_fit_label = [] # label for best fit distance

	for j in range(0,training_data.shape[0]):
		training = training_data.iloc[j]
		training = training.to_numpy() # faster
		save_train_label = training[0] # save label for knn
		training = np.delete(training,0) # delete so it isn't part of sum

		tot = np.sqrt(np.sum((test-training) ** 2)) # euclidean distance calculation

		# determine which training sets go into closest neighbors 
		if(len(best_fit) < k_knn): # if len < k then add to cloest neighbors list
			best_fit.append(tot)
			best_fit_label.append(training[0])
		else:
			if(tot < max(best_fit)): # check if this neighbor is closer
				best_fit[best_fit.index(max(best_fit))] = tot
				best_fit_label[best_fit.index(max(best_fit))] = save_train_label
	# predict label based on highest count in closest neighbors list
	counts = np.bincount(best_fit_label)
	if(np.argmax(counts) == save_test_label):
		total_correct = total_correct + 1
print("Total correct is "+str(total_correct)+" out of 50")
print(str(total_correct/50)+"%")