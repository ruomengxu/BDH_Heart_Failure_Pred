import models_partc
from sklearn.model_selection import KFold,ShuffleSplit
from numpy import mean

import utils

# USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

# USE THIS RANDOM STATE FOR ALL OF YOUR CROSS
# VALIDATION TESTS OR THE TESTS WILL NEVER PASS
RANDOM_STATE = 545510477

#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_kfold(X,Y,k=5):
	#TODO:First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the folds
	acc=0
	auc_=0
	kf = KFold(n_splits=k,random_state=RANDOM_STATE)
	for train_index,test_index in kf.split(X):
		Y_pred=models_partc.logistic_regression_pred(X[train_index],Y[train_index],X[test_index])
		acc_,auc__,precision_,recall_,f1score_=models_partc.classification_metrics(Y_pred, Y[test_index])
		acc+=acc_
		auc_+=auc__
	return acc/k,auc_/5


#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_randomisedCV(X,Y,iterNo=5,test_percent=0.2):
	#TODO: First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the iterations
	acc=0
	auc_=0
	ss = ShuffleSplit(n_splits=iterNo,test_size=test_percent,random_state=RANDOM_STATE)
	for train_index,test_index in ss.split(X):
		Y_pred=models_partc.logistic_regression_pred(X[train_index],Y[train_index],X[test_index])
		acc_,auc__,precision_,recall_,f1score_=models_partc.classification_metrics(Y_pred, Y[test_index])
		acc+=acc_
		auc_+=auc__
	return acc/iterNo,auc_/iterNo


def main():
	X,Y = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	print "Classifier: Logistic Regression__________"
	acc_k,auc_k = get_acc_auc_kfold(X,Y)
	print "Average Accuracy in KFold CV: "+str(acc_k)
	print "Average AUC in KFold CV: "+str(auc_k)
	acc_r,auc_r = get_acc_auc_randomisedCV(X,Y)
	print "Average Accuracy in Randomised CV: "+str(acc_r)
	print "Average AUC in Randomised CV: "+str(auc_r)

if __name__ == "__main__":
	main()

