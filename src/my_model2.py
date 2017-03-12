import utils
import etl
import cross
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import *
from sklearn.naive_bayes import GaussianNB


RANDOM_STATE = 545510477

#Note: You can reuse code that you wrote in etl.py and models.py and cross.py over here. It might help.
# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

'''
You may generate your own features over here.
Note that for the test data, all events are already filtered such that they fall in the observation window of their respective patients. Thus, if you were to generate features similar to those you constructed in code/etl.py for the test data, all you have to do is aggregate events for each patient.
IMPORTANT: Store your test data features in a file called "test_features.txt" where each line has the
patient_id followed by a space and the corresponding feature in sparse format.
Eg of a line:
60 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514
Here, 60 is the patient id and 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514 is the feature for the patient with id 60.

Save the file as "test_features.txt" and save it inside the folder deliverables

input:
output: X_train,Y_train,X_test
'''
def my_features():
	X_train, Y_train = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	filepath='../data/test/'
	filtered_events= pd.read_csv(filepath + 'events.csv')[['patient_id', 'event_id', 'value']]
	feature_map=pd.read_csv(filepath + 'event_feature_map.csv')
	aggregated_events=etl.aggregate_events(filtered_events, None, feature_map, '')
	patient_features=aggregated_events.groupby('patient_id')[['feature_id','feature_value']].apply(lambda x: [tuple(x) for x in x.values]).to_dict()
	events_mortality=pd.DataFrame(aggregated_events['patient_id'])
	events_mortality['label']=aggregated_events['patient_id']
	mortality=events_mortality.set_index('patient_id')['label'].to_dict()
	etl.save_svmlight(patient_features, mortality, '../deliverables/test_features.txt', '../deliverables/features.txt')
	X_test= load_svmlight_file('../deliverables/test_features.txt',n_features=3190)[0]
	X_testt, Y_testt = utils.get_data_from_svmlight("../data/features_svmlight.validate")
	clf=GradientBoostingClassifier()
	clf=clf.fit(X_train,Y_train)
	model = SelectFromModel(clf, prefit=True)
	X_train_n=model.transform(X_train)
	X_test_n=model.transform(X_test)
	X_testt_n=model.transform(X_testt)

	return X_train_n.todense(),Y_train,X_test_n.todense(),X_testt_n.todense(),Y_testt
	# print(len(X_train.todense()))
	# return X_train.todense(),Y_train,X_test.todense(),Y_test

'''
You can use any model you wish.

input: X_train, Y_train, X_test
output: Y_pred
'''
def my_classifier_predictions(X_train,Y_train,X_test,X_testt,Y_testt):
	#TODO: complete this

	clf1=GradientBoostingClassifier()
	clf3=GaussianNB()
	eclf=VotingClassifier(estimators=[('gbc',clf1),('gnb',clf3)],voting='soft')
	eclf=eclf.fit(X_train,Y_train)
	Y_train_pred=eclf.predict(X_train)
	print('train:')
	print(roc_auc_score(Y_train,Y_train_pred))
	Y_pred=eclf.predict_proba(X_test)[:,1]
	print('test:')

	Y_predt=eclf.predict(X_testt)
	print(roc_auc_score(Y_testt,Y_predt))

	# parameters1=np.arange(0.5,1,0.01)
	# score1=[]
	# score2=[]
	# for parameter1 in parameters1:
	# 	clf=GradientBoostingClassifier(subsample=parameter1,max_features=28,max_depth=3,learning_rate=0.16,n_estimators=60,random_state=RANDOM_STATE)
	# 	clf=clf.fit(X_train,Y_train)
	# 	score_train=clf.score(X_train,Y_train)
	# 	score_test=clf.score(X_test,Y_test)
	# 	score1.append(score_train)
	# 	score2.append(score_test)
	# 	print(parameter1)
	# print(score1)
	# print(score2)
	# return 0







# best_params_
	return Y_pred.flatten()


def main():
	X_train, Y_train, X_test,X_testt,Y_testt= my_features()
	# my_classifier_predictions(X_train,Y_train,X_test,Y_test)
	Y_pred = my_classifier_predictions(X_train,Y_train,X_test,X_testt,Y_testt)
	# print('test:')
	# print(roc_auc_score(Y_test,Y_pred))
	utils.generate_submission("../deliverables/test_features.txt",Y_pred)
	#The above function will generate a csv file of (patient_id,predicted label) and will be saved as "my_predictions.csv" in the deliverables folder.

if __name__ == "__main__":
    main()

	