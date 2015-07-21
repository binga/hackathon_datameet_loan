import pandas as pd
import numpy as np

from datetime import datetime

from sklearn.ensemble import RandomForestClassifier

from scipy.stats import randint as sp_randint
from sklearn.grid_search import ParameterSampler

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


del train['Net_relationship_value']
del test['Net_relationship_value']

del train['Net_relationship_value_bin']
del test['Net_relationship_value_bin']

del train['Branch_City']
del test['Branch_City']

#del train['AQB']
#del test['AQB']

del train['Base_tag']
del test['Base_tag']

del train['Total_Invest_in_MF']
del test['Total_Invest_in_MF']

del train['salary_dec14']
del test['salary_dec14']

train = pd.get_dummies(train)
test = pd.get_dummies(test)

ids_train = train['ID']
ids_test = test['ID']

y = train['RESPONDER']
target_final = test['RESPONDER']

train.drop(['ID', 'RESPONDER'], inplace = True, axis = 1)
test.drop(['ID', 'RESPONDER'], inplace=True, axis=1)

clf = RandomForestClassifier(n_estimators = 484, max_features = 22, max_depth=23, min_samples_leaf=8, min_samples_split=8,
							bootstrap=True, n_jobs = -1, verbose = False, random_state = 2244)

start = datetime.now()

# Randomized search CV

def zoneEvalMetric(actual, predicted, topTen=30):
    scores = pd.DataFrame({'actual':actual, 'predicted':predicted})
    scores.sort(columns='predicted', ascending=False, inplace=True)
    scores.reset_index(inplace = True)
    numerator = np.sum(scores.ix[0:int((float(topTen)*len(scores))/100), 'actual'])
    total = float(np.sum(scores['actual']))
    evalMetric = float(numerator)/total
    return evalMetric * 100

from scipy.stats import randint as sp_randint

param_dist = {
	'n_estimators' : sp_randint(5,10), # We have to set these correctly for good results
	'max_features' : sp_randint(5,10), # We have to set these correctly for good results
	'max_depth' : sp_randint(5,10), # We have to set these correctly for good results
	'min_samples_leaf':sp_randint(1,10),
	'min_samples_split' : sp_randint(1,10),
	'bootstrap' : [True, False]
}

param_list = list(ParameterSampler(param_dist, n_iter=5))

n_iter = 10 # We have to set these correctly for good results
best_score = 0
best_param = {}

print "Scheduling %s runs..." % n_iter

param_list = list(ParameterSampler(param_dist, n_iter))

for t, run in enumerate(param_list):
	#print run
	clf.set_params(**run)
	clf.fit(train,y)
	train_preds = clf.predict_proba(train)[:,1]
	test_preds = clf.predict_proba(test)[:,1]
	tr_score = zoneEvalMetric(y, train_preds, 30)
	te_score = zoneEvalMetric(target_final, test_preds, 30)

	if te_score > best_score:
		best_score = te_score
		best_param = run
		print "run: %s, train: %s, test: %s| params:%s" % (t, tr_score, te_score, run)

print "Best params: ", best_param
