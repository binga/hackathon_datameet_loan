import pandas as pd
import numpy as np

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

#train1 = pd.read_csv("train_5.csv", usecols=["AQB", "FEB15_Bal"])
#test1 = pd.read_csv("test_5.csv", usecols=["AQB", "FEB15_Bal"])

#train = pd.concat((train, train1), axis=1)
#test = pd.concat((test, test1), axis=1)

del train['Net_relationship_value']
del test['Net_relationship_value']

del train['Net_relationship_value_bin']
del test['Net_relationship_value_bin']

del train['Branch_City']
del test['Branch_City']

#train['Branch_City'] = list(pd.factorize(train['Branch_City']))

del train['AQB']
del test['AQB']

del train['Base_tag']
del test['Base_tag']

del train['Total_Invest_in_MF']
del test['Total_Invest_in_MF']

del train['salary_dec14']
del test['salary_dec14']

# print train['salary_Dec14'].head()

# exit()
# train['mva'] = np.median(train['salary_feb15'] + train['salary_jan15'])# + train['salary_dec14'])
# test['mva'] = np.median(test['salary_feb15'] + test['salary_jan15'])# +
# test['salary_dec14'])

dict_event = {'N': 0, 'Y': 1}
train['event'].replace(dict_event, inplace=True)
test['event'].replace(dict_event, inplace=True)

train['cc_cash_withdrawal_tag'].replace(dict_event, inplace=True)
test['cc_cash_withdrawal_tag'].replace(dict_event, inplace=True)

train['Event_and_Credit_Card_cash_withd'].replace(dict_event, inplace=True)
test['Event_and_Credit_Card_cash_withd'].replace(dict_event, inplace=True)

train['COC_DAE_EASY_EMI_tag'].replace(dict_event, inplace=True)
test['COC_DAE_EASY_EMI_tag'].replace(dict_event, inplace=True)

# train = train[['ID', 'pl_holding', 'RESPONDER']]
# test = test[['ID', 'pl_holding', 'RESPONDER']]

# PAPQ - others  or PA
# IF debit spends < 932 1 else 0
# AQB and Net releance - Standard deviation

train = pd.get_dummies(train)
test = pd.get_dummies(test)

ids_train = train['ID']
ids_test = test['ID']

y = train['RESPONDER']
target_final = test['RESPONDER']

train.drop(['ID', 'RESPONDER'], inplace=True, axis=1)
test.drop(['ID', 'RESPONDER'], inplace=True, axis=1)

import xgboost as xgb

dtrain = xgb.DMatrix(train, label=y, missing=-1)
dtest = xgb.DMatrix(test, target_final, missing=-1)
watchlist = [(dtest, 'test'), (dtrain, 'train')]

params = {'objective': 'binary:logistic', 'max_depth': 7, 'eta': 0.07, 'silent': 1, 'subsample': 0.6,  # 'eval_metric':'auc',
          'nthread': 4, 'colsample_bytree': 0.7, 'missing': -1,  'scale_pos_weight': 3}

num_rounds = 150

#xgb.cv(param, dtrain, num_round, nfold = 5, seed = 0, feval=evalfunc, obj = Giniii)


def zoneEvalMetric(actual, predicted, topTen=10):
    scores = pd.DataFrame({'actual': actual, 'predicted': predicted})
    scores.sort(columns='predicted', ascending=False, inplace=True)
    scores.reset_index(inplace=True)
    numerator = np.sum(
        scores.ix[0:int((float(topTen)*len(scores))/100), 'actual'])
    total = float(np.sum(scores['actual']))
    evalMetric = float(numerator)/total
    return evalMetric * 100


def zoneEvalMetricXgB(predicted, dtrain, topTen=10):
    actual = dtrain.get_label()
    scores = pd.DataFrame({'actual': actual, 'predicted': predicted})
    scores.sort(columns='predicted', ascending=False, inplace=True)
    scores.reset_index(inplace=True)
    numerator = np.sum(
        scores.ix[0:int((float(topTen)*len(scores))/100), 'actual'])
    total = float(np.sum(scores['actual']))
    evalMetric = float(numerator)/total
    return 'metric', evalMetric * 100

bst = xgb.train(params, dtrain, num_rounds, watchlist, feval=zoneEvalMetricXgB)


print bst.get_fscore()

#train_preds = bst.predict(train)
test_preds = bst.predict(dtest)
submit = pd.DataFrame(
    {'ID': ids_test, 'probability': test_preds, 'RESPONDER': target_final})
submit.to_csv("data/Finalprobs.csv", index=False)


# print "Train eval score:", zoneEvalMetric(y, train_preds, 10)
# print "Test eval score:", zoneEvalMetric(target_final, test_preds, 10)


#                             features  importance
# 69                      pl_holding_a    0.085316
# 71                      pl_holding_c    0.071364
# 2             Net_relationship_value    0.064362
# 14                         FEB15_Bal    0.061576
# 1                                AQB    0.060755
# 15                         FEB15_EOP    0.057640
# 4                        DC_SPEND_3M    0.051945
# 3        Number_Balance_enquiries_3m    0.051476
# 5                        CC_SPEND_3M    0.046574
# 0                                AGE    0.038941
# 13                      product_code    0.025612
# 7                Total_asset_holding    0.019522
# 27            COC_DAE_EASY_EMI_tag_N    0.018774
# 19                Total_Invest_in_MF    0.017487
# 25         Transactor_revolver_tag_R    0.016750
# 17                    RD_AMOUNT_BOOK    0.016475
# 28            COC_DAE_EASY_EMI_tag_Y    0.015820
# 18                    Dmat_Investing    0.015767
# 16                    FD_AMOUNT_BOOK    0.014933
# 58               ratio_eop_amb_bin_j    0.013485
# 10              Prematured_rd_cnt_3M    0.009060
# 65             balance_enquiry_bin_g    0.007844
# 70                      pl_holding_b    0.007735
# 32                    PAPQ_Tag_OTHER    0.007651
# 75        OUR_BANK_OTHER_BANK_loan_d    0.007216
# 33                       PAPQ_Tag_PA    0.007202
# 20                  Account_type_CSA    0.006831
# 66                     hnw_segment_a    0.006724
# 21                   Account_type_SA    0.006267
# 9               Prematured_fd_cnt_3M    0.005995
# ..                               ...         ...
# 11                  closed_rd_cnt_3M    0.002544
# 34                       PAPQ_Tag_PQ    0.002407
# 62             balance_enquiry_bin_d    0.002395
# 83      Net_relationship_value_bin_h    0.002391
# 6   Number_Accounts_under_Joint_acco    0.002346
# 24         Transactor_revolver_tag_N    0.002331
# 42            Prematured_FD_Tag_3M_Y    0.002325
# 81      Net_relationship_value_bin_f    0.002311
# 67                     hnw_segment_b    0.002264
# 57               ratio_eop_amb_bin_i    0.002239
# 53               ratio_eop_amb_bin_e    0.002146
# 84      Net_relationship_value_bin_i    0.002104
# 46                closed_RD_Tag_3M_Y    0.002011
# 38               Joint_account_tag_Y    0.001930
# 37               Joint_account_tag_N    0.001866
# 56               ratio_eop_amb_bin_h    0.001799
# 45                closed_RD_Tag_3M_N    0.001695
# 22         Transactor_revolver_tag_D    0.001687
# 55               ratio_eop_amb_bin_g    0.001377
# 47                closed_FD_Tag_3M_N    0.001301
# 36                     CITY_CHANGE_Y    0.001276
# 35                     CITY_CHANGE_N    0.001260
# 8                  CHQ_BOUNCE_CNT_3M    0.001232
# 40               CHQ_BOUNCE_TAG_3M_Y    0.001166
# 48                closed_FD_Tag_3M_Y    0.001128
# 39               CHQ_BOUNCE_TAG_3M_N    0.001080
# 49               ratio_eop_amb_bin_a    0.000352
# 76      Net_relationship_value_bin_a    0.000338
# 51               ratio_eop_amb_bin_c    0.000242
# 50               ratio_eop_amb_bin_b    0.000225


# [88]	test-auc:0.841064	train-auc:0.897911
# [89]	test-auc:0.841113	train-auc:0.898692
# [90]	test-auc:0.841356	train-auc:0.899177
# [91]	test-auc:0.841715	train-auc:0.899989
# [92]	test-auc:0.842159	train-auc:0.900909
# [93]	test-auc:0.842195	train-auc:0.901197
# [94]	test-auc:0.842321	train-auc:0.901914
# [95]	test-auc:0.842431	train-auc:0.902744
# [96]	test-auc:0.842443	train-auc:0.903364
# [97]	test-auc:0.842638	train-auc:0.904153
# [98]	test-auc:0.842991	train-auc:0.904884
# [99]	test-auc:0.842881	train-auc:0.905594
# [100]	test-auc:0.843099	train-auc:0.905982
# [101]	test-auc:0.843274	train-auc:0.906518
# [102]	test-auc:0.843193	train-auc:0.907391
# [103]	test-auc:0.843131	train-auc:0.908127
# [104]	test-auc:0.843154	train-auc:0.908959
# [105]	test-auc:0.843104	train-auc:0.909561
# [106]	test-auc:0.843318	train-auc:0.909962
