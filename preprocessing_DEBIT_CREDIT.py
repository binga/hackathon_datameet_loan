import pandas as pd
import numpy as np

data = pd.read_csv("data/Customer_level_data_PL", sep="|")
debit_card_transaction_data = pd.read_csv(
    "data/Debit_card_transaction_data_PL", sep="|")
credit_card_transaction_data_feb = pd.read_csv('data/casa_txn_feb15', sep='|')
credit_card_transaction_data_jan = pd.read_csv('data/casa_txn_jan15', sep='|')
credit_card_transaction_data_dec = pd.read_csv('data/casa_txn_dec14', sep='|')
# casa_transaction_data=pd.read_csv("data/Customer_level_data_PL", sep="|")


def add_debit_spendings(data, debit_card_transaction_data):
    debit_card_transaction_data.rename(columns={'UCIC': 'ID'}, inplace=True)
    ID_grouped_debit_transactions = debit_card_transaction_data.groupby(
        ['ID', 'month']).sum()['Spend']
    ID_grouped_debit_transactions = ID_grouped_debit_transactions.reset_index().pivot(
        index='ID', columns='month', values='Spend')
    ID_grouped_debit_transactions = ID_grouped_debit_transactions.fillna(0)
    data_new = pd.merge(data,
                        ID_grouped_debit_transactions, left_on='ID', right_index=True, how='left')
    data_new.rename(columns={'DEC14': 'DEC14_debit',
                             'JAN15': 'JAN15_debit', 'FEB15': 'FEB15_debit'}, inplace=True)
    return data_new


def add_credit_spendings(data, credit_card_transaction_data):
    credit_card_transaction_data.rename(
        columns={'N_F_BRANCH_TXN_AMT_LCY': 'Spend'}, inplace=True)
    credit_card_transaction_data = credit_card_transaction_data.loc[
        credit_card_transaction_data['F_F_BRANCH_TXN_DRCR_IND'] == 'C', ['ID', 'month_flag', 'Spend']]
    ID_grouped_credit_transactions = credit_card_transaction_data.groupby(
        ['ID', 'month_flag']).sum()['Spend']
    ID_grouped_credit_transactions = ID_grouped_credit_transactions.reset_index(
    ).pivot(index='ID', columns='month_flag', values='Spend')
    ID_grouped_credit_transactions = ID_grouped_credit_transactions.fillna(0)
    data_new = pd.merge(data,
                        ID_grouped_credit_transactions, left_on='ID', right_index=True, how='left')
    data_new.rename(columns={'DEC14': 'DEC14_credit',
                             'JAN15': 'JAN15_credit', 'FEB15': 'FEB15_credit'}, inplace=True)
    return data_new


def num_to_categorical(x):
    if x == 0:
        return 0
    else:
        return 1


def add_category_spend_percentage(data, debit_card_transaction_data):
    debit_card_transaction_data.rename(columns={'UCIC': 'ID'}, inplace=True)
    ID_total_debit_spend = debit_card_transaction_data.groupby(
        'ID').sum()['Spend'].reset_index()
    df_debit_transactions_normalized = pd.merge(
        debit_card_transaction_data, ID_total_debit_spend, left_on='ID', right_on='ID')
    df_debit_transactions_normalized['normalized_spend'] = df_debit_transactions_normalized[
        'Spend_x']*1.0/df_debit_transactions_normalized['Spend_y']
    df_debit_transactions_normalized_category_wise = df_debit_transactions_normalized.groupby(
        ['ID', 'V_D_VP_MCC_CAT_DESCR']).sum().reset_index()
    df_debit_transactions_normalized_category_wise = df_debit_transactions_normalized_category_wise.pivot(
        index='ID', columns='V_D_VP_MCC_CAT_DESCR', values='normalized_spend').fillna(0).applymap(num_to_categorical)[['PETROL', 'RESTAURANT', 'SUPERMKT', 'HOTELS', 'JEWELLERY']]
    # [['AIR', 'CASH', 'EDUCATION', 'JEWELLERY', 'TRAVEL', 'OTHERS', 'OXFAM']]
    data_new = pd.merge(
        data, df_debit_transactions_normalized_category_wise, left_on='ID', right_index=True, how='left')
    return data_new


data = add_debit_spendings(data, debit_card_transaction_data)
# WITHOUT add_category_spend_percentage function 83.5, eta=0.05
# With add_category_spend_percentage boolean, 83.0

data = add_category_spend_percentage(data, debit_card_transaction_data)
data = add_credit_spendings(data, credit_card_transaction_data_dec)
data = add_credit_spendings(data, credit_card_transaction_data_jan)
data = add_credit_spendings(data, credit_card_transaction_data_feb)

# print np.sum(pd.isnull(data))

data.ix[pd.isnull(data['JAN15_Bal']), 'JAN15_Bal'] = np.mean(data['JAN15_Bal'])
data.ix[pd.isnull(data['JAN15_EOP']), 'JAN15_EOP'] = np.mean(data['JAN15_EOP'])
data.ix[pd.isnull(data['DEC14_EOP']), 'DEC14_EOP'] = np.mean(data['DEC14_EOP'])
data.ix[pd.isnull(data['DEC14_Bal']), 'DEC14_Bal'] = np.mean(data['DEC14_Bal'])
data.ix[pd.isnull(data['NOV14_EOP']), 'NOV14_EOP'] = np.mean(data['NOV14_EOP'])
data.ix[pd.isnull(data['NOV14_Bal']), 'NOV14_Bal'] = np.mean(data['NOV14_Bal'])
data.ix[pd.isnull(data['OCT14_EOP']), 'OCT14_EOP'] = np.mean(data['OCT14_EOP'])
data.ix[pd.isnull(data['OCT14_Bal']), 'OCT14_Bal'] = np.mean(data['OCT14_Bal'])
data.ix[pd.isnull(data['SEP14_EOP']), 'SEP14_EOP'] = np.mean(data['SEP14_EOP'])
data.ix[pd.isnull(data['SEP14_Bal']), 'SEP14_Bal'] = np.mean(data['SEP14_Bal'])

data.ix[pd.isnull(data['DEC14_debit']), 'DEC14_debit'] = -1
data.ix[pd.isnull(data['JAN15_debit']), 'JAN15_debit'] = -1
data.ix[pd.isnull(data['FEB15_debit']), 'FEB15_debit'] = -1
data.ix[pd.isnull(data['DEC14_credit']), 'DEC14_credit'] = -1
data.ix[pd.isnull(data['JAN15_credit']), 'JAN15_credit'] = -1
data.ix[pd.isnull(data['FEB15_credit']), 'FEB15_credit'] = -1

# debit_cat_list = ['AIR', 'APPARELS', 'AUTOMOBILE', 'CASH', 'DEPTSTORE', 'EDUCATION', 'ELECTRONIC', 'ENTMNT', 'GROCERIES', 'HOBBY', 'HOME_DECOR', 'HOTELS',
#                   'JEWELLERY', 'MEDICAL', 'MOTO', 'OTHERS', 'OXFAM', 'PETROL', 'PRSNL CARE', 'RAILWAY', 'RESTAURANT', 'SERVICES', 'SUPERMKT', 'TELECOM', 'TRAVEL', 'UTILITY']

debit_cat_list = ['PETROL', 'RESTAURANT', 'SUPERMKT', 'HOTELS', 'JEWELLERY']
for category in debit_cat_list:
    data.ix[pd.isnull(data[category]), category] = -1


# np.mean(data['SEP14_Bal'])
data.ix[pd.isnull(data['salary_sep14']), 'salary_sep14'] = -1
# np.mean(data['SEP14_Bal'])
data.ix[pd.isnull(data['salary_oct14']), 'salary_oct14'] = -1
# np.mean(data['SEP14_Bal'])
data.ix[pd.isnull(data['salary_nov14']), 'salary_nov14'] = -1
# np.mean(data['SEP14_Bal'])
data.ix[pd.isnull(data['salary_dec14']), 'salary_dec14'] = -1
# np.mean(data['SEP14_Bal'])
data.ix[pd.isnull(data['salary_jan15']), 'salary_jan15'] = -1
# np.mean(data['SEP14_Bal'])
data.ix[pd.isnull(data['salary_feb15']), 'salary_feb15'] = -1


data.ix[pd.isnull(data['RATIO_EOP_BAL_FEB15']), 'RATIO_EOP_BAL_FEB15'] = np.mean(
    data['RATIO_EOP_BAL_FEB15'])

del data['ZIP_CODE']  # 20000
# del data['Branch_City'] # 2000
# data.drop(['salary_sep14', 'salary_oct14', 'salary_nov14', 'salary_dec14', 'salary_jan15', 'salary_feb15'], inplace=True, axis=1)
# data.drop(['SEP14_Bal', 'OCT14_Bal', 'NOV14_Bal', 'DEC14_Bal'], inplace=True, axis=1)
# data.drop(['SEP14_EOP', 'OCT14_EOP', 'NOV14_EOP', 'DEC14_EOP'], inplace=True, axis=1)
# del data['RATIO_EOP_BAL_FEB15']

train = data.ix[data['Base_tag'] == "D", :]
test = data.ix[data['Base_tag'] == "V", :]

train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)
print train.columns

print "train, test sets written to disk!"
