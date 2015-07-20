import pandas as pd
import numpy as np

data = pd.read_csv("Customer_level_data_PL", sep="|")

#print np.sum(pd.isnull(data))

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

data.ix[pd.isnull(data['salary_sep14']), 'salary_sep14'] = -1#np.mean(data['SEP14_Bal'])
data.ix[pd.isnull(data['salary_oct14']), 'salary_oct14'] = -1#np.mean(data['SEP14_Bal'])
data.ix[pd.isnull(data['salary_nov14']), 'salary_nov14'] = -1#np.mean(data['SEP14_Bal'])
data.ix[pd.isnull(data['salary_dec14']), 'salary_dec14'] = -1#np.mean(data['SEP14_Bal'])
data.ix[pd.isnull(data['salary_jan15']), 'salary_jan15'] = -1#np.mean(data['SEP14_Bal'])
data.ix[pd.isnull(data['salary_feb15']), 'salary_feb15'] = -1#np.mean(data['SEP14_Bal'])


data.ix[pd.isnull(data['RATIO_EOP_BAL_FEB15']), 'RATIO_EOP_BAL_FEB15'] = np.mean(data['RATIO_EOP_BAL_FEB15'])

del data['ZIP_CODE'] #20000
#del data['Branch_City'] # 2000
#data.drop(['salary_sep14', 'salary_oct14', 'salary_nov14', 'salary_dec14', 'salary_jan15', 'salary_feb15'], inplace=True, axis=1) 
#data.drop(['SEP14_Bal', 'OCT14_Bal', 'NOV14_Bal', 'DEC14_Bal'], inplace=True, axis=1) 
#data.drop(['SEP14_EOP', 'OCT14_EOP', 'NOV14_EOP', 'DEC14_EOP'], inplace=True, axis=1) 
#del data['RATIO_EOP_BAL_FEB15']

train = data.ix[data['Base_tag'] == "D",:]
test = data.ix[data['Base_tag'] == "V",:]

train.to_csv("train.csv", index = False)
test.to_csv("test.csv", index = False)

print "train, test sets written to disk!"