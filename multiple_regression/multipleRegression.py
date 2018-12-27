# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
#age, wage >50, education-num, sex = male 1
import pandas as pd
import numpy
import statsmodels.api as sm
import statsmodels.formula.api as smf

data = pd.read_csv('Data2.csv', names = ['age', 'workclass', 'fnlwgt', 'education', 'educYear', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours', 'nati-country', 'wage'])

data['age']= data['age'].convert_objects(convert_numeric= True)
data['wage']= data['wage'].convert_objects(convert_numeric= True)
data['educYear']= data['educYear'].convert_objects(convert_numeric= True)
data['sex']= data['sex'].convert_objects(convert_numeric= True)
data['hours']= data['hours'].convert_objects(convert_numeric= True)
#make sure that the categorical explanatory variables have one group code as zero
# in this case sex, female = 0; wage <= 50 coded 0
#Quantitative explanatory variables that do not have 0 as valid value
#need to be centered. This is the case for education. It starts in year 1.

#center
data['educYear_c'] = (data['educYear']-data['educYear'].mean())

print('OLS regression model for the association between age and hours per week worked')
reg1 = smf.ols('hours ~ age + wage + educYear_c + sex', data=data).fit()
print(reg1.summary())