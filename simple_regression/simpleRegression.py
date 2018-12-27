# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import pandas as pd
import numpy
import statsmodels.api as sm
import statsmodels.formula.api as smf

data = pd.read_csv('Data2.csv', names = ['age', 'workclass', 'fnlwgt', 'education', 'educYear', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours', 'nati-country', 'wage'])
print(data.head(5))
print(data['hours'].dtypes)
data['age']= data['age'].convert_objects(convert_numeric= True)
data['hours']= data['hours'].convert_objects(convert_numeric= True)

print('OLS regression model for the association between age and hours per week worked')
reg1 = smf.ols('hours ~ age', data=data).fit()
print(reg1.summary())