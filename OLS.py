import numpy as np  
import pandas as pd  
import statsmodels.api as sm 
import statsmodels.api as sm
from statsmodels.formula.api import ols 
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
from out import*
data = pd.read_csv(r'Real estate.csv')



y=data['Y house price of unit area']
x1=data['X1 transaction date']
x2=data['X2 house age']
x3=data['X3 distance to the nearest MRT station']
x4=data['X4 number of convenience stores']
x5=data['X5 latitude']
x6=data['X6 longitude']
x=np.column_stack((x1,x2,x3,x4,x5,x6))

x_n = sm.add_constant(x) 
model = sm.OLS(y, x) #modeling  
results = model.fit(disp=0) #fit model  


print(results.summary())

fig = plt.figure(figsize=(12,8))

#produce regression plots
fig = sm.graphics.plot_regress_exog(results, 'x1', fig=fig)
