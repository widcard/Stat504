import numpy as np  
import pandas as pd  
import statsmodels.api as sm 
import statsmodels.api as sm
from statsmodels.formula.api import ols 
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import shapiro


data = pd.read_csv(r'Real estate.csv')


def normality_test(data):
  stat, p_value = shapiro(data)    #Shapiro-Wilk test
  alpha = 0.05

  if p_value > alpha:
    print('Normality test: Gaussian')  #fail in reject H0 (null hypothesis H0: follow normal distribution)
  else:
    print('Normality test: Non Gaussian') #reject H0 (alternative hypothesis H1: does not follow normal distribution)


#data plot (2D and 3D)
def data_scatter(data_, pca, n_dim, ax, color):
  if(n_dim == 2):
    plt.scatter(data_[:,0], data_[:,1], color=color)    #plot 2D
  else:
    ax.scatter(data_[:,0], data_[:,1], data_[:,2], color=color)  #plot 3D
    ax.set_zlabel('Dimension 3 (%.f %%)' % (round(pca.explained_variance_ratio_.cumsum()[2], 2)*100)) #third principal component
  plt.xlabel('Dimension 1 (%.f %%)' % (round(pca.explained_variance_ratio_.cumsum()[0], 2)*100)) #first principal component
  plt.ylabel('Dimension 2 (%.f %%)' % (round(pca.explained_variance_ratio_.cumsum()[1], 2)*100)) #second principal component
  
# show the plot
plt.figure(figsize=(13,5))

for feat, grd in zip(data, range(231,237)):
  plt.subplot(grd)
  sns.boxplot(y=data[feat], color='grey')
  plt.ylabel('Value')
  plt.title('Boxplot\n%s'%feat)
plt.tight_layout()
plt.show()
