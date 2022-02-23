# Multiple Linear Regression

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



#buidling the optimal model using backward elimination
import statsmodels.api as sm
#Add columns of 1
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
print(X)
#step 1: select a SL 
sl = 0.05

# step 2: fit the full model with all possible predictors
X_opt = X[:,[0,1,2,3,4,5]]
  #X_opt array has a dtype of object and for that without casting it causes an error.
X_opt = np.array(X_opt, dtype=float) 
regressor_OLS = sm.OLS( y, X_opt).fit()


# step 3 : consider the predictor with the highest p-value. if P> SL, go to step 4 ,otherwise go to fin

regressor_OLS.summary()
print(regressor_OLS.summary())
df = pd.read_html(regressor_OLS.summary().tables[1].as_html(),header=0,index_col=0)[0]
c=df['P>|t|'].values[0]
print(c)



#size = np.shape(X_opt)[1]


for i in X_opt:
 
  m = 0.0
  index = 0 
  X_opt = np.array(X_opt, dtype=float) 
  regressor_OLS = sm.OLS( y, X_opt).fit()  
  regressor_OLS.summary()
  print(regressor_OLS.summary())
  df = pd.read_html(regressor_OLS.summary().tables[1].as_html(),header=0,index_col=0)[0]
  for j in range(np.shape(X_opt)[1]):
   
      a=df['P>|t|'].values[j]
      if a>=m:
          m=a
          index = j

  if abs(df['t'].values[index]) < m or df['P>|t|'].values[index] > 0.05:
      X_opt = np.delete(X_opt, index, axis=1)
     
  else:
      break
  
print(np.shape(X_opt)[1])
print(regressor_OLS.summary())



