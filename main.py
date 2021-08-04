#IMPORTING LIBRARIES
import pandas as pd
import numpy as np


#READING DATASET
dataset=pd.read_csv("salary.csv")

#read data of dependent variable from dataset
X=dataset.iloc[:,1:2].values
#read the data of independent variable from dataset
Y=dataset.iloc[:,-1].values
#feature scaling putting the data into a range for better visualzation
from sklearn.preprocessing import StandardScaler 
sc=StandardScaler()
X=sc.fit_transform(X)
sc_y=StandardScaler()
Y=sc_y.fit_transform(np.reshape(Y,(10,1)))

# Y=Y.reshape(len(Y),1)

#Performing Linear Regression

#Performing SVR model
from sklearn.svm import SVR
svr=SVR(kernel="rbf")
svr.fit(X,Y)

#visualizing data
import matplotlib.pyplot  as plt
plt.xlabel("EXPERIENCE")
plt.ylabel("SALARY")
plt.scatter(X,Y,color='green')
plt.plot(X,svr.predict(Y),color='red')
plt.show()

#visualizing data in highier density
# x_grid=np.arange(min(X),max(X),0.1)
# x_grid=x_grid.reshape(len(x_grid),1)
# plt.xlabel("EXPERIENCE")
# plt.ylabel("SALARY")
# plt.scatter(X,Y,color='green')
# plt.plot(x_grid,svr.predict(Y),color='red')
# plt.show()





