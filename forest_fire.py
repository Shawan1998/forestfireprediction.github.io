import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("Forest_fire.csv")
x=data.iloc[:,1:-1].values
y=data.iloc[:,-1:].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

log_reg=LogisticRegression()

log_reg.fit(x_train,y_train)

import pickle
with open('model.pkl','wb') as model_pkl:
    pickle.dump(log_reg,model_pkl)
