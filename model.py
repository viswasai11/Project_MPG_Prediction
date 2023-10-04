import pandas as pd
import pickle

mpg_df=pd.read_csv('Auto MPG Reg.csv')
mpg_df.horsepower=mpg_df.horsepower.fillna(
    mpg_df.horsepower.median())

y=mpg_df.mpg
X=mpg_df[['cylinders','displacement','horsepower','weight','acceleration','modelyear','origin']]

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X,y)
reg.score(X,y)

pickle.dump(reg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))