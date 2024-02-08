import pandas as pd
import numpy as np
import pickle

df=pd.read_csv("heart.csv")

X=df.iloc[:,:13]
y=df.iloc[:,[13]]
X = X.to_numpy()
y = y.to_numpy()
1
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5,random_state=42)

age=int(input("Enter Your Age:"))
Gender = int(input("Enter Your Gender(1-M/0-F):"))
cp=int(input("Enter Chest Pain Level:"))
trestpbs = int(input())
chols=int(input("Enter Cholestrol Level:"))
fbs = int(input())
restcg = int(input())
thalach = int(input())
exang = int(input())
oldpeak = int(input())
slope = int(input())
ca = int(input())
thal = int(input())

p = [age,Gender,cp,trestpbs,chols,fbs,restcg,thalach,exang,oldpeak,slope,ca,thal]

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


random_forest_model = RandomForestRegressor(n_estimators=13, random_state=42)
random_forest_model.fit(X_train, y_train)
predictions = random_forest_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)
new_input = np.array([[age,Gender,cp,trestpbs,chols,fbs,restcg,thalach,exang,oldpeak,slope,ca,thal]])
predicted_output = random_forest_model.predict(new_input)
print("Predicted Output:", predicted_output)


pickle.dump(random_forest_model, open('premodel.pkl', 'wb'))
