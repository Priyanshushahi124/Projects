#importing libraries
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#importing data
data=pd.read_csv('Crop_recommendation.csv')


#split features and labels
x=data.iloc[:,:-1]
y=data.iloc[:,-1]

# training the model
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model=RandomForestClassifier()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print('Accuracy:', accuracy)

# creating an executable file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
