import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd 
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

df=pd.read_csv('Dataset/diabetes_binary_health_indicators_BRFSS2015.csv')
y= df['Diabetes_binary'] 
x= df.drop(['Diabetes_binary'], axis=1)

# x = df[['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
#         'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
#         'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
#         'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']].values
# y = df['Diabetes_binary'].values

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)  

st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)  

classifier= RandomForestClassifier(n_estimators= 10, criterion="entropy")  
classifier.fit(x_train, y_train)  

y_pred= classifier.predict(x_test) 

print ("confussion matrix is : \n")
print (confusion_matrix(y_test, y_pred))
print ("-----------------------")
ACC = accuracy_score(y_test, y_pred)
print ("Accuracy is : \n", ACC)
print ("-----------------------")
Prec = precision_score(y_test, y_pred)
print ("Precsion is : \n", Prec)
print ("-----------------------")
F1 = f1_score(y_test, y_pred)
print ("F1 score is : \n", F1)
print ("-----------------------")