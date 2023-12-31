from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import pandas as pd

df=pd.read_csv('Dataset/diabetes_binary_health_indicators_BRFSS2015.csv')

x = df[['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
        'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']].values
y = df['Diabetes_binary'].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

y_pred = log_reg.predict(x_test)

print ("confussion matrix is : \n")
print (confusion_matrix(y_test, y_pred))    
print ("-----------------------")
ACC = accuracy_score(y_test, y_pred)
print ("Accuracy is : \n", ACC * 100)
print ("-----------------------")
Prec = precision_score(y_test, y_pred)
print ("Precsion is : \n", Prec * 100)
print ("-----------------------")
F1 = f1_score(y_test, y_pred)
print ("F1 score is : \n", F1 * 100)
print ("-----------------------")
