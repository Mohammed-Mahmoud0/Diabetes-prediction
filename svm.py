import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

df=pd.read_csv('Dataset/diabetes_binary_health_indicators_BRFSS2015.csv')
# x = df[['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
#         'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
#         'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
#         'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education','Income']].values
# y = df['Diabetes_binary'].values

y= df['Diabetes_binary'] 
x= df.drop(['Diabetes_binary'], axis=1)

from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(sampling_strategy=1)
x_res, y_res = rus.fit_resample(x, y)

x_train, x_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.33, random_state=1)
classifier= SVC(kernel= 'rbf', max_iter=100,C=1.0,gamma=1)
classifier.fit(x_train, y_train)
y_predict = classifier.predict(x_test)

CM = confusion_matrix(y_test, y_predict)
print('Confusion Matrix is : \n', CM)
AccScore = accuracy_score(y_test, y_predict)
print('Accuracy Score is : ', AccScore)
F1Score = f1_score(y_test, y_predict, average='micro') 
print('F1 Score is : ', F1Score)
PrecisionScore = precision_score(y_test, y_predict, average='micro')
print('Precision Score is : ', PrecisionScore)