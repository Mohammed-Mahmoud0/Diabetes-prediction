from cProfile import label
from turtle import color
from matplotlib.pyplot import cla
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ----------------------------Logistic Regression------------------------------------

df = pd.read_csv('Dataset/diabetes_binary_health_indicators_BRFSS2015.csv')


x = df[['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
        'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']].values
y = df['Diabetes_binary'].values

class logistic_regression:
    def __init__(self, l_rate = 0.001, iterations = 100):
        self.l_rate = l_rate
        self.iterations = iterations
    
    def fit(self, x, y):
        self.losses = []
        self.theta = np.zeros((1 + x.shape[1]))
        n = x.shape[0]
        for i in range (self.iterations):
            y_pred = self.theta[0] + np.dot(x, self.theta[1:])
            z = y_pred
            g_z = 1 / (1 + np.e**(-z))
            cost = (-y * np.log(g_z) - (1 - y) * np.log(1 - g_z)) / n
            self.losses.append(cost)
            d_theta1 = (1 / n) * np.dot(x.T, (g_z - y))
            d_theta0 = (1 / n) * np.sum(g_z - y)
            self.theta[1:] = self.theta[1:] - self.l_rate * d_theta1
            self.theta[0] = self.theta[0] - self.l_rate * d_theta0
        return self
    
    def predict(self, x):
        y_pred = self.theta[0] + np.dot(x, self.theta[1:])
        z = y_pred
        g_z = 1 / (1 + np.e**(-z))
        return [1 if i > 0.5 else 0 for i in g_z]

def scale(x):
    x_scaled = x - np.mean(x, axis=0)
    x_scaled = x_scaled / np.std(x_scaled, axis=0)
    return x_scaled

x_sd = scale(x)
model = logistic_regression()
model.fit(x_sd, y)

y_pred = model.predict(x_sd)

cm = confusion_matrix(y_pred, y, labels=[1,0])
print ("confussion matrix is : \n", cm)
TP = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[1][1]

ACC = (TP + TN) / (TP + TN + FP + FN)
print ("Accuracy is : \n", ACC)
print ("-----------------------")
Rec = TP / (TP + FN)
print ("Recall is : \n", Rec)
print ("-----------------------")
Prec = TP / (TP + FP)
print ("Precsion is : \n", Prec)
print ("-----------------------")
F1 = 2 * ((Prec * Rec) / (Prec + Rec))
print ("F1 score is : \n", F1)
print ("-----------------------")

