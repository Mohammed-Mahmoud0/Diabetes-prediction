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

#-------------------------Data Collection-----------------------------------------

df=pd.read_csv("Dataset/diabetes_binary_health_indicators_BRFSS2015.csv")
df.head()
df.info()
df.shape
df.sample(5)
df.columns
df.rename(columns={'Diabetes_binary': 'Diabetes_yes_no'}, inplace=True)
df.head()
Diabetes=df['Diabetes_yes_no']
Diabetes.value_counts()
df.Diabetes_yes_no.unique()
# 0=>Has Not Diabetes
# 1=>Has Diabetes
df.HighBP.unique()
# 0=>No High
# 1=>High BP
df.HighChol.unique()
# 0=>No High Cholesterol
# 1=>High Cholesterol
df.HeartDiseaseorAttack.unique()
# 0=>No
# 1=>Yes
df.GenHlth.unique()
# 1=>Excellent
# 2=>Very Good
# 3=>Good
# 4=>Fair
# 5=>Poor
df.DiffWalk.unique()
# 0=>No
# 1=>Yes
df.Sex.unique()
# 0=>Female
# 1=>Male
df.Age.unique()
# 1=>18-24
# 2=>25-30
# 3=>31-35
# 4=>36-40
# 5=>41-46
# 6=>47-51
# 7=>52-56
# 8=>57-61
# 9=>62-66
# 10=>67-71
# 11=>72-76
# 12=>77-80
# 13=>over 80 
df.Education.unique()
# 1=>Never Attended School
# 2=>primary
# 3=>preparatory
# 4=>secondary
# 5=>university
# 6=>graduate
df.Income.unique()
# 1=>less than or equal 3000
# 2=>greater than 3000 and less than or equal 5000
# 3=>greater than 5000 and less than or equal 10000
# 4=>greater than 10000 and less than or equal 15000
# 5=>greater than 15000 and less than or equal 20000
# 6=>greater than 20000 and less than or equal 30000
# 7=>greater than 30000 and less than 45000
# 8=>45000 or More
# Data Cleaning
df.isna().sum()
# Visualize correlation
df.corr()
df['Diabetes'] = df['Diabetes_yes_no'].map({0:'No Diabetes', 1:'Diabetes'})
df['High_blood'] = df['HighBP'].map({0:'No High',1:'High BP'})
df['High_Cholesterol'] = df['HighChol'].map({0:'No High Cholesterol',1:'High Cholesterol'})
df['Heart_Attack'] = df['HeartDiseaseorAttack'].map({0:'No',1:'Yes'})
df['Genral_Health'] = df['GenHlth'].map({1:'Excellent',2:'Very Good',3:'Good',4:'Fair',5:'Poor'})
df['Difficulty_Walking'] = df['DiffWalk'].map({0:'No',1:'Yes'})
df['Sex_Part'] = df['Sex'].map({0:'Female',1:'Male'})
df['Age_level'] = df['Age'].map({1:'18-24',2:'25-30',3:'31-35',4:'36-40',5:'41-46',6:'47-51',7:'52-56',8:'57-61',9:'62-66',10:'67-71',11:'72-76',12:'77-80',13:'Over 80'})
df['Education_level'] = df['Education'].map({1:'Never Attended School', 2:'primary',3:'preparatory',4:'secondary',5:'university',6:'graduate'})
df['Income_level'] = df['Income'].map({1:'less than or equal 3000',2:'greater than 3000 and less than or equal 5000',3:'greater than 5000 and less than or equal 10000',4:'greater than 10000 and less than or equal 15000',5:'greater than 15000 and less than or equal 20000',6:'greater than 20000 and less than or equal 30000',7:'greater than 30000 and less than 45000',8:'45000 or More'})
df.head()
df.info()
diabetes_bp = df.groupby(['Diabetes', 'High_blood']).size().reset_index(name = 'Count')
print(diabetes_bp)
diabetes_bp = df.groupby(['Diabetes', 'High_Cholesterol']).size().reset_index(name = 'Count')
print(diabetes_bp)
diabetes_bp = df.groupby(['Diabetes', 'Heart_Attack']).size().reset_index(name = 'Count')
print(diabetes_bp)
diabetes_bp = df.groupby(['Diabetes', 'Genral_Health']).size().reset_index(name = 'Count')
print(diabetes_bp)
diabetes_bp = df.groupby(['Diabetes', 'Difficulty_Walking']).size().reset_index(name ='Count')
print(diabetes_bp)
diabetes_bp = df.groupby(['Diabetes','Sex_Part']).size().reset_index(name ='Count')
print(diabetes_bp)
diabetes_bp = df.groupby(['Diabetes', 'Age_level']).size().reset_index(name = 'Count')
print(diabetes_bp)
diabetes_bp = df.groupby(['Diabetes','Education_level']).size().reset_index(name ='Count')
print(diabetes_bp)
diabetes_bp = df.groupby(['Diabetes','Income_level']).size().reset_index(name ='Count')
print(diabetes_bp)
df.shape
x = df[['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
        'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education','Income']].values
y = df['Diabetes_yes_no'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=44, shuffle =True)

#-------------------------Logistic Regression--------------------------------------------

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

#-------------------------SVM----------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=44, shuffle =True)
classifier= SVC(kernel= 'rbf', max_iter=100,C=1.0,gamma='auto')
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