# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 09:27:29 2020

@author: janak
"""
"""Let's begin """

# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Unfallorte_Combined.csv')
#Take only those variables which can have effect to accident, District(1),Month(5),Weekday(6),Light Condition(15) and Road Condition(18)
X = dataset.iloc[:, [1,5,6,18,19]].values
y = dataset.iloc[:, 21].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#For District
labelencoder_X_0 = LabelEncoder()
X[:, 0] = labelencoder_X_0.fit_transform(X[:, 0])
#7-6, 6-5, 1-0

#For Month
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
#12-11, 11-10, 1-0

#For Weekday
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#7-6, 6-5, 1-0

#Category for gender
#labelencoder_X_2 = LabelEncoder()
#X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#Create dummy variables for country only, not needed for gender
#onehotencoder = OneHotEncoder(categorical_features = [0])
#X = onehotencoder.fit_transform(X).toarray()
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
#X = np.array(ct.fit_transform(X))
X = np.array(ct.fit_transform(X), dtype=np.float)
X = X[:, 0:]
#Col 0-6

#Dummy variables for Month
ct = ColumnTransformer([('encoder', OneHotEncoder(), [7])], remainder='passthrough')
#X = np.array(ct.fit_transform(X))
X = np.array(ct.fit_transform(X), dtype=np.float)
X = X[:, 0:]
#From col 7-18

#Dummy variables for Weekday
ct = ColumnTransformer([('encoder', OneHotEncoder(), [19])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
X = np.array(ct.fit_transform(X), dtype=np.float)
X = X[:, 0:]
#From col 19-26

# Encoding the Dependent Variable
#labelencoder_y = LabelEncoder()
#y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Part 2 - Now let's make the ANN
#Importing keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

#Initialising the ANN 
classifier = Sequential()

#Adding the first hidden layer as output_dim and input layer as input_dim
#output_dm=(input_dm+output_node)/2
classifier.add(Dense(output_dim = 15, init = 'uniform', activation = 'relu', input_dim = 29))
#Do this during improving the ANN
#classifier.add(Dropout(p = 0.1)) #To resolve over fitting $never go beyond 0.5
#Adding second hidden layer
classifier.add(Dense(output_dim = 15, init = 'uniform', activation = 'relu'))
#Do this during improving the ANN
#classifier.add(Dropout(p = 0.1))
#Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#Compiling the ANN
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size  = 15, nb_epoch = 130)
#accuracy achieved is 0.8941
#Part 3 - Making predictions and evaluating the model
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.2) #Predicts the probability of death greater than 20%
#y_pred_inv = sc.inverse_transform(y_pred)

#Plot and compare the trend with original
# import matplotlib.pyplot as plt
# plt.plot(y_test, color = 'red', label = 'Real Accident Number')
# plt.plot(y_pred, color = 'blue', label = 'Predicted Accident Number')
# plt.title('Accident Prediction')
# plt.xlabel('Observation')
# plt.ylabel('Score of Accidents')
# plt.legend()
# plt.show()

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#Accuracy calculation
overall_accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
overall_accuracy
#0.8941 and 0.8802980903586399, when nb_epoch = 130 and batch_size = 10
#0.8937 and 0.8789007918025151, when nb_epoch = 130 and batch_size = 5
#0.8922 and 0.8721471821145785, when nb_epoch = 100 and batch_size = 5
#0.8940 and 0.8854215183977643, when nb_epoch = 100 and batch_size = 15

#0.8941 and 0.8919422449930136, when nb_epoch = 130 and batch_size = 15  

#0.8937 and 0.8772706101537029, when nb_epoch = 125 and batch_size = 15
#0.8940 and 0.8318584070796460, when nb_epoch = 130 and batch_size = 20
#0.8940 and 0.8432696786213321, when nb_epoch = 100 and batch_size = 20
#0.8939 and 0.884257102934327, when nb_epoch = 100 and batch_size = 15
#0.8929 and 0.8586399627387051, when nb_epoch = 100 and batch_size = 10
print(' Part2 Done...')
#Analytical Questions
#Q1. Will there be accidental death in district 4 on weekend of July at morning when the road 
#is dry?
"""Solution:
    District - 4 ---[1,0,0,0,0,1,0]
    Month - July - 7 --- [1,0,0,0,0,0,0,0,0,1,0,0]
    Weekday - Weekend - [0,0,0,0,0,0,0,0]
    Light Condition - Morning - 1 - [1]
    Road Condition - Dry - 0 - [0]
    """
new_data = [1.0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0]
new_pred = classifier.predict(sc.transform(np.array([new_data])))
new_pred = (new_pred>0.2)
print(f'There is tend to death at the said condition {new_pred}')
#There is tend to death at the said condition [[False]]

#Q2. What is the probability of death in district 3 on monday morning of 
#January when the road is smooth?
"""Solution:
    District - 3 ---[1,0,1,0,0,0,0]
    Month - Jan - 1 --- [0,0,1,0,0,0,0,0,0,0,0,0]
    Weekday - Monday - [0,0,0,0,0,0,1,0]
    Light Condition - Morning - 1 - [1]
    Road Condition - Smooth - 0 - [2]
    """
q2_data = [1.0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,2]
q2_pred = classifier.predict(sc.transform(np.array([q2_data])))
#q2_pred = (q2_pred>0.2)
print(f'The probability of death at the said condition is: {q2_pred*100}')
#7.39%

#Q3. What is the probability of death in district 1 during October wednesday 
#when the conditions are adverse?
"""Solution:
    By adverse: Light condition is daylight and road is wet
    District - 1 ---[0,1,0,0,0,0,0]
    Month - Oct - 10 --- [0,0,0,0,0,0,0,0,1,0,0,0]
    Weekday - Wednesday - [0,0,0,0,0,0,0,1]
    Light Condition - Daylight - 0 - [0]
    Road Condition - Wet - 1 - [1]
    """
q3_data = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1]
q3_pred = classifier.predict(sc.transform(np.array([q3_data])))
#q2_pred = (q2_pred>0.2)
print(f'The probability of death at the said condition is: {q3_pred*100}')
#13.1%

#Q4. What is the probability of death in district 7 during April Fiday 
#night when road is smooth?
"""Solution:
    By adverse: Light condition is daylight and road is wet
    District - 7 ---[1,0,0,0,0,1,0]
    Month - Apr - 4 --- [0,0,0,0,0,1,0,0,0,0,0,0]
    Weekday - Friday - [0,0,0,1,0,0,0,0]
    Light Condition - Darkness - 2 - [2]
    Road Condition - Smooth - 2 - [2]
    """
q4_data = [1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,2,2]
q4_pred = classifier.predict(sc.transform(np.array([q4_data])))
#q2_pred = (q2_pred>0.2)
print(f'The probability of death at the said condition is: {q4_pred*100}')
#3.86%

#Part 4 - Evaluating, improving and tuning the model
#Second Part  is not required as will be incorporated by Part 4
#Evaluating the model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()    
    classifier.add(Dense(output_dim = 15, init = 'uniform', activation = 'relu', input_dim = 29))
    classifier.add(Dense(output_dim = 15, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 15, nb_epoch = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 15, n_jobs = -1)    
mean = accuracies.mean()
variance = accuracies.std()

#Improving the ANN
#ANN is improved using dropout layer as above. It drops the number of nodes of hidden layers

#Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()    
    classifier.add(Dense(output_dim = 15, init = 'uniform', activation = 'relu', input_dim = 29))
    classifier.add(Dense(output_dim = 15, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size' : [25,15, 32],
              'epochs' : [150, 130],
              'optimizer' : ['adam', 'rmsprop']}
#Pass these values to grid search
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = None)
#Now fit this values
grid_search = grid_search.fit(X_train, y_train)
#find the best parameters and accuracy
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
best_accuracy
best_parameters
grid_search

#Analytical Questions
#Q1. Will there be accidental death in district 4 on weekend of July at morning when the road is dry?
"""Solution:
    District - 4 ---[1,0,0,0,0,1,0]
    Month - July - 7 --- [1,0,0,0,0,0,0,0,0,1,0,0]
    Weekday - Weekend - [0,0,0,0,0,0,0,0]
    Light Condition - Morning - 1 - [1]
    Road Condition - Dry - 0 - [0]
    """
q1_data = [1.0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0]
q1_pred_tuned = grid_search.best_estimator_.predict(sc.transform(np.array([q1_data])))
q1_pred_tuned = (q1_pred_tuned>0.2)
print(f'There is tend to death at the said condition {q1_pred_tuned}')

#Q2. What is the probability of death in district 3 on monday morning of January when the road is smooth?
"""Solution:
    District - 3 ---[1,0,1,0,0,0,0]
    Month - Jan - 1 --- [0,0,1,0,0,0,0,0,0,0,0,0]
    Weekday - Monday - [0,0,0,0,0,0,1,0]
    Light Condition - Morning - 1 - [1]
    Road Condition - Smooth - 0 - [2]
    """
q2_data_tuned = [1.0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,2]
q2_pred_tuned = grid_search.best_estimator_.predict(sc.transform(np.array([q2_data_tuned])))
#q2_pred = (q2_pred>0.2)
print(f'The probability of death at the said condition is: {q2_pred_tuned*100}')

#Q3. What is the probability of death in district 1 during October wednesday when the conditions are adverse?
"""Solution:
    By adverse: Light condition is daylight and road is wet
    District - 1 ---[0,1,0,0,0,0,0]
    Month - Oct - 10 --- [0,0,0,0,0,0,0,0,1,0,0,0]
    Weekday - Wednesday - [0,0,0,0,0,0,0,1]
    Light Condition - Daylight - 0 - [0]
    Road Condition - Wet - 1 - [1]
    """
q3_data_tuned = [0.0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1]
q3_pred_tuned = grid_search.predict(sc.transform(np.array([q3_data_tuned])))
#q2_pred = (q2_pred>0.2)
print(f'The probability of death at the said condition is: {q3_pred_tuned*100}')

#Q4. What is the probability of death in district 7 during April Fiday night when road is smooth?
"""Solution:
    By adverse: Light condition is daylight and road is wet
    District - 7 ---[1,0,0,0,0,1,0]
    Month - Apr - 4 --- [0,0,0,0,0,1,0,0,0,0,0,0]
    Weekday - Friday - [0,0,0,1,0,0,0,0]
    Light Condition - Darkness - 2 - [2]
    Road Condition - Smooth - 2 - [2]
    """
q4_data_tuned = [1.0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,2,2]
q4_pred_tuned = grid_search.predict(sc.transform(np.array([q4_data_tuned])))
#q2_pred = (q2_pred>0.2)
print(f'The probability of death at the said condition is: {q4_pred_tuned*100}')
grid_search.best_estimator_
