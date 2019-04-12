###In this program I will be using the machine learning library 'scikit-learn'.
###The program predicts the gender when there is an input of the height, weight and shoe size given. It does the prediction on the basis of previous data.

#import the tree s
from sklearn import tree

#[height, weight, shoe size]
X = [[181, 80, 44], [117, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'female', 'female', 'female', 'male', 'female', 'male', 'male', 'female', 'male']

#set up tree variable
clfr = tree.DecisionTreeClassifier()

#now that we have out tree variable,  let us train this variable on our dataset X and Y

#In a nutshell: fitting is equal to training. Then, after it is trained, the model can be used to make predictions, usually with a .predict() method call.

clfr = clfr.fit(X, Y)

prediction = clfr.predict([[100, 70, 43]])

print(prediction)

##Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.
