# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and Preprocess Data
Load the Iris dataset, separate features and target, split into training and testing sets, and standardize the features.

2.Initialize the SGD Classifier
Create an SGDClassifier model using the logistic loss (for classification).

3.Train the Model
Fit the classifier on the training data to learn the weights through stochastic gradient descent.

4.Predict the Species
Take user input, scale it, make a prediction using the trained model, and display the corresponding Iris species.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: JANANI R
RegisterNumber:  25018734
*/
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Iris dataset
iris = load_iris()
X = iris.data       # Features
y = iris.target     # Species (0,1,2)

# Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SGD Classifier model
model = SGDClassifier(loss="log_loss", max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Prediction
sepal_l = float(input("Enter Sepal Length: "))
sepal_w = float(input("Enter Sepal Width: "))
petal_l = float(input("Enter Petal Length: "))
petal_w = float(input("Enter Petal Width: "))

x_new = scaler.transform([[sepal_l, sepal_w, petal_l, petal_w]])

pred = model.predict(x_new)[0]

print("Predicted Species:", iris.target_names[pred])

```

## Output:
<img width="356" height="152" alt="image" src="https://github.com/user-attachments/assets/0b47da89-e309-4760-a0fa-1b11561e4e03" />




## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
