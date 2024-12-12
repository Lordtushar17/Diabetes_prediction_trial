import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import joblib  # Import joblib for saving the model

# Loading the data
diabetes_dataset = pd.read_csv('D:\\PICT Techfiesta\\diabetes_prediction\\diabetes.csv')

rows, columns = diabetes_dataset.shape[0], diabetes_dataset.shape[1]

# Separating the data
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

scaler = StandardScaler()
scaler.fit(X)

standardized_data = scaler.transform(X)

X = standardized_data
Y = diabetes_dataset['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
classifier = svm.SVC(kernel='linear')

# Training the support vector machine classifier
classifier.fit(X_train, Y_train)
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print("Accuracy score of the training data is: ", training_data_accuracy)

# Save the trained model to a .pkl file
model_filename = 'diabetes_prediction_model.pkl'
joblib.dump(classifier, model_filename)
print(f"Model saved to {model_filename}")

# Example input prediction
input_data = (0, 132, 78, 0, 0, 32.4, 0.393, 21)

# Change the data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
std_data = scaler.transform(input_data_reshaped)

print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if prediction[0] == 0:
    print("Congrats, you do not have diabetes")
else:
    print("You have diabetes")
