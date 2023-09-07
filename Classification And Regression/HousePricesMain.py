import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Dense , Dropout
from tensorflow.python.keras.models import Sequential
from keras import utils as np_utils
from keras.utils import np_utils
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from keras.optimizers import Adam


df = pd.read_csv(r'C:\HousePrice\housePrice.csv' , names=['Area', 'Room', 'Parking', 'Warehouse', 'Elevator' , 'Address' , 'Price'], skiprows=1)

#print(df)
#print(df.isna().sum())

number_of_rows_before_deletion = df.shape[0]
df = df.dropna(how ='any')
number_of_rows_after_deletion = df.shape[0]
#print(df.isna().sum())


df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
quartiles = df['Price'].quantile([0, 0.25, 0.5, 0.75, 1])
labels = ['cheap', 'underMean', 'upperMean', 'expensive']
df['priceLevel'] = pd.qcut(df['Price'], q=4, labels=labels)

#print("Quartiles of Price column:\n", df['Price'].describe())
#print("\nUpdated dataframe:\n", df)


selected_str_cols = ['Parking', 'Warehouse', 'Elevator' , 'Address']

le_Parking = LabelEncoder()
le_Warehouse = LabelEncoder()
le_Elevator = LabelEncoder()
le_Address = LabelEncoder()

df['Parking'] = le_Parking.fit_transform(df['Parking'])
df['Warehouse'] = le_Warehouse.fit_transform(df['Warehouse'])
df['Elevator'] = le_Elevator.fit_transform(df['Elevator'])
df['Address'] = le_Address.fit_transform(df['Address'])

#print(df[selected_str_cols])
#print("\nUpdated dataframe For Label Encoding:\n", df)


scaler = StandardScaler()
scaler.fit(df.get(['Area', 'Room', 'Parking', 'Warehouse', 'Elevator' , 'Address' , 'Price']))
scaled_data = scaler.transform(df.get(['Area', 'Room', 'Parking', 'Warehouse', 'Elevator' , 'Address' , 'Price']))
scaler.fit(scaled_data) 
df['Area'] = scaled_data[:, 0]
df['Room'] = scaled_data[:, 1]
df['Parking'] = scaled_data[:, 2]
df['Warehouse'] = scaled_data[:, 3]
df['Elevator'] = scaled_data[:, 4]
df['Address'] = scaled_data[:, 5]
df['Price'] = scaled_data[:, 6]
#print(df)


#choosing pricelevel as our target(Specifically for Classifcation and the neural network part)
X = df.drop(['priceLevel', 'Price'], axis=1)
y = df['priceLevel']

"""
if you want to perform regression replace the code above with this:
X = df.drop(['priceLevel', 'Price'], axis=1)
y = df['Price']

"""

# Encode the labels as integers using LabelEncoder(Comment This Part If You Want Yo Perform Regression)
le = LabelEncoder()
y = le.fit_transform(y)

# Perform one-hot encoding on the integer-encoded labels(If you want to perform classification make sure to remove this line,this is made for the neural network part)
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#print("Training set shape:", X_train.shape)
#print("Testing set shape:", X_test.shape)




#Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
train_r2, test_r2 = r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)
train_mse, test_mse = mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)

#print(f'Training set:\nR-squared: {train_r2:.3f}\nMSE: {train_mse:.3f}')
#print(f'Testing set:\nR-squared: {test_r2:.3f}\nMSE: {test_mse:.3f}')


#Polynominal Regression With 2 Degrees
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)
y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

#print(f'Training set R-squared: {train_r2:.3f}, MSE: {train_mse:.3f}\n')
#print(f'Testing set R-squared: {test_r2:.3f}, MSE: {test_mse:.3f}\n')


#Polynomial Regression With 3 Degrees
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
model = LinearRegression()
model.fit(X_train_poly, y_train)
y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

#print(f'Training set R-squared: {train_r2:.3f}, MSE: {train_mse:.3f}\n')
#print(f'Testing set R-squared: {test_r2:.3f}, MSE: {test_mse:.3f}\n')





#Decision Tree
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
training_accuracy = accuracy_score(y_train, clf.predict(X_train))
testing_accuracy = accuracy_score(y_test, y_pred)
print("Training Accuracy:", training_accuracy)
print("Testing Accuracy:", testing_accuracy)


#Random Forest
clf = RandomForestClassifier(criterion='entropy')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
training_accuracy = accuracy_score(y_train, clf.predict(X_train))
testing_accuracy = accuracy_score(y_test, y_pred)
print("Training Accuracy:", training_accuracy)
print("Testing Accuracy:", testing_accuracy)


#KNN algorithm for k=3, 5, and 7
for k in [3, 5, 7]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    # Evaluate the accuracy of the model on both training and testing sets
    train_acc = knn.score(X_train, y_train)
    test_acc = knn.score(X_test, y_test)
    print(f"k={k}, Training Accuracy: {train_acc:.3f}, Testing Accuracy: {test_acc:.3f}")


#SVM algorithm in linear mode
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
train_acc_linear = svm_linear.score(X_train, y_train)
test_acc_linear = svm_linear.score(X_test, y_test)
print(f"Linear SVM: Training Accuracy: {train_acc_linear:.3f}, Testing Accuracy: {test_acc_linear:.3f}")


#SVM algorithm in non-linear mode
svm_nonlinear = SVC(kernel='rbf')
svm_nonlinear.fit(X_train, y_train)
train_acc_nonlinear = svm_nonlinear.score(X_train, y_train)
test_acc_nonlinear = svm_nonlinear.score(X_test, y_test)
print(f"Non-Linear SVM: Training Accuracy: {train_acc_nonlinear:.3f}, Testing Accuracy: {test_acc_nonlinear:.3f}")



# Build the model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
opt = Adam(learning_rate=0.001)
history=model.fit(X_train, y_train, epochs=12000, batch_size=256, validation_data=(X_test, y_test))

# Evaluate the model
train_acc=model.evaluate(X_train, y_train)
print('Training accuracy:', train_acc[1])
test_acc=model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc[1])

# Predict the test data and print the confusion matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print('Confusion matrix:\n', cm)