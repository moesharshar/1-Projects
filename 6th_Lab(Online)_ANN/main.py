# Artificial Neural Network(ANN)
# Part_1 - Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding and spliting categorical data

# Method_1:-(scikit-learn=0.19.2)
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder_x_1 = LabelEncoder()
# x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])
# labelencoder_x_2 = LabelEncoder()
# x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])
# onehotencoder = OneHotEncoder(categorical_features = [1])
# x = onehotencoder.fit_transform(x).toarray()
# x = x[:, 1:]

# Method_2
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# # Encoding countries by converting it's string values into numerical such as (0 or 1 or 2)
# encoder = LabelEncoder()
# x[:, 0] = encoder.fit_transform(x[:, 0])
# # separating each country to be in a stand alone column, so country that is represented by 1 or 2 don't have higher
# #   effect upon country of 0
# # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
# # remainder='passthrough' to Leave the rest of the columns untouched
# encoder = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [0])], remainder='passthrough')
# x = encoder.fit_transform(x)

# Method_3
# We can apply both transformations (from text categories to integer categories,
# then from integer categories to one-hot vectors) in one shot using the LabelBinarizer class:-
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
encoder = LabelBinarizer()
geography_cat_1hot = encoder.fit_transform(x[:, 1])
encoder = LabelEncoder()
x[:, 2] = encoder.fit_transform(x[:, 2])
x = np.column_stack((x[:, 0], geography_cat_1hot[:, 1:], x[:, 2:]))
# Note that we can neglect one of the 3 encoded columns of Country, 
# so not to get into dumy variables trap

# End of encoding categorical data


# Splitting the dataset into the Training set and Testing set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                random_state=42, shuffle=True)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Part_2 Building ANN Model

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))

# Adding the second hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(x_train, y_train, batch_size=10, nb_epoch=100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Testing set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)