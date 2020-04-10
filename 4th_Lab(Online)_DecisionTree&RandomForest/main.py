# Importing the libraries
import pandas as pd

# Importing the dataset using pandas library
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, 4].values

DT_Sum = RF_Sum = 0

# Looping onto training & testing for having the average result of accuracy of model while enabling randomization
for i in range(0, 10):
    # Splitting the dataset into a Training set and Test set
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Fitting Decision Tree Classification to the Training set
    from sklearn.tree import DecisionTreeClassifier
    DT_Classifier = DecisionTreeClassifier()
    DT_Classifier.fit(x_train, y_train)

    # Predicting the Test set results
    y_DT_pred = DT_Classifier.predict(x_test)

    # Fitting Decision Tree Classification to the Training set
    from sklearn.ensemble import RandomForestClassifier
    RF_Classifier = RandomForestClassifier(n_estimators=100)
    RF_Classifier.fit(x_train, y_train)

    # Predicting the Test set results
    y_RF_pred = RF_Classifier.predict(x_test)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm1 = confusion_matrix(y_DT_pred, y_test)
    cm2 = confusion_matrix(y_RF_pred, y_test)

    DT_Accuracy = (cm1[0][0] + cm1[1][1])/cm1.sum()  # (cm1[0][0] + cm1[0][1] + cm1[1][0] + cm1[1][1])
    RF_Accuracy = (cm2[0][0] + cm2[1][1]) / (cm2[0][0] + cm2[0][1] + cm2[1][0] + cm2[1][1])

    # Displaying both Decision Tree accuracy & Random Forest accuracy
    print(i + 1, "-The accuracy if Decission Tree is ", '%.2f' % (DT_Accuracy * 100), "%")
    print("   The accuracy if Random Forest is ", '%.2f' % (RF_Accuracy * 100), "%")

    DT_Sum += DT_Accuracy
    RF_Sum += RF_Accuracy

# Displaying the average accuracy for each of Decision Tree and Random Forest
print("\nThe Avg accuracy if Decission Tree is ", '%.2f' % (DT_Sum / 10 * 100), "%")
print("The Avg accuracy if Random Forest is ", '%.2f' % (RF_Sum / 10 * 100), "%")
