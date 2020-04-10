# Importing the dataset using pandas library
import pandas as pd

dataset = pd.read_csv("Social_Network_Ads.csv")
x = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, 4].values

# Features Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x = sc.fit_transform(x)

loop = 10
SVM_acc_sum = KNN_acc_sum = 0
# Looping onto training & testing for having the average result of accuracy of model while enabling randomization
for i in range(0, loop):
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=None)

    # Building SVM Classification Model (Fitting SVC to the Training set)
    from sklearn.svm import SVC

    SVM_Classifier = SVC(kernel='linear')
    SVM_Classifier = SVM_Classifier.fit(x_train, y_train)

    # Predicting the Test set results
    SVM_y_pred = SVM_Classifier.predict(x_test)

    # Building KNN Classification Model
    from sklearn.neighbors import KNeighborsClassifier
    KNN_Classifier = KNeighborsClassifier(n_neighbors=7)
    KNN_Classifier = KNN_Classifier.fit(x_train, y_train)

    # Predicting the Test set results
    KNN_y_pred = KNN_Classifier.predict(x_test)

    # building the confusion matrix for calculating the accuracy of our Model
    from sklearn.metrics import confusion_matrix

    SVM_cm = confusion_matrix(SVM_y_pred, y_test)
    SVM_acc = (SVM_cm[0][0] + SVM_cm[1][1]) / SVM_cm.sum()

    KNN_cm = confusion_matrix(KNN_y_pred, y_test)
    KNN_acc = (KNN_cm[0][0] + KNN_cm[1][1]) / KNN_cm.sum()

    SVM_acc_sum += SVM_acc
    KNN_acc_sum += KNN_acc

# Displaying the average accuracy of SVM Classification Model
SVM_acc_avg = SVM_acc_sum / loop * 100
KNN_acc_avg = KNN_acc_sum / loop * 100
print('The average accuracy of prediction using:-\nSVM = ', '%.2f' % SVM_acc_avg, '%', '\nKNN = ', '%.2f' % KNN_acc_avg,
      '%')
