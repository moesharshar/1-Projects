import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')

x = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, 4].values

# feature scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x = sc.fit_transform(x)

rslt1 = rslt2 = 0

for i in range(5):
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=55)

    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression()
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    from sklearn.tree import DecisionTreeClassifier

    classifier2 = DecisionTreeClassifier()
    classifier2.fit(x_train, y_train)
    y2_pred = classifier2.predict(x_test)

    # Testing Model
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred)
    x1 = cm[0][0]
    x2 = cm[0][1]
    x3 = cm[1][0]
    x4 = cm[1][1]

    cm2 = confusion_matrix(y_test, y2_pred)
    x11 = cm2[0][0]
    x22 = cm2[0][1]
    x33 = cm2[1][0]
    x44 = cm2[1][1]
    # End of testing

    rslt1 += (x1 + x4) / (x1 + x2 + x3 + x4) * 100
    rslt2 += (x11 + x44) / (x11 + x22 + x33 + x44) * 100

# Printing result of using Logistic_Regression
print("Accuracy using Logistic_Regression = " + str(rslt1 / 5) + '%')

# Printing result of using Decision_Tree
print("Accuracy using Decision_Tree = " + str(rslt2 / 5) + '%')
