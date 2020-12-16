# How to use Isolation Forest in Python

*Last update: 16 December 2020*

[Isolation Forest (IF)](https://en.wikipedia.org/wiki/Isolation_forest) can be used to detect outliers in a dataset. IF is a model that is trained on some data and can be predicted on new data. Thus, IF makes it possible to identify outliers in new data in a similar way as in an original training dataset. This can be helpful when outliers in new data need to be identified in order to ensure the accuracy of a predictive model.

**1. Basic Example (sklearn)**

Before I go into more detail, I show a brief example that highlights how Isolation Forest with sklearn works.

First load some packages (I will use them throughout this example):

```
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import datasets
from random import randrange
```

Next I generate some data for illustration. 
```
df = {'y': [1,2,3,4,5,6,7,8,9,10], 'x': [11,12,13,14,15,16,17,23454,19,20]}
df = pd.DataFrame(data=df)
```
Variable x in the data frame clearly is an extreme value when compared to the rest of the data. 

Next I fit an Isolation Forest to variable x.

```
x = df['x'].to_numpy().reshape(-1, 1)

clf = IsolationForest(random_state=2020, behaviour="new")
clf.fit(x)

pred = clf.predict(x)

df = pd.concat([df,pd.Series(pred)], axis=1)
```

The result is a data frame in which the last row shows the predictions of the Isolation Forest.

The prediction is 1 if "no outlier" has been detected in x and -1 in case there is an outlier. As you can see, outlier detection works well in this example.
```
    y      x  0
0   1     11  1
1   2     12  1
2   3     13  1
3   4     14  1
4   5     15  1
5   6     16  1
6   7     17  1
7   8  23454 -1
8   9     19  1
9  10     20  1
```

Next suppose there is "new data" for which outliers need to be predicted by the model.

```
df_new = {'y': [1,2,3,4,5,6,7,8,9,10], 'x': [11,12,13,14,15,16,17,18,19,20]}
df_new = pd.DataFrame(data=df_new)

x = df_new['x'].to_numpy().reshape(-1, 1)
pred_new = clf.predict(x)

df_new = pd.concat([df_new,pd.Series(pred_new)], axis=1)
print(df_new)
```
Variable x in the new data frame does not contain any outliers, so we would expect the Isolation Forest to predict the data accordingly. In fact the Isolation Forest in the simple example works very well. For the new data, no outliers are predicted.

```
    y   x  0
0   1  11  1
1   2  12  1
2   3  13  1
3   4  14  1
4   5  15  1
5   6  16  1
6   7  17  1
7   8  18  1
8   9  19  1
9  10  20  1
```

**2. Example with Iris Data and ExtraTreesClassifier**

Below I use [Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) with the Iris data and an [ExtraTreesClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html). This is an arbitrary choice. You can use Isolation Forest with any type of classifier.

First I load the Iris data and I estimate a baseline model.

```
# Load Iris data
iris = datasets.load_iris()
x = iris.data[:, :]  
y = iris.target

# Test train split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=123)

# Benchmark model
et = ExtraTreesClassifier(n_estimators=1000, max_features=2, n_jobs=5, bootstrap=True, random_state=123, warm_start=True)
et.fit(xtrain, ytrain)
ypred = et.predict(xtest)

# Scores
scores = accuracy_score(ytest, ypred)
print(scores)
cm = confusion_matrix(ytest, ypred)
print(cm)
```

The model produces a good fit on the test set with an accuracy of just under 0.96.

In the next step I add some outliers to the data to see how the model performs in this case:

```
# Iris to data frame
df = pd.DataFrame(x)
df['y'] = pd.Series(y)

# Add outliers to the data
for c in [0,1,2]:
    for x in range(0,5):
        df.loc[df.shape[0]] = [randrange(100,200),randrange(200,350),randrange(150,300),randrange(500,1000),c]

# Check new data with outliers
print(df.tail(20))

y = df['y'].to_numpy()
x = df.drop(['y'], axis=1).to_numpy()

# Test train split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=123)

# Model
et = ExtraTreesClassifier(n_estimators=1000, max_features=2, n_jobs=5, bootstrap=True, random_state=123, warm_start=True)
et.fit(xtrain, ytrain)
ypred = et.predict(xtest)

# Scores
scores = accuracy_score(ytest, ypred)
print(scores)
cm = confusion_matrix(ytest, ypred)
print(cm)
```

The last 20 rows of the data frame illustrate hor the outlier look like. They are equally distributed across the target's classes (y) and are relatively extreme.

```
         0      1      2      3  y
145    6.7    3.0    5.2    2.3  2
146    6.3    2.5    5.0    1.9  2
147    6.5    3.0    5.2    2.0  2
148    6.2    3.4    5.4    2.3  2
149    5.9    3.0    5.1    1.8  2
150  118.0  312.0  286.0  564.0  0
151  141.0  276.0  266.0  537.0  0
152  188.0  258.0  216.0  746.0  0
153  150.0  344.0  213.0  723.0  0
154  184.0  291.0  253.0  635.0  0
155  176.0  231.0  215.0  909.0  1
156  116.0  308.0  284.0  574.0  1
157  149.0  224.0  285.0  683.0  1
158  127.0  267.0  155.0  921.0  1
159  136.0  252.0  219.0  634.0  1
160  183.0  337.0  259.0  573.0  2
161  113.0  290.0  281.0  556.0  2
162  124.0  288.0  249.0  575.0  2
163  114.0  303.0  280.0  626.0  2
164  148.0  342.0  188.0  686.0  2
```

The acuracy of the classifier with outliers in the data is 0.9 and worse than before. So it is beneficial to identify the outliers and remove them or treat them in some other way if possible.

Outliers are present in both, the train and test set. In the train set, about 10% of observations are outliers. The most important tuning paraneter in [Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) is `contamination=`. This parameter provides information on the share of outliers in the data. Often the exact share is unknown, so that evaluation on the test set is needed. However, in the present case, I choose `contamination=0.1` since 10% of the data are outliers.

Next I first train the Isolation Forest, identify and remove outliers in `train` and `test` and fit the classifier again on the data without the outliers.

```
# Train isolation forest
isof = IsolationForest(random_state=123, contamination=0.1, bootstrap=True, behaviour="new")
isof.fit(xtrain)

# Remove outlier from train data
isopred = isof.predict(xtrain)
print(isopred)
df = pd.DataFrame(xtrain)
df['y'] = pd.Series(ytrain)
df['iso'] = pd.Series(isopred) 
print(df[df[1]>=100])
print(df.shape)
df = df[df['iso'] == 1]
print(df[df[1]>=100])
print(df.shape)

# Return train data without extreme values
y = df['y']
df = df.drop(['y'], axis=1)
df = df.drop(['iso'], axis=1)
print(df.shape)
xtrain_clean = df.to_numpy()
ytrain_clean = y.to_numpy()

# Remove outlier from test data
isopred = isof.predict(xtest)
df = pd.DataFrame(xtest)
df['y'] = pd.Series(ytest)
df['iso'] = pd.Series(isopred) 
print(df[df[1]>=100])
print(df.shape)
df = df[df['iso'] == 1]
print(df[df[1]>=100])
print(df.shape)

# Return test data without extreme values
y = df['y']
df = df.drop(['y'], axis=1)
df = df.drop(['iso'], axis=1)
print(df.shape)
xtest = df.to_numpy()
ytest = y.to_numpy()

# Run classifier again (on data without outliers)
et = ExtraTreesClassifier(n_estimators=1000, max_features=2, n_jobs=5, random_state=123)
et.fit(xtrain, ytrain)
ypred = et.predict(xtest)

scores = accuracy_score(ytest, ypred)
print(scores)
cm = confusion_matrix(ytest, ypred)
print(cm)
```

A closer look at the data shows, that `ExtraTreesClassifier()` has identified 12 observations as outliers in `train` (while 10 are true outliers) and 6 observations in `test` (while 5 are true outliers). So the choice of `contamination=0.1` in `ExtraTreesClassifier()` is a little generous. However, after removing the outliers identified by `ExtraTreesClassifier()`, the accuracy of the classifier improves to just under 0.98 on the test set (compared to 0.9 with outliers and 0.96 in the benchmark model).

**3. Conclusion**

Removing "extreme values" in data can help to achieve a better predictive performance. However, in case a "new" unknown and unseen set of data is used to make predictions (which usually is the case, e.g. when using a test set), it is often unclear what observations can be seen as outliers and can be removed without the risk of misleading results. 

Isolation Forest provides an option to identify and predict outliers. Thus Isolation Forest can be used to treat outliers in new data (by prediction) which in turn can improve overall predictive performance (at least with respect to the observations not predicted as outliers).
