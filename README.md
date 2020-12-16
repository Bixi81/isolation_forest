# How to use Isolation Forest in Python

*Last update: 16 December 2020*

[Isolation Forest (IF)](https://en.wikipedia.org/wiki/Isolation_forest) can be used to detect outliers in a dataset. IF is a model that is trained on some data and can be predicted on new data. Thus, IF makes it possible to identify outliers in new data in a similar way as in an original training dataset. This can be helpful when outliers in new data need to be identified in order to ensure the accuracy of a predictive model.

**1. Basic Example (sklearn)**

Before I go into more detail, I show a brief example that highlights how Isolation Forest with sklearn works.

First load some packages (I will use them throughout this example)

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


```
code
```
