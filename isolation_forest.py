import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import datasets
from random import randrange

#########################################################
# Minimal Example
df = {'y': [1,2,3,4,5,6,7,8,9,10], 'x': [11,12,13,14,15,16,17,23454,19,20]}
df = pd.DataFrame(data=df)
print(df)

x = df['x'].to_numpy().reshape(-1, 1)
clf = IsolationForest(random_state=2020, behaviour="new")
clf.fit(x)
pred = clf.predict(x)
df = pd.concat([df,pd.Series(pred)], axis=1)
print(df)

df_new = {'y': [1,2,3,4,5,6,7,8,9,10], 'x': [11,12,13,14,15,16,17,18,19,20]}
df_new = pd.DataFrame(data=df_new)
x = df_new['x'].to_numpy().reshape(-1, 1)
pred_new = clf.predict(x)
df_new = pd.concat([df_new,pd.Series(pred_new)], axis=1)
print(df_new)

#########################################################
# Load Iris data
iris = datasets.load_iris()
x = iris.data[:, :]  
y = iris.target

#########################################################
# Benchmark model

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

#########################################################
# Data with extreme values (outliers)

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

#########################################################
# Isolation forest to detect outliers

# Train isolation forest
isof = IsolationForest(random_state=123, contamination=0.1, bootstrap=True, behaviour="new")
isof.fit(xtrain)

# Remove outlier from train data
isopred = isof.predict(xtrain)
print(isopred)
df = pd.DataFrame(xtrain)
df['y'] = pd.Series(ytrain)
df['iso'] = pd.Series(isopred) 
print("Trian data with outlier")
print(df[df[1]>=100])
print(df[df["iso"]== -1])
print(df.shape)
print("Trian data without outlier")
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
print("Test data with outlier")
print(df[df[1]>=100])
print(df.shape)
df = df[df['iso'] == 1]
print("Test data without outlier")
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
