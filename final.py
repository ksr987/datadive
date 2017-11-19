import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import sklearn.metrics as mt
from sklearn.model_selection import cross_val_score

in_data = pd.read_csv('data.csv', low_memory=False)
y = in_data['sub_grade']
x = in_data

x_train, x_test, y_train, y_test = train_test_split(x, y)


reg = Ridge(alpha = .5)

reg.fit(x_train, y_train)

reg_predictions = reg.predict(x_test)

print(y_test[0:5])
print(reg_predictions[0:5])
scores = cross_val_score(reg, x_test, y_test, cv=5)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)\n" % (scores.mean(), scores.std() * 2))

print(mt.classification_report(y_test, reg_predictions))
print(mt.confusion_matrix(y_test, reg_predictions))
