from sklearn import linear_model
import numpy as np

A = np.array([[2,5,7,9,11,16,19,23,22,29,29,35,37,40,46]]).T
b = np.array([[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]).T

lr = linear_model.LinearRegression()
result = lr.fit(A, b)

kp = result.predict([[12]])

print(kp)
