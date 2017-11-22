import pandas as pd
import numpy as np

from sklearn import linear_model


def read_data():
    df = pd.read_csv("../octave/ex1/ex1data2.txt", names=['X1', 'X2', 'y'])
    x = df.as_matrix(['X1', 'X2'])
    y = df.as_matrix(['y'])

    return (x, y)


X, y = read_data()
regression = linear_model.LinearRegression()
regression.fit(X, y)

to_estimate = np.array([1650, 3])

predictions = regression.predict([to_estimate])

price = predictions[0][0]
print('Predicted price of a 1650 sq-ft, 3 br house ')
print('(using sklearn linear_model):')
print(' {}'.format(price))
