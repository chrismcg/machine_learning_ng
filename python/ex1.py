import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../octave/ex1/ex1data1.txt", names=['X', 'y'])


def plot_data(x, y):
    plt.plot(x, y, 'rx', markersize=10)
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')

    plt.show()


plot_data(df['X'], df['y'])