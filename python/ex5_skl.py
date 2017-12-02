# The first half of this, up to polyfeatures, looks close to the Octave
# version. After that it gets different because of how the built in tools or
# doc examples use a score rather than plotting the cost.
# The random selection of train/test examples and how cross validation works also doesn't help
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, learning_curve, validation_curve, KFold, RepeatedKFold
from sklearn.pipeline import Pipeline

def read_data():
    # Load from ex5data1 and 'stitch' it back together again as going to use
    # sklearn tools for cross validation
    data = sio.loadmat('../octave/ex5/ex5data1.mat')
    X = data['X']
    y = data['y']
    Xval = data['Xval']
    yval = data['yval']
    Xtest = data['Xtest']
    ytest = data['ytest']

    return (np.vstack((X, Xval, Xtest)), np.vstack((y, yval, ytest)))


X, y = read_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=21) #, random_state=0)
m_train = X_train.shape[0]

fig, ax = plt.subplots()
ax.plot(X_train, y_train, 'rx', markersize=10, linewidth=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()

regression = linear_model.Ridge(alpha=0.01, normalize=False)
regression.fit(X_train, y_train)
prediction = regression.predict(X_train)

fig, ax = plt.subplots()
ax.plot(X_train, y_train, 'rx', markersize=10, linewidth=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
ax.plot(X_train, prediction, '--', linewidth=2)
plt.show()

# FROM scikit-learn docs
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5),
                        show_std=False):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    if show_std:
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Using the whole data set seems to give a better and more consistent graph.
# Don't need to split into train/validation/test like in Octave because the
# RepeatedKFold splitter that scikit-learn uses in the call to learning_curve does this
# for us and in a better way.
#
# Can't really do the increasing size by one each time that Octave does with
# it's 1:m loop as you specify percentages of the data for each step in
# learning_curve. I've picked 20 steps between 1% and 100% and this seems good enough to see the shape.
#
# The ylim's are necessary as you have have really bad scores that cause the
# graph axis to get to big to see important details
#
# NB: This plots % correct rather than number of errors so graph is "flipped" on y axis
plot_learning_curve(
    regression,
    "Test",
    X,
    y,
    train_sizes=np.linspace(.01, 1.0, 20),
    cv=RepeatedKFold(),
    ylim=(-2, 1)
)
# plot_learning_curve(regression, "Test", X_train, y_train, train_sizes=np.linspace(.1, 1.0, 20))
plt.show()

alpha = 0.000000001
pipeline = Pipeline([
    ('poly', PolynomialFeatures(8, include_bias=False)),
    ('scale', StandardScaler()),
    ('reg', linear_model.Ridge(alpha=alpha, normalize=False))
])

pipeline.fit(X_train, y_train)
X_predict = np.linspace(X_train.min() - 15, X_train.max() + 15, 100).reshape((-1, 1))
prediction = pipeline.predict(X_predict)

fig, ax = plt.subplots()
ax.plot(X_train, y_train, 'rx', markersize=10, linewidth=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
ax.plot(X_predict, prediction, '--', linewidth=2)

# NB: This uses all data like the one before it
# This really needs the ylim to be able to interpret the high variance output
plot_learning_curve(
    pipeline,
    "Polynomial Regression Learning Curve, alpha={}".format(alpha),
    X,
    y,
    train_sizes=np.linspace(0.01, 1.0, 20),
    cv=RepeatedKFold(),
    ylim=(-2, 1)
)
plt.show()

pipeline = Pipeline([
    ('poly', PolynomialFeatures(8, include_bias=False)),
    ('scale', StandardScaler()),
    ('reg', linear_model.Ridge(normalize=False))
])

# alphas = [0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
alphas = np.geomspace(0.001, 10, 10)
# This is different than the version in Octave as it works off the score not
# the error rate. It doesn't seem as useful for picking the learning rate, but
# they have CV versions that can do it automatically anyway. For visualizing I
# ended up using a log scale to try and make it clearer. It also seems very
# unstable and can change a lot depending on how the test/train split ends up.
train_scores, valid_scores = validation_curve(pipeline, X_train, y_train, "reg__alpha", alphas, verbose=1)

fig, ax = plt.subplots()
ax.set_xscale('log')
ax.plot(alphas, train_scores.mean(axis=1), label="Train")
ax.plot(alphas, valid_scores.mean(axis=1), label="Cross Validation")
plt.legend()
plt.xlabel('alpha')
plt.ylabel('Score')
plt.show()

# Automatically pick best fit with cross validation
pipeline = Pipeline([
    ('poly', PolynomialFeatures(8, include_bias=False)),
    ('scale', StandardScaler()),
    ('reg', linear_model.RidgeCV(alphas=alphas, normalize=False))
])
pipeline.fit(X_train, y_train)
score = pipeline.score(X_test, y_test)
print(score)
