import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

# Load Training Data
print('Loading and Visualizing Data ...')

# Load from ex5data1:
# You will have X, y, Xval, yval, Xtest, ytest in your environment
data = sio.loadmat('../octave/ex5/ex5data1.mat')
X = data['X']
y = data['y']
Xval = data['Xval']
yval = data['yval']
Xtest = data['Xtest']
ytest = data['ytest']

# m = Number of examples
m, n = X.shape

# Plot training data
fig, ax = plt.subplots()

ax.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()

#% =========== Part 2: Regularized Linear Regression Cost =============
#  You should now implement the cost function for regularized linear 
#  regression. 
#

def linear_reg_cost_function(x, y, theta, reg_lambda):
    m = y.size
    theta = theta.reshape((theta.size, 1))
    error = x.dot(theta) - y
    j = (error ** 2).sum() / (2 * m)
    regularization = (theta[1:, :] ** 2).sum() * (reg_lambda / (2 * m))
    j = j + regularization

    gradient = ((error * x).sum(axis=0, keepdims=True) / m)
    gradient_regularization = np.copy(theta).transpose()
    gradient_regularization[:, 0] = 0
    gradient_regularization = gradient_regularization * (reg_lambda / m)
    gradient = gradient + gradient_regularization

    return (j, gradient[0])

theta = np.array([1, 1]).reshape((2, 1))
cost, gradients = linear_reg_cost_function(np.c_[np.ones((m, 1)), X], y, theta, 1);

print('Cost at theta = [1 ; 1]: {}'.format(cost))
print('(this value should be about 303.993192)')

#% =========== Part 3: Regularized Linear Regression Gradient =============
#  You should now implement the gradient for regularized linear 
#  regression.
#

theta = np.array([1, 1]).reshape((2, 1))
cost, gradients = linear_reg_cost_function(np.c_[np.ones((m, 1)), X], y, theta, 1);

print('Gradient at theta = [1 ; 1]:  [{}; {}]'.format(gradients[0], gradients[1]))
print('(this value should be about [-15.303016; 598.250744])')

#% =========== Part 4: Train Linear Regression =============
#  Once you have implemented the cost and gradient correctly, the
#  trainLinearReg function will use your cost function to train 
#  regularized linear regression.
# 
#  Write Up Note: The data is non-linear, so this will not give a great 
#                 fit.
#

def train_linear_reg(x, y, reg_lambda):
    m, n = x.shape
    initial_theta = np.zeros((n, 1))

    result = op.minimize(
        lambda t: linear_reg_cost_function(x, y, t, reg_lambda),
        initial_theta,
        method="TNC",
        jac=True,
        options={'maxiter': 200, 'disp': False}
    )

    return result.x

#  Train linear regression with lambda = 0
reg_lambda = 0;
theta = train_linear_reg(np.c_[np.ones((m, 1)), X], y, reg_lambda)

# Plot fit over the data
fig, ax = plt.subplots()

ax.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
ax.plot(X, np.c_[np.ones((m, 1)), X].dot(theta), '--', linewidth=2)
plt.show()

#% =========== Part 5: Learning Curve for Linear Regression =============
#  Next, you should implement the learningCurve function. 
#
#  Write Up Note: Since the model is underfitting the data, we expect to
#                 see a graph with "high bias" -- Figure 3 in ex5.pdf 
#

def learning_curve(x, y, x_val, y_val, reg_lambda):
    m, _ = x.shape
    mval, _ = x_val.shape
    iterations = 50
    error_train = []
    error_validation = []

    for i in range(m):
        iteration_error_train = []
        iteration_error_validation = []

        for iteration in range(iterations):
            random_train_indices = np.random.permutation(m)
            x_i = x[random_train_indices[:i + 1], :]
            y_i = y[random_train_indices[:i + 1]]

            theta = train_linear_reg(x_i, y_i, reg_lambda)

            cost, _ = linear_reg_cost_function(x_i, y_i, theta, 0)
            iteration_error_train.append(cost)

            random_validation_indices = np.random.permutation(mval)
            x_val_i = x_val[random_validation_indices[:i + 1], :]
            y_val_i = y_val[random_validation_indices[:i + 1]]

            cost, _ = linear_reg_cost_function(x_val_i, y_val_i, theta, 0)
            iteration_error_validation.append(cost)

        error_train.append(np.mean(iteration_error_train))
        error_validation.append(np.mean(iteration_error_validation))

    return (error_train, error_validation)

reg_lambda = 0;
mval, nval = Xval.shape
error_train, error_validation = learning_curve(np.c_[np.ones((m, 1)), X], y, np.c_[np.ones((mval, 1)), Xval], yval, reg_lambda)

fig, ax = plt.subplots()

ax.plot(list(range(m)), error_train, label='Train')
ax.plot(list(range(m)), error_validation, label='Cross Validation')
ax.axis([0, 13, 0, 150])
plt.legend()
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.show()

print('# Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print('  \t{}\t\t{}\t{}'.format(i, error_train[i], error_validation[i]))

#% =========== Part 6: Feature Mapping for Polynomial Regression =============
#  One solution to this is to use polynomial regression. You should now
#  complete polyFeatures to map each example into its powers
#

def poly_features(x, p):
    powers = np.linspace(1, p, p)
    return x ** powers

def feature_normalize(x):
    mu = np.mean(x, axis=0)
    x_norm = x - mu

    sigma = np.std(x_norm, axis=0)
    x_norm = x_norm / sigma

    return (x_norm, mu, sigma)

p = 8;

# Map X onto Polynomial Features and Normalize
X_poly = poly_features(X, p)
[X_poly, mu, sigma] = feature_normalize(X_poly)  # Normalize
X_poly = np.c_[np.ones((m, 1)), X_poly]                   # Add Ones

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = poly_features(Xtest, p)
X_poly_test = (X_poly_test - mu) / sigma
m_xpt, _ = X_poly_test.shape
X_poly_test = np.c_[np.ones((m_xpt, 1)), X_poly_test]         # Add Ones

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = poly_features(Xval, p)
X_poly_val = (X_poly_val - mu) / sigma
m_xpv, _ = X_poly_val.shape
X_poly_val = np.c_[np.ones((m_xpv, 1)), X_poly_val]           # Add Ones

print('Normalized Training Example 1:')
print('  {}  \n'.format(X_poly[0, :]))

#% =========== Part 7: Learning Curve for Polynomial Regression =============
#  Now, you will get to experiment with polynomial regression with multiple
#  values of lambda. The code below runs polynomial regression with 
#  lambda = 0. You should try running the code with different values of
#  lambda to see how the fit and learning curve change.
#

reg_lambda = 0.01
theta = train_linear_reg(X_poly, y, reg_lambda)

# Plot training data and fit
def plot_fit(ax, min_x, max_x, mu, sigma, theta, p):
    x = np.arange(min_x - 15, max_x + 15, 0.05).reshape(-1, 1)
    x_poly = (poly_features(x, p) - mu) / sigma
    x_poly = np.c_[np.ones((x_poly.shape[0], 1)), x_poly]
    ax.plot(x, x_poly.dot(theta), '--', linewidth=2)

fig, ax = plt.subplots()
ax.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plot_fit(ax, min(X), max(X), mu, sigma, theta, p);
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title('Polynomial Regression Fit (lambda = {})'.format(reg_lambda))
plt.show()

fig, ax = plt.subplots()
error_train, error_val = learning_curve(X_poly, y, X_poly_val, yval, reg_lambda);
ax.plot(list(range(m)), error_train, label='Train')
ax.plot(list(range(m)), error_val, label='Cross Validation')
plt.title('Polynomial Regression Learning Curve (lambda = {})'.format(reg_lambda))
plt.xlabel('Number of training examples')
plt.ylabel('Error')
ax.axis([0, 13, 0, 100])
plt.legend()
plt.show()

print('Polynomial Regression (lambda = {})'.format(reg_lambda))
print('# Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print('  \t{}\t\t{}\t{}'.format(i, error_train[i], error_validation[i]))

#% =========== Part 8: Validation for Selecting Lambda =============
#  You will now implement validationCurve to test various values of 
#  lambda on a validation set. You will then use this to select the
#  "best" lambda value.
#

def validation_curve(x, y, x_val, y_val):
    lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    error_train = []
    error_val = []

    for reg_lambda in lambda_vec:
        theta = train_linear_reg(x, y, reg_lambda)

        cost, _ = linear_reg_cost_function(x, y, theta, 0)
        error_train.append(cost)

        cost, _ = linear_reg_cost_function(x_val, y_val, theta, 0)
        error_val.append(cost)

    return (lambda_vec, error_train, error_val)


lambda_vec, error_train, error_val = validation_curve(X_poly, y, X_poly_val, yval);

fig, ax = plt.subplots()
ax.plot(lambda_vec, error_train, label="Train")
ax.plot(lambda_vec, error_val, label="Cross Validation")
plt.legend()
plt.xlabel('lambda')
plt.ylabel('Error')
plt.show()

print('lambda\t\tTrain Error\tValidation Error');
for i in range(len(lambda_vec)):
    print(' {}\t{}\t{}'.format(lambda_vec[i], error_train[i], error_val[i]))

# testing against Xtest
reg_lambda = 3
theta = train_linear_reg(X_poly, y, reg_lambda)
cost, _ = linear_reg_cost_function(X_poly_test, ytest, theta, 0)
print('Test data error (lambda = {}): {}'.format(reg_lambda, cost))
