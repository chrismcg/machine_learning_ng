import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

#% =============== Part 1: Loading movie ratings dataset ================
#  You will start by loading the movie ratings dataset to understand the
#  structure of the data.
#  
print('Loading movie ratings dataset.')

#  Load data
data = sio.loadmat('../octave/ex8/ex8_movies.mat')
Y = data['Y']
R = data['R']

#  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on
#  943 users
#
#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
#  rating to movie i

#  From the matrix, we can compute statistics like average rating.
print('Average rating for movie 1 (Toy Story): {} / 5'.format(Y[0, R[0, :]].mean()))

#  We can "visualize" the ratings matrix by plotting it with imagesc
plt.imshow(Y)
plt.ylabel('Movies')
plt.xlabel('Users')
# plt.show()

#% ============ Part 2: Collaborative Filtering Cost Function ===========
#  You will now implement the cost function for collaborative filtering.
#  To help you debug your cost function, we have included set of weights
#  that we trained on that. Specifically, you should complete the code in 
#  cofiCostFunc.m to return J.

#  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
data = sio.loadmat('../octave/ex8/ex8_movieParams.mat')
X = data['X']
theta = data['Theta']

#  Reduce the data set size so that this runs faster
num_users = 4
num_movies = 5
num_features = 3

X = X[0:num_movies, 0:num_features]
theta = theta[0:num_users, 0:num_features]
Y = Y[0:num_movies, 0:num_users]
R = R[0:num_movies, 0:num_users]

def cofi_cost_func(params, y, r, num_users, num_movies, num_features, reg_lambda):
    x = params[0:num_movies * num_features].reshape((num_movies, num_features))
    theta = params[num_movies * num_features:].reshape((num_users, num_features))

    error = (x.dot(theta.T) - y) * r
    j = (error ** 2).sum() / 2
    user_reg = (theta ** 2).sum() * (reg_lambda / 2)
    movie_reg = (x ** 2).sum() * (reg_lambda / 2)
    j = j + user_reg + movie_reg

    x_grad = error.dot(theta) + (reg_lambda * x)
    theta_grad = error.T.dot(x) + (reg_lambda * theta)
    grad = np.r_[x_grad.ravel(), theta_grad.ravel()]

    return j, grad

#  Evaluate cost function
params = np.r_[X.ravel(), theta.ravel()]
J, _ = cofi_cost_func(params, Y, R, num_users, num_movies, num_features, 0)

print('Cost at loaded parameters: {} '.format(J))
print('(this value should be about 22.22)')

#% ============== Part 3: Collaborative Filtering Gradient ==============
#  Once your cost function matches up with ours, you should now implement 
#  the collaborative filtering gradient function. Specifically, you should 
#  complete the code in cofiCostFunc.m to return the grad argument.
#  
print('Checking Gradients (without regularization) ... ')

def compute_numerical_gradient(cost_function, theta):
    numerical_gradient = np.zeros(theta.shape)
    perturbations = np.zeros(theta.shape)
    e = 1e-4

    for p in range(theta.size):
        perturbations[p] = e
        loss1, _ = cost_function(theta - perturbations)
        loss2, _ = cost_function(theta + perturbations)
        numerical_gradient[p] = (loss2 - loss1) / (2 * e)
        perturbations[p] = 0

    return numerical_gradient

def check_cost_function(reg_lambda = 0):
    x_t = np.random.rand(4, 3)
    theta_t = np.random.rand(5, 3)
    y = x_t.dot(theta_t.T)
    y[np.random.rand(y.shape[0], y.shape[1]) > 0.5] = 0
    r = np.zeros(y.shape)
    r[np.logical_not(y) == False] = 1

    x = np.random.randn(x_t.shape[0], x_t.shape[1])
    theta = np.random.randn(theta_t.shape[0], theta_t.shape[1])
    num_users = y.shape[1]
    num_movies = y.shape[0]
    num_features = theta.shape[1]
    params = np.r_[x.ravel(), theta.ravel()]

    numerical_gradient = compute_numerical_gradient(
        lambda t: cofi_cost_func(t, y, r, num_users, num_movies, num_features, reg_lambda),
        params.copy()
    )

    cost, gradient = cofi_cost_func(params.copy(), y, r, num_users, num_movies, num_features, reg_lambda)

    print(np.c_[numerical_gradient, gradient])
    print('The above two columns you get should be very similar.')
    print('(Left-Your Numerical Gradient, Right-Analytical Gradient)')

    diff = np.linalg.norm(numerical_gradient - gradient) / np.linalg.norm(numerical_gradient + gradient)
    print('If your cost function implementation is correct, then')
    print('the relative difference will be small (less than 1e-9).')
    print('Relative Difference: {}'.format(diff))


#  Check gradients by running checkNNGradients
# check_cost_function()

#% ========= Part 4: Collaborative Filtering Cost Regularization ========
#  Now, you should implement regularization for the cost function for 
#  collaborative filtering. You can implement it by adding the cost of
#  regularization to the original cost computation.
#

#  Evaluate cost function
params = np.r_[X.ravel(), theta.ravel()]
J, _ = cofi_cost_func(params, Y, R, num_users, num_movies, num_features, 1.5);

print('Cost at loaded parameters (lambda = 1.5): {} '.format(J))
print('(this value should be about 31.34)')

#% ======= Part 5: Collaborative Filtering Gradient Regularization ======
#  Once your cost matches up with ours, you should proceed to implement 
#  regularization for the gradient. 
#


print('Checking Gradients (with regularization) ... ')

#  Check gradients by running checkNNGradients
check_cost_function(1.5)

#% ============== Part 6: Entering ratings for a new user ===============
#  Before we will train the collaborative filtering model, we will first
#  add ratings that correspond to a new user that we just observed. This
#  part of the code will also allow you to put in your own ratings for the
#  movies in our dataset!
#


def load_movie_list():
    movies = []
    with open('../octave/ex8/movie_ids.txt', encoding="latin1") as f:
        for line in f:
            movie = ' '.join(line.rstrip().split(' ')[1:])
            movies.append(movie)

    return movies


movie_list = load_movie_list()

#  Initialize my ratings
my_ratings = np.zeros((1682, 1))

# Check the file movie_idx.txt for id of each movie in our dataset
# For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
my_ratings[0] = 4

# Or suppose did not enjoy Silence of the Lambs (1991), you can set
my_ratings[97] = 2

# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
my_ratings[6] = 3
my_ratings[11]= 5
my_ratings[53] = 4
my_ratings[63]= 5
my_ratings[65]= 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354]= 5

print('New user ratings:')
for i in range(my_ratings.size):
    if my_ratings[i] > 0:
        print('Rated {} for {}'.format(my_ratings[i], movie_list[i]))

#% ================== Part 7: Learning Movie Ratings ====================
#  Now, you will train the collaborative filtering model on a movie rating
#  dataset of 1682 movies and 943 users
#

print('Training collaborative filtering...')

#  Load data
data = sio.loadmat('../octave/ex8/ex8_movies.mat')
Y = data['Y']
R = data['R']

#  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by
#  943 users
#
#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
#  rating to movie i

#  Add our own ratings to the data matrix
Y = np.c_[my_ratings, Y]
R = np.c_[np.not_equal(my_ratings, 0).astype(int), R]

def normalize_ratings(y, r):
    m, _ = y.shape
    ymean = np.zeros((m, 1))
    ynorm = np.zeros(y.shape)
    for i in range(m):
        idx = np.nonzero(r[i, :] == 1)
        ymean[i] = np.mean(y[i, idx])
        ynorm[i, idx] = y[i, idx] - ymean[i]

    return ynorm, ymean

#  Normalize Ratings
Ynorm, Ymean = normalize_ratings(Y, R)

#  Useful Values
num_users = R.shape[1]
num_movies = Y.shape[0]
num_features = 10

# Set Initial Parameters (Theta, X)
X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)

initial_parameters = np.r_[X.ravel(), Theta.ravel()]

# Set options for fmincg
# options = optimset('GradObj', 'on', 'MaxIter', 100);

# Set Regularization
reg_lambda = 10;
# theta = fmincg (@(t)(cofiCostFunc(t, Ynorm, R, num_users, num_movies, ...
#                                 num_features, lambda)), ...
#                 initial_parameters, options);

def cost_function(t):
    return cofi_cost_func(t, Ynorm, R, num_users, num_movies, num_features, reg_lambda)

result = op.minimize(
            cost_function,
            np.copy(initial_parameters),
            # method="Newton-CG",
            method="L-BFGS-B",
            jac=True,
            options={'maxiter': 50, 'disp': False}
        )

theta = result.x

# Unfold the returned theta back into U and W
# X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
X = theta[:num_movies*num_features].reshape((num_movies, num_features))
# Theta = reshape(theta(num_movies*num_features+1:end), ...
#                 num_users, num_features);
Theta = theta[num_movies*num_features:].reshape((num_users, num_features))

print('Recommender system learning completed.')

#% ================== Part 8: Recommendation for you ====================
#  After training the model, you can now make recommendations by computing
#  the predictions matrix.
#

p = X.dot(Theta.T)

my_predictions = (p + Ymean)[:, 0]

movie_list = load_movie_list()

ix = np.argsort(-my_predictions)
print('Top recommendations for you:')
for i in range(10):
    j = ix[i];
    print('Predicting rating {:0.1f} for movie {}'.format(my_predictions[j],  movie_list[j]))

print('Original ratings provided:')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print('Rated {} for {}'.format(my_ratings[i, :], movie_list[i]))
