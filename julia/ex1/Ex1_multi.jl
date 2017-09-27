data = readcsv("ex1data2.txt")
X = data[:, 1:2]
y = data[:, 3]
m = length(y)

function featureNormalize(X)
  mu = mean(X, 1)
  sigma = std(X, 1)

  X_norm = (X .- mu) ./ sigma

  X_norm, mu, sigma
end

X, mu, sigma = featureNormalize(X)
X = [ones(m, 1) X]

computeCostMulti(X, y, theta) = sum(((X * theta) - y) .^ 2) / 2m

function gradientDescentMulti(X, y, theta, alpha, iterations)
  J_history = zeros(iterations, 1)

  for iteration = 1:iterations
    h = X * theta
    errors = h - y
    theta = theta - ((alpha / m) * (X' * errors))
    J_history[iteration] = computeCostMulti(X, y, theta)
  end

  theta, J_history
end

alpha = 0.01
iterations = 400
theta = zeros(3, 1)

minTheta, J_history = gradientDescentMulti(X, y, theta, alpha, iterations)

print("theta from gradient descent: ")
display(map(x -> round(x, 2), minTheta))

using Gadfly
set_default_plot_size(800px, 600px)
plot(x = 1:size(J_history, 1), y = J_history, Geom.line)

to_estimate = [1650 3]
to_estimate_normalized = (to_estimate .- mu) ./ sigma
input = hcat(1, to_estimate_normalized)
price = input * minTheta

print("Gradient Descent estimated price of 1650 sq-ft, 3 br house: ")
@printf("%.2f", price[1])
println()

data = readcsv("ex1data2.txt")
X = data[:, 1:2]
y = data[:, 3]
m = length(y)

function normalEqn(X, y)
  theta = pinv(X' * X) * X' * y
end

X = [ones(m, 1) X]
minTheta = normalEqn(X, y)

print("theta from normal equation: ")
display(map(x -> round(x, 2), minTheta))
println()

to_estimate = [1 1650 3]
price = to_estimate * minTheta

print("Normal equation estimated price of 1650 sq-ft, 3 br house: ")
@printf("%.2f", price[1])
println()
