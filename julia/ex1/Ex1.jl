data = readcsv("ex1data1.txt")
y = data[:, 2]
m = length(y)

X = hcat(ones(m, 1), data[:, 1])
theta = zeros(2, 1)

iterations = 1500
alpha = 0.01

using Gadfly
using DataFrames

set_default_plot_size(800px, 600px)

df = rename!(DataFrame(data), Dict(:x1 => :population, :x2 => :profit))
plot(df, x = :population, y = :profit, shape = [Gadfly.cross], Geom.point)

computeCost(X, y, theta) = sum(((X * theta) - y) .^ 2) / 2m

print("should be ~ 32.07: ")
display(computeCost(X, y, theta))

print("should be ~ 54.24: ")
display(computeCost(X, y, [-1; 2]))

function gradientDescent(X, y, theta, alpha, iterations)
  J_history = zeros(iterations, 1)

  for iteration = 1:iterations
    h = X * theta
    errors = h - y
    theta = theta - ((alpha / m) * (X' * errors))
    J_history[iteration] = computeCost(X, y, theta)
  end

  [theta, J_history]
end

minTheta, J_history = gradientDescent(X, y, theta, alpha, iterations)

print("should be ~ [-3.6303; 1.1664]: ")
display(minTheta)

print("Prediction for population 35,000: ")
display(([1 3.5] * minTheta) * 10000)

print("Prediction for population 70,000: ")
display(([1 7] * minTheta) * 10000)
df[:linear] = (X * minTheta)[:]

plot(df,
  layer(x = :population, y = :profit, shape = [Gadfly.cross], Geom.point),
  layer(x = :population, y = :linear, Geom.line)
)

theta0_vals = linspace(-10, 10, 100)
theta1_vals = linspace(-1, 4, 100)

J_vals = zeros(length(theta0_vals), length(theta1_vals));

for i = 1:length(theta0_vals)
  for j = 1:length(theta1_vals)
    t = [theta0_vals[i]; theta1_vals[j]]
    J_vals[i,j] = computeCost(X, y, t)
  end
end

plot(
 layer(x = theta0_vals, y = theta1_vals, z = J_vals, Geom.contour(levels=logspace(-2, 3, 20))),
    layer(x = [minTheta[1]], y = [minTheta[2]], shape=[Gadfly.cross], Theme(default_color="red"), Geom.point),
    Guide.xlabel("Theta 0"),
    Guide.ylabel("Theta 1")
)
