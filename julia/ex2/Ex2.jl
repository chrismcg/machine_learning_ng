data = readcsv("ex2data1.txt")
X = data[:, 1:2]
y = data[:, 3]

using Gadfly
using DataFrames

set_default_plot_size(800px, 600px)

df = rename!(DataFrame(data), Dict(:x1 => :score1, :x2 => :score2, :x3 => :admitted))
pool!(df, [:admitted])
plot(df,
  x = :score1,
  y = :score2,
  shape = :admitted,
  Geom.point,
  color = :admitted,
  Theme(point_shapes = [circle, Gadfly.cross], highlight_width = 0pt),
  Scale.color_discrete,
  Coord.cartesian(xmin=30, xmax=100, ymin=30, ymax=100)
)

m, n = size(X)

X = hcat(ones(m, 1), X)

initial_theta = zeros(n + 1, 1)

sigmoid(z) = 1 ./ (1 .+ exp.(-z))

function costFunction(theta, X, y)
  h = sigmoid(X * theta)
  J = ((-y' * log.(h)) - ((1 - y') * (log.(1 - h)))) / m
  grad = (X' * (h - y)) / m
  [J, grad]
end

cost, grad = costFunction(initial_theta, X, y)

print("should be ~0.693: ")
display(cost)

print("should be ~ -0.1, -12.0092, -11.2686 :")
display(grad)

test_theta = [-24; 0.2; 0.2]

cost, grad = costFunction(test_theta, X, y)

print("should be ~0.218: ")
display(cost)

print("should be ~ -0.043, 2.566, 2.647 :")
display(grad)

using Optim

function test(theta)
  display(theta)
  J, grad = costFunction(theta, X, y)
  J
end

optimize(test, initial_theta, GradientDescent())
