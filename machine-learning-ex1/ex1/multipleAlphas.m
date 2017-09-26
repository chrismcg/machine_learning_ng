fprintf('Running multiple alpha test ...\n');
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

[X mu sigma] = featureNormalize(X);
X = [ones(m, 1) X];

% multiple runs
function J_history = testAlpha (X, y, alpha, iterations)
  theta = zeros(3, 1);
  [testTheta, J_history] = gradientDescentMulti(X, y, theta, alpha, iterations);
end

alphas = [0.01, 0.03, 0.1, 0.3, 1];
iterations = 50;

output = arrayfun(@(alpha) {alpha, testAlpha(X, y, alpha, iterations)}, alphas, "UniformOutput", false);

colors = ['k', 'r', 'g', 'b', 'm'];

csv_data = ones(iterations + 1, size(alphas, 2));

for i = 1:size(output, 2)
  alpha = output{i}{1};
  J_history = output{i}{2};
  csv_data(:, i) = [alpha; J_history];
  plot(1:50, J_history, colors(i));
  hold on;
endfor

csvwrite("multipleAlphas.csv", csv_data);

