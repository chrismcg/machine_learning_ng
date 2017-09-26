fprintf('Running multiple lambda test ...\n');

data = load('ex2data2.txt');
X = data(:, [1, 2]);
y = data(:, 3);
X = mapFeature(X(:,1), X(:,2));

lambdas = [0, 1, 10, 100];

function theta = testLambda(X, y, lambda)
  initial_theta = zeros(size(X, 2), 1);

  % Set Options
  options = optimset('GradObj', 'on', 'MaxIter', 400);

  % Optimize
  [theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
end

calculated_thetas = arrayfun(@(lambda) {lambda, testLambda(X, y, lambda)}, lambdas, "UniformOutput", false);

% output will be of form:
% lambda, x, y, z
% with 2500 rows per lambda
filename = "multipleLambdas.csv";
fid = fopen(filename, "w");
fputs(fid, "lambda,x,y,z\n");
for i = 1:size(calculated_thetas, 2)
  lambda = calculated_thetas{i}{1};
  theta = calculated_thetas{i}{2};

  x = linspace(-1, 1.5, 50);
  y = linspace(-1, 1.5, 50);
  for i = 1:length(x)
    for j = 1:length(y)
      z = mapFeature(x(i), y(j)) * theta;
      csvwrite("multipleLambdas.csv", [lambda, x(i), y(j), z], "append", "on");
    endfor
  endfor
endfor
