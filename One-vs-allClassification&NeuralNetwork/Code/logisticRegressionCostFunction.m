function [J, grad] = logisticRegressionCostFunction(theta, X, y, lambda)

m = length(y);
grad = zeros(size(theta));

z = X * theta;
y_hat = sigmoid(z);
reg_term = (lambda/(2 * m)) * (theta(2:length(theta)))' * theta(2:length(theta));

J = (1/m) * ((-y' * log(y_hat)) - (1 - y)' * log(1 - y_hat)) + reg_term;

grad(1) = (1/m) * (X(:,1)' * (y_hat - y));
grad(2:end) = (1/m) * (X(:,2:end)' * (y_hat - y)) + (lambda/m) * theta(2:end);

end