function [J, grad] = costFunction(theta, X, y)

m = length(y);

z = X * theta;
y_hat = sigmoid(z);
J = (1 / m) * ((-y' * log(y_hat)) - (1 - y)' * log(1 - y_hat));

grad = (1 / m) * (X' *(y_hat - y));

end