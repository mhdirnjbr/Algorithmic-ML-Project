function J = costFunction(X, y, theta)

m = length(y);
y_hat = X * theta;
J = (1 / (2 * m)) * (y_hat - y)' * (y_hat - y);

end