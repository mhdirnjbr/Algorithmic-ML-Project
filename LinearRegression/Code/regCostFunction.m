function J = regCostFunction(X, y, theta, lambda)

m = length(y);

% theta(1) is not considered in the cost function
reg_term = (lambda/(2*m)) * (theta(2:length(theta)))' * theta(2:length(theta));
y_hat = X * theta;
J = (1 / (2 * m)) * (y_hat - y)' * (y_hat - y) + reg_term;

end

