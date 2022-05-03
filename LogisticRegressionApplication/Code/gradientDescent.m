function [theta, J_history] = gradientDescent(X, y, theta, alpha, iterations, lambda)

m = length(y);
J_history = zeros(iterations, 1);

for iter = 1:iterations

    z = X * theta;
    y_hat = sigmoid(z);
   
    theta(1) = theta(1) - ((alpha / m) * X(:,1)' * (y_hat - y)); % Not regularize the bias term
    theta(2:end) = theta(2:end) * (1 - (alpha * (lambda / m))) - ((alpha / m) * X(:,2:end)' * (y_hat - y));

    % Save the cost J in every iteration
    J_history(iter) = costFunction(theta, X, y, lambda);
end
end