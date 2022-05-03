function [theta, J_history] = gradientDescent(X, y, theta, alpha, iterations)

m = length(y);
J_history = zeros(iterations, 1);

for iter = 1:iterations
   
    y_hat = (X * theta);
    theta = theta - ((alpha / m) * X'* (y_hat - y));

    % Save the cost J in every iteration
    J_history(iter) = costFunction(X, y, theta);
end
end
