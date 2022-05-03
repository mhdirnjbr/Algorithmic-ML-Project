function theta = regGradientDescent(X, y, theta, lambda, alpha, num_iters)

m = length(y);

for iter = 1:num_iters

    y_hat = (X * theta);
    theta(1) = theta(1) - ((alpha / m) * X(:,1)' * (y_hat - y));
    theta(2:end) = theta(2:end) * (1 - (alpha * (lambda / m))) - ((alpha / m) * X(:,2:end)' * (y_hat - y));

end
end

