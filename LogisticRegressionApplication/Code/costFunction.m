function J = costFunction(theta, X, y, lambda)

m = length(y); 

z = X * theta;
y_hat = sigmoid(z);
reg_term = (lambda/(2*m)) * (theta(2:length(theta)))' * theta(2:length(theta));

J = (1/m) * ((-y' * log(y_hat)) - (1 - y)' * log(1 - y_hat)) + reg_term;

end