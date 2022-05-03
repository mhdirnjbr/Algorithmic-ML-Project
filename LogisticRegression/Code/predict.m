function p = predict(theta, X)

z = X * theta;
y_hat = sigmoid(z);
p =  y_hat >= 0.5;

end