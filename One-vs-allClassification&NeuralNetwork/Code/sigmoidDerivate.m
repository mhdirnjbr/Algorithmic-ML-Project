function g_prime = sigmoidDerivate(z)

g_prime = sigmoid(z) .* (1 - sigmoid(z));

end