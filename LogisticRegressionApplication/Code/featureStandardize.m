function [X_norm, mu, sigma] = featureStandardize(X)

X_norm = X;

mu = mean(X); % Compute the mean
sigma = std(X); % Compute the standard deviation

X_norm = (X_norm - mu) ./ sigma;

end