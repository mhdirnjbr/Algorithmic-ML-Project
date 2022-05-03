function p = predictOneVsAll(all_theta, X)

m = size(X, 1);
X = [ones(m, 1) X];

predict = sigmoid(X * all_theta');
[~, p] = max(predict, [], 2);

% M = max(A,[],dim) returns the largest elements along dimension dim.
% For example, if A is a matrix, then max(A,[],2) is a column vector 
% containing the maximum value of each row.

end