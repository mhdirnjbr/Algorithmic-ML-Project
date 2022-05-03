function [J ,grad] = neuralNetworkCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


m = size(X, 1);
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


% Forward propagation

X = [ones(m,1) X]; % Adding 1 as first column in X
a1 = X; 
z2 = Theta1 * a1'; 
a2 = sigmoid(z2); 

a2 = [ones(m,1) a2'];

z3 = Theta2 * a2';
a3 = sigmoid(z3);

y_hat = a3;


% Creating the one-hot vector of labels

y_new = zeros(num_labels, m);
for i=1:m
  y_new(y(i),i)=1;
end

% Cost function
J = (1/m) * sum ( sum ( (-y_new) .* log(y_hat) - (1-y_new) .* log(1-y_hat)));

% We do not regularize the terms that correspond to the bias
t1 = Theta1(:,2:size(Theta1,2));
t2 = Theta2(:,2:size(Theta2,2));

% Regularization
reg_term = lambda  * (sum( sum ( t1.^ 2 )) + sum( sum ( t2.^ 2 ))) / (2*m);

% Regularized cost function
J = J + reg_term;



% Back propagation

for t=1:m

    % Step 1
	a1 = X(t,:); % (1 x 401)
    a1 = a1'; % (401 x 1)
	z2 = Theta1 * a1; % (25 x 401) * (401 x 1)
	a2 = sigmoid(z2); % (25 x 1)
    
    a2 = [1 ; a2]; % adding a bias (26 x 1)
	z3 = Theta2 * a2; % (10 x 26) * (26 x 1)
	a3 = sigmoid(z3); % a3 = y_hat (10 x 1)
    
    % Step 2
	delta_3 = a3 - y_new(:,t); % (10 x 1)
	
    z2 = [1; z2]; % adding a bias (26 x 1)

    % Step 3
    delta_2 = (Theta2' * delta_3) .* sigmoidDerivate(z2); % ((26 x 10) * (10 x 1))=(26 x 1)

    % Step 4
	delta_2 = delta_2(2:end); % skipping delta_2(0) (25 x 1)


	Theta2_grad = Theta2_grad + delta_3 * a2'; % (10 x 1) * (1 x 26)
	Theta1_grad = Theta1_grad + delta_2 * a1'; % (25 x 1) * (1 x 401)
    
end

% Step 5
Theta2_grad = (1/m) * Theta2_grad; % (10 x 26)
Theta1_grad = (1/m) * Theta1_grad; % (25 x 401)



% Regularization

% Theta1_grad(:, 1) = Theta1_grad(:, 1) ./ m; % for j = 0

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end)); % for j >= 1 

% Theta2_grad(:, 1) = Theta2_grad(:, 1) ./ m; % for j = 0

Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end)); % for j >= 1


% Unroll gradients

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end