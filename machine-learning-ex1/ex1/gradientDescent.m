function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %


diff = X*theta - y;
delta1 = (1/m)*sum(diff.*X(:,1));
delta2 = (1/m)*sum(diff.*X(:,2));
theta(1) = theta(1) - alpha*delta1;
theta(2) = theta(2) - alpha*delta2;

#{
diff1 = diff;
diff2 = diff;
for i = 1:length(y)
diff1(i) = diff1(i) * X(i,1);
diff2(i) = diff2(i)*X(i,2);
end
delta1 = sum(diff1) * (1/m);
delta2 = sum(diff2) * (1/m);
#}

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
