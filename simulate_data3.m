%% Data generation
% linear
function [X, s, y, F] = simulate_data3(n, m, sa, sb)
    % Initial parameter
    rho=0.1;
    sigma1=1.0;
    sigma=1.0;
    
    % Generate random predictor variables
    X = rand(n,2);
    
    % Generate spatial locations
    s = sa+(sb-sa)*rand(m, 2);
    
    % Generate covariance matrix for the random effects
    Sigma = zeros(m, m);
    for i = 1:m
        for j = 1:m
            distance = norm(s(i, :) - s(j, :)); % Compute the Euclidean distance
            r = exp(-distance/rho);
            Sigma(i, j) = sigma1^2 * r;
        end
    end
    
    % Generate random effects
    b = mvnrnd(zeros(m, 1), Sigma)'; % covariance matrix Sigma
    Z = eye(n, m);
    b1 = Z * b;
    
    % Generate fixed effects
    F = zeros(n, 1);
    for i = 1:length(X)
        x1 = X(:,1);
        x2 = X(:,2);
        F(i) = 1+x1(i)+x2(i);
    end
    C = sqrt(var(F));
    F = F / C;
    
    % Generate error term
    epsilon = normrnd(0, sigma, n, 1);
    
    % Generate response variable
    y = F + b1 + epsilon;
end
