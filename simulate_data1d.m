%% Data generation
% hajjem 1d
function [X, s, y, F] = simulate_data1d(n, m, sa, sb)
    % Initial parameter
    rho=0.1;
    sigma1=1;
    sigma=1;
   
    % Generate random predictor variables
    X = normrnd(0, 1, n, 1);
    
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
        x1 = X(i);
        %x2 = X(i,2);
        %x3 = X(i,3);
        %F(i) = 2 * X(i) + X(i)^2 + 4 * (X(i) > 0) + 2 * log(abs(X(i))) * X(i);
        %F(i) = 2 * x1 + x2^2 + 4 * (x3 > 0) + 2 * log(abs(x1)) * x3;
        F(i) = 2 * x1 + x1^2 + 4 * (x1 > 0) + 2 * log(abs(x1)) * x1;
    end
    C = sqrt(var(F));
    F = F / C;
    
    % Generate error term
    epsilon = normrnd(0, sigma, n, 1);
    
    % Generate response variable
    y = F + b1 + epsilon;
end
