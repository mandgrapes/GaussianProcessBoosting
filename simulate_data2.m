%% Data generation
% friedman3
function [X, s, y, F] = simulate_data2(n, m, sa, sb)
    % Initial parameter
    rho=0.1;
    sigma1=1;
    sigma=1;
    
    
    % Generate random predictor variables
    a = 0;
    b = 100;
    x1 = a+(b-a)*rand(n,1);
    
    a = 40*pi;
    b = 560*pi;
    x2 = a+(b-a)*rand(n,1);    
    
    a = 0;
    b = 1;
    x3 = a+(b-a)*rand(n,1);        

    a = 1;
    b = 11;
    x4 = a+(b-a)*rand(n,1);      
    
    X = [x1,x2,x3,x4];
    
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
        temp = x2(i)*x3(i)-1-1/(x2(i)*x4(i));
        F(i) = atan(temp/x1(i));
    end
    C = sqrt(var(F));
    F = F / C;
    
    % Generate error term
    epsilon = normrnd(0, sigma, n, 1);
    
    % Generate response variable
    y = F + b1 + epsilon;
end