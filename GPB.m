function [theta_pred,F_model,GP_model] = GPB(X,s,y,nu,M)

%nu = 0.05;
%M = 30;

% Input:
% X: input features, n x p matrix
% s: locations, n x d matrix
% y: output values, n x 1 vector
% n: number of data points
% m: number of boosting iterations
% theta0: initial value of theta, theta(1) = sigma_1,
% nu: learning rate
% M: number of boosting iterations
% BoostType: gradient

    
% Output:
% F_hat: Predictor function, n x 1 vector
% theta_hat: Covariance parameters, n x 1 vector

% Initialize F0
n = length(y);
F0 = ones(n, 1) * mean(y);
Fm = F0;
F_tree = cell(M+1,1);
F_tree{1} = F0;
Ftree_val = zeros(n,M);
ftree_val = zeros(n,M);
RES = zeros(n,M);
theta=zeros(M,3);
modelS = cell(M,1);
    
% Main loop
for mi = 1:M
    % Update theta
    [theta(mi,:),modelS{mi}] = update_theta(s, y, Fm);

    % Compute Psi
    Psi = compute_Psi(s, y, theta(mi,:));

	% Gradient boosting step
	[fm, res, tree]= argmin_gradient_boosting(X, y, Fm, Psi);
	ftree_val(:,mi) = fm;

	% Update Fm
	Fm = Fm + nu * fm;
	Ftree_val(:,mi) = Fm;
	RES(:,mi) = res;
	F_tree{mi+1}= tree;
end

F_model = F_tree;
GP_model = modelS{end};
theta_pred = theta(end,:);

end

%% Helper functions
% Update theta using GP
function [theta_new,gpModel] = update_theta(s, y, Fm)
    y_res = y - Fm;
    gpModel = fitrgp(s, y_res, 'KernelFunction','exponential');
    kernelParams = gpModel.KernelInformation.KernelParameters;
    lengthScale = kernelParams(1);
    signalStd = kernelParams(2);
    noiseStd = gpModel.Sigma;
    theta_new = [lengthScale, signalStd, noiseStd];
end


function Psi = compute_Psi(s, y, theta)
    % Compute the covariance matrix Psi
    n = length(y);
    Sigma = compute_Sigma(s, theta);
    noiseStd = theta(3);
    Psi = Sigma + noiseStd^2 * eye(n);
end

% Compute the covariance matrix Sigma
% theta(1): signal variance
% theta(2):  characteristic length scale
function Sigma = compute_Sigma(s, theta)
    m = size(s, 1);
    rho = theta(1);
    signalStd = theta(2);
    Sigma = zeros(m, m);
    for i = 1:m
        for j = 1:m
            distance = norm(s(i, :) - s(j, :)); % Compute the Euclidean distance
            r = exp(-distance/rho);
            Sigma(i, j) = signalStd^2 * r;
        end
    end
end

% Find the function fm that minimizes the gradient boosting objective
function [fm, Pseudo_res, tree] = argmin_gradient_boosting(X, y, Fm, Psi)
    y_res = y - Fm;
    Pseudo_res = Psi \ y_res;
    %tree = buildTree(X, Pseudo_res, MaxDepth, MinLeafSize);
    
    n = length(y_res);
    prop = 0.5;
    K = n*prop;  % Number of observations to select randomly
    % Randomly select K observations
    randomIndices = datasample(1:length(X(:,1)), K, 'Replace', false);
    X_selected = X(randomIndices,:);
    y_selected = Pseudo_res(randomIndices);
    tree = fitrtree(X_selected, y_selected, 'MinLeafSize', 10);
    fm = predict(tree, X);
end