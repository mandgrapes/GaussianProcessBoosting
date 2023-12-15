% linear GP
function [theta_pred,F_model,GP_model] = LGP(X,s,y)

% Train the linear model for the mean using X
linear_model = fitlm(X, y);

% Subtract the linear model predictions from y
y_res = y - predict(linear_model, X);

% Fit Gaussian Process model
gpModel = fitrgp(s, y_res, 'KernelFunction','exponential');

kernelParams = gpModel.KernelInformation.KernelParameters;
lengthScale = kernelParams(1);
signalStd = kernelParams(2);
noiseStd = gpModel.Sigma;
theta_pred = [lengthScale, signalStd, noiseStd];

F_model = linear_model;
GP_model = gpModel;

% Make predictions
y_pred = predict(gpModel, s);
F_pred = predict(linear_model, X);
y_pred_full = y_pred + F_pred;

end