clear
clc

rng(44)

n=500;
m=500;
sa=0;
sb=0.5;
sa_ext=0.5;
sb_ext=1.0;
nu = 0.05;
M = 30;

[X_train, s_train, y_train, F_train] = simulate_data1d(n, m, sa, sb); % train
[X_test, s_test, y_test, F_test] = simulate_data1d(n, m, sa, sb); % test 
[X_ext, s_ext, y_ext, F_ext] = simulate_data1d(n, m, sa_ext, sb_ext); % extrapolation

% Linear Gaussian Process
% train
[~,F_model,GP_model] = LGP(X_train,s_train,y_train);
y_train_pred_LG = predict(GP_model, s_train);
F_train_pred_LG = predict(F_model, X_train);
y_train_pred_full_LG = y_train_pred_LG + F_train_pred_LG;

% test
y_test_pred_LG = predict(GP_model, s_test);
F_test_pred_LG = predict(F_model, X_test);
y_test_pred_full_LG = y_test_pred_LG + F_test_pred_LG;


% Gaussian Process Boosting
% train
[~,F_model,GP_model] = GPB(X_train,s_train,y_train,nu,M);
y_train_pred_GBP = predict(GP_model, s_train);
F_train_pred_GBP = Tree_predict(F_model, X_train, nu);
y_train_pred_full_GBP = y_train_pred_GBP + F_train_pred_GBP;

% test
y_test_pred_GBP = predict(GP_model, s_test);
F_test_pred_GBP = Tree_predict(F_model, X_test, nu);
y_test_pred_full_GBP = y_test_pred_GBP + F_test_pred_GBP;


%% Plot for training data
figure(1)
subplot(3,1,1)
hold on
plot(X_train,y_train,'.')
plot(X_train,F_train,'.')
hold off
xlabel('X')
ylabel('y or F')
legend({'y','F'})
title('Ground truth')

subplot(3,1,2)
hold on
plot(X_train,y_train_pred_full_LG,'.')
plot(X_train,F_train_pred_LG,'.')
hold off
xlabel('X')
ylabel('y or F')
legend({'y','F'})
title('Linear Gaussian Process')


subplot(3,1,3)
hold on
plot(X_train,y_train_pred_full_GBP,'.')
plot(X_train,F_train_pred_GBP,'.')
hold off
xlabel('X')
ylabel('y or F')
legend({'y','F'})
title('Gaussian Process Boosting')



%% Plot for test data
figure(2)
subplot(3,1,1)
hold on
plot(X_test,y_test,'.')
plot(X_test,F_test,'.')
hold off
xlabel('X')
ylabel('y or F')
legend({'y','F'})
title('Ground truth')

subplot(3,1,2)
hold on
plot(X_test,y_test_pred_full_LG,'.')
plot(X_test,F_test_pred_LG,'.')
hold off
xlabel('X')
ylabel('y or F')
legend({'y','F'})
title('Linear Gaussian Process')


subplot(3,1,3)
hold on
plot(X_test,y_test_pred_full_GBP,'.')
plot(X_test,F_test_pred_GBP,'.')
hold off
xlabel('X')
ylabel('y or F')
legend({'y','F'})
title('Gaussian Process Boosting')





