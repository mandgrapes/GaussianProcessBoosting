clear
clc

rng(44)

sim1=1;
sim2=0;
sim3=0;



%% Hajjem simulation 

if sim1==1

n=500;
m=500;
sa=0;
sb=0.5;
sa_ext=0.5;
sb_ext=1.0;
nu = 0.05;
M = 30;

nr = 100;
mse_y_LFG = zeros(nr,1);
mse_y_LFG_ext = zeros(nr,1);
mse_F_LFG = zeros(nr,1);
theta_LFG = cell(nr,1);
t_LGP = zeros(nr,1);

mse_y_GPB = zeros(nr,1);
mse_y_GPB_ext = zeros(nr,1);
mse_F_GPB = zeros(nr,1);
theta_GPB = cell(nr,1);
t_GPB = zeros(nr,1);

parfor ii=1:nr

    disp(['iteration = ',num2str(ii)])
    
    % Data
    [X_train, s_train, y_train] = simulate_data1(n, m, sa, sb); % train
    [X_test, s_test, y_test, F_test] = simulate_data1(n, m, sa, sb); % test 
    [X_ext, s_ext, y_ext, ~] = simulate_data1(n, m, sa_ext, sb_ext); % extrapolation

    % Linear Gaussian Process
    tic
    % train
    [theta_pred,F_model,GP_model] = LGP(X_train,s_train,y_train);
    % test
    y_test_pred = predict(GP_model, s_test);
    F_test_pred = predict(F_model, X_test);
    y_test_pred_full = y_test_pred + F_test_pred;
   
    mse_y_LFG(ii) = sqrt(mean((y_test - y_test_pred_full).^2));
    mse_F_LFG(ii) = sqrt(mean((F_test - F_test_pred).^2));
    theta_LFG{ii} = theta_pred;
    % ext
    y_ext_pred = predict(GP_model, s_ext);
    F_ext_pred = predict(F_model, X_ext);
    y_ext_pred_full = y_ext_pred + F_ext_pred;
    mse_y_LFG_ext(ii) = sqrt(mean((y_ext - y_ext_pred_full).^2));    
    
    % time
    t_LGP(ii) = toc;

    % Gaussian Process Boosting
    tic
    % train
    [theta_pred,F_model,GP_model] = GPB(X_train,s_train,y_train,nu,M);
    % test
    y_test_pred = predict(GP_model, s_test);
    F_test_pred = Tree_predict(F_model, X_test, nu);
    y_test_pred_full = y_test_pred + F_test_pred;
   
    mse_y_GPB(ii) = sqrt(mean((y_test - y_test_pred_full).^2));
    mse_F_GPB(ii) = sqrt(mean((F_test - F_test_pred).^2));
    theta_GPB{ii} = theta_pred;
    % ext
    y_ext_pred = predict(GP_model, s_ext);
    F_ext_pred = Tree_predict(F_model, X_ext, nu);
    y_ext_pred_full = y_ext_pred + F_ext_pred;
    mse_y_GPB_ext(ii) = sqrt(mean((y_ext - y_ext_pred_full).^2));    
    
    % time
    t_GPB(ii) = toc;
    
end

theta_LFG_m = zeros(nr:3);
theta_GP_m = zeros(nr:3);
for ii=1:nr
    theta_LFG_m(ii,:) = theta_LFG{ii};  
    theta_GPB_m(ii,:) = theta_GPB{ii};  
end

% Results for Linear Gaussian Process
nmse_y_LFG = mean(mse_y_LFG)
nmse_y_LFG_ext = mean(mse_y_LFG_ext)
nmse_F_LFG = mean(mse_F_LFG)
nmse_var1_LPG = mean(sqrt(mean((theta_LFG_m(:,2).^2 - 1).^2)))
bias_var1_LPG = mean(theta_LFG_m(:,2).^2)-1
nmse_rho_LPG = mean(sqrt(mean((theta_LFG_m(:,1) - 0.1).^2)))
bias_rho_LPG = mean(theta_LFG_m(:,1))-0.1
nmse_var_LPG = mean(sqrt(mean((theta_LFG_m(:,3).^2 - 1).^2)))
bias_var_LPG = mean(theta_LFG_m(:,3).^2)-1
mean_t_LPG = mean(t_LGP)

% Results for Gaussian Process Boosting
nmse_y_GPB = mean(mse_y_GPB)
nmse_y_GPB_ext = mean(mse_y_GPB_ext)
nmse_F_GPB = mean(mse_F_GPB)
nmse_var1_GPB = mean(sqrt(mean((theta_GPB_m(:,2).^2 - 1).^2)))
bias_var1_GPB = mean(theta_GPB_m(:,2).^2)-1
nmse_rho_GPB = mean(sqrt(mean((theta_GPB_m(:,1) - 0.1).^2)))
bias_rho_GPB = mean(theta_GPB_m(:,1))-0.1
nmse_var_GPB = mean(sqrt(mean((theta_GPB_m(:,3).^2 - 1).^2)))
bias_var_GPB = mean(theta_GPB_m(:,3).^2)-1
mean_t_GPB = mean(t_GPB)

end


%% friedman3 simulation 

if sim2==1

n=500;
m=500;
sa=0;
sb=0.5;
sa_ext=0.5;
sb_ext=1.0;
nu = 0.05;
M = 30;

nr = 100;
mse_y_LFG = zeros(nr,1);
mse_y_LFG_ext = zeros(nr,1);
mse_F_LFG = zeros(nr,1);
theta_LFG = cell(nr,1);
t_LGP = zeros(nr,1);

mse_y_GPB = zeros(nr,1);
mse_y_GPB_ext = zeros(nr,1);
mse_F_GPB = zeros(nr,1);
theta_GPB = cell(nr,1);
t_GPB = zeros(nr,1);

parfor ii=1:nr

    disp(['iteration = ',num2str(ii)])
    
    % Data
    [X_train, s_train, y_train] = simulate_data2(n, m, sa, sb); % train
    [X_test, s_test, y_test, F_test] = simulate_data2(n, m, sa, sb); % test 
    [X_ext, s_ext, y_ext, ~] = simulate_data2(n, m, sa_ext, sb_ext); % extrapolation

    % Linear Gaussian Process
    tic
    % train
    [theta_pred,F_model,GP_model] = LGP(X_train,s_train,y_train);
    % test
    y_test_pred = predict(GP_model, s_test);
    F_test_pred = predict(F_model, X_test);
    y_test_pred_full = y_test_pred + F_test_pred;
   
    mse_y_LFG(ii) = sqrt(mean((y_test - y_test_pred_full).^2));
    mse_F_LFG(ii) = sqrt(mean((F_test - F_test_pred).^2));
    theta_LFG{ii} = theta_pred;
    % ext
    y_ext_pred = predict(GP_model, s_ext);
    F_ext_pred = predict(F_model, X_ext);
    y_ext_pred_full = y_ext_pred + F_ext_pred;
    mse_y_LFG_ext(ii) = sqrt(mean((y_ext - y_ext_pred_full).^2));    
    
    % time
    t_LGP(ii) = toc;

    % Gaussian Process Boosting
    tic
    % train
    [theta_pred,F_model,GP_model] = GPB(X_train,s_train,y_train,nu,M);
    % test
    y_test_pred = predict(GP_model, s_test);
    F_test_pred = Tree_predict(F_model, X_test, nu);
    y_test_pred_full = y_test_pred + F_test_pred;
   
    mse_y_GPB(ii) = sqrt(mean((y_test - y_test_pred_full).^2));
    mse_F_GPB(ii) = sqrt(mean((F_test - F_test_pred).^2));
    theta_GPB{ii} = theta_pred;
    % ext
    y_ext_pred = predict(GP_model, s_ext);
    F_ext_pred = Tree_predict(F_model, X_ext, nu);
    y_ext_pred_full = y_ext_pred + F_ext_pred;
    mse_y_GPB_ext(ii) = sqrt(mean((y_ext - y_ext_pred_full).^2));    
    
    % time
    t_GPB(ii) = toc;
    
end

theta_LFG_m = zeros(nr:3);
theta_GP_m = zeros(nr:3);
for ii=1:nr
    theta_LFG_m(ii,:) = theta_LFG{ii};  
    theta_GPB_m(ii,:) = theta_GPB{ii};  
end

% Results for Linear Gaussian Process
nmse_y_LFG = mean(mse_y_LFG)
nmse_y_LFG_ext = mean(mse_y_LFG_ext)
nmse_F_LFG = mean(mse_F_LFG)
nmse_var1_LPG = mean(sqrt(mean((theta_LFG_m(:,2).^2 - 1).^2)))
bias_var1_LPG = mean(theta_LFG_m(:,2).^2)-1
nmse_rho_LPG = mean(sqrt(mean((theta_LFG_m(:,1) - 0.1).^2)))
bias_rho_LPG = mean(theta_LFG_m(:,1))-0.1
nmse_var_LPG = mean(sqrt(mean((theta_LFG_m(:,3).^2 - 1).^2)))
bias_var_LPG = mean(theta_LFG_m(:,3).^2)-1
mean_t_LPG = mean(t_LGP)

% Results for Gaussian Process Boosting
nmse_y_GPB = mean(mse_y_GPB)
nmse_y_GPB_ext = mean(mse_y_GPB_ext)
nmse_F_GPB = mean(mse_F_GPB)
nmse_var1_GPB = mean(sqrt(mean((theta_GPB_m(:,2).^2 - 1).^2)))
bias_var1_GPB = mean(theta_GPB_m(:,2).^2)-1
nmse_rho_GPB = mean(sqrt(mean((theta_GPB_m(:,1) - 0.1).^2)))
bias_rho_GPB = mean(theta_GPB_m(:,1))-0.1
nmse_var_GPB = mean(sqrt(mean((theta_GPB_m(:,3).^2 - 1).^2)))
bias_var_GPB = mean(theta_GPB_m(:,3).^2)-1
mean_t_GPB = mean(t_GPB)

end


%% Linear simulation

if sim3==1

n=500;
m=500;
sa=0;
sb=0.5;
sa_ext=0.5;
sb_ext=1.0;
nu = 0.05;
M = 30;

nr = 100;
mse_y_LFG = zeros(nr,1);
mse_y_LFG_ext = zeros(nr,1);
mse_F_LFG = zeros(nr,1);
theta_LFG = cell(nr,1);
t_LGP = zeros(nr,1);

mse_y_GPB = zeros(nr,1);
mse_y_GPB_ext = zeros(nr,1);
mse_F_GPB = zeros(nr,1);
theta_GPB = cell(nr,1);
t_GPB = zeros(nr,1);

parfor ii=1:nr

    disp(['iteration = ',num2str(ii)])
    
    % Data
    [X_train, s_train, y_train] = simulate_data3(n, m, sa, sb); % train
    [X_test, s_test, y_test, F_test] = simulate_data3(n, m, sa, sb); % test 
    [X_ext, s_ext, y_ext, ~] = simulate_data3(n, m, sa_ext, sb_ext); % extrapolation

    % Linear Gaussian Process
    tic
    % train
    [theta_pred,F_model,GP_model] = LGP(X_train,s_train,y_train);
    % test
    y_test_pred = predict(GP_model, s_test);
    F_test_pred = predict(F_model, X_test);
    y_test_pred_full = y_test_pred + F_test_pred;
   
    mse_y_LFG(ii) = sqrt(mean((y_test - y_test_pred_full).^2));
    mse_F_LFG(ii) = sqrt(mean((F_test - F_test_pred).^2));
    theta_LFG{ii} = theta_pred;
    % ext
    y_ext_pred = predict(GP_model, s_ext);
    F_ext_pred = predict(F_model, X_ext);
    y_ext_pred_full = y_ext_pred + F_ext_pred;
    mse_y_LFG_ext(ii) = sqrt(mean((y_ext - y_ext_pred_full).^2));    
    
    % time
    t_LGP(ii) = toc;

    % Gaussian Process Boosting
    tic
    % train
    [theta_pred,F_model,GP_model] = GPB(X_train,s_train,y_train,nu,M);
    % test
    y_test_pred = predict(GP_model, s_test);
    F_test_pred = Tree_predict(F_model, X_test, nu);
    y_test_pred_full = y_test_pred + F_test_pred;
   
    mse_y_GPB(ii) = sqrt(mean((y_test - y_test_pred_full).^2));
    mse_F_GPB(ii) = sqrt(mean((F_test - F_test_pred).^2));
    theta_GPB{ii} = theta_pred;
    % ext
    y_ext_pred = predict(GP_model, s_ext);
    F_ext_pred = Tree_predict(F_model, X_ext, nu);
    y_ext_pred_full = y_ext_pred + F_ext_pred;
    mse_y_GPB_ext(ii) = sqrt(mean((y_ext - y_ext_pred_full).^2));    
    
    % time
    t_GPB(ii) = toc;
    
end

theta_LFG_m = zeros(nr:3);
theta_GP_m = zeros(nr:3);
for ii=1:nr
    theta_LFG_m(ii,:) = theta_LFG{ii};  
    theta_GPB_m(ii,:) = theta_GPB{ii};  
end

% Results for Linear Gaussian Process
nmse_y_LFG = mean(mse_y_LFG)
nmse_y_LFG_ext = mean(mse_y_LFG_ext)
nmse_F_LFG = mean(mse_F_LFG)
nmse_var1_LPG = mean(sqrt(mean((theta_LFG_m(:,2).^2 - 1).^2)))
bias_var1_LPG = mean(theta_LFG_m(:,2).^2)-1
nmse_rho_LPG = mean(sqrt(mean((theta_LFG_m(:,1) - 0.1).^2)))
bias_rho_LPG = mean(theta_LFG_m(:,1))-0.1
nmse_var_LPG = mean(sqrt(mean((theta_LFG_m(:,3).^2 - 1).^2)))
bias_var_LPG = mean(theta_LFG_m(:,3).^2)-1
mean_t_LPG = mean(t_LGP)

% Results for Gaussian Process Boosting
nmse_y_GPB = mean(mse_y_GPB)
nmse_y_GPB_ext = mean(mse_y_GPB_ext)
nmse_F_GPB = mean(mse_F_GPB)
nmse_var1_GPB = mean(sqrt(mean((theta_GPB_m(:,2).^2 - 1).^2)))
bias_var1_GPB = mean(theta_GPB_m(:,2).^2)-1
nmse_rho_GPB = mean(sqrt(mean((theta_GPB_m(:,1) - 0.1).^2)))
bias_rho_GPB = mean(theta_GPB_m(:,1))-0.1
nmse_var_GPB = mean(sqrt(mean((theta_GPB_m(:,3).^2 - 1).^2)))
bias_var_GPB = mean(theta_GPB_m(:,3).^2)-1
mean_t_GPB = mean(t_GPB)

end
