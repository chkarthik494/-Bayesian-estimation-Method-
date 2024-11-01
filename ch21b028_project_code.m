%% PNSE TERM PROJECT
% Name: Abhiram S
% Roll no.: CH19B037

%% Question 1

% Loading data
load dataset1;
Phi = [y(2:end-1)' t(2:end-1)' -t(1:end-2)']; % regressor matrix

% Prior information on parameters
mu_a1_pri = 0.4; sigma2_a1_pri = 0.2;
mu_b1_pri = 0.5; sigma2_b1_pri = 0.2;
mu_b2_pri = 0.3; sigma2_b2_pri = 0.3;

% Sampling from priors
S = 1000; % number of parameter samples
rng(0); % fixing seed for reproducible results
a1_pri = normrnd(mu_a1_pri, sqrt(sigma2_a1_pri), 1, S);
b1_pri = normrnd(mu_b1_pri, sqrt(sigma2_b1_pri), 1, S);
b2_pri = normrnd(mu_b2_pri, sqrt(sigma2_b2_pri), 1, S);
theta_pri = [a1_pri; b1_pri; b2_pri]; % matrix of priors

% ABC rejection algorithm
yhat = Phi*theta_pri; % predictions from each parameter samples
distance = sqrt(sum((yhat - y(3:end)').^2)); % distance function: 2-norm
epsilon = 10; % tolerance on distance
selected = distance <= epsilon;
theta_post = theta_pri(:,selected);
a1_post = theta_post(1,:);
b1_post = theta_post(2,:);
b2_post = theta_post(3,:);

% Plotting histograms and fitting normal distributions
figure('Position', [0 0 1.5*560 1.5*420]);
subplot(2,2,1); histfit(a1_post); title('Posterior Distribution of a1');
subplot(2,2,2); histfit(b1_post); title('Posterior Distribution of b1');
subplot(2,2,3); histfit(b2_post); title('Posterior Distribution of b2');

% Finding sample mean and sample variance of posteriors
mu_a1_post = mean(a1_post); sigma2_a1_post = var(a1_post);
mu_b1_post = mean(b1_post); sigma2_b1_post = var(b1_post);
mu_b2_post = mean(b2_post); sigma2_b2_post = var(b2_post);

% Reduction in uncertainty calculated as percentage change in variances
delta_var = [(sigma2_a1_pri - sigma2_a1_post)/sigma2_a1_pri*100;
             (sigma2_b1_pri - sigma2_b1_post)/sigma2_b1_pri*100;
             (sigma2_b2_pri - sigma2_b2_post)/sigma2_b2_pri*100];

% Point estimate calculated as mean of posterior                    
point_est = [mu_a1_post;
             mu_b1_post;
             mu_b2_post];

% Information gain calculated using Bhattacharyya distance
info_gains = [info_gain(mu_a1_pri, mu_a1_post, sigma2_a1_pri, sigma2_a1_post);
              info_gain(mu_b1_pri, mu_b1_post, sigma2_b1_pri, sigma2_b1_post);
              info_gain(mu_b2_pri, mu_b2_post, sigma2_b2_pri, sigma2_b2_post)];

% Analysing residuals after fitting
yhat = Phi*point_est;
residuals = y(3:end)'-yhat;
subplot(2,2,4); histfit(residuals); title('Distribution of Residuals');
% Residuals seem to be Gaussian, hence the fit is good

% Collecting results in a table and displaying                
results = table(delta_var, point_est, info_gains, 'VariableNames', ...
                {'Reduction in Uncertainty', 'Point Estimate', 'Information Gain'}, ...
                'RowNames', {'a1', 'b1', 'b2'});
fprintf('PROBLEM 1 RESULTS: \n');
disp(results); fprintf('\n');

%% Question 2

% Loading data
load dataset2;
u = data2(:,1);
y = data2(:,2);

% Prior information on parameters
mu_pri = 0.5; sigma2_pri = 1;

% Number of sample points and tolerance on distance
S = 5000; epsilon_1 = 45; epsilon_2 = 60;

% First order state space model

% Sampling from priors
rng(0); % fixing seed for reproducible results
a1_pri_1 = normrnd(mu_pri, sqrt(sigma2_pri), 1, S);
b0_pri_1 = normrnd(mu_pri, sqrt(sigma2_pri), 1, S);
b1_pri_1 = normrnd(mu_pri, sqrt(sigma2_pri), 1, S);

% Declaring posteriors
a1_post_1 = zeros(size(a1_pri_1));
b0_post_1 = zeros(size(b0_pri_1));
b1_post_1 = zeros(size(b1_pri_1));

% ABC rejection algorithm
j = 0;
for i = 1:S
    a1 = a1_pri_1(i); b0 = b0_pri_1(i); b1 = b1_pri_1(i);
    A = -a1;
    B = b1 - a1*b0;
    C = 1;
    D = b0;
    sys = ss(A, B, C, D, 1); % constructing state space model
    x_init = y(1) - b0*u(1); % initial state
    yhat = lsim(sys, u, [], x_init); % simulating to get predictions
    if norm(yhat-y) <= epsilon_1 && ~(isinf(norm(yhat-y))) && ~(isnan(norm(yhat-y)))
        j = j + 1;
        a1_post_1(j) = a1; b0_post_1(j) = b0; b1_post_1(j) = b1;
    end
end
a1_post_1 = a1_post_1(1:j);
b0_post_1 = b0_post_1(1:j);
b1_post_1 = b1_post_1(1:j);

% Plotting histograms and fitting normal distributions
figure('Name', '1st Order Fit', 'Position', [0 0 1.5*560 1.5*420]);
subplot(2,2,1); histfit(a1_post_1); title('Posterior Distribution of a1');
subplot(2,2,2); histfit(b0_post_1); title('Posterior Distribution of b0');
subplot(2,2,3); histfit(b1_post_1); title('Posterior Distribution of b1');

% Finding sample mean and sample variance of posteriors
mu_a1_post_1 = mean(a1_post_1); sigma2_a1_post_1 = var(a1_post_1);
mu_b0_post_1 = mean(b0_post_1); sigma2_b0_post_1 = var(b0_post_1);
mu_b1_post_1 = mean(b1_post_1); sigma2_b1_post_1 = var(b1_post_1);

% Estimated state space model obtained using sample mean as point estimate
A_1 = -mu_a1_post_1;
B_1 = mu_b1_post_1 - mu_a1_post_1*mu_b0_post_1;
C_1 = 1;
D_1 = mu_b0_post_1;

% Using estimated state space model to analyse residuals
sys = ss(A_1, B_1, C_1, D_1, 1); % constructing state space model
x_init = y(1) - b0*u(1); % initial state
yhat_1 = lsim(sys, u, [], x_init); % simulating to get predictions
residuals_1 = y - yhat_1;
subplot(2,2,4); histfit(residuals_1); title('Distribution of Residuals');
% Residuals seem to be Gaussian, hence the fit is good
MSE_1 = mean(residuals_1.^2);

% Information gain calculated using Bhattacharyya distance
info_gains_1 = [info_gain(mu_pri, mu_a1_post_1, sigma2_pri, sigma2_a1_post_1);
                info_gain(mu_pri, mu_b0_post_1, sigma2_pri, sigma2_b0_post_1);
                info_gain(mu_pri, mu_b1_post_1, sigma2_pri, sigma2_b1_post_1)];
            
% Second order state space model

% Sampling from priors
rng(0); % fixing seed for reproducible results
a1_pri_2 = normrnd(mu_pri, sqrt(sigma2_pri), 1, S);
a2_pri_2 = normrnd(mu_pri, sqrt(sigma2_pri), 1, S);
b0_pri_2 = normrnd(mu_pri, sqrt(sigma2_pri), 1, S);
b1_pri_2 = normrnd(mu_pri, sqrt(sigma2_pri), 1, S);
b2_pri_2 = normrnd(mu_pri, sqrt(sigma2_pri), 1, S);

% Declaring posteriors
a1_post_2 = zeros(size(a1_pri_2));
a2_post_2 = zeros(size(a2_pri_2));
b0_post_2 = zeros(size(b0_pri_2));
b1_post_2 = zeros(size(b1_pri_2));
b2_post_2 = zeros(size(b2_pri_2));

% ABC rejection algorithm
j = 0;
for i = 1:S
    a1 = a1_pri_2(i); a2 = a2_pri_2(i);
    b0 = b0_pri_2(i); b1 = b1_pri_2(i); b2 = b2_pri_2(i);
    A = [-a1 1;
         -a2 0];
    B = [b1 - a1*b0;
         b2 - a2*b0];
    C = [1 0];
    D = b0;
    sys = ss(A, B, C, D, 1); % constructing state space model
    x_init = [y(1) - b0*u(1); 0]; % initial state
    yhat = lsim(sys, u, [], x_init); % simulating to get predictions
    if norm(yhat-y) <= epsilon_2 && ~(isinf(norm(yhat-y))) && ~(isnan(norm(yhat-y)))
        j = j + 1;
        a1_post_2(j) = a1; a2_post_2(j) = a2;
        b0_post_2(j) = b0; b1_post_2(j) = b1; b2_post_2(j) = b2;
    end
end
a1_post_2 = a1_post_2(1:j);
a2_post_2 = a2_post_2(1:j);
b0_post_2 = b0_post_2(1:j);
b1_post_2 = b1_post_2(1:j);
b2_post_2 = b2_post_2(1:j);

% Plotting histograms and fitting normal distributions
figure('Name', '2nd Order Fit', 'Position', [0 0 2.35*560 1.5*420]);
subplot(2,3,1); histfit(a1_post_2); title('Posterior Distribution of a1');
subplot(2,3,2); histfit(a2_post_2); title('Posterior Distribution of a2');
subplot(2,3,3); histfit(b0_post_2); title('Posterior Distribution of b0');
subplot(2,3,4); histfit(b1_post_2); title('Posterior Distribution of b1');
subplot(2,3,5); histfit(b2_post_2); title('Posterior Distribution of b2');

% Finding sample mean and sample variance of posteriors
mu_a1_post_2 = mean(a1_post_2); sigma2_a1_post_2 = var(a1_post_2);
mu_a2_post_2 = mean(a2_post_2); sigma2_a2_post_2 = var(a2_post_2);
mu_b0_post_2 = mean(b0_post_2); sigma2_b0_post_2 = var(b0_post_2);
mu_b1_post_2 = mean(b1_post_2); sigma2_b1_post_2 = var(b1_post_2);
mu_b2_post_2 = mean(b2_post_2); sigma2_b2_post_2 = var(b2_post_2);

% Estimated state space model obtained using sample mean as point estimate
A_2 = [-mu_a1_post_2 1;
       -mu_a2_post_2 0];
B_2 = [mu_b1_post_2 - mu_a1_post_2*mu_b0_post_2;
       mu_b2_post_2 - mu_a2_post_2*mu_b0_post_2];
C_2 = [1 0];
D_2 = mu_b0_post_2;

% Using estimated state space model to analyse residuals
sys = ss(A_2, B_2, C_2, D_2, 1); % constructing state space model
x_init = [y(1) - b0*u(1); 0]; % initial state
yhat_2 = lsim(sys, u, [], x_init); % simulating to get predictions
residuals_2 = y - yhat_2;
subplot(2,3,6); histfit(residuals_2); title('Distribution of Residuals');
% Residuals seem to be Gaussian, hence the fit is good
MSE_2 = mean(residuals_2.^2);

% Information gain calculated using Bhattacharyya distance
info_gains_2 = [info_gain(mu_pri, mu_a1_post_2, sigma2_pri, sigma2_a1_post_2);
                info_gain(mu_pri, mu_b0_post_2, sigma2_pri, sigma2_b0_post_2);
                info_gain(mu_pri, mu_b1_post_2, sigma2_pri, sigma2_b1_post_2)];

% Improvement in fit quantified as percentage change in MSE
fit_improvement = (MSE_1 - MSE_2)/MSE_1*100;

% Information loss from first-order fit to second-order fit calculated as
% percentage change in information gains of the two models
info_loss = (info_gains_1 - info_gains_2)./info_gains_1*100;

% Displaying results
fprintf('PROBLEM 2 RESULTS: \n');
fprintf('Estimated First-Order State Space Model: \n');
fprintf('%.4f | %.4f\n', A_1, B_1);
fprintf(' -------+-------\n');
fprintf(' %.4f | %.4f\n\n', C_1, D_1);

fprintf('Estimated Second-Order State Space Model: \n');
fprintf('%.4f %.4f |  %.4f\n', A_2(1,:), B_2(1));
fprintf('%.4f %.4f |  %.4f\n', A_2(2,:), B_2(2));
fprintf(' --------------+--------\n');
fprintf(' %.4f %.4f | %.4f\n\n', C_2, D_2);

results1 = table(MSE_1, MSE_2, fit_improvement, 'VariableNames', ...
                 {'MSE (1st Order)', 'MSE (2nd Order)', '% Reduction in MSE'});
fprintf('Improvement in Fit: \n');
disp(results1);

results2 = table(info_gains_1, info_gains_2, info_loss, 'VariableNames', ...
                 {'Info Gain (1st Order)', 'Info Gain (2nd Order)', ...
                 '% Loss in Info'}, 'RowNames', {'a1', 'b0', 'b1'});
fprintf('\nLoss in Information: \n');
disp(results2); fprintf('\n');

%% Functions

% Function for information gain                
function beta = info_gain(mu1, mu2, s21, s22)
    Bd = 1/4*(log(1/4*(s21/s22 + s22/s21 + 2)) + (mu1 - mu2)^2/(s21 + s22));
    beta = 1 - exp(-Bd);
end