% This file generates a matrix with N samples of the realization of
% M 2-dimensional discrete-time white-noise processes.
% It's just a temporary solution, since it's probably better to implement
% disturbance generation in Julia in the end

rng(123);

% 2-dimensional, pole excess 1
A = [0 -(4^2);
  1 -(2*4*0.1);];
B = [0.5; 0];
% % 3-dimensional, pole excess 2
% A = [0 0 -(4^2);
%            1 0 -(4^2+2*4*0.1);
%            0 1 -(2*4*0.1+1)]
% B = [0.5; 0.0; 0.0];
% % 4-dimensional, pole excess 4
% A = [0 0 0 -24;
%            1 0 0 -33;
%            0 1 0 -19;
%            0 0 1 -2.8]
% B = [0.5; 0.0; 0.0; 0.0]

t0 = 0;            % Initial time of noise model simulation
Ts = 0.05;         % Sampling frequency of noise model
N  = 100;          % Number of simulated time steps of noise model
M = 5000;             % Number of noise realizations

n = size(A, 1);
C = eye(2);         % So that we get state as output
Mexp  = [A B*(B'); zeros(size(A)) -A'];
MTs   = expm(Mexp*Ts);
Ad  = MTs(1:n, 1:n);
Bd2Ts = MTs(1:n, n+1:end)*(Ad');
Bd    = chol(Bd2Ts);        % Might need to wrap matrices in Hermitian()
a = norm(Ad)^2;
b = norm(Bd)^2;
% MTs   = expm(Mexp*Ts);
% Ad  = MTs(n+1:end, n+1:end)';
% Bd2Ts = Ad*MTs(1:n, n+1:end);
% Bd    = chol(Bd2Ts);        % Might need to wrap matrices in Hermitian()
% a = norm(Ad)^2;
% b = norm(Bd)^2;

xM = nan(N+1, 2*M);
t = t0:Ts:t0+N*Ts;
sys = ss(Ad, Bd, C, zeros(size(C)), Ts);

for m = 1:M
    z = randn(2, N+1);
    x = lsim(sys, z, t);
    xM(:, 2*m-1:2*m) = x;
end

% Row k in xM corresponds to sample k, every pair of columns corresponds to
% one realization of the noise

writematrix(xM, 'x_mat.csv');

%%  DEBUG
% CHECKING CONDITIONAL DISTRIBUTION

k = 20; % We only look at the k:th and k+1:th samples

% Expected value of x_{k+1} given x_k should be Ad*x_k, so
% x_{k+1}-Ad*x_{k} should be zero-mean for all realizations. If we average
% over M realizations, we should get a value close to 0. Mean should
% actually be (Ad^k+1)*x0 if we are not given x_k, but we use conditional
% mean here.


% Conditional variance of x_k should be Bd*(Bd'), for all k
xk = NaN(2, M);
xkp1 = NaN(2, M);
means_kp1 = NaN(2, M);  % Expected value of x_{k+1}, given x_{k}
for m = 1:M
    xk(:, m) = xM(k, 2*m-1:2*m)';
    means_kp1(:, m) = Ad*xk(:, m);
    xkp1(:, m) = xM(k+1, 2*m-1:2*m)';
end

deviation_kp1 = xkp1-means_kp1;
cond_mean = mean(deviation_kp1, 2);

cond_var = cov(deviation_kp1'); % Nice, very close to Bd*(Bd')

% CHECKING NON-CONDITIONAL DISTRIBUTION
xk = NaN(2, M);
xkp1 = NaN(2, M);
for m = 1:M
    xk(:, m) = xM(k, 2*m-1:2*m)';
    xkp1(:, m) = xM(k+1, 2*m-1:2*m)';
end

sk   = cov(xk');
skp1 = cov(xkp1');
