% This file generates a matrix with N samples of the realization of 
% M 2-dimensional discrete-time white-noise processes.
% It's just a temporary solution, since it's probably better to implement
% disturbance generation in Julia in the end

A = [-1 0; 0 -2];
B = [1 0; 0 0.5];
C = eye(2);         % So that we get state as output

t0 = 0;            % Initial time of noise model simulation
Ts = 0.05;         % Sampling frequency of noise model
N  = 100;          % Number of simulated time steps of noise model
M = 2;             % Number of noise realizations

n = size(A, 1);
Mexp  = [A B*B'; zeros(size(A)) -A'];
MTs   = expm(Mexp*Ts);
Ad  = MTs(1:n, 1:n);
Bd2Ts = MTs(1:n, n+1:end)*Ad;
Bd    = chol(Bd2Ts);        % Might need to wrap matrices in Hermitian()

z = randn(2, N+1);
xM = nan(N+1, 2*M);
t = t0:Ts:t0+N*Ts;
sys = ss(Ad, Bd, C, zeros(size(C)), Ts);

for m = 1:M
    x = lsim(sys, z, t);
    xM(:, 2*(m-1)+1:2*m) = x;
end

writematrix(xM, 'x_mat.csv');
