A = [-1 0; 0 -2];
B = [1 0; 0 0.5];
C = eye(2);         % So that we get state as output

t0 = 0;            % Initial time of noise model simulation
Ts = 0.05;         % Sampling frequency of noise model
N  = 100;          % Number of simulated time steps of noise model
M = 2;             % Number of noise realizations

Ad = expm(A*Ts);
Bd = ( A\(Ad - eye(size(Ad))) )*B;

z = randn(2, N+1);
xM = nan(N+1, 2*M);
t = t0:Ts:t0+N*Ts;
sys = ss(Ad, Bd, C, zeros(size(C)), Ts);

for m = 1:M
    x = lsim(sys, z, t);
    xM(:, 2*(m-1)+1:2*m) = x;
end

writematrix(xM, 'x_mat.csv');
