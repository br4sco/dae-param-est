rng(123);
N = 11;
time = 0:N;
t = 6;

Ad = [0.7 0; 0 0.6];
Bd = [0.2 0; 0 0.8];
% Ad = [0.7 0.3; -0.2 0.6];
% Bd = [0.2 -0.15; -0.9 0.8];
x0 = [1; 0];
n = size(x0, 1);

% In the vector [xN, ..., x1]^T, t21([t1,..,t1]) gives the row indices of 
% xt1, ..., xtq ordered such that the ordering of the elements is the same
% as in the original vector
t2i = @(time) t2i_func(time, N, n);

z = randn(size(Bd,1), N+1);
sys = ss(Ad, Bd, eye(n), zeros(n,n));
x = lsim(sys, z, time, x0);
x = reshape(flipud(x).', [], 1);    % Puts x into a single column, with xN at the top
x = x(1:end-n, :);                  % Removes x0

x_past = x(t2i(1:(t-1)));
x_t = x(t2i(t));
x_fut = x(t2i((t+1):N));
x_tm1 = x(t2i(t-1));
x_tp1 = x(t2i(t+1));

mu = nan(N*n, 1);
sig_diag = nan(N*n, n);
for i = 1:N
    mu(t2i(i)) = (Ad^i)*x0;
    if i == 1
        sig_diag(t2i(i), :) = Bd*(Bd');
    else
        sig_diag(t2i(i), :) = Ad*sig_diag(t2i(i-1), :)*(Ad') + Bd*(Bd');
    end
end

sig = nan(N*n, N*n);
for j = 1:N
    for i = 1:N
        % i:row time, j:col time
        if i == j
            sig(t2i(i), t2i(j)) = sig_diag(t2i(i), :);
        elseif i > j
            sig(t2i(i), t2i(j)) = (Ad^(i-j))*sig_diag(t2i(j), :);
        else
            % j > i
            sig(t2i(i), t2i(j)) = ((Ad^(j-i))*sig_diag(t2i(i), :))';
        end
    end
end

mu_past = mu(t2i(1:(t-1)));
mu_t = mu(t2i(t));
mu_fut = mu(t2i((t+1):N));
mu_tm1 = mu(t2i(t-1));
mu_tp1 = mu(t2i(t-1));

sig_past = sig(t2i(1:t-1), t2i(1:t-1));
sig_t = sig(t2i(t), t2i(t));
sig_fut = sig(t2i(t+1:N), t2i(t+1:N));
sig_past_t = sig(t2i(1:t-1), t2i(t));
sig_past_fut = sig(t2i(1:t-1), t2i(t+1:N));
sig_t_fut = sig(t2i(t), t2i(t+1:N));

sig_tm1 = sig(t2i(t-1), t2i(t-1));
sig_tp1 = sig(t2i(t+1), t2i(t+1));
sig_tm1_t = sig(t2i(t-1), t2i(t));
sig_tm1_tp1 = sig(t2i(t-1), t2i(t+1));
sig_t_tp1 = sig(t2i(t), t2i(t+1));

% temp1 = [sig_past sig_past_t sig_past_fut; sig_past_t' sig_t sig_t_fut; sig_past_fut' sig_t_fut' sig_fut];
% temp2 = [sig_tm1 sig_tm1_t sig_tm1_tp1; sig_tm1_t' sig_t sig_t_tp1; sig_tm1_tp1' sig_t_tp1' sig_tp1];

m_t_tm1 = mu_t + (sig_tm1_t')* ( sig_tm1\(x_tm1-mu_tm1));
m_t_past = mu_t + (sig_past_t')* ( sig_past\(x_past - mu_past) );

m_t_tp1 = mu_t + (sig_t_tp1)* ( sig_tp1\(x_tp1-mu_tp1));
m_t_fut = mu_t + (sig_t_fut)* ( sig_fut\(x_fut - mu_fut) );

% z represents x_tp1 and x_tm1, y represents x_past and x_fut
x_z = [x_tp1; x_tm1];
mu_z = [mu_tp1; mu_tm1];
sig_t_z = [sig_t_tp1 sig_tm1_t'];
sig_z = [sig_tp1 sig_tm1_tp1'; sig_tm1_tp1 sig_tm1];
m_t_z = mu_t + sig_t_z* ( sig_z\(x_z - mu_z));

x_y = [x_fut; x_past];
mu_y = [mu_fut; mu_past];
sig_t_y = [sig_t_fut sig_past_t'];
sig_y = [sig_fut sig_past_fut'; sig_past_fut sig_past];
m_t_y = mu_t + sig_t_y* ( sig_y\(x_y - mu_y));


past_mean_y = sig_t_y/sig_y;
part_mean_z = sig_t_z/sig_z;

exp1 = sig_t_fut - (sig_past_t')*( sig_past\sig_past_fut );
exp2 = sig_fut - (sig_past_fut')*( sig_past\sig_past_fut);

% I'm not exactly sure what I was doing here...
% mat1 = sig_t_fut - (sig_past_t')*( sig_past\sig_past_fut );
% mat2 = sig_fut - (sig_past_fut')*( sig_fut\(sig_past_fut')) - sig_past;
% test_mat = mat1/mat2;
