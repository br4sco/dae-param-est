%===============  *Examples of disturbance models of w(t)* ===============%
%
% We simulate independent realizations via two methods:
% (1) assuming the spectrum is 0 above a frequency w_max, then discretizing
% the frequency band [0,w_max] and using the spectral representation of the
% process to define a random continuous (and smooth) function approximating
% realizations of w(t)
% (2) discretizing the the linear filter and simulating its response to a
% white noise input
%
% Two given spectra are used. One with pole excess 4, and the other has
% pole excess 2 leading to realizations that are 2 times differentiable and
% one time differentiable, respectively.

% Mohamed Abdalmoaty
% abda@kth.se
% January 22, 2021

% ======================================================================= %
clear
close all

% == user's options
% choose model
model = 2;  % either 1 or 2
% request values at random times?
randTimes = true; % if false a uniform grid is fitted within the original one
n         = 50; %number of requested samples; used only if randTimes = true;
M         = 1;     % number of required realizations (not really used now)
dt        = 0.01;  % initial time resolution in seconds
tfinal    = 2; % length of realizations in seconds

%=========================================================================%



% == the linear filter == %
% Suppose that is Gw(s) = sigma/(a*s^2 + b*2*s +c)
% the parameters sigma, a, b, and c are known (for now)
sigma = 0.5;   % this parameter tunes the magnitude of the disturbance
omega = 4;   % natural freq. in rad/s (tunes freq. contents/fluctuations)
zeta  = 0.1; % dampling coefficient (tunes damping)
a     = 1;
b     = 2*omega*zeta;
c     = omega^2;
s     = tf('s');
Gw    = sigma/(a*s^2 + b*s +c);
% isstable(Gw) % the filter is stable




% == the spectral factor
if model == 1
    w_max = 50;           % rad/sec
    dw    = 0.01;        % rad/sec  delta w (the realizations are periodic with period 2*pi/dw)
elseif model == 2
    % use these values for model 2
    w_max = 350;
    dw    = 0.01;
end
W     = dw:dw:w_max;  % a grid of frequencies
% S(w) = G(iw) G(-iw)/ 2*pi
% approximation is periodic with period 2*pi/dw
if model == 1
    % model 1
    specFac = @(s) (sigma)./(sqrt(2*pi)*(a*s.^2 + b*s +c));
elseif model == 2
    % model 2: the one used in spectral_Monte_Carlo
    specFac = @(s) (s+1)./(sqrt(2*pi)*(1*s.^2 + sqrt(12)*s +1));%
end
spec    = (specFac(W*1i)).*specFac(-W*1i);  % S(w)
figure('position',  [200, 400, 1500, 500])
subplot(1,3,1)
loglog(W,spec); grid on
title('The spectral density of w(t)')
xlabel('angular frequecy \omega (rad/s)'); ylabel('\Phi(\omega)')



% == spectral approximation
Phi = rand(M,length(W))*2*pi; % sample random phases uniformly in [0,2*pi]
% w(t) = sum_k A_k cos(w_k t + phi_k)
% Ak = 2 sqrt(S(w_k)*dw)
% phi_k is uniform RV
w = @(t) sum(2*sqrt(dw*spec).*cos(W*t + Phi),2)'; % the process w(t)

% evaluating w(t) on a uniform time grid for inspection
T = dt:dt:tfinal;
wT = zeros(length(T),M); % allocate memory space
for i = 1:length(T)
    wT(i,:) = w(T(i));
    % each column of wT is a realization of w(t), 0<t<tfinal
end
subplot(1,3,2)
plot(T, wT(:,1)); grid on
title('A realization obtained using spectral approximation')
xlabel('time (s)'); ylabel('w(t)')



% == discrete-time model simulation
if model == 1
    % continuous-time state-space model matrices of Gw
    Aw = [0 -c;
        1 -b;];
    Bw = [sigma; 0;];
    C  = [0 1];
elseif model == 2
    % The model we tried in spectral_Monte_Carlo.m
    Aw = [0 -1;
        1 -sqrt(12);];
    Bw = [1; 1;];
    C  = [0 1];
end
% find the corresponding discrete-time matrices
F       = [-Aw             Bw*Bw';
    zeros(size(Aw))   Aw';]*dt;
expF    = expm(F);
% the indices need to be updated to work generically
Awd     = expF(3:4,3:4)';
Sigma_w = Awd* expF(1:2,3:4); % cov matrix of discrete-time nosie
Bwd     = chol(Sigma_w,'lower');

Gwd = ss(Awd,eye(2),C,0,dt); % the discret-time linear filter
Q = dlyap(Awd,Bwd*Bwd'); % initial state-covariance
% evaluating the discretized w(t) for inspection
wTd = zeros(length(T),M); % allocate memory space
x = zeros(length(T),length(Awd),M); % state trajectories

for mc = 1:M
    [wTd(:,mc),~, x(:,:,mc)] = lsim(Gwd, chol(Sigma_w,'lower')*randn(2,length(T)),[0 T(1:end-1)], chol(Q,'lower')*randn(2,1));
    % a random initial state is used; Here lsim is used for efficiency but it is not needed.
    % One can simply use the state-space equations.
end
subplot(1,3,3)
plot(T, wTd(:,1));  grid on
title('A realization simulated via a discretized model')
xlabel('time (s)'); ylabel('w(t)')



% == Generating arbitrary intersamples values using the discretized model
% For this we need to record the state trajectory
realization     = 1; % just chose to work on the first realization here, x(:,:,1)
switch randTimes
    case true
        requested_times = rand(n,1)*(tfinal-dt)  + dt;  % random values in [dt,tfinal]
        %         n               = 1; %number of requested samples;
        %         requested_times = 1.45865;  % random values in [dt,tfinal]
    case false
        requested_times = (dt+0.8*dt):10*dt:tfinal;
        n = length(requested_times);
end

% the following random variables are used to generate the required states
% I'm fixing them here so that I can use the same number later when trying
% an alternative method below
zeta = randn(2,n);

prev_x_indx      = floor(requested_times./dt);
prev_t           = prev_x_indx*dt;
dt_requested     = requested_times - prev_t;
requested_states = zeros(length(Awd),n);

for r = 1:n
    x_given          = [x(prev_x_indx(r),:,1)';  x(prev_x_indx(r)+1,:,1)';];  % given state
    expF_prev        = expm(F*(prev_t(r)/dt));
    expF_requested   = expm(F*(requested_times(r)/dt));
    expF_next        = expm(F*((prev_t(r)+dt)/dt));
    
    % the indices need to be updated to work generically
    % Note here that we can use P_prev = P_requested = P_next = Q
    % the computations as used in this loop are do not converge in general
    % due to numerical issues (!). It is used here for illustration/comparison
    P_prev = expF_prev(3:4,3:4)'*Q*expF_prev(3:4,3:4)+...
        expF_prev(3:4,3:4)'*expF_prev (1:2,3:4);
    P_requested =  expF_requested(3:4,3:4)'*Q*expF_requested(3:4,3:4)+...
        expF_requested(3:4,3:4)'*expF_requested(1:2,3:4);
    P_next = expF_next(3:4,3:4)'*Q*expF_next(3:4,3:4)+...
        expF_next(3:4,3:4)'*expF_next(1:2,3:4);
    
    cov_x_requested_given = [expm(Aw*(requested_times(r)-prev_t(r)))*P_prev...
                             P_next*expm(Aw*(prev_t(r)+dt-requested_times(r)))'];
    P_prev_next           =  P_next*Awd';
    cov_x_given           = [P_prev         P_prev_next
                             P_prev_next'   P_next];
    
    cond_mean_x_req = cov_x_requested_given*(cov_x_given\x_given);
    cond_cov_x_req  = P_requested - (cov_x_requested_given*(cov_x_given\(cov_x_requested_given')));
    
    % eigendecomposition
    [V,D] = eig((cond_cov_x_req'*cond_cov_x_req)/2);
    D(D<0) = 0; % replace negative eigenvalues by zero;
    % chol(cond_cov_x_req,'lower')
    
    % realize the state
    requested_states(:,r) =cond_mean_x_req + (V*sqrt(D))*zeta(:,r);
    % requested_states(:,r) = mvnrnd(cond_mean_x_req,cond_cov_x_req,1); % alternative way
end
requested_samples = requested_states(2,:);  % measure

subplot(1,3,3)
hold all
plot(requested_times,requested_samples,'Marker','.','MarkerSize',22,'LineStyle','none')



% another method for conditional sampling
requested_states_alt_method = zeros(length(Awd),n);
for r = 1:n
    x_given = [x(prev_x_indx(r),:,1)';  x(prev_x_indx(r)+1,:,1)';];  % given state
    
    dt1 = requested_times(r)-prev_t(r);
    dt2 = prev_t(r)+dt-requested_times(r);
    
    expF_prev    = expm(F*(dt1/dt));
    Sigma_w_prev = expF_prev(3:4,3:4)'* expF_prev(1:2,3:4); % cov matrix of discrete-time nosie
    Awd_prev     = expF_prev(3:4,3:4)';
    
    expF_next    = expm(F*(dt2/dt));
    Sigma_w_next = expF_next(3:4,3:4)'* expF_next(1:2,3:4); % cov matrix of discrete-time nosie
    Awd_next     = expF_next(3:4,3:4)';
    % the increment in the time interval where the value is requested
    Z = x_given(length(Aw)+1:end) - Awd*x_given(1:length(Aw));
    
    % cond. distribution of increment from requested time to next time on the
    % grid
    cov_Z  = (Awd_next*Sigma_w_prev*Awd_next') + Sigma_w_next;
    
    cond_mean_w_req = (Sigma_w_prev*Awd_next')*(cov_Z\Z);
    cond_cov_w_req  = Sigma_w_prev - ((Sigma_w_prev*Awd_next')*(cov_Z\((Sigma_w_prev*Awd_next')')));
    % eigendecomposition is used to find a square root of the cov matrix
    [V,D] = eig((cond_cov_w_req'*cond_cov_w_req)/2);
    D(D<0) = 0; % replace negative eigenvalues by zero;
    % chol(cond_cov_w_req,'lower') can be used if we are sure cond_cov_w_req
    % is positive definite matrix. Here, when dt1 and dt2 are small, the
    % matrix cond_cov_w_req can be singular
    increment_prev = cond_mean_w_req + (V*sqrt(D))*zeta(:,r);
    increment_next = Z - Awd_next*increment_prev;
    
    % realize the state
    requested_states_alt_method(:,r) = Awd_prev*x_given(1:length(Aw))+increment_prev;
end
requested_samples_alt_method = requested_states_alt_method(2,:);  % measure

subplot(1,3,3)
hold all
plot(requested_times,requested_samples_alt_method,'Marker','.',...
    'MarkerSize',12,'LineStyle','none', 'color','g')



% Checking validity of method algebrically. Use the last value of r
% further statistical checks can be done, but I haven't implemented any.

Z = x_given(length(Aw)+1:end) - Awd*x_given(1:length(Aw));
inc1 = requested_states(:,r) - Awd_prev*x_given(1:length(Aw));
inc2 =  x_given(length(Aw)+1:end) - Awd_next*requested_states(:,r);
% checkValues must be zero if the methods are successful
checkValue1 = Z - Awd_next*inc1 - inc2
checkValue2 = x_given(length(Aw)+1:end) - (Awd_next*(Awd_prev*x_given(1:length(Aw)) + inc1) +  inc2)
checkValue3 = x_given(length(Aw)+1:end)- (Awd_next*Awd_prev*x_given(1:length(Aw))...
    + Awd_next*increment_prev +  increment_next)












% x_{t-1} = A_{t-1} x_{t-2} + z_{t-2}
% x_{t} = A_{t|t-1} x_{t-1} + z_{t-1}
% 
% 
% x_s = A_{s|t-1} x_{t-1} + v
% x_{t} =  A_{t|s} x_s + w
% 
% 
% 
% 
% x_{t} = A_{t|s} (A_{s|t-1} x_{t-1} + v) + w
%       = A_{t|s} A_{s|t-1} x_{t-1} + A_{t|s} v + w
%       
%  {  A_{t|s} A_{s|t-1} = A_{t|t-1}  } ==>
%        
%       = A_{t|t-1} x_{t-1} + (A_{t|s} v + w)
%       
%       
%       
% z_{t-1} = x_{t} - A_{t|t-1} x_{t-1} = (A_{t|s} v + w)
% 
% 
% z_{t-1} = (A_{t|s} v + w)
% 
% 
% z_{t-1}
% v
% w          

































