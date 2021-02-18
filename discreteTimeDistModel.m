%===============  *Examples of simulating disturbance w(t)* ===============%
%
% Given:
% 1) a shaping filter (spectrum)
% 2) the value x(t_k): the state corresponding to w(t_k)
% 3) time t_s > t_k
% 
% we have a function returning a simulated value w(t_s) and the corresponding
% state x(t_s)

% Mohamed Abdalmoaty
% abda@kth.se
% January 29, 2021

% ======================================================================= %
clear
% close all

% === User choices
K = 10000;  % The maximum number expected for k. 
            % that is the number of time points t_1,t_2, ..., t_k = T
            
% the state-space model is
% dx_w(t) = Aw x(t) dt + Bw dv(t)
%    w(t) = C x(t)
% A, B, and C are determinzed by the tranfer function Gw(s) = a(s)/b(s)
% that is, the coefficients of the numerator and denomenator of G
% continuous-time state-space matrices of Gw:
% @Oscar: this is the model I gave you today
Aw = [0 -(4^2);    
      1 -(2*4*0.1);];
Bw = [1; 0;];  % replace by Bw = [c; 0;]; where c is the factor tuning the variance of w
C  = [0 1];

% initial state (fixed to zero here, but can be changed in the main code)
x0 = zeros(length(Aw),1);  % the initial value of w is the second entry of x0
                          % by definition (see the matrix c). The first
                          % entry of x can be fixed to 0 or any other value
                          % as long as it is free and not required to
                          % assume a specific value to get consistent
                          % initial condition.

% generate random variables. These are to be fixed during the simulation          
% we assume here that scalar w(t)
z = randn(K,1);

% ======================================================================= %

% Illustration ( the main function is at the end of the file)

% initialization
prev_xw.x = x0; % value of the previous state
prev_xw.t = 0;  % time of the previous state
prev_xw.k = 1;  % the location on the grid

% generate random time points in [0, T]
% number of point unknown a priori
% assume max step size dt for the purpose of this illustration
T = 100; %second
dt = 0.1; % assumed max step size
time_points = 0;
current_time = 0;
while current_time < T
    current_time = current_time +rand*dt;
    time_points = [time_points;current_time; ];
end

% generate a hypothetical trajectory
w = [x0(2); zeros(length(time_points)-1,1)];
for idx = 2:length(time_points)
    next_xw =  dist_model(Aw,Bw, prev_xw, time_points(idx), z(prev_xw.k));
    w(idx) = next_xw.x(2);
    prev_xw = next_xw;
end

figure
plot(time_points, w)
xlabel('Time in seconds')
ylabel('w(t)')
grid on

% ======================================================================= %

%===  function to generate next values ===%
function next_xw = dist_model(Aw,Bw, prev_xw, t, z)
% size of Aw is n times n.
% first form F
F       = [-Aw             Bw*Bw';
            zeros(size(Aw))   Aw';]*(t-prev_xw.t);
expF    = expm(F);  % F is to be partitioned into n time n 4 blocks
Awd     = expF(length(Aw)+1:end,length(Aw)+1:end)';  % the transpose of the lower right block
Sigma_w = Awd* expF(1:length(Aw),length(Aw)+1:end); % cov matrix of discrete-time noise
%       = Awd* upper right block of expF         
Bwd     = chol(Sigma_w,'lower');

% generate next state
xw = Awd*prev_xw.x + Bwd*z;

next_xw.x = xw;           % value of the next state
next_xw.t = t;           % value of the previous state
next_xw.k = prev_xw.k+1;  % the location on the grid

end