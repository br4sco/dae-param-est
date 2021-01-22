% Given a continuous rational spectrum of a CT stochastic process w(t), 
% we seek to generate a realization of w
% It is assumed that the spectrum phi_w(w) is characterized via a 
% linear filter Gw with n poles and m zeros such that all poles are stable.
% The spectrum is discretized and the spectral representation is used.

% === This code is not optimized for computational efficiency === %

% Mohamed Abdalmoaty
% abda@kth.se
% January 20, 2021

% ========================================================

close all
clear

% rng(100) % uncomment to fix realization (done when used for parameter estimation)

%number of required realizations
M = 1000;
tfinal = 5; % seconds. This is the length of each realization

% Suppose that Gw(s) = (s+z1)/(a*s^2 + b*2*s +c)
% the parameters z1, a, b, and c are known (for now)
s= tf('s');
Gw = (s+1)/(s^2 + sqrt(12)*s +2);
specFac = @(s) (s+1)./(pi*(s.^2 + sqrt(12)*s +2));
% th pi factor corresponds to the spectral factor of white noise
% The Fourier transform of the Dirac delta is
% 1/(2*pi) = spectral density of white noise

% bodemag(Gw); % inspect to check frequency w_max, above which the spectrum is
% almost zero

w_max = 100; % rad/sec
dw = 0.001; %rad/sec  delta w ( the realizations are periodic with period 2*pi/dw)
W = dw:dw:w_max;  % a grid of frequencies
% value of the spectral density at the chosen frequencies
spec = (specFac(W*1i)).*specFac(-W*1i);
% loglog(W,spec)
k = length(W); % number of used frequencies
Phi = rand(M,k)*2*pi;  % these should be fixed during the whole estimation procedure
w = @(t) sum(2*sqrt(dw*spec).*cos(W*t + Phi),2)'; % the process w(t)

% =============== end =================%

% some illustrations and checks

% Evaluating w(t) on a uniform time grid for inspection
wT = zeros(tfinal,M); % allocate memory space
dt = 0.1; % seconds
T = dt:dt:tfinal;
for i = 1:length(T)
    wT(i,:) = w(T(i));  
    % each column of wT is a realization of w(t), 0<t<tfinal
end


% plot the 10th realization
figure
plot(T, wT(:,10))

figure
hist(wT(10,:))

% statistics
% figure
% plot(mean(wT)) % each realization should be zero
% check variances at different times
% var(wT(100,:))
% var(wT(500,:))
% var(wT(end,:))
var(wT(20,:))

% averages
var(wT(1,:))
var(wT(:,1))

% variance check
covar(Gw,1)*(2*pi)
[A,B,C,D] = tf2ss(Gw.num{1},Gw.den{1});
C*lyap(A,B*B')*C'*(2*pi)

spect = @(w)   (w.^2 + 1) ./ (w.^4 + 8*w.^2 + 4);
integral(spect,-inf, inf)

sum(2*dw*spec)

% comparizon with exact discrete-time method
% canonical state-space model corresponding to Gw
Aw = [0 -2;
      1 -sqrt(12);];
Bw = [1; 1;];
C = [0 1];  
F = [-Aw Bw*Bw';
    zeros(2) Aw';]*dt;
expF = expm(F);
Awd = expF(3:4,3:4)';
sigma_w = expF(3:4,3:4)'* expF(1:2,3:4);

Gwd = ss(Awd,eye(2),C,0,dt);
[P,Q] = covar(Gwd,sigma_w) % steady-state covariance of state and output
for mc = 1:M
   wTd(:,mc) = lsim(Gwd, chol(sigma_w,'lower')*randn(2,length(T)),T, chol(Q,'lower')*randn(2,1))';
end
figure(1)
hold all
plot(T,wTd(:,2))
var(wTd(50,:))
var(wTd(:,10))

figure
hist(wTd(10,:))
figure
hist(wTd(:,10))

figure
plot(T,wT)

figure
plot(T,wTd)
