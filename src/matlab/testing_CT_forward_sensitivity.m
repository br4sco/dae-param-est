theta = -1;
Ts = 0.01;
eps = 0.001;

%% Sensitivity first, discretization later

A1 = [theta 0; 1 theta];
B1 = [1 0; 0 0];
C1 = eye(2);

[A1d, B1d] = discretize_CT_ss(A1, B1, Ts, eps);

%% Discretization first, sensitivity later

A2 = theta;
B2 = 1;
C2 = 1;

[A2d, B2d] = discretize_CT_ss(A2, B2, Ts, eps);