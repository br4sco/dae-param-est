function delta_robot_f!(res, dz, z, u, w, θ)
     L0 = θ[1]
     L1 = θ[2]
     L2 = θ[3]
     L3 = θ[4]
     LC1 = θ[5]
     LC2 = θ[6]
     M1 = θ[7]
     M2 = θ[8]
     M3 = θ[9]
     J1 = θ[10]
     J2 = θ[11]
     g = θ[12]
     γ = θ[13]

     q1 = z[1:3]
     q2 = z[4:6]
     q3 = z[7:9]
     v1 = z[10:12]
     v2 = z[13:15]
     v3 = z[16:18]
     dq1 = dz[1:3]
     dq2 = dz[4:6]
     dq3 = dz[7:9]
     dv1 = dz[10:12]
     dv2 = dz[13:15]
     dv3 = dz[16:18]
     κ   = z[19:24]

     b = [1.; 0.; 0.]

     # -------------------------- For q1 -------------------------
     mvec1 = zeros(3,3)
     c1_1  = zeros(3)
     c2_1  = zeros(3)
     c3_1  = zeros(3)
     c4_1  = zeros(3)
     c5_1  = zeros(3)
     c6_1  = zeros(3)
     G_1   = zeros(3)

     mvec1[1,1]=J1+LC1^2*M1+L1^2*(M2+M3)
     mvec1[1,2]=L1*(LC2*M2+L2*M3)*(cos(q1[1])*cos(q1[2])*cos(q1[3])+sin(q1[1])*sin(q1[2]))
     mvec1[1,3]=-L1*(LC2*M2+L2*M3)*cos(q1[1])*sin(q1[2])*sin(q1[3])
     mvec1[2,1]=mvec1[1,2]
     mvec1[2,2]=J2+M2*LC2^2+M3*L2^2
     mvec1[2,3]=0.0
     mvec1[3,1]=mvec1[1,3]
     mvec1[3,2]=0.0
     mvec1[3,3]=(J2+M2*LC2^2+M3*L2^2)*sin(q1[2])^2

     c1_1[1]=0.0
     c1_1[2]=L1*(LC2*M2+L2*M3)*(cos(q1[1])*sin(q1[2]) - cos(q1[2])*cos(q1[3])*sin(q1[1]))
     c1_1[3]=L1*(LC2*M2+L2*M3)*sin(q1[1])*sin(q1[2])*sin(q1[3])
     c2_1[1]=L1*(LC2*M2+L2*M3)*(cos(q1[2])*sin(q1[1]) - cos(q1[1])*cos(q1[3])*sin(q1[2]))
     c2_1[2]=0.0
     c2_1[3]=0.0
     c3_1[1]=-(L1*(LC2*M2+L2*M3)*cos(q1[3])*cos(q1[1])*sin(q1[2]))
     c3_1[2]=-(J2+LC2^2*M2+L2^2*M3)*cos(q1[2])*sin(q1[2])
     c3_1[3]=0.0
     c4_1[1]=0.0
     c4_1[2]=0.0
     c4_1[3]=0.0
     c5_1[1]=-2.0*L1*(LC2*M2+L2*M3)*cos(q1[1])*cos(q1[2])*sin(q1[3])
     c5_1[2]=0.0
     c5_1[3]=(J2+LC2^2*M2+L2^2*M3)*sin(2.0*q1[2])
     c6_1[1]=0.0
     c6_1[2]=0.0
     c6_1[3]=0.0

     G_1[1]=-g*(LC1*M1+L1*(M2+M3))*cos(q1[1])
     G_1[2]=-g*(LC2*M2+L2*M3)*cos(q1[2])*cos(q1[3])
     G_1[3]= g*(LC2*M2+L2*M3)*sin(q1[2])*sin(q1[3])

     cg1 = c1_1*v1[1]^2+c2_1*v1[2]^2+c3_1*v1[3]^2+c4_1*v1[1]*v1[2]+c5_1*v1[2]*v1[3]+c6_1*v1[1]*v1[3]+G_1+γ*v1

     # -------------------------- For q2 -------------------------
     mvec2 = zeros(3,3)
     c1_2  = zeros(3)
     c2_2  = zeros(3)
     c3_2  = zeros(3)
     c4_2  = zeros(3)
     c5_2  = zeros(3)
     c6_2  = zeros(3)
     G_2   = zeros(3)

     mvec2[1,1]=J1+LC1^2*M1+L1^2*(M2+M3)
     mvec2[1,2]=L1*(LC2*M2+L2*M3)*(cos(q2[1])*cos(q2[2])*cos(q2[3])+sin(q2[1])*sin(q2[2]))
     mvec2[1,3]=-L1*(LC2*M2+L2*M3)*cos(q2[1])*sin(q2[2])*sin(q2[3])
     mvec2[2,1]=mvec2[1,2]
     mvec2[2,2]=J2+M2*LC2^2+M3*L2^2
     mvec2[2,3]=0.0
     mvec2[3,1]=mvec2[1,3]
     mvec2[3,2]=0.0
     mvec2[3,3]=(J2+M2*LC2^2+M3*L2^2)*sin(q2[2])^2

     c1_2[1]=0.0
     c1_2[2]=L1*(LC2*M2+L2*M3)*(cos(q2[1])*sin(q2[2]) - cos(q2[2])*cos(q2[3])*sin(q2[1]))
     c1_2[3]=L1*(LC2*M2+L2*M3)*sin(q2[1])*sin(q2[2])*sin(q2[3])
     c2_2[1]=L1*(LC2*M2+L2*M3)*(cos(q2[2])*sin(q2[1])-cos(q2[1])*cos(q2[3])*sin(q2[2]))
     c2_2[2]=0.0
     c2_2[3]=0.0
     c3_2[1]=-(L1*(LC2*M2+L2*M3)*cos(q2[3])*cos(q2[1])*sin(q2[2]))
     c3_2[2]=-(J2+LC2^2*M2+L2^2*M3)*cos(q2[2])*sin(q2[2])
     c3_2[3]=0.0
     c4_2[1]=0.0
     c4_2[2]=0.0
     c4_2[3]=0.0
     c5_2[1]=-2.0*L1*(LC2*M2+L2*M3)*cos(q2[1])*cos(q2[2])*sin(q2[3])
     c5_2[2]=0.0
     c5_2[3]=(J2+LC2^2*M2+L2^2*M3)*sin(2.0*q2[2])
     c6_2[1]=0.0
     c6_2[2]=0.0
     c6_2[3]=0.0

     G_2[1]=-g*(LC1*M1+L1*(M2+M3))*cos(q2[1])
     G_2[2]=-g*(LC2*M2+L2*M3)*cos(q2[2])*cos(q2[3])
     G_2[3]= g*(LC2*M2+L2*M3)*sin(q2[2])*sin(q2[3])

     cg2 = c1_2*v2[1]^2+c2_2*v2[2]^2+c3_2*v2[3]^2+c4_2*v2[1]*v2[2]+c5_2*v2[2]*v2[3]+c6_2*v2[1]*v2[3]+G_2+ γ*v2

     # -------------------------- For q3 -------------------------
     mvec3 = zeros(3,3)
     c1_3  = zeros(3)
     c2_3  = zeros(3)
     c3_3  = zeros(3)
     c4_3  = zeros(3)
     c5_3  = zeros(3)
     c6_3  = zeros(3)
     G_3   = zeros(3)

     mvec3[1,1]=J1+LC1^2*M1+L1^2*(M2+M3)
     mvec3[1,2]=L1*(LC2*M2+L2*M3)*(cos(q3[1])*cos(q3[2])*cos(q3[3])+sin(q3[1])*sin(q3[2]))
     mvec3[1,3]=-L1*(LC2*M2+L2*M3)*cos(q3[1])*sin(q3[2])*sin(q3[3])
     mvec3[2,1]=mvec3[1,2]
     mvec3[2,2]=J2+M2*LC2^2+M3*L2^2
     mvec3[2,3]=0.0
     mvec3[3,1]=mvec3[1,3]
     mvec3[3,2]=0.0
     mvec3[3,3]=(J2+M2*LC2^2+M3*L2^2)*sin(q3[2])^2

     c1_3[1]=0.0
     c1_3[2]=L1*(LC2*M2+L2*M3)*(cos(q3[1])*sin(q3[2]) - cos(q3[2])*cos(q3[3])*sin(q3[1]))
     c1_3[3]=L1*(LC2*M2+L2*M3)*sin(q3[1])*sin(q3[2])*sin(q3[3])
     c2_3[1]=L1*(LC2*M2+L2*M3)*(cos(q3[2])*sin(q3[1])-cos(q3[1])*cos(q3[3])*sin(q3[2]))
     c2_3[2]=0.0
     c2_3[3]=0.0
     c3_3[1]=-(L1*(LC2*M2+L2*M3)*cos(q3[3])*cos(q3[1])*sin(q3[2]))
     c3_3[2]=-(J2+LC2^2*M2+L2^2*M3)*cos(q3[2])*sin(q3[2])
     c3_3[3]=0.0
     c4_3[1]=0.0
     c4_3[2]=0.0
     c4_3[3]=0.0
     c5_3[1]=-2.0*L1*(LC2*M2+L2*M3)*cos(q3[1])*cos(q3[2])*sin(q3[3])
     c5_3[2]=0.0
     c5_3[3]=(J2+LC2^2*M2+L2^2*M3)*sin(2.0*q3[2])
     c6_3[1]=0.0
     c6_3[2]=0.0
     c6_3[3]=0.0

     G_3[1]=-g*(LC1*M1+L1*(M2+M3))*cos(q3[1])
     G_3[2]=-g*(LC2*M2+L2*M3)*cos(q3[2])*cos(q3[3])
     G_3[3]= g*(LC2*M2+L2*M3)*sin(q3[2])*sin(q3[3])

     cg3 = c1_3*v3[1]^2+c2_3*v3[2]^2+c3_3*v3[3]^2+c4_3*v3[1]*v3[2]+c5_3*v3[2]*v3[3]+c6_3*v3[1]*v3[3]+G_3+ γ*v3

     # -------------------------- For psi, dpsi ----------------------------------
     # psi_1[1] = L2*sin(q1[2])*sin(q1[3]);
     # psi_1[2] = L1*cos(q1[1]) + L2*cos(q1[2]) + L0 - L3;
     # psi_1[3] = L1*sin(q1[1]) + L2*sin(q1[2])*cos(q1[3]);

     dpsi_1 = zeros(3,3)
     dpsi_1[1,1] = 0.0;
     dpsi_1[1,2] = L2*cos(q1[2])*sin(q1[3]);
     dpsi_1[1,3] = L2*sin(q1[2])*cos(q1[3]);
     dpsi_1[2,1] =-L1*sin(q1[1]);
     dpsi_1[2,2] =-L2*sin(q1[2]);
     dpsi_1[2,3] = 0.0;
     dpsi_1[3,1] = L1*cos(q1[1]);
     dpsi_1[3,2] = L2*cos(q1[2])*cos(q1[3]);
     dpsi_1[3,3] = -L2*sin(q1[2])*sin(q1[3]);

     # psi_2[1] = L2*sin(q2[2])*sin(q2[3]);
     # psi_2[2] = L1*cos(q2[1]) + L2*cos(q2[2]) + L0 - L3;
     # psi_2[3] = L1*sin(q2[1]) + L2*sin(q2[2])*cos(q2[3]);

     dpsi_2 = zeros(3,3)
     dpsi_2[1,1] = 0.0;
     dpsi_2[1,2] = L2*cos(q2[2])*sin(q2[3]);
     dpsi_2[1,3] = L2*sin(q2[2])*cos(q2[3]); # Good up to here
     dpsi_2[2,1] =-L1*sin(q2[1]);
     dpsi_2[2,2] =-L2*sin(q2[2]);
     dpsi_2[2,3] = 0.0;
     dpsi_2[3,1] = L1*cos(q2[1]);
     dpsi_2[3,2] = L2*cos(q2[2])*cos(q2[3]);
     dpsi_2[3,3] = -L2*sin(q2[2])*sin(q2[3]);

     # psi_3[1] = L2*sin(q3[2])*sin(q3[3]);
     # psi_3[2] = L1*cos(q3[1]) + L2*cos(q3[2]) + L0 - L3;
     # psi_3[3] = L1*sin(q3[1]) + L2*sin(q3[2])*cos(q3[3]);

     dpsi_3 = zeros(3,3)
     dpsi_3[1,1] = 0.0;
     dpsi_3[1,2] = L2*cos(q3[2])*sin(q3[3]);
     dpsi_3[1,3] = L2*sin(q3[2])*cos(q3[3]);
     dpsi_3[2,1] =-L1*sin(q3[1]);
     dpsi_3[2,2] =-L2*sin(q3[2]);
     dpsi_3[2,3] = 0.0;
     dpsi_3[3,1] = L1*cos(q3[1]);
     dpsi_3[3,2] = L2*cos(q3[2])*cos(q3[3]);
     dpsi_3[3,3] = -L2*sin(q3[2])*sin(q3[3]);

     # ---------------------- For last equations ---------------------

     psi(q) = [  L2*sin(q[2])*sin(q[3])
                    L1*cos(q[1])+L2*cos(q[2]) + L0 - L3;
                    L1*sin(q[1])+L2*sin(q[2])*cos(q[3])]

     # dpsi(q) = [ 0.0                L2*cos(q[2])*sin(q[3])    L2*sin(q[2])*cos(q[3])
     #             -L1*sin(q[1])      -L2*sin(q[2])             0.0
     #             L1*cos(q[1])       L2*cos(q[2])*cos(q[3])    -L2*sin(q[2])*sin(q[3])]

     Rz(ϕ) = [cos(ϕ) -sin(ϕ) 0.; sin(ϕ) cos(ϕ) 0.; 0. 0. 1.]

     h = [psi(q1) - Rz(2*pi/3)*psi(q2)
          psi(q1) - Rz(-2*pi/3)*psi(q3)]

     H = [dpsi_1 -Rz(2*pi/3)*dpsi_2           zeros(3,3)
          dpsi_1      zeros(3,3)         -Rz(-2*pi/3)*dpsi_3]

     # From Matlab symbolic toolbox
     dH = [                    0.    L2*cos(q1[2])*cos(q1[3])*v1[3] - L2*sin(q1[2])*sin(q1[3])*v1[2]    L2*cos(q1[2])*cos(q1[3])*v1[2] - L2*sin(q1[2])*sin(q1[3])*v1[3]  -L1*sin((2*pi)/3)*cos(q2[1])*v2[1]  L2*cos((2*pi)/3)*sin(q2[2])*sin(q2[3])*v2[2] - L2*cos((2*pi)/3)*cos(q2[2])*cos(q2[3])*v2[3] - L2*sin((2*pi)/3)*cos(q2[2])*v2[2]  L2*cos((2*pi)/3)*sin(q2[2])*sin(q2[3])*v2[3] - L2*cos((2*pi)/3)*cos(q2[2])*cos(q2[3])*v2[2]                                 0.                                                                                                                               0.                                                                                           0.
               -L1*cos(q1[1])*v1[1]                                               -L2*cos(q1[2])*v1[2]                                                                 0.   L1*cos((2*pi)/3)*cos(q2[1])*v2[1]  L2*cos((2*pi)/3)*cos(q2[2])*v2[2] - L2*sin((2*pi)/3)*cos(q2[2])*cos(q2[3])*v2[3] + L2*sin((2*pi)/3)*sin(q2[2])*sin(q2[3])*v2[2]  L2*sin((2*pi)/3)*sin(q2[2])*sin(q2[3])*v2[3] - L2*sin((2*pi)/3)*cos(q2[2])*cos(q2[3])*v2[2]                                 0.                                                                                                                               0.                                                                                           0.
               -L1*sin(q1[1])*v1[1]  -L2*cos(q1[3])*sin(q1[2])*v1[2] - L2*cos(q1[2])*sin(q1[3])*v1[3]  -L2*cos(q1[2])*sin(q1[3])*v1[2] - L2*cos(q1[3])*sin(q1[2])*v1[3]                 L1*sin(q2[1])*v2[1]                                                                  L2*cos(q2[3])*sin(q2[2])*v2[2] + L2*cos(q2[2])*sin(q2[3])*v2[3]                              L2*cos(q2[2])*sin(q2[3])*v2[2] + L2*cos(q2[3])*sin(q2[2])*v2[3]                                 0.                                                                                                                               0.                                                                                           0.
                              0.    L2*cos(q1[2])*cos(q1[3])*v1[3] - L2*sin(q1[2])*sin(q1[3])*v1[2]    L2*cos(q1[2])*cos(q1[3])*v1[2] - L2*sin(q1[2])*sin(q1[3])*v1[3]                                  0.                                                                                                                               0.                                                                                           0.  L1*sin((2*pi)/3)*cos(q3[1])*v3[1]  L2*sin((2*pi)/3)*cos(q3[2])*v3[2] - L2*cos((2*pi)/3)*cos(q3[2])*cos(q3[3])*v3[3] + L2*cos((2*pi)/3)*sin(q3[2])*sin(q3[3])*v3[2]  L2*cos((2*pi)/3)*sin(q3[2])*sin(q3[3])*v3[3] - L2*cos((2*pi)/3)*cos(q3[2])*cos(q3[3])*v3[2]
               -L1*cos(q1[1])*v1[1]                                               -L2*cos(q1[2])*v1[2]                                                                 0.                                  0.                                                                                                                               0.                                                                                           0.  L1*cos((2*pi)/3)*cos(q3[1])*v3[1]  L2*cos((2*pi)/3)*cos(q3[2])*v3[2] + L2*sin((2*pi)/3)*cos(q3[2])*cos(q3[3])*v3[3] - L2*sin((2*pi)/3)*sin(q3[2])*sin(q3[3])*v3[2]  L2*sin((2*pi)/3)*cos(q3[2])*cos(q3[3])*v3[2] - L2*sin((2*pi)/3)*sin(q3[2])*sin(q3[3])*v3[3]
               -L1*sin(q1[1])*v1[1]  -L2*cos(q1[3])*sin(q1[2])*v1[2] - L2*cos(q1[2])*sin(q1[3])*v1[3]  -L2*cos(q1[2])*sin(q1[3])*v1[2] - L2*cos(q1[3])*sin(q1[2])*v1[3]                                  0.                                                                                                                               0.                                                                                           0.                L1*sin(q3[1])*v3[1]                                                                  L2*cos(q3[3])*sin(q3[2])*v3[2] + L2*cos(q3[2])*sin(q3[3])*v3[3]                            L2*cos(q3[2])*sin(q3[3])*v3[2] + L2*cos(q3[3])*sin(q3[2])*v3[3]]

     myb = sparse(reshape(b, (3,1)))
     B = blockdiag(myb,myb,myb)
     mysol = Matrix([B transpose(H)])\vcat(G_1, G_2, G_3)
     utot = mysol[1:3] + u# + w.^2 

     # tau = HˆT(q) * κ + b*u
     tau1 = transpose(dpsi_1)*κ[1:3] + transpose(dpsi_1)*κ[4:6] + b*utot[1]
     tau2 = -transpose(Rz(2*pi/3)*dpsi_2)*κ[1:3] + b*utot[2]
     tau3 = -transpose(Rz(-2*pi/3)*dpsi_3)*κ[4:6] + b*utot[3]

     Minvterm = [mvec1\(tau1 - (c1_1*v1[1]^2+c2_1*v1[2]^2+c3_1*v1[3]^2+c4_1*v1[1]*v1[2]+c5_1*v1[2]*v1[3]+c6_1*v1[1]*v1[3]+G_1+ γ*v1))
                    mvec2\(tau2 - (c1_2*v2[1]^2+c2_2*v2[2]^2+c3_2*v2[3]^2+c4_2*v2[1]*v2[2]+c5_2*v2[2]*v2[3]+c6_2*v2[1]*v2[3]+G_2+ γ*v2))
                    mvec3\(tau3 - (c1_3*v3[1]^2+c2_3*v3[2]^2+c3_3*v3[3]^2+c4_3*v3[1]*v3[2]+c5_3*v3[2]*v3[3]+c6_3*v3[1]*v3[3]+G_3+ γ*v3))]

     z0 = h
     z1 = H*[v1; v2; v3]
     z2 = dH*[v1; v2; v3] + H*Minvterm
     pole = 5
     alpha1 = 2*pole
     alpha0 = pole^2

     res[1:3] = dq1 - v1
     res[4:6] = dq2 - v2
     res[7:9] = dq3 - v3
     res[10:12] = mvec1*dv1 + cg1 - tau1
     res[13:15] = mvec2*dv2 + cg2 - tau2
     res[16:18] = mvec3*dv3 + cg3 - tau3
     res[19:24] = z2 + alpha1*z1 + alpha0*z0

end

function get_delta_initial_dv(θ, u0, w0)
     L0 = θ[1]
     L1 = θ[2]
     L2 = θ[3]
     L3 = θ[4]
     LC1 = θ[5]
     LC2 = θ[6]
     M1 = θ[7]
     M2 = θ[8]
     M3 = θ[9]
     J1 = θ[10]
     J2 = θ[11]
     g = θ[12]
     γ = θ[13]

     q1 = [π/6; acos(-0.5*sqrt(3)*L1/L2 + (L3-L0)/L2); 0.0]
     q2 = [π/6; acos(-0.5*sqrt(3)*L1/L2 + (L3-L0)/L2); 0.0]
     q3 = [π/6; acos(-0.5*sqrt(3)*L1/L2 + (L3-L0)/L2); 0.0]
     v1 = zeros(3)
     v2 = zeros(3)
     v3 = zeros(3)

     b = sparse(reshape([1.; 0.; 0.], (3,1)))

     # -------------------------- For q1 -------------------------
     mvec1 = zeros(3,3)
     c1_1  = zeros(3)
     c2_1  = zeros(3)
     c3_1  = zeros(3)
     c4_1  = zeros(3)
     c5_1  = zeros(3)
     c6_1  = zeros(3)
     G_1   = zeros(3)

     mvec1[1,1]=J1+LC1^2*M1+L1^2*(M2+M3)
     mvec1[1,2]=L1*(LC2*M2+L2*M3)*(cos(q1[1])*cos(q1[2])*cos(q1[3])+sin(q1[1])*sin(q1[2]))
     mvec1[1,3]=-L1*(LC2*M2+L2*M3)*cos(q1[1])*sin(q1[2])*sin(q1[3])
     mvec1[2,1]=mvec1[1,2]
     mvec1[2,2]=J2+M2*LC2^2+M3*L2^2
     mvec1[2,3]=0.0
     mvec1[3,1]=mvec1[1,3]
     mvec1[3,2]=0.0
     mvec1[3,3]=(J2+M2*LC2^2+M3*L2^2)*sin(q1[2])^2
     mvec1 = sparse(mvec1)

     c1_1[1]=0.0
     c1_1[2]=L1*(LC2*M2+L2*M3)*(cos(q1[1])*sin(q1[2]) - cos(q1[2])*cos(q1[3])*sin(q1[1]))
     c1_1[3]=L1*(LC2*M2+L2*M3)*sin(q1[1])*sin(q1[2])*sin(q1[3])
     c2_1[1]=L1*(LC2*M2+L2*M3)*(cos(q1[2])*sin(q1[1])-cos(q1[1])*cos(q1[3])*sin(q1[2]))
     c2_1[2]=0.0
     c2_1[3]=0.0
     c3_1[1]=-(L1*(LC2*M2+L2*M3)*cos(q1[3])*cos(q1[1])*sin(q1[2]))
     c3_1[2]=-(J2+LC2^2*M2+L2^2*M3)*cos(q1[2])*sin(q1[2])
     c3_1[3]=0.0
     c4_1[1]=0.0
     c4_1[2]=0.0
     c4_1[3]=0.0
     c5_1[1]=-2.0*L1*(LC2*M2+L2*M3)*cos(q1[1])*cos(q1[2])*sin(q1[3])
     c5_1[2]=0.0
     c5_1[3]=(J2+LC2^2*M2+L2^2*M3)*sin(2.0*q1[2])
     c6_1[1]=0.0
     c6_1[2]=0.0
     c6_1[3]=0.0

     G_1[1]=-g*(LC1*M1+L1*(M2+M3))*cos(q1[1])
     G_1[2]=-g*(LC2*M2+L2*M3)*cos(q1[2])*cos(q1[3])
     G_1[3]= g*(LC2*M2+L2*M3)*sin(q1[2])*sin(q1[3])

     cg1 = c1_1*v1[1]^2+c2_1*v1[2]^2+c3_1*v1[3]^2+c4_1*v1[1]*v1[2]+c5_1*v1[2]*v1[3]+c6_1*v1[1]*v1[3]+G_1+γ*v1
     # cg1 = sparse(reshape(c1_1*v1[1]^2+c2_1*v1[2]^2+c3_1*v1[3]^2+c4_1*v1[1]*v1[2]+c5_1*v1[2]*v1[3]+c6_1*v1[1]*v1[3]+G_1+γ*v1, (3,1)))

     # -------------------------- For q2 -------------------------
     mvec2 = zeros(3,3)
     c1_2  = zeros(3)
     c2_2  = zeros(3)
     c3_2  = zeros(3)
     c4_2  = zeros(3)
     c5_2  = zeros(3)
     c6_2  = zeros(3)
     G_2   = zeros(3)

     mvec2[1,1]=J1+LC1^2*M1+L1^2*(M2+M3)
     mvec2[1,2]=L1*(LC2*M2+L2*M3)*(cos(q2[1])*cos(q2[2])*cos(q2[3])+sin(q2[1])*sin(q2[2]))
     mvec2[1,3]=-L1*(LC2*M2+L2*M3)*cos(q2[1])*sin(q2[2])*sin(q2[3])
     mvec2[2,1]=mvec2[1,2]
     mvec2[2,2]=J2+M2*LC2^2+M3*L2^2
     mvec2[2,3]=0.0
     mvec2[3,1]=mvec2[1,3]
     mvec2[3,2]=0.0
     mvec2[3,3]=(J2+M2*LC2^2+M3*L2^2)*sin(q2[2])^2
     mvec2 = sparse(mvec2)

     c1_2[1]=0.0
     c1_2[2]=L1*(LC2*M2+L2*M3)*(cos(q2[1])*sin(q2[2]) - cos(q2[2])*cos(q2[3])*sin(q2[1]))
     c1_2[3]=L1*(LC2*M2+L2*M3)*sin(q2[1])*sin(q2[2])*sin(q2[3])
     c2_2[1]=L1*(LC2*M2+L2*M3)*(cos(q2[2])*sin(q2[1])-cos(q2[1])*cos(q2[3])*sin(q2[2]))
     c2_2[2]=0.0
     c2_2[3]=0.0
     c3_2[1]=-(L1*(LC2*M2+L2*M3)*cos(q2[3])*cos(q2[1])*sin(q2[2]))
     c3_2[2]=-(J2+LC2^2*M2+L2^2*M3)*cos(q2[2])*sin(q2[2])
     c3_2[3]=0.0
     c4_2[1]=0.0
     c4_2[2]=0.0
     c4_2[3]=0.0
     c5_2[1]=-2.0*L1*(LC2*M2+L2*M3)*cos(q2[1])*cos(q2[2])*sin(q2[3])
     c5_2[2]=0.0
     c5_2[3]=(J2+LC2^2*M2+L2^2*M3)*sin(2.0*q2[2])
     c6_2[1]=0.0
     c6_2[2]=0.0
     c6_2[3]=0.0

     G_2[1]=-g*(LC1*M1+L1*(M2+M3))*cos(q2[1])
     G_2[2]=-g*(LC2*M2+L2*M3)*cos(q2[2])*cos(q2[3])
     G_2[3]= g*(LC2*M2+L2*M3)*sin(q2[2])*sin(q2[3])

     cg2 = c1_2*v2[1]^2+c2_2*v2[2]^2+c3_2*v2[3]^2+c4_2*v2[1]*v2[2]+c5_2*v2[2]*v2[3]+c6_2*v2[1]*v2[3]+G_2+ γ*v2
     # cg2 = sparse(reshape(c1_2*v2[1]^2+c2_2*v2[2]^2+c3_2*v2[3]^2+c4_2*v2[1]*v2[2]+c5_2*v2[2]*v2[3]+c6_2*v2[1]*v2[3]+G_2+ γ*v2, (3,1)))

     # -------------------------- For q3 -------------------------
     mvec3 = zeros(3,3)
     c1_3  = zeros(3)
     c2_3  = zeros(3)
     c3_3  = zeros(3)
     c4_3  = zeros(3)
     c5_3  = zeros(3)
     c6_3  = zeros(3)
     G_3   = zeros(3)

     mvec3[1,1]=J1+LC1^2*M1+L1^2*(M2+M3)
     mvec3[1,2]=L1*(LC2*M2+L2*M3)*(cos(q3[1])*cos(q3[2])*cos(q3[3])+sin(q3[1])*sin(q3[2]))
     mvec3[1,3]=-L1*(LC2*M2+L2*M3)*cos(q3[1])*sin(q3[2])*sin(q3[3])
     mvec3[2,1]=mvec3[1,2]
     mvec3[2,2]=J2+M2*LC2^2+M3*L2^2
     mvec3[2,3]=0.0
     mvec3[3,1]=mvec3[1,3]
     mvec3[3,2]=0.0
     mvec3[3,3]=(J2+M2*LC2^2+M3*L2^2)*sin(q3[2])^2
     mvec3 = sparse(mvec3)

     c1_3[1]=0.0
     c1_3[2]=L1*(LC2*M2+L2*M3)*(cos(q3[1])*sin(q3[2]) - cos(q3[2])*cos(q3[3])*sin(q3[1]))
     c1_3[3]=L1*(LC2*M2+L2*M3)*sin(q3[1])*sin(q3[2])*sin(q3[3])
     c2_3[1]=L1*(LC2*M2+L2*M3)*(cos(q3[2])*sin(q3[1])-cos(q3[1])*cos(q3[3])*sin(q3[2]))
     c2_3[2]=0.0
     c2_3[3]=0.0
     c3_3[1]=-(L1*(LC2*M2+L2*M3)*cos(q3[3])*cos(q3[1])*sin(q3[2]))
     c3_3[2]=-(J2+LC2^2*M2+L2^2*M3)*cos(q3[2])*sin(q3[2])
     c3_3[3]=0.0
     c4_3[1]=0.0
     c4_3[2]=0.0
     c4_3[3]=0.0
     c5_3[1]=-2.0*L1*(LC2*M2+L2*M3)*cos(q3[1])*cos(q3[2])*sin(q3[3])
     c5_3[2]=0.0
     c5_3[3]=(J2+LC2^2*M2+L2^2*M3)*sin(2.0*q3[2])
     c6_3[1]=0.0
     c6_3[2]=0.0
     c6_3[3]=0.0

     G_3[1]=-g*(LC1*M1+L1*(M2+M3))*cos(q3[1])
     G_3[2]=-g*(LC2*M2+L2*M3)*cos(q3[2])*cos(q3[3])
     G_3[3]= g*(LC2*M2+L2*M3)*sin(q3[2])*sin(q3[3])

     cg3 = c1_3*v3[1]^2+c2_3*v3[2]^2+c3_3*v3[3]^2+c4_3*v3[1]*v3[2]+c5_3*v3[2]*v3[3]+c6_3*v3[1]*v3[3]+G_3+ γ*v3
     # cg3 = sparse(reshape(c1_3*v3[1]^2+c2_3*v3[2]^2+c3_3*v3[3]^2+c4_3*v3[1]*v3[2]+c5_3*v3[2]*v3[3]+c6_3*v3[1]*v3[3]+G_3+ γ*v3, (3,1)))
     # -------------------------- For psi, dpsi ----------------------------------
     # psi_1[1] = L2*sin(q1[2])*sin(q1[3]);
     # psi_1[2] = L1*cos(q1[1]) + L2*cos(q1[2]) + L0 - L3;
     # psi_1[3] = L1*sin(q1[1]) + L2*sin(q1[2])*cos(q1[3]);

     dpsi_1 = zeros(3,3)
     dpsi_1[1,1] = 0.0;
     dpsi_1[1,2] = L2*cos(q1[2])*sin(q1[3]);
     dpsi_1[1,3] = L2*sin(q1[2])*cos(q1[3]);
     dpsi_1[2,1] =-L1*sin(q1[1]);
     dpsi_1[2,2] =-L2*sin(q1[2]);
     dpsi_1[2,3] = 0.0;
     dpsi_1[3,1] = L1*cos(q1[1]);
     dpsi_1[3,2] = L2*cos(q1[2])*cos(q1[3]);
     dpsi_1[3,3] = -L2*sin(q1[2])*sin(q1[3]);

     # psi_2[1] = L2*sin(q2[2])*sin(q2[3]);
     # psi_2[2] = L1*cos(q2[1]) + L2*cos(q2[2]) + L0 - L3;
     # psi_2[3] = L1*sin(q2[1]) + L2*sin(q2[2])*cos(q2[3]);

     dpsi_2 = zeros(3,3)
     dpsi_2[1,1] = 0.0;
     dpsi_2[1,2] = L2*cos(q2[2])*sin(q2[3]);
     dpsi_2[1,3] = L2*sin(q2[2])*cos(q2[3]);
     dpsi_2[2,1] =-L1*sin(q2[1]);
     dpsi_2[2,2] =-L2*sin(q2[2]);
     dpsi_2[2,3] = 0.0;
     dpsi_2[3,1] = L1*cos(q2[1]);
     dpsi_2[3,2] = L2*cos(q2[2])*cos(q2[3]);
     dpsi_2[3,3] = -L2*sin(q2[2])*sin(q2[3]);

     # psi_3[1] = L2*sin(q3[2])*sin(q3[3]);
     # psi_3[2] = L1*cos(q3[1]) + L2*cos(q3[2]) + L0 - L3;
     # psi_3[3] = L1*sin(q3[1]) + L2*sin(q3[2])*cos(q3[3]);

     dpsi_3 = zeros(3,3)
     dpsi_3[1,1] = 0.0;
     dpsi_3[1,2] = L2*cos(q3[2])*sin(q3[3]);
     dpsi_3[1,3] = L2*sin(q3[2])*cos(q3[3]);
     dpsi_3[2,1] =-L1*sin(q3[1]);
     dpsi_3[2,2] =-L2*sin(q3[2]);
     dpsi_3[2,3] = 0.0;
     dpsi_3[3,1] = L1*cos(q3[1]);
     dpsi_3[3,2] = L2*cos(q3[2])*cos(q3[3]);
     dpsi_3[3,3] = -L2*sin(q3[2])*sin(q3[3]);

     # ---------------------- For last equations ---------------------

     # psi(q) = [  L2*sin(q[2])*sin(q[3])
     #                L1*cos(q[1])+L2*cos(q[2]) + L0 - L3;
     #                L1*sin(q[1])+L2*sin(q[2])*cos(q[3])]

     dpsi(q) = [ 0.0                L2*cos(q[2])*sin(q[3])    L2*sin(q[2])*cos(q[3])
                    -L1*sin(q[1])      -L2*sin(q[2])             0.0
                    L1*cos(q[1])       L2*cos(q[2])*cos(q[3])    -L2*sin(q[2])*sin(q[3])]

     Rz(ϕ) = [cos(ϕ) -sin(ϕ) 0.; sin(ϕ) cos(ϕ) 0.; 0. 0. 1.]
     # # tau = HˆT(q) * κ + b*u0
     # tau1 = transpose(dpsi_1)*κ[1:3] + transpose(dpsi_1)*κ[4:6] + b*u0[1]
     # tau2 = -transpose(Rz(2*pi/3)*dpsi_2)*κ[1:3] + b*u0[2]
     # tau3 = -transpose(Rz(-2*pi/3)*dpsi_3)*κ[4:6] + b*u0[3]

     # Minvterm = [mvec1\(tau1 - (c1_1*v1[1]^2+c2_1*v1[2]^2+c3_1*v1[3]^2+c4_1*v1[1]*v1[2]+c5_1*v1[2]*v1[3]+c6_1*v1[1]*v1[3]+G_1+ γ*v1))
     #                mvec2\(tau2 - (c1_2*v2[1]^2+c2_2*v2[2]^2+c3_2*v2[3]^2+c4_2*v2[1]*v2[2]+c5_2*v2[2]*v2[3]+c6_2*v2[1]*v2[3]+G_2+ γ*v2))
     #                mvec3\(tau3 - (c1_3*v3[1]^2+c2_3*v3[2]^2+c3_3*v3[3]^2+c4_3*v3[1]*v3[2]+c5_3*v3[2]*v3[3]+c6_3*v3[1]*v3[3]+G_3+ γ*v3))]
     #
     # h = [psi(q1) - Rz(2*pi/3)*psi(q2)
     #      psi(q1) - Rz(-2*pi/3)*psi(q3)]

     H = [dpsi_1 -Rz(2*pi/3)*dpsi_2           zeros(3,3)
          dpsi_1      zeros(3,3)         -Rz(-2*pi/3)*dpsi_3]

     # # From Matlab symbolic toolbox
     # dH = [                    0.    L2*cos(q1[2])*cos(q1[3])*v1[3] - L2*sin(q1[2])*sin(q1[3])*v1[2]    L2*cos(q1[2])*cos(q1[3])*v1[2] - L2*sin(q1[2])*sin(q1[3])*v1[3]  -L1*sin((2*pi)/3)*cos(q2[1])*v2[1]  L2*cos((2*pi)/3)*sin(q2[2])*sin(q2[3])*v2[2] - L2*cos((2*pi)/3)*cos(q2[2])*cos(q2[3])*v2[3] - L2*sin((2*pi)/3)*cos(q2[2])*v2[2]  L2*cos((2*pi)/3)*sin(q2[2])*sin(q2[3])*v2[3] - L2*cos((2*pi)/3)*cos(q2[2])*cos(q2[3])*v2[2]                                 0.                                                                                                                               0.                                                                                           0.
     #           -L1*cos(q1[1])*v1[1]                                               -L2*cos(q1[2])*v1[2]                                                                 0.   L1*cos((2*pi)/3)*cos(q2[1])*v2[1]  L2*cos((2*pi)/3)*cos(q2[2])*v2[2] - L2*sin((2*pi)/3)*cos(q2[2])*cos(q2[3])*v2[3] + L2*sin((2*pi)/3)*sin(q2[2])*sin(q2[3])*v2[2]  L2*sin((2*pi)/3)*sin(q2[2])*sin(q2[3])*v2[3] - L2*sin((2*pi)/3)*cos(q2[2])*cos(q2[3])*v2[2]                                 0.                                                                                                                               0.                                                                                           0.
     #           -L1*sin(q1[1])*v1[1]  -L2*cos(q1[3])*sin(q1[2])*v1[2] - L2*cos(q1[2])*sin(q1[3])*v1[3]  -L2*cos(q1[2])*sin(q1[3])*v1[2] - L2*cos(q1[3])*sin(q1[2])*v1[3]                 L1*sin(q2[1])*v2[1]                                                                  L2*cos(q2[3])*sin(q2[2])*v2[2] + L2*cos(q2[2])*sin(q2[3])*v2[3]                              L2*cos(q2[2])*sin(q2[3])*v2[2] + L2*cos(q2[3])*sin(q2[2])*v2[3]                                 0.                                                                                                                               0.                                                                                           0.
     #                          0.    L2*cos(q1[2])*cos(q1[3])*v1[3] - L2*sin(q1[2])*sin(q1[3])*v1[2]    L2*cos(q1[2])*cos(q1[3])*v1[2] - L2*sin(q1[2])*sin(q1[3])*v1[3]                                  0.                                                                                                                               0.                                                                                           0.  L1*sin((2*pi)/3)*cos(q3[1])*v3[1]  L2*sin((2*pi)/3)*cos(q3[2])*v3[2] - L2*cos((2*pi)/3)*cos(q3[2])*cos(q3[3])*v3[3] + L2*cos((2*pi)/3)*sin(q3[2])*sin(q3[3])*v3[2]  L2*cos((2*pi)/3)*sin(q3[2])*sin(q3[3])*v3[3] - L2*cos((2*pi)/3)*cos(q3[2])*cos(q3[3])*v3[2]
     #           -L1*cos(q1[1])*v1[1]                                               -L2*cos(q1[2])*v1[2]                                                                 0.                                  0.                                                                                                                               0.                                                                                           0.  L1*cos((2*pi)/3)*cos(q3[1])*v3[1]  L2*cos((2*pi)/3)*cos(q3[2])*v3[2] + L2*sin((2*pi)/3)*cos(q3[2])*cos(q3[3])*v3[3] - L2*sin((2*pi)/3)*sin(q3[2])*sin(q3[3])*v3[2]  L2*sin((2*pi)/3)*cos(q3[2])*cos(q3[3])*v3[2] - L2*sin((2*pi)/3)*sin(q3[2])*sin(q3[3])*v3[3]
     #           -L1*sin(q1[1])*v1[1]  -L2*cos(q1[3])*sin(q1[2])*v1[2] - L2*cos(q1[2])*sin(q1[3])*v1[3]  -L2*cos(q1[2])*sin(q1[3])*v1[2] - L2*cos(q1[3])*sin(q1[2])*v1[3]                                  0.                                                                                                                               0.                                                                                           0.                L1*sin(q3[1])*v3[1]                                                                  L2*cos(q3[3])*sin(q3[2])*v3[2] + L2*cos(q3[2])*sin(q3[3])*v3[3]                            L2*cos(q3[2])*sin(q3[3])*v3[2] + L2*cos(q3[3])*sin(q3[2])*v3[3]]

     # -------------------------------

     M = blockdiag(mvec1,mvec2, mvec3)
     CG = vcat(cg1, cg2, cg3)
     B = blockdiag(b,b,b)
     mysol = [B transpose(H)]\vcat(G_1, G_2, G_3)
     # u = mysol[1:3] + u0[1]*ones(3) + (w0[1].^2)*ones(3)  # Assumed scalar values for u0 and w0 (or rather, used just the first element)
     u = mysol[1:3] + u0 + w0.^2    # Assumes u0 and w0 are vectors, which here are used in whole
     κ = (H*(M\transpose(H)))\( H*(M\(CG - B*u)) )
     dv = M\(transpose(H)*κ + B*u - CG)
     dκ = zeros(6)

     z0  = vcat(q1, q2, q3, v1, v2, v3, κ)
     dz0 = vcat(v1, v2, v3, dv, dκ)

     return z0, dz0
end
