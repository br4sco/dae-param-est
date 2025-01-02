L0 = 1.0
L1 = 1.5
L2 = 2.0
L3 = 0.5
LC1 = 0.75
LC2 = 1.0
M1 = 0.1
M2 = 0.1
M3 = 0.3
J1 = 0.4
J2 = 0.4
γ = 1.0
σ = 0.002                 # measurement noise variance

# TODO: Finish!

# const free_dyn_pars_true = [L0, L1, L2, L3, LC1, LC2, M1, M2, M3, J1, J2, γ]#Array{Float64}(undef, 0) # TODO: Change dyn_par_bounds if changing parameter
# const num_dyn_vars = 30#24#30
# const num_dyn_vars_adj = 33#27#33 # For adjoint method, there might be additional state variables, since outputs need to be baked into the state
# use_adjoint = true
# use_new_adj = true
# get_all_θs(pars::Vector{Float64}) = vcat(pars[1:11], [g], pars[12])#[L0, L1, L2, L3, LC1, LC2, M1, M2, M3, J1, J2, g, γ]
# # dyn_par_bounds = Array{Float64}(undef, 0, 2)
# dyn_par_bounds = hcat(fill(0.01, 12, 1), fill(1e4, 12, 1))#[0.01 1e4]#[2*(L3-L0-L2)/sqrt(3)+0.01 2*(L2+L3-L0)/sqrt(3)-0.01; 0.01 1e4; 0.01 1e4]#[0.01 1e4]
# dyn_par_bounds[3,1] = 1.0 # Setting lower bound for L2
# @warn "The learning rate dimension doesn't deal with disturbance parameters in any nice way, other info comes from W_meta, and this part is hard coded" # Oooh, what if we define what function of nx, n_in etc to use here, and in get_experiment_data that function is simply used? Instead of having to define stuff there since only then are nx and n_in defined
# # const_learning_rate = [0.1]#[0.1, 0.1, 0.1, 0.05, 0.05, 0.1, 0.02, 0.02, 0.05, 0.05, 0.05, 0.2]
# const_learning_rate = [0.1, 0.1, 0.1, 0.05, 0.05, 0.1, 0.02, 0.02, 0.05, 0.05, 0.05, 0.2]#, 0.1, 0.2, 0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05] #[0.1, 0.2, 0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05] # For disturbance model
# model_sens_to_use = delta_robot_gc_allparsens#_alldist_FAKE#delta_robot_gc_γsens    # Ah, when using _FAKE version, baseline fails because of computation of Jacobian
# # TODO: Add length assertions here in file instead of in functions? So they crash during include? Or maybe that's worse
# model_to_use = delta_robot_gc
# model_adj_to_use = delta_robot_gc_adjoint_allpar_new
# model_adj_to_use_dist_sens = delta_robot_gc_adjoint_allpar_alldist  # Old adjoint approach, i.e. not using foradj
# model_adj_to_use_dist_sens_new = delta_robot_gc_foradj_allpar_alldist
# sgd_version_to_use = perform_SGD_adam_new_deltaversion  # Needs to update bounds of L3 dynamically based on L0
# # Models for debug:
# model_stepbystep = delta_adj_stepbystep_NEW

# # Only used for adjoint debugging purposes
# FpL1 = (x, dx) -> [cos(x[1])*dx[27]+cos(x[1])*dx[30]-sin(x[1])*dx[26]-sin(x[1])*dx[29]; 0.0; 0.0; -cos(x[4])*dx[27]-(sin(x[4])*dx[26])*0.5-(sqrt(3)*sin(x[4])*dx[25])*0.5; 0.0; 0.0; (sqrt(3)*sin(x[7])*dx[28])*0.5-(sin(x[7])*dx[29])*0.5-cos(x[7])*dx[30]; 0.0; 0.0; sin(x[1])*dx[20]-cos(x[1])*dx[24]-cos(x[1])*dx[21]+sin(x[1])*dx[23]-0.0*cos(x[1])*(M2+M3)+dx[11]*(L2*M3+LC2*M2)*(sin(x[1])*sin(x[2])+cos(x[1])*cos(x[2])*cos(x[3]))+2*L1*dx[10]*(M2+M3)+x[11]^2*(L2*M3+LC2*M2)*(cos(x[2])*sin(x[1])-cos(x[1])*cos(x[3])*sin(x[2]))-cos(x[1])*sin(x[2])*sin(x[3])*dx[12]*(L2*M3+LC2*M2)-cos(x[1])*cos(x[3])*sin(x[2])*x[12]^2*(L2*M3+LC2*M2)-2*cos(x[1])*cos(x[2])*sin(x[3])*x[11]*x[12]*(L2*M3+LC2*M2); dx[10]*(L2*M3+LC2*M2)*(sin(x[1])*sin(x[2])+cos(x[1])*cos(x[2])*cos(x[3]))+x[10]^2*(L2*M3+LC2*M2)*(cos(x[1])*sin(x[2])-cos(x[2])*cos(x[3])*sin(x[1])); sin(x[1])*sin(x[2])*sin(x[3])*x[10]^2*(L2*M3+LC2*M2)-cos(x[1])*sin(x[2])*sin(x[3])*dx[10]*(L2*M3+LC2*M2); cos(x[4])*dx[21]+(sin(x[4])*dx[20])*0.5-0.0*cos(x[4])*(M2+M3)+dx[14]*(L2*M3+LC2*M2)*(sin(x[4])*sin(x[5])+cos(x[4])*cos(x[5])*cos(x[6]))+2*L1*dx[13]*(M2+M3)+x[14]^2*(L2*M3+LC2*M2)*(cos(x[5])*sin(x[4])-cos(x[4])*cos(x[6])*sin(x[5]))+(sqrt(3)*sin(x[4])*dx[19])*0.5-cos(x[4])*sin(x[5])*sin(x[6])*dx[15]*(L2*M3+LC2*M2)-cos(x[4])*cos(x[6])*sin(x[5])*x[15]^2*(L2*M3+LC2*M2)-2*cos(x[4])*cos(x[5])*sin(x[6])*x[14]*x[15]*(L2*M3+LC2*M2); dx[13]*(L2*M3+LC2*M2)*(sin(x[4])*sin(x[5])+cos(x[4])*cos(x[5])*cos(x[6]))+x[13]^2*(L2*M3+LC2*M2)*(cos(x[4])*sin(x[5])-cos(x[5])*cos(x[6])*sin(x[4])); sin(x[4])*sin(x[5])*sin(x[6])*x[13]^2*(L2*M3+LC2*M2)-cos(x[4])*sin(x[5])*sin(x[6])*dx[13]*(L2*M3+LC2*M2); cos(x[7])*dx[24]+(sin(x[7])*dx[23])*0.5-0.0*cos(x[7])*(M2+M3)+dx[17]*(L2*M3+LC2*M2)*(sin(x[7])*sin(x[8])+cos(x[7])*cos(x[8])*cos(x[9]))+2*L1*dx[16]*(M2+M3)+x[17]^2*(L2*M3+LC2*M2)*(cos(x[8])*sin(x[7])-cos(x[7])*cos(x[9])*sin(x[8]))-(sqrt(3)*sin(x[7])*dx[22])*0.5-cos(x[7])*sin(x[8])*sin(x[9])*dx[18]*(L2*M3+LC2*M2)-cos(x[7])*cos(x[9])*sin(x[8])*x[18]^2*(L2*M3+LC2*M2)-2*cos(x[7])*cos(x[8])*sin(x[9])*x[17]*x[18]*(L2*M3+LC2*M2); dx[16]*(L2*M3+LC2*M2)*(sin(x[7])*sin(x[8])+cos(x[7])*cos(x[8])*cos(x[9]))+x[16]^2*(L2*M3+LC2*M2)*(cos(x[7])*sin(x[8])-cos(x[8])*cos(x[9])*sin(x[7])); sin(x[7])*sin(x[8])*sin(x[9])*x[16]^2*(L2*M3+LC2*M2)-cos(x[7])*sin(x[8])*sin(x[9])*dx[16]*(L2*M3+LC2*M2); (sqrt(3)*cos(x[4]))*0.5; cos(x[1])+cos(x[4])*0.5; sin(x[1])-sin(x[4]); -(sqrt(3)*cos(x[7]))*0.5; cos(x[1])+cos(x[7])*0.5; sin(x[1])-sin(x[7]); -(sqrt(3)*sin(x[4])*x[13])*0.5; -sin(x[1])*x[10]-(sin(x[4])*x[13])*0.5; cos(x[1])*x[10]-cos(x[4])*x[13]; (sqrt(3)*sin(x[7])*x[16])*0.5; -sin(x[1])*x[10]-(sin(x[7])*x[16])*0.5; cos(x[1])*x[10]-cos(x[7])*x[16]; 0.0; -cos(x[1]); -sin(x[1]);;]
# FpL2 = (x, dx) -> [0.0; cos(x[2])*cos(x[3])*dx[27]-sin(x[2])*dx[29]-sin(x[2])*dx[26]+cos(x[2])*cos(x[3])*dx[30]+cos(x[2])*sin(x[3])*dx[25]+cos(x[2])*sin(x[3])*dx[28]; cos(x[3])*sin(x[2])*dx[25]+cos(x[3])*sin(x[2])*dx[28]-sin(x[2])*sin(x[3])*dx[27]-sin(x[2])*sin(x[3])*dx[30]; 0.0; dx[25]*((cos(x[5])*sin(x[6]))*0.5-(sqrt(3)*sin(x[5]))*0.5)-dx[26]*(sin(x[5])*0.5+(sqrt(3)*cos(x[5])*sin(x[6]))*0.5)-cos(x[5])*cos(x[6])*dx[27]; (cos(x[6])*sin(x[5])*dx[25])*0.5+sin(x[5])*sin(x[6])*dx[27]-(sqrt(3)*cos(x[6])*sin(x[5])*dx[26])*0.5; 0.0; dx[28]*((cos(x[8])*sin(x[9]))*0.5+(sqrt(3)*sin(x[8]))*0.5)-dx[29]*(sin(x[8])*0.5-(sqrt(3)*cos(x[8])*sin(x[9]))*0.5)-cos(x[8])*cos(x[9])*dx[30]; (cos(x[9])*sin(x[8])*dx[28])*0.5+sin(x[8])*sin(x[9])*dx[30]+(sqrt(3)*cos(x[9])*sin(x[8])*dx[29])*0.5; L1*M3*dx[11]*(sin(x[1])*sin(x[2])+cos(x[1])*cos(x[2])*cos(x[3]))+L1*M3*x[11]^2*(cos(x[2])*sin(x[1])-cos(x[1])*cos(x[3])*sin(x[2]))-L1*M3*cos(x[1])*cos(x[3])*sin(x[2])*x[12]^2-L1*M3*cos(x[1])*sin(x[2])*sin(x[3])*dx[12]-2*L1*M3*cos(x[1])*cos(x[2])*sin(x[3])*x[11]*x[12]; sin(x[2])*dx[20]+sin(x[2])*dx[23]-cos(x[2])*cos(x[3])*dx[21]-cos(x[2])*cos(x[3])*dx[24]+2*L2*M3*dx[11]-cos(x[2])*sin(x[3])*dx[19]-cos(x[2])*sin(x[3])*dx[22]+L1*M3*dx[10]*(sin(x[1])*sin(x[2])+cos(x[1])*cos(x[2])*cos(x[3]))-M3*g*cos(x[2])*cos(x[3])+L1*M3*x[10]^2*(cos(x[1])*sin(x[2])-cos(x[2])*cos(x[3])*sin(x[1]))-2*L2*M3*cos(x[2])*sin(x[2])*x[12]^2; sin(x[2])*sin(x[3])*dx[21]-cos(x[3])*sin(x[2])*dx[22]-cos(x[3])*sin(x[2])*dx[19]+sin(x[2])*sin(x[3])*dx[24]+2*L2*M3*sin(x[2])^2*dx[12]+M3*g*sin(x[2])*sin(x[3])+2*L2*M3*sin(2*x[2])*x[11]*x[12]+L1*M3*sin(x[1])*sin(x[2])*sin(x[3])*x[10]^2-L1*M3*cos(x[1])*sin(x[2])*sin(x[3])*dx[10]; L1*M3*dx[14]*(sin(x[4])*sin(x[5])+cos(x[4])*cos(x[5])*cos(x[6]))+L1*M3*x[14]^2*(cos(x[5])*sin(x[4])-cos(x[4])*cos(x[6])*sin(x[5]))-L1*M3*cos(x[4])*cos(x[6])*sin(x[5])*x[15]^2-L1*M3*cos(x[4])*sin(x[5])*sin(x[6])*dx[15]-2*L1*M3*cos(x[4])*cos(x[5])*sin(x[6])*x[14]*x[15]; dx[20]*(sin(x[5])*0.5+(sqrt(3)*cos(x[5])*sin(x[6]))*0.5)-dx[19]*((cos(x[5])*sin(x[6]))*0.5-(sqrt(3)*sin(x[5]))*0.5)+cos(x[5])*cos(x[6])*dx[21]+2*L2*M3*dx[14]+L1*M3*dx[13]*(sin(x[4])*sin(x[5])+cos(x[4])*cos(x[5])*cos(x[6]))-M3*g*cos(x[5])*cos(x[6])+L1*M3*x[13]^2*(cos(x[4])*sin(x[5])-cos(x[5])*cos(x[6])*sin(x[4]))-2*L2*M3*cos(x[5])*sin(x[5])*x[15]^2; (sqrt(3)*cos(x[6])*sin(x[5])*dx[20])*0.5-sin(x[5])*sin(x[6])*dx[21]-(cos(x[6])*sin(x[5])*dx[19])*0.5+2*L2*M3*sin(x[5])^2*dx[15]+M3*g*sin(x[5])*sin(x[6])+2*L2*M3*sin(2*x[5])*x[14]*x[15]+L1*M3*sin(x[4])*sin(x[5])*sin(x[6])*x[13]^2-L1*M3*cos(x[4])*sin(x[5])*sin(x[6])*dx[13]; L1*M3*dx[17]*(sin(x[7])*sin(x[8])+cos(x[7])*cos(x[8])*cos(x[9]))+L1*M3*x[17]^2*(cos(x[8])*sin(x[7])-cos(x[7])*cos(x[9])*sin(x[8]))-L1*M3*cos(x[7])*cos(x[9])*sin(x[8])*x[18]^2-L1*M3*cos(x[7])*sin(x[8])*sin(x[9])*dx[18]-2*L1*M3*cos(x[7])*cos(x[8])*sin(x[9])*x[17]*x[18]; dx[23]*(sin(x[8])*0.5-(sqrt(3)*cos(x[8])*sin(x[9]))*0.5)-dx[22]*((cos(x[8])*sin(x[9]))*0.5+(sqrt(3)*sin(x[8]))*0.5)+cos(x[8])*cos(x[9])*dx[24]+2*L2*M3*dx[17]+L1*M3*dx[16]*(sin(x[7])*sin(x[8])+cos(x[7])*cos(x[8])*cos(x[9]))-M3*g*cos(x[8])*cos(x[9])+L1*M3*x[16]^2*(cos(x[7])*sin(x[8])-cos(x[8])*cos(x[9])*sin(x[7]))-2*L2*M3*cos(x[8])*sin(x[8])*x[18]^2; 2*L2*M3*sin(x[8])^2*dx[18]-sin(x[8])*sin(x[9])*dx[24]-(sqrt(3)*cos(x[9])*sin(x[8])*dx[23])*0.5-(cos(x[9])*sin(x[8])*dx[22])*0.5+M3*g*sin(x[8])*sin(x[9])+2*L2*M3*sin(2*x[8])*x[17]*x[18]+L1*M3*sin(x[7])*sin(x[8])*sin(x[9])*x[16]^2-L1*M3*cos(x[7])*sin(x[8])*sin(x[9])*dx[16]; (sqrt(3)*cos(x[5]))*0.5+sin(x[2])*sin(x[3])+(sin(x[5])*sin(x[6]))*0.5; cos(x[2])+cos(x[5])*0.5-(sqrt(3)*sin(x[5])*sin(x[6]))*0.5; cos(x[3])*sin(x[2])-cos(x[6])*sin(x[5]); sin(x[2])*sin(x[3])-(sqrt(3)*cos(x[8]))*0.5+(sin(x[8])*sin(x[9]))*0.5; cos(x[2])+cos(x[8])*0.5+(sqrt(3)*sin(x[8])*sin(x[9]))*0.5; cos(x[3])*sin(x[2])-cos(x[9])*sin(x[8]); cos(x[2])*sin(x[3])*x[11]-(sqrt(3)*sin(x[5])*x[14])*0.5+cos(x[3])*sin(x[2])*x[12]+(cos(x[5])*sin(x[6])*x[14])*0.5+(cos(x[6])*sin(x[5])*x[15])*0.5; -sin(x[2])*x[11]-(sin(x[5])*x[14])*0.5-(sqrt(3)*cos(x[5])*sin(x[6])*x[14])*0.5-(sqrt(3)*cos(x[6])*sin(x[5])*x[15])*0.5; cos(x[2])*cos(x[3])*x[11]-cos(x[5])*cos(x[6])*x[14]-sin(x[2])*sin(x[3])*x[12]+sin(x[5])*sin(x[6])*x[15]; (sqrt(3)*sin(x[8])*x[17])*0.5+cos(x[2])*sin(x[3])*x[11]+cos(x[3])*sin(x[2])*x[12]+(cos(x[8])*sin(x[9])*x[17])*0.5+(cos(x[9])*sin(x[8])*x[18])*0.5; (sqrt(3)*cos(x[8])*sin(x[9])*x[17])*0.5-(sin(x[8])*x[17])*0.5-sin(x[2])*x[11]+(sqrt(3)*cos(x[9])*sin(x[8])*x[18])*0.5; cos(x[2])*cos(x[3])*x[11]-cos(x[8])*cos(x[9])*x[17]-sin(x[2])*sin(x[3])*x[12]+sin(x[8])*sin(x[9])*x[18]; -sin(x[2])*sin(x[3]); -cos(x[2]); -cos(x[3])*sin(x[2]) ;;]
# Fpγ  = (x, dx) -> [0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; x[10]; x[11]; x[12]; x[13]; x[14]; x[15]; x[16]; x[17]; x[18]; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
# deb_Fp = Fpγ

# # # If output is all three servo angles
# # f(x::Vector{Float64}) = [x[1],x[4],x[7]]    # All three servo angles
# # f_sens(x::Vector{Float64}, θ::Vector{Float64}) = [x[25],x[28],x[31]]
# # # If output is position of end effector, expressed in angles of first arm
# f(x::Vector{Float64}, θ::Vector{Float64}) = [θ[3]*sin(x[2])*sin(x[3]) #L2*sin(x[2])*sin(x[3])
#     θ[2]*cos(x[1]) + θ[3]*cos(x[2]) + θ[1] - θ[4] #L1*cos(x[1]) + L2*cos(x[2]) + L0 - L3
#     θ[2]*sin(x[1]) + θ[3]*sin(x[2])*cos(x[3])] #L1*sin(x[1]) + L2*sin(x[2])*cos(x[3])]
# # f(x::Vector{Float64}) = x[1:30]   # DEBUG

# ##################################################################################################################################################
# # f_sens should return a matrix with each row corresponding to a different output component and each column corresponding to a different parameter
# ##################################################################################################################################################

# # sans_p-part
# f_sens_base(x::Vector{Float64}, θ::Vector{Float64}, par_ind::Int)::Matrix{Float64} = 
#     [θ[3]*cos(x[2])*sin(x[3])*x[30*par_ind+2]+θ[3]*cos(x[3])*sin(x[2])*x[30*par_ind+3] #L2*cos(x[2])*sin(x[3])*x[30*par_ind+2]+L2*cos(x[3])*sin(x[2])*x[30*par_ind+3]
#     -θ[2]*sin(x[1])*x[30*par_ind+1]-θ[3]*sin(x[2])*x[30*par_ind+2] #-L1*sin(x[1])*x[30*par_ind+1]-L2*sin(x[2])*x[30*par_ind+2]
#     θ[2]*cos(x[1])*x[30*par_ind+1]+θ[3]*cos(x[2])*cos(x[3])*x[30*par_ind+2]-θ[3]*sin(x[2])*sin(x[3])*x[30*par_ind+3];;] #L1*cos(x[1])*x[30*par_ind+1]+L2*cos(x[2])*cos(x[3])*x[30*par_ind+2]-L2*sin(x[2])*sin(x[3])*x[30*par_ind+3];;]
# # p-parts
# f_sens_L0(x::Vector{Float64})::Matrix{Float64} = [0.0; 1.0; 0.0;;]
# f_sens_L1(x::Vector{Float64})::Matrix{Float64} = [0.0; cos(x[1]); sin(x[1]);;]
# f_sens_L2(x::Vector{Float64})::Matrix{Float64} = [sin(x[2])*sin(x[3]); cos(x[2]); cos(x[3])*sin(x[2]);;]
# f_sens_L3(x::Vector{Float64})::Matrix{Float64} = [0.0; -1.0; 0.0;;]
# f_sens_other(x::Vector{Float64})::Matrix{Float64} = zeros(3,1)

# # # Sensitivity wrt to L1 (currently for stabilised model). To create a column-matrix, make sure to use ;; at the end, e.g. [...;;]
# # f_sens(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = f_sens_base(x, θ, 1)+f_sens_L1(x)

# # # Sensitivity wrt to whichever individual parameter except L0, L1, L2, L3, all others are the same
# # f_sens(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = f_sens_base(x, θ, 1)+f_sens_other(x)

# # # Sensitivity wrt to [L1, M1, J1]
# # f_sens(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = hcat(f_sens_L1(x)+f_sens_base(x,θ,1), f_sens_other(x)+f_sens_base(x,θ,2), f_sens_other(x)+f_sens_base(x,θ,3))

# # # Sensitivity wrt to γ and one disturbance parameter
# # f_sens(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = [f_sens_base(x, θ, 1)+f_sens_other(x)    f_sens_base(x, θ, 2)+f_sens_other(x)]

# # Sensitivity wrt to debug2-case parameters
# # f_sens(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = hcat(f_sens_base(x, θ, 1)+f_sens_other(x), f_sens_base(x, θ, 2)+f_sens_other(x))#, f_sens_base(x, θ, 3)+f_sens_L2(x))#, 
# #     # f_sens_base(x, θ, 4)+f_sens_L3(x), f_sens_base(x, θ, 5)+f_sens_other(x), f_sens_base(x, θ, 6)+f_sens_other(x), f_sens_base(x, θ, 7)+f_sens_other(x),
# #     # f_sens_base(x, θ, 8)+f_sens_other(x), f_sens_base(x, θ, 9)+f_sens_other(x), f_sens_base(x, θ, 10)+f_sens_other(x), f_sens_base(x, θ, 11)+f_sens_other(x),
# #     # f_sens_base(x, θ, 12)+f_sens_other(x))
# # f_sens(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = hcat(f_sens_base(x, θ, 1)+f_sens_L1(x), f_sens_base(x, θ, 2)+f_sens_other(x))

# # # Sensitivity for deb1 tests
# # f_sens(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = f_sens_base(x, θ, 1)+f_sens_L2(x)

# # # Sensitivity wrt to one disturbance parameter
# # f_sens(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = f_sens_base(x, θ, 1)+f_sens_other(x)

# # Sensitivity wrt to all parameters
# f_sens(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = hcat(f_sens_base(x, θ, 1)+f_sens_L0(x), f_sens_base(x, θ, 2)+f_sens_L1(x), f_sens_base(x, θ, 3)+f_sens_L2(x), 
#     f_sens_base(x, θ, 4)+f_sens_L3(x), f_sens_base(x, θ, 5)+f_sens_other(x), f_sens_base(x, θ, 6)+f_sens_other(x), f_sens_base(x, θ, 7)+f_sens_other(x),
#     f_sens_base(x, θ, 8)+f_sens_other(x), f_sens_base(x, θ, 9)+f_sens_other(x), f_sens_base(x, θ, 10)+f_sens_other(x), f_sens_base(x, θ, 11)+f_sens_other(x),
#     f_sens_base(x, θ, 12)+f_sens_other(x))

# # # Sensitivity wrt to all disturbance parameters
# # f_sens(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = hcat(f_sens_base(x, θ, 1)+f_sens_other(x), f_sens_base(x, θ, 2)+f_sens_other(x), f_sens_base(x, θ, 3)+f_sens_other(x), 
# #     f_sens_base(x, θ, 4)+f_sens_other(x), f_sens_base(x, θ, 5)+f_sens_other(x), f_sens_base(x, θ, 6)+f_sens_other(x), f_sens_base(x, θ, 7)+f_sens_other(x),
# #     f_sens_base(x, θ, 8)+f_sens_other(x), f_sens_base(x, θ, 9)+f_sens_other(x), f_sens_base(x, θ, 10)+f_sens_other(x), f_sens_base(x, θ, 11)+f_sens_other(x),
# #     f_sens_base(x, θ, 12)+f_sens_other(x))

# # # Sensitivity wrt to all dynamical parameters AND all disturbance parameters
# # f_sens(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = hcat(f_sens_base(x, θ, 1)+f_sens_L0(x), f_sens_base(x, θ, 2)+f_sens_L1(x), f_sens_base(x, θ, 3)+f_sens_L2(x), 
# #     f_sens_base(x, θ, 4)+f_sens_L3(x), f_sens_base(x, θ, 5)+f_sens_other(x), f_sens_base(x, θ, 6)+f_sens_other(x), f_sens_base(x, θ, 7)+f_sens_other(x),
# #     f_sens_base(x, θ, 8)+f_sens_other(x), f_sens_base(x, θ, 9)+f_sens_other(x), f_sens_base(x, θ, 10)+f_sens_other(x), f_sens_base(x, θ, 11)+f_sens_other(x),
# #     f_sens_base(x, θ, 12)+f_sens_other(x), f_sens_base(x, θ, 13)+f_sens_other(x), f_sens_base(x, θ, 14)+f_sens_other(x), f_sens_base(x, θ, 15)+f_sens_other(x), 
# #     f_sens_base(x, θ, 16)+f_sens_other(x), f_sens_base(x, θ, 17)+f_sens_other(x), f_sens_base(x, θ, 18)+f_sens_other(x), f_sens_base(x, θ, 19)+f_sens_other(x),
# #     f_sens_base(x, θ, 20)+f_sens_other(x), f_sens_base(x, θ, 21)+f_sens_other(x), f_sens_base(x, θ, 22)+f_sens_other(x), f_sens_base(x, θ, 23)+f_sens_other(x),
# #     f_sens_base(x, θ, 24)+f_sens_other(x))

# # BASELINE: SHOULD NOT INCLUDE DISTURBANCE PARAMETERS, SINCE BASELINE METHOD CANNOT IDENTIFY THEM ANYWAY
# # Sensitivity wrt to all dynamical parameters
# f_sens_baseline(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = hcat(f_sens_base(x, θ, 1)+f_sens_L0(x), f_sens_base(x, θ, 2)+f_sens_L1(x), f_sens_base(x, θ, 3)+f_sens_L2(x), 
# f_sens_base(x, θ, 4)+f_sens_L3(x), f_sens_base(x, θ, 5)+f_sens_other(x), f_sens_base(x, θ, 6)+f_sens_other(x), f_sens_base(x, θ, 7)+f_sens_other(x),
# f_sens_base(x, θ, 8)+f_sens_other(x), f_sens_base(x, θ, 9)+f_sens_other(x), f_sens_base(x, θ, 10)+f_sens_other(x), f_sens_base(x, θ, 11)+f_sens_other(x),
# f_sens_base(x, θ, 12)+f_sens_other(x))
# # # Sensitivity wrt to whichever individual parameter except L0, L1, L2, L3, all others are the same
# # f_sens_baseline(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = f_sens_base(x, θ, 1)+f_sens_other(x)
# # # Sensitivity wrt whichever parameters I felt like while debugging (γ)
# # f_sens_baseline(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = hcat(f_sens_base(x, θ, 1)+f_sens_other(x))


# # # Just getting all states
# # f(x::Vector{Float64}) = x[1:24]
# # f_sens(x::Vector{Float64}, θ::Vector{Float64}) = x[1:48]
# # Since none of the state variables are the outputs, we add output sensitivites at the end. Those three extra states are e.g. needed for adjoint method.
# f_sens_deb(x::Vector{Float64}, θ::Vector{Float64}) = inject_adj_sens(x, f_sens(x, θ))
# f_debug(x::Vector{Float64}, θ::Vector{Float64}) = vcat(x[1:num_dyn_vars], f(x, θ))