import Interpolations.CubicSplineInterpolation
import Interpolations.LinearInterpolation
import Random

using DifferentialEquations
using Sundials
using DelimitedFiles
using Plots
using ProgressMeter
using SparseArrays

include("delta_robot_helper.jl")
# include("noise_model.jl")

struct Model
  f!::Function                  # residual function
  jac!::Function               # jacobian function
  x0::Vector{Float64}         # initial values of x
  dx0::Vector{Float64}        # initial values of x'
  dvars::Array{Bool, 1}         # bool array indicating differential variables
  ic_check::Vector{Float64}   # residual at t0 (DEBUG)
end

struct Model_ode
    f::Function             # ODE Function
    x0::Vector{Float64}   # initial values of x
end

function interpolation(T::Float64, xs::Vector{Float64})
  let ts = range(0, T, length=length(xs))
    LinearInterpolation(ts, xs)
  end
end

function delta_robot(Φ::Float64, u::Function, w::Function, θ::Vector{Float64})::Model
    let L0 = θ[1], L1 = θ[2], L2 = θ[3], L3 = θ[4], LC1 = θ[5], LC2 = θ[6], M1 = θ[7], M2 = θ[8], M3 = θ[9], J1 = θ[10], J2 = θ[11], g = θ[12], γ = θ[13]
        pole = 5
        α1 = 2*pole
        α0 = pole^2
        function f!(res, dz, z, _, t)
            # ug_lg = BHT\G
            ug_lg = [
                1   0.0   0.0   0.0   -L1*sin(z[1])   L1*cos(z[1])   0.0   -L1*sin(z[1])   L1*cos(z[1])
                0.0   0.0   0.0   L2*cos(z[2])*sin(z[3])   -L2*sin(z[2])   L2*cos(z[2])*cos(z[3])   L2*cos(z[2])*sin(z[3])   -L2*sin(z[2])   L2*cos(z[2])*cos(z[3])
                0.0   0.0   0.0   L2*cos(z[3])*sin(z[2])   0.0   -L2*sin(z[2])*sin(z[3])   L2*cos(z[3])*sin(z[2])   0.0   -L2*sin(z[2])*sin(z[3])
                0.0   1   0.0   -(sqrt(3)*L1*sin(z[4]))*0.5   -(L1*sin(z[4]))*0.5   -L1*cos(z[4])   0.0   0.0   0
                0.0   0.0   0.0   (L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5   -(L2*sin(z[5]))*0.5-(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5   -L2*cos(z[5])*cos(z[6])   0.0   0.0   0
                0.0   0.0   0.0   (L2*cos(z[6])*sin(z[5]))*0.5   -(sqrt(3)*L2*cos(z[6])*sin(z[5]))*0.5   L2*sin(z[5])*sin(z[6])   0.0   0.0   0
                0.0   0.0   1   0.0   0.0   0.0   (sqrt(3)*L1*sin(z[7]))*0.5   -(L1*sin(z[7]))*0.5   -L1*cos(z[7])
                0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5   (sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5-(L2*sin(z[8]))*0.5   -L2*cos(z[8])*cos(z[9])
                0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(z[9])*sin(z[8]))*0.5   (sqrt(3)*L2*cos(z[9])*sin(z[8]))*0.5   L2*sin(z[8])*sin(z[9])
            ]\[
                -g*cos(z[1])*(L1*(M2+M3)+LC1*M1)
                -g*cos(z[2])*cos(z[3])*(L2*M3+LC2*M2)
                g*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)
                -g*cos(z[4])*(L1*(M2+M3)+LC1*M1)
                -g*cos(z[5])*cos(z[6])*(L2*M3+LC2*M2)
                g*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)
                -g*cos(z[7])*(L1*(M2+M3)+LC1*M1)
                -g*cos(z[8])*cos(z[9])*(L2*M3+LC2*M2)
                g*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)
            ]
            ut = ug_lg[1:3] + u(t)

            # Differential equations
            res[1] = dz[1]-z[10]
            res[2] = dz[2]-z[11]
            res[3] = dz[3]-z[12]
            res[4] = dz[4]-z[13]
            res[5] = dz[5]-z[14]
            res[6] = dz[6]-z[15]
            res[7] = dz[7]-z[16]
            res[8] = dz[8]-z[17]
            res[9] = dz[9]-z[18]
            res[10] = dz[10]*(J1+L1^2*(M2+M3)+LC1^2*M1)-ut[1]+γ*z[10]-g*cos(z[1])*(L1*(M2+M3)+LC1*M1)-L1*cos(z[1])*z[21]-L1*cos(z[1])*z[24]+L1*sin(z[1])*z[20]+L1*sin(z[1])*z[23]+L1*dz[11]*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*dz[12]*(L2*M3+LC2*M2)-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2)
            res[11] = dz[11]*(J2+L2^2*M3+LC2^2*M2)+γ*z[11]+L2*sin(z[2])*z[20]+L2*sin(z[2])*z[23]-L2*cos(z[2])*cos(z[3])*z[21]-L2*cos(z[2])*cos(z[3])*z[24]-L2*cos(z[2])*sin(z[3])*z[19]-L2*cos(z[2])*sin(z[3])*z[22]+L1*dz[10]*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))-cos(z[2])*sin(z[2])*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-g*cos(z[2])*cos(z[3])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))
            res[12] = γ*z[12]+sin(z[2])^2*dz[12]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[3])*sin(z[2])*z[19]-L2*cos(z[3])*sin(z[2])*z[22]+L2*sin(z[2])*sin(z[3])*z[21]+L2*sin(z[2])*sin(z[3])*z[24]+sin(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+g*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*sin(z[1])*sin(z[2])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*dz[10]*(L2*M3+LC2*M2)
            res[13] = dz[13]*(J1+L1^2*(M2+M3)+LC1^2*M1)-ut[2]+γ*z[13]-g*cos(z[4])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[4])*z[21]+(L1*sin(z[4])*z[20])*0.5+(sqrt(3)*L1*sin(z[4])*z[19])*0.5+L1*dz[14]*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)-L1*cos(z[4])*sin(z[5])*sin(z[6])*dz[15]*(L2*M3+LC2*M2)-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2)
            res[14] = z[20]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-z[19]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)+dz[14]*(J2+L2^2*M3+LC2^2*M2)+γ*z[14]+L2*cos(z[5])*cos(z[6])*z[21]+L1*dz[13]*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))-cos(z[5])*sin(z[5])*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-g*cos(z[5])*cos(z[6])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))
            res[15] = γ*z[15]+sin(z[5])^2*dz[15]*(J2+L2^2*M3+LC2^2*M2)-(L2*cos(z[6])*sin(z[5])*z[19])*0.5-L2*sin(z[5])*sin(z[6])*z[21]+sin(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)+g*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[6])*sin(z[5])*z[20])*0.5+L1*sin(z[4])*sin(z[5])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)-L1*cos(z[4])*sin(z[5])*sin(z[6])*dz[13]*(L2*M3+LC2*M2)
            res[16] = dz[16]*(J1+L1^2*(M2+M3)+LC1^2*M1)-ut[3]+γ*z[16]-g*cos(z[7])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[7])*z[24]+(L1*sin(z[7])*z[23])*0.5-(sqrt(3)*L1*sin(z[7])*z[22])*0.5+L1*dz[17]*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)-L1*cos(z[7])*sin(z[8])*sin(z[9])*dz[18]*(L2*M3+LC2*M2)-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2)
            res[17] = z[23]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)-z[22]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)+dz[17]*(J2+L2^2*M3+LC2^2*M2)+γ*z[17]+L2*cos(z[8])*cos(z[9])*z[24]+L1*dz[16]*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))-cos(z[8])*sin(z[8])*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-g*cos(z[8])*cos(z[9])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))
            res[18] = γ*z[18]+sin(z[8])^2*dz[18]*(J2+L2^2*M3+LC2^2*M2)-(L2*cos(z[9])*sin(z[8])*z[22])*0.5-L2*sin(z[8])*sin(z[9])*z[24]+sin(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)+g*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)-(sqrt(3)*L2*cos(z[9])*sin(z[8])*z[23])*0.5+L1*sin(z[7])*sin(z[8])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)-L1*cos(z[7])*sin(z[8])*sin(z[9])*dz[16]*(L2*M3+LC2*M2)
            
            # Algebraic equations

            dv = [ # inv(M)*Mv_term
                [J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))   -L1*cos(z[1])*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)
                 L1*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))   J2+L2^2*M3+LC2^2*M2   0
                 -L1*cos(z[1])*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)   0.0   sin(z[2])^2*(J2+L2^2*M3+LC2^2*M2)]\
                [ut[1]-γ*z[10]+g*cos(z[1])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[1])*z[21]+L1*cos(z[1])*z[24]-L1*sin(z[1])*z[20]-L1*sin(z[1])*z[23]-L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))+L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2)
                 L2*cos(z[2])*cos(z[3])*z[21]-L2*sin(z[2])*z[20]-L2*sin(z[2])*z[23]-γ*z[11]+L2*cos(z[2])*cos(z[3])*z[24]+L2*cos(z[2])*sin(z[3])*z[19]+L2*cos(z[2])*sin(z[3])*z[22]+cos(z[2])*sin(z[2])*z[12]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[2])*cos(z[3])*(L2*M3+LC2*M2)-L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))
                 L2*cos(z[3])*sin(z[2])*z[19]-γ*z[12]+L2*cos(z[3])*sin(z[2])*z[22]-L2*sin(z[2])*sin(z[3])*z[21]-L2*sin(z[2])*sin(z[3])*z[24]-sin(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)-g*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)-L1*sin(z[1])*sin(z[2])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)]

                [J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))   -L1*cos(z[4])*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)
                 L1*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))   J2+L2^2*M3+LC2^2*M2   0
                 -L1*cos(z[4])*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)   0.0   sin(z[5])^2*(J2+L2^2*M3+LC2^2*M2)]\
                [ut[2]-γ*z[13]+g*cos(z[4])*(L1*(M2+M3)+LC1*M1)-L1*cos(z[4])*z[21]-(L1*sin(z[4])*z[20])*0.5-(sqrt(3)*L1*sin(z[4])*z[19])*0.5-L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))+L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2)
                 z[19]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)-z[20]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-γ*z[14]-L2*cos(z[5])*cos(z[6])*z[21]+cos(z[5])*sin(z[5])*z[15]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[5])*cos(z[6])*(L2*M3+LC2*M2)-L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))
                 (L2*cos(z[6])*sin(z[5])*z[19])*0.5-γ*z[15]+L2*sin(z[5])*sin(z[6])*z[21]-sin(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)-g*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)-(sqrt(3)*L2*cos(z[6])*sin(z[5])*z[20])*0.5-L1*sin(z[4])*sin(z[5])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)]
                
                [J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))   -L1*cos(z[7])*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)
                 L1*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))   J2+L2^2*M3+LC2^2*M2   0
                 -L1*cos(z[7])*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)   0.0   sin(z[8])^2*(J2+L2^2*M3+LC2^2*M2)]\
                [ut[3]-γ*z[16]+g*cos(z[7])*(L1*(M2+M3)+LC1*M1)-L1*cos(z[7])*z[24]-(L1*sin(z[7])*z[23])*0.5+(sqrt(3)*L1*sin(z[7])*z[22])*0.5-L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))+L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2)
                 z[22]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)-z[23]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)-γ*z[17]-L2*cos(z[8])*cos(z[9])*z[24]+cos(z[8])*sin(z[8])*z[18]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[8])*cos(z[9])*(L2*M3+LC2*M2)-L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))
                 (L2*cos(z[9])*sin(z[8])*z[22])*0.5-γ*z[18]+L2*sin(z[8])*sin(z[9])*z[24]-sin(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)-g*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[9])*sin(z[8])*z[23])*0.5-L1*sin(z[7])*sin(z[8])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)]
                ]

            res[19] = α1*(dz[5]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)+L2*cos(z[2])*sin(z[3])*dz[2]+L2*cos(z[3])*sin(z[2])*dz[3]+(L2*cos(z[6])*sin(z[5])*dz[6])*0.5-(sqrt(3)*L1*sin(z[4])*dz[4])*0.5)-z[14]*((sqrt(3)*L2*cos(z[5])*dz[5])*0.5-(L2*cos(z[5])*cos(z[6])*dz[6])*0.5+(L2*sin(z[5])*sin(z[6])*dz[5])*0.5)+dv[5]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)+α0*((sqrt(3)*(L0-L3+L1*cos(z[4])+L2*cos(z[5])))*0.5+L2*sin(z[2])*sin(z[3])+(L2*sin(z[5])*sin(z[6]))*0.5)+z[11]*(L2*cos(z[2])*cos(z[3])*dz[3]-L2*sin(z[2])*sin(z[3])*dz[2])+z[12]*(L2*cos(z[2])*cos(z[3])*dz[2]-L2*sin(z[2])*sin(z[3])*dz[3])+z[15]*((L2*cos(z[5])*cos(z[6])*dz[5])*0.5-(L2*sin(z[5])*sin(z[6])*dz[6])*0.5)+L2*cos(z[2])*sin(z[3])*dv[2]+L2*cos(z[3])*sin(z[2])*dv[3]+(L2*cos(z[6])*sin(z[5])*dv[6])*0.5-(sqrt(3)*L1*sin(z[4])*dv[4])*0.5-(sqrt(3)*L1*cos(z[4])*dz[4]*z[13])*0.5
            res[20] = α0*((3*L0)*0.5-(3*L3)*0.5+L1*cos(z[1])+L2*cos(z[2])+(L1*cos(z[4]))*0.5+(L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5)-dv[5]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-z[14]*((L2*cos(z[5])*dz[5])*0.5+(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[6])*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6])*dz[5])*0.5)-α1*(dz[5]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)+L1*sin(z[1])*dz[1]+L2*sin(z[2])*dz[2]+(L1*sin(z[4])*dz[4])*0.5+(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[6])*0.5)-z[15]*((sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[5])*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6])*dz[6])*0.5)-L1*sin(z[1])*dv[1]-L2*sin(z[2])*dv[2]-(L1*sin(z[4])*dv[4])*0.5-L1*cos(z[1])*dz[1]*z[10]-L2*cos(z[2])*dz[2]*z[11]-(L1*cos(z[4])*dz[4]*z[13])*0.5-(sqrt(3)*L2*cos(z[6])*sin(z[5])*dv[6])*0.5
            res[21] = α1*(L1*cos(z[1])*dz[1]-L1*cos(z[4])*dz[4]+L2*cos(z[2])*cos(z[3])*dz[2]-L2*cos(z[5])*cos(z[6])*dz[5]-L2*sin(z[2])*sin(z[3])*dz[3]+L2*sin(z[5])*sin(z[6])*dz[6])+α0*(L1*sin(z[1])-L1*sin(z[4])+L2*cos(z[3])*sin(z[2])-L2*cos(z[6])*sin(z[5]))-z[11]*(L2*cos(z[3])*sin(z[2])*dz[2]+L2*cos(z[2])*sin(z[3])*dz[3])-z[12]*(L2*cos(z[2])*sin(z[3])*dz[2]+L2*cos(z[3])*sin(z[2])*dz[3])+z[14]*(L2*cos(z[6])*sin(z[5])*dz[5]+L2*cos(z[5])*sin(z[6])*dz[6])+z[15]*(L2*cos(z[5])*sin(z[6])*dz[5]+L2*cos(z[6])*sin(z[5])*dz[6])+L1*cos(z[1])*dv[1]-L1*cos(z[4])*dv[4]+L2*cos(z[2])*cos(z[3])*dv[2]-L2*cos(z[5])*cos(z[6])*dv[5]-L2*sin(z[2])*sin(z[3])*dv[3]+L2*sin(z[5])*sin(z[6])*dv[6]-L1*sin(z[1])*dz[1]*z[10]+L1*sin(z[4])*dz[4]*z[13]
            res[22] = α1*(dz[8]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)+L2*cos(z[2])*sin(z[3])*dz[2]+L2*cos(z[3])*sin(z[2])*dz[3]+(L2*cos(z[9])*sin(z[8])*dz[9])*0.5+(sqrt(3)*L1*sin(z[7])*dz[7])*0.5)+z[17]*((L2*cos(z[8])*cos(z[9])*dz[9])*0.5+(sqrt(3)*L2*cos(z[8])*dz[8])*0.5-(L2*sin(z[8])*sin(z[9])*dz[8])*0.5)+dv[8]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)+α0*(L2*sin(z[2])*sin(z[3])-(sqrt(3)*(L0-L3+L1*cos(z[7])+L2*cos(z[8])))*0.5+(L2*sin(z[8])*sin(z[9]))*0.5)+z[11]*(L2*cos(z[2])*cos(z[3])*dz[3]-L2*sin(z[2])*sin(z[3])*dz[2])+z[12]*(L2*cos(z[2])*cos(z[3])*dz[2]-L2*sin(z[2])*sin(z[3])*dz[3])+z[18]*((L2*cos(z[8])*cos(z[9])*dz[8])*0.5-(L2*sin(z[8])*sin(z[9])*dz[9])*0.5)+L2*cos(z[2])*sin(z[3])*dv[2]+L2*cos(z[3])*sin(z[2])*dv[3]+(L2*cos(z[9])*sin(z[8])*dv[9])*0.5+(sqrt(3)*L1*sin(z[7])*dv[7])*0.5+(sqrt(3)*L1*cos(z[7])*dz[7]*z[16])*0.5
            res[23] = α0*((3*L0)*0.5-(3*L3)*0.5+L1*cos(z[1])+L2*cos(z[2])+(L1*cos(z[7]))*0.5+(L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5)-dv[8]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)-z[17]*((L2*cos(z[8])*dz[8])*0.5-(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[9])*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9])*dz[8])*0.5)-α1*(dz[8]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)+L1*sin(z[1])*dz[1]+L2*sin(z[2])*dz[2]+(L1*sin(z[7])*dz[7])*0.5-(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[9])*0.5)+z[18]*((sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[8])*0.5-(sqrt(3)*L2*sin(z[8])*sin(z[9])*dz[9])*0.5)-L1*sin(z[1])*dv[1]-L2*sin(z[2])*dv[2]-(L1*sin(z[7])*dv[7])*0.5-L1*cos(z[1])*dz[1]*z[10]-L2*cos(z[2])*dz[2]*z[11]-(L1*cos(z[7])*dz[7]*z[16])*0.5+(sqrt(3)*L2*cos(z[9])*sin(z[8])*dv[9])*0.5
            res[24] = α1*(L1*cos(z[1])*dz[1]-L1*cos(z[7])*dz[7]+L2*cos(z[2])*cos(z[3])*dz[2]-L2*cos(z[8])*cos(z[9])*dz[8]-L2*sin(z[2])*sin(z[3])*dz[3]+L2*sin(z[8])*sin(z[9])*dz[9])+α0*(L1*sin(z[1])-L1*sin(z[7])+L2*cos(z[3])*sin(z[2])-L2*cos(z[9])*sin(z[8]))-z[11]*(L2*cos(z[3])*sin(z[2])*dz[2]+L2*cos(z[2])*sin(z[3])*dz[3])-z[12]*(L2*cos(z[2])*sin(z[3])*dz[2]+L2*cos(z[3])*sin(z[2])*dz[3])+z[17]*(L2*cos(z[9])*sin(z[8])*dz[8]+L2*cos(z[8])*sin(z[9])*dz[9])+z[18]*(L2*cos(z[8])*sin(z[9])*dz[8]+L2*cos(z[9])*sin(z[8])*dz[9])+L1*cos(z[1])*dv[1]-L1*cos(z[7])*dv[7]+L2*cos(z[2])*cos(z[3])*dv[2]-L2*cos(z[8])*cos(z[9])*dv[8]-L2*sin(z[2])*sin(z[3])*dv[3]+L2*sin(z[8])*sin(z[9])*dv[9]-L1*sin(z[1])*dz[1]*z[10]+L1*sin(z[7])*dz[7]*z[16]

            nothing
        end

        z0, dz0 = get_delta_initial(θ, u(0.0), du0, w(0.0))

        dvars = vcat(fill(true, 18), fill(false, 6))
        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 for delta robot is: $r0"

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, z0, dz0, dvars, r0)
    end
end

function delta_robot_gc(Φ::Float64, u::Function, w::Function, θ::Vector{Float64})::Model
    let L0 = θ[1], L1 = θ[2], L2 = θ[3], L3 = θ[4], LC1 = θ[5], LC2 = θ[6], M1 = θ[7], M2 = θ[8], M3 = θ[9], J1 = θ[10], J2 = θ[11], g = 0.0, γ = θ[13]
        # @info "Samples: $(u(0.3)), $(u(0.46)), $(u(1.23)), $(w(0.3)), $(w(0.46)), $(w(1.23))"
        function f!(res, dz, z, _, t)
            ut = u(t)
            wt = w(t)

            res[1] = dz[1]-z[10]+L1*cos(z[1])*dz[27]+L1*cos(z[1])*dz[30]-L1*sin(z[1])*dz[26]-L1*sin(z[1])*dz[29]
            res[2] = dz[2]-z[11]-L2*sin(z[2])*dz[26]-L2*sin(z[2])*dz[29]+L2*cos(z[2])*cos(z[3])*dz[27]+L2*cos(z[2])*cos(z[3])*dz[30]+L2*cos(z[2])*sin(z[3])*dz[25]+L2*cos(z[2])*sin(z[3])*dz[28]
            res[3] = dz[3]-z[12]+L2*cos(z[3])*sin(z[2])*dz[25]+L2*cos(z[3])*sin(z[2])*dz[28]-L2*sin(z[2])*sin(z[3])*dz[27]-L2*sin(z[2])*sin(z[3])*dz[30]
            res[4] = dz[4]-z[13]-L1*cos(z[4])*dz[27]-(L1*sin(z[4])*dz[26])*0.5-(sqrt(3)*L1*sin(z[4])*dz[25])*0.5
            res[5] = dz[5]-z[14]+dz[25]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)-dz[26]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-L2*cos(z[5])*cos(z[6])*dz[27]
            res[6] = dz[6]-z[15]+(L2*cos(z[6])*sin(z[5])*dz[25])*0.5+L2*sin(z[5])*sin(z[6])*dz[27]-(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[26])*0.5
            res[7] = dz[7]-z[16]-L1*cos(z[7])*dz[30]-(L1*sin(z[7])*dz[29])*0.5+(sqrt(3)*L1*sin(z[7])*dz[28])*0.5
            res[8] = dz[8]-z[17]+dz[28]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)-dz[29]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)-L2*cos(z[8])*cos(z[9])*dz[30]
            res[9] = dz[9]-z[18]+(L2*cos(z[9])*sin(z[8])*dz[28])*0.5+L2*sin(z[8])*sin(z[9])*dz[30]+(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[29])*0.5
            res[10] = dz[10]*(J1+L1^2*(M2+M3)+LC1^2*M1)-ut[1]-wt[1]^2+γ*z[10]-0.0*cos(z[1])*(L1*(M2+M3)+LC1*M1)-L1*cos(z[1])*dz[21]-L1*cos(z[1])*dz[24]+L1*sin(z[1])*dz[20]+L1*sin(z[1])*dz[23]+L1*dz[11]*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*dz[12]*(L2*M3+LC2*M2)-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2)
            res[11] = dz[11]*(J2+L2^2*M3+LC2^2*M2)+γ*z[11]+L2*sin(z[2])*dz[20]+L2*sin(z[2])*dz[23]-L2*cos(z[2])*cos(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[24]-L2*cos(z[2])*sin(z[3])*dz[19]-L2*cos(z[2])*sin(z[3])*dz[22]+L1*dz[10]*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))-cos(z[2])*sin(z[2])*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-0.0*cos(z[2])*cos(z[3])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))
            res[12] = γ*z[12]+sin(z[2])^2*dz[12]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[3])*sin(z[2])*dz[19]-L2*cos(z[3])*sin(z[2])*dz[22]+L2*sin(z[2])*sin(z[3])*dz[21]+L2*sin(z[2])*sin(z[3])*dz[24]+sin(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+0.0*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*sin(z[1])*sin(z[2])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*dz[10]*(L2*M3+LC2*M2)
            res[13] = dz[13]*(J1+L1^2*(M2+M3)+LC1^2*M1)-ut[2]-wt[2]^2+γ*z[13]-0.0*cos(z[4])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[4])*dz[21]+(L1*sin(z[4])*dz[20])*0.5+(sqrt(3)*L1*sin(z[4])*dz[19])*0.5+L1*dz[14]*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)-L1*cos(z[4])*sin(z[5])*sin(z[6])*dz[15]*(L2*M3+LC2*M2)-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2)
            res[14] = dz[20]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-dz[19]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)+dz[14]*(J2+L2^2*M3+LC2^2*M2)+γ*z[14]+L2*cos(z[5])*cos(z[6])*dz[21]+L1*dz[13]*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))-cos(z[5])*sin(z[5])*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-0.0*cos(z[5])*cos(z[6])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))
            res[15] = γ*z[15]+sin(z[5])^2*dz[15]*(J2+L2^2*M3+LC2^2*M2)-(L2*cos(z[6])*sin(z[5])*dz[19])*0.5-L2*sin(z[5])*sin(z[6])*dz[21]+sin(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)+0.0*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[20])*0.5+L1*sin(z[4])*sin(z[5])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)-L1*cos(z[4])*sin(z[5])*sin(z[6])*dz[13]*(L2*M3+LC2*M2)
            res[16] = dz[16]*(J1+L1^2*(M2+M3)+LC1^2*M1)-ut[3]-wt[3]^2+γ*z[16]-0.0*cos(z[7])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[7])*dz[24]+(L1*sin(z[7])*dz[23])*0.5-(sqrt(3)*L1*sin(z[7])*dz[22])*0.5+L1*dz[17]*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)-L1*cos(z[7])*sin(z[8])*sin(z[9])*dz[18]*(L2*M3+LC2*M2)-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2)
            res[17] = dz[23]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)-dz[22]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)+dz[17]*(J2+L2^2*M3+LC2^2*M2)+γ*z[17]+L2*cos(z[8])*cos(z[9])*dz[24]+L1*dz[16]*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))-cos(z[8])*sin(z[8])*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-0.0*cos(z[8])*cos(z[9])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))
            res[18] = γ*z[18]+sin(z[8])^2*dz[18]*(J2+L2^2*M3+LC2^2*M2)-(L2*cos(z[9])*sin(z[8])*dz[22])*0.5-L2*sin(z[8])*sin(z[9])*dz[24]+sin(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)+0.0*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)-(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[23])*0.5+L1*sin(z[7])*sin(z[8])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)-L1*cos(z[7])*sin(z[8])*sin(z[9])*dz[16]*(L2*M3+LC2*M2)
            res[19] = (sqrt(3)*(L0-L3+L1*cos(z[4])+L2*cos(z[5])))*0.5+L2*sin(z[2])*sin(z[3])+(L2*sin(z[5])*sin(z[6]))*0.5
            res[20] = (3*L0)*0.5-(3*L3)*0.5+L1*cos(z[1])+L2*cos(z[2])+(L1*cos(z[4]))*0.5+(L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5
            res[21] = L1*sin(z[1])-L1*sin(z[4])+L2*cos(z[3])*sin(z[2])-L2*cos(z[6])*sin(z[5])
            res[22] = L2*sin(z[2])*sin(z[3])-(sqrt(3)*(L0-L3+L1*cos(z[7])+L2*cos(z[8])))*0.5+(L2*sin(z[8])*sin(z[9]))*0.5
            res[23] = (3*L0)*0.5-(3*L3)*0.5+L1*cos(z[1])+L2*cos(z[2])+(L1*cos(z[7]))*0.5+(L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5
            res[24] = L1*sin(z[1])-L1*sin(z[7])+L2*cos(z[3])*sin(z[2])-L2*cos(z[9])*sin(z[8])
            res[25] = L2*cos(z[2])*sin(z[3])*z[11]-(sqrt(3)*(L1*sin(z[4])*z[13]+L2*sin(z[5])*z[14]))*0.5+L2*cos(z[3])*sin(z[2])*z[12]+(L2*cos(z[5])*sin(z[6])*z[14])*0.5+(L2*cos(z[6])*sin(z[5])*z[15])*0.5
            res[26] = -L1*sin(z[1])*z[10]-L2*sin(z[2])*z[11]-(L1*sin(z[4])*z[13])*0.5-(L2*sin(z[5])*z[14])*0.5-(sqrt(3)*L2*cos(z[5])*sin(z[6])*z[14])*0.5-(sqrt(3)*L2*cos(z[6])*sin(z[5])*z[15])*0.5
            res[27] = L1*cos(z[1])*z[10]-L1*cos(z[4])*z[13]+L2*cos(z[2])*cos(z[3])*z[11]-L2*cos(z[5])*cos(z[6])*z[14]-L2*sin(z[2])*sin(z[3])*z[12]+L2*sin(z[5])*sin(z[6])*z[15]
            res[28] = (sqrt(3)*(L1*sin(z[7])*z[16]+L2*sin(z[8])*z[17]))*0.5+L2*cos(z[2])*sin(z[3])*z[11]+L2*cos(z[3])*sin(z[2])*z[12]+(L2*cos(z[8])*sin(z[9])*z[17])*0.5+(L2*cos(z[9])*sin(z[8])*z[18])*0.5
            res[29] = (sqrt(3)*L2*cos(z[8])*sin(z[9])*z[17])*0.5-L2*sin(z[2])*z[11]-(L1*sin(z[7])*z[16])*0.5-(L2*sin(z[8])*z[17])*0.5-L1*sin(z[1])*z[10]+(sqrt(3)*L2*cos(z[9])*sin(z[8])*z[18])*0.5
            res[30] = L1*cos(z[1])*z[10]-L1*cos(z[7])*z[16]+L2*cos(z[2])*cos(z[3])*z[11]-L2*cos(z[8])*cos(z[9])*z[17]-L2*sin(z[2])*sin(z[3])*z[12]+L2*sin(z[8])*sin(z[9])*z[18]

            nothing
        end

        # @info "In here, we get θ = $θ, u=$(u(0.0)) and w=$(w(0.0))"
        z0, dz0 = get_delta_initial_comp(θ, u(0.0), w(0.0))

        dvars = fill(true, 30)
        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 for delta robot is: $r0"

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, z0, dz0, dvars, r0)
    end
end

function delta_robot_gc_L1sens(Φ::Float64, u::Function, w::Function, θ::Vector{Float64})::Model 
    let L0 = θ[1], L1 = θ[2], L2 = θ[3], L3 = θ[4], LC1 = θ[5], LC2 = θ[6], M1 = θ[7], M2 = θ[8], M3 = θ[9], J1 = θ[10], J2 = θ[11], g = 0.0, γ = θ[13]
        # @info "Samples: $(u(0.3)), $(u(0.46)), $(u(1.23)), $(w(0.3)), $(w(0.46)), $(w(1.23))"
        function f!(res, dz, z, _, t)
            ut = u(t)
            wt = w(t)

            res[1] = dz[1]-z[10]+L1*cos(z[1])*dz[27]+L1*cos(z[1])*dz[30]-L1*sin(z[1])*dz[26]-L1*sin(z[1])*dz[29]
            res[2] = dz[2]-z[11]-L2*sin(z[2])*dz[26]-L2*sin(z[2])*dz[29]+L2*cos(z[2])*cos(z[3])*dz[27]+L2*cos(z[2])*cos(z[3])*dz[30]+L2*cos(z[2])*sin(z[3])*dz[25]+L2*cos(z[2])*sin(z[3])*dz[28]
            res[3] = dz[3]-z[12]+L2*cos(z[3])*sin(z[2])*dz[25]+L2*cos(z[3])*sin(z[2])*dz[28]-L2*sin(z[2])*sin(z[3])*dz[27]-L2*sin(z[2])*sin(z[3])*dz[30]
            res[4] = dz[4]-z[13]-L1*cos(z[4])*dz[27]-(L1*sin(z[4])*dz[26])*0.5-(sqrt(3)*L1*sin(z[4])*dz[25])*0.5
            res[5] = dz[5]-z[14]+dz[25]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)-dz[26]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-L2*cos(z[5])*cos(z[6])*dz[27]
            res[6] = dz[6]-z[15]+(L2*cos(z[6])*sin(z[5])*dz[25])*0.5+L2*sin(z[5])*sin(z[6])*dz[27]-(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[26])*0.5
            res[7] = dz[7]-z[16]-L1*cos(z[7])*dz[30]-(L1*sin(z[7])*dz[29])*0.5+(sqrt(3)*L1*sin(z[7])*dz[28])*0.5
            res[8] = dz[8]-z[17]+dz[28]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)-dz[29]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)-L2*cos(z[8])*cos(z[9])*dz[30]
            res[9] = dz[9]-z[18]+(L2*cos(z[9])*sin(z[8])*dz[28])*0.5+L2*sin(z[8])*sin(z[9])*dz[30]+(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[29])*0.5
            res[10] = dz[10]*(J1+L1^2*(M2+M3)+LC1^2*M1)-ut[1]-wt[1]^2+γ*z[10]-0.0*cos(z[1])*(L1*(M2+M3)+LC1*M1)-L1*cos(z[1])*dz[21]-L1*cos(z[1])*dz[24]+L1*sin(z[1])*dz[20]+L1*sin(z[1])*dz[23]+L1*dz[11]*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*dz[12]*(L2*M3+LC2*M2)-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2)
            res[11] = dz[11]*(J2+L2^2*M3+LC2^2*M2)+γ*z[11]+L2*sin(z[2])*dz[20]+L2*sin(z[2])*dz[23]-L2*cos(z[2])*cos(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[24]-L2*cos(z[2])*sin(z[3])*dz[19]-L2*cos(z[2])*sin(z[3])*dz[22]+L1*dz[10]*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))-cos(z[2])*sin(z[2])*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-0.0*cos(z[2])*cos(z[3])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))
            res[12] = γ*z[12]+sin(z[2])^2*dz[12]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[3])*sin(z[2])*dz[19]-L2*cos(z[3])*sin(z[2])*dz[22]+L2*sin(z[2])*sin(z[3])*dz[21]+L2*sin(z[2])*sin(z[3])*dz[24]+sin(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+0.0*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*sin(z[1])*sin(z[2])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*dz[10]*(L2*M3+LC2*M2)
            res[13] = dz[13]*(J1+L1^2*(M2+M3)+LC1^2*M1)-ut[2]-wt[2]^2+γ*z[13]-0.0*cos(z[4])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[4])*dz[21]+(L1*sin(z[4])*dz[20])*0.5+(sqrt(3)*L1*sin(z[4])*dz[19])*0.5+L1*dz[14]*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)-L1*cos(z[4])*sin(z[5])*sin(z[6])*dz[15]*(L2*M3+LC2*M2)-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2)
            res[14] = dz[20]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-dz[19]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)+dz[14]*(J2+L2^2*M3+LC2^2*M2)+γ*z[14]+L2*cos(z[5])*cos(z[6])*dz[21]+L1*dz[13]*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))-cos(z[5])*sin(z[5])*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-0.0*cos(z[5])*cos(z[6])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))
            res[15] = γ*z[15]+sin(z[5])^2*dz[15]*(J2+L2^2*M3+LC2^2*M2)-(L2*cos(z[6])*sin(z[5])*dz[19])*0.5-L2*sin(z[5])*sin(z[6])*dz[21]+sin(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)+0.0*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[20])*0.5+L1*sin(z[4])*sin(z[5])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)-L1*cos(z[4])*sin(z[5])*sin(z[6])*dz[13]*(L2*M3+LC2*M2)
            res[16] = dz[16]*(J1+L1^2*(M2+M3)+LC1^2*M1)-ut[3]-wt[3]^2+γ*z[16]-0.0*cos(z[7])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[7])*dz[24]+(L1*sin(z[7])*dz[23])*0.5-(sqrt(3)*L1*sin(z[7])*dz[22])*0.5+L1*dz[17]*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)-L1*cos(z[7])*sin(z[8])*sin(z[9])*dz[18]*(L2*M3+LC2*M2)-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2)
            res[17] = dz[23]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)-dz[22]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)+dz[17]*(J2+L2^2*M3+LC2^2*M2)+γ*z[17]+L2*cos(z[8])*cos(z[9])*dz[24]+L1*dz[16]*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))-cos(z[8])*sin(z[8])*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-0.0*cos(z[8])*cos(z[9])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))
            res[18] = γ*z[18]+sin(z[8])^2*dz[18]*(J2+L2^2*M3+LC2^2*M2)-(L2*cos(z[9])*sin(z[8])*dz[22])*0.5-L2*sin(z[8])*sin(z[9])*dz[24]+sin(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)+0.0*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)-(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[23])*0.5+L1*sin(z[7])*sin(z[8])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)-L1*cos(z[7])*sin(z[8])*sin(z[9])*dz[16]*(L2*M3+LC2*M2)
            res[19] = (sqrt(3)*(L0-L3+L1*cos(z[4])+L2*cos(z[5])))*0.5+L2*sin(z[2])*sin(z[3])+(L2*sin(z[5])*sin(z[6]))*0.5
            res[20] = (3*L0)*0.5-(3*L3)*0.5+L1*cos(z[1])+L2*cos(z[2])+(L1*cos(z[4]))*0.5+(L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5
            res[21] = L1*sin(z[1])-L1*sin(z[4])+L2*cos(z[3])*sin(z[2])-L2*cos(z[6])*sin(z[5])
            res[22] = L2*sin(z[2])*sin(z[3])-(sqrt(3)*(L0-L3+L1*cos(z[7])+L2*cos(z[8])))*0.5+(L2*sin(z[8])*sin(z[9]))*0.5
            res[23] = (3*L0)*0.5-(3*L3)*0.5+L1*cos(z[1])+L2*cos(z[2])+(L1*cos(z[7]))*0.5+(L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5
            res[24] = L1*sin(z[1])-L1*sin(z[7])+L2*cos(z[3])*sin(z[2])-L2*cos(z[9])*sin(z[8])
            res[25] = L2*cos(z[2])*sin(z[3])*z[11]-(sqrt(3)*(L1*sin(z[4])*z[13]+L2*sin(z[5])*z[14]))*0.5+L2*cos(z[3])*sin(z[2])*z[12]+(L2*cos(z[5])*sin(z[6])*z[14])*0.5+(L2*cos(z[6])*sin(z[5])*z[15])*0.5
            res[26] = -L1*sin(z[1])*z[10]-L2*sin(z[2])*z[11]-(L1*sin(z[4])*z[13])*0.5-(L2*sin(z[5])*z[14])*0.5-(sqrt(3)*L2*cos(z[5])*sin(z[6])*z[14])*0.5-(sqrt(3)*L2*cos(z[6])*sin(z[5])*z[15])*0.5
            res[27] = L1*cos(z[1])*z[10]-L1*cos(z[4])*z[13]+L2*cos(z[2])*cos(z[3])*z[11]-L2*cos(z[5])*cos(z[6])*z[14]-L2*sin(z[2])*sin(z[3])*z[12]+L2*sin(z[5])*sin(z[6])*z[15]
            res[28] = (sqrt(3)*(L1*sin(z[7])*z[16]+L2*sin(z[8])*z[17]))*0.5+L2*cos(z[2])*sin(z[3])*z[11]+L2*cos(z[3])*sin(z[2])*z[12]+(L2*cos(z[8])*sin(z[9])*z[17])*0.5+(L2*cos(z[9])*sin(z[8])*z[18])*0.5
            res[29] = (sqrt(3)*L2*cos(z[8])*sin(z[9])*z[17])*0.5-L2*sin(z[2])*z[11]-(L1*sin(z[7])*z[16])*0.5-(L2*sin(z[8])*z[17])*0.5-L1*sin(z[1])*z[10]+(sqrt(3)*L2*cos(z[9])*sin(z[8])*z[18])*0.5
            res[30] = L1*cos(z[1])*z[10]-L1*cos(z[7])*z[16]+L2*cos(z[2])*cos(z[3])*z[11]-L2*cos(z[8])*cos(z[9])*z[17]-L2*sin(z[2])*sin(z[3])*z[12]+L2*sin(z[8])*sin(z[9])*z[18]
            # Sensitivity equations
            res[31] = dz[31]-z[40]-z[31]*(L1*cos(z[1])*dz[26]+L1*cos(z[1])*dz[29]+L1*sin(z[1])*dz[27]+L1*sin(z[1])*dz[30])+L1*cos(z[1])*dz[57]+L1*cos(z[1])*dz[60]-L1*sin(z[1])*dz[56]-L1*sin(z[1])*dz[59]
            res[32] = dz[32]-z[41]+z[33]*(L2*cos(z[2])*cos(z[3])*dz[25]+L2*cos(z[2])*cos(z[3])*dz[28]-L2*cos(z[2])*sin(z[3])*dz[27]-L2*cos(z[2])*sin(z[3])*dz[30])-z[32]*(L2*cos(z[2])*dz[26]+L2*cos(z[2])*dz[29]+L2*cos(z[3])*sin(z[2])*dz[27]+L2*cos(z[3])*sin(z[2])*dz[30]+L2*sin(z[2])*sin(z[3])*dz[25]+L2*sin(z[2])*sin(z[3])*dz[28])-L2*sin(z[2])*dz[56]-L2*sin(z[2])*dz[59]+L2*cos(z[2])*cos(z[3])*dz[57]+L2*cos(z[2])*cos(z[3])*dz[60]+L2*cos(z[2])*sin(z[3])*dz[55]+L2*cos(z[2])*sin(z[3])*dz[58]
            res[33] = dz[33]-z[42]+z[32]*(L2*cos(z[2])*cos(z[3])*dz[25]+L2*cos(z[2])*cos(z[3])*dz[28]-L2*cos(z[2])*sin(z[3])*dz[27]-L2*cos(z[2])*sin(z[3])*dz[30])-z[33]*(L2*cos(z[3])*sin(z[2])*dz[27]+L2*cos(z[3])*sin(z[2])*dz[30]+L2*sin(z[2])*sin(z[3])*dz[25]+L2*sin(z[2])*sin(z[3])*dz[28])+L2*cos(z[3])*sin(z[2])*dz[55]+L2*cos(z[3])*sin(z[2])*dz[58]-L2*sin(z[2])*sin(z[3])*dz[57]-L2*sin(z[2])*sin(z[3])*dz[60]
            res[34] = dz[34]-z[43]-z[34]*((L1*cos(z[4])*dz[26])*0.5-L1*sin(z[4])*dz[27]+(sqrt(3)*L1*cos(z[4])*dz[25])*0.5)-L1*cos(z[4])*dz[57]-(L1*sin(z[4])*dz[56])*0.5-(sqrt(3)*L1*sin(z[4])*dz[55])*0.5
            res[35] = dz[35]-z[44]+dz[55]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)-dz[56]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)+z[36]*((L2*cos(z[5])*cos(z[6])*dz[25])*0.5+L2*cos(z[5])*sin(z[6])*dz[27]-(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[26])*0.5)-z[35]*(dz[25]*((sqrt(3)*L2*cos(z[5]))*0.5+(L2*sin(z[5])*sin(z[6]))*0.5)+dz[26]*((L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5)-L2*cos(z[6])*sin(z[5])*dz[27])-L2*cos(z[5])*cos(z[6])*dz[57]
            res[36] = dz[36]-z[45]+z[35]*((L2*cos(z[5])*cos(z[6])*dz[25])*0.5+L2*cos(z[5])*sin(z[6])*dz[27]-(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[26])*0.5)+z[36]*(L2*cos(z[6])*sin(z[5])*dz[27]-(L2*sin(z[5])*sin(z[6])*dz[25])*0.5+(sqrt(3)*L2*sin(z[5])*sin(z[6])*dz[26])*0.5)+(L2*cos(z[6])*sin(z[5])*dz[55])*0.5+L2*sin(z[5])*sin(z[6])*dz[57]-(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[56])*0.5
            res[37] = dz[37]-z[46]+z[37]*(L1*sin(z[7])*dz[30]-(L1*cos(z[7])*dz[29])*0.5+(sqrt(3)*L1*cos(z[7])*dz[28])*0.5)-L1*cos(z[7])*dz[60]-(L1*sin(z[7])*dz[59])*0.5+(sqrt(3)*L1*sin(z[7])*dz[58])*0.5
            res[38] = dz[38]-z[47]+dz[58]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)-dz[59]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)+z[39]*((L2*cos(z[8])*cos(z[9])*dz[28])*0.5+L2*cos(z[8])*sin(z[9])*dz[30]+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[29])*0.5)+z[38]*(dz[28]*((sqrt(3)*L2*cos(z[8]))*0.5-(L2*sin(z[8])*sin(z[9]))*0.5)-dz[29]*((L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5)+L2*cos(z[9])*sin(z[8])*dz[30])-L2*cos(z[8])*cos(z[9])*dz[60]
            res[39] = dz[39]-z[48]+z[38]*((L2*cos(z[8])*cos(z[9])*dz[28])*0.5+L2*cos(z[8])*sin(z[9])*dz[30]+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[29])*0.5)-z[39]*((L2*sin(z[8])*sin(z[9])*dz[28])*0.5-L2*cos(z[9])*sin(z[8])*dz[30]+(sqrt(3)*L2*sin(z[8])*sin(z[9])*dz[29])*0.5)+(L2*cos(z[9])*sin(z[8])*dz[58])*0.5+L2*sin(z[8])*sin(z[9])*dz[60]+(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[59])*0.5
            res[40] = z[41]*(2*L1*z[11]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[12]*(L2*M3+LC2*M2))+dz[40]*(J1+L1^2*(M2+M3)+LC1^2*M1)-z[42]*(2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]*(L2*M3+LC2*M2))-z[33]*(L1*cos(z[1])*cos(z[2])*sin(z[3])*dz[11]*(L2*M3+LC2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*z[12]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]^2*(L2*M3+LC2*M2)+L1*cos(z[1])*cos(z[3])*sin(z[2])*dz[12]*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[2])*cos(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+z[31]*(0.0*sin(z[1])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[1])*dz[20]+L1*cos(z[1])*dz[23]+L1*sin(z[1])*dz[21]+L1*sin(z[1])*dz[24]+L1*dz[11]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2]))+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)+L1*sin(z[1])*sin(z[2])*sin(z[3])*dz[12]*(L2*M3+LC2*M2)+2*L1*cos(z[2])*sin(z[1])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+γ*z[40]-z[32]*(L1*z[11]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))-L1*dz[11]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))+L1*cos(z[1])*cos(z[2])*cos(z[3])*z[12]^2*(L2*M3+LC2*M2)+L1*cos(z[1])*cos(z[2])*sin(z[3])*dz[12]*(L2*M3+LC2*M2)-2*L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))-L1*cos(z[1])*dz[51]-L1*cos(z[1])*dz[54]+L1*sin(z[1])*dz[50]+L1*sin(z[1])*dz[53]+L1*dz[41]*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))-L1*cos(z[1])*sin(z[2])*sin(z[3])*dz[42]*(L2*M3+LC2*M2)
            res[41] = z[31]*(L1*dz[10]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))-L1*z[10]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3])))+z[32]*(sin(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)+L2*cos(z[2])*dz[20]+L2*cos(z[2])*dz[23]+L2*cos(z[3])*sin(z[2])*dz[21]+L2*cos(z[3])*sin(z[2])*dz[24]+L2*sin(z[2])*sin(z[3])*dz[19]+L2*sin(z[2])*sin(z[3])*dz[22]+L1*dz[10]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))+0.0*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2])))+dz[41]*(J2+L2^2*M3+LC2^2*M2)+γ*z[41]+z[33]*(L2*cos(z[2])*sin(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[22]-L2*cos(z[2])*cos(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[24]+0.0*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[2])*sin(z[3])*dz[10]*(L2*M3+LC2*M2))+L2*sin(z[2])*dz[50]+L2*sin(z[2])*dz[53]-L2*cos(z[2])*cos(z[3])*dz[51]-L2*cos(z[2])*cos(z[3])*dz[54]-L2*cos(z[2])*sin(z[3])*dz[49]-L2*cos(z[2])*sin(z[3])*dz[52]+L1*dz[40]*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))-2*cos(z[2])*sin(z[2])*z[42]*z[12]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[40]*z[10]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))
            res[42] = z[33]*(L2*cos(z[3])*sin(z[2])*dz[21]+L2*cos(z[3])*sin(z[2])*dz[24]+L2*sin(z[2])*sin(z[3])*dz[19]+L2*sin(z[2])*sin(z[3])*dz[22]+0.0*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[10]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[3])*sin(z[2])*dz[10]*(L2*M3+LC2*M2))+z[32]*(L2*cos(z[2])*sin(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[22]-L2*cos(z[2])*cos(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[24]+2*cos(z[2])*sin(z[2])*dz[12]*(J2+L2^2*M3+LC2^2*M2)+2*cos(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+0.0*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[2])*sin(z[3])*dz[10]*(L2*M3+LC2*M2))+z[42]*(γ+sin(2*z[2])*z[11]*(J2+L2^2*M3+LC2^2*M2))+z[31]*(L1*cos(z[1])*sin(z[2])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)+L1*sin(z[1])*sin(z[2])*sin(z[3])*dz[10]*(L2*M3+LC2*M2))+sin(z[2])^2*dz[42]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[3])*sin(z[2])*dz[49]-L2*cos(z[3])*sin(z[2])*dz[52]+L2*sin(z[2])*sin(z[3])*dz[51]+L2*sin(z[2])*sin(z[3])*dz[54]+sin(2*z[2])*z[41]*z[12]*(J2+L2^2*M3+LC2^2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*dz[40]*(L2*M3+LC2*M2)+2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[40]*z[10]*(L2*M3+LC2*M2)
            res[43] = z[44]*(2*L1*z[14]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[15]*(L2*M3+LC2*M2))+dz[43]*(J1+L1^2*(M2+M3)+LC1^2*M1)-z[45]*(2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]*(L2*M3+LC2*M2))-z[36]*(L1*cos(z[4])*cos(z[5])*sin(z[6])*dz[14]*(L2*M3+LC2*M2)-L1*cos(z[4])*sin(z[5])*sin(z[6])*z[15]^2*(L2*M3+LC2*M2)-L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*cos(z[6])*sin(z[5])*dz[15]*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[5])*cos(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+γ*z[43]-z[35]*(L1*z[14]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))-L1*dz[14]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))+L1*cos(z[4])*cos(z[5])*cos(z[6])*z[15]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*cos(z[5])*sin(z[6])*dz[15]*(L2*M3+LC2*M2)-2*L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+z[34]*(0.0*sin(z[4])*(L1*(M2+M3)+LC1*M1)+(L1*cos(z[4])*dz[20])*0.5-L1*sin(z[4])*dz[21]+(sqrt(3)*L1*cos(z[4])*dz[19])*0.5+L1*dz[14]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5]))+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)+L1*sin(z[4])*sin(z[5])*sin(z[6])*dz[15]*(L2*M3+LC2*M2)+2*L1*cos(z[5])*sin(z[4])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+L1*cos(z[4])*dz[51]+(L1*sin(z[4])*dz[50])*0.5+(sqrt(3)*L1*sin(z[4])*dz[49])*0.5+L1*dz[44]*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))-L1*cos(z[4])*sin(z[5])*sin(z[6])*dz[45]*(L2*M3+LC2*M2)
            res[44] = z[34]*(L1*dz[13]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))-L1*z[13]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6])))-z[36]*((L2*cos(z[5])*cos(z[6])*dz[19])*0.5+L2*cos(z[5])*sin(z[6])*dz[21]-0.0*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)-(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[20])*0.5-L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*cos(z[5])*sin(z[6])*dz[13]*(L2*M3+LC2*M2))-dz[49]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)+dz[50]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)+dz[44]*(J2+L2^2*M3+LC2^2*M2)+γ*z[44]+z[35]*(dz[19]*((sqrt(3)*L2*cos(z[5]))*0.5+(L2*sin(z[5])*sin(z[6]))*0.5)+dz[20]*((L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5)-cos(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)+sin(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[6])*sin(z[5])*dz[21]+L1*dz[13]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))+0.0*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5])))+L2*cos(z[5])*cos(z[6])*dz[51]+L1*dz[43]*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))-2*cos(z[5])*sin(z[5])*z[45]*z[15]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[43]*z[13]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))
            res[45] = z[45]*(γ+sin(2*z[5])*z[14]*(J2+L2^2*M3+LC2^2*M2))-z[36]*(L2*cos(z[6])*sin(z[5])*dz[21]-(L2*sin(z[5])*sin(z[6])*dz[19])*0.5-0.0*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+(sqrt(3)*L2*sin(z[5])*sin(z[6])*dz[20])*0.5-L1*cos(z[6])*sin(z[4])*sin(z[5])*z[13]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*cos(z[6])*sin(z[5])*dz[13]*(L2*M3+LC2*M2))+z[35]*(2*cos(z[5])*sin(z[5])*dz[15]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[5])*sin(z[6])*dz[21]-(L2*cos(z[5])*cos(z[6])*dz[19])*0.5+2*cos(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)+0.0*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[20])*0.5+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[5])*sin(z[6])*dz[13]*(L2*M3+LC2*M2))+z[34]*(L1*cos(z[4])*sin(z[5])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)+L1*sin(z[4])*sin(z[5])*sin(z[6])*dz[13]*(L2*M3+LC2*M2))+sin(z[5])^2*dz[45]*(J2+L2^2*M3+LC2^2*M2)-(L2*cos(z[6])*sin(z[5])*dz[49])*0.5-L2*sin(z[5])*sin(z[6])*dz[51]+sin(2*z[5])*z[44]*z[15]*(J2+L2^2*M3+LC2^2*M2)+(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[50])*0.5-L1*cos(z[4])*sin(z[5])*sin(z[6])*dz[43]*(L2*M3+LC2*M2)+2*L1*sin(z[4])*sin(z[5])*sin(z[6])*z[43]*z[13]*(L2*M3+LC2*M2)
            res[46] = z[47]*(2*L1*z[17]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[18]*(L2*M3+LC2*M2))+dz[46]*(J1+L1^2*(M2+M3)+LC1^2*M1)-z[48]*(2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]*(L2*M3+LC2*M2))-z[39]*(L1*cos(z[7])*cos(z[8])*sin(z[9])*dz[17]*(L2*M3+LC2*M2)-L1*cos(z[7])*sin(z[8])*sin(z[9])*z[18]^2*(L2*M3+LC2*M2)-L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*cos(z[9])*sin(z[8])*dz[18]*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[8])*cos(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+γ*z[46]-z[38]*(L1*z[17]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))-L1*dz[17]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))+L1*cos(z[7])*cos(z[8])*cos(z[9])*z[18]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*cos(z[8])*sin(z[9])*dz[18]*(L2*M3+LC2*M2)-2*L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+z[37]*(0.0*sin(z[7])*(L1*(M2+M3)+LC1*M1)+(L1*cos(z[7])*dz[23])*0.5-L1*sin(z[7])*dz[24]-(sqrt(3)*L1*cos(z[7])*dz[22])*0.5+L1*dz[17]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8]))+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)+L1*sin(z[7])*sin(z[8])*sin(z[9])*dz[18]*(L2*M3+LC2*M2)+2*L1*cos(z[8])*sin(z[7])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+L1*cos(z[7])*dz[54]+(L1*sin(z[7])*dz[53])*0.5-(sqrt(3)*L1*sin(z[7])*dz[52])*0.5+L1*dz[47]*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))-L1*cos(z[7])*sin(z[8])*sin(z[9])*dz[48]*(L2*M3+LC2*M2)
            res[47] = z[37]*(L1*dz[16]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))-L1*z[16]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9])))-z[39]*((L2*cos(z[8])*cos(z[9])*dz[22])*0.5+L2*cos(z[8])*sin(z[9])*dz[24]-0.0*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[23])*0.5-L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*cos(z[8])*sin(z[9])*dz[16]*(L2*M3+LC2*M2))-dz[52]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)+dz[53]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)+dz[47]*(J2+L2^2*M3+LC2^2*M2)+γ*z[47]+z[38]*(dz[23]*((L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5)-dz[22]*((sqrt(3)*L2*cos(z[8]))*0.5-(L2*sin(z[8])*sin(z[9]))*0.5)-cos(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)+sin(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[9])*sin(z[8])*dz[24]+L1*dz[16]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))+0.0*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8])))+L2*cos(z[8])*cos(z[9])*dz[54]+L1*dz[46]*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))-2*cos(z[8])*sin(z[8])*z[48]*z[18]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[46]*z[16]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))
            res[48] = z[39]*((L2*sin(z[8])*sin(z[9])*dz[22])*0.5-L2*cos(z[9])*sin(z[8])*dz[24]+0.0*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+(sqrt(3)*L2*sin(z[8])*sin(z[9])*dz[23])*0.5+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[16]^2*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[9])*sin(z[8])*dz[16]*(L2*M3+LC2*M2))+z[48]*(γ+sin(2*z[8])*z[17]*(J2+L2^2*M3+LC2^2*M2))-z[38]*((L2*cos(z[8])*cos(z[9])*dz[22])*0.5+L2*cos(z[8])*sin(z[9])*dz[24]-2*cos(z[8])*sin(z[8])*dz[18]*(J2+L2^2*M3+LC2^2*M2)-2*cos(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)-0.0*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[23])*0.5-L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*cos(z[8])*sin(z[9])*dz[16]*(L2*M3+LC2*M2))+z[37]*(L1*cos(z[7])*sin(z[8])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)+L1*sin(z[7])*sin(z[8])*sin(z[9])*dz[16]*(L2*M3+LC2*M2))+sin(z[8])^2*dz[48]*(J2+L2^2*M3+LC2^2*M2)-(L2*cos(z[9])*sin(z[8])*dz[52])*0.5-L2*sin(z[8])*sin(z[9])*dz[54]+sin(2*z[8])*z[47]*z[18]*(J2+L2^2*M3+LC2^2*M2)-(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[53])*0.5-L1*cos(z[7])*sin(z[8])*sin(z[9])*dz[46]*(L2*M3+LC2*M2)+2*L1*sin(z[7])*sin(z[8])*sin(z[9])*z[46]*z[16]*(L2*M3+LC2*M2)
            res[49] = z[35]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)+L2*cos(z[2])*sin(z[3])*z[32]+L2*cos(z[3])*sin(z[2])*z[33]+(L2*cos(z[6])*sin(z[5])*z[36])*0.5-(sqrt(3)*L1*sin(z[4])*z[34])*0.5
            res[50] = -z[35]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-L1*sin(z[1])*z[31]-L2*sin(z[2])*z[32]-(L1*sin(z[4])*z[34])*0.5-(sqrt(3)*L2*cos(z[6])*sin(z[5])*z[36])*0.5
            res[51] = L1*cos(z[1])*z[31]-L1*cos(z[4])*z[34]+L2*cos(z[2])*cos(z[3])*z[32]-L2*cos(z[5])*cos(z[6])*z[35]-L2*sin(z[2])*sin(z[3])*z[33]+L2*sin(z[5])*sin(z[6])*z[36]
            res[52] = z[38]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)+L2*cos(z[2])*sin(z[3])*z[32]+L2*cos(z[3])*sin(z[2])*z[33]+(L2*cos(z[9])*sin(z[8])*z[39])*0.5+(sqrt(3)*L1*sin(z[7])*z[37])*0.5
            res[53] = (sqrt(3)*L2*cos(z[9])*sin(z[8])*z[39])*0.5-L1*sin(z[1])*z[31]-L2*sin(z[2])*z[32]-(L1*sin(z[7])*z[37])*0.5-z[38]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)
            res[54] = L1*cos(z[1])*z[31]-L1*cos(z[7])*z[37]+L2*cos(z[2])*cos(z[3])*z[32]-L2*cos(z[8])*cos(z[9])*z[38]-L2*sin(z[2])*sin(z[3])*z[33]+L2*sin(z[8])*sin(z[9])*z[39]
            res[55] = z[32]*(L2*cos(z[2])*cos(z[3])*z[12]-L2*sin(z[2])*sin(z[3])*z[11])+z[33]*(L2*cos(z[2])*cos(z[3])*z[11]-L2*sin(z[2])*sin(z[3])*z[12])+z[36]*((L2*cos(z[5])*cos(z[6])*z[14])*0.5-(L2*sin(z[5])*sin(z[6])*z[15])*0.5)-z[35]*((sqrt(3)*L2*cos(z[5])*z[14])*0.5-(L2*cos(z[5])*cos(z[6])*z[15])*0.5+(L2*sin(z[5])*sin(z[6])*z[14])*0.5)+z[44]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)+L2*cos(z[2])*sin(z[3])*z[41]+L2*cos(z[3])*sin(z[2])*z[42]+(L2*cos(z[6])*sin(z[5])*z[45])*0.5-(sqrt(3)*L1*sin(z[4])*z[43])*0.5-(sqrt(3)*L1*cos(z[4])*z[34]*z[13])*0.5
            res[56] = -z[44]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-z[35]*((L2*cos(z[5])*z[14])*0.5+(sqrt(3)*L2*cos(z[5])*cos(z[6])*z[15])*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6])*z[14])*0.5)-z[36]*((sqrt(3)*L2*cos(z[5])*cos(z[6])*z[14])*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6])*z[15])*0.5)-L1*sin(z[1])*z[40]-L2*sin(z[2])*z[41]-(L1*sin(z[4])*z[43])*0.5-L1*cos(z[1])*z[31]*z[10]-L2*cos(z[2])*z[32]*z[11]-(L1*cos(z[4])*z[34]*z[13])*0.5-(sqrt(3)*L2*cos(z[6])*sin(z[5])*z[45])*0.5
            res[57] = z[35]*(L2*cos(z[6])*sin(z[5])*z[14]+L2*cos(z[5])*sin(z[6])*z[15])-z[33]*(L2*cos(z[2])*sin(z[3])*z[11]+L2*cos(z[3])*sin(z[2])*z[12])-z[32]*(L2*cos(z[3])*sin(z[2])*z[11]+L2*cos(z[2])*sin(z[3])*z[12])+z[36]*(L2*cos(z[5])*sin(z[6])*z[14]+L2*cos(z[6])*sin(z[5])*z[15])+L1*cos(z[1])*z[40]-L1*cos(z[4])*z[43]+L2*cos(z[2])*cos(z[3])*z[41]-L2*cos(z[5])*cos(z[6])*z[44]-L2*sin(z[2])*sin(z[3])*z[42]+L2*sin(z[5])*sin(z[6])*z[45]-L1*sin(z[1])*z[31]*z[10]+L1*sin(z[4])*z[34]*z[13]
            res[58] = z[32]*(L2*cos(z[2])*cos(z[3])*z[12]-L2*sin(z[2])*sin(z[3])*z[11])+z[33]*(L2*cos(z[2])*cos(z[3])*z[11]-L2*sin(z[2])*sin(z[3])*z[12])+z[39]*((L2*cos(z[8])*cos(z[9])*z[17])*0.5-(L2*sin(z[8])*sin(z[9])*z[18])*0.5)+z[38]*((L2*cos(z[8])*cos(z[9])*z[18])*0.5+(sqrt(3)*L2*cos(z[8])*z[17])*0.5-(L2*sin(z[8])*sin(z[9])*z[17])*0.5)+z[47]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)+L2*cos(z[2])*sin(z[3])*z[41]+L2*cos(z[3])*sin(z[2])*z[42]+(L2*cos(z[9])*sin(z[8])*z[48])*0.5+(sqrt(3)*L1*sin(z[7])*z[46])*0.5+(sqrt(3)*L1*cos(z[7])*z[37]*z[16])*0.5
            res[59] = z[39]*((sqrt(3)*L2*cos(z[8])*cos(z[9])*z[17])*0.5-(sqrt(3)*L2*sin(z[8])*sin(z[9])*z[18])*0.5)-z[38]*((L2*cos(z[8])*z[17])*0.5-(sqrt(3)*L2*cos(z[8])*cos(z[9])*z[18])*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9])*z[17])*0.5)-z[47]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)-L1*sin(z[1])*z[40]-L2*sin(z[2])*z[41]-(L1*sin(z[7])*z[46])*0.5-L1*cos(z[1])*z[31]*z[10]-L2*cos(z[2])*z[32]*z[11]-(L1*cos(z[7])*z[37]*z[16])*0.5+(sqrt(3)*L2*cos(z[9])*sin(z[8])*z[48])*0.5
            res[60] = z[38]*(L2*cos(z[9])*sin(z[8])*z[17]+L2*cos(z[8])*sin(z[9])*z[18])-z[33]*(L2*cos(z[2])*sin(z[3])*z[11]+L2*cos(z[3])*sin(z[2])*z[12])-z[32]*(L2*cos(z[3])*sin(z[2])*z[11]+L2*cos(z[2])*sin(z[3])*z[12])+z[39]*(L2*cos(z[8])*sin(z[9])*z[17]+L2*cos(z[9])*sin(z[8])*z[18])+L1*cos(z[1])*z[40]-L1*cos(z[7])*z[46]+L2*cos(z[2])*cos(z[3])*z[41]-L2*cos(z[8])*cos(z[9])*z[47]-L2*sin(z[2])*sin(z[3])*z[42]+L2*sin(z[8])*sin(z[9])*z[48]-L1*sin(z[1])*z[31]*z[10]+L1*sin(z[7])*z[37]*z[16]
            # Parameter-specific part for L1
            res[31] += cos(z[1])*dz[27]+cos(z[1])*dz[30]-sin(z[1])*dz[26]-sin(z[1])*dz[29]
            res[34] += -cos(z[4])*dz[27]-(sin(z[4])*dz[26])*0.5-(sqrt(3)*sin(z[4])*dz[25])*0.5
            res[37] += (sqrt(3)*sin(z[7])*dz[28])*0.5-(sin(z[7])*dz[29])*0.5-cos(z[7])*dz[30]
            res[40] += sin(z[1])*dz[20]-cos(z[1])*dz[24]-cos(z[1])*dz[21]+sin(z[1])*dz[23]-0.0*cos(z[1])*(M2+M3)+dz[11]*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))+2*L1*dz[10]*(M2+M3)+z[11]^2*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-cos(z[1])*sin(z[2])*sin(z[3])*dz[12]*(L2*M3+LC2*M2)-cos(z[1])*cos(z[3])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)-2*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2)
            res[41] += dz[10]*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))+z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))
            res[42] += sin(z[1])*sin(z[2])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)-cos(z[1])*sin(z[2])*sin(z[3])*dz[10]*(L2*M3+LC2*M2)
            res[43] += cos(z[4])*dz[21]+(sin(z[4])*dz[20])*0.5-0.0*cos(z[4])*(M2+M3)+dz[14]*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))+2*L1*dz[13]*(M2+M3)+z[14]^2*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))+(sqrt(3)*sin(z[4])*dz[19])*0.5-cos(z[4])*sin(z[5])*sin(z[6])*dz[15]*(L2*M3+LC2*M2)-cos(z[4])*cos(z[6])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)-2*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2)
            res[44] += dz[13]*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))+z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))
            res[45] += sin(z[4])*sin(z[5])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)-cos(z[4])*sin(z[5])*sin(z[6])*dz[13]*(L2*M3+LC2*M2)
            res[46] += cos(z[7])*dz[24]+(sin(z[7])*dz[23])*0.5-0.0*cos(z[7])*(M2+M3)+dz[17]*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))+2*L1*dz[16]*(M2+M3)+z[17]^2*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-(sqrt(3)*sin(z[7])*dz[22])*0.5-cos(z[7])*sin(z[8])*sin(z[9])*dz[18]*(L2*M3+LC2*M2)-cos(z[7])*cos(z[9])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)-2*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2)
            res[47] += dz[16]*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))+z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))
            res[48] += sin(z[7])*sin(z[8])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)-cos(z[7])*sin(z[8])*sin(z[9])*dz[16]*(L2*M3+LC2*M2)
            res[49] += (sqrt(3)*cos(z[4]))*0.5
            res[50] += cos(z[1])+cos(z[4])*0.5
            res[51] += sin(z[1])-sin(z[4])
            res[52] += -(sqrt(3)*cos(z[7]))*0.5
            res[53] += cos(z[1])+cos(z[7])*0.5
            res[54] += sin(z[1])-sin(z[7])
            res[55] += -(sqrt(3)*sin(z[4])*z[13])*0.5
            res[56] += -sin(z[1])*z[10]-(sin(z[4])*z[13])*0.5
            res[57] += cos(z[1])*z[10]-cos(z[4])*z[13]
            res[58] += (sqrt(3)*sin(z[7])*z[16])*0.5
            res[59] += -sin(z[1])*z[10]-(sin(z[7])*z[16])*0.5
            res[60] += cos(z[1])*z[10]-cos(z[7])*z[16]

            nothing
        end

        z0, dz0 = get_delta_initial_L1sens_comp(θ, u(0.0), w(0.0))

        dvars = fill(true, 60)
        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 for L1 forward sensitivity delta robot is: $r0"

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, z0, dz0, dvars, r0)
    end
end

function delta_robot_gc_J1sens(Φ::Float64, u::Function, w::Function, θ::Vector{Float64})::Model
    let L0 = θ[1], L1 = θ[2], L2 = θ[3], L3 = θ[4], LC1 = θ[5], LC2 = θ[6], M1 = θ[7], M2 = θ[8], M3 = θ[9], J1 = θ[10], J2 = θ[11], g = 0.0, γ = θ[13]
        # @info "Samples: $(u(0.3)), $(u(0.46)), $(u(1.23)), $(w(0.3)), $(w(0.46)), $(w(1.23))"
        function f!(res, dz, z, _, t)
            ut = u(t)
            wt = w(t)

            res[1] = dz[1]-z[10]+L1*cos(z[1])*dz[27]+L1*cos(z[1])*dz[30]-L1*sin(z[1])*dz[26]-L1*sin(z[1])*dz[29]
            res[2] = dz[2]-z[11]-L2*sin(z[2])*dz[26]-L2*sin(z[2])*dz[29]+L2*cos(z[2])*cos(z[3])*dz[27]+L2*cos(z[2])*cos(z[3])*dz[30]+L2*cos(z[2])*sin(z[3])*dz[25]+L2*cos(z[2])*sin(z[3])*dz[28]
            res[3] = dz[3]-z[12]+L2*cos(z[3])*sin(z[2])*dz[25]+L2*cos(z[3])*sin(z[2])*dz[28]-L2*sin(z[2])*sin(z[3])*dz[27]-L2*sin(z[2])*sin(z[3])*dz[30]
            res[4] = dz[4]-z[13]-L1*cos(z[4])*dz[27]-(L1*sin(z[4])*dz[26])*0.5-(sqrt(3)*L1*sin(z[4])*dz[25])*0.5
            res[5] = dz[5]-z[14]+dz[25]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)-dz[26]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-L2*cos(z[5])*cos(z[6])*dz[27]
            res[6] = dz[6]-z[15]+(L2*cos(z[6])*sin(z[5])*dz[25])*0.5+L2*sin(z[5])*sin(z[6])*dz[27]-(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[26])*0.5
            res[7] = dz[7]-z[16]-L1*cos(z[7])*dz[30]-(L1*sin(z[7])*dz[29])*0.5+(sqrt(3)*L1*sin(z[7])*dz[28])*0.5
            res[8] = dz[8]-z[17]+dz[28]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)-dz[29]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)-L2*cos(z[8])*cos(z[9])*dz[30]
            res[9] = dz[9]-z[18]+(L2*cos(z[9])*sin(z[8])*dz[28])*0.5+L2*sin(z[8])*sin(z[9])*dz[30]+(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[29])*0.5
            res[10] = dz[10]*(J1+L1^2*(M2+M3)+LC1^2*M1)-ut[1]-wt[1]^2+γ*z[10]-0.0*cos(z[1])*(L1*(M2+M3)+LC1*M1)-L1*cos(z[1])*dz[21]-L1*cos(z[1])*dz[24]+L1*sin(z[1])*dz[20]+L1*sin(z[1])*dz[23]+L1*dz[11]*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*dz[12]*(L2*M3+LC2*M2)-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2)
            res[11] = dz[11]*(J2+L2^2*M3+LC2^2*M2)+γ*z[11]+L2*sin(z[2])*dz[20]+L2*sin(z[2])*dz[23]-L2*cos(z[2])*cos(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[24]-L2*cos(z[2])*sin(z[3])*dz[19]-L2*cos(z[2])*sin(z[3])*dz[22]+L1*dz[10]*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))-cos(z[2])*sin(z[2])*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-0.0*cos(z[2])*cos(z[3])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))
            res[12] = γ*z[12]+sin(z[2])^2*dz[12]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[3])*sin(z[2])*dz[19]-L2*cos(z[3])*sin(z[2])*dz[22]+L2*sin(z[2])*sin(z[3])*dz[21]+L2*sin(z[2])*sin(z[3])*dz[24]+sin(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+0.0*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*sin(z[1])*sin(z[2])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*dz[10]*(L2*M3+LC2*M2)
            res[13] = dz[13]*(J1+L1^2*(M2+M3)+LC1^2*M1)-ut[2]-wt[2]^2+γ*z[13]-0.0*cos(z[4])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[4])*dz[21]+(L1*sin(z[4])*dz[20])*0.5+(sqrt(3)*L1*sin(z[4])*dz[19])*0.5+L1*dz[14]*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)-L1*cos(z[4])*sin(z[5])*sin(z[6])*dz[15]*(L2*M3+LC2*M2)-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2)
            res[14] = dz[20]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-dz[19]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)+dz[14]*(J2+L2^2*M3+LC2^2*M2)+γ*z[14]+L2*cos(z[5])*cos(z[6])*dz[21]+L1*dz[13]*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))-cos(z[5])*sin(z[5])*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-0.0*cos(z[5])*cos(z[6])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))
            res[15] = γ*z[15]+sin(z[5])^2*dz[15]*(J2+L2^2*M3+LC2^2*M2)-(L2*cos(z[6])*sin(z[5])*dz[19])*0.5-L2*sin(z[5])*sin(z[6])*dz[21]+sin(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)+0.0*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[20])*0.5+L1*sin(z[4])*sin(z[5])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)-L1*cos(z[4])*sin(z[5])*sin(z[6])*dz[13]*(L2*M3+LC2*M2)
            res[16] = dz[16]*(J1+L1^2*(M2+M3)+LC1^2*M1)-ut[3]-wt[3]^2+γ*z[16]-0.0*cos(z[7])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[7])*dz[24]+(L1*sin(z[7])*dz[23])*0.5-(sqrt(3)*L1*sin(z[7])*dz[22])*0.5+L1*dz[17]*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)-L1*cos(z[7])*sin(z[8])*sin(z[9])*dz[18]*(L2*M3+LC2*M2)-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2)
            res[17] = dz[23]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)-dz[22]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)+dz[17]*(J2+L2^2*M3+LC2^2*M2)+γ*z[17]+L2*cos(z[8])*cos(z[9])*dz[24]+L1*dz[16]*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))-cos(z[8])*sin(z[8])*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-0.0*cos(z[8])*cos(z[9])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))
            res[18] = γ*z[18]+sin(z[8])^2*dz[18]*(J2+L2^2*M3+LC2^2*M2)-(L2*cos(z[9])*sin(z[8])*dz[22])*0.5-L2*sin(z[8])*sin(z[9])*dz[24]+sin(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)+0.0*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)-(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[23])*0.5+L1*sin(z[7])*sin(z[8])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)-L1*cos(z[7])*sin(z[8])*sin(z[9])*dz[16]*(L2*M3+LC2*M2)
            res[19] = (sqrt(3)*(L0-L3+L1*cos(z[4])+L2*cos(z[5])))*0.5+L2*sin(z[2])*sin(z[3])+(L2*sin(z[5])*sin(z[6]))*0.5
            res[20] = (3*L0)*0.5-(3*L3)*0.5+L1*cos(z[1])+L2*cos(z[2])+(L1*cos(z[4]))*0.5+(L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5
            res[21] = L1*sin(z[1])-L1*sin(z[4])+L2*cos(z[3])*sin(z[2])-L2*cos(z[6])*sin(z[5])
            res[22] = L2*sin(z[2])*sin(z[3])-(sqrt(3)*(L0-L3+L1*cos(z[7])+L2*cos(z[8])))*0.5+(L2*sin(z[8])*sin(z[9]))*0.5
            res[23] = (3*L0)*0.5-(3*L3)*0.5+L1*cos(z[1])+L2*cos(z[2])+(L1*cos(z[7]))*0.5+(L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5
            res[24] = L1*sin(z[1])-L1*sin(z[7])+L2*cos(z[3])*sin(z[2])-L2*cos(z[9])*sin(z[8])
            res[25] = L2*cos(z[2])*sin(z[3])*z[11]-(sqrt(3)*(L1*sin(z[4])*z[13]+L2*sin(z[5])*z[14]))*0.5+L2*cos(z[3])*sin(z[2])*z[12]+(L2*cos(z[5])*sin(z[6])*z[14])*0.5+(L2*cos(z[6])*sin(z[5])*z[15])*0.5
            res[26] = -L1*sin(z[1])*z[10]-L2*sin(z[2])*z[11]-(L1*sin(z[4])*z[13])*0.5-(L2*sin(z[5])*z[14])*0.5-(sqrt(3)*L2*cos(z[5])*sin(z[6])*z[14])*0.5-(sqrt(3)*L2*cos(z[6])*sin(z[5])*z[15])*0.5
            res[27] = L1*cos(z[1])*z[10]-L1*cos(z[4])*z[13]+L2*cos(z[2])*cos(z[3])*z[11]-L2*cos(z[5])*cos(z[6])*z[14]-L2*sin(z[2])*sin(z[3])*z[12]+L2*sin(z[5])*sin(z[6])*z[15]
            res[28] = (sqrt(3)*(L1*sin(z[7])*z[16]+L2*sin(z[8])*z[17]))*0.5+L2*cos(z[2])*sin(z[3])*z[11]+L2*cos(z[3])*sin(z[2])*z[12]+(L2*cos(z[8])*sin(z[9])*z[17])*0.5+(L2*cos(z[9])*sin(z[8])*z[18])*0.5
            res[29] = (sqrt(3)*L2*cos(z[8])*sin(z[9])*z[17])*0.5-L2*sin(z[2])*z[11]-(L1*sin(z[7])*z[16])*0.5-(L2*sin(z[8])*z[17])*0.5-L1*sin(z[1])*z[10]+(sqrt(3)*L2*cos(z[9])*sin(z[8])*z[18])*0.5
            res[30] = L1*cos(z[1])*z[10]-L1*cos(z[7])*z[16]+L2*cos(z[2])*cos(z[3])*z[11]-L2*cos(z[8])*cos(z[9])*z[17]-L2*sin(z[2])*sin(z[3])*z[12]+L2*sin(z[8])*sin(z[9])*z[18]
            # Sensitivity equations
            res[31] = dz[31]-z[40]-z[31]*(L1*cos(z[1])*dz[26]+L1*cos(z[1])*dz[29]+L1*sin(z[1])*dz[27]+L1*sin(z[1])*dz[30])+L1*cos(z[1])*dz[57]+L1*cos(z[1])*dz[60]-L1*sin(z[1])*dz[56]-L1*sin(z[1])*dz[59]
            res[32] = dz[32]-z[41]+z[33]*(L2*cos(z[2])*cos(z[3])*dz[25]+L2*cos(z[2])*cos(z[3])*dz[28]-L2*cos(z[2])*sin(z[3])*dz[27]-L2*cos(z[2])*sin(z[3])*dz[30])-z[32]*(L2*cos(z[2])*dz[26]+L2*cos(z[2])*dz[29]+L2*cos(z[3])*sin(z[2])*dz[27]+L2*cos(z[3])*sin(z[2])*dz[30]+L2*sin(z[2])*sin(z[3])*dz[25]+L2*sin(z[2])*sin(z[3])*dz[28])-L2*sin(z[2])*dz[56]-L2*sin(z[2])*dz[59]+L2*cos(z[2])*cos(z[3])*dz[57]+L2*cos(z[2])*cos(z[3])*dz[60]+L2*cos(z[2])*sin(z[3])*dz[55]+L2*cos(z[2])*sin(z[3])*dz[58]
            res[33] = dz[33]-z[42]+z[32]*(L2*cos(z[2])*cos(z[3])*dz[25]+L2*cos(z[2])*cos(z[3])*dz[28]-L2*cos(z[2])*sin(z[3])*dz[27]-L2*cos(z[2])*sin(z[3])*dz[30])-z[33]*(L2*cos(z[3])*sin(z[2])*dz[27]+L2*cos(z[3])*sin(z[2])*dz[30]+L2*sin(z[2])*sin(z[3])*dz[25]+L2*sin(z[2])*sin(z[3])*dz[28])+L2*cos(z[3])*sin(z[2])*dz[55]+L2*cos(z[3])*sin(z[2])*dz[58]-L2*sin(z[2])*sin(z[3])*dz[57]-L2*sin(z[2])*sin(z[3])*dz[60]
            res[34] = dz[34]-z[43]-z[34]*((L1*cos(z[4])*dz[26])*0.5-L1*sin(z[4])*dz[27]+(sqrt(3)*L1*cos(z[4])*dz[25])*0.5)-L1*cos(z[4])*dz[57]-(L1*sin(z[4])*dz[56])*0.5-(sqrt(3)*L1*sin(z[4])*dz[55])*0.5
            res[35] = dz[35]-z[44]+dz[55]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)-dz[56]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)+z[36]*((L2*cos(z[5])*cos(z[6])*dz[25])*0.5+L2*cos(z[5])*sin(z[6])*dz[27]-(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[26])*0.5)-z[35]*(dz[25]*((sqrt(3)*L2*cos(z[5]))*0.5+(L2*sin(z[5])*sin(z[6]))*0.5)+dz[26]*((L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5)-L2*cos(z[6])*sin(z[5])*dz[27])-L2*cos(z[5])*cos(z[6])*dz[57]
            res[36] = dz[36]-z[45]+z[35]*((L2*cos(z[5])*cos(z[6])*dz[25])*0.5+L2*cos(z[5])*sin(z[6])*dz[27]-(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[26])*0.5)+z[36]*(L2*cos(z[6])*sin(z[5])*dz[27]-(L2*sin(z[5])*sin(z[6])*dz[25])*0.5+(sqrt(3)*L2*sin(z[5])*sin(z[6])*dz[26])*0.5)+(L2*cos(z[6])*sin(z[5])*dz[55])*0.5+L2*sin(z[5])*sin(z[6])*dz[57]-(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[56])*0.5
            res[37] = dz[37]-z[46]+z[37]*(L1*sin(z[7])*dz[30]-(L1*cos(z[7])*dz[29])*0.5+(sqrt(3)*L1*cos(z[7])*dz[28])*0.5)-L1*cos(z[7])*dz[60]-(L1*sin(z[7])*dz[59])*0.5+(sqrt(3)*L1*sin(z[7])*dz[58])*0.5
            res[38] = dz[38]-z[47]+dz[58]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)-dz[59]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)+z[39]*((L2*cos(z[8])*cos(z[9])*dz[28])*0.5+L2*cos(z[8])*sin(z[9])*dz[30]+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[29])*0.5)+z[38]*(dz[28]*((sqrt(3)*L2*cos(z[8]))*0.5-(L2*sin(z[8])*sin(z[9]))*0.5)-dz[29]*((L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5)+L2*cos(z[9])*sin(z[8])*dz[30])-L2*cos(z[8])*cos(z[9])*dz[60]
            res[39] = dz[39]-z[48]+z[38]*((L2*cos(z[8])*cos(z[9])*dz[28])*0.5+L2*cos(z[8])*sin(z[9])*dz[30]+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[29])*0.5)-z[39]*((L2*sin(z[8])*sin(z[9])*dz[28])*0.5-L2*cos(z[9])*sin(z[8])*dz[30]+(sqrt(3)*L2*sin(z[8])*sin(z[9])*dz[29])*0.5)+(L2*cos(z[9])*sin(z[8])*dz[58])*0.5+L2*sin(z[8])*sin(z[9])*dz[60]+(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[59])*0.5
            res[40] = z[41]*(2*L1*z[11]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[12]*(L2*M3+LC2*M2))+dz[40]*(J1+L1^2*(M2+M3)+LC1^2*M1)-z[42]*(2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]*(L2*M3+LC2*M2))-z[33]*(L1*cos(z[1])*cos(z[2])*sin(z[3])*dz[11]*(L2*M3+LC2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*z[12]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]^2*(L2*M3+LC2*M2)+L1*cos(z[1])*cos(z[3])*sin(z[2])*dz[12]*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[2])*cos(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+z[31]*(0.0*sin(z[1])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[1])*dz[20]+L1*cos(z[1])*dz[23]+L1*sin(z[1])*dz[21]+L1*sin(z[1])*dz[24]+L1*dz[11]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2]))+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)+L1*sin(z[1])*sin(z[2])*sin(z[3])*dz[12]*(L2*M3+LC2*M2)+2*L1*cos(z[2])*sin(z[1])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+γ*z[40]-z[32]*(L1*z[11]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))-L1*dz[11]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))+L1*cos(z[1])*cos(z[2])*cos(z[3])*z[12]^2*(L2*M3+LC2*M2)+L1*cos(z[1])*cos(z[2])*sin(z[3])*dz[12]*(L2*M3+LC2*M2)-2*L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))-L1*cos(z[1])*dz[51]-L1*cos(z[1])*dz[54]+L1*sin(z[1])*dz[50]+L1*sin(z[1])*dz[53]+L1*dz[41]*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))-L1*cos(z[1])*sin(z[2])*sin(z[3])*dz[42]*(L2*M3+LC2*M2)
            res[41] = z[31]*(L1*dz[10]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))-L1*z[10]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3])))+z[32]*(sin(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)+L2*cos(z[2])*dz[20]+L2*cos(z[2])*dz[23]+L2*cos(z[3])*sin(z[2])*dz[21]+L2*cos(z[3])*sin(z[2])*dz[24]+L2*sin(z[2])*sin(z[3])*dz[19]+L2*sin(z[2])*sin(z[3])*dz[22]+L1*dz[10]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))+0.0*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2])))+dz[41]*(J2+L2^2*M3+LC2^2*M2)+γ*z[41]+z[33]*(L2*cos(z[2])*sin(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[22]-L2*cos(z[2])*cos(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[24]+0.0*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[2])*sin(z[3])*dz[10]*(L2*M3+LC2*M2))+L2*sin(z[2])*dz[50]+L2*sin(z[2])*dz[53]-L2*cos(z[2])*cos(z[3])*dz[51]-L2*cos(z[2])*cos(z[3])*dz[54]-L2*cos(z[2])*sin(z[3])*dz[49]-L2*cos(z[2])*sin(z[3])*dz[52]+L1*dz[40]*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))-2*cos(z[2])*sin(z[2])*z[42]*z[12]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[40]*z[10]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))
            res[42] = z[33]*(L2*cos(z[3])*sin(z[2])*dz[21]+L2*cos(z[3])*sin(z[2])*dz[24]+L2*sin(z[2])*sin(z[3])*dz[19]+L2*sin(z[2])*sin(z[3])*dz[22]+0.0*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[10]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[3])*sin(z[2])*dz[10]*(L2*M3+LC2*M2))+z[32]*(L2*cos(z[2])*sin(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[22]-L2*cos(z[2])*cos(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[24]+2*cos(z[2])*sin(z[2])*dz[12]*(J2+L2^2*M3+LC2^2*M2)+2*cos(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+0.0*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[2])*sin(z[3])*dz[10]*(L2*M3+LC2*M2))+z[42]*(γ+sin(2*z[2])*z[11]*(J2+L2^2*M3+LC2^2*M2))+z[31]*(L1*cos(z[1])*sin(z[2])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)+L1*sin(z[1])*sin(z[2])*sin(z[3])*dz[10]*(L2*M3+LC2*M2))+sin(z[2])^2*dz[42]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[3])*sin(z[2])*dz[49]-L2*cos(z[3])*sin(z[2])*dz[52]+L2*sin(z[2])*sin(z[3])*dz[51]+L2*sin(z[2])*sin(z[3])*dz[54]+sin(2*z[2])*z[41]*z[12]*(J2+L2^2*M3+LC2^2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*dz[40]*(L2*M3+LC2*M2)+2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[40]*z[10]*(L2*M3+LC2*M2)
            res[43] = z[44]*(2*L1*z[14]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[15]*(L2*M3+LC2*M2))+dz[43]*(J1+L1^2*(M2+M3)+LC1^2*M1)-z[45]*(2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]*(L2*M3+LC2*M2))-z[36]*(L1*cos(z[4])*cos(z[5])*sin(z[6])*dz[14]*(L2*M3+LC2*M2)-L1*cos(z[4])*sin(z[5])*sin(z[6])*z[15]^2*(L2*M3+LC2*M2)-L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*cos(z[6])*sin(z[5])*dz[15]*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[5])*cos(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+γ*z[43]-z[35]*(L1*z[14]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))-L1*dz[14]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))+L1*cos(z[4])*cos(z[5])*cos(z[6])*z[15]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*cos(z[5])*sin(z[6])*dz[15]*(L2*M3+LC2*M2)-2*L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+z[34]*(0.0*sin(z[4])*(L1*(M2+M3)+LC1*M1)+(L1*cos(z[4])*dz[20])*0.5-L1*sin(z[4])*dz[21]+(sqrt(3)*L1*cos(z[4])*dz[19])*0.5+L1*dz[14]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5]))+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)+L1*sin(z[4])*sin(z[5])*sin(z[6])*dz[15]*(L2*M3+LC2*M2)+2*L1*cos(z[5])*sin(z[4])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+L1*cos(z[4])*dz[51]+(L1*sin(z[4])*dz[50])*0.5+(sqrt(3)*L1*sin(z[4])*dz[49])*0.5+L1*dz[44]*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))-L1*cos(z[4])*sin(z[5])*sin(z[6])*dz[45]*(L2*M3+LC2*M2)
            res[44] = z[34]*(L1*dz[13]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))-L1*z[13]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6])))-z[36]*((L2*cos(z[5])*cos(z[6])*dz[19])*0.5+L2*cos(z[5])*sin(z[6])*dz[21]-0.0*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)-(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[20])*0.5-L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*cos(z[5])*sin(z[6])*dz[13]*(L2*M3+LC2*M2))-dz[49]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)+dz[50]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)+dz[44]*(J2+L2^2*M3+LC2^2*M2)+γ*z[44]+z[35]*(dz[19]*((sqrt(3)*L2*cos(z[5]))*0.5+(L2*sin(z[5])*sin(z[6]))*0.5)+dz[20]*((L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5)-cos(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)+sin(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[6])*sin(z[5])*dz[21]+L1*dz[13]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))+0.0*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5])))+L2*cos(z[5])*cos(z[6])*dz[51]+L1*dz[43]*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))-2*cos(z[5])*sin(z[5])*z[45]*z[15]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[43]*z[13]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))
            res[45] = z[45]*(γ+sin(2*z[5])*z[14]*(J2+L2^2*M3+LC2^2*M2))-z[36]*(L2*cos(z[6])*sin(z[5])*dz[21]-(L2*sin(z[5])*sin(z[6])*dz[19])*0.5-0.0*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+(sqrt(3)*L2*sin(z[5])*sin(z[6])*dz[20])*0.5-L1*cos(z[6])*sin(z[4])*sin(z[5])*z[13]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*cos(z[6])*sin(z[5])*dz[13]*(L2*M3+LC2*M2))+z[35]*(2*cos(z[5])*sin(z[5])*dz[15]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[5])*sin(z[6])*dz[21]-(L2*cos(z[5])*cos(z[6])*dz[19])*0.5+2*cos(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)+0.0*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[20])*0.5+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[5])*sin(z[6])*dz[13]*(L2*M3+LC2*M2))+z[34]*(L1*cos(z[4])*sin(z[5])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)+L1*sin(z[4])*sin(z[5])*sin(z[6])*dz[13]*(L2*M3+LC2*M2))+sin(z[5])^2*dz[45]*(J2+L2^2*M3+LC2^2*M2)-(L2*cos(z[6])*sin(z[5])*dz[49])*0.5-L2*sin(z[5])*sin(z[6])*dz[51]+sin(2*z[5])*z[44]*z[15]*(J2+L2^2*M3+LC2^2*M2)+(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[50])*0.5-L1*cos(z[4])*sin(z[5])*sin(z[6])*dz[43]*(L2*M3+LC2*M2)+2*L1*sin(z[4])*sin(z[5])*sin(z[6])*z[43]*z[13]*(L2*M3+LC2*M2)
            res[46] = z[47]*(2*L1*z[17]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[18]*(L2*M3+LC2*M2))+dz[46]*(J1+L1^2*(M2+M3)+LC1^2*M1)-z[48]*(2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]*(L2*M3+LC2*M2))-z[39]*(L1*cos(z[7])*cos(z[8])*sin(z[9])*dz[17]*(L2*M3+LC2*M2)-L1*cos(z[7])*sin(z[8])*sin(z[9])*z[18]^2*(L2*M3+LC2*M2)-L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*cos(z[9])*sin(z[8])*dz[18]*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[8])*cos(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+γ*z[46]-z[38]*(L1*z[17]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))-L1*dz[17]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))+L1*cos(z[7])*cos(z[8])*cos(z[9])*z[18]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*cos(z[8])*sin(z[9])*dz[18]*(L2*M3+LC2*M2)-2*L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+z[37]*(0.0*sin(z[7])*(L1*(M2+M3)+LC1*M1)+(L1*cos(z[7])*dz[23])*0.5-L1*sin(z[7])*dz[24]-(sqrt(3)*L1*cos(z[7])*dz[22])*0.5+L1*dz[17]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8]))+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)+L1*sin(z[7])*sin(z[8])*sin(z[9])*dz[18]*(L2*M3+LC2*M2)+2*L1*cos(z[8])*sin(z[7])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+L1*cos(z[7])*dz[54]+(L1*sin(z[7])*dz[53])*0.5-(sqrt(3)*L1*sin(z[7])*dz[52])*0.5+L1*dz[47]*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))-L1*cos(z[7])*sin(z[8])*sin(z[9])*dz[48]*(L2*M3+LC2*M2)
            res[47] = z[37]*(L1*dz[16]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))-L1*z[16]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9])))-z[39]*((L2*cos(z[8])*cos(z[9])*dz[22])*0.5+L2*cos(z[8])*sin(z[9])*dz[24]-0.0*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[23])*0.5-L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*cos(z[8])*sin(z[9])*dz[16]*(L2*M3+LC2*M2))-dz[52]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)+dz[53]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)+dz[47]*(J2+L2^2*M3+LC2^2*M2)+γ*z[47]+z[38]*(dz[23]*((L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5)-dz[22]*((sqrt(3)*L2*cos(z[8]))*0.5-(L2*sin(z[8])*sin(z[9]))*0.5)-cos(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)+sin(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[9])*sin(z[8])*dz[24]+L1*dz[16]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))+0.0*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8])))+L2*cos(z[8])*cos(z[9])*dz[54]+L1*dz[46]*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))-2*cos(z[8])*sin(z[8])*z[48]*z[18]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[46]*z[16]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))
            res[48] = z[39]*((L2*sin(z[8])*sin(z[9])*dz[22])*0.5-L2*cos(z[9])*sin(z[8])*dz[24]+0.0*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+(sqrt(3)*L2*sin(z[8])*sin(z[9])*dz[23])*0.5+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[16]^2*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[9])*sin(z[8])*dz[16]*(L2*M3+LC2*M2))+z[48]*(γ+sin(2*z[8])*z[17]*(J2+L2^2*M3+LC2^2*M2))-z[38]*((L2*cos(z[8])*cos(z[9])*dz[22])*0.5+L2*cos(z[8])*sin(z[9])*dz[24]-2*cos(z[8])*sin(z[8])*dz[18]*(J2+L2^2*M3+LC2^2*M2)-2*cos(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)-0.0*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[23])*0.5-L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*cos(z[8])*sin(z[9])*dz[16]*(L2*M3+LC2*M2))+z[37]*(L1*cos(z[7])*sin(z[8])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)+L1*sin(z[7])*sin(z[8])*sin(z[9])*dz[16]*(L2*M3+LC2*M2))+sin(z[8])^2*dz[48]*(J2+L2^2*M3+LC2^2*M2)-(L2*cos(z[9])*sin(z[8])*dz[52])*0.5-L2*sin(z[8])*sin(z[9])*dz[54]+sin(2*z[8])*z[47]*z[18]*(J2+L2^2*M3+LC2^2*M2)-(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[53])*0.5-L1*cos(z[7])*sin(z[8])*sin(z[9])*dz[46]*(L2*M3+LC2*M2)+2*L1*sin(z[7])*sin(z[8])*sin(z[9])*z[46]*z[16]*(L2*M3+LC2*M2)
            res[49] = z[35]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)+L2*cos(z[2])*sin(z[3])*z[32]+L2*cos(z[3])*sin(z[2])*z[33]+(L2*cos(z[6])*sin(z[5])*z[36])*0.5-(sqrt(3)*L1*sin(z[4])*z[34])*0.5
            res[50] = -z[35]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-L1*sin(z[1])*z[31]-L2*sin(z[2])*z[32]-(L1*sin(z[4])*z[34])*0.5-(sqrt(3)*L2*cos(z[6])*sin(z[5])*z[36])*0.5
            res[51] = L1*cos(z[1])*z[31]-L1*cos(z[4])*z[34]+L2*cos(z[2])*cos(z[3])*z[32]-L2*cos(z[5])*cos(z[6])*z[35]-L2*sin(z[2])*sin(z[3])*z[33]+L2*sin(z[5])*sin(z[6])*z[36]
            res[52] = z[38]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)+L2*cos(z[2])*sin(z[3])*z[32]+L2*cos(z[3])*sin(z[2])*z[33]+(L2*cos(z[9])*sin(z[8])*z[39])*0.5+(sqrt(3)*L1*sin(z[7])*z[37])*0.5
            res[53] = (sqrt(3)*L2*cos(z[9])*sin(z[8])*z[39])*0.5-L1*sin(z[1])*z[31]-L2*sin(z[2])*z[32]-(L1*sin(z[7])*z[37])*0.5-z[38]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)
            res[54] = L1*cos(z[1])*z[31]-L1*cos(z[7])*z[37]+L2*cos(z[2])*cos(z[3])*z[32]-L2*cos(z[8])*cos(z[9])*z[38]-L2*sin(z[2])*sin(z[3])*z[33]+L2*sin(z[8])*sin(z[9])*z[39]
            res[55] = z[32]*(L2*cos(z[2])*cos(z[3])*z[12]-L2*sin(z[2])*sin(z[3])*z[11])+z[33]*(L2*cos(z[2])*cos(z[3])*z[11]-L2*sin(z[2])*sin(z[3])*z[12])+z[36]*((L2*cos(z[5])*cos(z[6])*z[14])*0.5-(L2*sin(z[5])*sin(z[6])*z[15])*0.5)-z[35]*((sqrt(3)*L2*cos(z[5])*z[14])*0.5-(L2*cos(z[5])*cos(z[6])*z[15])*0.5+(L2*sin(z[5])*sin(z[6])*z[14])*0.5)+z[44]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)+L2*cos(z[2])*sin(z[3])*z[41]+L2*cos(z[3])*sin(z[2])*z[42]+(L2*cos(z[6])*sin(z[5])*z[45])*0.5-(sqrt(3)*L1*sin(z[4])*z[43])*0.5-(sqrt(3)*L1*cos(z[4])*z[34]*z[13])*0.5
            res[56] = -z[44]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-z[35]*((L2*cos(z[5])*z[14])*0.5+(sqrt(3)*L2*cos(z[5])*cos(z[6])*z[15])*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6])*z[14])*0.5)-z[36]*((sqrt(3)*L2*cos(z[5])*cos(z[6])*z[14])*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6])*z[15])*0.5)-L1*sin(z[1])*z[40]-L2*sin(z[2])*z[41]-(L1*sin(z[4])*z[43])*0.5-L1*cos(z[1])*z[31]*z[10]-L2*cos(z[2])*z[32]*z[11]-(L1*cos(z[4])*z[34]*z[13])*0.5-(sqrt(3)*L2*cos(z[6])*sin(z[5])*z[45])*0.5
            res[57] = z[35]*(L2*cos(z[6])*sin(z[5])*z[14]+L2*cos(z[5])*sin(z[6])*z[15])-z[33]*(L2*cos(z[2])*sin(z[3])*z[11]+L2*cos(z[3])*sin(z[2])*z[12])-z[32]*(L2*cos(z[3])*sin(z[2])*z[11]+L2*cos(z[2])*sin(z[3])*z[12])+z[36]*(L2*cos(z[5])*sin(z[6])*z[14]+L2*cos(z[6])*sin(z[5])*z[15])+L1*cos(z[1])*z[40]-L1*cos(z[4])*z[43]+L2*cos(z[2])*cos(z[3])*z[41]-L2*cos(z[5])*cos(z[6])*z[44]-L2*sin(z[2])*sin(z[3])*z[42]+L2*sin(z[5])*sin(z[6])*z[45]-L1*sin(z[1])*z[31]*z[10]+L1*sin(z[4])*z[34]*z[13]
            res[58] = z[32]*(L2*cos(z[2])*cos(z[3])*z[12]-L2*sin(z[2])*sin(z[3])*z[11])+z[33]*(L2*cos(z[2])*cos(z[3])*z[11]-L2*sin(z[2])*sin(z[3])*z[12])+z[39]*((L2*cos(z[8])*cos(z[9])*z[17])*0.5-(L2*sin(z[8])*sin(z[9])*z[18])*0.5)+z[38]*((L2*cos(z[8])*cos(z[9])*z[18])*0.5+(sqrt(3)*L2*cos(z[8])*z[17])*0.5-(L2*sin(z[8])*sin(z[9])*z[17])*0.5)+z[47]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)+L2*cos(z[2])*sin(z[3])*z[41]+L2*cos(z[3])*sin(z[2])*z[42]+(L2*cos(z[9])*sin(z[8])*z[48])*0.5+(sqrt(3)*L1*sin(z[7])*z[46])*0.5+(sqrt(3)*L1*cos(z[7])*z[37]*z[16])*0.5
            res[59] = z[39]*((sqrt(3)*L2*cos(z[8])*cos(z[9])*z[17])*0.5-(sqrt(3)*L2*sin(z[8])*sin(z[9])*z[18])*0.5)-z[38]*((L2*cos(z[8])*z[17])*0.5-(sqrt(3)*L2*cos(z[8])*cos(z[9])*z[18])*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9])*z[17])*0.5)-z[47]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)-L1*sin(z[1])*z[40]-L2*sin(z[2])*z[41]-(L1*sin(z[7])*z[46])*0.5-L1*cos(z[1])*z[31]*z[10]-L2*cos(z[2])*z[32]*z[11]-(L1*cos(z[7])*z[37]*z[16])*0.5+(sqrt(3)*L2*cos(z[9])*sin(z[8])*z[48])*0.5
            res[60] = z[38]*(L2*cos(z[9])*sin(z[8])*z[17]+L2*cos(z[8])*sin(z[9])*z[18])-z[33]*(L2*cos(z[2])*sin(z[3])*z[11]+L2*cos(z[3])*sin(z[2])*z[12])-z[32]*(L2*cos(z[3])*sin(z[2])*z[11]+L2*cos(z[2])*sin(z[3])*z[12])+z[39]*(L2*cos(z[8])*sin(z[9])*z[17]+L2*cos(z[9])*sin(z[8])*z[18])+L1*cos(z[1])*z[40]-L1*cos(z[7])*z[46]+L2*cos(z[2])*cos(z[3])*z[41]-L2*cos(z[8])*cos(z[9])*z[47]-L2*sin(z[2])*sin(z[3])*z[42]+L2*sin(z[8])*sin(z[9])*z[48]-L1*sin(z[1])*z[31]*z[10]+L1*sin(z[7])*z[37]*z[16]
            # Parameter-specific part for J1
            res[40] += dz[10]
            res[43] += dz[13]
            res[46] += dz[16]

            nothing
        end

        z0, dz0 = get_delta_initial_J1sens_comp(θ, u(0.0), w(0.0))

        dvars = fill(true, 60)
        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 for delta robot is: $r0"

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, z0, dz0, dvars, r0)
    end
end

function delta_robot_gc_γsens(Φ::Float64, u::Function, w::Function, θ::Vector{Float64})::Model
    let L0 = θ[1], L1 = θ[2], L2 = θ[3], L3 = θ[4], LC1 = θ[5], LC2 = θ[6], M1 = θ[7], M2 = θ[8], M3 = θ[9], J1 = θ[10], J2 = θ[11], g = 0.0, γ = θ[13]
        # @info "Samples: $(u(0.3)), $(u(0.46)), $(u(1.23)), $(w(0.3)), $(w(0.46)), $(w(1.23))"
        function f!(res, dz, z, _, t)
            ut = u(t)
            wt = w(t)

            res[1] = dz[1]-z[10]+L1*cos(z[1])*dz[27]+L1*cos(z[1])*dz[30]-L1*sin(z[1])*dz[26]-L1*sin(z[1])*dz[29]
            res[2] = dz[2]-z[11]-L2*sin(z[2])*dz[26]-L2*sin(z[2])*dz[29]+L2*cos(z[2])*cos(z[3])*dz[27]+L2*cos(z[2])*cos(z[3])*dz[30]+L2*cos(z[2])*sin(z[3])*dz[25]+L2*cos(z[2])*sin(z[3])*dz[28]
            res[3] = dz[3]-z[12]+L2*cos(z[3])*sin(z[2])*dz[25]+L2*cos(z[3])*sin(z[2])*dz[28]-L2*sin(z[2])*sin(z[3])*dz[27]-L2*sin(z[2])*sin(z[3])*dz[30]
            res[4] = dz[4]-z[13]-L1*cos(z[4])*dz[27]-(L1*sin(z[4])*dz[26])*0.5-(sqrt(3)*L1*sin(z[4])*dz[25])*0.5
            res[5] = dz[5]-z[14]+dz[25]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)-dz[26]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-L2*cos(z[5])*cos(z[6])*dz[27]
            res[6] = dz[6]-z[15]+(L2*cos(z[6])*sin(z[5])*dz[25])*0.5+L2*sin(z[5])*sin(z[6])*dz[27]-(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[26])*0.5
            res[7] = dz[7]-z[16]-L1*cos(z[7])*dz[30]-(L1*sin(z[7])*dz[29])*0.5+(sqrt(3)*L1*sin(z[7])*dz[28])*0.5
            res[8] = dz[8]-z[17]+dz[28]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)-dz[29]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)-L2*cos(z[8])*cos(z[9])*dz[30]
            res[9] = dz[9]-z[18]+(L2*cos(z[9])*sin(z[8])*dz[28])*0.5+L2*sin(z[8])*sin(z[9])*dz[30]+(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[29])*0.5
            res[10] = dz[10]*(J1+L1^2*(M2+M3)+LC1^2*M1)-ut[1]-wt[1]^2+γ*z[10]-0.0*cos(z[1])*(L1*(M2+M3)+LC1*M1)-L1*cos(z[1])*dz[21]-L1*cos(z[1])*dz[24]+L1*sin(z[1])*dz[20]+L1*sin(z[1])*dz[23]+L1*dz[11]*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*dz[12]*(L2*M3+LC2*M2)-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2)
            res[11] = dz[11]*(J2+L2^2*M3+LC2^2*M2)+γ*z[11]+L2*sin(z[2])*dz[20]+L2*sin(z[2])*dz[23]-L2*cos(z[2])*cos(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[24]-L2*cos(z[2])*sin(z[3])*dz[19]-L2*cos(z[2])*sin(z[3])*dz[22]+L1*dz[10]*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))-cos(z[2])*sin(z[2])*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-0.0*cos(z[2])*cos(z[3])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))
            res[12] = γ*z[12]+sin(z[2])^2*dz[12]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[3])*sin(z[2])*dz[19]-L2*cos(z[3])*sin(z[2])*dz[22]+L2*sin(z[2])*sin(z[3])*dz[21]+L2*sin(z[2])*sin(z[3])*dz[24]+sin(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+0.0*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*sin(z[1])*sin(z[2])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*dz[10]*(L2*M3+LC2*M2)
            res[13] = dz[13]*(J1+L1^2*(M2+M3)+LC1^2*M1)-ut[2]-wt[2]^2+γ*z[13]-0.0*cos(z[4])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[4])*dz[21]+(L1*sin(z[4])*dz[20])*0.5+(sqrt(3)*L1*sin(z[4])*dz[19])*0.5+L1*dz[14]*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)-L1*cos(z[4])*sin(z[5])*sin(z[6])*dz[15]*(L2*M3+LC2*M2)-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2)
            res[14] = dz[20]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-dz[19]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)+dz[14]*(J2+L2^2*M3+LC2^2*M2)+γ*z[14]+L2*cos(z[5])*cos(z[6])*dz[21]+L1*dz[13]*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))-cos(z[5])*sin(z[5])*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-0.0*cos(z[5])*cos(z[6])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))
            res[15] = γ*z[15]+sin(z[5])^2*dz[15]*(J2+L2^2*M3+LC2^2*M2)-(L2*cos(z[6])*sin(z[5])*dz[19])*0.5-L2*sin(z[5])*sin(z[6])*dz[21]+sin(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)+0.0*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[20])*0.5+L1*sin(z[4])*sin(z[5])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)-L1*cos(z[4])*sin(z[5])*sin(z[6])*dz[13]*(L2*M3+LC2*M2)
            res[16] = dz[16]*(J1+L1^2*(M2+M3)+LC1^2*M1)-ut[3]-wt[3]^2+γ*z[16]-0.0*cos(z[7])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[7])*dz[24]+(L1*sin(z[7])*dz[23])*0.5-(sqrt(3)*L1*sin(z[7])*dz[22])*0.5+L1*dz[17]*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)-L1*cos(z[7])*sin(z[8])*sin(z[9])*dz[18]*(L2*M3+LC2*M2)-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2)
            res[17] = dz[23]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)-dz[22]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)+dz[17]*(J2+L2^2*M3+LC2^2*M2)+γ*z[17]+L2*cos(z[8])*cos(z[9])*dz[24]+L1*dz[16]*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))-cos(z[8])*sin(z[8])*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-0.0*cos(z[8])*cos(z[9])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))
            res[18] = γ*z[18]+sin(z[8])^2*dz[18]*(J2+L2^2*M3+LC2^2*M2)-(L2*cos(z[9])*sin(z[8])*dz[22])*0.5-L2*sin(z[8])*sin(z[9])*dz[24]+sin(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)+0.0*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)-(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[23])*0.5+L1*sin(z[7])*sin(z[8])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)-L1*cos(z[7])*sin(z[8])*sin(z[9])*dz[16]*(L2*M3+LC2*M2)
            res[19] = (sqrt(3)*(L0-L3+L1*cos(z[4])+L2*cos(z[5])))*0.5+L2*sin(z[2])*sin(z[3])+(L2*sin(z[5])*sin(z[6]))*0.5
            res[20] = (3*L0)*0.5-(3*L3)*0.5+L1*cos(z[1])+L2*cos(z[2])+(L1*cos(z[4]))*0.5+(L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5
            res[21] = L1*sin(z[1])-L1*sin(z[4])+L2*cos(z[3])*sin(z[2])-L2*cos(z[6])*sin(z[5])
            res[22] = L2*sin(z[2])*sin(z[3])-(sqrt(3)*(L0-L3+L1*cos(z[7])+L2*cos(z[8])))*0.5+(L2*sin(z[8])*sin(z[9]))*0.5
            res[23] = (3*L0)*0.5-(3*L3)*0.5+L1*cos(z[1])+L2*cos(z[2])+(L1*cos(z[7]))*0.5+(L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5
            res[24] = L1*sin(z[1])-L1*sin(z[7])+L2*cos(z[3])*sin(z[2])-L2*cos(z[9])*sin(z[8])
            res[25] = L2*cos(z[2])*sin(z[3])*z[11]-(sqrt(3)*(L1*sin(z[4])*z[13]+L2*sin(z[5])*z[14]))*0.5+L2*cos(z[3])*sin(z[2])*z[12]+(L2*cos(z[5])*sin(z[6])*z[14])*0.5+(L2*cos(z[6])*sin(z[5])*z[15])*0.5
            res[26] = -L1*sin(z[1])*z[10]-L2*sin(z[2])*z[11]-(L1*sin(z[4])*z[13])*0.5-(L2*sin(z[5])*z[14])*0.5-(sqrt(3)*L2*cos(z[5])*sin(z[6])*z[14])*0.5-(sqrt(3)*L2*cos(z[6])*sin(z[5])*z[15])*0.5
            res[27] = L1*cos(z[1])*z[10]-L1*cos(z[4])*z[13]+L2*cos(z[2])*cos(z[3])*z[11]-L2*cos(z[5])*cos(z[6])*z[14]-L2*sin(z[2])*sin(z[3])*z[12]+L2*sin(z[5])*sin(z[6])*z[15]
            res[28] = (sqrt(3)*(L1*sin(z[7])*z[16]+L2*sin(z[8])*z[17]))*0.5+L2*cos(z[2])*sin(z[3])*z[11]+L2*cos(z[3])*sin(z[2])*z[12]+(L2*cos(z[8])*sin(z[9])*z[17])*0.5+(L2*cos(z[9])*sin(z[8])*z[18])*0.5
            res[29] = (sqrt(3)*L2*cos(z[8])*sin(z[9])*z[17])*0.5-L2*sin(z[2])*z[11]-(L1*sin(z[7])*z[16])*0.5-(L2*sin(z[8])*z[17])*0.5-L1*sin(z[1])*z[10]+(sqrt(3)*L2*cos(z[9])*sin(z[8])*z[18])*0.5
            res[30] = L1*cos(z[1])*z[10]-L1*cos(z[7])*z[16]+L2*cos(z[2])*cos(z[3])*z[11]-L2*cos(z[8])*cos(z[9])*z[17]-L2*sin(z[2])*sin(z[3])*z[12]+L2*sin(z[8])*sin(z[9])*z[18]
            # Sensitivity equations
            res[31] = dz[31]-z[40]-z[31]*(L1*cos(z[1])*dz[26]+L1*cos(z[1])*dz[29]+L1*sin(z[1])*dz[27]+L1*sin(z[1])*dz[30])+L1*cos(z[1])*dz[57]+L1*cos(z[1])*dz[60]-L1*sin(z[1])*dz[56]-L1*sin(z[1])*dz[59]
            res[32] = dz[32]-z[41]+z[33]*(L2*cos(z[2])*cos(z[3])*dz[25]+L2*cos(z[2])*cos(z[3])*dz[28]-L2*cos(z[2])*sin(z[3])*dz[27]-L2*cos(z[2])*sin(z[3])*dz[30])-z[32]*(L2*cos(z[2])*dz[26]+L2*cos(z[2])*dz[29]+L2*cos(z[3])*sin(z[2])*dz[27]+L2*cos(z[3])*sin(z[2])*dz[30]+L2*sin(z[2])*sin(z[3])*dz[25]+L2*sin(z[2])*sin(z[3])*dz[28])-L2*sin(z[2])*dz[56]-L2*sin(z[2])*dz[59]+L2*cos(z[2])*cos(z[3])*dz[57]+L2*cos(z[2])*cos(z[3])*dz[60]+L2*cos(z[2])*sin(z[3])*dz[55]+L2*cos(z[2])*sin(z[3])*dz[58]
            res[33] = dz[33]-z[42]+z[32]*(L2*cos(z[2])*cos(z[3])*dz[25]+L2*cos(z[2])*cos(z[3])*dz[28]-L2*cos(z[2])*sin(z[3])*dz[27]-L2*cos(z[2])*sin(z[3])*dz[30])-z[33]*(L2*cos(z[3])*sin(z[2])*dz[27]+L2*cos(z[3])*sin(z[2])*dz[30]+L2*sin(z[2])*sin(z[3])*dz[25]+L2*sin(z[2])*sin(z[3])*dz[28])+L2*cos(z[3])*sin(z[2])*dz[55]+L2*cos(z[3])*sin(z[2])*dz[58]-L2*sin(z[2])*sin(z[3])*dz[57]-L2*sin(z[2])*sin(z[3])*dz[60]
            res[34] = dz[34]-z[43]-z[34]*((L1*cos(z[4])*dz[26])*0.5-L1*sin(z[4])*dz[27]+(sqrt(3)*L1*cos(z[4])*dz[25])*0.5)-L1*cos(z[4])*dz[57]-(L1*sin(z[4])*dz[56])*0.5-(sqrt(3)*L1*sin(z[4])*dz[55])*0.5
            res[35] = dz[35]-z[44]+dz[55]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)-dz[56]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)+z[36]*((L2*cos(z[5])*cos(z[6])*dz[25])*0.5+L2*cos(z[5])*sin(z[6])*dz[27]-(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[26])*0.5)-z[35]*(dz[25]*((sqrt(3)*L2*cos(z[5]))*0.5+(L2*sin(z[5])*sin(z[6]))*0.5)+dz[26]*((L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5)-L2*cos(z[6])*sin(z[5])*dz[27])-L2*cos(z[5])*cos(z[6])*dz[57]
            res[36] = dz[36]-z[45]+z[35]*((L2*cos(z[5])*cos(z[6])*dz[25])*0.5+L2*cos(z[5])*sin(z[6])*dz[27]-(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[26])*0.5)+z[36]*(L2*cos(z[6])*sin(z[5])*dz[27]-(L2*sin(z[5])*sin(z[6])*dz[25])*0.5+(sqrt(3)*L2*sin(z[5])*sin(z[6])*dz[26])*0.5)+(L2*cos(z[6])*sin(z[5])*dz[55])*0.5+L2*sin(z[5])*sin(z[6])*dz[57]-(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[56])*0.5
            res[37] = dz[37]-z[46]+z[37]*(L1*sin(z[7])*dz[30]-(L1*cos(z[7])*dz[29])*0.5+(sqrt(3)*L1*cos(z[7])*dz[28])*0.5)-L1*cos(z[7])*dz[60]-(L1*sin(z[7])*dz[59])*0.5+(sqrt(3)*L1*sin(z[7])*dz[58])*0.5
            res[38] = dz[38]-z[47]+dz[58]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)-dz[59]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)+z[39]*((L2*cos(z[8])*cos(z[9])*dz[28])*0.5+L2*cos(z[8])*sin(z[9])*dz[30]+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[29])*0.5)+z[38]*(dz[28]*((sqrt(3)*L2*cos(z[8]))*0.5-(L2*sin(z[8])*sin(z[9]))*0.5)-dz[29]*((L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5)+L2*cos(z[9])*sin(z[8])*dz[30])-L2*cos(z[8])*cos(z[9])*dz[60]
            res[39] = dz[39]-z[48]+z[38]*((L2*cos(z[8])*cos(z[9])*dz[28])*0.5+L2*cos(z[8])*sin(z[9])*dz[30]+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[29])*0.5)-z[39]*((L2*sin(z[8])*sin(z[9])*dz[28])*0.5-L2*cos(z[9])*sin(z[8])*dz[30]+(sqrt(3)*L2*sin(z[8])*sin(z[9])*dz[29])*0.5)+(L2*cos(z[9])*sin(z[8])*dz[58])*0.5+L2*sin(z[8])*sin(z[9])*dz[60]+(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[59])*0.5
            res[40] = z[41]*(2*L1*z[11]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[12]*(L2*M3+LC2*M2))+dz[40]*(J1+L1^2*(M2+M3)+LC1^2*M1)-z[42]*(2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]*(L2*M3+LC2*M2))-z[33]*(L1*cos(z[1])*cos(z[2])*sin(z[3])*dz[11]*(L2*M3+LC2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*z[12]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]^2*(L2*M3+LC2*M2)+L1*cos(z[1])*cos(z[3])*sin(z[2])*dz[12]*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[2])*cos(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+z[31]*(0.0*sin(z[1])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[1])*dz[20]+L1*cos(z[1])*dz[23]+L1*sin(z[1])*dz[21]+L1*sin(z[1])*dz[24]+L1*dz[11]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2]))+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)+L1*sin(z[1])*sin(z[2])*sin(z[3])*dz[12]*(L2*M3+LC2*M2)+2*L1*cos(z[2])*sin(z[1])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+γ*z[40]-z[32]*(L1*z[11]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))-L1*dz[11]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))+L1*cos(z[1])*cos(z[2])*cos(z[3])*z[12]^2*(L2*M3+LC2*M2)+L1*cos(z[1])*cos(z[2])*sin(z[3])*dz[12]*(L2*M3+LC2*M2)-2*L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))-L1*cos(z[1])*dz[51]-L1*cos(z[1])*dz[54]+L1*sin(z[1])*dz[50]+L1*sin(z[1])*dz[53]+L1*dz[41]*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))-L1*cos(z[1])*sin(z[2])*sin(z[3])*dz[42]*(L2*M3+LC2*M2)
            res[41] = z[31]*(L1*dz[10]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))-L1*z[10]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3])))+z[32]*(sin(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)+L2*cos(z[2])*dz[20]+L2*cos(z[2])*dz[23]+L2*cos(z[3])*sin(z[2])*dz[21]+L2*cos(z[3])*sin(z[2])*dz[24]+L2*sin(z[2])*sin(z[3])*dz[19]+L2*sin(z[2])*sin(z[3])*dz[22]+L1*dz[10]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))+0.0*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2])))+dz[41]*(J2+L2^2*M3+LC2^2*M2)+γ*z[41]+z[33]*(L2*cos(z[2])*sin(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[22]-L2*cos(z[2])*cos(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[24]+0.0*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[2])*sin(z[3])*dz[10]*(L2*M3+LC2*M2))+L2*sin(z[2])*dz[50]+L2*sin(z[2])*dz[53]-L2*cos(z[2])*cos(z[3])*dz[51]-L2*cos(z[2])*cos(z[3])*dz[54]-L2*cos(z[2])*sin(z[3])*dz[49]-L2*cos(z[2])*sin(z[3])*dz[52]+L1*dz[40]*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))-2*cos(z[2])*sin(z[2])*z[42]*z[12]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[40]*z[10]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))
            res[42] = z[33]*(L2*cos(z[3])*sin(z[2])*dz[21]+L2*cos(z[3])*sin(z[2])*dz[24]+L2*sin(z[2])*sin(z[3])*dz[19]+L2*sin(z[2])*sin(z[3])*dz[22]+0.0*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[10]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[3])*sin(z[2])*dz[10]*(L2*M3+LC2*M2))+z[32]*(L2*cos(z[2])*sin(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[22]-L2*cos(z[2])*cos(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[24]+2*cos(z[2])*sin(z[2])*dz[12]*(J2+L2^2*M3+LC2^2*M2)+2*cos(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+0.0*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[2])*sin(z[3])*dz[10]*(L2*M3+LC2*M2))+z[42]*(γ+sin(2*z[2])*z[11]*(J2+L2^2*M3+LC2^2*M2))+z[31]*(L1*cos(z[1])*sin(z[2])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)+L1*sin(z[1])*sin(z[2])*sin(z[3])*dz[10]*(L2*M3+LC2*M2))+sin(z[2])^2*dz[42]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[3])*sin(z[2])*dz[49]-L2*cos(z[3])*sin(z[2])*dz[52]+L2*sin(z[2])*sin(z[3])*dz[51]+L2*sin(z[2])*sin(z[3])*dz[54]+sin(2*z[2])*z[41]*z[12]*(J2+L2^2*M3+LC2^2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*dz[40]*(L2*M3+LC2*M2)+2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[40]*z[10]*(L2*M3+LC2*M2)
            res[43] = z[44]*(2*L1*z[14]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[15]*(L2*M3+LC2*M2))+dz[43]*(J1+L1^2*(M2+M3)+LC1^2*M1)-z[45]*(2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]*(L2*M3+LC2*M2))-z[36]*(L1*cos(z[4])*cos(z[5])*sin(z[6])*dz[14]*(L2*M3+LC2*M2)-L1*cos(z[4])*sin(z[5])*sin(z[6])*z[15]^2*(L2*M3+LC2*M2)-L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*cos(z[6])*sin(z[5])*dz[15]*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[5])*cos(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+γ*z[43]-z[35]*(L1*z[14]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))-L1*dz[14]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))+L1*cos(z[4])*cos(z[5])*cos(z[6])*z[15]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*cos(z[5])*sin(z[6])*dz[15]*(L2*M3+LC2*M2)-2*L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+z[34]*(0.0*sin(z[4])*(L1*(M2+M3)+LC1*M1)+(L1*cos(z[4])*dz[20])*0.5-L1*sin(z[4])*dz[21]+(sqrt(3)*L1*cos(z[4])*dz[19])*0.5+L1*dz[14]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5]))+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)+L1*sin(z[4])*sin(z[5])*sin(z[6])*dz[15]*(L2*M3+LC2*M2)+2*L1*cos(z[5])*sin(z[4])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+L1*cos(z[4])*dz[51]+(L1*sin(z[4])*dz[50])*0.5+(sqrt(3)*L1*sin(z[4])*dz[49])*0.5+L1*dz[44]*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))-L1*cos(z[4])*sin(z[5])*sin(z[6])*dz[45]*(L2*M3+LC2*M2)
            res[44] = z[34]*(L1*dz[13]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))-L1*z[13]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6])))-z[36]*((L2*cos(z[5])*cos(z[6])*dz[19])*0.5+L2*cos(z[5])*sin(z[6])*dz[21]-0.0*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)-(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[20])*0.5-L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*cos(z[5])*sin(z[6])*dz[13]*(L2*M3+LC2*M2))-dz[49]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)+dz[50]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)+dz[44]*(J2+L2^2*M3+LC2^2*M2)+γ*z[44]+z[35]*(dz[19]*((sqrt(3)*L2*cos(z[5]))*0.5+(L2*sin(z[5])*sin(z[6]))*0.5)+dz[20]*((L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5)-cos(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)+sin(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[6])*sin(z[5])*dz[21]+L1*dz[13]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))+0.0*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5])))+L2*cos(z[5])*cos(z[6])*dz[51]+L1*dz[43]*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))-2*cos(z[5])*sin(z[5])*z[45]*z[15]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[43]*z[13]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))
            res[45] = z[45]*(γ+sin(2*z[5])*z[14]*(J2+L2^2*M3+LC2^2*M2))-z[36]*(L2*cos(z[6])*sin(z[5])*dz[21]-(L2*sin(z[5])*sin(z[6])*dz[19])*0.5-0.0*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+(sqrt(3)*L2*sin(z[5])*sin(z[6])*dz[20])*0.5-L1*cos(z[6])*sin(z[4])*sin(z[5])*z[13]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*cos(z[6])*sin(z[5])*dz[13]*(L2*M3+LC2*M2))+z[35]*(2*cos(z[5])*sin(z[5])*dz[15]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[5])*sin(z[6])*dz[21]-(L2*cos(z[5])*cos(z[6])*dz[19])*0.5+2*cos(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)+0.0*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[20])*0.5+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[5])*sin(z[6])*dz[13]*(L2*M3+LC2*M2))+z[34]*(L1*cos(z[4])*sin(z[5])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)+L1*sin(z[4])*sin(z[5])*sin(z[6])*dz[13]*(L2*M3+LC2*M2))+sin(z[5])^2*dz[45]*(J2+L2^2*M3+LC2^2*M2)-(L2*cos(z[6])*sin(z[5])*dz[49])*0.5-L2*sin(z[5])*sin(z[6])*dz[51]+sin(2*z[5])*z[44]*z[15]*(J2+L2^2*M3+LC2^2*M2)+(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[50])*0.5-L1*cos(z[4])*sin(z[5])*sin(z[6])*dz[43]*(L2*M3+LC2*M2)+2*L1*sin(z[4])*sin(z[5])*sin(z[6])*z[43]*z[13]*(L2*M3+LC2*M2)
            res[46] = z[47]*(2*L1*z[17]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[18]*(L2*M3+LC2*M2))+dz[46]*(J1+L1^2*(M2+M3)+LC1^2*M1)-z[48]*(2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]*(L2*M3+LC2*M2))-z[39]*(L1*cos(z[7])*cos(z[8])*sin(z[9])*dz[17]*(L2*M3+LC2*M2)-L1*cos(z[7])*sin(z[8])*sin(z[9])*z[18]^2*(L2*M3+LC2*M2)-L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*cos(z[9])*sin(z[8])*dz[18]*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[8])*cos(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+γ*z[46]-z[38]*(L1*z[17]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))-L1*dz[17]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))+L1*cos(z[7])*cos(z[8])*cos(z[9])*z[18]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*cos(z[8])*sin(z[9])*dz[18]*(L2*M3+LC2*M2)-2*L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+z[37]*(0.0*sin(z[7])*(L1*(M2+M3)+LC1*M1)+(L1*cos(z[7])*dz[23])*0.5-L1*sin(z[7])*dz[24]-(sqrt(3)*L1*cos(z[7])*dz[22])*0.5+L1*dz[17]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8]))+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)+L1*sin(z[7])*sin(z[8])*sin(z[9])*dz[18]*(L2*M3+LC2*M2)+2*L1*cos(z[8])*sin(z[7])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+L1*cos(z[7])*dz[54]+(L1*sin(z[7])*dz[53])*0.5-(sqrt(3)*L1*sin(z[7])*dz[52])*0.5+L1*dz[47]*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))-L1*cos(z[7])*sin(z[8])*sin(z[9])*dz[48]*(L2*M3+LC2*M2)
            res[47] = z[37]*(L1*dz[16]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))-L1*z[16]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9])))-z[39]*((L2*cos(z[8])*cos(z[9])*dz[22])*0.5+L2*cos(z[8])*sin(z[9])*dz[24]-0.0*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[23])*0.5-L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*cos(z[8])*sin(z[9])*dz[16]*(L2*M3+LC2*M2))-dz[52]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)+dz[53]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)+dz[47]*(J2+L2^2*M3+LC2^2*M2)+γ*z[47]+z[38]*(dz[23]*((L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5)-dz[22]*((sqrt(3)*L2*cos(z[8]))*0.5-(L2*sin(z[8])*sin(z[9]))*0.5)-cos(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)+sin(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[9])*sin(z[8])*dz[24]+L1*dz[16]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))+0.0*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8])))+L2*cos(z[8])*cos(z[9])*dz[54]+L1*dz[46]*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))-2*cos(z[8])*sin(z[8])*z[48]*z[18]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[46]*z[16]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))
            res[48] = z[39]*((L2*sin(z[8])*sin(z[9])*dz[22])*0.5-L2*cos(z[9])*sin(z[8])*dz[24]+0.0*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+(sqrt(3)*L2*sin(z[8])*sin(z[9])*dz[23])*0.5+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[16]^2*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[9])*sin(z[8])*dz[16]*(L2*M3+LC2*M2))+z[48]*(γ+sin(2*z[8])*z[17]*(J2+L2^2*M3+LC2^2*M2))-z[38]*((L2*cos(z[8])*cos(z[9])*dz[22])*0.5+L2*cos(z[8])*sin(z[9])*dz[24]-2*cos(z[8])*sin(z[8])*dz[18]*(J2+L2^2*M3+LC2^2*M2)-2*cos(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)-0.0*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[23])*0.5-L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*cos(z[8])*sin(z[9])*dz[16]*(L2*M3+LC2*M2))+z[37]*(L1*cos(z[7])*sin(z[8])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)+L1*sin(z[7])*sin(z[8])*sin(z[9])*dz[16]*(L2*M3+LC2*M2))+sin(z[8])^2*dz[48]*(J2+L2^2*M3+LC2^2*M2)-(L2*cos(z[9])*sin(z[8])*dz[52])*0.5-L2*sin(z[8])*sin(z[9])*dz[54]+sin(2*z[8])*z[47]*z[18]*(J2+L2^2*M3+LC2^2*M2)-(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[53])*0.5-L1*cos(z[7])*sin(z[8])*sin(z[9])*dz[46]*(L2*M3+LC2*M2)+2*L1*sin(z[7])*sin(z[8])*sin(z[9])*z[46]*z[16]*(L2*M3+LC2*M2)
            res[49] = z[35]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)+L2*cos(z[2])*sin(z[3])*z[32]+L2*cos(z[3])*sin(z[2])*z[33]+(L2*cos(z[6])*sin(z[5])*z[36])*0.5-(sqrt(3)*L1*sin(z[4])*z[34])*0.5
            res[50] = -z[35]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-L1*sin(z[1])*z[31]-L2*sin(z[2])*z[32]-(L1*sin(z[4])*z[34])*0.5-(sqrt(3)*L2*cos(z[6])*sin(z[5])*z[36])*0.5
            res[51] = L1*cos(z[1])*z[31]-L1*cos(z[4])*z[34]+L2*cos(z[2])*cos(z[3])*z[32]-L2*cos(z[5])*cos(z[6])*z[35]-L2*sin(z[2])*sin(z[3])*z[33]+L2*sin(z[5])*sin(z[6])*z[36]
            res[52] = z[38]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)+L2*cos(z[2])*sin(z[3])*z[32]+L2*cos(z[3])*sin(z[2])*z[33]+(L2*cos(z[9])*sin(z[8])*z[39])*0.5+(sqrt(3)*L1*sin(z[7])*z[37])*0.5
            res[53] = (sqrt(3)*L2*cos(z[9])*sin(z[8])*z[39])*0.5-L1*sin(z[1])*z[31]-L2*sin(z[2])*z[32]-(L1*sin(z[7])*z[37])*0.5-z[38]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)
            res[54] = L1*cos(z[1])*z[31]-L1*cos(z[7])*z[37]+L2*cos(z[2])*cos(z[3])*z[32]-L2*cos(z[8])*cos(z[9])*z[38]-L2*sin(z[2])*sin(z[3])*z[33]+L2*sin(z[8])*sin(z[9])*z[39]
            res[55] = z[32]*(L2*cos(z[2])*cos(z[3])*z[12]-L2*sin(z[2])*sin(z[3])*z[11])+z[33]*(L2*cos(z[2])*cos(z[3])*z[11]-L2*sin(z[2])*sin(z[3])*z[12])+z[36]*((L2*cos(z[5])*cos(z[6])*z[14])*0.5-(L2*sin(z[5])*sin(z[6])*z[15])*0.5)-z[35]*((sqrt(3)*L2*cos(z[5])*z[14])*0.5-(L2*cos(z[5])*cos(z[6])*z[15])*0.5+(L2*sin(z[5])*sin(z[6])*z[14])*0.5)+z[44]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)+L2*cos(z[2])*sin(z[3])*z[41]+L2*cos(z[3])*sin(z[2])*z[42]+(L2*cos(z[6])*sin(z[5])*z[45])*0.5-(sqrt(3)*L1*sin(z[4])*z[43])*0.5-(sqrt(3)*L1*cos(z[4])*z[34]*z[13])*0.5
            res[56] = -z[44]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-z[35]*((L2*cos(z[5])*z[14])*0.5+(sqrt(3)*L2*cos(z[5])*cos(z[6])*z[15])*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6])*z[14])*0.5)-z[36]*((sqrt(3)*L2*cos(z[5])*cos(z[6])*z[14])*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6])*z[15])*0.5)-L1*sin(z[1])*z[40]-L2*sin(z[2])*z[41]-(L1*sin(z[4])*z[43])*0.5-L1*cos(z[1])*z[31]*z[10]-L2*cos(z[2])*z[32]*z[11]-(L1*cos(z[4])*z[34]*z[13])*0.5-(sqrt(3)*L2*cos(z[6])*sin(z[5])*z[45])*0.5
            res[57] = z[35]*(L2*cos(z[6])*sin(z[5])*z[14]+L2*cos(z[5])*sin(z[6])*z[15])-z[33]*(L2*cos(z[2])*sin(z[3])*z[11]+L2*cos(z[3])*sin(z[2])*z[12])-z[32]*(L2*cos(z[3])*sin(z[2])*z[11]+L2*cos(z[2])*sin(z[3])*z[12])+z[36]*(L2*cos(z[5])*sin(z[6])*z[14]+L2*cos(z[6])*sin(z[5])*z[15])+L1*cos(z[1])*z[40]-L1*cos(z[4])*z[43]+L2*cos(z[2])*cos(z[3])*z[41]-L2*cos(z[5])*cos(z[6])*z[44]-L2*sin(z[2])*sin(z[3])*z[42]+L2*sin(z[5])*sin(z[6])*z[45]-L1*sin(z[1])*z[31]*z[10]+L1*sin(z[4])*z[34]*z[13]
            res[58] = z[32]*(L2*cos(z[2])*cos(z[3])*z[12]-L2*sin(z[2])*sin(z[3])*z[11])+z[33]*(L2*cos(z[2])*cos(z[3])*z[11]-L2*sin(z[2])*sin(z[3])*z[12])+z[39]*((L2*cos(z[8])*cos(z[9])*z[17])*0.5-(L2*sin(z[8])*sin(z[9])*z[18])*0.5)+z[38]*((L2*cos(z[8])*cos(z[9])*z[18])*0.5+(sqrt(3)*L2*cos(z[8])*z[17])*0.5-(L2*sin(z[8])*sin(z[9])*z[17])*0.5)+z[47]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)+L2*cos(z[2])*sin(z[3])*z[41]+L2*cos(z[3])*sin(z[2])*z[42]+(L2*cos(z[9])*sin(z[8])*z[48])*0.5+(sqrt(3)*L1*sin(z[7])*z[46])*0.5+(sqrt(3)*L1*cos(z[7])*z[37]*z[16])*0.5
            res[59] = z[39]*((sqrt(3)*L2*cos(z[8])*cos(z[9])*z[17])*0.5-(sqrt(3)*L2*sin(z[8])*sin(z[9])*z[18])*0.5)-z[38]*((L2*cos(z[8])*z[17])*0.5-(sqrt(3)*L2*cos(z[8])*cos(z[9])*z[18])*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9])*z[17])*0.5)-z[47]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)-L1*sin(z[1])*z[40]-L2*sin(z[2])*z[41]-(L1*sin(z[7])*z[46])*0.5-L1*cos(z[1])*z[31]*z[10]-L2*cos(z[2])*z[32]*z[11]-(L1*cos(z[7])*z[37]*z[16])*0.5+(sqrt(3)*L2*cos(z[9])*sin(z[8])*z[48])*0.5
            res[60] = z[38]*(L2*cos(z[9])*sin(z[8])*z[17]+L2*cos(z[8])*sin(z[9])*z[18])-z[33]*(L2*cos(z[2])*sin(z[3])*z[11]+L2*cos(z[3])*sin(z[2])*z[12])-z[32]*(L2*cos(z[3])*sin(z[2])*z[11]+L2*cos(z[2])*sin(z[3])*z[12])+z[39]*(L2*cos(z[8])*sin(z[9])*z[17]+L2*cos(z[9])*sin(z[8])*z[18])+L1*cos(z[1])*z[40]-L1*cos(z[7])*z[46]+L2*cos(z[2])*cos(z[3])*z[41]-L2*cos(z[8])*cos(z[9])*z[47]-L2*sin(z[2])*sin(z[3])*z[42]+L2*sin(z[8])*sin(z[9])*z[48]-L1*sin(z[1])*z[31]*z[10]+L1*sin(z[7])*z[37]*z[16]
            # Parameter-specific part for J1
            res[40] += z[10]
            res[41] += z[11]
            res[42] += z[12]
            res[43] += z[13]
            res[44] += z[14]
            res[45] += z[15]
            res[46] += z[16]
            res[47] += z[17]
            res[48] += z[18]

            nothing
        end

        z0, dz0 = get_delta_initial_γsens_comp(θ, u(0.0), w(0.0))

        dvars = fill(true, 60)
        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 for delta robot is: $r0"

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, z0, dz0, dvars, r0)
    end
end

function delta_robot_gc_M1sens(Φ::Float64, u::Function, w::Function, θ::Vector{Float64})::Model 
    let L0 = θ[1], L1 = θ[2], L2 = θ[3], L3 = θ[4], LC1 = θ[5], LC2 = θ[6], M1 = θ[7], M2 = θ[8], M3 = θ[9], J1 = θ[10], J2 = θ[11], g = 0.0, γ = θ[13]
        # @info "Samples: $(u(0.3)), $(u(0.46)), $(u(1.23)), $(w(0.3)), $(w(0.46)), $(w(1.23))"
        function f!(res, dz, z, _, t)
            ut = u(t)
            wt = w(t)

            res[1] = dz[1]-z[10]+L1*cos(z[1])*dz[27]+L1*cos(z[1])*dz[30]-L1*sin(z[1])*dz[26]-L1*sin(z[1])*dz[29]
            res[2] = dz[2]-z[11]-L2*sin(z[2])*dz[26]-L2*sin(z[2])*dz[29]+L2*cos(z[2])*cos(z[3])*dz[27]+L2*cos(z[2])*cos(z[3])*dz[30]+L2*cos(z[2])*sin(z[3])*dz[25]+L2*cos(z[2])*sin(z[3])*dz[28]
            res[3] = dz[3]-z[12]+L2*cos(z[3])*sin(z[2])*dz[25]+L2*cos(z[3])*sin(z[2])*dz[28]-L2*sin(z[2])*sin(z[3])*dz[27]-L2*sin(z[2])*sin(z[3])*dz[30]
            res[4] = dz[4]-z[13]-L1*cos(z[4])*dz[27]-(L1*sin(z[4])*dz[26])*0.5-(sqrt(3)*L1*sin(z[4])*dz[25])*0.5
            res[5] = dz[5]-z[14]+dz[25]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)-dz[26]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-L2*cos(z[5])*cos(z[6])*dz[27]
            res[6] = dz[6]-z[15]+(L2*cos(z[6])*sin(z[5])*dz[25])*0.5+L2*sin(z[5])*sin(z[6])*dz[27]-(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[26])*0.5
            res[7] = dz[7]-z[16]-L1*cos(z[7])*dz[30]-(L1*sin(z[7])*dz[29])*0.5+(sqrt(3)*L1*sin(z[7])*dz[28])*0.5
            res[8] = dz[8]-z[17]+dz[28]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)-dz[29]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)-L2*cos(z[8])*cos(z[9])*dz[30]
            res[9] = dz[9]-z[18]+(L2*cos(z[9])*sin(z[8])*dz[28])*0.5+L2*sin(z[8])*sin(z[9])*dz[30]+(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[29])*0.5
            res[10] = dz[10]*(J1+L1^2*(M2+M3)+LC1^2*M1)-ut[1]-wt[1]^2+γ*z[10]-0.0*cos(z[1])*(L1*(M2+M3)+LC1*M1)-L1*cos(z[1])*dz[21]-L1*cos(z[1])*dz[24]+L1*sin(z[1])*dz[20]+L1*sin(z[1])*dz[23]+L1*dz[11]*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*dz[12]*(L2*M3+LC2*M2)-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2)
            res[11] = dz[11]*(J2+L2^2*M3+LC2^2*M2)+γ*z[11]+L2*sin(z[2])*dz[20]+L2*sin(z[2])*dz[23]-L2*cos(z[2])*cos(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[24]-L2*cos(z[2])*sin(z[3])*dz[19]-L2*cos(z[2])*sin(z[3])*dz[22]+L1*dz[10]*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))-cos(z[2])*sin(z[2])*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-0.0*cos(z[2])*cos(z[3])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))
            res[12] = γ*z[12]+sin(z[2])^2*dz[12]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[3])*sin(z[2])*dz[19]-L2*cos(z[3])*sin(z[2])*dz[22]+L2*sin(z[2])*sin(z[3])*dz[21]+L2*sin(z[2])*sin(z[3])*dz[24]+sin(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+0.0*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*sin(z[1])*sin(z[2])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*dz[10]*(L2*M3+LC2*M2)
            res[13] = dz[13]*(J1+L1^2*(M2+M3)+LC1^2*M1)-ut[2]-wt[2]^2+γ*z[13]-0.0*cos(z[4])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[4])*dz[21]+(L1*sin(z[4])*dz[20])*0.5+(sqrt(3)*L1*sin(z[4])*dz[19])*0.5+L1*dz[14]*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)-L1*cos(z[4])*sin(z[5])*sin(z[6])*dz[15]*(L2*M3+LC2*M2)-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2)
            res[14] = dz[20]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-dz[19]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)+dz[14]*(J2+L2^2*M3+LC2^2*M2)+γ*z[14]+L2*cos(z[5])*cos(z[6])*dz[21]+L1*dz[13]*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))-cos(z[5])*sin(z[5])*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-0.0*cos(z[5])*cos(z[6])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))
            res[15] = γ*z[15]+sin(z[5])^2*dz[15]*(J2+L2^2*M3+LC2^2*M2)-(L2*cos(z[6])*sin(z[5])*dz[19])*0.5-L2*sin(z[5])*sin(z[6])*dz[21]+sin(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)+0.0*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[20])*0.5+L1*sin(z[4])*sin(z[5])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)-L1*cos(z[4])*sin(z[5])*sin(z[6])*dz[13]*(L2*M3+LC2*M2)
            res[16] = dz[16]*(J1+L1^2*(M2+M3)+LC1^2*M1)-ut[3]-wt[3]^2+γ*z[16]-0.0*cos(z[7])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[7])*dz[24]+(L1*sin(z[7])*dz[23])*0.5-(sqrt(3)*L1*sin(z[7])*dz[22])*0.5+L1*dz[17]*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)-L1*cos(z[7])*sin(z[8])*sin(z[9])*dz[18]*(L2*M3+LC2*M2)-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2)
            res[17] = dz[23]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)-dz[22]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)+dz[17]*(J2+L2^2*M3+LC2^2*M2)+γ*z[17]+L2*cos(z[8])*cos(z[9])*dz[24]+L1*dz[16]*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))-cos(z[8])*sin(z[8])*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-0.0*cos(z[8])*cos(z[9])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))
            res[18] = γ*z[18]+sin(z[8])^2*dz[18]*(J2+L2^2*M3+LC2^2*M2)-(L2*cos(z[9])*sin(z[8])*dz[22])*0.5-L2*sin(z[8])*sin(z[9])*dz[24]+sin(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)+0.0*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)-(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[23])*0.5+L1*sin(z[7])*sin(z[8])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)-L1*cos(z[7])*sin(z[8])*sin(z[9])*dz[16]*(L2*M3+LC2*M2)
            res[19] = (sqrt(3)*(L0-L3+L1*cos(z[4])+L2*cos(z[5])))*0.5+L2*sin(z[2])*sin(z[3])+(L2*sin(z[5])*sin(z[6]))*0.5
            res[20] = (3*L0)*0.5-(3*L3)*0.5+L1*cos(z[1])+L2*cos(z[2])+(L1*cos(z[4]))*0.5+(L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5
            res[21] = L1*sin(z[1])-L1*sin(z[4])+L2*cos(z[3])*sin(z[2])-L2*cos(z[6])*sin(z[5])
            res[22] = L2*sin(z[2])*sin(z[3])-(sqrt(3)*(L0-L3+L1*cos(z[7])+L2*cos(z[8])))*0.5+(L2*sin(z[8])*sin(z[9]))*0.5
            res[23] = (3*L0)*0.5-(3*L3)*0.5+L1*cos(z[1])+L2*cos(z[2])+(L1*cos(z[7]))*0.5+(L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5
            res[24] = L1*sin(z[1])-L1*sin(z[7])+L2*cos(z[3])*sin(z[2])-L2*cos(z[9])*sin(z[8])
            res[25] = L2*cos(z[2])*sin(z[3])*z[11]-(sqrt(3)*(L1*sin(z[4])*z[13]+L2*sin(z[5])*z[14]))*0.5+L2*cos(z[3])*sin(z[2])*z[12]+(L2*cos(z[5])*sin(z[6])*z[14])*0.5+(L2*cos(z[6])*sin(z[5])*z[15])*0.5
            res[26] = -L1*sin(z[1])*z[10]-L2*sin(z[2])*z[11]-(L1*sin(z[4])*z[13])*0.5-(L2*sin(z[5])*z[14])*0.5-(sqrt(3)*L2*cos(z[5])*sin(z[6])*z[14])*0.5-(sqrt(3)*L2*cos(z[6])*sin(z[5])*z[15])*0.5
            res[27] = L1*cos(z[1])*z[10]-L1*cos(z[4])*z[13]+L2*cos(z[2])*cos(z[3])*z[11]-L2*cos(z[5])*cos(z[6])*z[14]-L2*sin(z[2])*sin(z[3])*z[12]+L2*sin(z[5])*sin(z[6])*z[15]
            res[28] = (sqrt(3)*(L1*sin(z[7])*z[16]+L2*sin(z[8])*z[17]))*0.5+L2*cos(z[2])*sin(z[3])*z[11]+L2*cos(z[3])*sin(z[2])*z[12]+(L2*cos(z[8])*sin(z[9])*z[17])*0.5+(L2*cos(z[9])*sin(z[8])*z[18])*0.5
            res[29] = (sqrt(3)*L2*cos(z[8])*sin(z[9])*z[17])*0.5-L2*sin(z[2])*z[11]-(L1*sin(z[7])*z[16])*0.5-(L2*sin(z[8])*z[17])*0.5-L1*sin(z[1])*z[10]+(sqrt(3)*L2*cos(z[9])*sin(z[8])*z[18])*0.5
            res[30] = L1*cos(z[1])*z[10]-L1*cos(z[7])*z[16]+L2*cos(z[2])*cos(z[3])*z[11]-L2*cos(z[8])*cos(z[9])*z[17]-L2*sin(z[2])*sin(z[3])*z[12]+L2*sin(z[8])*sin(z[9])*z[18]
            # Sensitivity equations
            res[31] = dz[31]-z[40]-z[31]*(L1*cos(z[1])*dz[26]+L1*cos(z[1])*dz[29]+L1*sin(z[1])*dz[27]+L1*sin(z[1])*dz[30])+L1*cos(z[1])*dz[57]+L1*cos(z[1])*dz[60]-L1*sin(z[1])*dz[56]-L1*sin(z[1])*dz[59]
            res[32] = dz[32]-z[41]+z[33]*(L2*cos(z[2])*cos(z[3])*dz[25]+L2*cos(z[2])*cos(z[3])*dz[28]-L2*cos(z[2])*sin(z[3])*dz[27]-L2*cos(z[2])*sin(z[3])*dz[30])-z[32]*(L2*cos(z[2])*dz[26]+L2*cos(z[2])*dz[29]+L2*cos(z[3])*sin(z[2])*dz[27]+L2*cos(z[3])*sin(z[2])*dz[30]+L2*sin(z[2])*sin(z[3])*dz[25]+L2*sin(z[2])*sin(z[3])*dz[28])-L2*sin(z[2])*dz[56]-L2*sin(z[2])*dz[59]+L2*cos(z[2])*cos(z[3])*dz[57]+L2*cos(z[2])*cos(z[3])*dz[60]+L2*cos(z[2])*sin(z[3])*dz[55]+L2*cos(z[2])*sin(z[3])*dz[58]
            res[33] = dz[33]-z[42]+z[32]*(L2*cos(z[2])*cos(z[3])*dz[25]+L2*cos(z[2])*cos(z[3])*dz[28]-L2*cos(z[2])*sin(z[3])*dz[27]-L2*cos(z[2])*sin(z[3])*dz[30])-z[33]*(L2*cos(z[3])*sin(z[2])*dz[27]+L2*cos(z[3])*sin(z[2])*dz[30]+L2*sin(z[2])*sin(z[3])*dz[25]+L2*sin(z[2])*sin(z[3])*dz[28])+L2*cos(z[3])*sin(z[2])*dz[55]+L2*cos(z[3])*sin(z[2])*dz[58]-L2*sin(z[2])*sin(z[3])*dz[57]-L2*sin(z[2])*sin(z[3])*dz[60]
            res[34] = dz[34]-z[43]-z[34]*((L1*cos(z[4])*dz[26])*0.5-L1*sin(z[4])*dz[27]+(sqrt(3)*L1*cos(z[4])*dz[25])*0.5)-L1*cos(z[4])*dz[57]-(L1*sin(z[4])*dz[56])*0.5-(sqrt(3)*L1*sin(z[4])*dz[55])*0.5
            res[35] = dz[35]-z[44]+dz[55]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)-dz[56]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)+z[36]*((L2*cos(z[5])*cos(z[6])*dz[25])*0.5+L2*cos(z[5])*sin(z[6])*dz[27]-(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[26])*0.5)-z[35]*(dz[25]*((sqrt(3)*L2*cos(z[5]))*0.5+(L2*sin(z[5])*sin(z[6]))*0.5)+dz[26]*((L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5)-L2*cos(z[6])*sin(z[5])*dz[27])-L2*cos(z[5])*cos(z[6])*dz[57]
            res[36] = dz[36]-z[45]+z[35]*((L2*cos(z[5])*cos(z[6])*dz[25])*0.5+L2*cos(z[5])*sin(z[6])*dz[27]-(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[26])*0.5)+z[36]*(L2*cos(z[6])*sin(z[5])*dz[27]-(L2*sin(z[5])*sin(z[6])*dz[25])*0.5+(sqrt(3)*L2*sin(z[5])*sin(z[6])*dz[26])*0.5)+(L2*cos(z[6])*sin(z[5])*dz[55])*0.5+L2*sin(z[5])*sin(z[6])*dz[57]-(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[56])*0.5
            res[37] = dz[37]-z[46]+z[37]*(L1*sin(z[7])*dz[30]-(L1*cos(z[7])*dz[29])*0.5+(sqrt(3)*L1*cos(z[7])*dz[28])*0.5)-L1*cos(z[7])*dz[60]-(L1*sin(z[7])*dz[59])*0.5+(sqrt(3)*L1*sin(z[7])*dz[58])*0.5
            res[38] = dz[38]-z[47]+dz[58]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)-dz[59]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)+z[39]*((L2*cos(z[8])*cos(z[9])*dz[28])*0.5+L2*cos(z[8])*sin(z[9])*dz[30]+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[29])*0.5)+z[38]*(dz[28]*((sqrt(3)*L2*cos(z[8]))*0.5-(L2*sin(z[8])*sin(z[9]))*0.5)-dz[29]*((L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5)+L2*cos(z[9])*sin(z[8])*dz[30])-L2*cos(z[8])*cos(z[9])*dz[60]
            res[39] = dz[39]-z[48]+z[38]*((L2*cos(z[8])*cos(z[9])*dz[28])*0.5+L2*cos(z[8])*sin(z[9])*dz[30]+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[29])*0.5)-z[39]*((L2*sin(z[8])*sin(z[9])*dz[28])*0.5-L2*cos(z[9])*sin(z[8])*dz[30]+(sqrt(3)*L2*sin(z[8])*sin(z[9])*dz[29])*0.5)+(L2*cos(z[9])*sin(z[8])*dz[58])*0.5+L2*sin(z[8])*sin(z[9])*dz[60]+(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[59])*0.5
            res[40] = z[41]*(2*L1*z[11]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[12]*(L2*M3+LC2*M2))+dz[40]*(J1+L1^2*(M2+M3)+LC1^2*M1)-z[42]*(2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]*(L2*M3+LC2*M2))-z[33]*(L1*cos(z[1])*cos(z[2])*sin(z[3])*dz[11]*(L2*M3+LC2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*z[12]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]^2*(L2*M3+LC2*M2)+L1*cos(z[1])*cos(z[3])*sin(z[2])*dz[12]*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[2])*cos(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+z[31]*(0.0*sin(z[1])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[1])*dz[20]+L1*cos(z[1])*dz[23]+L1*sin(z[1])*dz[21]+L1*sin(z[1])*dz[24]+L1*dz[11]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2]))+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)+L1*sin(z[1])*sin(z[2])*sin(z[3])*dz[12]*(L2*M3+LC2*M2)+2*L1*cos(z[2])*sin(z[1])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+γ*z[40]-z[32]*(L1*z[11]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))-L1*dz[11]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))+L1*cos(z[1])*cos(z[2])*cos(z[3])*z[12]^2*(L2*M3+LC2*M2)+L1*cos(z[1])*cos(z[2])*sin(z[3])*dz[12]*(L2*M3+LC2*M2)-2*L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))-L1*cos(z[1])*dz[51]-L1*cos(z[1])*dz[54]+L1*sin(z[1])*dz[50]+L1*sin(z[1])*dz[53]+L1*dz[41]*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))-L1*cos(z[1])*sin(z[2])*sin(z[3])*dz[42]*(L2*M3+LC2*M2)
            res[41] = z[31]*(L1*dz[10]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))-L1*z[10]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3])))+z[32]*(sin(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)+L2*cos(z[2])*dz[20]+L2*cos(z[2])*dz[23]+L2*cos(z[3])*sin(z[2])*dz[21]+L2*cos(z[3])*sin(z[2])*dz[24]+L2*sin(z[2])*sin(z[3])*dz[19]+L2*sin(z[2])*sin(z[3])*dz[22]+L1*dz[10]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))+0.0*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2])))+dz[41]*(J2+L2^2*M3+LC2^2*M2)+γ*z[41]+z[33]*(L2*cos(z[2])*sin(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[22]-L2*cos(z[2])*cos(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[24]+0.0*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[2])*sin(z[3])*dz[10]*(L2*M3+LC2*M2))+L2*sin(z[2])*dz[50]+L2*sin(z[2])*dz[53]-L2*cos(z[2])*cos(z[3])*dz[51]-L2*cos(z[2])*cos(z[3])*dz[54]-L2*cos(z[2])*sin(z[3])*dz[49]-L2*cos(z[2])*sin(z[3])*dz[52]+L1*dz[40]*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))-2*cos(z[2])*sin(z[2])*z[42]*z[12]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[40]*z[10]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))
            res[42] = z[33]*(L2*cos(z[3])*sin(z[2])*dz[21]+L2*cos(z[3])*sin(z[2])*dz[24]+L2*sin(z[2])*sin(z[3])*dz[19]+L2*sin(z[2])*sin(z[3])*dz[22]+0.0*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[10]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[3])*sin(z[2])*dz[10]*(L2*M3+LC2*M2))+z[32]*(L2*cos(z[2])*sin(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[22]-L2*cos(z[2])*cos(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[24]+2*cos(z[2])*sin(z[2])*dz[12]*(J2+L2^2*M3+LC2^2*M2)+2*cos(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+0.0*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[2])*sin(z[3])*dz[10]*(L2*M3+LC2*M2))+z[42]*(γ+sin(2*z[2])*z[11]*(J2+L2^2*M3+LC2^2*M2))+z[31]*(L1*cos(z[1])*sin(z[2])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)+L1*sin(z[1])*sin(z[2])*sin(z[3])*dz[10]*(L2*M3+LC2*M2))+sin(z[2])^2*dz[42]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[3])*sin(z[2])*dz[49]-L2*cos(z[3])*sin(z[2])*dz[52]+L2*sin(z[2])*sin(z[3])*dz[51]+L2*sin(z[2])*sin(z[3])*dz[54]+sin(2*z[2])*z[41]*z[12]*(J2+L2^2*M3+LC2^2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*dz[40]*(L2*M3+LC2*M2)+2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[40]*z[10]*(L2*M3+LC2*M2)
            res[43] = z[44]*(2*L1*z[14]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[15]*(L2*M3+LC2*M2))+dz[43]*(J1+L1^2*(M2+M3)+LC1^2*M1)-z[45]*(2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]*(L2*M3+LC2*M2))-z[36]*(L1*cos(z[4])*cos(z[5])*sin(z[6])*dz[14]*(L2*M3+LC2*M2)-L1*cos(z[4])*sin(z[5])*sin(z[6])*z[15]^2*(L2*M3+LC2*M2)-L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*cos(z[6])*sin(z[5])*dz[15]*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[5])*cos(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+γ*z[43]-z[35]*(L1*z[14]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))-L1*dz[14]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))+L1*cos(z[4])*cos(z[5])*cos(z[6])*z[15]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*cos(z[5])*sin(z[6])*dz[15]*(L2*M3+LC2*M2)-2*L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+z[34]*(0.0*sin(z[4])*(L1*(M2+M3)+LC1*M1)+(L1*cos(z[4])*dz[20])*0.5-L1*sin(z[4])*dz[21]+(sqrt(3)*L1*cos(z[4])*dz[19])*0.5+L1*dz[14]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5]))+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)+L1*sin(z[4])*sin(z[5])*sin(z[6])*dz[15]*(L2*M3+LC2*M2)+2*L1*cos(z[5])*sin(z[4])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+L1*cos(z[4])*dz[51]+(L1*sin(z[4])*dz[50])*0.5+(sqrt(3)*L1*sin(z[4])*dz[49])*0.5+L1*dz[44]*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))-L1*cos(z[4])*sin(z[5])*sin(z[6])*dz[45]*(L2*M3+LC2*M2)
            res[44] = z[34]*(L1*dz[13]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))-L1*z[13]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6])))-z[36]*((L2*cos(z[5])*cos(z[6])*dz[19])*0.5+L2*cos(z[5])*sin(z[6])*dz[21]-0.0*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)-(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[20])*0.5-L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*cos(z[5])*sin(z[6])*dz[13]*(L2*M3+LC2*M2))-dz[49]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)+dz[50]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)+dz[44]*(J2+L2^2*M3+LC2^2*M2)+γ*z[44]+z[35]*(dz[19]*((sqrt(3)*L2*cos(z[5]))*0.5+(L2*sin(z[5])*sin(z[6]))*0.5)+dz[20]*((L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5)-cos(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)+sin(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[6])*sin(z[5])*dz[21]+L1*dz[13]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))+0.0*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5])))+L2*cos(z[5])*cos(z[6])*dz[51]+L1*dz[43]*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))-2*cos(z[5])*sin(z[5])*z[45]*z[15]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[43]*z[13]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))
            res[45] = z[45]*(γ+sin(2*z[5])*z[14]*(J2+L2^2*M3+LC2^2*M2))-z[36]*(L2*cos(z[6])*sin(z[5])*dz[21]-(L2*sin(z[5])*sin(z[6])*dz[19])*0.5-0.0*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+(sqrt(3)*L2*sin(z[5])*sin(z[6])*dz[20])*0.5-L1*cos(z[6])*sin(z[4])*sin(z[5])*z[13]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*cos(z[6])*sin(z[5])*dz[13]*(L2*M3+LC2*M2))+z[35]*(2*cos(z[5])*sin(z[5])*dz[15]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[5])*sin(z[6])*dz[21]-(L2*cos(z[5])*cos(z[6])*dz[19])*0.5+2*cos(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)+0.0*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[20])*0.5+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[5])*sin(z[6])*dz[13]*(L2*M3+LC2*M2))+z[34]*(L1*cos(z[4])*sin(z[5])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)+L1*sin(z[4])*sin(z[5])*sin(z[6])*dz[13]*(L2*M3+LC2*M2))+sin(z[5])^2*dz[45]*(J2+L2^2*M3+LC2^2*M2)-(L2*cos(z[6])*sin(z[5])*dz[49])*0.5-L2*sin(z[5])*sin(z[6])*dz[51]+sin(2*z[5])*z[44]*z[15]*(J2+L2^2*M3+LC2^2*M2)+(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[50])*0.5-L1*cos(z[4])*sin(z[5])*sin(z[6])*dz[43]*(L2*M3+LC2*M2)+2*L1*sin(z[4])*sin(z[5])*sin(z[6])*z[43]*z[13]*(L2*M3+LC2*M2)
            res[46] = z[47]*(2*L1*z[17]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[18]*(L2*M3+LC2*M2))+dz[46]*(J1+L1^2*(M2+M3)+LC1^2*M1)-z[48]*(2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]*(L2*M3+LC2*M2))-z[39]*(L1*cos(z[7])*cos(z[8])*sin(z[9])*dz[17]*(L2*M3+LC2*M2)-L1*cos(z[7])*sin(z[8])*sin(z[9])*z[18]^2*(L2*M3+LC2*M2)-L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*cos(z[9])*sin(z[8])*dz[18]*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[8])*cos(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+γ*z[46]-z[38]*(L1*z[17]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))-L1*dz[17]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))+L1*cos(z[7])*cos(z[8])*cos(z[9])*z[18]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*cos(z[8])*sin(z[9])*dz[18]*(L2*M3+LC2*M2)-2*L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+z[37]*(0.0*sin(z[7])*(L1*(M2+M3)+LC1*M1)+(L1*cos(z[7])*dz[23])*0.5-L1*sin(z[7])*dz[24]-(sqrt(3)*L1*cos(z[7])*dz[22])*0.5+L1*dz[17]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8]))+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)+L1*sin(z[7])*sin(z[8])*sin(z[9])*dz[18]*(L2*M3+LC2*M2)+2*L1*cos(z[8])*sin(z[7])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+L1*cos(z[7])*dz[54]+(L1*sin(z[7])*dz[53])*0.5-(sqrt(3)*L1*sin(z[7])*dz[52])*0.5+L1*dz[47]*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))-L1*cos(z[7])*sin(z[8])*sin(z[9])*dz[48]*(L2*M3+LC2*M2)
            res[47] = z[37]*(L1*dz[16]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))-L1*z[16]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9])))-z[39]*((L2*cos(z[8])*cos(z[9])*dz[22])*0.5+L2*cos(z[8])*sin(z[9])*dz[24]-0.0*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[23])*0.5-L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*cos(z[8])*sin(z[9])*dz[16]*(L2*M3+LC2*M2))-dz[52]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)+dz[53]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)+dz[47]*(J2+L2^2*M3+LC2^2*M2)+γ*z[47]+z[38]*(dz[23]*((L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5)-dz[22]*((sqrt(3)*L2*cos(z[8]))*0.5-(L2*sin(z[8])*sin(z[9]))*0.5)-cos(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)+sin(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[9])*sin(z[8])*dz[24]+L1*dz[16]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))+0.0*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8])))+L2*cos(z[8])*cos(z[9])*dz[54]+L1*dz[46]*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))-2*cos(z[8])*sin(z[8])*z[48]*z[18]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[46]*z[16]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))
            res[48] = z[39]*((L2*sin(z[8])*sin(z[9])*dz[22])*0.5-L2*cos(z[9])*sin(z[8])*dz[24]+0.0*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+(sqrt(3)*L2*sin(z[8])*sin(z[9])*dz[23])*0.5+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[16]^2*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[9])*sin(z[8])*dz[16]*(L2*M3+LC2*M2))+z[48]*(γ+sin(2*z[8])*z[17]*(J2+L2^2*M3+LC2^2*M2))-z[38]*((L2*cos(z[8])*cos(z[9])*dz[22])*0.5+L2*cos(z[8])*sin(z[9])*dz[24]-2*cos(z[8])*sin(z[8])*dz[18]*(J2+L2^2*M3+LC2^2*M2)-2*cos(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)-0.0*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[23])*0.5-L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*cos(z[8])*sin(z[9])*dz[16]*(L2*M3+LC2*M2))+z[37]*(L1*cos(z[7])*sin(z[8])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)+L1*sin(z[7])*sin(z[8])*sin(z[9])*dz[16]*(L2*M3+LC2*M2))+sin(z[8])^2*dz[48]*(J2+L2^2*M3+LC2^2*M2)-(L2*cos(z[9])*sin(z[8])*dz[52])*0.5-L2*sin(z[8])*sin(z[9])*dz[54]+sin(2*z[8])*z[47]*z[18]*(J2+L2^2*M3+LC2^2*M2)-(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[53])*0.5-L1*cos(z[7])*sin(z[8])*sin(z[9])*dz[46]*(L2*M3+LC2*M2)+2*L1*sin(z[7])*sin(z[8])*sin(z[9])*z[46]*z[16]*(L2*M3+LC2*M2)
            res[49] = z[35]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)+L2*cos(z[2])*sin(z[3])*z[32]+L2*cos(z[3])*sin(z[2])*z[33]+(L2*cos(z[6])*sin(z[5])*z[36])*0.5-(sqrt(3)*L1*sin(z[4])*z[34])*0.5
            res[50] = -z[35]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-L1*sin(z[1])*z[31]-L2*sin(z[2])*z[32]-(L1*sin(z[4])*z[34])*0.5-(sqrt(3)*L2*cos(z[6])*sin(z[5])*z[36])*0.5
            res[51] = L1*cos(z[1])*z[31]-L1*cos(z[4])*z[34]+L2*cos(z[2])*cos(z[3])*z[32]-L2*cos(z[5])*cos(z[6])*z[35]-L2*sin(z[2])*sin(z[3])*z[33]+L2*sin(z[5])*sin(z[6])*z[36]
            res[52] = z[38]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)+L2*cos(z[2])*sin(z[3])*z[32]+L2*cos(z[3])*sin(z[2])*z[33]+(L2*cos(z[9])*sin(z[8])*z[39])*0.5+(sqrt(3)*L1*sin(z[7])*z[37])*0.5
            res[53] = (sqrt(3)*L2*cos(z[9])*sin(z[8])*z[39])*0.5-L1*sin(z[1])*z[31]-L2*sin(z[2])*z[32]-(L1*sin(z[7])*z[37])*0.5-z[38]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)
            res[54] = L1*cos(z[1])*z[31]-L1*cos(z[7])*z[37]+L2*cos(z[2])*cos(z[3])*z[32]-L2*cos(z[8])*cos(z[9])*z[38]-L2*sin(z[2])*sin(z[3])*z[33]+L2*sin(z[8])*sin(z[9])*z[39]
            res[55] = z[32]*(L2*cos(z[2])*cos(z[3])*z[12]-L2*sin(z[2])*sin(z[3])*z[11])+z[33]*(L2*cos(z[2])*cos(z[3])*z[11]-L2*sin(z[2])*sin(z[3])*z[12])+z[36]*((L2*cos(z[5])*cos(z[6])*z[14])*0.5-(L2*sin(z[5])*sin(z[6])*z[15])*0.5)-z[35]*((sqrt(3)*L2*cos(z[5])*z[14])*0.5-(L2*cos(z[5])*cos(z[6])*z[15])*0.5+(L2*sin(z[5])*sin(z[6])*z[14])*0.5)+z[44]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)+L2*cos(z[2])*sin(z[3])*z[41]+L2*cos(z[3])*sin(z[2])*z[42]+(L2*cos(z[6])*sin(z[5])*z[45])*0.5-(sqrt(3)*L1*sin(z[4])*z[43])*0.5-(sqrt(3)*L1*cos(z[4])*z[34]*z[13])*0.5
            res[56] = -z[44]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-z[35]*((L2*cos(z[5])*z[14])*0.5+(sqrt(3)*L2*cos(z[5])*cos(z[6])*z[15])*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6])*z[14])*0.5)-z[36]*((sqrt(3)*L2*cos(z[5])*cos(z[6])*z[14])*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6])*z[15])*0.5)-L1*sin(z[1])*z[40]-L2*sin(z[2])*z[41]-(L1*sin(z[4])*z[43])*0.5-L1*cos(z[1])*z[31]*z[10]-L2*cos(z[2])*z[32]*z[11]-(L1*cos(z[4])*z[34]*z[13])*0.5-(sqrt(3)*L2*cos(z[6])*sin(z[5])*z[45])*0.5
            res[57] = z[35]*(L2*cos(z[6])*sin(z[5])*z[14]+L2*cos(z[5])*sin(z[6])*z[15])-z[33]*(L2*cos(z[2])*sin(z[3])*z[11]+L2*cos(z[3])*sin(z[2])*z[12])-z[32]*(L2*cos(z[3])*sin(z[2])*z[11]+L2*cos(z[2])*sin(z[3])*z[12])+z[36]*(L2*cos(z[5])*sin(z[6])*z[14]+L2*cos(z[6])*sin(z[5])*z[15])+L1*cos(z[1])*z[40]-L1*cos(z[4])*z[43]+L2*cos(z[2])*cos(z[3])*z[41]-L2*cos(z[5])*cos(z[6])*z[44]-L2*sin(z[2])*sin(z[3])*z[42]+L2*sin(z[5])*sin(z[6])*z[45]-L1*sin(z[1])*z[31]*z[10]+L1*sin(z[4])*z[34]*z[13]
            res[58] = z[32]*(L2*cos(z[2])*cos(z[3])*z[12]-L2*sin(z[2])*sin(z[3])*z[11])+z[33]*(L2*cos(z[2])*cos(z[3])*z[11]-L2*sin(z[2])*sin(z[3])*z[12])+z[39]*((L2*cos(z[8])*cos(z[9])*z[17])*0.5-(L2*sin(z[8])*sin(z[9])*z[18])*0.5)+z[38]*((L2*cos(z[8])*cos(z[9])*z[18])*0.5+(sqrt(3)*L2*cos(z[8])*z[17])*0.5-(L2*sin(z[8])*sin(z[9])*z[17])*0.5)+z[47]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)+L2*cos(z[2])*sin(z[3])*z[41]+L2*cos(z[3])*sin(z[2])*z[42]+(L2*cos(z[9])*sin(z[8])*z[48])*0.5+(sqrt(3)*L1*sin(z[7])*z[46])*0.5+(sqrt(3)*L1*cos(z[7])*z[37]*z[16])*0.5
            res[59] = z[39]*((sqrt(3)*L2*cos(z[8])*cos(z[9])*z[17])*0.5-(sqrt(3)*L2*sin(z[8])*sin(z[9])*z[18])*0.5)-z[38]*((L2*cos(z[8])*z[17])*0.5-(sqrt(3)*L2*cos(z[8])*cos(z[9])*z[18])*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9])*z[17])*0.5)-z[47]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)-L1*sin(z[1])*z[40]-L2*sin(z[2])*z[41]-(L1*sin(z[7])*z[46])*0.5-L1*cos(z[1])*z[31]*z[10]-L2*cos(z[2])*z[32]*z[11]-(L1*cos(z[7])*z[37]*z[16])*0.5+(sqrt(3)*L2*cos(z[9])*sin(z[8])*z[48])*0.5
            res[60] = z[38]*(L2*cos(z[9])*sin(z[8])*z[17]+L2*cos(z[8])*sin(z[9])*z[18])-z[33]*(L2*cos(z[2])*sin(z[3])*z[11]+L2*cos(z[3])*sin(z[2])*z[12])-z[32]*(L2*cos(z[3])*sin(z[2])*z[11]+L2*cos(z[2])*sin(z[3])*z[12])+z[39]*(L2*cos(z[8])*sin(z[9])*z[17]+L2*cos(z[9])*sin(z[8])*z[18])+L1*cos(z[1])*z[40]-L1*cos(z[7])*z[46]+L2*cos(z[2])*cos(z[3])*z[41]-L2*cos(z[8])*cos(z[9])*z[47]-L2*sin(z[2])*sin(z[3])*z[42]+L2*sin(z[8])*sin(z[9])*z[48]-L1*sin(z[1])*z[31]*z[10]+L1*sin(z[7])*z[37]*z[16]
            # Parameter-specific part for M1
            res[40] += LC1^2*dz[10]-LC1*0.0*cos(z[1])
            res[43] += LC1^2*dz[13]-LC1*0.0*cos(z[4])
            res[46] += LC1^2*dz[16]-LC1*0.0*cos(z[7])
            nothing
        end

        z0, dz0 = get_delta_initial_M1sens_comp(θ, u(0.0), w(0.0))

        dvars = fill(true, 60)
        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 for delta robot is: $r0"

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, z0, dz0, dvars, r0)
    end
end

function delta_robot_gc_L1M1J1sens(Φ::Float64, u::Function, w::Function, θ::Vector{Float64})::Model 
    let L0 = θ[1], L1 = θ[2], L2 = θ[3], L3 = θ[4], LC1 = θ[5], LC2 = θ[6], M1 = θ[7], M2 = θ[8], M3 = θ[9], J1 = θ[10], J2 = θ[11], g = 0.0, γ = θ[13]
        # @info "Samples: $(u(0.3)), $(u(0.46)), $(u(1.23)), $(w(0.3)), $(w(0.46)), $(w(1.23))"
        function f!(res, dz, z, _, t)
            ut = u(t)
            wt = w(t)

            res[1] = dz[1]-z[10]+L1*cos(z[1])*dz[27]+L1*cos(z[1])*dz[30]-L1*sin(z[1])*dz[26]-L1*sin(z[1])*dz[29]
            res[2] = dz[2]-z[11]-L2*sin(z[2])*dz[26]-L2*sin(z[2])*dz[29]+L2*cos(z[2])*cos(z[3])*dz[27]+L2*cos(z[2])*cos(z[3])*dz[30]+L2*cos(z[2])*sin(z[3])*dz[25]+L2*cos(z[2])*sin(z[3])*dz[28]
            res[3] = dz[3]-z[12]+L2*cos(z[3])*sin(z[2])*dz[25]+L2*cos(z[3])*sin(z[2])*dz[28]-L2*sin(z[2])*sin(z[3])*dz[27]-L2*sin(z[2])*sin(z[3])*dz[30]
            res[4] = dz[4]-z[13]-L1*cos(z[4])*dz[27]-(L1*sin(z[4])*dz[26])*0.5-(sqrt(3)*L1*sin(z[4])*dz[25])*0.5
            res[5] = dz[5]-z[14]+dz[25]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)-dz[26]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-L2*cos(z[5])*cos(z[6])*dz[27]
            res[6] = dz[6]-z[15]+(L2*cos(z[6])*sin(z[5])*dz[25])*0.5+L2*sin(z[5])*sin(z[6])*dz[27]-(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[26])*0.5
            res[7] = dz[7]-z[16]-L1*cos(z[7])*dz[30]-(L1*sin(z[7])*dz[29])*0.5+(sqrt(3)*L1*sin(z[7])*dz[28])*0.5
            res[8] = dz[8]-z[17]+dz[28]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)-dz[29]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)-L2*cos(z[8])*cos(z[9])*dz[30]
            res[9] = dz[9]-z[18]+(L2*cos(z[9])*sin(z[8])*dz[28])*0.5+L2*sin(z[8])*sin(z[9])*dz[30]+(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[29])*0.5
            res[10] = dz[10]*(J1+L1^2*(M2+M3)+LC1^2*M1)-ut[1]-wt[1]^2+γ*z[10]-0.0*cos(z[1])*(L1*(M2+M3)+LC1*M1)-L1*cos(z[1])*dz[21]-L1*cos(z[1])*dz[24]+L1*sin(z[1])*dz[20]+L1*sin(z[1])*dz[23]+L1*dz[11]*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*dz[12]*(L2*M3+LC2*M2)-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2)
            res[11] = dz[11]*(J2+L2^2*M3+LC2^2*M2)+γ*z[11]+L2*sin(z[2])*dz[20]+L2*sin(z[2])*dz[23]-L2*cos(z[2])*cos(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[24]-L2*cos(z[2])*sin(z[3])*dz[19]-L2*cos(z[2])*sin(z[3])*dz[22]+L1*dz[10]*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))-cos(z[2])*sin(z[2])*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-0.0*cos(z[2])*cos(z[3])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))
            res[12] = γ*z[12]+sin(z[2])^2*dz[12]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[3])*sin(z[2])*dz[19]-L2*cos(z[3])*sin(z[2])*dz[22]+L2*sin(z[2])*sin(z[3])*dz[21]+L2*sin(z[2])*sin(z[3])*dz[24]+sin(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+0.0*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*sin(z[1])*sin(z[2])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*dz[10]*(L2*M3+LC2*M2)
            res[13] = dz[13]*(J1+L1^2*(M2+M3)+LC1^2*M1)-ut[2]-wt[2]^2+γ*z[13]-0.0*cos(z[4])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[4])*dz[21]+(L1*sin(z[4])*dz[20])*0.5+(sqrt(3)*L1*sin(z[4])*dz[19])*0.5+L1*dz[14]*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)-L1*cos(z[4])*sin(z[5])*sin(z[6])*dz[15]*(L2*M3+LC2*M2)-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2)
            res[14] = dz[20]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-dz[19]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)+dz[14]*(J2+L2^2*M3+LC2^2*M2)+γ*z[14]+L2*cos(z[5])*cos(z[6])*dz[21]+L1*dz[13]*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))-cos(z[5])*sin(z[5])*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-0.0*cos(z[5])*cos(z[6])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))
            res[15] = γ*z[15]+sin(z[5])^2*dz[15]*(J2+L2^2*M3+LC2^2*M2)-(L2*cos(z[6])*sin(z[5])*dz[19])*0.5-L2*sin(z[5])*sin(z[6])*dz[21]+sin(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)+0.0*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[20])*0.5+L1*sin(z[4])*sin(z[5])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)-L1*cos(z[4])*sin(z[5])*sin(z[6])*dz[13]*(L2*M3+LC2*M2)
            res[16] = dz[16]*(J1+L1^2*(M2+M3)+LC1^2*M1)-ut[3]-wt[3]^2+γ*z[16]-0.0*cos(z[7])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[7])*dz[24]+(L1*sin(z[7])*dz[23])*0.5-(sqrt(3)*L1*sin(z[7])*dz[22])*0.5+L1*dz[17]*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)-L1*cos(z[7])*sin(z[8])*sin(z[9])*dz[18]*(L2*M3+LC2*M2)-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2)
            res[17] = dz[23]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)-dz[22]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)+dz[17]*(J2+L2^2*M3+LC2^2*M2)+γ*z[17]+L2*cos(z[8])*cos(z[9])*dz[24]+L1*dz[16]*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))-cos(z[8])*sin(z[8])*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-0.0*cos(z[8])*cos(z[9])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))
            res[18] = γ*z[18]+sin(z[8])^2*dz[18]*(J2+L2^2*M3+LC2^2*M2)-(L2*cos(z[9])*sin(z[8])*dz[22])*0.5-L2*sin(z[8])*sin(z[9])*dz[24]+sin(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)+0.0*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)-(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[23])*0.5+L1*sin(z[7])*sin(z[8])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)-L1*cos(z[7])*sin(z[8])*sin(z[9])*dz[16]*(L2*M3+LC2*M2)
            res[19] = (sqrt(3)*(L0-L3+L1*cos(z[4])+L2*cos(z[5])))*0.5+L2*sin(z[2])*sin(z[3])+(L2*sin(z[5])*sin(z[6]))*0.5
            res[20] = (3*L0)*0.5-(3*L3)*0.5+L1*cos(z[1])+L2*cos(z[2])+(L1*cos(z[4]))*0.5+(L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5
            res[21] = L1*sin(z[1])-L1*sin(z[4])+L2*cos(z[3])*sin(z[2])-L2*cos(z[6])*sin(z[5])
            res[22] = L2*sin(z[2])*sin(z[3])-(sqrt(3)*(L0-L3+L1*cos(z[7])+L2*cos(z[8])))*0.5+(L2*sin(z[8])*sin(z[9]))*0.5
            res[23] = (3*L0)*0.5-(3*L3)*0.5+L1*cos(z[1])+L2*cos(z[2])+(L1*cos(z[7]))*0.5+(L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5
            res[24] = L1*sin(z[1])-L1*sin(z[7])+L2*cos(z[3])*sin(z[2])-L2*cos(z[9])*sin(z[8])
            res[25] = L2*cos(z[2])*sin(z[3])*z[11]-(sqrt(3)*(L1*sin(z[4])*z[13]+L2*sin(z[5])*z[14]))*0.5+L2*cos(z[3])*sin(z[2])*z[12]+(L2*cos(z[5])*sin(z[6])*z[14])*0.5+(L2*cos(z[6])*sin(z[5])*z[15])*0.5
            res[26] = -L1*sin(z[1])*z[10]-L2*sin(z[2])*z[11]-(L1*sin(z[4])*z[13])*0.5-(L2*sin(z[5])*z[14])*0.5-(sqrt(3)*L2*cos(z[5])*sin(z[6])*z[14])*0.5-(sqrt(3)*L2*cos(z[6])*sin(z[5])*z[15])*0.5
            res[27] = L1*cos(z[1])*z[10]-L1*cos(z[4])*z[13]+L2*cos(z[2])*cos(z[3])*z[11]-L2*cos(z[5])*cos(z[6])*z[14]-L2*sin(z[2])*sin(z[3])*z[12]+L2*sin(z[5])*sin(z[6])*z[15]
            res[28] = (sqrt(3)*(L1*sin(z[7])*z[16]+L2*sin(z[8])*z[17]))*0.5+L2*cos(z[2])*sin(z[3])*z[11]+L2*cos(z[3])*sin(z[2])*z[12]+(L2*cos(z[8])*sin(z[9])*z[17])*0.5+(L2*cos(z[9])*sin(z[8])*z[18])*0.5
            res[29] = (sqrt(3)*L2*cos(z[8])*sin(z[9])*z[17])*0.5-L2*sin(z[2])*z[11]-(L1*sin(z[7])*z[16])*0.5-(L2*sin(z[8])*z[17])*0.5-L1*sin(z[1])*z[10]+(sqrt(3)*L2*cos(z[9])*sin(z[8])*z[18])*0.5
            res[30] = L1*cos(z[1])*z[10]-L1*cos(z[7])*z[16]+L2*cos(z[2])*cos(z[3])*z[11]-L2*cos(z[8])*cos(z[9])*z[17]-L2*sin(z[2])*sin(z[3])*z[12]+L2*sin(z[8])*sin(z[9])*z[18]
            # Sensitivity equations wrt L1
            res[31] = dz[31]-z[40]-z[31]*(L1*cos(z[1])*dz[26]+L1*cos(z[1])*dz[29]+L1*sin(z[1])*dz[27]+L1*sin(z[1])*dz[30])+L1*cos(z[1])*dz[57]+L1*cos(z[1])*dz[60]-L1*sin(z[1])*dz[56]-L1*sin(z[1])*dz[59]
            res[32] = dz[32]-z[41]+z[33]*(L2*cos(z[2])*cos(z[3])*dz[25]+L2*cos(z[2])*cos(z[3])*dz[28]-L2*cos(z[2])*sin(z[3])*dz[27]-L2*cos(z[2])*sin(z[3])*dz[30])-z[32]*(L2*cos(z[2])*dz[26]+L2*cos(z[2])*dz[29]+L2*cos(z[3])*sin(z[2])*dz[27]+L2*cos(z[3])*sin(z[2])*dz[30]+L2*sin(z[2])*sin(z[3])*dz[25]+L2*sin(z[2])*sin(z[3])*dz[28])-L2*sin(z[2])*dz[56]-L2*sin(z[2])*dz[59]+L2*cos(z[2])*cos(z[3])*dz[57]+L2*cos(z[2])*cos(z[3])*dz[60]+L2*cos(z[2])*sin(z[3])*dz[55]+L2*cos(z[2])*sin(z[3])*dz[58]
            res[33] = dz[33]-z[42]+z[32]*(L2*cos(z[2])*cos(z[3])*dz[25]+L2*cos(z[2])*cos(z[3])*dz[28]-L2*cos(z[2])*sin(z[3])*dz[27]-L2*cos(z[2])*sin(z[3])*dz[30])-z[33]*(L2*cos(z[3])*sin(z[2])*dz[27]+L2*cos(z[3])*sin(z[2])*dz[30]+L2*sin(z[2])*sin(z[3])*dz[25]+L2*sin(z[2])*sin(z[3])*dz[28])+L2*cos(z[3])*sin(z[2])*dz[55]+L2*cos(z[3])*sin(z[2])*dz[58]-L2*sin(z[2])*sin(z[3])*dz[57]-L2*sin(z[2])*sin(z[3])*dz[60]
            res[34] = dz[34]-z[43]-z[34]*((L1*cos(z[4])*dz[26])*0.5-L1*sin(z[4])*dz[27]+(sqrt(3)*L1*cos(z[4])*dz[25])*0.5)-L1*cos(z[4])*dz[57]-(L1*sin(z[4])*dz[56])*0.5-(sqrt(3)*L1*sin(z[4])*dz[55])*0.5
            res[35] = dz[35]-z[44]+dz[55]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)-dz[56]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)+z[36]*((L2*cos(z[5])*cos(z[6])*dz[25])*0.5+L2*cos(z[5])*sin(z[6])*dz[27]-(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[26])*0.5)-z[35]*(dz[25]*((sqrt(3)*L2*cos(z[5]))*0.5+(L2*sin(z[5])*sin(z[6]))*0.5)+dz[26]*((L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5)-L2*cos(z[6])*sin(z[5])*dz[27])-L2*cos(z[5])*cos(z[6])*dz[57]
            res[36] = dz[36]-z[45]+z[35]*((L2*cos(z[5])*cos(z[6])*dz[25])*0.5+L2*cos(z[5])*sin(z[6])*dz[27]-(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[26])*0.5)+z[36]*(L2*cos(z[6])*sin(z[5])*dz[27]-(L2*sin(z[5])*sin(z[6])*dz[25])*0.5+(sqrt(3)*L2*sin(z[5])*sin(z[6])*dz[26])*0.5)+(L2*cos(z[6])*sin(z[5])*dz[55])*0.5+L2*sin(z[5])*sin(z[6])*dz[57]-(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[56])*0.5
            res[37] = dz[37]-z[46]+z[37]*(L1*sin(z[7])*dz[30]-(L1*cos(z[7])*dz[29])*0.5+(sqrt(3)*L1*cos(z[7])*dz[28])*0.5)-L1*cos(z[7])*dz[60]-(L1*sin(z[7])*dz[59])*0.5+(sqrt(3)*L1*sin(z[7])*dz[58])*0.5
            res[38] = dz[38]-z[47]+dz[58]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)-dz[59]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)+z[39]*((L2*cos(z[8])*cos(z[9])*dz[28])*0.5+L2*cos(z[8])*sin(z[9])*dz[30]+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[29])*0.5)+z[38]*(dz[28]*((sqrt(3)*L2*cos(z[8]))*0.5-(L2*sin(z[8])*sin(z[9]))*0.5)-dz[29]*((L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5)+L2*cos(z[9])*sin(z[8])*dz[30])-L2*cos(z[8])*cos(z[9])*dz[60]
            res[39] = dz[39]-z[48]+z[38]*((L2*cos(z[8])*cos(z[9])*dz[28])*0.5+L2*cos(z[8])*sin(z[9])*dz[30]+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[29])*0.5)-z[39]*((L2*sin(z[8])*sin(z[9])*dz[28])*0.5-L2*cos(z[9])*sin(z[8])*dz[30]+(sqrt(3)*L2*sin(z[8])*sin(z[9])*dz[29])*0.5)+(L2*cos(z[9])*sin(z[8])*dz[58])*0.5+L2*sin(z[8])*sin(z[9])*dz[60]+(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[59])*0.5
            res[40] = z[41]*(2*L1*z[11]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[12]*(L2*M3+LC2*M2))+dz[40]*(J1+L1^2*(M2+M3)+LC1^2*M1)-z[42]*(2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]*(L2*M3+LC2*M2))-z[33]*(L1*cos(z[1])*cos(z[2])*sin(z[3])*dz[11]*(L2*M3+LC2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*z[12]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]^2*(L2*M3+LC2*M2)+L1*cos(z[1])*cos(z[3])*sin(z[2])*dz[12]*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[2])*cos(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+z[31]*(0.0*sin(z[1])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[1])*dz[20]+L1*cos(z[1])*dz[23]+L1*sin(z[1])*dz[21]+L1*sin(z[1])*dz[24]+L1*dz[11]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2]))+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)+L1*sin(z[1])*sin(z[2])*sin(z[3])*dz[12]*(L2*M3+LC2*M2)+2*L1*cos(z[2])*sin(z[1])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+γ*z[40]-z[32]*(L1*z[11]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))-L1*dz[11]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))+L1*cos(z[1])*cos(z[2])*cos(z[3])*z[12]^2*(L2*M3+LC2*M2)+L1*cos(z[1])*cos(z[2])*sin(z[3])*dz[12]*(L2*M3+LC2*M2)-2*L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))-L1*cos(z[1])*dz[51]-L1*cos(z[1])*dz[54]+L1*sin(z[1])*dz[50]+L1*sin(z[1])*dz[53]+L1*dz[41]*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))-L1*cos(z[1])*sin(z[2])*sin(z[3])*dz[42]*(L2*M3+LC2*M2)
            res[41] = z[31]*(L1*dz[10]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))-L1*z[10]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3])))+z[32]*(sin(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)+L2*cos(z[2])*dz[20]+L2*cos(z[2])*dz[23]+L2*cos(z[3])*sin(z[2])*dz[21]+L2*cos(z[3])*sin(z[2])*dz[24]+L2*sin(z[2])*sin(z[3])*dz[19]+L2*sin(z[2])*sin(z[3])*dz[22]+L1*dz[10]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))+0.0*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2])))+dz[41]*(J2+L2^2*M3+LC2^2*M2)+γ*z[41]+z[33]*(L2*cos(z[2])*sin(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[22]-L2*cos(z[2])*cos(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[24]+0.0*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[2])*sin(z[3])*dz[10]*(L2*M3+LC2*M2))+L2*sin(z[2])*dz[50]+L2*sin(z[2])*dz[53]-L2*cos(z[2])*cos(z[3])*dz[51]-L2*cos(z[2])*cos(z[3])*dz[54]-L2*cos(z[2])*sin(z[3])*dz[49]-L2*cos(z[2])*sin(z[3])*dz[52]+L1*dz[40]*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))-2*cos(z[2])*sin(z[2])*z[42]*z[12]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[40]*z[10]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))
            res[42] = z[33]*(L2*cos(z[3])*sin(z[2])*dz[21]+L2*cos(z[3])*sin(z[2])*dz[24]+L2*sin(z[2])*sin(z[3])*dz[19]+L2*sin(z[2])*sin(z[3])*dz[22]+0.0*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[10]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[3])*sin(z[2])*dz[10]*(L2*M3+LC2*M2))+z[32]*(L2*cos(z[2])*sin(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[22]-L2*cos(z[2])*cos(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[24]+2*cos(z[2])*sin(z[2])*dz[12]*(J2+L2^2*M3+LC2^2*M2)+2*cos(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+0.0*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[2])*sin(z[3])*dz[10]*(L2*M3+LC2*M2))+z[42]*(γ+sin(2*z[2])*z[11]*(J2+L2^2*M3+LC2^2*M2))+z[31]*(L1*cos(z[1])*sin(z[2])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)+L1*sin(z[1])*sin(z[2])*sin(z[3])*dz[10]*(L2*M3+LC2*M2))+sin(z[2])^2*dz[42]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[3])*sin(z[2])*dz[49]-L2*cos(z[3])*sin(z[2])*dz[52]+L2*sin(z[2])*sin(z[3])*dz[51]+L2*sin(z[2])*sin(z[3])*dz[54]+sin(2*z[2])*z[41]*z[12]*(J2+L2^2*M3+LC2^2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*dz[40]*(L2*M3+LC2*M2)+2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[40]*z[10]*(L2*M3+LC2*M2)
            res[43] = z[44]*(2*L1*z[14]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[15]*(L2*M3+LC2*M2))+dz[43]*(J1+L1^2*(M2+M3)+LC1^2*M1)-z[45]*(2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]*(L2*M3+LC2*M2))-z[36]*(L1*cos(z[4])*cos(z[5])*sin(z[6])*dz[14]*(L2*M3+LC2*M2)-L1*cos(z[4])*sin(z[5])*sin(z[6])*z[15]^2*(L2*M3+LC2*M2)-L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*cos(z[6])*sin(z[5])*dz[15]*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[5])*cos(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+γ*z[43]-z[35]*(L1*z[14]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))-L1*dz[14]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))+L1*cos(z[4])*cos(z[5])*cos(z[6])*z[15]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*cos(z[5])*sin(z[6])*dz[15]*(L2*M3+LC2*M2)-2*L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+z[34]*(0.0*sin(z[4])*(L1*(M2+M3)+LC1*M1)+(L1*cos(z[4])*dz[20])*0.5-L1*sin(z[4])*dz[21]+(sqrt(3)*L1*cos(z[4])*dz[19])*0.5+L1*dz[14]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5]))+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)+L1*sin(z[4])*sin(z[5])*sin(z[6])*dz[15]*(L2*M3+LC2*M2)+2*L1*cos(z[5])*sin(z[4])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+L1*cos(z[4])*dz[51]+(L1*sin(z[4])*dz[50])*0.5+(sqrt(3)*L1*sin(z[4])*dz[49])*0.5+L1*dz[44]*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))-L1*cos(z[4])*sin(z[5])*sin(z[6])*dz[45]*(L2*M3+LC2*M2)
            res[44] = z[34]*(L1*dz[13]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))-L1*z[13]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6])))-z[36]*((L2*cos(z[5])*cos(z[6])*dz[19])*0.5+L2*cos(z[5])*sin(z[6])*dz[21]-0.0*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)-(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[20])*0.5-L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*cos(z[5])*sin(z[6])*dz[13]*(L2*M3+LC2*M2))-dz[49]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)+dz[50]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)+dz[44]*(J2+L2^2*M3+LC2^2*M2)+γ*z[44]+z[35]*(dz[19]*((sqrt(3)*L2*cos(z[5]))*0.5+(L2*sin(z[5])*sin(z[6]))*0.5)+dz[20]*((L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5)-cos(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)+sin(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[6])*sin(z[5])*dz[21]+L1*dz[13]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))+0.0*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5])))+L2*cos(z[5])*cos(z[6])*dz[51]+L1*dz[43]*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))-2*cos(z[5])*sin(z[5])*z[45]*z[15]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[43]*z[13]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))
            res[45] = z[45]*(γ+sin(2*z[5])*z[14]*(J2+L2^2*M3+LC2^2*M2))-z[36]*(L2*cos(z[6])*sin(z[5])*dz[21]-(L2*sin(z[5])*sin(z[6])*dz[19])*0.5-0.0*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+(sqrt(3)*L2*sin(z[5])*sin(z[6])*dz[20])*0.5-L1*cos(z[6])*sin(z[4])*sin(z[5])*z[13]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*cos(z[6])*sin(z[5])*dz[13]*(L2*M3+LC2*M2))+z[35]*(2*cos(z[5])*sin(z[5])*dz[15]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[5])*sin(z[6])*dz[21]-(L2*cos(z[5])*cos(z[6])*dz[19])*0.5+2*cos(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)+0.0*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[20])*0.5+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[5])*sin(z[6])*dz[13]*(L2*M3+LC2*M2))+z[34]*(L1*cos(z[4])*sin(z[5])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)+L1*sin(z[4])*sin(z[5])*sin(z[6])*dz[13]*(L2*M3+LC2*M2))+sin(z[5])^2*dz[45]*(J2+L2^2*M3+LC2^2*M2)-(L2*cos(z[6])*sin(z[5])*dz[49])*0.5-L2*sin(z[5])*sin(z[6])*dz[51]+sin(2*z[5])*z[44]*z[15]*(J2+L2^2*M3+LC2^2*M2)+(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[50])*0.5-L1*cos(z[4])*sin(z[5])*sin(z[6])*dz[43]*(L2*M3+LC2*M2)+2*L1*sin(z[4])*sin(z[5])*sin(z[6])*z[43]*z[13]*(L2*M3+LC2*M2)
            res[46] = z[47]*(2*L1*z[17]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[18]*(L2*M3+LC2*M2))+dz[46]*(J1+L1^2*(M2+M3)+LC1^2*M1)-z[48]*(2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]*(L2*M3+LC2*M2))-z[39]*(L1*cos(z[7])*cos(z[8])*sin(z[9])*dz[17]*(L2*M3+LC2*M2)-L1*cos(z[7])*sin(z[8])*sin(z[9])*z[18]^2*(L2*M3+LC2*M2)-L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*cos(z[9])*sin(z[8])*dz[18]*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[8])*cos(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+γ*z[46]-z[38]*(L1*z[17]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))-L1*dz[17]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))+L1*cos(z[7])*cos(z[8])*cos(z[9])*z[18]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*cos(z[8])*sin(z[9])*dz[18]*(L2*M3+LC2*M2)-2*L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+z[37]*(0.0*sin(z[7])*(L1*(M2+M3)+LC1*M1)+(L1*cos(z[7])*dz[23])*0.5-L1*sin(z[7])*dz[24]-(sqrt(3)*L1*cos(z[7])*dz[22])*0.5+L1*dz[17]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8]))+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)+L1*sin(z[7])*sin(z[8])*sin(z[9])*dz[18]*(L2*M3+LC2*M2)+2*L1*cos(z[8])*sin(z[7])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+L1*cos(z[7])*dz[54]+(L1*sin(z[7])*dz[53])*0.5-(sqrt(3)*L1*sin(z[7])*dz[52])*0.5+L1*dz[47]*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))-L1*cos(z[7])*sin(z[8])*sin(z[9])*dz[48]*(L2*M3+LC2*M2)
            res[47] = z[37]*(L1*dz[16]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))-L1*z[16]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9])))-z[39]*((L2*cos(z[8])*cos(z[9])*dz[22])*0.5+L2*cos(z[8])*sin(z[9])*dz[24]-0.0*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[23])*0.5-L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*cos(z[8])*sin(z[9])*dz[16]*(L2*M3+LC2*M2))-dz[52]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)+dz[53]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)+dz[47]*(J2+L2^2*M3+LC2^2*M2)+γ*z[47]+z[38]*(dz[23]*((L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5)-dz[22]*((sqrt(3)*L2*cos(z[8]))*0.5-(L2*sin(z[8])*sin(z[9]))*0.5)-cos(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)+sin(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[9])*sin(z[8])*dz[24]+L1*dz[16]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))+0.0*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8])))+L2*cos(z[8])*cos(z[9])*dz[54]+L1*dz[46]*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))-2*cos(z[8])*sin(z[8])*z[48]*z[18]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[46]*z[16]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))
            res[48] = z[39]*((L2*sin(z[8])*sin(z[9])*dz[22])*0.5-L2*cos(z[9])*sin(z[8])*dz[24]+0.0*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+(sqrt(3)*L2*sin(z[8])*sin(z[9])*dz[23])*0.5+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[16]^2*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[9])*sin(z[8])*dz[16]*(L2*M3+LC2*M2))+z[48]*(γ+sin(2*z[8])*z[17]*(J2+L2^2*M3+LC2^2*M2))-z[38]*((L2*cos(z[8])*cos(z[9])*dz[22])*0.5+L2*cos(z[8])*sin(z[9])*dz[24]-2*cos(z[8])*sin(z[8])*dz[18]*(J2+L2^2*M3+LC2^2*M2)-2*cos(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)-0.0*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[23])*0.5-L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*cos(z[8])*sin(z[9])*dz[16]*(L2*M3+LC2*M2))+z[37]*(L1*cos(z[7])*sin(z[8])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)+L1*sin(z[7])*sin(z[8])*sin(z[9])*dz[16]*(L2*M3+LC2*M2))+sin(z[8])^2*dz[48]*(J2+L2^2*M3+LC2^2*M2)-(L2*cos(z[9])*sin(z[8])*dz[52])*0.5-L2*sin(z[8])*sin(z[9])*dz[54]+sin(2*z[8])*z[47]*z[18]*(J2+L2^2*M3+LC2^2*M2)-(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[53])*0.5-L1*cos(z[7])*sin(z[8])*sin(z[9])*dz[46]*(L2*M3+LC2*M2)+2*L1*sin(z[7])*sin(z[8])*sin(z[9])*z[46]*z[16]*(L2*M3+LC2*M2)
            res[49] = z[35]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)+L2*cos(z[2])*sin(z[3])*z[32]+L2*cos(z[3])*sin(z[2])*z[33]+(L2*cos(z[6])*sin(z[5])*z[36])*0.5-(sqrt(3)*L1*sin(z[4])*z[34])*0.5
            res[50] = -z[35]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-L1*sin(z[1])*z[31]-L2*sin(z[2])*z[32]-(L1*sin(z[4])*z[34])*0.5-(sqrt(3)*L2*cos(z[6])*sin(z[5])*z[36])*0.5
            res[51] = L1*cos(z[1])*z[31]-L1*cos(z[4])*z[34]+L2*cos(z[2])*cos(z[3])*z[32]-L2*cos(z[5])*cos(z[6])*z[35]-L2*sin(z[2])*sin(z[3])*z[33]+L2*sin(z[5])*sin(z[6])*z[36]
            res[52] = z[38]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)+L2*cos(z[2])*sin(z[3])*z[32]+L2*cos(z[3])*sin(z[2])*z[33]+(L2*cos(z[9])*sin(z[8])*z[39])*0.5+(sqrt(3)*L1*sin(z[7])*z[37])*0.5
            res[53] = (sqrt(3)*L2*cos(z[9])*sin(z[8])*z[39])*0.5-L1*sin(z[1])*z[31]-L2*sin(z[2])*z[32]-(L1*sin(z[7])*z[37])*0.5-z[38]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)
            res[54] = L1*cos(z[1])*z[31]-L1*cos(z[7])*z[37]+L2*cos(z[2])*cos(z[3])*z[32]-L2*cos(z[8])*cos(z[9])*z[38]-L2*sin(z[2])*sin(z[3])*z[33]+L2*sin(z[8])*sin(z[9])*z[39]
            res[55] = z[32]*(L2*cos(z[2])*cos(z[3])*z[12]-L2*sin(z[2])*sin(z[3])*z[11])+z[33]*(L2*cos(z[2])*cos(z[3])*z[11]-L2*sin(z[2])*sin(z[3])*z[12])+z[36]*((L2*cos(z[5])*cos(z[6])*z[14])*0.5-(L2*sin(z[5])*sin(z[6])*z[15])*0.5)-z[35]*((sqrt(3)*L2*cos(z[5])*z[14])*0.5-(L2*cos(z[5])*cos(z[6])*z[15])*0.5+(L2*sin(z[5])*sin(z[6])*z[14])*0.5)+z[44]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)+L2*cos(z[2])*sin(z[3])*z[41]+L2*cos(z[3])*sin(z[2])*z[42]+(L2*cos(z[6])*sin(z[5])*z[45])*0.5-(sqrt(3)*L1*sin(z[4])*z[43])*0.5-(sqrt(3)*L1*cos(z[4])*z[34]*z[13])*0.5
            res[56] = -z[44]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-z[35]*((L2*cos(z[5])*z[14])*0.5+(sqrt(3)*L2*cos(z[5])*cos(z[6])*z[15])*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6])*z[14])*0.5)-z[36]*((sqrt(3)*L2*cos(z[5])*cos(z[6])*z[14])*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6])*z[15])*0.5)-L1*sin(z[1])*z[40]-L2*sin(z[2])*z[41]-(L1*sin(z[4])*z[43])*0.5-L1*cos(z[1])*z[31]*z[10]-L2*cos(z[2])*z[32]*z[11]-(L1*cos(z[4])*z[34]*z[13])*0.5-(sqrt(3)*L2*cos(z[6])*sin(z[5])*z[45])*0.5
            res[57] = z[35]*(L2*cos(z[6])*sin(z[5])*z[14]+L2*cos(z[5])*sin(z[6])*z[15])-z[33]*(L2*cos(z[2])*sin(z[3])*z[11]+L2*cos(z[3])*sin(z[2])*z[12])-z[32]*(L2*cos(z[3])*sin(z[2])*z[11]+L2*cos(z[2])*sin(z[3])*z[12])+z[36]*(L2*cos(z[5])*sin(z[6])*z[14]+L2*cos(z[6])*sin(z[5])*z[15])+L1*cos(z[1])*z[40]-L1*cos(z[4])*z[43]+L2*cos(z[2])*cos(z[3])*z[41]-L2*cos(z[5])*cos(z[6])*z[44]-L2*sin(z[2])*sin(z[3])*z[42]+L2*sin(z[5])*sin(z[6])*z[45]-L1*sin(z[1])*z[31]*z[10]+L1*sin(z[4])*z[34]*z[13]
            res[58] = z[32]*(L2*cos(z[2])*cos(z[3])*z[12]-L2*sin(z[2])*sin(z[3])*z[11])+z[33]*(L2*cos(z[2])*cos(z[3])*z[11]-L2*sin(z[2])*sin(z[3])*z[12])+z[39]*((L2*cos(z[8])*cos(z[9])*z[17])*0.5-(L2*sin(z[8])*sin(z[9])*z[18])*0.5)+z[38]*((L2*cos(z[8])*cos(z[9])*z[18])*0.5+(sqrt(3)*L2*cos(z[8])*z[17])*0.5-(L2*sin(z[8])*sin(z[9])*z[17])*0.5)+z[47]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)+L2*cos(z[2])*sin(z[3])*z[41]+L2*cos(z[3])*sin(z[2])*z[42]+(L2*cos(z[9])*sin(z[8])*z[48])*0.5+(sqrt(3)*L1*sin(z[7])*z[46])*0.5+(sqrt(3)*L1*cos(z[7])*z[37]*z[16])*0.5
            res[59] = z[39]*((sqrt(3)*L2*cos(z[8])*cos(z[9])*z[17])*0.5-(sqrt(3)*L2*sin(z[8])*sin(z[9])*z[18])*0.5)-z[38]*((L2*cos(z[8])*z[17])*0.5-(sqrt(3)*L2*cos(z[8])*cos(z[9])*z[18])*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9])*z[17])*0.5)-z[47]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)-L1*sin(z[1])*z[40]-L2*sin(z[2])*z[41]-(L1*sin(z[7])*z[46])*0.5-L1*cos(z[1])*z[31]*z[10]-L2*cos(z[2])*z[32]*z[11]-(L1*cos(z[7])*z[37]*z[16])*0.5+(sqrt(3)*L2*cos(z[9])*sin(z[8])*z[48])*0.5
            res[60] = z[38]*(L2*cos(z[9])*sin(z[8])*z[17]+L2*cos(z[8])*sin(z[9])*z[18])-z[33]*(L2*cos(z[2])*sin(z[3])*z[11]+L2*cos(z[3])*sin(z[2])*z[12])-z[32]*(L2*cos(z[3])*sin(z[2])*z[11]+L2*cos(z[2])*sin(z[3])*z[12])+z[39]*(L2*cos(z[8])*sin(z[9])*z[17]+L2*cos(z[9])*sin(z[8])*z[18])+L1*cos(z[1])*z[40]-L1*cos(z[7])*z[46]+L2*cos(z[2])*cos(z[3])*z[41]-L2*cos(z[8])*cos(z[9])*z[47]-L2*sin(z[2])*sin(z[3])*z[42]+L2*sin(z[8])*sin(z[9])*z[48]-L1*sin(z[1])*z[31]*z[10]+L1*sin(z[7])*z[37]*z[16]
            # Parameter-specific part for L1
            res[31] += cos(z[1])*dz[27]+cos(z[1])*dz[30]-sin(z[1])*dz[26]-sin(z[1])*dz[29]
            res[34] += -cos(z[4])*dz[27]-(sin(z[4])*dz[26])*0.5-(sqrt(3)*sin(z[4])*dz[25])*0.5
            res[37] += (sqrt(3)*sin(z[7])*dz[28])*0.5-(sin(z[7])*dz[29])*0.5-cos(z[7])*dz[30]
            res[40] += sin(z[1])*dz[20]-cos(z[1])*dz[24]-cos(z[1])*dz[21]+sin(z[1])*dz[23]-0.0*cos(z[1])*(M2+M3)+dz[11]*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))+2*L1*dz[10]*(M2+M3)+z[11]^2*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-cos(z[1])*sin(z[2])*sin(z[3])*dz[12]*(L2*M3+LC2*M2)-cos(z[1])*cos(z[3])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)-2*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2)
            res[41] += dz[10]*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))+z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))
            res[42] += sin(z[1])*sin(z[2])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)-cos(z[1])*sin(z[2])*sin(z[3])*dz[10]*(L2*M3+LC2*M2)
            res[43] += cos(z[4])*dz[21]+(sin(z[4])*dz[20])*0.5-0.0*cos(z[4])*(M2+M3)+dz[14]*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))+2*L1*dz[13]*(M2+M3)+z[14]^2*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))+(sqrt(3)*sin(z[4])*dz[19])*0.5-cos(z[4])*sin(z[5])*sin(z[6])*dz[15]*(L2*M3+LC2*M2)-cos(z[4])*cos(z[6])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)-2*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2)
            res[44] += dz[13]*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))+z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))
            res[45] += sin(z[4])*sin(z[5])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)-cos(z[4])*sin(z[5])*sin(z[6])*dz[13]*(L2*M3+LC2*M2)
            res[46] += cos(z[7])*dz[24]+(sin(z[7])*dz[23])*0.5-0.0*cos(z[7])*(M2+M3)+dz[17]*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))+2*L1*dz[16]*(M2+M3)+z[17]^2*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-(sqrt(3)*sin(z[7])*dz[22])*0.5-cos(z[7])*sin(z[8])*sin(z[9])*dz[18]*(L2*M3+LC2*M2)-cos(z[7])*cos(z[9])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)-2*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2)
            res[47] += dz[16]*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))+z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))
            res[48] += sin(z[7])*sin(z[8])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)-cos(z[7])*sin(z[8])*sin(z[9])*dz[16]*(L2*M3+LC2*M2)
            res[49] += (sqrt(3)*cos(z[4]))*0.5
            res[50] += cos(z[1])+cos(z[4])*0.5
            res[51] += sin(z[1])-sin(z[4])
            res[52] += -(sqrt(3)*cos(z[7]))*0.5
            res[53] += cos(z[1])+cos(z[7])*0.5
            res[54] += sin(z[1])-sin(z[7])
            res[55] += -(sqrt(3)*sin(z[4])*z[13])*0.5
            res[56] += -sin(z[1])*z[10]-(sin(z[4])*z[13])*0.5
            res[57] += cos(z[1])*z[10]-cos(z[4])*z[13]
            res[58] += (sqrt(3)*sin(z[7])*z[16])*0.5
            res[59] += -sin(z[1])*z[10]-(sin(z[7])*z[16])*0.5
            res[60] += cos(z[1])*z[10]-cos(z[7])*z[16]
            # Sensitivity equations wrt to M1
            res[61] += dz[61]-z[70]-z[61]*(L1*cos(z[1])*dz[26]+L1*cos(z[1])*dz[29]+L1*sin(z[1])*dz[27]+L1*sin(z[1])*dz[30])+L1*cos(z[1])*dz[87]+L1*cos(z[1])*dz[90]-L1*sin(z[1])*dz[86]-L1*sin(z[1])*dz[89]
            res[62] += dz[62]-z[71]+z[63]*(L2*cos(z[2])*cos(z[3])*dz[25]+L2*cos(z[2])*cos(z[3])*dz[28]-L2*cos(z[2])*sin(z[3])*dz[27]-L2*cos(z[2])*sin(z[3])*dz[30])-z[62]*(L2*cos(z[2])*dz[26]+L2*cos(z[2])*dz[29]+L2*cos(z[3])*sin(z[2])*dz[27]+L2*cos(z[3])*sin(z[2])*dz[30]+L2*sin(z[2])*sin(z[3])*dz[25]+L2*sin(z[2])*sin(z[3])*dz[28])-L2*sin(z[2])*dz[86]-L2*sin(z[2])*dz[89]+L2*cos(z[2])*cos(z[3])*dz[87]+L2*cos(z[2])*cos(z[3])*dz[90]+L2*cos(z[2])*sin(z[3])*dz[85]+L2*cos(z[2])*sin(z[3])*dz[88]
            res[63] += dz[63]-z[72]+z[62]*(L2*cos(z[2])*cos(z[3])*dz[25]+L2*cos(z[2])*cos(z[3])*dz[28]-L2*cos(z[2])*sin(z[3])*dz[27]-L2*cos(z[2])*sin(z[3])*dz[30])-z[63]*(L2*cos(z[3])*sin(z[2])*dz[27]+L2*cos(z[3])*sin(z[2])*dz[30]+L2*sin(z[2])*sin(z[3])*dz[25]+L2*sin(z[2])*sin(z[3])*dz[28])+L2*cos(z[3])*sin(z[2])*dz[85]+L2*cos(z[3])*sin(z[2])*dz[88]-L2*sin(z[2])*sin(z[3])*dz[87]-L2*sin(z[2])*sin(z[3])*dz[90]
            res[64] += dz[64]-z[73]-z[64]*((L1*cos(z[4])*dz[26])*0.5-L1*sin(z[4])*dz[27]+(sqrt(3)*L1*cos(z[4])*dz[25])*0.5)-L1*cos(z[4])*dz[87]-(L1*sin(z[4])*dz[86])*0.5-(sqrt(3)*L1*sin(z[4])*dz[85])*0.5
            res[65] += dz[65]-z[74]+dz[85]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)-dz[86]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)+z[66]*((L2*cos(z[5])*cos(z[6])*dz[25])*0.5+L2*cos(z[5])*sin(z[6])*dz[27]-(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[26])*0.5)-z[65]*(dz[25]*((sqrt(3)*L2*cos(z[5]))*0.5+(L2*sin(z[5])*sin(z[6]))*0.5)+dz[26]*((L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5)-L2*cos(z[6])*sin(z[5])*dz[27])-L2*cos(z[5])*cos(z[6])*dz[87]
            res[66] += dz[66]-z[75]+z[65]*((L2*cos(z[5])*cos(z[6])*dz[25])*0.5+L2*cos(z[5])*sin(z[6])*dz[27]-(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[26])*0.5)+z[66]*(L2*cos(z[6])*sin(z[5])*dz[27]-(L2*sin(z[5])*sin(z[6])*dz[25])*0.5+(sqrt(3)*L2*sin(z[5])*sin(z[6])*dz[26])*0.5)+(L2*cos(z[6])*sin(z[5])*dz[85])*0.5+L2*sin(z[5])*sin(z[6])*dz[87]-(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[86])*0.5
            res[67] += dz[67]-z[76]+z[67]*(L1*sin(z[7])*dz[30]-(L1*cos(z[7])*dz[29])*0.5+(sqrt(3)*L1*cos(z[7])*dz[28])*0.5)-L1*cos(z[7])*dz[90]-(L1*sin(z[7])*dz[89])*0.5+(sqrt(3)*L1*sin(z[7])*dz[88])*0.5
            res[68] += dz[68]-z[77]+dz[88]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)-dz[89]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)+z[69]*((L2*cos(z[8])*cos(z[9])*dz[28])*0.5+L2*cos(z[8])*sin(z[9])*dz[30]+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[29])*0.5)+z[68]*(dz[28]*((sqrt(3)*L2*cos(z[8]))*0.5-(L2*sin(z[8])*sin(z[9]))*0.5)-dz[29]*((L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5)+L2*cos(z[9])*sin(z[8])*dz[30])-L2*cos(z[8])*cos(z[9])*dz[90]
            res[69] += dz[69]-z[78]+z[68]*((L2*cos(z[8])*cos(z[9])*dz[28])*0.5+L2*cos(z[8])*sin(z[9])*dz[30]+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[29])*0.5)-z[69]*((L2*sin(z[8])*sin(z[9])*dz[28])*0.5-L2*cos(z[9])*sin(z[8])*dz[30]+(sqrt(3)*L2*sin(z[8])*sin(z[9])*dz[29])*0.5)+(L2*cos(z[9])*sin(z[8])*dz[88])*0.5+L2*sin(z[8])*sin(z[9])*dz[90]+(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[89])*0.5
            res[70] += z[71]*(2*L1*z[11]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[12]*(L2*M3+LC2*M2))+dz[70]*(J1+L1^2*(M2+M3)+LC1^2*M1)-z[72]*(2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]*(L2*M3+LC2*M2))-z[63]*(L1*cos(z[1])*cos(z[2])*sin(z[3])*dz[11]*(L2*M3+LC2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*z[12]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]^2*(L2*M3+LC2*M2)+L1*cos(z[1])*cos(z[3])*sin(z[2])*dz[12]*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[2])*cos(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+z[61]*(0.0*sin(z[1])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[1])*dz[20]+L1*cos(z[1])*dz[23]+L1*sin(z[1])*dz[21]+L1*sin(z[1])*dz[24]+L1*dz[11]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2]))+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)+L1*sin(z[1])*sin(z[2])*sin(z[3])*dz[12]*(L2*M3+LC2*M2)+2*L1*cos(z[2])*sin(z[1])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+γ*z[70]-z[62]*(L1*z[11]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))-L1*dz[11]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))+L1*cos(z[1])*cos(z[2])*cos(z[3])*z[12]^2*(L2*M3+LC2*M2)+L1*cos(z[1])*cos(z[2])*sin(z[3])*dz[12]*(L2*M3+LC2*M2)-2*L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))-L1*cos(z[1])*dz[81]-L1*cos(z[1])*dz[84]+L1*sin(z[1])*dz[80]+L1*sin(z[1])*dz[83]+L1*dz[71]*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))-L1*cos(z[1])*sin(z[2])*sin(z[3])*dz[72]*(L2*M3+LC2*M2)
            res[71] += z[61]*(L1*dz[10]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))-L1*z[10]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3])))+z[62]*(sin(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)+L2*cos(z[2])*dz[20]+L2*cos(z[2])*dz[23]+L2*cos(z[3])*sin(z[2])*dz[21]+L2*cos(z[3])*sin(z[2])*dz[24]+L2*sin(z[2])*sin(z[3])*dz[19]+L2*sin(z[2])*sin(z[3])*dz[22]+L1*dz[10]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))+0.0*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2])))+dz[71]*(J2+L2^2*M3+LC2^2*M2)+γ*z[71]+z[63]*(L2*cos(z[2])*sin(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[22]-L2*cos(z[2])*cos(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[24]+0.0*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[2])*sin(z[3])*dz[10]*(L2*M3+LC2*M2))+L2*sin(z[2])*dz[80]+L2*sin(z[2])*dz[83]-L2*cos(z[2])*cos(z[3])*dz[81]-L2*cos(z[2])*cos(z[3])*dz[84]-L2*cos(z[2])*sin(z[3])*dz[79]-L2*cos(z[2])*sin(z[3])*dz[82]+L1*dz[70]*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))-2*cos(z[2])*sin(z[2])*z[72]*z[12]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[70]*z[10]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))
            res[72] += z[63]*(L2*cos(z[3])*sin(z[2])*dz[21]+L2*cos(z[3])*sin(z[2])*dz[24]+L2*sin(z[2])*sin(z[3])*dz[19]+L2*sin(z[2])*sin(z[3])*dz[22]+0.0*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[10]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[3])*sin(z[2])*dz[10]*(L2*M3+LC2*M2))+z[62]*(L2*cos(z[2])*sin(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[22]-L2*cos(z[2])*cos(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[24]+2*cos(z[2])*sin(z[2])*dz[12]*(J2+L2^2*M3+LC2^2*M2)+2*cos(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+0.0*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[2])*sin(z[3])*dz[10]*(L2*M3+LC2*M2))+z[72]*(γ+sin(2*z[2])*z[11]*(J2+L2^2*M3+LC2^2*M2))+z[61]*(L1*cos(z[1])*sin(z[2])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)+L1*sin(z[1])*sin(z[2])*sin(z[3])*dz[10]*(L2*M3+LC2*M2))+sin(z[2])^2*dz[72]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[3])*sin(z[2])*dz[79]-L2*cos(z[3])*sin(z[2])*dz[82]+L2*sin(z[2])*sin(z[3])*dz[81]+L2*sin(z[2])*sin(z[3])*dz[84]+sin(2*z[2])*z[71]*z[12]*(J2+L2^2*M3+LC2^2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*dz[70]*(L2*M3+LC2*M2)+2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[70]*z[10]*(L2*M3+LC2*M2)
            res[73] += z[74]*(2*L1*z[14]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[15]*(L2*M3+LC2*M2))+dz[73]*(J1+L1^2*(M2+M3)+LC1^2*M1)-z[75]*(2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]*(L2*M3+LC2*M2))-z[66]*(L1*cos(z[4])*cos(z[5])*sin(z[6])*dz[14]*(L2*M3+LC2*M2)-L1*cos(z[4])*sin(z[5])*sin(z[6])*z[15]^2*(L2*M3+LC2*M2)-L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*cos(z[6])*sin(z[5])*dz[15]*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[5])*cos(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+γ*z[73]-z[65]*(L1*z[14]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))-L1*dz[14]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))+L1*cos(z[4])*cos(z[5])*cos(z[6])*z[15]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*cos(z[5])*sin(z[6])*dz[15]*(L2*M3+LC2*M2)-2*L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+z[64]*(0.0*sin(z[4])*(L1*(M2+M3)+LC1*M1)+(L1*cos(z[4])*dz[20])*0.5-L1*sin(z[4])*dz[21]+(sqrt(3)*L1*cos(z[4])*dz[19])*0.5+L1*dz[14]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5]))+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)+L1*sin(z[4])*sin(z[5])*sin(z[6])*dz[15]*(L2*M3+LC2*M2)+2*L1*cos(z[5])*sin(z[4])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+L1*cos(z[4])*dz[81]+(L1*sin(z[4])*dz[80])*0.5+(sqrt(3)*L1*sin(z[4])*dz[79])*0.5+L1*dz[74]*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))-L1*cos(z[4])*sin(z[5])*sin(z[6])*dz[75]*(L2*M3+LC2*M2)
            res[74] += z[64]*(L1*dz[13]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))-L1*z[13]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6])))-z[66]*((L2*cos(z[5])*cos(z[6])*dz[19])*0.5+L2*cos(z[5])*sin(z[6])*dz[21]-0.0*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)-(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[20])*0.5-L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*cos(z[5])*sin(z[6])*dz[13]*(L2*M3+LC2*M2))-dz[79]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)+dz[80]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)+dz[74]*(J2+L2^2*M3+LC2^2*M2)+γ*z[74]+z[65]*(dz[19]*((sqrt(3)*L2*cos(z[5]))*0.5+(L2*sin(z[5])*sin(z[6]))*0.5)+dz[20]*((L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5)-cos(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)+sin(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[6])*sin(z[5])*dz[21]+L1*dz[13]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))+0.0*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5])))+L2*cos(z[5])*cos(z[6])*dz[81]+L1*dz[73]*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))-2*cos(z[5])*sin(z[5])*z[75]*z[15]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[73]*z[13]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))
            res[75] += z[75]*(γ+sin(2*z[5])*z[14]*(J2+L2^2*M3+LC2^2*M2))-z[66]*(L2*cos(z[6])*sin(z[5])*dz[21]-(L2*sin(z[5])*sin(z[6])*dz[19])*0.5-0.0*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+(sqrt(3)*L2*sin(z[5])*sin(z[6])*dz[20])*0.5-L1*cos(z[6])*sin(z[4])*sin(z[5])*z[13]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*cos(z[6])*sin(z[5])*dz[13]*(L2*M3+LC2*M2))+z[65]*(2*cos(z[5])*sin(z[5])*dz[15]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[5])*sin(z[6])*dz[21]-(L2*cos(z[5])*cos(z[6])*dz[19])*0.5+2*cos(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)+0.0*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[20])*0.5+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[5])*sin(z[6])*dz[13]*(L2*M3+LC2*M2))+z[64]*(L1*cos(z[4])*sin(z[5])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)+L1*sin(z[4])*sin(z[5])*sin(z[6])*dz[13]*(L2*M3+LC2*M2))+sin(z[5])^2*dz[75]*(J2+L2^2*M3+LC2^2*M2)-(L2*cos(z[6])*sin(z[5])*dz[79])*0.5-L2*sin(z[5])*sin(z[6])*dz[81]+sin(2*z[5])*z[74]*z[15]*(J2+L2^2*M3+LC2^2*M2)+(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[80])*0.5-L1*cos(z[4])*sin(z[5])*sin(z[6])*dz[73]*(L2*M3+LC2*M2)+2*L1*sin(z[4])*sin(z[5])*sin(z[6])*z[73]*z[13]*(L2*M3+LC2*M2)
            res[76] += z[77]*(2*L1*z[17]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[18]*(L2*M3+LC2*M2))+dz[76]*(J1+L1^2*(M2+M3)+LC1^2*M1)-z[78]*(2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]*(L2*M3+LC2*M2))-z[69]*(L1*cos(z[7])*cos(z[8])*sin(z[9])*dz[17]*(L2*M3+LC2*M2)-L1*cos(z[7])*sin(z[8])*sin(z[9])*z[18]^2*(L2*M3+LC2*M2)-L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*cos(z[9])*sin(z[8])*dz[18]*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[8])*cos(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+γ*z[76]-z[68]*(L1*z[17]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))-L1*dz[17]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))+L1*cos(z[7])*cos(z[8])*cos(z[9])*z[18]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*cos(z[8])*sin(z[9])*dz[18]*(L2*M3+LC2*M2)-2*L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+z[67]*(0.0*sin(z[7])*(L1*(M2+M3)+LC1*M1)+(L1*cos(z[7])*dz[23])*0.5-L1*sin(z[7])*dz[24]-(sqrt(3)*L1*cos(z[7])*dz[22])*0.5+L1*dz[17]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8]))+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)+L1*sin(z[7])*sin(z[8])*sin(z[9])*dz[18]*(L2*M3+LC2*M2)+2*L1*cos(z[8])*sin(z[7])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+L1*cos(z[7])*dz[84]+(L1*sin(z[7])*dz[83])*0.5-(sqrt(3)*L1*sin(z[7])*dz[82])*0.5+L1*dz[77]*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))-L1*cos(z[7])*sin(z[8])*sin(z[9])*dz[78]*(L2*M3+LC2*M2)
            res[77] += z[67]*(L1*dz[16]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))-L1*z[16]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9])))-z[69]*((L2*cos(z[8])*cos(z[9])*dz[22])*0.5+L2*cos(z[8])*sin(z[9])*dz[24]-0.0*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[23])*0.5-L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*cos(z[8])*sin(z[9])*dz[16]*(L2*M3+LC2*M2))-dz[82]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)+dz[83]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)+dz[77]*(J2+L2^2*M3+LC2^2*M2)+γ*z[77]+z[68]*(dz[23]*((L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5)-dz[22]*((sqrt(3)*L2*cos(z[8]))*0.5-(L2*sin(z[8])*sin(z[9]))*0.5)-cos(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)+sin(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[9])*sin(z[8])*dz[24]+L1*dz[16]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))+0.0*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8])))+L2*cos(z[8])*cos(z[9])*dz[84]+L1*dz[76]*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))-2*cos(z[8])*sin(z[8])*z[78]*z[18]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[76]*z[16]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))
            res[78] += z[69]*((L2*sin(z[8])*sin(z[9])*dz[22])*0.5-L2*cos(z[9])*sin(z[8])*dz[24]+0.0*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+(sqrt(3)*L2*sin(z[8])*sin(z[9])*dz[23])*0.5+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[16]^2*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[9])*sin(z[8])*dz[16]*(L2*M3+LC2*M2))+z[78]*(γ+sin(2*z[8])*z[17]*(J2+L2^2*M3+LC2^2*M2))-z[68]*((L2*cos(z[8])*cos(z[9])*dz[22])*0.5+L2*cos(z[8])*sin(z[9])*dz[24]-2*cos(z[8])*sin(z[8])*dz[18]*(J2+L2^2*M3+LC2^2*M2)-2*cos(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)-0.0*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[23])*0.5-L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*cos(z[8])*sin(z[9])*dz[16]*(L2*M3+LC2*M2))+z[67]*(L1*cos(z[7])*sin(z[8])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)+L1*sin(z[7])*sin(z[8])*sin(z[9])*dz[16]*(L2*M3+LC2*M2))+sin(z[8])^2*dz[78]*(J2+L2^2*M3+LC2^2*M2)-(L2*cos(z[9])*sin(z[8])*dz[82])*0.5-L2*sin(z[8])*sin(z[9])*dz[84]+sin(2*z[8])*z[77]*z[18]*(J2+L2^2*M3+LC2^2*M2)-(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[83])*0.5-L1*cos(z[7])*sin(z[8])*sin(z[9])*dz[76]*(L2*M3+LC2*M2)+2*L1*sin(z[7])*sin(z[8])*sin(z[9])*z[76]*z[16]*(L2*M3+LC2*M2)
            res[79] += z[65]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)+L2*cos(z[2])*sin(z[3])*z[62]+L2*cos(z[3])*sin(z[2])*z[63]+(L2*cos(z[6])*sin(z[5])*z[66])*0.5-(sqrt(3)*L1*sin(z[4])*z[64])*0.5
            res[80] += -z[65]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-L1*sin(z[1])*z[61]-L2*sin(z[2])*z[62]-(L1*sin(z[4])*z[64])*0.5-(sqrt(3)*L2*cos(z[6])*sin(z[5])*z[66])*0.5
            res[81] += L1*cos(z[1])*z[61]-L1*cos(z[4])*z[64]+L2*cos(z[2])*cos(z[3])*z[62]-L2*cos(z[5])*cos(z[6])*z[65]-L2*sin(z[2])*sin(z[3])*z[63]+L2*sin(z[5])*sin(z[6])*z[66]
            res[82] += z[68]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)+L2*cos(z[2])*sin(z[3])*z[62]+L2*cos(z[3])*sin(z[2])*z[63]+(L2*cos(z[9])*sin(z[8])*z[69])*0.5+(sqrt(3)*L1*sin(z[7])*z[67])*0.5
            res[83] += (sqrt(3)*L2*cos(z[9])*sin(z[8])*z[69])*0.5-L1*sin(z[1])*z[61]-L2*sin(z[2])*z[62]-(L1*sin(z[7])*z[67])*0.5-z[68]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)
            res[84] += L1*cos(z[1])*z[61]-L1*cos(z[7])*z[67]+L2*cos(z[2])*cos(z[3])*z[62]-L2*cos(z[8])*cos(z[9])*z[68]-L2*sin(z[2])*sin(z[3])*z[63]+L2*sin(z[8])*sin(z[9])*z[69]
            res[85] += z[62]*(L2*cos(z[2])*cos(z[3])*z[12]-L2*sin(z[2])*sin(z[3])*z[11])+z[63]*(L2*cos(z[2])*cos(z[3])*z[11]-L2*sin(z[2])*sin(z[3])*z[12])+z[66]*((L2*cos(z[5])*cos(z[6])*z[14])*0.5-(L2*sin(z[5])*sin(z[6])*z[15])*0.5)-z[65]*((sqrt(3)*L2*cos(z[5])*z[14])*0.5-(L2*cos(z[5])*cos(z[6])*z[15])*0.5+(L2*sin(z[5])*sin(z[6])*z[14])*0.5)+z[74]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)+L2*cos(z[2])*sin(z[3])*z[71]+L2*cos(z[3])*sin(z[2])*z[72]+(L2*cos(z[6])*sin(z[5])*z[75])*0.5-(sqrt(3)*L1*sin(z[4])*z[73])*0.5-(sqrt(3)*L1*cos(z[4])*z[64]*z[13])*0.5
            res[86] += -z[74]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-z[65]*((L2*cos(z[5])*z[14])*0.5+(sqrt(3)*L2*cos(z[5])*cos(z[6])*z[15])*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6])*z[14])*0.5)-z[66]*((sqrt(3)*L2*cos(z[5])*cos(z[6])*z[14])*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6])*z[15])*0.5)-L1*sin(z[1])*z[70]-L2*sin(z[2])*z[71]-(L1*sin(z[4])*z[73])*0.5-L1*cos(z[1])*z[61]*z[10]-L2*cos(z[2])*z[62]*z[11]-(L1*cos(z[4])*z[64]*z[13])*0.5-(sqrt(3)*L2*cos(z[6])*sin(z[5])*z[75])*0.5
            res[87] += z[65]*(L2*cos(z[6])*sin(z[5])*z[14]+L2*cos(z[5])*sin(z[6])*z[15])-z[63]*(L2*cos(z[2])*sin(z[3])*z[11]+L2*cos(z[3])*sin(z[2])*z[12])-z[62]*(L2*cos(z[3])*sin(z[2])*z[11]+L2*cos(z[2])*sin(z[3])*z[12])+z[66]*(L2*cos(z[5])*sin(z[6])*z[14]+L2*cos(z[6])*sin(z[5])*z[15])+L1*cos(z[1])*z[70]-L1*cos(z[4])*z[73]+L2*cos(z[2])*cos(z[3])*z[71]-L2*cos(z[5])*cos(z[6])*z[74]-L2*sin(z[2])*sin(z[3])*z[72]+L2*sin(z[5])*sin(z[6])*z[75]-L1*sin(z[1])*z[61]*z[10]+L1*sin(z[4])*z[64]*z[13]
            res[88] += z[62]*(L2*cos(z[2])*cos(z[3])*z[12]-L2*sin(z[2])*sin(z[3])*z[11])+z[63]*(L2*cos(z[2])*cos(z[3])*z[11]-L2*sin(z[2])*sin(z[3])*z[12])+z[69]*((L2*cos(z[8])*cos(z[9])*z[17])*0.5-(L2*sin(z[8])*sin(z[9])*z[18])*0.5)+z[68]*((L2*cos(z[8])*cos(z[9])*z[18])*0.5+(sqrt(3)*L2*cos(z[8])*z[17])*0.5-(L2*sin(z[8])*sin(z[9])*z[17])*0.5)+z[77]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)+L2*cos(z[2])*sin(z[3])*z[71]+L2*cos(z[3])*sin(z[2])*z[72]+(L2*cos(z[9])*sin(z[8])*z[78])*0.5+(sqrt(3)*L1*sin(z[7])*z[76])*0.5+(sqrt(3)*L1*cos(z[7])*z[67]*z[16])*0.5
            res[89] += z[69]*((sqrt(3)*L2*cos(z[8])*cos(z[9])*z[17])*0.5-(sqrt(3)*L2*sin(z[8])*sin(z[9])*z[18])*0.5)-z[68]*((L2*cos(z[8])*z[17])*0.5-(sqrt(3)*L2*cos(z[8])*cos(z[9])*z[18])*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9])*z[17])*0.5)-z[77]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)-L1*sin(z[1])*z[70]-L2*sin(z[2])*z[71]-(L1*sin(z[7])*z[76])*0.5-L1*cos(z[1])*z[61]*z[10]-L2*cos(z[2])*z[62]*z[11]-(L1*cos(z[7])*z[67]*z[16])*0.5+(sqrt(3)*L2*cos(z[9])*sin(z[8])*z[78])*0.5
            res[90] += z[68]*(L2*cos(z[9])*sin(z[8])*z[17]+L2*cos(z[8])*sin(z[9])*z[18])-z[63]*(L2*cos(z[2])*sin(z[3])*z[11]+L2*cos(z[3])*sin(z[2])*z[12])-z[62]*(L2*cos(z[3])*sin(z[2])*z[11]+L2*cos(z[2])*sin(z[3])*z[12])+z[69]*(L2*cos(z[8])*sin(z[9])*z[17]+L2*cos(z[9])*sin(z[8])*z[18])+L1*cos(z[1])*z[70]-L1*cos(z[7])*z[76]+L2*cos(z[2])*cos(z[3])*z[71]-L2*cos(z[8])*cos(z[9])*z[77]-L2*sin(z[2])*sin(z[3])*z[72]+L2*sin(z[8])*sin(z[9])*z[78]-L1*sin(z[1])*z[61]*z[10]+L1*sin(z[7])*z[67]*z[16]
            # Parameter-specific part for M1
            res[70] += LC1^2*dz[10]-LC1*0.0*cos(z[1])
            res[73] += LC1^2*dz[13]-LC1*0.0*cos(z[4])
            res[76] += LC1^2*dz[16]-LC1*0.0*cos(z[7])
            # Sensitivity equations wrt to J1
            res[91] = dz[91]-z[100]-z[91]*(L1*cos(z[1])*dz[26]+L1*cos(z[1])*dz[29]+L1*sin(z[1])*dz[27]+L1*sin(z[1])*dz[30])+L1*cos(z[1])*dz[117]+L1*cos(z[1])*dz[120]-L1*sin(z[1])*dz[116]-L1*sin(z[1])*dz[119]
            res[92] = dz[92]-z[101]+z[93]*(L2*cos(z[2])*cos(z[3])*dz[25]+L2*cos(z[2])*cos(z[3])*dz[28]-L2*cos(z[2])*sin(z[3])*dz[27]-L2*cos(z[2])*sin(z[3])*dz[30])-z[92]*(L2*cos(z[2])*dz[26]+L2*cos(z[2])*dz[29]+L2*cos(z[3])*sin(z[2])*dz[27]+L2*cos(z[3])*sin(z[2])*dz[30]+L2*sin(z[2])*sin(z[3])*dz[25]+L2*sin(z[2])*sin(z[3])*dz[28])-L2*sin(z[2])*dz[116]-L2*sin(z[2])*dz[119]+L2*cos(z[2])*cos(z[3])*dz[117]+L2*cos(z[2])*cos(z[3])*dz[120]+L2*cos(z[2])*sin(z[3])*dz[115]+L2*cos(z[2])*sin(z[3])*dz[118]
            res[93] = dz[93]-z[102]+z[92]*(L2*cos(z[2])*cos(z[3])*dz[25]+L2*cos(z[2])*cos(z[3])*dz[28]-L2*cos(z[2])*sin(z[3])*dz[27]-L2*cos(z[2])*sin(z[3])*dz[30])-z[93]*(L2*cos(z[3])*sin(z[2])*dz[27]+L2*cos(z[3])*sin(z[2])*dz[30]+L2*sin(z[2])*sin(z[3])*dz[25]+L2*sin(z[2])*sin(z[3])*dz[28])+L2*cos(z[3])*sin(z[2])*dz[115]+L2*cos(z[3])*sin(z[2])*dz[118]-L2*sin(z[2])*sin(z[3])*dz[117]-L2*sin(z[2])*sin(z[3])*dz[120]
            res[94] = dz[94]-z[103]-z[94]*((L1*cos(z[4])*dz[26])*0.5-L1*sin(z[4])*dz[27]+(sqrt(3)*L1*cos(z[4])*dz[25])*0.5)-L1*cos(z[4])*dz[117]-(L1*sin(z[4])*dz[116])*0.5-(sqrt(3)*L1*sin(z[4])*dz[115])*0.5
            res[95] = dz[95]-z[104]+dz[115]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)-dz[116]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)+z[96]*((L2*cos(z[5])*cos(z[6])*dz[25])*0.5+L2*cos(z[5])*sin(z[6])*dz[27]-(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[26])*0.5)-z[95]*(dz[25]*((sqrt(3)*L2*cos(z[5]))*0.5+(L2*sin(z[5])*sin(z[6]))*0.5)+dz[26]*((L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5)-L2*cos(z[6])*sin(z[5])*dz[27])-L2*cos(z[5])*cos(z[6])*dz[117]
            res[96] = dz[96]-z[105]+z[95]*((L2*cos(z[5])*cos(z[6])*dz[25])*0.5+L2*cos(z[5])*sin(z[6])*dz[27]-(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[26])*0.5)+z[96]*(L2*cos(z[6])*sin(z[5])*dz[27]-(L2*sin(z[5])*sin(z[6])*dz[25])*0.5+(sqrt(3)*L2*sin(z[5])*sin(z[6])*dz[26])*0.5)+(L2*cos(z[6])*sin(z[5])*dz[115])*0.5+L2*sin(z[5])*sin(z[6])*dz[117]-(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[116])*0.5
            res[97] = dz[97]-z[106]+z[97]*(L1*sin(z[7])*dz[30]-(L1*cos(z[7])*dz[29])*0.5+(sqrt(3)*L1*cos(z[7])*dz[28])*0.5)-L1*cos(z[7])*dz[120]-(L1*sin(z[7])*dz[119])*0.5+(sqrt(3)*L1*sin(z[7])*dz[118])*0.5
            res[98] = dz[98]-z[107]+dz[118]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)-dz[119]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)+z[99]*((L2*cos(z[8])*cos(z[9])*dz[28])*0.5+L2*cos(z[8])*sin(z[9])*dz[30]+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[29])*0.5)+z[98]*(dz[28]*((sqrt(3)*L2*cos(z[8]))*0.5-(L2*sin(z[8])*sin(z[9]))*0.5)-dz[29]*((L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5)+L2*cos(z[9])*sin(z[8])*dz[30])-L2*cos(z[8])*cos(z[9])*dz[120]
            res[99] = dz[99]-z[108]+z[98]*((L2*cos(z[8])*cos(z[9])*dz[28])*0.5+L2*cos(z[8])*sin(z[9])*dz[30]+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[29])*0.5)-z[99]*((L2*sin(z[8])*sin(z[9])*dz[28])*0.5-L2*cos(z[9])*sin(z[8])*dz[30]+(sqrt(3)*L2*sin(z[8])*sin(z[9])*dz[29])*0.5)+(L2*cos(z[9])*sin(z[8])*dz[118])*0.5+L2*sin(z[8])*sin(z[9])*dz[120]+(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[119])*0.5
            res[100] = z[101]*(2*L1*z[11]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[12]*(L2*M3+LC2*M2))+dz[100]*(J1+L1^2*(M2+M3)+LC1^2*M1)-z[102]*(2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]*(L2*M3+LC2*M2))-z[93]*(L1*cos(z[1])*cos(z[2])*sin(z[3])*dz[11]*(L2*M3+LC2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*z[12]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]^2*(L2*M3+LC2*M2)+L1*cos(z[1])*cos(z[3])*sin(z[2])*dz[12]*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[2])*cos(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+z[91]*(0.0*sin(z[1])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[1])*dz[20]+L1*cos(z[1])*dz[23]+L1*sin(z[1])*dz[21]+L1*sin(z[1])*dz[24]+L1*dz[11]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2]))+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)+L1*sin(z[1])*sin(z[2])*sin(z[3])*dz[12]*(L2*M3+LC2*M2)+2*L1*cos(z[2])*sin(z[1])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+γ*z[100]-z[92]*(L1*z[11]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))-L1*dz[11]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))+L1*cos(z[1])*cos(z[2])*cos(z[3])*z[12]^2*(L2*M3+LC2*M2)+L1*cos(z[1])*cos(z[2])*sin(z[3])*dz[12]*(L2*M3+LC2*M2)-2*L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))-L1*cos(z[1])*dz[111]-L1*cos(z[1])*dz[114]+L1*sin(z[1])*dz[110]+L1*sin(z[1])*dz[113]+L1*dz[101]*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))-L1*cos(z[1])*sin(z[2])*sin(z[3])*dz[102]*(L2*M3+LC2*M2)
            res[101] = z[91]*(L1*dz[10]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))-L1*z[10]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3])))+z[92]*(sin(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)+L2*cos(z[2])*dz[20]+L2*cos(z[2])*dz[23]+L2*cos(z[3])*sin(z[2])*dz[21]+L2*cos(z[3])*sin(z[2])*dz[24]+L2*sin(z[2])*sin(z[3])*dz[19]+L2*sin(z[2])*sin(z[3])*dz[22]+L1*dz[10]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))+0.0*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2])))+dz[101]*(J2+L2^2*M3+LC2^2*M2)+γ*z[101]+z[93]*(L2*cos(z[2])*sin(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[22]-L2*cos(z[2])*cos(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[24]+0.0*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[2])*sin(z[3])*dz[10]*(L2*M3+LC2*M2))+L2*sin(z[2])*dz[110]+L2*sin(z[2])*dz[113]-L2*cos(z[2])*cos(z[3])*dz[111]-L2*cos(z[2])*cos(z[3])*dz[114]-L2*cos(z[2])*sin(z[3])*dz[109]-L2*cos(z[2])*sin(z[3])*dz[112]+L1*dz[100]*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))-2*cos(z[2])*sin(z[2])*z[102]*z[12]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[100]*z[10]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))
            res[102] = z[93]*(L2*cos(z[3])*sin(z[2])*dz[21]+L2*cos(z[3])*sin(z[2])*dz[24]+L2*sin(z[2])*sin(z[3])*dz[19]+L2*sin(z[2])*sin(z[3])*dz[22]+0.0*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[10]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[3])*sin(z[2])*dz[10]*(L2*M3+LC2*M2))+z[92]*(L2*cos(z[2])*sin(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[22]-L2*cos(z[2])*cos(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[24]+2*cos(z[2])*sin(z[2])*dz[12]*(J2+L2^2*M3+LC2^2*M2)+2*cos(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+0.0*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[2])*sin(z[3])*dz[10]*(L2*M3+LC2*M2))+z[102]*(γ+sin(2*z[2])*z[11]*(J2+L2^2*M3+LC2^2*M2))+z[91]*(L1*cos(z[1])*sin(z[2])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)+L1*sin(z[1])*sin(z[2])*sin(z[3])*dz[10]*(L2*M3+LC2*M2))+sin(z[2])^2*dz[102]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[3])*sin(z[2])*dz[109]-L2*cos(z[3])*sin(z[2])*dz[112]+L2*sin(z[2])*sin(z[3])*dz[111]+L2*sin(z[2])*sin(z[3])*dz[114]+sin(2*z[2])*z[101]*z[12]*(J2+L2^2*M3+LC2^2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*dz[100]*(L2*M3+LC2*M2)+2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[100]*z[10]*(L2*M3+LC2*M2)
            res[103] = z[104]*(2*L1*z[14]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[15]*(L2*M3+LC2*M2))+dz[103]*(J1+L1^2*(M2+M3)+LC1^2*M1)-z[105]*(2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]*(L2*M3+LC2*M2))-z[96]*(L1*cos(z[4])*cos(z[5])*sin(z[6])*dz[14]*(L2*M3+LC2*M2)-L1*cos(z[4])*sin(z[5])*sin(z[6])*z[15]^2*(L2*M3+LC2*M2)-L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*cos(z[6])*sin(z[5])*dz[15]*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[5])*cos(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+γ*z[103]-z[95]*(L1*z[14]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))-L1*dz[14]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))+L1*cos(z[4])*cos(z[5])*cos(z[6])*z[15]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*cos(z[5])*sin(z[6])*dz[15]*(L2*M3+LC2*M2)-2*L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+z[94]*(0.0*sin(z[4])*(L1*(M2+M3)+LC1*M1)+(L1*cos(z[4])*dz[20])*0.5-L1*sin(z[4])*dz[21]+(sqrt(3)*L1*cos(z[4])*dz[19])*0.5+L1*dz[14]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5]))+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)+L1*sin(z[4])*sin(z[5])*sin(z[6])*dz[15]*(L2*M3+LC2*M2)+2*L1*cos(z[5])*sin(z[4])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+L1*cos(z[4])*dz[111]+(L1*sin(z[4])*dz[110])*0.5+(sqrt(3)*L1*sin(z[4])*dz[109])*0.5+L1*dz[104]*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))-L1*cos(z[4])*sin(z[5])*sin(z[6])*dz[105]*(L2*M3+LC2*M2)
            res[104] = z[94]*(L1*dz[13]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))-L1*z[13]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6])))-z[96]*((L2*cos(z[5])*cos(z[6])*dz[19])*0.5+L2*cos(z[5])*sin(z[6])*dz[21]-0.0*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)-(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[20])*0.5-L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*cos(z[5])*sin(z[6])*dz[13]*(L2*M3+LC2*M2))-dz[109]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)+dz[110]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)+dz[104]*(J2+L2^2*M3+LC2^2*M2)+γ*z[104]+z[95]*(dz[19]*((sqrt(3)*L2*cos(z[5]))*0.5+(L2*sin(z[5])*sin(z[6]))*0.5)+dz[20]*((L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5)-cos(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)+sin(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[6])*sin(z[5])*dz[21]+L1*dz[13]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))+0.0*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5])))+L2*cos(z[5])*cos(z[6])*dz[111]+L1*dz[103]*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))-2*cos(z[5])*sin(z[5])*z[105]*z[15]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[103]*z[13]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))
            res[105] = z[105]*(γ+sin(2*z[5])*z[14]*(J2+L2^2*M3+LC2^2*M2))-z[96]*(L2*cos(z[6])*sin(z[5])*dz[21]-(L2*sin(z[5])*sin(z[6])*dz[19])*0.5-0.0*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+(sqrt(3)*L2*sin(z[5])*sin(z[6])*dz[20])*0.5-L1*cos(z[6])*sin(z[4])*sin(z[5])*z[13]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*cos(z[6])*sin(z[5])*dz[13]*(L2*M3+LC2*M2))+z[95]*(2*cos(z[5])*sin(z[5])*dz[15]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[5])*sin(z[6])*dz[21]-(L2*cos(z[5])*cos(z[6])*dz[19])*0.5+2*cos(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)+0.0*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[20])*0.5+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[5])*sin(z[6])*dz[13]*(L2*M3+LC2*M2))+z[94]*(L1*cos(z[4])*sin(z[5])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)+L1*sin(z[4])*sin(z[5])*sin(z[6])*dz[13]*(L2*M3+LC2*M2))+sin(z[5])^2*dz[105]*(J2+L2^2*M3+LC2^2*M2)-(L2*cos(z[6])*sin(z[5])*dz[109])*0.5-L2*sin(z[5])*sin(z[6])*dz[111]+sin(2*z[5])*z[104]*z[15]*(J2+L2^2*M3+LC2^2*M2)+(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[110])*0.5-L1*cos(z[4])*sin(z[5])*sin(z[6])*dz[103]*(L2*M3+LC2*M2)+2*L1*sin(z[4])*sin(z[5])*sin(z[6])*z[103]*z[13]*(L2*M3+LC2*M2)
            res[106] = z[107]*(2*L1*z[17]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[18]*(L2*M3+LC2*M2))+dz[106]*(J1+L1^2*(M2+M3)+LC1^2*M1)-z[108]*(2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]*(L2*M3+LC2*M2))-z[99]*(L1*cos(z[7])*cos(z[8])*sin(z[9])*dz[17]*(L2*M3+LC2*M2)-L1*cos(z[7])*sin(z[8])*sin(z[9])*z[18]^2*(L2*M3+LC2*M2)-L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*cos(z[9])*sin(z[8])*dz[18]*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[8])*cos(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+γ*z[106]-z[98]*(L1*z[17]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))-L1*dz[17]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))+L1*cos(z[7])*cos(z[8])*cos(z[9])*z[18]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*cos(z[8])*sin(z[9])*dz[18]*(L2*M3+LC2*M2)-2*L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+z[97]*(0.0*sin(z[7])*(L1*(M2+M3)+LC1*M1)+(L1*cos(z[7])*dz[23])*0.5-L1*sin(z[7])*dz[24]-(sqrt(3)*L1*cos(z[7])*dz[22])*0.5+L1*dz[17]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8]))+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)+L1*sin(z[7])*sin(z[8])*sin(z[9])*dz[18]*(L2*M3+LC2*M2)+2*L1*cos(z[8])*sin(z[7])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+L1*cos(z[7])*dz[114]+(L1*sin(z[7])*dz[113])*0.5-(sqrt(3)*L1*sin(z[7])*dz[112])*0.5+L1*dz[107]*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))-L1*cos(z[7])*sin(z[8])*sin(z[9])*dz[108]*(L2*M3+LC2*M2)
            res[107] = z[97]*(L1*dz[16]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))-L1*z[16]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9])))-z[99]*((L2*cos(z[8])*cos(z[9])*dz[22])*0.5+L2*cos(z[8])*sin(z[9])*dz[24]-0.0*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[23])*0.5-L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*cos(z[8])*sin(z[9])*dz[16]*(L2*M3+LC2*M2))-dz[112]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)+dz[113]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)+dz[107]*(J2+L2^2*M3+LC2^2*M2)+γ*z[107]+z[98]*(dz[23]*((L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5)-dz[22]*((sqrt(3)*L2*cos(z[8]))*0.5-(L2*sin(z[8])*sin(z[9]))*0.5)-cos(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)+sin(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[9])*sin(z[8])*dz[24]+L1*dz[16]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))+0.0*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8])))+L2*cos(z[8])*cos(z[9])*dz[114]+L1*dz[106]*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))-2*cos(z[8])*sin(z[8])*z[108]*z[18]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[106]*z[16]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))
            res[108] = z[99]*((L2*sin(z[8])*sin(z[9])*dz[22])*0.5-L2*cos(z[9])*sin(z[8])*dz[24]+0.0*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+(sqrt(3)*L2*sin(z[8])*sin(z[9])*dz[23])*0.5+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[16]^2*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[9])*sin(z[8])*dz[16]*(L2*M3+LC2*M2))+z[108]*(γ+sin(2*z[8])*z[17]*(J2+L2^2*M3+LC2^2*M2))-z[98]*((L2*cos(z[8])*cos(z[9])*dz[22])*0.5+L2*cos(z[8])*sin(z[9])*dz[24]-2*cos(z[8])*sin(z[8])*dz[18]*(J2+L2^2*M3+LC2^2*M2)-2*cos(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)-0.0*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[23])*0.5-L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*cos(z[8])*sin(z[9])*dz[16]*(L2*M3+LC2*M2))+z[97]*(L1*cos(z[7])*sin(z[8])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)+L1*sin(z[7])*sin(z[8])*sin(z[9])*dz[16]*(L2*M3+LC2*M2))+sin(z[8])^2*dz[108]*(J2+L2^2*M3+LC2^2*M2)-(L2*cos(z[9])*sin(z[8])*dz[112])*0.5-L2*sin(z[8])*sin(z[9])*dz[114]+sin(2*z[8])*z[107]*z[18]*(J2+L2^2*M3+LC2^2*M2)-(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[113])*0.5-L1*cos(z[7])*sin(z[8])*sin(z[9])*dz[106]*(L2*M3+LC2*M2)+2*L1*sin(z[7])*sin(z[8])*sin(z[9])*z[106]*z[16]*(L2*M3+LC2*M2)
            res[109] = z[95]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)+L2*cos(z[2])*sin(z[3])*z[92]+L2*cos(z[3])*sin(z[2])*z[93]+(L2*cos(z[6])*sin(z[5])*z[96])*0.5-(sqrt(3)*L1*sin(z[4])*z[94])*0.5
            res[110] = -z[95]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-L1*sin(z[1])*z[91]-L2*sin(z[2])*z[92]-(L1*sin(z[4])*z[94])*0.5-(sqrt(3)*L2*cos(z[6])*sin(z[5])*z[96])*0.5
            res[111] = L1*cos(z[1])*z[91]-L1*cos(z[4])*z[94]+L2*cos(z[2])*cos(z[3])*z[92]-L2*cos(z[5])*cos(z[6])*z[95]-L2*sin(z[2])*sin(z[3])*z[93]+L2*sin(z[5])*sin(z[6])*z[96]
            res[112] = z[98]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)+L2*cos(z[2])*sin(z[3])*z[92]+L2*cos(z[3])*sin(z[2])*z[93]+(L2*cos(z[9])*sin(z[8])*z[99])*0.5+(sqrt(3)*L1*sin(z[7])*z[97])*0.5
            res[113] = (sqrt(3)*L2*cos(z[9])*sin(z[8])*z[99])*0.5-L1*sin(z[1])*z[91]-L2*sin(z[2])*z[92]-(L1*sin(z[7])*z[97])*0.5-z[98]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)
            res[114] = L1*cos(z[1])*z[91]-L1*cos(z[7])*z[97]+L2*cos(z[2])*cos(z[3])*z[92]-L2*cos(z[8])*cos(z[9])*z[98]-L2*sin(z[2])*sin(z[3])*z[93]+L2*sin(z[8])*sin(z[9])*z[99]
            res[115] = z[92]*(L2*cos(z[2])*cos(z[3])*z[12]-L2*sin(z[2])*sin(z[3])*z[11])+z[93]*(L2*cos(z[2])*cos(z[3])*z[11]-L2*sin(z[2])*sin(z[3])*z[12])+z[96]*((L2*cos(z[5])*cos(z[6])*z[14])*0.5-(L2*sin(z[5])*sin(z[6])*z[15])*0.5)-z[95]*((sqrt(3)*L2*cos(z[5])*z[14])*0.5-(L2*cos(z[5])*cos(z[6])*z[15])*0.5+(L2*sin(z[5])*sin(z[6])*z[14])*0.5)+z[104]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)+L2*cos(z[2])*sin(z[3])*z[101]+L2*cos(z[3])*sin(z[2])*z[102]+(L2*cos(z[6])*sin(z[5])*z[105])*0.5-(sqrt(3)*L1*sin(z[4])*z[103])*0.5-(sqrt(3)*L1*cos(z[4])*z[94]*z[13])*0.5
            res[116] = -z[104]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-z[95]*((L2*cos(z[5])*z[14])*0.5+(sqrt(3)*L2*cos(z[5])*cos(z[6])*z[15])*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6])*z[14])*0.5)-z[96]*((sqrt(3)*L2*cos(z[5])*cos(z[6])*z[14])*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6])*z[15])*0.5)-L1*sin(z[1])*z[100]-L2*sin(z[2])*z[101]-(L1*sin(z[4])*z[103])*0.5-L1*cos(z[1])*z[91]*z[10]-L2*cos(z[2])*z[92]*z[11]-(L1*cos(z[4])*z[94]*z[13])*0.5-(sqrt(3)*L2*cos(z[6])*sin(z[5])*z[105])*0.5
            res[117] = z[95]*(L2*cos(z[6])*sin(z[5])*z[14]+L2*cos(z[5])*sin(z[6])*z[15])-z[93]*(L2*cos(z[2])*sin(z[3])*z[11]+L2*cos(z[3])*sin(z[2])*z[12])-z[92]*(L2*cos(z[3])*sin(z[2])*z[11]+L2*cos(z[2])*sin(z[3])*z[12])+z[96]*(L2*cos(z[5])*sin(z[6])*z[14]+L2*cos(z[6])*sin(z[5])*z[15])+L1*cos(z[1])*z[100]-L1*cos(z[4])*z[103]+L2*cos(z[2])*cos(z[3])*z[101]-L2*cos(z[5])*cos(z[6])*z[104]-L2*sin(z[2])*sin(z[3])*z[102]+L2*sin(z[5])*sin(z[6])*z[105]-L1*sin(z[1])*z[91]*z[10]+L1*sin(z[4])*z[94]*z[13]
            res[118] = z[92]*(L2*cos(z[2])*cos(z[3])*z[12]-L2*sin(z[2])*sin(z[3])*z[11])+z[93]*(L2*cos(z[2])*cos(z[3])*z[11]-L2*sin(z[2])*sin(z[3])*z[12])+z[99]*((L2*cos(z[8])*cos(z[9])*z[17])*0.5-(L2*sin(z[8])*sin(z[9])*z[18])*0.5)+z[98]*((L2*cos(z[8])*cos(z[9])*z[18])*0.5+(sqrt(3)*L2*cos(z[8])*z[17])*0.5-(L2*sin(z[8])*sin(z[9])*z[17])*0.5)+z[107]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)+L2*cos(z[2])*sin(z[3])*z[101]+L2*cos(z[3])*sin(z[2])*z[102]+(L2*cos(z[9])*sin(z[8])*z[108])*0.5+(sqrt(3)*L1*sin(z[7])*z[106])*0.5+(sqrt(3)*L1*cos(z[7])*z[97]*z[16])*0.5
            res[119] = z[99]*((sqrt(3)*L2*cos(z[8])*cos(z[9])*z[17])*0.5-(sqrt(3)*L2*sin(z[8])*sin(z[9])*z[18])*0.5)-z[98]*((L2*cos(z[8])*z[17])*0.5-(sqrt(3)*L2*cos(z[8])*cos(z[9])*z[18])*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9])*z[17])*0.5)-z[107]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)-L1*sin(z[1])*z[100]-L2*sin(z[2])*z[101]-(L1*sin(z[7])*z[106])*0.5-L1*cos(z[1])*z[91]*z[10]-L2*cos(z[2])*z[92]*z[11]-(L1*cos(z[7])*z[97]*z[16])*0.5+(sqrt(3)*L2*cos(z[9])*sin(z[8])*z[108])*0.5
            res[120] = z[98]*(L2*cos(z[9])*sin(z[8])*z[17]+L2*cos(z[8])*sin(z[9])*z[18])-z[93]*(L2*cos(z[2])*sin(z[3])*z[11]+L2*cos(z[3])*sin(z[2])*z[12])-z[92]*(L2*cos(z[3])*sin(z[2])*z[11]+L2*cos(z[2])*sin(z[3])*z[12])+z[99]*(L2*cos(z[8])*sin(z[9])*z[17]+L2*cos(z[9])*sin(z[8])*z[18])+L1*cos(z[1])*z[100]-L1*cos(z[7])*z[106]+L2*cos(z[2])*cos(z[3])*z[101]-L2*cos(z[8])*cos(z[9])*z[107]-L2*sin(z[2])*sin(z[3])*z[102]+L2*sin(z[8])*sin(z[9])*z[108]-L1*sin(z[1])*z[91]*z[10]+L1*sin(z[7])*z[97]*z[16]
            # Parameter-specific part for J1
            res[100] += dz[10]
            res[103] += dz[13]
            res[106] += dz[16]

            nothing
        end

        z0, dz0 = get_delta_initial_L1M1J1sens_comp(θ, u(0.0), w(0.0))

        dvars = fill(true, 120)
        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 for delta robot is: $r0"

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, z0, dz0, dvars, r0)
    end
end

function delta_robot_gc_adjoint_γonly(u::Function, w::Function, θ::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Matrix{Float64}, dx::Function, dx2::Function)#::Tuple{Model, Function, Function}
    let L0 = θ[1], L1 = θ[2], L2 = θ[3], L3 = θ[4], LC1 = θ[5], LC2 = θ[6], M1 = θ[7], M2 = θ[8], M3 = θ[9], J1 = θ[10], J2 = θ[11], g = 0.0, γ = θ[13]
        
        (nx, np) = size(xp0)
        # np = size(xp0,2)
        @assert (np == 1) "delta_robot_gc_adjoint_γonly is hard-coded to only handle one parameter γ, make sure to pass correct xp0"
        # nx = size(xp0,1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        Fx = [-L1*cos(x(T)[1])*dx(T)[26]-L1*cos(x(T)[1])*dx(T)[29]-L1*sin(x(T)[1])*dx(T)[27]-L1*sin(x(T)[1])*dx(T)[30]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   -L2*cos(x(T)[2])*dx(T)[26]-L2*cos(x(T)[2])*dx(T)[29]-L2*cos(x(T)[3])*sin(x(T)[2])*dx(T)[27]-L2*cos(x(T)[3])*sin(x(T)[2])*dx(T)[30]-L2*sin(x(T)[2])*sin(x(T)[3])*dx(T)[25]-L2*sin(x(T)[2])*sin(x(T)[3])*dx(T)[28]   L2*cos(x(T)[2])*cos(x(T)[3])*dx(T)[25]+L2*cos(x(T)[2])*cos(x(T)[3])*dx(T)[28]-L2*cos(x(T)[2])*sin(x(T)[3])*dx(T)[27]-L2*cos(x(T)[2])*sin(x(T)[3])*dx(T)[30]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   L2*cos(x(T)[2])*cos(x(T)[3])*dx(T)[25]+L2*cos(x(T)[2])*cos(x(T)[3])*dx(T)[28]-L2*cos(x(T)[2])*sin(x(T)[3])*dx(T)[27]-L2*cos(x(T)[2])*sin(x(T)[3])*dx(T)[30]   -L2*cos(x(T)[3])*sin(x(T)[2])*dx(T)[27]-L2*cos(x(T)[3])*sin(x(T)[2])*dx(T)[30]-L2*sin(x(T)[2])*sin(x(T)[3])*dx(T)[25]-L2*sin(x(T)[2])*sin(x(T)[3])*dx(T)[28]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   L1*sin(x(T)[4])*dx(T)[27]-(L1*cos(x(T)[4])*dx(T)[26])*0.5-(sqrt(3)*L1*cos(x(T)[4])*dx(T)[25])*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   L2*cos(x(T)[6])*sin(x(T)[5])*dx(T)[27]-dx(T)[26]*((L2*cos(x(T)[5]))*0.5-(sqrt(3)*L2*sin(x(T)[5])*sin(x(T)[6]))*0.5)-dx(T)[25]*((sqrt(3)*L2*cos(x(T)[5]))*0.5+(L2*sin(x(T)[5])*sin(x(T)[6]))*0.5)   (L2*cos(x(T)[5])*cos(x(T)[6])*dx(T)[25])*0.5+L2*cos(x(T)[5])*sin(x(T)[6])*dx(T)[27]-(sqrt(3)*L2*cos(x(T)[5])*cos(x(T)[6])*dx(T)[26])*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   (L2*cos(x(T)[5])*cos(x(T)[6])*dx(T)[25])*0.5+L2*cos(x(T)[5])*sin(x(T)[6])*dx(T)[27]-(sqrt(3)*L2*cos(x(T)[5])*cos(x(T)[6])*dx(T)[26])*0.5   L2*cos(x(T)[6])*sin(x(T)[5])*dx(T)[27]-(L2*sin(x(T)[5])*sin(x(T)[6])*dx(T)[25])*0.5+(sqrt(3)*L2*sin(x(T)[5])*sin(x(T)[6])*dx(T)[26])*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   L1*sin(x(T)[7])*dx(T)[30]-(L1*cos(x(T)[7])*dx(T)[29])*0.5+(sqrt(3)*L1*cos(x(T)[7])*dx(T)[28])*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   dx(T)[28]*((sqrt(3)*L2*cos(x(T)[8]))*0.5-(L2*sin(x(T)[8])*sin(x(T)[9]))*0.5)-dx(T)[29]*((L2*cos(x(T)[8]))*0.5+(sqrt(3)*L2*sin(x(T)[8])*sin(x(T)[9]))*0.5)+L2*cos(x(T)[9])*sin(x(T)[8])*dx(T)[30]   (L2*cos(x(T)[8])*cos(x(T)[9])*dx(T)[28])*0.5+L2*cos(x(T)[8])*sin(x(T)[9])*dx(T)[30]+(sqrt(3)*L2*cos(x(T)[8])*cos(x(T)[9])*dx(T)[29])*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x(T)[8])*cos(x(T)[9])*dx(T)[28])*0.5+L2*cos(x(T)[8])*sin(x(T)[9])*dx(T)[30]+(sqrt(3)*L2*cos(x(T)[8])*cos(x(T)[9])*dx(T)[29])*0.5   L2*cos(x(T)[9])*sin(x(T)[8])*dx(T)[30]-(L2*sin(x(T)[8])*sin(x(T)[9])*dx(T)[28])*0.5-(sqrt(3)*L2*sin(x(T)[8])*sin(x(T)[9])*dx(T)[29])*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0*sin(x(T)[1])*(L1*(M2+M3)+LC1*M1)+L1*cos(x(T)[1])*dx(T)[20]+L1*cos(x(T)[1])*dx(T)[23]+L1*sin(x(T)[1])*dx(T)[21]+L1*sin(x(T)[1])*dx(T)[24]+L1*dx(T)[11]*(L2*M3+LC2*M2)*(cos(x(T)[1])*sin(x(T)[2])-cos(x(T)[2])*cos(x(T)[3])*sin(x(T)[1]))+L1*x(T)[11]^2*(L2*M3+LC2*M2)*(cos(x(T)[1])*cos(x(T)[2])+cos(x(T)[3])*sin(x(T)[1])*sin(x(T)[2]))+L1*cos(x(T)[3])*sin(x(T)[1])*sin(x(T)[2])*x(T)[12]^2*(L2*M3+LC2*M2)+L1*sin(x(T)[1])*sin(x(T)[2])*sin(x(T)[3])*dx(T)[12]*(L2*M3+LC2*M2)+2*L1*cos(x(T)[2])*sin(x(T)[1])*sin(x(T)[3])*x(T)[11]*x(T)[12]*(L2*M3+LC2*M2)   L1*dx(T)[11]*(L2*M3+LC2*M2)*(cos(x(T)[2])*sin(x(T)[1])-cos(x(T)[1])*cos(x(T)[3])*sin(x(T)[2]))-L1*x(T)[11]^2*(L2*M3+LC2*M2)*(sin(x(T)[1])*sin(x(T)[2])+cos(x(T)[1])*cos(x(T)[2])*cos(x(T)[3]))-L1*cos(x(T)[1])*cos(x(T)[2])*cos(x(T)[3])*x(T)[12]^2*(L2*M3+LC2*M2)-L1*cos(x(T)[1])*cos(x(T)[2])*sin(x(T)[3])*dx(T)[12]*(L2*M3+LC2*M2)+2*L1*cos(x(T)[1])*sin(x(T)[2])*sin(x(T)[3])*x(T)[11]*x(T)[12]*(L2*M3+LC2*M2)   L1*cos(x(T)[1])*sin(x(T)[2])*sin(x(T)[3])*x(T)[11]^2*(L2*M3+LC2*M2)+L1*cos(x(T)[1])*sin(x(T)[2])*sin(x(T)[3])*x(T)[12]^2*(L2*M3+LC2*M2)-L1*cos(x(T)[1])*cos(x(T)[2])*sin(x(T)[3])*dx(T)[11]*(L2*M3+LC2*M2)-L1*cos(x(T)[1])*cos(x(T)[3])*sin(x(T)[2])*dx(T)[12]*(L2*M3+LC2*M2)-2*L1*cos(x(T)[1])*cos(x(T)[2])*cos(x(T)[3])*x(T)[11]*x(T)[12]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   γ   2*L1*x(T)[11]*(L2*M3+LC2*M2)*(cos(x(T)[2])*sin(x(T)[1])-cos(x(T)[1])*cos(x(T)[3])*sin(x(T)[2]))-2*L1*cos(x(T)[1])*cos(x(T)[2])*sin(x(T)[3])*x(T)[12]*(L2*M3+LC2*M2)   -2*L1*cos(x(T)[1])*cos(x(T)[2])*sin(x(T)[3])*x(T)[11]*(L2*M3+LC2*M2)-2*L1*cos(x(T)[1])*cos(x(T)[3])*sin(x(T)[2])*x(T)[12]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            L1*dx(T)[10]*(L2*M3+LC2*M2)*(cos(x(T)[1])*sin(x(T)[2])-cos(x(T)[2])*cos(x(T)[3])*sin(x(T)[1]))-L1*x(T)[10]^2*(L2*M3+LC2*M2)*(sin(x(T)[1])*sin(x(T)[2])+cos(x(T)[1])*cos(x(T)[2])*cos(x(T)[3]))   sin(x(T)[2])^2*x(T)[12]^2*(J2+L2^2*M3+LC2^2*M2)-cos(x(T)[2])^2*x(T)[12]^2*(J2+L2^2*M3+LC2^2*M2)+L2*cos(x(T)[2])*dx(T)[20]+L2*cos(x(T)[2])*dx(T)[23]+L2*cos(x(T)[3])*sin(x(T)[2])*dx(T)[21]+L2*cos(x(T)[3])*sin(x(T)[2])*dx(T)[24]+L2*sin(x(T)[2])*sin(x(T)[3])*dx(T)[19]+L2*sin(x(T)[2])*sin(x(T)[3])*dx(T)[22]+L1*dx(T)[10]*(L2*M3+LC2*M2)*(cos(x(T)[2])*sin(x(T)[1])-cos(x(T)[1])*cos(x(T)[3])*sin(x(T)[2]))+0.0*cos(x(T)[3])*sin(x(T)[2])*(L2*M3+LC2*M2)+L1*x(T)[10]^2*(L2*M3+LC2*M2)*(cos(x(T)[1])*cos(x(T)[2])+cos(x(T)[3])*sin(x(T)[1])*sin(x(T)[2]))   L2*cos(x(T)[2])*sin(x(T)[3])*dx(T)[21]-L2*cos(x(T)[2])*cos(x(T)[3])*dx(T)[22]-L2*cos(x(T)[2])*cos(x(T)[3])*dx(T)[19]+L2*cos(x(T)[2])*sin(x(T)[3])*dx(T)[24]+0.0*cos(x(T)[2])*sin(x(T)[3])*(L2*M3+LC2*M2)+L1*cos(x(T)[2])*sin(x(T)[1])*sin(x(T)[3])*x(T)[10]^2*(L2*M3+LC2*M2)-L1*cos(x(T)[1])*cos(x(T)[2])*sin(x(T)[3])*dx(T)[10]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   2*L1*x(T)[10]*(L2*M3+LC2*M2)*(cos(x(T)[1])*sin(x(T)[2])-cos(x(T)[2])*cos(x(T)[3])*sin(x(T)[1]))   γ   -2*cos(x(T)[2])*sin(x(T)[2])*x(T)[12]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            L1*cos(x(T)[1])*sin(x(T)[2])*sin(x(T)[3])*x(T)[10]^2*(L2*M3+LC2*M2)+L1*sin(x(T)[1])*sin(x(T)[2])*sin(x(T)[3])*dx(T)[10]*(L2*M3+LC2*M2)   L2*cos(x(T)[2])*sin(x(T)[3])*dx(T)[21]-L2*cos(x(T)[2])*cos(x(T)[3])*dx(T)[22]-L2*cos(x(T)[2])*cos(x(T)[3])*dx(T)[19]+L2*cos(x(T)[2])*sin(x(T)[3])*dx(T)[24]+2*cos(x(T)[2])*sin(x(T)[2])*dx(T)[12]*(J2+L2^2*M3+LC2^2*M2)+2*cos(2*x(T)[2])*x(T)[11]*x(T)[12]*(J2+L2^2*M3+LC2^2*M2)+0.0*cos(x(T)[2])*sin(x(T)[3])*(L2*M3+LC2*M2)+L1*cos(x(T)[2])*sin(x(T)[1])*sin(x(T)[3])*x(T)[10]^2*(L2*M3+LC2*M2)-L1*cos(x(T)[1])*cos(x(T)[2])*sin(x(T)[3])*dx(T)[10]*(L2*M3+LC2*M2)   L2*cos(x(T)[3])*sin(x(T)[2])*dx(T)[21]+L2*cos(x(T)[3])*sin(x(T)[2])*dx(T)[24]+L2*sin(x(T)[2])*sin(x(T)[3])*dx(T)[19]+L2*sin(x(T)[2])*sin(x(T)[3])*dx(T)[22]+0.0*cos(x(T)[3])*sin(x(T)[2])*(L2*M3+LC2*M2)+L1*cos(x(T)[3])*sin(x(T)[1])*sin(x(T)[2])*x(T)[10]^2*(L2*M3+LC2*M2)-L1*cos(x(T)[1])*cos(x(T)[3])*sin(x(T)[2])*dx(T)[10]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   2*L1*sin(x(T)[1])*sin(x(T)[2])*sin(x(T)[3])*x(T)[10]*(L2*M3+LC2*M2)   sin(2*x(T)[2])*x(T)[12]*(J2+L2^2*M3+LC2^2*M2)   γ+sin(2*x(T)[2])*x(T)[11]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0*sin(x(T)[4])*(L1*(M2+M3)+LC1*M1)+(L1*cos(x(T)[4])*dx(T)[20])*0.5-L1*sin(x(T)[4])*dx(T)[21]+(sqrt(3)*L1*cos(x(T)[4])*dx(T)[19])*0.5+L1*dx(T)[14]*(L2*M3+LC2*M2)*(cos(x(T)[4])*sin(x(T)[5])-cos(x(T)[5])*cos(x(T)[6])*sin(x(T)[4]))+L1*x(T)[14]^2*(L2*M3+LC2*M2)*(cos(x(T)[4])*cos(x(T)[5])+cos(x(T)[6])*sin(x(T)[4])*sin(x(T)[5]))+L1*cos(x(T)[6])*sin(x(T)[4])*sin(x(T)[5])*x(T)[15]^2*(L2*M3+LC2*M2)+L1*sin(x(T)[4])*sin(x(T)[5])*sin(x(T)[6])*dx(T)[15]*(L2*M3+LC2*M2)+2*L1*cos(x(T)[5])*sin(x(T)[4])*sin(x(T)[6])*x(T)[14]*x(T)[15]*(L2*M3+LC2*M2)   L1*dx(T)[14]*(L2*M3+LC2*M2)*(cos(x(T)[5])*sin(x(T)[4])-cos(x(T)[4])*cos(x(T)[6])*sin(x(T)[5]))-L1*x(T)[14]^2*(L2*M3+LC2*M2)*(sin(x(T)[4])*sin(x(T)[5])+cos(x(T)[4])*cos(x(T)[5])*cos(x(T)[6]))-L1*cos(x(T)[4])*cos(x(T)[5])*cos(x(T)[6])*x(T)[15]^2*(L2*M3+LC2*M2)-L1*cos(x(T)[4])*cos(x(T)[5])*sin(x(T)[6])*dx(T)[15]*(L2*M3+LC2*M2)+2*L1*cos(x(T)[4])*sin(x(T)[5])*sin(x(T)[6])*x(T)[14]*x(T)[15]*(L2*M3+LC2*M2)   L1*cos(x(T)[4])*sin(x(T)[5])*sin(x(T)[6])*x(T)[14]^2*(L2*M3+LC2*M2)+L1*cos(x(T)[4])*sin(x(T)[5])*sin(x(T)[6])*x(T)[15]^2*(L2*M3+LC2*M2)-L1*cos(x(T)[4])*cos(x(T)[5])*sin(x(T)[6])*dx(T)[14]*(L2*M3+LC2*M2)-L1*cos(x(T)[4])*cos(x(T)[6])*sin(x(T)[5])*dx(T)[15]*(L2*M3+LC2*M2)-2*L1*cos(x(T)[4])*cos(x(T)[5])*cos(x(T)[6])*x(T)[14]*x(T)[15]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   γ   2*L1*x(T)[14]*(L2*M3+LC2*M2)*(cos(x(T)[5])*sin(x(T)[4])-cos(x(T)[4])*cos(x(T)[6])*sin(x(T)[5]))-2*L1*cos(x(T)[4])*cos(x(T)[5])*sin(x(T)[6])*x(T)[15]*(L2*M3+LC2*M2)   -2*L1*cos(x(T)[4])*cos(x(T)[5])*sin(x(T)[6])*x(T)[14]*(L2*M3+LC2*M2)-2*L1*cos(x(T)[4])*cos(x(T)[6])*sin(x(T)[5])*x(T)[15]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   L1*dx(T)[13]*(L2*M3+LC2*M2)*(cos(x(T)[4])*sin(x(T)[5])-cos(x(T)[5])*cos(x(T)[6])*sin(x(T)[4]))-L1*x(T)[13]^2*(L2*M3+LC2*M2)*(sin(x(T)[4])*sin(x(T)[5])+cos(x(T)[4])*cos(x(T)[5])*cos(x(T)[6]))   dx(T)[19]*((sqrt(3)*L2*cos(x(T)[5]))*0.5+(L2*sin(x(T)[5])*sin(x(T)[6]))*0.5)+dx(T)[20]*((L2*cos(x(T)[5]))*0.5-(sqrt(3)*L2*sin(x(T)[5])*sin(x(T)[6]))*0.5)-cos(x(T)[5])^2*x(T)[15]^2*(J2+L2^2*M3+LC2^2*M2)+sin(x(T)[5])^2*x(T)[15]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(x(T)[6])*sin(x(T)[5])*dx(T)[21]+L1*dx(T)[13]*(L2*M3+LC2*M2)*(cos(x(T)[5])*sin(x(T)[4])-cos(x(T)[4])*cos(x(T)[6])*sin(x(T)[5]))+0.0*cos(x(T)[6])*sin(x(T)[5])*(L2*M3+LC2*M2)+L1*x(T)[13]^2*(L2*M3+LC2*M2)*(cos(x(T)[4])*cos(x(T)[5])+cos(x(T)[6])*sin(x(T)[4])*sin(x(T)[5]))   0.0*cos(x(T)[5])*sin(x(T)[6])*(L2*M3+LC2*M2)-L2*cos(x(T)[5])*sin(x(T)[6])*dx(T)[21]-(L2*cos(x(T)[5])*cos(x(T)[6])*dx(T)[19])*0.5+(sqrt(3)*L2*cos(x(T)[5])*cos(x(T)[6])*dx(T)[20])*0.5+L1*cos(x(T)[5])*sin(x(T)[4])*sin(x(T)[6])*x(T)[13]^2*(L2*M3+LC2*M2)-L1*cos(x(T)[4])*cos(x(T)[5])*sin(x(T)[6])*dx(T)[13]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   2*L1*x(T)[13]*(L2*M3+LC2*M2)*(cos(x(T)[4])*sin(x(T)[5])-cos(x(T)[5])*cos(x(T)[6])*sin(x(T)[4]))   γ   -2*cos(x(T)[5])*sin(x(T)[5])*x(T)[15]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   L1*cos(x(T)[4])*sin(x(T)[5])*sin(x(T)[6])*x(T)[13]^2*(L2*M3+LC2*M2)+L1*sin(x(T)[4])*sin(x(T)[5])*sin(x(T)[6])*dx(T)[13]*(L2*M3+LC2*M2)   2*cos(x(T)[5])*sin(x(T)[5])*dx(T)[15]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(x(T)[5])*sin(x(T)[6])*dx(T)[21]-(L2*cos(x(T)[5])*cos(x(T)[6])*dx(T)[19])*0.5+2*cos(2*x(T)[5])*x(T)[14]*x(T)[15]*(J2+L2^2*M3+LC2^2*M2)+0.0*cos(x(T)[5])*sin(x(T)[6])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(x(T)[5])*cos(x(T)[6])*dx(T)[20])*0.5+L1*cos(x(T)[5])*sin(x(T)[4])*sin(x(T)[6])*x(T)[13]^2*(L2*M3+LC2*M2)-L1*cos(x(T)[4])*cos(x(T)[5])*sin(x(T)[6])*dx(T)[13]*(L2*M3+LC2*M2)   (L2*sin(x(T)[5])*sin(x(T)[6])*dx(T)[19])*0.5-L2*cos(x(T)[6])*sin(x(T)[5])*dx(T)[21]+0.0*cos(x(T)[6])*sin(x(T)[5])*(L2*M3+LC2*M2)-(sqrt(3)*L2*sin(x(T)[5])*sin(x(T)[6])*dx(T)[20])*0.5+L1*cos(x(T)[6])*sin(x(T)[4])*sin(x(T)[5])*x(T)[13]^2*(L2*M3+LC2*M2)-L1*cos(x(T)[4])*cos(x(T)[6])*sin(x(T)[5])*dx(T)[13]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   2*L1*sin(x(T)[4])*sin(x(T)[5])*sin(x(T)[6])*x(T)[13]*(L2*M3+LC2*M2)   sin(2*x(T)[5])*x(T)[15]*(J2+L2^2*M3+LC2^2*M2)   γ+sin(2*x(T)[5])*x(T)[14]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0*sin(x(T)[7])*(L1*(M2+M3)+LC1*M1)+(L1*cos(x(T)[7])*dx(T)[23])*0.5-L1*sin(x(T)[7])*dx(T)[24]-(sqrt(3)*L1*cos(x(T)[7])*dx(T)[22])*0.5+L1*dx(T)[17]*(L2*M3+LC2*M2)*(cos(x(T)[7])*sin(x(T)[8])-cos(x(T)[8])*cos(x(T)[9])*sin(x(T)[7]))+L1*x(T)[17]^2*(L2*M3+LC2*M2)*(cos(x(T)[7])*cos(x(T)[8])+cos(x(T)[9])*sin(x(T)[7])*sin(x(T)[8]))+L1*cos(x(T)[9])*sin(x(T)[7])*sin(x(T)[8])*x(T)[18]^2*(L2*M3+LC2*M2)+L1*sin(x(T)[7])*sin(x(T)[8])*sin(x(T)[9])*dx(T)[18]*(L2*M3+LC2*M2)+2*L1*cos(x(T)[8])*sin(x(T)[7])*sin(x(T)[9])*x(T)[17]*x(T)[18]*(L2*M3+LC2*M2)   L1*dx(T)[17]*(L2*M3+LC2*M2)*(cos(x(T)[8])*sin(x(T)[7])-cos(x(T)[7])*cos(x(T)[9])*sin(x(T)[8]))-L1*x(T)[17]^2*(L2*M3+LC2*M2)*(sin(x(T)[7])*sin(x(T)[8])+cos(x(T)[7])*cos(x(T)[8])*cos(x(T)[9]))-L1*cos(x(T)[7])*cos(x(T)[8])*cos(x(T)[9])*x(T)[18]^2*(L2*M3+LC2*M2)-L1*cos(x(T)[7])*cos(x(T)[8])*sin(x(T)[9])*dx(T)[18]*(L2*M3+LC2*M2)+2*L1*cos(x(T)[7])*sin(x(T)[8])*sin(x(T)[9])*x(T)[17]*x(T)[18]*(L2*M3+LC2*M2)   L1*cos(x(T)[7])*sin(x(T)[8])*sin(x(T)[9])*x(T)[17]^2*(L2*M3+LC2*M2)+L1*cos(x(T)[7])*sin(x(T)[8])*sin(x(T)[9])*x(T)[18]^2*(L2*M3+LC2*M2)-L1*cos(x(T)[7])*cos(x(T)[8])*sin(x(T)[9])*dx(T)[17]*(L2*M3+LC2*M2)-L1*cos(x(T)[7])*cos(x(T)[9])*sin(x(T)[8])*dx(T)[18]*(L2*M3+LC2*M2)-2*L1*cos(x(T)[7])*cos(x(T)[8])*cos(x(T)[9])*x(T)[17]*x(T)[18]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   γ   2*L1*x(T)[17]*(L2*M3+LC2*M2)*(cos(x(T)[8])*sin(x(T)[7])-cos(x(T)[7])*cos(x(T)[9])*sin(x(T)[8]))-2*L1*cos(x(T)[7])*cos(x(T)[8])*sin(x(T)[9])*x(T)[18]*(L2*M3+LC2*M2)   -2*L1*cos(x(T)[7])*cos(x(T)[8])*sin(x(T)[9])*x(T)[17]*(L2*M3+LC2*M2)-2*L1*cos(x(T)[7])*cos(x(T)[9])*sin(x(T)[8])*x(T)[18]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   L1*dx(T)[16]*(L2*M3+LC2*M2)*(cos(x(T)[7])*sin(x(T)[8])-cos(x(T)[8])*cos(x(T)[9])*sin(x(T)[7]))-L1*x(T)[16]^2*(L2*M3+LC2*M2)*(sin(x(T)[7])*sin(x(T)[8])+cos(x(T)[7])*cos(x(T)[8])*cos(x(T)[9]))   dx(T)[23]*((L2*cos(x(T)[8]))*0.5+(sqrt(3)*L2*sin(x(T)[8])*sin(x(T)[9]))*0.5)-dx(T)[22]*((sqrt(3)*L2*cos(x(T)[8]))*0.5-(L2*sin(x(T)[8])*sin(x(T)[9]))*0.5)-cos(x(T)[8])^2*x(T)[18]^2*(J2+L2^2*M3+LC2^2*M2)+sin(x(T)[8])^2*x(T)[18]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(x(T)[9])*sin(x(T)[8])*dx(T)[24]+L1*dx(T)[16]*(L2*M3+LC2*M2)*(cos(x(T)[8])*sin(x(T)[7])-cos(x(T)[7])*cos(x(T)[9])*sin(x(T)[8]))+0.0*cos(x(T)[9])*sin(x(T)[8])*(L2*M3+LC2*M2)+L1*x(T)[16]^2*(L2*M3+LC2*M2)*(cos(x(T)[7])*cos(x(T)[8])+cos(x(T)[9])*sin(x(T)[7])*sin(x(T)[8]))   0.0*cos(x(T)[8])*sin(x(T)[9])*(L2*M3+LC2*M2)-L2*cos(x(T)[8])*sin(x(T)[9])*dx(T)[24]-(L2*cos(x(T)[8])*cos(x(T)[9])*dx(T)[22])*0.5-(sqrt(3)*L2*cos(x(T)[8])*cos(x(T)[9])*dx(T)[23])*0.5+L1*cos(x(T)[8])*sin(x(T)[7])*sin(x(T)[9])*x(T)[16]^2*(L2*M3+LC2*M2)-L1*cos(x(T)[7])*cos(x(T)[8])*sin(x(T)[9])*dx(T)[16]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   2*L1*x(T)[16]*(L2*M3+LC2*M2)*(cos(x(T)[7])*sin(x(T)[8])-cos(x(T)[8])*cos(x(T)[9])*sin(x(T)[7]))   γ   -2*cos(x(T)[8])*sin(x(T)[8])*x(T)[18]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   L1*cos(x(T)[7])*sin(x(T)[8])*sin(x(T)[9])*x(T)[16]^2*(L2*M3+LC2*M2)+L1*sin(x(T)[7])*sin(x(T)[8])*sin(x(T)[9])*dx(T)[16]*(L2*M3+LC2*M2)   2*cos(x(T)[8])*sin(x(T)[8])*dx(T)[18]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(x(T)[8])*sin(x(T)[9])*dx(T)[24]-(L2*cos(x(T)[8])*cos(x(T)[9])*dx(T)[22])*0.5+2*cos(2*x(T)[8])*x(T)[17]*x(T)[18]*(J2+L2^2*M3+LC2^2*M2)+0.0*cos(x(T)[8])*sin(x(T)[9])*(L2*M3+LC2*M2)-(sqrt(3)*L2*cos(x(T)[8])*cos(x(T)[9])*dx(T)[23])*0.5+L1*cos(x(T)[8])*sin(x(T)[7])*sin(x(T)[9])*x(T)[16]^2*(L2*M3+LC2*M2)-L1*cos(x(T)[7])*cos(x(T)[8])*sin(x(T)[9])*dx(T)[16]*(L2*M3+LC2*M2)   (L2*sin(x(T)[8])*sin(x(T)[9])*dx(T)[22])*0.5-L2*cos(x(T)[9])*sin(x(T)[8])*dx(T)[24]+0.0*cos(x(T)[9])*sin(x(T)[8])*(L2*M3+LC2*M2)+(sqrt(3)*L2*sin(x(T)[8])*sin(x(T)[9])*dx(T)[23])*0.5+L1*cos(x(T)[9])*sin(x(T)[7])*sin(x(T)[8])*x(T)[16]^2*(L2*M3+LC2*M2)-L1*cos(x(T)[7])*cos(x(T)[9])*sin(x(T)[8])*dx(T)[16]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   2*L1*sin(x(T)[7])*sin(x(T)[8])*sin(x(T)[9])*x(T)[16]*(L2*M3+LC2*M2)   sin(2*x(T)[8])*x(T)[18]*(J2+L2^2*M3+LC2^2*M2)   γ+sin(2*x(T)[8])*x(T)[17]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   L2*cos(x(T)[2])*sin(x(T)[3])   L2*cos(x(T)[3])*sin(x(T)[2])   -(sqrt(3)*L1*sin(x(T)[4]))*0.5   (L2*cos(x(T)[5])*sin(x(T)[6]))*0.5-(sqrt(3)*L2*sin(x(T)[5]))*0.5   (L2*cos(x(T)[6])*sin(x(T)[5]))*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            -L1*sin(x(T)[1])   -L2*sin(x(T)[2])   0.0   -(L1*sin(x(T)[4]))*0.5   -(L2*sin(x(T)[5]))*0.5-(sqrt(3)*L2*cos(x(T)[5])*sin(x(T)[6]))*0.5   -(sqrt(3)*L2*cos(x(T)[6])*sin(x(T)[5]))*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            L1*cos(x(T)[1])   L2*cos(x(T)[2])*cos(x(T)[3])   -L2*sin(x(T)[2])*sin(x(T)[3])   -L1*cos(x(T)[4])   -L2*cos(x(T)[5])*cos(x(T)[6])   L2*sin(x(T)[5])*sin(x(T)[6])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   L2*cos(x(T)[2])*sin(x(T)[3])   L2*cos(x(T)[3])*sin(x(T)[2])   0.0   0.0   0.0   (sqrt(3)*L1*sin(x(T)[7]))*0.5   (L2*cos(x(T)[8])*sin(x(T)[9]))*0.5+(sqrt(3)*L2*sin(x(T)[8]))*0.5   (L2*cos(x(T)[9])*sin(x(T)[8]))*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            -L1*sin(x(T)[1])   -L2*sin(x(T)[2])   0.0   0.0   0.0   0.0   -(L1*sin(x(T)[7]))*0.5   (sqrt(3)*L2*cos(x(T)[8])*sin(x(T)[9]))*0.5-(L2*sin(x(T)[8]))*0.5   (sqrt(3)*L2*cos(x(T)[9])*sin(x(T)[8]))*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            L1*cos(x(T)[1])   L2*cos(x(T)[2])*cos(x(T)[3])   -L2*sin(x(T)[2])*sin(x(T)[3])   0.0   0.0   0.0   -L1*cos(x(T)[7])   -L2*cos(x(T)[8])*cos(x(T)[9])   L2*sin(x(T)[8])*sin(x(T)[9])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   L2*cos(x(T)[2])*cos(x(T)[3])*x(T)[12]-L2*sin(x(T)[2])*sin(x(T)[3])*x(T)[11]   L2*cos(x(T)[2])*cos(x(T)[3])*x(T)[11]-L2*sin(x(T)[2])*sin(x(T)[3])*x(T)[12]   -(sqrt(3)*L1*cos(x(T)[4])*x(T)[13])*0.5   (L2*cos(x(T)[5])*cos(x(T)[6])*x(T)[15])*0.5-(sqrt(3)*L2*cos(x(T)[5])*x(T)[14])*0.5-(L2*sin(x(T)[5])*sin(x(T)[6])*x(T)[14])*0.5   (L2*cos(x(T)[5])*cos(x(T)[6])*x(T)[14])*0.5-(L2*sin(x(T)[5])*sin(x(T)[6])*x(T)[15])*0.5   0.0   0.0   0.0   0.0   L2*cos(x(T)[2])*sin(x(T)[3])   L2*cos(x(T)[3])*sin(x(T)[2])   -(sqrt(3)*L1*sin(x(T)[4]))*0.5   (L2*cos(x(T)[5])*sin(x(T)[6]))*0.5-(sqrt(3)*L2*sin(x(T)[5]))*0.5   (L2*cos(x(T)[6])*sin(x(T)[5]))*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            -L1*cos(x(T)[1])*x(T)[10]   -L2*cos(x(T)[2])*x(T)[11]   0.0   -(L1*cos(x(T)[4])*x(T)[13])*0.5   (sqrt(3)*L2*sin(x(T)[5])*sin(x(T)[6])*x(T)[14])*0.5-(sqrt(3)*L2*cos(x(T)[5])*cos(x(T)[6])*x(T)[15])*0.5-(L2*cos(x(T)[5])*x(T)[14])*0.5   (sqrt(3)*L2*sin(x(T)[5])*sin(x(T)[6])*x(T)[15])*0.5-(sqrt(3)*L2*cos(x(T)[5])*cos(x(T)[6])*x(T)[14])*0.5   0.0   0.0   0.0   -L1*sin(x(T)[1])   -L2*sin(x(T)[2])   0.0   -(L1*sin(x(T)[4]))*0.5   -(L2*sin(x(T)[5]))*0.5-(sqrt(3)*L2*cos(x(T)[5])*sin(x(T)[6]))*0.5   -(sqrt(3)*L2*cos(x(T)[6])*sin(x(T)[5]))*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            -L1*sin(x(T)[1])*x(T)[10]   -L2*cos(x(T)[3])*sin(x(T)[2])*x(T)[11]-L2*cos(x(T)[2])*sin(x(T)[3])*x(T)[12]   -L2*cos(x(T)[2])*sin(x(T)[3])*x(T)[11]-L2*cos(x(T)[3])*sin(x(T)[2])*x(T)[12]   L1*sin(x(T)[4])*x(T)[13]   L2*cos(x(T)[6])*sin(x(T)[5])*x(T)[14]+L2*cos(x(T)[5])*sin(x(T)[6])*x(T)[15]   L2*cos(x(T)[5])*sin(x(T)[6])*x(T)[14]+L2*cos(x(T)[6])*sin(x(T)[5])*x(T)[15]   0.0   0.0   0.0   L1*cos(x(T)[1])   L2*cos(x(T)[2])*cos(x(T)[3])   -L2*sin(x(T)[2])*sin(x(T)[3])   -L1*cos(x(T)[4])   -L2*cos(x(T)[5])*cos(x(T)[6])   L2*sin(x(T)[5])*sin(x(T)[6])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   L2*cos(x(T)[2])*cos(x(T)[3])*x(T)[12]-L2*sin(x(T)[2])*sin(x(T)[3])*x(T)[11]   L2*cos(x(T)[2])*cos(x(T)[3])*x(T)[11]-L2*sin(x(T)[2])*sin(x(T)[3])*x(T)[12]   0.0   0.0   0.0   (sqrt(3)*L1*cos(x(T)[7])*x(T)[16])*0.5   (L2*cos(x(T)[8])*cos(x(T)[9])*x(T)[18])*0.5+(sqrt(3)*L2*cos(x(T)[8])*x(T)[17])*0.5-(L2*sin(x(T)[8])*sin(x(T)[9])*x(T)[17])*0.5   (L2*cos(x(T)[8])*cos(x(T)[9])*x(T)[17])*0.5-(L2*sin(x(T)[8])*sin(x(T)[9])*x(T)[18])*0.5   0.0   L2*cos(x(T)[2])*sin(x(T)[3])   L2*cos(x(T)[3])*sin(x(T)[2])   0.0   0.0   0.0   (sqrt(3)*L1*sin(x(T)[7]))*0.5   (L2*cos(x(T)[8])*sin(x(T)[9]))*0.5+(sqrt(3)*L2*sin(x(T)[8]))*0.5   (L2*cos(x(T)[9])*sin(x(T)[8]))*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            -L1*cos(x(T)[1])*x(T)[10]   -L2*cos(x(T)[2])*x(T)[11]   0.0   0.0   0.0   0.0   -(L1*cos(x(T)[7])*x(T)[16])*0.5   (sqrt(3)*L2*cos(x(T)[8])*cos(x(T)[9])*x(T)[18])*0.5-(L2*cos(x(T)[8])*x(T)[17])*0.5-(sqrt(3)*L2*sin(x(T)[8])*sin(x(T)[9])*x(T)[17])*0.5   (sqrt(3)*L2*cos(x(T)[8])*cos(x(T)[9])*x(T)[17])*0.5-(sqrt(3)*L2*sin(x(T)[8])*sin(x(T)[9])*x(T)[18])*0.5   -L1*sin(x(T)[1])   -L2*sin(x(T)[2])   0.0   0.0   0.0   0.0   -(L1*sin(x(T)[7]))*0.5   (sqrt(3)*L2*cos(x(T)[8])*sin(x(T)[9]))*0.5-(L2*sin(x(T)[8]))*0.5   (sqrt(3)*L2*cos(x(T)[9])*sin(x(T)[8]))*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            -L1*sin(x(T)[1])*x(T)[10]   -L2*cos(x(T)[3])*sin(x(T)[2])*x(T)[11]-L2*cos(x(T)[2])*sin(x(T)[3])*x(T)[12]   -L2*cos(x(T)[2])*sin(x(T)[3])*x(T)[11]-L2*cos(x(T)[3])*sin(x(T)[2])*x(T)[12]   0.0   0.0   0.0   L1*sin(x(T)[7])*x(T)[16]   L2*cos(x(T)[9])*sin(x(T)[8])*x(T)[17]+L2*cos(x(T)[8])*sin(x(T)[9])*x(T)[18]   L2*cos(x(T)[8])*sin(x(T)[9])*x(T)[17]+L2*cos(x(T)[9])*sin(x(T)[8])*x(T)[18]   L1*cos(x(T)[1])   L2*cos(x(T)[2])*cos(x(T)[3])   -L2*sin(x(T)[2])*sin(x(T)[3])   0.0   0.0   0.0   -L1*cos(x(T)[7])   -L2*cos(x(T)[8])*cos(x(T)[9])   L2*sin(x(T)[8])*sin(x(T)[9])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   -L2*cos(x(T)[2])*sin(x(T)[3])   -L2*cos(x(T)[3])*sin(x(T)[2])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0
            L1*sin(x(T)[1])   L2*sin(x(T)[2])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0
            -L1*cos(x(T)[1])   -L2*cos(x(T)[2])*cos(x(T)[3])   L2*sin(x(T)[2])*sin(x(T)[3])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0]
        Fdx = t -> [ # TODO: Should we use sparse matrices? Is that worthwile? Might improve efficiency
            1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*sin(x(t)[1])   L1*cos(x(t)[1])   0.0   -L1*sin(x(t)[1])   L1*cos(x(t)[1])   0.0   0.0   0.0
            0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L2*cos(x(t)[2])*sin(x(t)[3])   -L2*sin(x(t)[2])   L2*cos(x(t)[2])*cos(x(t)[3])   L2*cos(x(t)[2])*sin(x(t)[3])   -L2*sin(x(t)[2])   L2*cos(x(t)[2])*cos(x(t)[3])   0.0   0.0   0.0
            0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L2*cos(x(t)[3])*sin(x(t)[2])   0.0   -L2*sin(x(t)[2])*sin(x(t)[3])   L2*cos(x(t)[3])*sin(x(t)[2])   0.0   -L2*sin(x(t)[2])*sin(x(t)[3])   0.0   0.0   0.0
            0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -(sqrt(3)*L1*sin(x(t)[4]))*0.5   -(L1*sin(x(t)[4]))*0.5   -L1*cos(x(t)[4])   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x(t)[5])*sin(x(t)[6]))*0.5-(sqrt(3)*L2*sin(x(t)[5]))*0.5   -(L2*sin(x(t)[5]))*0.5-(sqrt(3)*L2*cos(x(t)[5])*sin(x(t)[6]))*0.5   -L2*cos(x(t)[5])*cos(x(t)[6])   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x(t)[6])*sin(x(t)[5]))*0.5   -(sqrt(3)*L2*cos(x(t)[6])*sin(x(t)[5]))*0.5   L2*sin(x(t)[5])*sin(x(t)[6])   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (sqrt(3)*L1*sin(x(t)[7]))*0.5   -(L1*sin(x(t)[7]))*0.5   -L1*cos(x(t)[7])   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x(t)[8])*sin(x(t)[9]))*0.5+(sqrt(3)*L2*sin(x(t)[8]))*0.5   (sqrt(3)*L2*cos(x(t)[8])*sin(x(t)[9]))*0.5-(L2*sin(x(t)[8]))*0.5   -L2*cos(x(t)[8])*cos(x(t)[9])   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x(t)[9])*sin(x(t)[8]))*0.5   (sqrt(3)*L2*cos(x(t)[9])*sin(x(t)[8]))*0.5   L2*sin(x(t)[8])*sin(x(t)[9])   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(x(t)[1])*sin(x(t)[2])+cos(x(t)[1])*cos(x(t)[2])*cos(x(t)[3]))   -L1*cos(x(t)[1])*sin(x(t)[2])*sin(x(t)[3])*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*sin(x(t)[1])   -L1*cos(x(t)[1])   0.0   L1*sin(x(t)[1])   -L1*cos(x(t)[1])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*(L2*M3+LC2*M2)*(sin(x(t)[1])*sin(x(t)[2])+cos(x(t)[1])*cos(x(t)[2])*cos(x(t)[3]))   J2+L2^2*M3+LC2^2*M2   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L2*cos(x(t)[2])*sin(x(t)[3])   L2*sin(x(t)[2])   -L2*cos(x(t)[2])*cos(x(t)[3])   -L2*cos(x(t)[2])*sin(x(t)[3])   L2*sin(x(t)[2])   -L2*cos(x(t)[2])*cos(x(t)[3])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*cos(x(t)[1])*sin(x(t)[2])*sin(x(t)[3])*(L2*M3+LC2*M2)   0.0   sin(x(t)[2])^2*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   -L2*cos(x(t)[3])*sin(x(t)[2])   0.0   L2*sin(x(t)[2])*sin(x(t)[3])   -L2*cos(x(t)[3])*sin(x(t)[2])   0.0   L2*sin(x(t)[2])*sin(x(t)[3])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(x(t)[4])*sin(x(t)[5])+cos(x(t)[4])*cos(x(t)[5])*cos(x(t)[6]))   -L1*cos(x(t)[4])*sin(x(t)[5])*sin(x(t)[6])*(L2*M3+LC2*M2)   0.0   0.0   0.0   (sqrt(3)*L1*sin(x(t)[4]))*0.5   (L1*sin(x(t)[4]))*0.5   L1*cos(x(t)[4])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*(L2*M3+LC2*M2)*(sin(x(t)[4])*sin(x(t)[5])+cos(x(t)[4])*cos(x(t)[5])*cos(x(t)[6]))   J2+L2^2*M3+LC2^2*M2   0.0   0.0   0.0   0.0   (sqrt(3)*L2*sin(x(t)[5]))*0.5-(L2*cos(x(t)[5])*sin(x(t)[6]))*0.5   (L2*sin(x(t)[5]))*0.5+(sqrt(3)*L2*cos(x(t)[5])*sin(x(t)[6]))*0.5   L2*cos(x(t)[5])*cos(x(t)[6])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*cos(x(t)[4])*sin(x(t)[5])*sin(x(t)[6])*(L2*M3+LC2*M2)   0.0   sin(x(t)[5])^2*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   -(L2*cos(x(t)[6])*sin(x(t)[5]))*0.5   (sqrt(3)*L2*cos(x(t)[6])*sin(x(t)[5]))*0.5   -L2*sin(x(t)[5])*sin(x(t)[6])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(x(t)[7])*sin(x(t)[8])+cos(x(t)[7])*cos(x(t)[8])*cos(x(t)[9]))   -L1*cos(x(t)[7])*sin(x(t)[8])*sin(x(t)[9])*(L2*M3+LC2*M2)   0.0   0.0   0.0   -(sqrt(3)*L1*sin(x(t)[7]))*0.5   (L1*sin(x(t)[7]))*0.5   L1*cos(x(t)[7])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*(L2*M3+LC2*M2)*(sin(x(t)[7])*sin(x(t)[8])+cos(x(t)[7])*cos(x(t)[8])*cos(x(t)[9]))   J2+L2^2*M3+LC2^2*M2   0.0   0.0   0.0   0.0   -(L2*cos(x(t)[8])*sin(x(t)[9]))*0.5-(sqrt(3)*L2*sin(x(t)[8]))*0.5   (L2*sin(x(t)[8]))*0.5-(sqrt(3)*L2*cos(x(t)[8])*sin(x(t)[9]))*0.5   L2*cos(x(t)[8])*cos(x(t)[9])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*cos(x(t)[7])*sin(x(t)[8])*sin(x(t)[9])*(L2*M3+LC2*M2)   0.0   sin(x(t)[8])^2*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   -(L2*cos(x(t)[9])*sin(x(t)[8]))*0.5   -(sqrt(3)*L2*cos(x(t)[9])*sin(x(t)[8]))*0.5   -L2*sin(x(t)[8])*sin(x(t)[9])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0]
        Fddx = [
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*cos(x(T)[1])*dx(T)[1]   -L1*sin(x(T)[1])*dx(T)[1]   0.0   -L1*cos(x(T)[1])*dx(T)[1]   -L1*sin(x(T)[1])*dx(T)[1]   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L2*cos(x(T)[2])*cos(x(T)[3])*dx(T)[3]-L2*sin(x(T)[2])*sin(x(T)[3])*dx(T)[2]   -L2*cos(x(T)[2])*dx(T)[2]   -L2*cos(x(T)[3])*sin(x(T)[2])*dx(T)[2]-L2*cos(x(T)[2])*sin(x(T)[3])*dx(T)[3]   L2*cos(x(T)[2])*cos(x(T)[3])*dx(T)[3]-L2*sin(x(T)[2])*sin(x(T)[3])*dx(T)[2]   -L2*cos(x(T)[2])*dx(T)[2]   -L2*cos(x(T)[3])*sin(x(T)[2])*dx(T)[2]-L2*cos(x(T)[2])*sin(x(T)[3])*dx(T)[3]   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L2*cos(x(T)[2])*cos(x(T)[3])*dx(T)[2]-L2*sin(x(T)[2])*sin(x(T)[3])*dx(T)[3]   0.0   -L2*cos(x(T)[2])*sin(x(T)[3])*dx(T)[2]-L2*cos(x(T)[3])*sin(x(T)[2])*dx(T)[3]   L2*cos(x(T)[2])*cos(x(T)[3])*dx(T)[2]-L2*sin(x(T)[2])*sin(x(T)[3])*dx(T)[3]   0.0   -L2*cos(x(T)[2])*sin(x(T)[3])*dx(T)[2]-L2*cos(x(T)[3])*sin(x(T)[2])*dx(T)[3]   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -(sqrt(3)*L1*cos(x(T)[4])*dx(T)[4])*0.5   -(L1*cos(x(T)[4])*dx(T)[4])*0.5   L1*sin(x(T)[4])*dx(T)[4]   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x(T)[5])*cos(x(T)[6])*dx(T)[6])*0.5-(sqrt(3)*L2*cos(x(T)[5])*dx(T)[5])*0.5-(L2*sin(x(T)[5])*sin(x(T)[6])*dx(T)[5])*0.5   (sqrt(3)*L2*sin(x(T)[5])*sin(x(T)[6])*dx(T)[5])*0.5-(sqrt(3)*L2*cos(x(T)[5])*cos(x(T)[6])*dx(T)[6])*0.5-(L2*cos(x(T)[5])*dx(T)[5])*0.5   L2*cos(x(T)[6])*sin(x(T)[5])*dx(T)[5]+L2*cos(x(T)[5])*sin(x(T)[6])*dx(T)[6]   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x(T)[5])*cos(x(T)[6])*dx(T)[5])*0.5-(L2*sin(x(T)[5])*sin(x(T)[6])*dx(T)[6])*0.5   (sqrt(3)*L2*sin(x(T)[5])*sin(x(T)[6])*dx(T)[6])*0.5-(sqrt(3)*L2*cos(x(T)[5])*cos(x(T)[6])*dx(T)[5])*0.5   L2*cos(x(T)[5])*sin(x(T)[6])*dx(T)[5]+L2*cos(x(T)[6])*sin(x(T)[5])*dx(T)[6]   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (sqrt(3)*L1*cos(x(T)[7])*dx(T)[7])*0.5   -(L1*cos(x(T)[7])*dx(T)[7])*0.5   L1*sin(x(T)[7])*dx(T)[7]   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x(T)[8])*cos(x(T)[9])*dx(T)[9])*0.5+(sqrt(3)*L2*cos(x(T)[8])*dx(T)[8])*0.5-(L2*sin(x(T)[8])*sin(x(T)[9])*dx(T)[8])*0.5   (sqrt(3)*L2*cos(x(T)[8])*cos(x(T)[9])*dx(T)[9])*0.5-(L2*cos(x(T)[8])*dx(T)[8])*0.5-(sqrt(3)*L2*sin(x(T)[8])*sin(x(T)[9])*dx(T)[8])*0.5   L2*cos(x(T)[9])*sin(x(T)[8])*dx(T)[8]+L2*cos(x(T)[8])*sin(x(T)[9])*dx(T)[9]   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x(T)[8])*cos(x(T)[9])*dx(T)[8])*0.5-(L2*sin(x(T)[8])*sin(x(T)[9])*dx(T)[9])*0.5   (sqrt(3)*L2*cos(x(T)[8])*cos(x(T)[9])*dx(T)[8])*0.5-(sqrt(3)*L2*sin(x(T)[8])*sin(x(T)[9])*dx(T)[9])*0.5   L2*cos(x(T)[8])*sin(x(T)[9])*dx(T)[8]+L2*cos(x(T)[9])*sin(x(T)[8])*dx(T)[9]   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*(L2*M3+LC2*M2)*(cos(x(T)[2])*cos(x(T)[3])*sin(x(T)[1])*dx(T)[1]-cos(x(T)[2])*sin(x(T)[1])*dx(T)[2]-cos(x(T)[1])*sin(x(T)[2])*dx(T)[1]+cos(x(T)[1])*cos(x(T)[3])*sin(x(T)[2])*dx(T)[2]+cos(x(T)[1])*cos(x(T)[2])*sin(x(T)[3])*dx(T)[3])   L1*sin(x(T)[1])*sin(x(T)[2])*sin(x(T)[3])*dx(T)[1]*(L2*M3+LC2*M2)-L1*cos(x(T)[1])*cos(x(T)[3])*sin(x(T)[2])*dx(T)[3]*(L2*M3+LC2*M2)-L1*cos(x(T)[1])*cos(x(T)[2])*sin(x(T)[3])*dx(T)[2]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*cos(x(T)[1])*dx(T)[1]   L1*sin(x(T)[1])*dx(T)[1]   0.0   L1*cos(x(T)[1])*dx(T)[1]   L1*sin(x(T)[1])*dx(T)[1]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*(L2*M3+LC2*M2)*(cos(x(T)[2])*cos(x(T)[3])*sin(x(T)[1])*dx(T)[1]-cos(x(T)[2])*sin(x(T)[1])*dx(T)[2]-cos(x(T)[1])*sin(x(T)[2])*dx(T)[1]+cos(x(T)[1])*cos(x(T)[3])*sin(x(T)[2])*dx(T)[2]+cos(x(T)[1])*cos(x(T)[2])*sin(x(T)[3])*dx(T)[3])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L2*sin(x(T)[2])*sin(x(T)[3])*dx(T)[2]-L2*cos(x(T)[2])*cos(x(T)[3])*dx(T)[3]   L2*cos(x(T)[2])*dx(T)[2]   L2*cos(x(T)[3])*sin(x(T)[2])*dx(T)[2]+L2*cos(x(T)[2])*sin(x(T)[3])*dx(T)[3]   L2*sin(x(T)[2])*sin(x(T)[3])*dx(T)[2]-L2*cos(x(T)[2])*cos(x(T)[3])*dx(T)[3]   L2*cos(x(T)[2])*dx(T)[2]   L2*cos(x(T)[3])*sin(x(T)[2])*dx(T)[2]+L2*cos(x(T)[2])*sin(x(T)[3])*dx(T)[3]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*sin(x(T)[1])*sin(x(T)[2])*sin(x(T)[3])*dx(T)[1]*(L2*M3+LC2*M2)-L1*cos(x(T)[1])*cos(x(T)[3])*sin(x(T)[2])*dx(T)[3]*(L2*M3+LC2*M2)-L1*cos(x(T)[1])*cos(x(T)[2])*sin(x(T)[3])*dx(T)[2]*(L2*M3+LC2*M2)   0.0   2*cos(x(T)[2])*sin(x(T)[2])*dx(T)[2]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   L2*sin(x(T)[2])*sin(x(T)[3])*dx(T)[3]-L2*cos(x(T)[2])*cos(x(T)[3])*dx(T)[2]   0.0   L2*cos(x(T)[2])*sin(x(T)[3])*dx(T)[2]+L2*cos(x(T)[3])*sin(x(T)[2])*dx(T)[3]   L2*sin(x(T)[2])*sin(x(T)[3])*dx(T)[3]-L2*cos(x(T)[2])*cos(x(T)[3])*dx(T)[2]   0.0   L2*cos(x(T)[2])*sin(x(T)[3])*dx(T)[2]+L2*cos(x(T)[3])*sin(x(T)[2])*dx(T)[3]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*(L2*M3+LC2*M2)*(cos(x(T)[5])*cos(x(T)[6])*sin(x(T)[4])*dx(T)[4]-cos(x(T)[5])*sin(x(T)[4])*dx(T)[5]-cos(x(T)[4])*sin(x(T)[5])*dx(T)[4]+cos(x(T)[4])*cos(x(T)[6])*sin(x(T)[5])*dx(T)[5]+cos(x(T)[4])*cos(x(T)[5])*sin(x(T)[6])*dx(T)[6])   L1*sin(x(T)[4])*sin(x(T)[5])*sin(x(T)[6])*dx(T)[4]*(L2*M3+LC2*M2)-L1*cos(x(T)[4])*cos(x(T)[6])*sin(x(T)[5])*dx(T)[6]*(L2*M3+LC2*M2)-L1*cos(x(T)[4])*cos(x(T)[5])*sin(x(T)[6])*dx(T)[5]*(L2*M3+LC2*M2)   0.0   0.0   0.0   (sqrt(3)*L1*cos(x(T)[4])*dx(T)[4])*0.5   (L1*cos(x(T)[4])*dx(T)[4])*0.5   -L1*sin(x(T)[4])*dx(T)[4]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*(L2*M3+LC2*M2)*(cos(x(T)[5])*cos(x(T)[6])*sin(x(T)[4])*dx(T)[4]-cos(x(T)[5])*sin(x(T)[4])*dx(T)[5]-cos(x(T)[4])*sin(x(T)[5])*dx(T)[4]+cos(x(T)[4])*cos(x(T)[6])*sin(x(T)[5])*dx(T)[5]+cos(x(T)[4])*cos(x(T)[5])*sin(x(T)[6])*dx(T)[6])   0.0   0.0   0.0   0.0   0.0   (sqrt(3)*L2*cos(x(T)[5])*dx(T)[5])*0.5-(L2*cos(x(T)[5])*cos(x(T)[6])*dx(T)[6])*0.5+(L2*sin(x(T)[5])*sin(x(T)[6])*dx(T)[5])*0.5   (L2*cos(x(T)[5])*dx(T)[5])*0.5+(sqrt(3)*L2*cos(x(T)[5])*cos(x(T)[6])*dx(T)[6])*0.5-(sqrt(3)*L2*sin(x(T)[5])*sin(x(T)[6])*dx(T)[5])*0.5   -L2*cos(x(T)[6])*sin(x(T)[5])*dx(T)[5]-L2*cos(x(T)[5])*sin(x(T)[6])*dx(T)[6]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*sin(x(T)[4])*sin(x(T)[5])*sin(x(T)[6])*dx(T)[4]*(L2*M3+LC2*M2)-L1*cos(x(T)[4])*cos(x(T)[6])*sin(x(T)[5])*dx(T)[6]*(L2*M3+LC2*M2)-L1*cos(x(T)[4])*cos(x(T)[5])*sin(x(T)[6])*dx(T)[5]*(L2*M3+LC2*M2)   0.0   2*cos(x(T)[5])*sin(x(T)[5])*dx(T)[5]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   (L2*sin(x(T)[5])*sin(x(T)[6])*dx(T)[6])*0.5-(L2*cos(x(T)[5])*cos(x(T)[6])*dx(T)[5])*0.5   (sqrt(3)*L2*cos(x(T)[5])*cos(x(T)[6])*dx(T)[5])*0.5-(sqrt(3)*L2*sin(x(T)[5])*sin(x(T)[6])*dx(T)[6])*0.5   -L2*cos(x(T)[5])*sin(x(T)[6])*dx(T)[5]-L2*cos(x(T)[6])*sin(x(T)[5])*dx(T)[6]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*(L2*M3+LC2*M2)*(cos(x(T)[8])*cos(x(T)[9])*sin(x(T)[7])*dx(T)[7]-cos(x(T)[8])*sin(x(T)[7])*dx(T)[8]-cos(x(T)[7])*sin(x(T)[8])*dx(T)[7]+cos(x(T)[7])*cos(x(T)[9])*sin(x(T)[8])*dx(T)[8]+cos(x(T)[7])*cos(x(T)[8])*sin(x(T)[9])*dx(T)[9])   L1*sin(x(T)[7])*sin(x(T)[8])*sin(x(T)[9])*dx(T)[7]*(L2*M3+LC2*M2)-L1*cos(x(T)[7])*cos(x(T)[9])*sin(x(T)[8])*dx(T)[9]*(L2*M3+LC2*M2)-L1*cos(x(T)[7])*cos(x(T)[8])*sin(x(T)[9])*dx(T)[8]*(L2*M3+LC2*M2)   0.0   0.0   0.0   -(sqrt(3)*L1*cos(x(T)[7])*dx(T)[7])*0.5   (L1*cos(x(T)[7])*dx(T)[7])*0.5   -L1*sin(x(T)[7])*dx(T)[7]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*(L2*M3+LC2*M2)*(cos(x(T)[8])*cos(x(T)[9])*sin(x(T)[7])*dx(T)[7]-cos(x(T)[8])*sin(x(T)[7])*dx(T)[8]-cos(x(T)[7])*sin(x(T)[8])*dx(T)[7]+cos(x(T)[7])*cos(x(T)[9])*sin(x(T)[8])*dx(T)[8]+cos(x(T)[7])*cos(x(T)[8])*sin(x(T)[9])*dx(T)[9])   0.0   0.0   0.0   0.0   0.0   (L2*sin(x(T)[8])*sin(x(T)[9])*dx(T)[8])*0.5-(sqrt(3)*L2*cos(x(T)[8])*dx(T)[8])*0.5-(L2*cos(x(T)[8])*cos(x(T)[9])*dx(T)[9])*0.5   (L2*cos(x(T)[8])*dx(T)[8])*0.5-(sqrt(3)*L2*cos(x(T)[8])*cos(x(T)[9])*dx(T)[9])*0.5+(sqrt(3)*L2*sin(x(T)[8])*sin(x(T)[9])*dx(T)[8])*0.5   -L2*cos(x(T)[9])*sin(x(T)[8])*dx(T)[8]-L2*cos(x(T)[8])*sin(x(T)[9])*dx(T)[9]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*sin(x(T)[7])*sin(x(T)[8])*sin(x(T)[9])*dx(T)[7]*(L2*M3+LC2*M2)-L1*cos(x(T)[7])*cos(x(T)[9])*sin(x(T)[8])*dx(T)[9]*(L2*M3+LC2*M2)-L1*cos(x(T)[7])*cos(x(T)[8])*sin(x(T)[9])*dx(T)[8]*(L2*M3+LC2*M2)   0.0   2*cos(x(T)[8])*sin(x(T)[8])*dx(T)[8]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   (L2*sin(x(T)[8])*sin(x(T)[9])*dx(T)[9])*0.5-(L2*cos(x(T)[8])*cos(x(T)[9])*dx(T)[8])*0.5   (sqrt(3)*L2*sin(x(T)[8])*sin(x(T)[9])*dx(T)[9])*0.5-(sqrt(3)*L2*cos(x(T)[8])*cos(x(T)[9])*dx(T)[8])*0.5   -L2*cos(x(T)[8])*sin(x(T)[9])*dx(T)[8]-L2*cos(x(T)[9])*sin(x(T)[8])*dx(T)[9]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0]
        Fp = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; x(T)[10]; x(T)[11]; x(T)[12]; x(T)[13]; x(T)[14]; x(T)[15]; x(T)[16]; x(T)[17]; x(T)[18]; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
        gₓ = [0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -(2*y(T)[1]-2*x2(T)[31])/T   -(2*y(T)[2]-2*x2(T)[32])/T   -(2*y(T)[3]-2*x2(T)[33])/T]
        gdₓ = [0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -(2*dy(T)[1]-2*dx2(T)[31])/T   -(2*dy(T)[2]-2*dx2(T)[32])/T   -(2*dy(T)[3]-2*dx2(T)[33])/T]

        # NOTE: Convention is used that derivatives wrt to θ stack by adding new cols
        # while derivatives wrt to x stack by adding new rows

        ###################### INITIALIZING ADJOINT SYSTEM ####################
        # Indices 1-18 are differential (d), while 19-33 are algebraic (a)
        dinds = 1:18
        ainds = 19:33
        λT  = zeros(33)
        dλT = zeros(33)
        temp = (-gₓ)/vcat(Fdx(T)[dinds,:], -Fx[ainds,:])
        λT[dinds] = zeros(length(dinds))
        λT[ainds] = temp[ainds]
        dλT[dinds] = temp[dinds]
        temp = (-gdₓ + (dλT[dinds]')*(Fx[dinds,:] - Fddx[dinds,:] - Fdx(T)[dinds,:]) + (λT[ainds]')*Fx[ainds,:])/vcat(Fdx(T)[dinds,:], -Fx[ainds,:])
        dλT[ainds] = temp[ainds]

        function f!(res, dz, z, _, t)
            # ut = u(t)
            # wt = w(t)
            xt = x(t)
            dxt = dx(t)
            yt = y(t)

            # Adjoint equations
            res[1] = dz[1]-z[11]*(L1*dxt[10]*(L2*M3+LC2*M2)*(cos(xt[1])*sin(xt[2])-cos(xt[2])*cos(xt[3])*sin(xt[1]))-L1*xt[10]^2*(L2*M3+LC2*M2)*(sin(xt[1])*sin(xt[2])+cos(xt[1])*cos(xt[2])*cos(xt[3])))-z[10]*(0.0*sin(xt[1])*(L1*(M2+M3)+LC1*M1)+L1*cos(xt[1])*dxt[20]+L1*cos(xt[1])*dxt[23]+L1*sin(xt[1])*dxt[21]+L1*sin(xt[1])*dxt[24]+L1*dxt[11]*(L2*M3+LC2*M2)*(cos(xt[1])*sin(xt[2])-cos(xt[2])*cos(xt[3])*sin(xt[1]))+L1*xt[11]^2*(L2*M3+LC2*M2)*(cos(xt[1])*cos(xt[2])+cos(xt[3])*sin(xt[1])*sin(xt[2]))+L1*cos(xt[3])*sin(xt[1])*sin(xt[2])*xt[12]^2*(L2*M3+LC2*M2)+L1*sin(xt[1])*sin(xt[2])*sin(xt[3])*dxt[12]*(L2*M3+LC2*M2)+2*L1*cos(xt[2])*sin(xt[1])*sin(xt[3])*xt[11]*xt[12]*(L2*M3+LC2*M2))-z[12]*(L1*cos(xt[1])*sin(xt[2])*sin(xt[3])*xt[10]^2*(L2*M3+LC2*M2)+L1*sin(xt[1])*sin(xt[2])*sin(xt[3])*dxt[10]*(L2*M3+LC2*M2))+z[1]*(L1*cos(xt[1])*dxt[26]+L1*cos(xt[1])*dxt[29]+L1*sin(xt[1])*dxt[27]+L1*sin(xt[1])*dxt[30])-L1*cos(xt[1])*z[21]-L1*cos(xt[1])*z[24]+L1*cos(xt[1])*z[33]+L1*sin(xt[1])*z[20]+L1*sin(xt[1])*z[23]-L1*sin(xt[1])*z[32]+L1*cos(xt[1])*z[26]*xt[10]+L1*cos(xt[1])*z[29]*xt[10]+L1*sin(xt[1])*z[27]*xt[10]+L1*sin(xt[1])*z[30]*xt[10]
            res[2] = dz[2]-z[25]*(L2*cos(xt[2])*cos(xt[3])*xt[12]-L2*sin(xt[2])*sin(xt[3])*xt[11])+z[27]*(L2*cos(xt[3])*sin(xt[2])*xt[11]+L2*cos(xt[2])*sin(xt[3])*xt[12])-z[28]*(L2*cos(xt[2])*cos(xt[3])*xt[12]-L2*sin(xt[2])*sin(xt[3])*xt[11])+z[30]*(L2*cos(xt[3])*sin(xt[2])*xt[11]+L2*cos(xt[2])*sin(xt[3])*xt[12])-z[3]*(L2*cos(xt[2])*cos(xt[3])*dxt[25]+L2*cos(xt[2])*cos(xt[3])*dxt[28]-L2*cos(xt[2])*sin(xt[3])*dxt[27]-L2*cos(xt[2])*sin(xt[3])*dxt[30])-z[12]*(L2*cos(xt[2])*sin(xt[3])*dxt[21]-L2*cos(xt[2])*cos(xt[3])*dxt[22]-L2*cos(xt[2])*cos(xt[3])*dxt[19]+L2*cos(xt[2])*sin(xt[3])*dxt[24]+2*cos(xt[2])*sin(xt[2])*dxt[12]*(J2+L2^2*M3+LC2^2*M2)+2*cos(2*xt[2])*xt[11]*xt[12]*(J2+L2^2*M3+LC2^2*M2)+0.0*cos(xt[2])*sin(xt[3])*(L2*M3+LC2*M2)+L1*cos(xt[2])*sin(xt[1])*sin(xt[3])*xt[10]^2*(L2*M3+LC2*M2)-L1*cos(xt[1])*cos(xt[2])*sin(xt[3])*dxt[10]*(L2*M3+LC2*M2))-z[11]*(sin(xt[2])^2*xt[12]^2*(J2+L2^2*M3+LC2^2*M2)-cos(xt[2])^2*xt[12]^2*(J2+L2^2*M3+LC2^2*M2)+L2*cos(xt[2])*dxt[20]+L2*cos(xt[2])*dxt[23]+L2*cos(xt[3])*sin(xt[2])*dxt[21]+L2*cos(xt[3])*sin(xt[2])*dxt[24]+L2*sin(xt[2])*sin(xt[3])*dxt[19]+L2*sin(xt[2])*sin(xt[3])*dxt[22]+L1*dxt[10]*(L2*M3+LC2*M2)*(cos(xt[2])*sin(xt[1])-cos(xt[1])*cos(xt[3])*sin(xt[2]))+0.0*cos(xt[3])*sin(xt[2])*(L2*M3+LC2*M2)+L1*xt[10]^2*(L2*M3+LC2*M2)*(cos(xt[1])*cos(xt[2])+cos(xt[3])*sin(xt[1])*sin(xt[2])))+z[10]*(L1*xt[11]^2*(L2*M3+LC2*M2)*(sin(xt[1])*sin(xt[2])+cos(xt[1])*cos(xt[2])*cos(xt[3]))-L1*dxt[11]*(L2*M3+LC2*M2)*(cos(xt[2])*sin(xt[1])-cos(xt[1])*cos(xt[3])*sin(xt[2]))+L1*cos(xt[1])*cos(xt[2])*cos(xt[3])*xt[12]^2*(L2*M3+LC2*M2)+L1*cos(xt[1])*cos(xt[2])*sin(xt[3])*dxt[12]*(L2*M3+LC2*M2)-2*L1*cos(xt[1])*sin(xt[2])*sin(xt[3])*xt[11]*xt[12]*(L2*M3+LC2*M2))+z[2]*(L2*cos(xt[2])*dxt[26]+L2*cos(xt[2])*dxt[29]+L2*cos(xt[3])*sin(xt[2])*dxt[27]+L2*cos(xt[3])*sin(xt[2])*dxt[30]+L2*sin(xt[2])*sin(xt[3])*dxt[25]+L2*sin(xt[2])*sin(xt[3])*dxt[28])+L2*sin(xt[2])*z[20]+L2*sin(xt[2])*z[23]-L2*sin(xt[2])*z[32]-L2*cos(xt[2])*cos(xt[3])*z[21]-L2*cos(xt[2])*cos(xt[3])*z[24]+L2*cos(xt[2])*cos(xt[3])*z[33]-L2*cos(xt[2])*sin(xt[3])*z[19]-L2*cos(xt[2])*sin(xt[3])*z[22]+L2*cos(xt[2])*sin(xt[3])*z[31]+L2*cos(xt[2])*z[26]*xt[11]+L2*cos(xt[2])*z[29]*xt[11]
            res[3] = dz[3]-z[12]*(L2*cos(xt[3])*sin(xt[2])*dxt[21]+L2*cos(xt[3])*sin(xt[2])*dxt[24]+L2*sin(xt[2])*sin(xt[3])*dxt[19]+L2*sin(xt[2])*sin(xt[3])*dxt[22]+0.0*cos(xt[3])*sin(xt[2])*(L2*M3+LC2*M2)+L1*cos(xt[3])*sin(xt[1])*sin(xt[2])*xt[10]^2*(L2*M3+LC2*M2)-L1*cos(xt[1])*cos(xt[3])*sin(xt[2])*dxt[10]*(L2*M3+LC2*M2))-z[25]*(L2*cos(xt[2])*cos(xt[3])*xt[11]-L2*sin(xt[2])*sin(xt[3])*xt[12])+z[27]*(L2*cos(xt[2])*sin(xt[3])*xt[11]+L2*cos(xt[3])*sin(xt[2])*xt[12])-z[28]*(L2*cos(xt[2])*cos(xt[3])*xt[11]-L2*sin(xt[2])*sin(xt[3])*xt[12])+z[30]*(L2*cos(xt[2])*sin(xt[3])*xt[11]+L2*cos(xt[3])*sin(xt[2])*xt[12])-z[2]*(L2*cos(xt[2])*cos(xt[3])*dxt[25]+L2*cos(xt[2])*cos(xt[3])*dxt[28]-L2*cos(xt[2])*sin(xt[3])*dxt[27]-L2*cos(xt[2])*sin(xt[3])*dxt[30])+z[10]*(L1*cos(xt[1])*cos(xt[2])*sin(xt[3])*dxt[11]*(L2*M3+LC2*M2)-L1*cos(xt[1])*sin(xt[2])*sin(xt[3])*xt[12]^2*(L2*M3+LC2*M2)-L1*cos(xt[1])*sin(xt[2])*sin(xt[3])*xt[11]^2*(L2*M3+LC2*M2)+L1*cos(xt[1])*cos(xt[3])*sin(xt[2])*dxt[12]*(L2*M3+LC2*M2)+2*L1*cos(xt[1])*cos(xt[2])*cos(xt[3])*xt[11]*xt[12]*(L2*M3+LC2*M2))+z[3]*(L2*cos(xt[3])*sin(xt[2])*dxt[27]+L2*cos(xt[3])*sin(xt[2])*dxt[30]+L2*sin(xt[2])*sin(xt[3])*dxt[25]+L2*sin(xt[2])*sin(xt[3])*dxt[28])-z[11]*(L2*cos(xt[2])*sin(xt[3])*dxt[21]-L2*cos(xt[2])*cos(xt[3])*dxt[22]-L2*cos(xt[2])*cos(xt[3])*dxt[19]+L2*cos(xt[2])*sin(xt[3])*dxt[24]+0.0*cos(xt[2])*sin(xt[3])*(L2*M3+LC2*M2)+L1*cos(xt[2])*sin(xt[1])*sin(xt[3])*xt[10]^2*(L2*M3+LC2*M2)-L1*cos(xt[1])*cos(xt[2])*sin(xt[3])*dxt[10]*(L2*M3+LC2*M2))-L2*cos(xt[3])*sin(xt[2])*z[19]-L2*cos(xt[3])*sin(xt[2])*z[22]+L2*cos(xt[3])*sin(xt[2])*z[31]+L2*sin(xt[2])*sin(xt[3])*z[21]+L2*sin(xt[2])*sin(xt[3])*z[24]-L2*sin(xt[2])*sin(xt[3])*z[33]
            res[4] = dz[4]-z[14]*(L1*dxt[13]*(L2*M3+LC2*M2)*(cos(xt[4])*sin(xt[5])-cos(xt[5])*cos(xt[6])*sin(xt[4]))-L1*xt[13]^2*(L2*M3+LC2*M2)*(sin(xt[4])*sin(xt[5])+cos(xt[4])*cos(xt[5])*cos(xt[6])))+z[4]*((L1*cos(xt[4])*dxt[26])*0.5-L1*sin(xt[4])*dxt[27]+(sqrt(3)*L1*cos(xt[4])*dxt[25])*0.5)-z[15]*(L1*cos(xt[4])*sin(xt[5])*sin(xt[6])*xt[13]^2*(L2*M3+LC2*M2)+L1*sin(xt[4])*sin(xt[5])*sin(xt[6])*dxt[13]*(L2*M3+LC2*M2))-z[13]*(0.0*sin(xt[4])*(L1*(M2+M3)+LC1*M1)+(L1*cos(xt[4])*dxt[20])*0.5-L1*sin(xt[4])*dxt[21]+(sqrt(3)*L1*cos(xt[4])*dxt[19])*0.5+L1*dxt[14]*(L2*M3+LC2*M2)*(cos(xt[4])*sin(xt[5])-cos(xt[5])*cos(xt[6])*sin(xt[4]))+L1*xt[14]^2*(L2*M3+LC2*M2)*(cos(xt[4])*cos(xt[5])+cos(xt[6])*sin(xt[4])*sin(xt[5]))+L1*cos(xt[6])*sin(xt[4])*sin(xt[5])*xt[15]^2*(L2*M3+LC2*M2)+L1*sin(xt[4])*sin(xt[5])*sin(xt[6])*dxt[15]*(L2*M3+LC2*M2)+2*L1*cos(xt[5])*sin(xt[4])*sin(xt[6])*xt[14]*xt[15]*(L2*M3+LC2*M2))+L1*cos(xt[4])*z[21]+(L1*sin(xt[4])*z[20])*0.5+(sqrt(3)*L1*sin(xt[4])*z[19])*0.5+(L1*cos(xt[4])*z[26]*xt[13])*0.5-L1*sin(xt[4])*z[27]*xt[13]+(sqrt(3)*L1*cos(xt[4])*z[25]*xt[13])*0.5
            res[5] = dz[5]-z[27]*(L2*cos(xt[6])*sin(xt[5])*xt[14]+L2*cos(xt[5])*sin(xt[6])*xt[15])+z[25]*((sqrt(3)*L2*cos(xt[5])*xt[14])*0.5-(L2*cos(xt[5])*cos(xt[6])*xt[15])*0.5+(L2*sin(xt[5])*sin(xt[6])*xt[14])*0.5)-z[6]*((L2*cos(xt[5])*cos(xt[6])*dxt[25])*0.5+L2*cos(xt[5])*sin(xt[6])*dxt[27]-(sqrt(3)*L2*cos(xt[5])*cos(xt[6])*dxt[26])*0.5)-z[19]*((L2*cos(xt[5])*sin(xt[6]))*0.5-(sqrt(3)*L2*sin(xt[5]))*0.5)+z[20]*((L2*sin(xt[5]))*0.5+(sqrt(3)*L2*cos(xt[5])*sin(xt[6]))*0.5)+z[5]*(dxt[25]*((sqrt(3)*L2*cos(xt[5]))*0.5+(L2*sin(xt[5])*sin(xt[6]))*0.5)+dxt[26]*((L2*cos(xt[5]))*0.5-(sqrt(3)*L2*sin(xt[5])*sin(xt[6]))*0.5)-L2*cos(xt[6])*sin(xt[5])*dxt[27])-z[15]*(2*cos(xt[5])*sin(xt[5])*dxt[15]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(xt[5])*sin(xt[6])*dxt[21]-(L2*cos(xt[5])*cos(xt[6])*dxt[19])*0.5+2*cos(2*xt[5])*xt[14]*xt[15]*(J2+L2^2*M3+LC2^2*M2)+0.0*cos(xt[5])*sin(xt[6])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(xt[5])*cos(xt[6])*dxt[20])*0.5+L1*cos(xt[5])*sin(xt[4])*sin(xt[6])*xt[13]^2*(L2*M3+LC2*M2)-L1*cos(xt[4])*cos(xt[5])*sin(xt[6])*dxt[13]*(L2*M3+LC2*M2))+z[26]*((L2*cos(xt[5])*xt[14])*0.5+(sqrt(3)*L2*cos(xt[5])*cos(xt[6])*xt[15])*0.5-(sqrt(3)*L2*sin(xt[5])*sin(xt[6])*xt[14])*0.5)+z[13]*(L1*xt[14]^2*(L2*M3+LC2*M2)*(sin(xt[4])*sin(xt[5])+cos(xt[4])*cos(xt[5])*cos(xt[6]))-L1*dxt[14]*(L2*M3+LC2*M2)*(cos(xt[5])*sin(xt[4])-cos(xt[4])*cos(xt[6])*sin(xt[5]))+L1*cos(xt[4])*cos(xt[5])*cos(xt[6])*xt[15]^2*(L2*M3+LC2*M2)+L1*cos(xt[4])*cos(xt[5])*sin(xt[6])*dxt[15]*(L2*M3+LC2*M2)-2*L1*cos(xt[4])*sin(xt[5])*sin(xt[6])*xt[14]*xt[15]*(L2*M3+LC2*M2))-z[14]*(dxt[19]*((sqrt(3)*L2*cos(xt[5]))*0.5+(L2*sin(xt[5])*sin(xt[6]))*0.5)+dxt[20]*((L2*cos(xt[5]))*0.5-(sqrt(3)*L2*sin(xt[5])*sin(xt[6]))*0.5)-cos(xt[5])^2*xt[15]^2*(J2+L2^2*M3+LC2^2*M2)+sin(xt[5])^2*xt[15]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(xt[6])*sin(xt[5])*dxt[21]+L1*dxt[13]*(L2*M3+LC2*M2)*(cos(xt[5])*sin(xt[4])-cos(xt[4])*cos(xt[6])*sin(xt[5]))+0.0*cos(xt[6])*sin(xt[5])*(L2*M3+LC2*M2)+L1*xt[13]^2*(L2*M3+LC2*M2)*(cos(xt[4])*cos(xt[5])+cos(xt[6])*sin(xt[4])*sin(xt[5])))+L2*cos(xt[5])*cos(xt[6])*z[21]
            res[6] = dz[6]+z[14]*((L2*cos(xt[5])*cos(xt[6])*dxt[19])*0.5+L2*cos(xt[5])*sin(xt[6])*dxt[21]-0.0*cos(xt[5])*sin(xt[6])*(L2*M3+LC2*M2)-(sqrt(3)*L2*cos(xt[5])*cos(xt[6])*dxt[20])*0.5-L1*cos(xt[5])*sin(xt[4])*sin(xt[6])*xt[13]^2*(L2*M3+LC2*M2)+L1*cos(xt[4])*cos(xt[5])*sin(xt[6])*dxt[13]*(L2*M3+LC2*M2))-z[27]*(L2*cos(xt[5])*sin(xt[6])*xt[14]+L2*cos(xt[6])*sin(xt[5])*xt[15])-z[25]*((L2*cos(xt[5])*cos(xt[6])*xt[14])*0.5-(L2*sin(xt[5])*sin(xt[6])*xt[15])*0.5)+z[15]*(L2*cos(xt[6])*sin(xt[5])*dxt[21]-(L2*sin(xt[5])*sin(xt[6])*dxt[19])*0.5-0.0*cos(xt[6])*sin(xt[5])*(L2*M3+LC2*M2)+(sqrt(3)*L2*sin(xt[5])*sin(xt[6])*dxt[20])*0.5-L1*cos(xt[6])*sin(xt[4])*sin(xt[5])*xt[13]^2*(L2*M3+LC2*M2)+L1*cos(xt[4])*cos(xt[6])*sin(xt[5])*dxt[13]*(L2*M3+LC2*M2))+z[13]*(L1*cos(xt[4])*cos(xt[5])*sin(xt[6])*dxt[14]*(L2*M3+LC2*M2)-L1*cos(xt[4])*sin(xt[5])*sin(xt[6])*xt[15]^2*(L2*M3+LC2*M2)-L1*cos(xt[4])*sin(xt[5])*sin(xt[6])*xt[14]^2*(L2*M3+LC2*M2)+L1*cos(xt[4])*cos(xt[6])*sin(xt[5])*dxt[15]*(L2*M3+LC2*M2)+2*L1*cos(xt[4])*cos(xt[5])*cos(xt[6])*xt[14]*xt[15]*(L2*M3+LC2*M2))-z[5]*((L2*cos(xt[5])*cos(xt[6])*dxt[25])*0.5+L2*cos(xt[5])*sin(xt[6])*dxt[27]-(sqrt(3)*L2*cos(xt[5])*cos(xt[6])*dxt[26])*0.5)-z[6]*(L2*cos(xt[6])*sin(xt[5])*dxt[27]-(L2*sin(xt[5])*sin(xt[6])*dxt[25])*0.5+(sqrt(3)*L2*sin(xt[5])*sin(xt[6])*dxt[26])*0.5)+z[26]*((sqrt(3)*L2*cos(xt[5])*cos(xt[6])*xt[14])*0.5-(sqrt(3)*L2*sin(xt[5])*sin(xt[6])*xt[15])*0.5)-(L2*cos(xt[6])*sin(xt[5])*z[19])*0.5-L2*sin(xt[5])*sin(xt[6])*z[21]+(sqrt(3)*L2*cos(xt[6])*sin(xt[5])*z[20])*0.5
            res[7] = dz[7]-z[17]*(L1*dxt[16]*(L2*M3+LC2*M2)*(cos(xt[7])*sin(xt[8])-cos(xt[8])*cos(xt[9])*sin(xt[7]))-L1*xt[16]^2*(L2*M3+LC2*M2)*(sin(xt[7])*sin(xt[8])+cos(xt[7])*cos(xt[8])*cos(xt[9])))-z[7]*(L1*sin(xt[7])*dxt[30]-(L1*cos(xt[7])*dxt[29])*0.5+(sqrt(3)*L1*cos(xt[7])*dxt[28])*0.5)-z[18]*(L1*cos(xt[7])*sin(xt[8])*sin(xt[9])*xt[16]^2*(L2*M3+LC2*M2)+L1*sin(xt[7])*sin(xt[8])*sin(xt[9])*dxt[16]*(L2*M3+LC2*M2))-z[16]*(0.0*sin(xt[7])*(L1*(M2+M3)+LC1*M1)+(L1*cos(xt[7])*dxt[23])*0.5-L1*sin(xt[7])*dxt[24]-(sqrt(3)*L1*cos(xt[7])*dxt[22])*0.5+L1*dxt[17]*(L2*M3+LC2*M2)*(cos(xt[7])*sin(xt[8])-cos(xt[8])*cos(xt[9])*sin(xt[7]))+L1*xt[17]^2*(L2*M3+LC2*M2)*(cos(xt[7])*cos(xt[8])+cos(xt[9])*sin(xt[7])*sin(xt[8]))+L1*cos(xt[9])*sin(xt[7])*sin(xt[8])*xt[18]^2*(L2*M3+LC2*M2)+L1*sin(xt[7])*sin(xt[8])*sin(xt[9])*dxt[18]*(L2*M3+LC2*M2)+2*L1*cos(xt[8])*sin(xt[7])*sin(xt[9])*xt[17]*xt[18]*(L2*M3+LC2*M2))+L1*cos(xt[7])*z[24]+(L1*sin(xt[7])*z[23])*0.5-(sqrt(3)*L1*sin(xt[7])*z[22])*0.5+(L1*cos(xt[7])*z[29]*xt[16])*0.5-L1*sin(xt[7])*z[30]*xt[16]-(sqrt(3)*L1*cos(xt[7])*z[28]*xt[16])*0.5
            res[8] = dz[8]-z[30]*(L2*cos(xt[9])*sin(xt[8])*xt[17]+L2*cos(xt[8])*sin(xt[9])*xt[18])-z[28]*((L2*cos(xt[8])*cos(xt[9])*xt[18])*0.5+(sqrt(3)*L2*cos(xt[8])*xt[17])*0.5-(L2*sin(xt[8])*sin(xt[9])*xt[17])*0.5)-z[9]*((L2*cos(xt[8])*cos(xt[9])*dxt[28])*0.5+L2*cos(xt[8])*sin(xt[9])*dxt[30]+(sqrt(3)*L2*cos(xt[8])*cos(xt[9])*dxt[29])*0.5)-z[22]*((L2*cos(xt[8])*sin(xt[9]))*0.5+(sqrt(3)*L2*sin(xt[8]))*0.5)+z[23]*((L2*sin(xt[8]))*0.5-(sqrt(3)*L2*cos(xt[8])*sin(xt[9]))*0.5)-z[8]*(dxt[28]*((sqrt(3)*L2*cos(xt[8]))*0.5-(L2*sin(xt[8])*sin(xt[9]))*0.5)-dxt[29]*((L2*cos(xt[8]))*0.5+(sqrt(3)*L2*sin(xt[8])*sin(xt[9]))*0.5)+L2*cos(xt[9])*sin(xt[8])*dxt[30])+z[18]*((L2*cos(xt[8])*cos(xt[9])*dxt[22])*0.5+L2*cos(xt[8])*sin(xt[9])*dxt[24]-2*cos(xt[8])*sin(xt[8])*dxt[18]*(J2+L2^2*M3+LC2^2*M2)-2*cos(2*xt[8])*xt[17]*xt[18]*(J2+L2^2*M3+LC2^2*M2)-0.0*cos(xt[8])*sin(xt[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(xt[8])*cos(xt[9])*dxt[23])*0.5-L1*cos(xt[8])*sin(xt[7])*sin(xt[9])*xt[16]^2*(L2*M3+LC2*M2)+L1*cos(xt[7])*cos(xt[8])*sin(xt[9])*dxt[16]*(L2*M3+LC2*M2))+z[29]*((L2*cos(xt[8])*xt[17])*0.5-(sqrt(3)*L2*cos(xt[8])*cos(xt[9])*xt[18])*0.5+(sqrt(3)*L2*sin(xt[8])*sin(xt[9])*xt[17])*0.5)+z[16]*(L1*xt[17]^2*(L2*M3+LC2*M2)*(sin(xt[7])*sin(xt[8])+cos(xt[7])*cos(xt[8])*cos(xt[9]))-L1*dxt[17]*(L2*M3+LC2*M2)*(cos(xt[8])*sin(xt[7])-cos(xt[7])*cos(xt[9])*sin(xt[8]))+L1*cos(xt[7])*cos(xt[8])*cos(xt[9])*xt[18]^2*(L2*M3+LC2*M2)+L1*cos(xt[7])*cos(xt[8])*sin(xt[9])*dxt[18]*(L2*M3+LC2*M2)-2*L1*cos(xt[7])*sin(xt[8])*sin(xt[9])*xt[17]*xt[18]*(L2*M3+LC2*M2))-z[17]*(dxt[23]*((L2*cos(xt[8]))*0.5+(sqrt(3)*L2*sin(xt[8])*sin(xt[9]))*0.5)-dxt[22]*((sqrt(3)*L2*cos(xt[8]))*0.5-(L2*sin(xt[8])*sin(xt[9]))*0.5)-cos(xt[8])^2*xt[18]^2*(J2+L2^2*M3+LC2^2*M2)+sin(xt[8])^2*xt[18]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(xt[9])*sin(xt[8])*dxt[24]+L1*dxt[16]*(L2*M3+LC2*M2)*(cos(xt[8])*sin(xt[7])-cos(xt[7])*cos(xt[9])*sin(xt[8]))+0.0*cos(xt[9])*sin(xt[8])*(L2*M3+LC2*M2)+L1*xt[16]^2*(L2*M3+LC2*M2)*(cos(xt[7])*cos(xt[8])+cos(xt[9])*sin(xt[7])*sin(xt[8])))+L2*cos(xt[8])*cos(xt[9])*z[24]
            res[9] = dz[9]+z[17]*((L2*cos(xt[8])*cos(xt[9])*dxt[22])*0.5+L2*cos(xt[8])*sin(xt[9])*dxt[24]-0.0*cos(xt[8])*sin(xt[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(xt[8])*cos(xt[9])*dxt[23])*0.5-L1*cos(xt[8])*sin(xt[7])*sin(xt[9])*xt[16]^2*(L2*M3+LC2*M2)+L1*cos(xt[7])*cos(xt[8])*sin(xt[9])*dxt[16]*(L2*M3+LC2*M2))-z[30]*(L2*cos(xt[8])*sin(xt[9])*xt[17]+L2*cos(xt[9])*sin(xt[8])*xt[18])-z[28]*((L2*cos(xt[8])*cos(xt[9])*xt[17])*0.5-(L2*sin(xt[8])*sin(xt[9])*xt[18])*0.5)-z[18]*((L2*sin(xt[8])*sin(xt[9])*dxt[22])*0.5-L2*cos(xt[9])*sin(xt[8])*dxt[24]+0.0*cos(xt[9])*sin(xt[8])*(L2*M3+LC2*M2)+(sqrt(3)*L2*sin(xt[8])*sin(xt[9])*dxt[23])*0.5+L1*cos(xt[9])*sin(xt[7])*sin(xt[8])*xt[16]^2*(L2*M3+LC2*M2)-L1*cos(xt[7])*cos(xt[9])*sin(xt[8])*dxt[16]*(L2*M3+LC2*M2))+z[16]*(L1*cos(xt[7])*cos(xt[8])*sin(xt[9])*dxt[17]*(L2*M3+LC2*M2)-L1*cos(xt[7])*sin(xt[8])*sin(xt[9])*xt[18]^2*(L2*M3+LC2*M2)-L1*cos(xt[7])*sin(xt[8])*sin(xt[9])*xt[17]^2*(L2*M3+LC2*M2)+L1*cos(xt[7])*cos(xt[9])*sin(xt[8])*dxt[18]*(L2*M3+LC2*M2)+2*L1*cos(xt[7])*cos(xt[8])*cos(xt[9])*xt[17]*xt[18]*(L2*M3+LC2*M2))-z[8]*((L2*cos(xt[8])*cos(xt[9])*dxt[28])*0.5+L2*cos(xt[8])*sin(xt[9])*dxt[30]+(sqrt(3)*L2*cos(xt[8])*cos(xt[9])*dxt[29])*0.5)+z[9]*((L2*sin(xt[8])*sin(xt[9])*dxt[28])*0.5-L2*cos(xt[9])*sin(xt[8])*dxt[30]+(sqrt(3)*L2*sin(xt[8])*sin(xt[9])*dxt[29])*0.5)-z[29]*((sqrt(3)*L2*cos(xt[8])*cos(xt[9])*xt[17])*0.5-(sqrt(3)*L2*sin(xt[8])*sin(xt[9])*xt[18])*0.5)-(L2*cos(xt[9])*sin(xt[8])*z[22])*0.5-L2*sin(xt[8])*sin(xt[9])*z[24]-(sqrt(3)*L2*cos(xt[9])*sin(xt[8])*z[23])*0.5
            res[10] = z[1]+dz[10]*(J1+L1^2*(M2+M3)+LC1^2*M1)-z[11]*(L1*(L2*M3+LC2*M2)*(cos(xt[2])*cos(xt[3])*sin(xt[1])*dxt[1]-cos(xt[2])*sin(xt[1])*dxt[2]-cos(xt[1])*sin(xt[2])*dxt[1]+cos(xt[1])*cos(xt[3])*sin(xt[2])*dxt[2]+cos(xt[1])*cos(xt[2])*sin(xt[3])*dxt[3])+2*L1*xt[10]*(L2*M3+LC2*M2)*(cos(xt[1])*sin(xt[2])-cos(xt[2])*cos(xt[3])*sin(xt[1])))-γ*z[10]-z[12]*(L1*cos(xt[1])*cos(xt[2])*sin(xt[3])*dxt[2]*(L2*M3+LC2*M2)+L1*cos(xt[1])*cos(xt[3])*sin(xt[2])*dxt[3]*(L2*M3+LC2*M2)-L1*sin(xt[1])*sin(xt[2])*sin(xt[3])*dxt[1]*(L2*M3+LC2*M2)+2*L1*sin(xt[1])*sin(xt[2])*sin(xt[3])*xt[10]*(L2*M3+LC2*M2))-L1*cos(xt[1])*z[27]-L1*cos(xt[1])*z[30]+L1*sin(xt[1])*z[26]+L1*sin(xt[1])*z[29]+L1*dz[11]*(L2*M3+LC2*M2)*(sin(xt[1])*sin(xt[2])+cos(xt[1])*cos(xt[2])*cos(xt[3]))-L1*cos(xt[1])*sin(xt[2])*sin(xt[3])*dz[12]*(L2*M3+LC2*M2)
            res[11] = z[2]-z[10]*(L1*(L2*M3+LC2*M2)*(cos(xt[2])*cos(xt[3])*sin(xt[1])*dxt[1]-cos(xt[2])*sin(xt[1])*dxt[2]-cos(xt[1])*sin(xt[2])*dxt[1]+cos(xt[1])*cos(xt[3])*sin(xt[2])*dxt[2]+cos(xt[1])*cos(xt[2])*sin(xt[3])*dxt[3])+2*L1*xt[11]*(L2*M3+LC2*M2)*(cos(xt[2])*sin(xt[1])-cos(xt[1])*cos(xt[3])*sin(xt[2]))-2*L1*cos(xt[1])*cos(xt[2])*sin(xt[3])*xt[12]*(L2*M3+LC2*M2))+dz[11]*(J2+L2^2*M3+LC2^2*M2)-γ*z[11]+L2*sin(xt[2])*z[26]+L2*sin(xt[2])*z[29]-L2*cos(xt[2])*cos(xt[3])*z[27]-L2*cos(xt[2])*cos(xt[3])*z[30]-L2*cos(xt[2])*sin(xt[3])*z[25]-L2*cos(xt[2])*sin(xt[3])*z[28]-sin(2*xt[2])*z[12]*xt[12]*(J2+L2^2*M3+LC2^2*M2)+L1*dz[10]*(L2*M3+LC2*M2)*(sin(xt[1])*sin(xt[2])+cos(xt[1])*cos(xt[2])*cos(xt[3]))
            res[12] = z[3]-z[12]*(γ+sin(2*xt[2])*xt[11]*(J2+L2^2*M3+LC2^2*M2)-2*cos(xt[2])*sin(xt[2])*dxt[2]*(J2+L2^2*M3+LC2^2*M2))+z[10]*(2*L1*cos(xt[1])*cos(xt[2])*sin(xt[3])*xt[11]*(L2*M3+LC2*M2)-L1*cos(xt[1])*cos(xt[3])*sin(xt[2])*dxt[3]*(L2*M3+LC2*M2)-L1*cos(xt[1])*cos(xt[2])*sin(xt[3])*dxt[2]*(L2*M3+LC2*M2)+2*L1*cos(xt[1])*cos(xt[3])*sin(xt[2])*xt[12]*(L2*M3+LC2*M2)+L1*sin(xt[1])*sin(xt[2])*sin(xt[3])*dxt[1]*(L2*M3+LC2*M2))+sin(xt[2])^2*dz[12]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(xt[3])*sin(xt[2])*z[25]-L2*cos(xt[3])*sin(xt[2])*z[28]+L2*sin(xt[2])*sin(xt[3])*z[27]+L2*sin(xt[2])*sin(xt[3])*z[30]+2*cos(xt[2])*sin(xt[2])*z[11]*xt[12]*(J2+L2^2*M3+LC2^2*M2)-L1*cos(xt[1])*sin(xt[2])*sin(xt[3])*dz[10]*(L2*M3+LC2*M2)
            res[13] = z[4]+dz[13]*(J1+L1^2*(M2+M3)+LC1^2*M1)-z[14]*(L1*(L2*M3+LC2*M2)*(cos(xt[5])*cos(xt[6])*sin(xt[4])*dxt[4]-cos(xt[5])*sin(xt[4])*dxt[5]-cos(xt[4])*sin(xt[5])*dxt[4]+cos(xt[4])*cos(xt[6])*sin(xt[5])*dxt[5]+cos(xt[4])*cos(xt[5])*sin(xt[6])*dxt[6])+2*L1*xt[13]*(L2*M3+LC2*M2)*(cos(xt[4])*sin(xt[5])-cos(xt[5])*cos(xt[6])*sin(xt[4])))-γ*z[13]-z[15]*(L1*cos(xt[4])*cos(xt[5])*sin(xt[6])*dxt[5]*(L2*M3+LC2*M2)+L1*cos(xt[4])*cos(xt[6])*sin(xt[5])*dxt[6]*(L2*M3+LC2*M2)-L1*sin(xt[4])*sin(xt[5])*sin(xt[6])*dxt[4]*(L2*M3+LC2*M2)+2*L1*sin(xt[4])*sin(xt[5])*sin(xt[6])*xt[13]*(L2*M3+LC2*M2))+L1*cos(xt[4])*z[27]+(L1*sin(xt[4])*z[26])*0.5+(sqrt(3)*L1*sin(xt[4])*z[25])*0.5+L1*dz[14]*(L2*M3+LC2*M2)*(sin(xt[4])*sin(xt[5])+cos(xt[4])*cos(xt[5])*cos(xt[6]))-L1*cos(xt[4])*sin(xt[5])*sin(xt[6])*dz[15]*(L2*M3+LC2*M2)
            res[14] = z[5]-z[25]*((L2*cos(xt[5])*sin(xt[6]))*0.5-(sqrt(3)*L2*sin(xt[5]))*0.5)+z[26]*((L2*sin(xt[5]))*0.5+(sqrt(3)*L2*cos(xt[5])*sin(xt[6]))*0.5)-z[13]*(L1*(L2*M3+LC2*M2)*(cos(xt[5])*cos(xt[6])*sin(xt[4])*dxt[4]-cos(xt[5])*sin(xt[4])*dxt[5]-cos(xt[4])*sin(xt[5])*dxt[4]+cos(xt[4])*cos(xt[6])*sin(xt[5])*dxt[5]+cos(xt[4])*cos(xt[5])*sin(xt[6])*dxt[6])+2*L1*xt[14]*(L2*M3+LC2*M2)*(cos(xt[5])*sin(xt[4])-cos(xt[4])*cos(xt[6])*sin(xt[5]))-2*L1*cos(xt[4])*cos(xt[5])*sin(xt[6])*xt[15]*(L2*M3+LC2*M2))+dz[14]*(J2+L2^2*M3+LC2^2*M2)-γ*z[14]+L2*cos(xt[5])*cos(xt[6])*z[27]-sin(2*xt[5])*z[15]*xt[15]*(J2+L2^2*M3+LC2^2*M2)+L1*dz[13]*(L2*M3+LC2*M2)*(sin(xt[4])*sin(xt[5])+cos(xt[4])*cos(xt[5])*cos(xt[6]))
            res[15] = z[6]-z[15]*(γ+sin(2*xt[5])*xt[14]*(J2+L2^2*M3+LC2^2*M2)-2*cos(xt[5])*sin(xt[5])*dxt[5]*(J2+L2^2*M3+LC2^2*M2))+z[13]*(2*L1*cos(xt[4])*cos(xt[5])*sin(xt[6])*xt[14]*(L2*M3+LC2*M2)-L1*cos(xt[4])*cos(xt[6])*sin(xt[5])*dxt[6]*(L2*M3+LC2*M2)-L1*cos(xt[4])*cos(xt[5])*sin(xt[6])*dxt[5]*(L2*M3+LC2*M2)+2*L1*cos(xt[4])*cos(xt[6])*sin(xt[5])*xt[15]*(L2*M3+LC2*M2)+L1*sin(xt[4])*sin(xt[5])*sin(xt[6])*dxt[4]*(L2*M3+LC2*M2))+sin(xt[5])^2*dz[15]*(J2+L2^2*M3+LC2^2*M2)-(L2*cos(xt[6])*sin(xt[5])*z[25])*0.5-L2*sin(xt[5])*sin(xt[6])*z[27]+2*cos(xt[5])*sin(xt[5])*z[14]*xt[15]*(J2+L2^2*M3+LC2^2*M2)+(sqrt(3)*L2*cos(xt[6])*sin(xt[5])*z[26])*0.5-L1*cos(xt[4])*sin(xt[5])*sin(xt[6])*dz[13]*(L2*M3+LC2*M2)
            res[16] = z[7]+dz[16]*(J1+L1^2*(M2+M3)+LC1^2*M1)-z[17]*(L1*(L2*M3+LC2*M2)*(cos(xt[8])*cos(xt[9])*sin(xt[7])*dxt[7]-cos(xt[8])*sin(xt[7])*dxt[8]-cos(xt[7])*sin(xt[8])*dxt[7]+cos(xt[7])*cos(xt[9])*sin(xt[8])*dxt[8]+cos(xt[7])*cos(xt[8])*sin(xt[9])*dxt[9])+2*L1*xt[16]*(L2*M3+LC2*M2)*(cos(xt[7])*sin(xt[8])-cos(xt[8])*cos(xt[9])*sin(xt[7])))-γ*z[16]-z[18]*(L1*cos(xt[7])*cos(xt[8])*sin(xt[9])*dxt[8]*(L2*M3+LC2*M2)+L1*cos(xt[7])*cos(xt[9])*sin(xt[8])*dxt[9]*(L2*M3+LC2*M2)-L1*sin(xt[7])*sin(xt[8])*sin(xt[9])*dxt[7]*(L2*M3+LC2*M2)+2*L1*sin(xt[7])*sin(xt[8])*sin(xt[9])*xt[16]*(L2*M3+LC2*M2))+L1*cos(xt[7])*z[30]+(L1*sin(xt[7])*z[29])*0.5-(sqrt(3)*L1*sin(xt[7])*z[28])*0.5+L1*dz[17]*(L2*M3+LC2*M2)*(sin(xt[7])*sin(xt[8])+cos(xt[7])*cos(xt[8])*cos(xt[9]))-L1*cos(xt[7])*sin(xt[8])*sin(xt[9])*dz[18]*(L2*M3+LC2*M2)
            res[17] = z[8]-z[28]*((L2*cos(xt[8])*sin(xt[9]))*0.5+(sqrt(3)*L2*sin(xt[8]))*0.5)+z[29]*((L2*sin(xt[8]))*0.5-(sqrt(3)*L2*cos(xt[8])*sin(xt[9]))*0.5)-z[16]*(L1*(L2*M3+LC2*M2)*(cos(xt[8])*cos(xt[9])*sin(xt[7])*dxt[7]-cos(xt[8])*sin(xt[7])*dxt[8]-cos(xt[7])*sin(xt[8])*dxt[7]+cos(xt[7])*cos(xt[9])*sin(xt[8])*dxt[8]+cos(xt[7])*cos(xt[8])*sin(xt[9])*dxt[9])+2*L1*xt[17]*(L2*M3+LC2*M2)*(cos(xt[8])*sin(xt[7])-cos(xt[7])*cos(xt[9])*sin(xt[8]))-2*L1*cos(xt[7])*cos(xt[8])*sin(xt[9])*xt[18]*(L2*M3+LC2*M2))+dz[17]*(J2+L2^2*M3+LC2^2*M2)-γ*z[17]+L2*cos(xt[8])*cos(xt[9])*z[30]-sin(2*xt[8])*z[18]*xt[18]*(J2+L2^2*M3+LC2^2*M2)+L1*dz[16]*(L2*M3+LC2*M2)*(sin(xt[7])*sin(xt[8])+cos(xt[7])*cos(xt[8])*cos(xt[9]))
            res[18] = z[9]-z[18]*(γ+sin(2*xt[8])*xt[17]*(J2+L2^2*M3+LC2^2*M2)-2*cos(xt[8])*sin(xt[8])*dxt[8]*(J2+L2^2*M3+LC2^2*M2))+z[16]*(2*L1*cos(xt[7])*cos(xt[8])*sin(xt[9])*xt[17]*(L2*M3+LC2*M2)-L1*cos(xt[7])*cos(xt[9])*sin(xt[8])*dxt[9]*(L2*M3+LC2*M2)-L1*cos(xt[7])*cos(xt[8])*sin(xt[9])*dxt[8]*(L2*M3+LC2*M2)+2*L1*cos(xt[7])*cos(xt[9])*sin(xt[8])*xt[18]*(L2*M3+LC2*M2)+L1*sin(xt[7])*sin(xt[8])*sin(xt[9])*dxt[7]*(L2*M3+LC2*M2))+sin(xt[8])^2*dz[18]*(J2+L2^2*M3+LC2^2*M2)-(L2*cos(xt[9])*sin(xt[8])*z[28])*0.5-L2*sin(xt[8])*sin(xt[9])*z[30]+2*cos(xt[8])*sin(xt[8])*z[17]*xt[18]*(J2+L2^2*M3+LC2^2*M2)-(sqrt(3)*L2*cos(xt[9])*sin(xt[8])*z[29])*0.5-L1*cos(xt[7])*sin(xt[8])*sin(xt[9])*dz[16]*(L2*M3+LC2*M2)
            res[19] = z[14]*((sqrt(3)*L2*cos(xt[5])*dxt[5])*0.5-(L2*cos(xt[5])*cos(xt[6])*dxt[6])*0.5+(L2*sin(xt[5])*sin(xt[6])*dxt[5])*0.5)-dz[14]*((L2*cos(xt[5])*sin(xt[6]))*0.5-(sqrt(3)*L2*sin(xt[5]))*0.5)-z[11]*(L2*cos(xt[2])*cos(xt[3])*dxt[3]-L2*sin(xt[2])*sin(xt[3])*dxt[2])-z[12]*(L2*cos(xt[2])*cos(xt[3])*dxt[2]-L2*sin(xt[2])*sin(xt[3])*dxt[3])-z[15]*((L2*cos(xt[5])*cos(xt[6])*dxt[5])*0.5-(L2*sin(xt[5])*sin(xt[6])*dxt[6])*0.5)-L2*cos(xt[2])*sin(xt[3])*dz[11]-L2*cos(xt[3])*sin(xt[2])*dz[12]-(L2*cos(xt[6])*sin(xt[5])*dz[15])*0.5+(sqrt(3)*L1*sin(xt[4])*dz[13])*0.5+(sqrt(3)*L1*cos(xt[4])*dxt[4]*z[13])*0.5
            res[20] = dz[14]*((L2*sin(xt[5]))*0.5+(sqrt(3)*L2*cos(xt[5])*sin(xt[6]))*0.5)+z[14]*((L2*cos(xt[5])*dxt[5])*0.5+(sqrt(3)*L2*cos(xt[5])*cos(xt[6])*dxt[6])*0.5-(sqrt(3)*L2*sin(xt[5])*sin(xt[6])*dxt[5])*0.5)+z[15]*((sqrt(3)*L2*cos(xt[5])*cos(xt[6])*dxt[5])*0.5-(sqrt(3)*L2*sin(xt[5])*sin(xt[6])*dxt[6])*0.5)+L1*sin(xt[1])*dz[10]+L2*sin(xt[2])*dz[11]+(L1*sin(xt[4])*dz[13])*0.5+L1*cos(xt[1])*dxt[1]*z[10]+L2*cos(xt[2])*dxt[2]*z[11]+(L1*cos(xt[4])*dxt[4]*z[13])*0.5+(sqrt(3)*L2*cos(xt[6])*sin(xt[5])*dz[15])*0.5
            res[21] = z[11]*(L2*cos(xt[3])*sin(xt[2])*dxt[2]+L2*cos(xt[2])*sin(xt[3])*dxt[3])+z[12]*(L2*cos(xt[2])*sin(xt[3])*dxt[2]+L2*cos(xt[3])*sin(xt[2])*dxt[3])-z[14]*(L2*cos(xt[6])*sin(xt[5])*dxt[5]+L2*cos(xt[5])*sin(xt[6])*dxt[6])-z[15]*(L2*cos(xt[5])*sin(xt[6])*dxt[5]+L2*cos(xt[6])*sin(xt[5])*dxt[6])-L1*cos(xt[1])*dz[10]+L1*cos(xt[4])*dz[13]-L2*cos(xt[2])*cos(xt[3])*dz[11]+L2*cos(xt[5])*cos(xt[6])*dz[14]+L2*sin(xt[2])*sin(xt[3])*dz[12]-L2*sin(xt[5])*sin(xt[6])*dz[15]+L1*sin(xt[1])*dxt[1]*z[10]-L1*sin(xt[4])*dxt[4]*z[13]
            res[22] = -z[17]*((L2*cos(xt[8])*cos(xt[9])*dxt[9])*0.5+(sqrt(3)*L2*cos(xt[8])*dxt[8])*0.5-(L2*sin(xt[8])*sin(xt[9])*dxt[8])*0.5)-dz[17]*((L2*cos(xt[8])*sin(xt[9]))*0.5+(sqrt(3)*L2*sin(xt[8]))*0.5)-z[11]*(L2*cos(xt[2])*cos(xt[3])*dxt[3]-L2*sin(xt[2])*sin(xt[3])*dxt[2])-z[12]*(L2*cos(xt[2])*cos(xt[3])*dxt[2]-L2*sin(xt[2])*sin(xt[3])*dxt[3])-z[18]*((L2*cos(xt[8])*cos(xt[9])*dxt[8])*0.5-(L2*sin(xt[8])*sin(xt[9])*dxt[9])*0.5)-L2*cos(xt[2])*sin(xt[3])*dz[11]-L2*cos(xt[3])*sin(xt[2])*dz[12]-(L2*cos(xt[9])*sin(xt[8])*dz[18])*0.5-(sqrt(3)*L1*sin(xt[7])*dz[16])*0.5-(sqrt(3)*L1*cos(xt[7])*dxt[7]*z[16])*0.5
            res[23] = dz[17]*((L2*sin(xt[8]))*0.5-(sqrt(3)*L2*cos(xt[8])*sin(xt[9]))*0.5)+z[17]*((L2*cos(xt[8])*dxt[8])*0.5-(sqrt(3)*L2*cos(xt[8])*cos(xt[9])*dxt[9])*0.5+(sqrt(3)*L2*sin(xt[8])*sin(xt[9])*dxt[8])*0.5)-z[18]*((sqrt(3)*L2*cos(xt[8])*cos(xt[9])*dxt[8])*0.5-(sqrt(3)*L2*sin(xt[8])*sin(xt[9])*dxt[9])*0.5)+L1*sin(xt[1])*dz[10]+L2*sin(xt[2])*dz[11]+(L1*sin(xt[7])*dz[16])*0.5+L1*cos(xt[1])*dxt[1]*z[10]+L2*cos(xt[2])*dxt[2]*z[11]+(L1*cos(xt[7])*dxt[7]*z[16])*0.5-(sqrt(3)*L2*cos(xt[9])*sin(xt[8])*dz[18])*0.5
            res[24] = z[11]*(L2*cos(xt[3])*sin(xt[2])*dxt[2]+L2*cos(xt[2])*sin(xt[3])*dxt[3])+z[12]*(L2*cos(xt[2])*sin(xt[3])*dxt[2]+L2*cos(xt[3])*sin(xt[2])*dxt[3])-z[17]*(L2*cos(xt[9])*sin(xt[8])*dxt[8]+L2*cos(xt[8])*sin(xt[9])*dxt[9])-z[18]*(L2*cos(xt[8])*sin(xt[9])*dxt[8]+L2*cos(xt[9])*sin(xt[8])*dxt[9])-L1*cos(xt[1])*dz[10]+L1*cos(xt[7])*dz[16]-L2*cos(xt[2])*cos(xt[3])*dz[11]+L2*cos(xt[8])*cos(xt[9])*dz[17]+L2*sin(xt[2])*sin(xt[3])*dz[12]-L2*sin(xt[8])*sin(xt[9])*dz[18]+L1*sin(xt[1])*dxt[1]*z[10]-L1*sin(xt[7])*dxt[7]*z[16]
            res[25] = dz[5]*((L2*cos(xt[5])*sin(xt[6]))*0.5-(sqrt(3)*L2*sin(xt[5]))*0.5)-z[5]*((sqrt(3)*L2*cos(xt[5])*dxt[5])*0.5-(L2*cos(xt[5])*cos(xt[6])*dxt[6])*0.5+(L2*sin(xt[5])*sin(xt[6])*dxt[5])*0.5)+z[2]*(L2*cos(xt[2])*cos(xt[3])*dxt[3]-L2*sin(xt[2])*sin(xt[3])*dxt[2])+z[3]*(L2*cos(xt[2])*cos(xt[3])*dxt[2]-L2*sin(xt[2])*sin(xt[3])*dxt[3])+z[6]*((L2*cos(xt[5])*cos(xt[6])*dxt[5])*0.5-(L2*sin(xt[5])*sin(xt[6])*dxt[6])*0.5)+L2*cos(xt[2])*sin(xt[3])*dz[2]+L2*cos(xt[3])*sin(xt[2])*dz[3]+(L2*cos(xt[6])*sin(xt[5])*dz[6])*0.5-(sqrt(3)*L1*sin(xt[4])*dz[4])*0.5-(sqrt(3)*L1*cos(xt[4])*dxt[4]*z[4])*0.5
            res[26] = -dz[5]*((L2*sin(xt[5]))*0.5+(sqrt(3)*L2*cos(xt[5])*sin(xt[6]))*0.5)-z[5]*((L2*cos(xt[5])*dxt[5])*0.5+(sqrt(3)*L2*cos(xt[5])*cos(xt[6])*dxt[6])*0.5-(sqrt(3)*L2*sin(xt[5])*sin(xt[6])*dxt[5])*0.5)-z[6]*((sqrt(3)*L2*cos(xt[5])*cos(xt[6])*dxt[5])*0.5-(sqrt(3)*L2*sin(xt[5])*sin(xt[6])*dxt[6])*0.5)-L1*sin(xt[1])*dz[1]-L2*sin(xt[2])*dz[2]-(L1*sin(xt[4])*dz[4])*0.5-L1*cos(xt[1])*dxt[1]*z[1]-L2*cos(xt[2])*dxt[2]*z[2]-(L1*cos(xt[4])*dxt[4]*z[4])*0.5-(sqrt(3)*L2*cos(xt[6])*sin(xt[5])*dz[6])*0.5
            res[27] = z[5]*(L2*cos(xt[6])*sin(xt[5])*dxt[5]+L2*cos(xt[5])*sin(xt[6])*dxt[6])-z[3]*(L2*cos(xt[2])*sin(xt[3])*dxt[2]+L2*cos(xt[3])*sin(xt[2])*dxt[3])-z[2]*(L2*cos(xt[3])*sin(xt[2])*dxt[2]+L2*cos(xt[2])*sin(xt[3])*dxt[3])+z[6]*(L2*cos(xt[5])*sin(xt[6])*dxt[5]+L2*cos(xt[6])*sin(xt[5])*dxt[6])+L1*cos(xt[1])*dz[1]-L1*cos(xt[4])*dz[4]+L2*cos(xt[2])*cos(xt[3])*dz[2]-L2*cos(xt[5])*cos(xt[6])*dz[5]-L2*sin(xt[2])*sin(xt[3])*dz[3]+L2*sin(xt[5])*sin(xt[6])*dz[6]-L1*sin(xt[1])*dxt[1]*z[1]+L1*sin(xt[4])*dxt[4]*z[4]
            res[28] = z[8]*((L2*cos(xt[8])*cos(xt[9])*dxt[9])*0.5+(sqrt(3)*L2*cos(xt[8])*dxt[8])*0.5-(L2*sin(xt[8])*sin(xt[9])*dxt[8])*0.5)+dz[8]*((L2*cos(xt[8])*sin(xt[9]))*0.5+(sqrt(3)*L2*sin(xt[8]))*0.5)+z[2]*(L2*cos(xt[2])*cos(xt[3])*dxt[3]-L2*sin(xt[2])*sin(xt[3])*dxt[2])+z[3]*(L2*cos(xt[2])*cos(xt[3])*dxt[2]-L2*sin(xt[2])*sin(xt[3])*dxt[3])+z[9]*((L2*cos(xt[8])*cos(xt[9])*dxt[8])*0.5-(L2*sin(xt[8])*sin(xt[9])*dxt[9])*0.5)+L2*cos(xt[2])*sin(xt[3])*dz[2]+L2*cos(xt[3])*sin(xt[2])*dz[3]+(L2*cos(xt[9])*sin(xt[8])*dz[9])*0.5+(sqrt(3)*L1*sin(xt[7])*dz[7])*0.5+(sqrt(3)*L1*cos(xt[7])*dxt[7]*z[7])*0.5
            res[29] = z[9]*((sqrt(3)*L2*cos(xt[8])*cos(xt[9])*dxt[8])*0.5-(sqrt(3)*L2*sin(xt[8])*sin(xt[9])*dxt[9])*0.5)-z[8]*((L2*cos(xt[8])*dxt[8])*0.5-(sqrt(3)*L2*cos(xt[8])*cos(xt[9])*dxt[9])*0.5+(sqrt(3)*L2*sin(xt[8])*sin(xt[9])*dxt[8])*0.5)-dz[8]*((L2*sin(xt[8]))*0.5-(sqrt(3)*L2*cos(xt[8])*sin(xt[9]))*0.5)-L1*sin(xt[1])*dz[1]-L2*sin(xt[2])*dz[2]-(L1*sin(xt[7])*dz[7])*0.5-L1*cos(xt[1])*dxt[1]*z[1]-L2*cos(xt[2])*dxt[2]*z[2]-(L1*cos(xt[7])*dxt[7]*z[7])*0.5+(sqrt(3)*L2*cos(xt[9])*sin(xt[8])*dz[9])*0.5
            res[30] = z[8]*(L2*cos(xt[9])*sin(xt[8])*dxt[8]+L2*cos(xt[8])*sin(xt[9])*dxt[9])-z[3]*(L2*cos(xt[2])*sin(xt[3])*dxt[2]+L2*cos(xt[3])*sin(xt[2])*dxt[3])-z[2]*(L2*cos(xt[3])*sin(xt[2])*dxt[2]+L2*cos(xt[2])*sin(xt[3])*dxt[3])+z[9]*(L2*cos(xt[8])*sin(xt[9])*dxt[8]+L2*cos(xt[9])*sin(xt[8])*dxt[9])+L1*cos(xt[1])*dz[1]-L1*cos(xt[7])*dz[7]+L2*cos(xt[2])*cos(xt[3])*dz[2]-L2*cos(xt[8])*cos(xt[9])*dz[8]-L2*sin(xt[2])*sin(xt[3])*dz[3]+L2*sin(xt[8])*sin(xt[9])*dz[9]-L1*sin(xt[1])*dxt[1]*z[1]+L1*sin(xt[7])*dxt[7]*z[7]
            # TODO: Do we really need to simulate with these three equations or could we remove them somehow since they are quite trivial?
            res[31] = -z[31]-(2*yt[1]-2*xt[31])/T
            res[32] = -z[32]-(2*yt[2]-2*xt[32])/T
            res[33] = -z[33]-(2*yt[3]-2*xt[33])/T
            # \beta equations
            res[34] = dz[34] - (z[10]*xt[10]+z[11]*xt[11]+z[12]*xt[12]+z[13]*xt[13]+z[14]*xt[14]+z[15]*xt[15]+z[16]*xt[16]+z[17]*xt[17]+z[18]*xt[18])

            nothing
        end

        z0  = vcat(λT[:], zeros(np))
        dz0 = vcat(dλT[:], (λT')*Fp)   # For some reason (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired

        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            # NOTE: Changes signs to match what I had in my manual calculations, seems correct now
            # Gp = adj_sol.u[end][end-np+1:end] + (((adj_sol.u[end][1:end-np]')*Fdx(0.0))*xp0)[:]
            # Gp = adj_sol.u[end-N_trans][nx+1:nx+np] + (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)[:]
            Gp = adj_sol.u[end][nx+1:nx+np] .+ (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)
            # return 0.0
        end

        function get_Gp_debug(adj_sol::DAESolution)
            integral = adj_sol.u[end][nx+1:nx+np]
            term = (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)
            Gp = integral .+ term
            return Gp, integral, term
        end

        function get_term_debug(adj_sol::DAESolution, xps::Matrix{Float64}, times::AbstractVector{Float64})
            term = zeros(length(adj_sol.u))
            for ind=eachindex(adj_sol.u)
                term[ind] = ((adj_sol.u[end+1-ind][1:nx]')*Fdx(times[ind]))*xps[:,ind]
            end
            return term
        end

        # The only adjoint variables whose derivatives occur in the adjoint equations are those adjoint variables corresponding to the differential equations of the nominal system
        dvars = vcat(fill(true, 18), fill(false, 15), fill(true, np))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 for adjoint delta robot is: $r0"

        debugs = (get_Gp_debug, get_term_debug)

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp, debugs
    end
end

function delta_robot_gc_adjoint_L1only(u::Function, w::Function, θ::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Matrix{Float64}, dx::Function, dx2::Function)#::Tuple{Model, Function, Function}
    let L0 = θ[1], L1 = θ[2], L2 = θ[3], L3 = θ[4], LC1 = θ[5], LC2 = θ[6], M1 = θ[7], M2 = θ[8], M3 = θ[9], J1 = θ[10], J2 = θ[11], g = 0.0, γ = θ[13]
        
        (nx, np) = size(xp0)
        # np = size(xp0,2)
        @assert (np == 1) "delta_robot_gc_adjoint_γonly is hard-coded to only handle one parameter L1, make sure to pass correct xp0"
        # nx = size(xp0,1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        Fx = [-L1*cos(x(T)[1])*dx(T)[26]-L1*cos(x(T)[1])*dx(T)[29]-L1*sin(x(T)[1])*dx(T)[27]-L1*sin(x(T)[1])*dx(T)[30]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   -L2*cos(x(T)[2])*dx(T)[26]-L2*cos(x(T)[2])*dx(T)[29]-L2*cos(x(T)[3])*sin(x(T)[2])*dx(T)[27]-L2*cos(x(T)[3])*sin(x(T)[2])*dx(T)[30]-L2*sin(x(T)[2])*sin(x(T)[3])*dx(T)[25]-L2*sin(x(T)[2])*sin(x(T)[3])*dx(T)[28]   L2*cos(x(T)[2])*cos(x(T)[3])*dx(T)[25]+L2*cos(x(T)[2])*cos(x(T)[3])*dx(T)[28]-L2*cos(x(T)[2])*sin(x(T)[3])*dx(T)[27]-L2*cos(x(T)[2])*sin(x(T)[3])*dx(T)[30]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   L2*cos(x(T)[2])*cos(x(T)[3])*dx(T)[25]+L2*cos(x(T)[2])*cos(x(T)[3])*dx(T)[28]-L2*cos(x(T)[2])*sin(x(T)[3])*dx(T)[27]-L2*cos(x(T)[2])*sin(x(T)[3])*dx(T)[30]   -L2*cos(x(T)[3])*sin(x(T)[2])*dx(T)[27]-L2*cos(x(T)[3])*sin(x(T)[2])*dx(T)[30]-L2*sin(x(T)[2])*sin(x(T)[3])*dx(T)[25]-L2*sin(x(T)[2])*sin(x(T)[3])*dx(T)[28]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   L1*sin(x(T)[4])*dx(T)[27]-(L1*cos(x(T)[4])*dx(T)[26])*0.5-(sqrt(3)*L1*cos(x(T)[4])*dx(T)[25])*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   L2*cos(x(T)[6])*sin(x(T)[5])*dx(T)[27]-dx(T)[26]*((L2*cos(x(T)[5]))*0.5-(sqrt(3)*L2*sin(x(T)[5])*sin(x(T)[6]))*0.5)-dx(T)[25]*((sqrt(3)*L2*cos(x(T)[5]))*0.5+(L2*sin(x(T)[5])*sin(x(T)[6]))*0.5)   (L2*cos(x(T)[5])*cos(x(T)[6])*dx(T)[25])*0.5+L2*cos(x(T)[5])*sin(x(T)[6])*dx(T)[27]-(sqrt(3)*L2*cos(x(T)[5])*cos(x(T)[6])*dx(T)[26])*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   (L2*cos(x(T)[5])*cos(x(T)[6])*dx(T)[25])*0.5+L2*cos(x(T)[5])*sin(x(T)[6])*dx(T)[27]-(sqrt(3)*L2*cos(x(T)[5])*cos(x(T)[6])*dx(T)[26])*0.5   L2*cos(x(T)[6])*sin(x(T)[5])*dx(T)[27]-(L2*sin(x(T)[5])*sin(x(T)[6])*dx(T)[25])*0.5+(sqrt(3)*L2*sin(x(T)[5])*sin(x(T)[6])*dx(T)[26])*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   L1*sin(x(T)[7])*dx(T)[30]-(L1*cos(x(T)[7])*dx(T)[29])*0.5+(sqrt(3)*L1*cos(x(T)[7])*dx(T)[28])*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   dx(T)[28]*((sqrt(3)*L2*cos(x(T)[8]))*0.5-(L2*sin(x(T)[8])*sin(x(T)[9]))*0.5)-dx(T)[29]*((L2*cos(x(T)[8]))*0.5+(sqrt(3)*L2*sin(x(T)[8])*sin(x(T)[9]))*0.5)+L2*cos(x(T)[9])*sin(x(T)[8])*dx(T)[30]   (L2*cos(x(T)[8])*cos(x(T)[9])*dx(T)[28])*0.5+L2*cos(x(T)[8])*sin(x(T)[9])*dx(T)[30]+(sqrt(3)*L2*cos(x(T)[8])*cos(x(T)[9])*dx(T)[29])*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x(T)[8])*cos(x(T)[9])*dx(T)[28])*0.5+L2*cos(x(T)[8])*sin(x(T)[9])*dx(T)[30]+(sqrt(3)*L2*cos(x(T)[8])*cos(x(T)[9])*dx(T)[29])*0.5   L2*cos(x(T)[9])*sin(x(T)[8])*dx(T)[30]-(L2*sin(x(T)[8])*sin(x(T)[9])*dx(T)[28])*0.5-(sqrt(3)*L2*sin(x(T)[8])*sin(x(T)[9])*dx(T)[29])*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0*sin(x(T)[1])*(L1*(M2+M3)+LC1*M1)+L1*cos(x(T)[1])*dx(T)[20]+L1*cos(x(T)[1])*dx(T)[23]+L1*sin(x(T)[1])*dx(T)[21]+L1*sin(x(T)[1])*dx(T)[24]+L1*dx(T)[11]*(L2*M3+LC2*M2)*(cos(x(T)[1])*sin(x(T)[2])-cos(x(T)[2])*cos(x(T)[3])*sin(x(T)[1]))+L1*x(T)[11]^2*(L2*M3+LC2*M2)*(cos(x(T)[1])*cos(x(T)[2])+cos(x(T)[3])*sin(x(T)[1])*sin(x(T)[2]))+L1*cos(x(T)[3])*sin(x(T)[1])*sin(x(T)[2])*x(T)[12]^2*(L2*M3+LC2*M2)+L1*sin(x(T)[1])*sin(x(T)[2])*sin(x(T)[3])*dx(T)[12]*(L2*M3+LC2*M2)+2*L1*cos(x(T)[2])*sin(x(T)[1])*sin(x(T)[3])*x(T)[11]*x(T)[12]*(L2*M3+LC2*M2)   L1*dx(T)[11]*(L2*M3+LC2*M2)*(cos(x(T)[2])*sin(x(T)[1])-cos(x(T)[1])*cos(x(T)[3])*sin(x(T)[2]))-L1*x(T)[11]^2*(L2*M3+LC2*M2)*(sin(x(T)[1])*sin(x(T)[2])+cos(x(T)[1])*cos(x(T)[2])*cos(x(T)[3]))-L1*cos(x(T)[1])*cos(x(T)[2])*cos(x(T)[3])*x(T)[12]^2*(L2*M3+LC2*M2)-L1*cos(x(T)[1])*cos(x(T)[2])*sin(x(T)[3])*dx(T)[12]*(L2*M3+LC2*M2)+2*L1*cos(x(T)[1])*sin(x(T)[2])*sin(x(T)[3])*x(T)[11]*x(T)[12]*(L2*M3+LC2*M2)   L1*cos(x(T)[1])*sin(x(T)[2])*sin(x(T)[3])*x(T)[11]^2*(L2*M3+LC2*M2)+L1*cos(x(T)[1])*sin(x(T)[2])*sin(x(T)[3])*x(T)[12]^2*(L2*M3+LC2*M2)-L1*cos(x(T)[1])*cos(x(T)[2])*sin(x(T)[3])*dx(T)[11]*(L2*M3+LC2*M2)-L1*cos(x(T)[1])*cos(x(T)[3])*sin(x(T)[2])*dx(T)[12]*(L2*M3+LC2*M2)-2*L1*cos(x(T)[1])*cos(x(T)[2])*cos(x(T)[3])*x(T)[11]*x(T)[12]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   γ   2*L1*x(T)[11]*(L2*M3+LC2*M2)*(cos(x(T)[2])*sin(x(T)[1])-cos(x(T)[1])*cos(x(T)[3])*sin(x(T)[2]))-2*L1*cos(x(T)[1])*cos(x(T)[2])*sin(x(T)[3])*x(T)[12]*(L2*M3+LC2*M2)   -2*L1*cos(x(T)[1])*cos(x(T)[2])*sin(x(T)[3])*x(T)[11]*(L2*M3+LC2*M2)-2*L1*cos(x(T)[1])*cos(x(T)[3])*sin(x(T)[2])*x(T)[12]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            L1*dx(T)[10]*(L2*M3+LC2*M2)*(cos(x(T)[1])*sin(x(T)[2])-cos(x(T)[2])*cos(x(T)[3])*sin(x(T)[1]))-L1*x(T)[10]^2*(L2*M3+LC2*M2)*(sin(x(T)[1])*sin(x(T)[2])+cos(x(T)[1])*cos(x(T)[2])*cos(x(T)[3]))   sin(x(T)[2])^2*x(T)[12]^2*(J2+L2^2*M3+LC2^2*M2)-cos(x(T)[2])^2*x(T)[12]^2*(J2+L2^2*M3+LC2^2*M2)+L2*cos(x(T)[2])*dx(T)[20]+L2*cos(x(T)[2])*dx(T)[23]+L2*cos(x(T)[3])*sin(x(T)[2])*dx(T)[21]+L2*cos(x(T)[3])*sin(x(T)[2])*dx(T)[24]+L2*sin(x(T)[2])*sin(x(T)[3])*dx(T)[19]+L2*sin(x(T)[2])*sin(x(T)[3])*dx(T)[22]+L1*dx(T)[10]*(L2*M3+LC2*M2)*(cos(x(T)[2])*sin(x(T)[1])-cos(x(T)[1])*cos(x(T)[3])*sin(x(T)[2]))+0.0*cos(x(T)[3])*sin(x(T)[2])*(L2*M3+LC2*M2)+L1*x(T)[10]^2*(L2*M3+LC2*M2)*(cos(x(T)[1])*cos(x(T)[2])+cos(x(T)[3])*sin(x(T)[1])*sin(x(T)[2]))   L2*cos(x(T)[2])*sin(x(T)[3])*dx(T)[21]-L2*cos(x(T)[2])*cos(x(T)[3])*dx(T)[22]-L2*cos(x(T)[2])*cos(x(T)[3])*dx(T)[19]+L2*cos(x(T)[2])*sin(x(T)[3])*dx(T)[24]+0.0*cos(x(T)[2])*sin(x(T)[3])*(L2*M3+LC2*M2)+L1*cos(x(T)[2])*sin(x(T)[1])*sin(x(T)[3])*x(T)[10]^2*(L2*M3+LC2*M2)-L1*cos(x(T)[1])*cos(x(T)[2])*sin(x(T)[3])*dx(T)[10]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   2*L1*x(T)[10]*(L2*M3+LC2*M2)*(cos(x(T)[1])*sin(x(T)[2])-cos(x(T)[2])*cos(x(T)[3])*sin(x(T)[1]))   γ   -2*cos(x(T)[2])*sin(x(T)[2])*x(T)[12]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            L1*cos(x(T)[1])*sin(x(T)[2])*sin(x(T)[3])*x(T)[10]^2*(L2*M3+LC2*M2)+L1*sin(x(T)[1])*sin(x(T)[2])*sin(x(T)[3])*dx(T)[10]*(L2*M3+LC2*M2)   L2*cos(x(T)[2])*sin(x(T)[3])*dx(T)[21]-L2*cos(x(T)[2])*cos(x(T)[3])*dx(T)[22]-L2*cos(x(T)[2])*cos(x(T)[3])*dx(T)[19]+L2*cos(x(T)[2])*sin(x(T)[3])*dx(T)[24]+2*cos(x(T)[2])*sin(x(T)[2])*dx(T)[12]*(J2+L2^2*M3+LC2^2*M2)+2*cos(2*x(T)[2])*x(T)[11]*x(T)[12]*(J2+L2^2*M3+LC2^2*M2)+0.0*cos(x(T)[2])*sin(x(T)[3])*(L2*M3+LC2*M2)+L1*cos(x(T)[2])*sin(x(T)[1])*sin(x(T)[3])*x(T)[10]^2*(L2*M3+LC2*M2)-L1*cos(x(T)[1])*cos(x(T)[2])*sin(x(T)[3])*dx(T)[10]*(L2*M3+LC2*M2)   L2*cos(x(T)[3])*sin(x(T)[2])*dx(T)[21]+L2*cos(x(T)[3])*sin(x(T)[2])*dx(T)[24]+L2*sin(x(T)[2])*sin(x(T)[3])*dx(T)[19]+L2*sin(x(T)[2])*sin(x(T)[3])*dx(T)[22]+0.0*cos(x(T)[3])*sin(x(T)[2])*(L2*M3+LC2*M2)+L1*cos(x(T)[3])*sin(x(T)[1])*sin(x(T)[2])*x(T)[10]^2*(L2*M3+LC2*M2)-L1*cos(x(T)[1])*cos(x(T)[3])*sin(x(T)[2])*dx(T)[10]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   2*L1*sin(x(T)[1])*sin(x(T)[2])*sin(x(T)[3])*x(T)[10]*(L2*M3+LC2*M2)   sin(2*x(T)[2])*x(T)[12]*(J2+L2^2*M3+LC2^2*M2)   γ+sin(2*x(T)[2])*x(T)[11]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0*sin(x(T)[4])*(L1*(M2+M3)+LC1*M1)+(L1*cos(x(T)[4])*dx(T)[20])*0.5-L1*sin(x(T)[4])*dx(T)[21]+(sqrt(3)*L1*cos(x(T)[4])*dx(T)[19])*0.5+L1*dx(T)[14]*(L2*M3+LC2*M2)*(cos(x(T)[4])*sin(x(T)[5])-cos(x(T)[5])*cos(x(T)[6])*sin(x(T)[4]))+L1*x(T)[14]^2*(L2*M3+LC2*M2)*(cos(x(T)[4])*cos(x(T)[5])+cos(x(T)[6])*sin(x(T)[4])*sin(x(T)[5]))+L1*cos(x(T)[6])*sin(x(T)[4])*sin(x(T)[5])*x(T)[15]^2*(L2*M3+LC2*M2)+L1*sin(x(T)[4])*sin(x(T)[5])*sin(x(T)[6])*dx(T)[15]*(L2*M3+LC2*M2)+2*L1*cos(x(T)[5])*sin(x(T)[4])*sin(x(T)[6])*x(T)[14]*x(T)[15]*(L2*M3+LC2*M2)   L1*dx(T)[14]*(L2*M3+LC2*M2)*(cos(x(T)[5])*sin(x(T)[4])-cos(x(T)[4])*cos(x(T)[6])*sin(x(T)[5]))-L1*x(T)[14]^2*(L2*M3+LC2*M2)*(sin(x(T)[4])*sin(x(T)[5])+cos(x(T)[4])*cos(x(T)[5])*cos(x(T)[6]))-L1*cos(x(T)[4])*cos(x(T)[5])*cos(x(T)[6])*x(T)[15]^2*(L2*M3+LC2*M2)-L1*cos(x(T)[4])*cos(x(T)[5])*sin(x(T)[6])*dx(T)[15]*(L2*M3+LC2*M2)+2*L1*cos(x(T)[4])*sin(x(T)[5])*sin(x(T)[6])*x(T)[14]*x(T)[15]*(L2*M3+LC2*M2)   L1*cos(x(T)[4])*sin(x(T)[5])*sin(x(T)[6])*x(T)[14]^2*(L2*M3+LC2*M2)+L1*cos(x(T)[4])*sin(x(T)[5])*sin(x(T)[6])*x(T)[15]^2*(L2*M3+LC2*M2)-L1*cos(x(T)[4])*cos(x(T)[5])*sin(x(T)[6])*dx(T)[14]*(L2*M3+LC2*M2)-L1*cos(x(T)[4])*cos(x(T)[6])*sin(x(T)[5])*dx(T)[15]*(L2*M3+LC2*M2)-2*L1*cos(x(T)[4])*cos(x(T)[5])*cos(x(T)[6])*x(T)[14]*x(T)[15]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   γ   2*L1*x(T)[14]*(L2*M3+LC2*M2)*(cos(x(T)[5])*sin(x(T)[4])-cos(x(T)[4])*cos(x(T)[6])*sin(x(T)[5]))-2*L1*cos(x(T)[4])*cos(x(T)[5])*sin(x(T)[6])*x(T)[15]*(L2*M3+LC2*M2)   -2*L1*cos(x(T)[4])*cos(x(T)[5])*sin(x(T)[6])*x(T)[14]*(L2*M3+LC2*M2)-2*L1*cos(x(T)[4])*cos(x(T)[6])*sin(x(T)[5])*x(T)[15]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   L1*dx(T)[13]*(L2*M3+LC2*M2)*(cos(x(T)[4])*sin(x(T)[5])-cos(x(T)[5])*cos(x(T)[6])*sin(x(T)[4]))-L1*x(T)[13]^2*(L2*M3+LC2*M2)*(sin(x(T)[4])*sin(x(T)[5])+cos(x(T)[4])*cos(x(T)[5])*cos(x(T)[6]))   dx(T)[19]*((sqrt(3)*L2*cos(x(T)[5]))*0.5+(L2*sin(x(T)[5])*sin(x(T)[6]))*0.5)+dx(T)[20]*((L2*cos(x(T)[5]))*0.5-(sqrt(3)*L2*sin(x(T)[5])*sin(x(T)[6]))*0.5)-cos(x(T)[5])^2*x(T)[15]^2*(J2+L2^2*M3+LC2^2*M2)+sin(x(T)[5])^2*x(T)[15]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(x(T)[6])*sin(x(T)[5])*dx(T)[21]+L1*dx(T)[13]*(L2*M3+LC2*M2)*(cos(x(T)[5])*sin(x(T)[4])-cos(x(T)[4])*cos(x(T)[6])*sin(x(T)[5]))+0.0*cos(x(T)[6])*sin(x(T)[5])*(L2*M3+LC2*M2)+L1*x(T)[13]^2*(L2*M3+LC2*M2)*(cos(x(T)[4])*cos(x(T)[5])+cos(x(T)[6])*sin(x(T)[4])*sin(x(T)[5]))   0.0*cos(x(T)[5])*sin(x(T)[6])*(L2*M3+LC2*M2)-L2*cos(x(T)[5])*sin(x(T)[6])*dx(T)[21]-(L2*cos(x(T)[5])*cos(x(T)[6])*dx(T)[19])*0.5+(sqrt(3)*L2*cos(x(T)[5])*cos(x(T)[6])*dx(T)[20])*0.5+L1*cos(x(T)[5])*sin(x(T)[4])*sin(x(T)[6])*x(T)[13]^2*(L2*M3+LC2*M2)-L1*cos(x(T)[4])*cos(x(T)[5])*sin(x(T)[6])*dx(T)[13]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   2*L1*x(T)[13]*(L2*M3+LC2*M2)*(cos(x(T)[4])*sin(x(T)[5])-cos(x(T)[5])*cos(x(T)[6])*sin(x(T)[4]))   γ   -2*cos(x(T)[5])*sin(x(T)[5])*x(T)[15]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   L1*cos(x(T)[4])*sin(x(T)[5])*sin(x(T)[6])*x(T)[13]^2*(L2*M3+LC2*M2)+L1*sin(x(T)[4])*sin(x(T)[5])*sin(x(T)[6])*dx(T)[13]*(L2*M3+LC2*M2)   2*cos(x(T)[5])*sin(x(T)[5])*dx(T)[15]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(x(T)[5])*sin(x(T)[6])*dx(T)[21]-(L2*cos(x(T)[5])*cos(x(T)[6])*dx(T)[19])*0.5+2*cos(2*x(T)[5])*x(T)[14]*x(T)[15]*(J2+L2^2*M3+LC2^2*M2)+0.0*cos(x(T)[5])*sin(x(T)[6])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(x(T)[5])*cos(x(T)[6])*dx(T)[20])*0.5+L1*cos(x(T)[5])*sin(x(T)[4])*sin(x(T)[6])*x(T)[13]^2*(L2*M3+LC2*M2)-L1*cos(x(T)[4])*cos(x(T)[5])*sin(x(T)[6])*dx(T)[13]*(L2*M3+LC2*M2)   (L2*sin(x(T)[5])*sin(x(T)[6])*dx(T)[19])*0.5-L2*cos(x(T)[6])*sin(x(T)[5])*dx(T)[21]+0.0*cos(x(T)[6])*sin(x(T)[5])*(L2*M3+LC2*M2)-(sqrt(3)*L2*sin(x(T)[5])*sin(x(T)[6])*dx(T)[20])*0.5+L1*cos(x(T)[6])*sin(x(T)[4])*sin(x(T)[5])*x(T)[13]^2*(L2*M3+LC2*M2)-L1*cos(x(T)[4])*cos(x(T)[6])*sin(x(T)[5])*dx(T)[13]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   2*L1*sin(x(T)[4])*sin(x(T)[5])*sin(x(T)[6])*x(T)[13]*(L2*M3+LC2*M2)   sin(2*x(T)[5])*x(T)[15]*(J2+L2^2*M3+LC2^2*M2)   γ+sin(2*x(T)[5])*x(T)[14]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0*sin(x(T)[7])*(L1*(M2+M3)+LC1*M1)+(L1*cos(x(T)[7])*dx(T)[23])*0.5-L1*sin(x(T)[7])*dx(T)[24]-(sqrt(3)*L1*cos(x(T)[7])*dx(T)[22])*0.5+L1*dx(T)[17]*(L2*M3+LC2*M2)*(cos(x(T)[7])*sin(x(T)[8])-cos(x(T)[8])*cos(x(T)[9])*sin(x(T)[7]))+L1*x(T)[17]^2*(L2*M3+LC2*M2)*(cos(x(T)[7])*cos(x(T)[8])+cos(x(T)[9])*sin(x(T)[7])*sin(x(T)[8]))+L1*cos(x(T)[9])*sin(x(T)[7])*sin(x(T)[8])*x(T)[18]^2*(L2*M3+LC2*M2)+L1*sin(x(T)[7])*sin(x(T)[8])*sin(x(T)[9])*dx(T)[18]*(L2*M3+LC2*M2)+2*L1*cos(x(T)[8])*sin(x(T)[7])*sin(x(T)[9])*x(T)[17]*x(T)[18]*(L2*M3+LC2*M2)   L1*dx(T)[17]*(L2*M3+LC2*M2)*(cos(x(T)[8])*sin(x(T)[7])-cos(x(T)[7])*cos(x(T)[9])*sin(x(T)[8]))-L1*x(T)[17]^2*(L2*M3+LC2*M2)*(sin(x(T)[7])*sin(x(T)[8])+cos(x(T)[7])*cos(x(T)[8])*cos(x(T)[9]))-L1*cos(x(T)[7])*cos(x(T)[8])*cos(x(T)[9])*x(T)[18]^2*(L2*M3+LC2*M2)-L1*cos(x(T)[7])*cos(x(T)[8])*sin(x(T)[9])*dx(T)[18]*(L2*M3+LC2*M2)+2*L1*cos(x(T)[7])*sin(x(T)[8])*sin(x(T)[9])*x(T)[17]*x(T)[18]*(L2*M3+LC2*M2)   L1*cos(x(T)[7])*sin(x(T)[8])*sin(x(T)[9])*x(T)[17]^2*(L2*M3+LC2*M2)+L1*cos(x(T)[7])*sin(x(T)[8])*sin(x(T)[9])*x(T)[18]^2*(L2*M3+LC2*M2)-L1*cos(x(T)[7])*cos(x(T)[8])*sin(x(T)[9])*dx(T)[17]*(L2*M3+LC2*M2)-L1*cos(x(T)[7])*cos(x(T)[9])*sin(x(T)[8])*dx(T)[18]*(L2*M3+LC2*M2)-2*L1*cos(x(T)[7])*cos(x(T)[8])*cos(x(T)[9])*x(T)[17]*x(T)[18]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   γ   2*L1*x(T)[17]*(L2*M3+LC2*M2)*(cos(x(T)[8])*sin(x(T)[7])-cos(x(T)[7])*cos(x(T)[9])*sin(x(T)[8]))-2*L1*cos(x(T)[7])*cos(x(T)[8])*sin(x(T)[9])*x(T)[18]*(L2*M3+LC2*M2)   -2*L1*cos(x(T)[7])*cos(x(T)[8])*sin(x(T)[9])*x(T)[17]*(L2*M3+LC2*M2)-2*L1*cos(x(T)[7])*cos(x(T)[9])*sin(x(T)[8])*x(T)[18]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   L1*dx(T)[16]*(L2*M3+LC2*M2)*(cos(x(T)[7])*sin(x(T)[8])-cos(x(T)[8])*cos(x(T)[9])*sin(x(T)[7]))-L1*x(T)[16]^2*(L2*M3+LC2*M2)*(sin(x(T)[7])*sin(x(T)[8])+cos(x(T)[7])*cos(x(T)[8])*cos(x(T)[9]))   dx(T)[23]*((L2*cos(x(T)[8]))*0.5+(sqrt(3)*L2*sin(x(T)[8])*sin(x(T)[9]))*0.5)-dx(T)[22]*((sqrt(3)*L2*cos(x(T)[8]))*0.5-(L2*sin(x(T)[8])*sin(x(T)[9]))*0.5)-cos(x(T)[8])^2*x(T)[18]^2*(J2+L2^2*M3+LC2^2*M2)+sin(x(T)[8])^2*x(T)[18]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(x(T)[9])*sin(x(T)[8])*dx(T)[24]+L1*dx(T)[16]*(L2*M3+LC2*M2)*(cos(x(T)[8])*sin(x(T)[7])-cos(x(T)[7])*cos(x(T)[9])*sin(x(T)[8]))+0.0*cos(x(T)[9])*sin(x(T)[8])*(L2*M3+LC2*M2)+L1*x(T)[16]^2*(L2*M3+LC2*M2)*(cos(x(T)[7])*cos(x(T)[8])+cos(x(T)[9])*sin(x(T)[7])*sin(x(T)[8]))   0.0*cos(x(T)[8])*sin(x(T)[9])*(L2*M3+LC2*M2)-L2*cos(x(T)[8])*sin(x(T)[9])*dx(T)[24]-(L2*cos(x(T)[8])*cos(x(T)[9])*dx(T)[22])*0.5-(sqrt(3)*L2*cos(x(T)[8])*cos(x(T)[9])*dx(T)[23])*0.5+L1*cos(x(T)[8])*sin(x(T)[7])*sin(x(T)[9])*x(T)[16]^2*(L2*M3+LC2*M2)-L1*cos(x(T)[7])*cos(x(T)[8])*sin(x(T)[9])*dx(T)[16]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   2*L1*x(T)[16]*(L2*M3+LC2*M2)*(cos(x(T)[7])*sin(x(T)[8])-cos(x(T)[8])*cos(x(T)[9])*sin(x(T)[7]))   γ   -2*cos(x(T)[8])*sin(x(T)[8])*x(T)[18]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   L1*cos(x(T)[7])*sin(x(T)[8])*sin(x(T)[9])*x(T)[16]^2*(L2*M3+LC2*M2)+L1*sin(x(T)[7])*sin(x(T)[8])*sin(x(T)[9])*dx(T)[16]*(L2*M3+LC2*M2)   2*cos(x(T)[8])*sin(x(T)[8])*dx(T)[18]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(x(T)[8])*sin(x(T)[9])*dx(T)[24]-(L2*cos(x(T)[8])*cos(x(T)[9])*dx(T)[22])*0.5+2*cos(2*x(T)[8])*x(T)[17]*x(T)[18]*(J2+L2^2*M3+LC2^2*M2)+0.0*cos(x(T)[8])*sin(x(T)[9])*(L2*M3+LC2*M2)-(sqrt(3)*L2*cos(x(T)[8])*cos(x(T)[9])*dx(T)[23])*0.5+L1*cos(x(T)[8])*sin(x(T)[7])*sin(x(T)[9])*x(T)[16]^2*(L2*M3+LC2*M2)-L1*cos(x(T)[7])*cos(x(T)[8])*sin(x(T)[9])*dx(T)[16]*(L2*M3+LC2*M2)   (L2*sin(x(T)[8])*sin(x(T)[9])*dx(T)[22])*0.5-L2*cos(x(T)[9])*sin(x(T)[8])*dx(T)[24]+0.0*cos(x(T)[9])*sin(x(T)[8])*(L2*M3+LC2*M2)+(sqrt(3)*L2*sin(x(T)[8])*sin(x(T)[9])*dx(T)[23])*0.5+L1*cos(x(T)[9])*sin(x(T)[7])*sin(x(T)[8])*x(T)[16]^2*(L2*M3+LC2*M2)-L1*cos(x(T)[7])*cos(x(T)[9])*sin(x(T)[8])*dx(T)[16]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   2*L1*sin(x(T)[7])*sin(x(T)[8])*sin(x(T)[9])*x(T)[16]*(L2*M3+LC2*M2)   sin(2*x(T)[8])*x(T)[18]*(J2+L2^2*M3+LC2^2*M2)   γ+sin(2*x(T)[8])*x(T)[17]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   L2*cos(x(T)[2])*sin(x(T)[3])   L2*cos(x(T)[3])*sin(x(T)[2])   -(sqrt(3)*L1*sin(x(T)[4]))*0.5   (L2*cos(x(T)[5])*sin(x(T)[6]))*0.5-(sqrt(3)*L2*sin(x(T)[5]))*0.5   (L2*cos(x(T)[6])*sin(x(T)[5]))*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            -L1*sin(x(T)[1])   -L2*sin(x(T)[2])   0.0   -(L1*sin(x(T)[4]))*0.5   -(L2*sin(x(T)[5]))*0.5-(sqrt(3)*L2*cos(x(T)[5])*sin(x(T)[6]))*0.5   -(sqrt(3)*L2*cos(x(T)[6])*sin(x(T)[5]))*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            L1*cos(x(T)[1])   L2*cos(x(T)[2])*cos(x(T)[3])   -L2*sin(x(T)[2])*sin(x(T)[3])   -L1*cos(x(T)[4])   -L2*cos(x(T)[5])*cos(x(T)[6])   L2*sin(x(T)[5])*sin(x(T)[6])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   L2*cos(x(T)[2])*sin(x(T)[3])   L2*cos(x(T)[3])*sin(x(T)[2])   0.0   0.0   0.0   (sqrt(3)*L1*sin(x(T)[7]))*0.5   (L2*cos(x(T)[8])*sin(x(T)[9]))*0.5+(sqrt(3)*L2*sin(x(T)[8]))*0.5   (L2*cos(x(T)[9])*sin(x(T)[8]))*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            -L1*sin(x(T)[1])   -L2*sin(x(T)[2])   0.0   0.0   0.0   0.0   -(L1*sin(x(T)[7]))*0.5   (sqrt(3)*L2*cos(x(T)[8])*sin(x(T)[9]))*0.5-(L2*sin(x(T)[8]))*0.5   (sqrt(3)*L2*cos(x(T)[9])*sin(x(T)[8]))*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            L1*cos(x(T)[1])   L2*cos(x(T)[2])*cos(x(T)[3])   -L2*sin(x(T)[2])*sin(x(T)[3])   0.0   0.0   0.0   -L1*cos(x(T)[7])   -L2*cos(x(T)[8])*cos(x(T)[9])   L2*sin(x(T)[8])*sin(x(T)[9])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   L2*cos(x(T)[2])*cos(x(T)[3])*x(T)[12]-L2*sin(x(T)[2])*sin(x(T)[3])*x(T)[11]   L2*cos(x(T)[2])*cos(x(T)[3])*x(T)[11]-L2*sin(x(T)[2])*sin(x(T)[3])*x(T)[12]   -(sqrt(3)*L1*cos(x(T)[4])*x(T)[13])*0.5   (L2*cos(x(T)[5])*cos(x(T)[6])*x(T)[15])*0.5-(sqrt(3)*L2*cos(x(T)[5])*x(T)[14])*0.5-(L2*sin(x(T)[5])*sin(x(T)[6])*x(T)[14])*0.5   (L2*cos(x(T)[5])*cos(x(T)[6])*x(T)[14])*0.5-(L2*sin(x(T)[5])*sin(x(T)[6])*x(T)[15])*0.5   0.0   0.0   0.0   0.0   L2*cos(x(T)[2])*sin(x(T)[3])   L2*cos(x(T)[3])*sin(x(T)[2])   -(sqrt(3)*L1*sin(x(T)[4]))*0.5   (L2*cos(x(T)[5])*sin(x(T)[6]))*0.5-(sqrt(3)*L2*sin(x(T)[5]))*0.5   (L2*cos(x(T)[6])*sin(x(T)[5]))*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            -L1*cos(x(T)[1])*x(T)[10]   -L2*cos(x(T)[2])*x(T)[11]   0.0   -(L1*cos(x(T)[4])*x(T)[13])*0.5   (sqrt(3)*L2*sin(x(T)[5])*sin(x(T)[6])*x(T)[14])*0.5-(sqrt(3)*L2*cos(x(T)[5])*cos(x(T)[6])*x(T)[15])*0.5-(L2*cos(x(T)[5])*x(T)[14])*0.5   (sqrt(3)*L2*sin(x(T)[5])*sin(x(T)[6])*x(T)[15])*0.5-(sqrt(3)*L2*cos(x(T)[5])*cos(x(T)[6])*x(T)[14])*0.5   0.0   0.0   0.0   -L1*sin(x(T)[1])   -L2*sin(x(T)[2])   0.0   -(L1*sin(x(T)[4]))*0.5   -(L2*sin(x(T)[5]))*0.5-(sqrt(3)*L2*cos(x(T)[5])*sin(x(T)[6]))*0.5   -(sqrt(3)*L2*cos(x(T)[6])*sin(x(T)[5]))*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            -L1*sin(x(T)[1])*x(T)[10]   -L2*cos(x(T)[3])*sin(x(T)[2])*x(T)[11]-L2*cos(x(T)[2])*sin(x(T)[3])*x(T)[12]   -L2*cos(x(T)[2])*sin(x(T)[3])*x(T)[11]-L2*cos(x(T)[3])*sin(x(T)[2])*x(T)[12]   L1*sin(x(T)[4])*x(T)[13]   L2*cos(x(T)[6])*sin(x(T)[5])*x(T)[14]+L2*cos(x(T)[5])*sin(x(T)[6])*x(T)[15]   L2*cos(x(T)[5])*sin(x(T)[6])*x(T)[14]+L2*cos(x(T)[6])*sin(x(T)[5])*x(T)[15]   0.0   0.0   0.0   L1*cos(x(T)[1])   L2*cos(x(T)[2])*cos(x(T)[3])   -L2*sin(x(T)[2])*sin(x(T)[3])   -L1*cos(x(T)[4])   -L2*cos(x(T)[5])*cos(x(T)[6])   L2*sin(x(T)[5])*sin(x(T)[6])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   L2*cos(x(T)[2])*cos(x(T)[3])*x(T)[12]-L2*sin(x(T)[2])*sin(x(T)[3])*x(T)[11]   L2*cos(x(T)[2])*cos(x(T)[3])*x(T)[11]-L2*sin(x(T)[2])*sin(x(T)[3])*x(T)[12]   0.0   0.0   0.0   (sqrt(3)*L1*cos(x(T)[7])*x(T)[16])*0.5   (L2*cos(x(T)[8])*cos(x(T)[9])*x(T)[18])*0.5+(sqrt(3)*L2*cos(x(T)[8])*x(T)[17])*0.5-(L2*sin(x(T)[8])*sin(x(T)[9])*x(T)[17])*0.5   (L2*cos(x(T)[8])*cos(x(T)[9])*x(T)[17])*0.5-(L2*sin(x(T)[8])*sin(x(T)[9])*x(T)[18])*0.5   0.0   L2*cos(x(T)[2])*sin(x(T)[3])   L2*cos(x(T)[3])*sin(x(T)[2])   0.0   0.0   0.0   (sqrt(3)*L1*sin(x(T)[7]))*0.5   (L2*cos(x(T)[8])*sin(x(T)[9]))*0.5+(sqrt(3)*L2*sin(x(T)[8]))*0.5   (L2*cos(x(T)[9])*sin(x(T)[8]))*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            -L1*cos(x(T)[1])*x(T)[10]   -L2*cos(x(T)[2])*x(T)[11]   0.0   0.0   0.0   0.0   -(L1*cos(x(T)[7])*x(T)[16])*0.5   (sqrt(3)*L2*cos(x(T)[8])*cos(x(T)[9])*x(T)[18])*0.5-(L2*cos(x(T)[8])*x(T)[17])*0.5-(sqrt(3)*L2*sin(x(T)[8])*sin(x(T)[9])*x(T)[17])*0.5   (sqrt(3)*L2*cos(x(T)[8])*cos(x(T)[9])*x(T)[17])*0.5-(sqrt(3)*L2*sin(x(T)[8])*sin(x(T)[9])*x(T)[18])*0.5   -L1*sin(x(T)[1])   -L2*sin(x(T)[2])   0.0   0.0   0.0   0.0   -(L1*sin(x(T)[7]))*0.5   (sqrt(3)*L2*cos(x(T)[8])*sin(x(T)[9]))*0.5-(L2*sin(x(T)[8]))*0.5   (sqrt(3)*L2*cos(x(T)[9])*sin(x(T)[8]))*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            -L1*sin(x(T)[1])*x(T)[10]   -L2*cos(x(T)[3])*sin(x(T)[2])*x(T)[11]-L2*cos(x(T)[2])*sin(x(T)[3])*x(T)[12]   -L2*cos(x(T)[2])*sin(x(T)[3])*x(T)[11]-L2*cos(x(T)[3])*sin(x(T)[2])*x(T)[12]   0.0   0.0   0.0   L1*sin(x(T)[7])*x(T)[16]   L2*cos(x(T)[9])*sin(x(T)[8])*x(T)[17]+L2*cos(x(T)[8])*sin(x(T)[9])*x(T)[18]   L2*cos(x(T)[8])*sin(x(T)[9])*x(T)[17]+L2*cos(x(T)[9])*sin(x(T)[8])*x(T)[18]   L1*cos(x(T)[1])   L2*cos(x(T)[2])*cos(x(T)[3])   -L2*sin(x(T)[2])*sin(x(T)[3])   0.0   0.0   0.0   -L1*cos(x(T)[7])   -L2*cos(x(T)[8])*cos(x(T)[9])   L2*sin(x(T)[8])*sin(x(T)[9])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   -L2*cos(x(T)[2])*sin(x(T)[3])   -L2*cos(x(T)[3])*sin(x(T)[2])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0
            L1*sin(x(T)[1])   L2*sin(x(T)[2])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0
            -L1*cos(x(T)[1])   -L2*cos(x(T)[2])*cos(x(T)[3])   L2*sin(x(T)[2])*sin(x(T)[3])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0]
        Fdx = t -> [ # TODO: Should we use sparse matrices? Is that worthwile? Might improve efficiency
            1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*sin(x(t)[1])   L1*cos(x(t)[1])   0.0   -L1*sin(x(t)[1])   L1*cos(x(t)[1])   0.0   0.0   0.0
            0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L2*cos(x(t)[2])*sin(x(t)[3])   -L2*sin(x(t)[2])   L2*cos(x(t)[2])*cos(x(t)[3])   L2*cos(x(t)[2])*sin(x(t)[3])   -L2*sin(x(t)[2])   L2*cos(x(t)[2])*cos(x(t)[3])   0.0   0.0   0.0
            0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L2*cos(x(t)[3])*sin(x(t)[2])   0.0   -L2*sin(x(t)[2])*sin(x(t)[3])   L2*cos(x(t)[3])*sin(x(t)[2])   0.0   -L2*sin(x(t)[2])*sin(x(t)[3])   0.0   0.0   0.0
            0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -(sqrt(3)*L1*sin(x(t)[4]))*0.5   -(L1*sin(x(t)[4]))*0.5   -L1*cos(x(t)[4])   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x(t)[5])*sin(x(t)[6]))*0.5-(sqrt(3)*L2*sin(x(t)[5]))*0.5   -(L2*sin(x(t)[5]))*0.5-(sqrt(3)*L2*cos(x(t)[5])*sin(x(t)[6]))*0.5   -L2*cos(x(t)[5])*cos(x(t)[6])   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x(t)[6])*sin(x(t)[5]))*0.5   -(sqrt(3)*L2*cos(x(t)[6])*sin(x(t)[5]))*0.5   L2*sin(x(t)[5])*sin(x(t)[6])   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (sqrt(3)*L1*sin(x(t)[7]))*0.5   -(L1*sin(x(t)[7]))*0.5   -L1*cos(x(t)[7])   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x(t)[8])*sin(x(t)[9]))*0.5+(sqrt(3)*L2*sin(x(t)[8]))*0.5   (sqrt(3)*L2*cos(x(t)[8])*sin(x(t)[9]))*0.5-(L2*sin(x(t)[8]))*0.5   -L2*cos(x(t)[8])*cos(x(t)[9])   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x(t)[9])*sin(x(t)[8]))*0.5   (sqrt(3)*L2*cos(x(t)[9])*sin(x(t)[8]))*0.5   L2*sin(x(t)[8])*sin(x(t)[9])   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(x(t)[1])*sin(x(t)[2])+cos(x(t)[1])*cos(x(t)[2])*cos(x(t)[3]))   -L1*cos(x(t)[1])*sin(x(t)[2])*sin(x(t)[3])*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*sin(x(t)[1])   -L1*cos(x(t)[1])   0.0   L1*sin(x(t)[1])   -L1*cos(x(t)[1])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*(L2*M3+LC2*M2)*(sin(x(t)[1])*sin(x(t)[2])+cos(x(t)[1])*cos(x(t)[2])*cos(x(t)[3]))   J2+L2^2*M3+LC2^2*M2   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L2*cos(x(t)[2])*sin(x(t)[3])   L2*sin(x(t)[2])   -L2*cos(x(t)[2])*cos(x(t)[3])   -L2*cos(x(t)[2])*sin(x(t)[3])   L2*sin(x(t)[2])   -L2*cos(x(t)[2])*cos(x(t)[3])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*cos(x(t)[1])*sin(x(t)[2])*sin(x(t)[3])*(L2*M3+LC2*M2)   0.0   sin(x(t)[2])^2*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   -L2*cos(x(t)[3])*sin(x(t)[2])   0.0   L2*sin(x(t)[2])*sin(x(t)[3])   -L2*cos(x(t)[3])*sin(x(t)[2])   0.0   L2*sin(x(t)[2])*sin(x(t)[3])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(x(t)[4])*sin(x(t)[5])+cos(x(t)[4])*cos(x(t)[5])*cos(x(t)[6]))   -L1*cos(x(t)[4])*sin(x(t)[5])*sin(x(t)[6])*(L2*M3+LC2*M2)   0.0   0.0   0.0   (sqrt(3)*L1*sin(x(t)[4]))*0.5   (L1*sin(x(t)[4]))*0.5   L1*cos(x(t)[4])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*(L2*M3+LC2*M2)*(sin(x(t)[4])*sin(x(t)[5])+cos(x(t)[4])*cos(x(t)[5])*cos(x(t)[6]))   J2+L2^2*M3+LC2^2*M2   0.0   0.0   0.0   0.0   (sqrt(3)*L2*sin(x(t)[5]))*0.5-(L2*cos(x(t)[5])*sin(x(t)[6]))*0.5   (L2*sin(x(t)[5]))*0.5+(sqrt(3)*L2*cos(x(t)[5])*sin(x(t)[6]))*0.5   L2*cos(x(t)[5])*cos(x(t)[6])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*cos(x(t)[4])*sin(x(t)[5])*sin(x(t)[6])*(L2*M3+LC2*M2)   0.0   sin(x(t)[5])^2*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   -(L2*cos(x(t)[6])*sin(x(t)[5]))*0.5   (sqrt(3)*L2*cos(x(t)[6])*sin(x(t)[5]))*0.5   -L2*sin(x(t)[5])*sin(x(t)[6])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(x(t)[7])*sin(x(t)[8])+cos(x(t)[7])*cos(x(t)[8])*cos(x(t)[9]))   -L1*cos(x(t)[7])*sin(x(t)[8])*sin(x(t)[9])*(L2*M3+LC2*M2)   0.0   0.0   0.0   -(sqrt(3)*L1*sin(x(t)[7]))*0.5   (L1*sin(x(t)[7]))*0.5   L1*cos(x(t)[7])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*(L2*M3+LC2*M2)*(sin(x(t)[7])*sin(x(t)[8])+cos(x(t)[7])*cos(x(t)[8])*cos(x(t)[9]))   J2+L2^2*M3+LC2^2*M2   0.0   0.0   0.0   0.0   -(L2*cos(x(t)[8])*sin(x(t)[9]))*0.5-(sqrt(3)*L2*sin(x(t)[8]))*0.5   (L2*sin(x(t)[8]))*0.5-(sqrt(3)*L2*cos(x(t)[8])*sin(x(t)[9]))*0.5   L2*cos(x(t)[8])*cos(x(t)[9])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*cos(x(t)[7])*sin(x(t)[8])*sin(x(t)[9])*(L2*M3+LC2*M2)   0.0   sin(x(t)[8])^2*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   -(L2*cos(x(t)[9])*sin(x(t)[8]))*0.5   -(sqrt(3)*L2*cos(x(t)[9])*sin(x(t)[8]))*0.5   -L2*sin(x(t)[8])*sin(x(t)[9])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0]
        Fddx = [
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*cos(x(T)[1])*dx(T)[1]   -L1*sin(x(T)[1])*dx(T)[1]   0.0   -L1*cos(x(T)[1])*dx(T)[1]   -L1*sin(x(T)[1])*dx(T)[1]   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L2*cos(x(T)[2])*cos(x(T)[3])*dx(T)[3]-L2*sin(x(T)[2])*sin(x(T)[3])*dx(T)[2]   -L2*cos(x(T)[2])*dx(T)[2]   -L2*cos(x(T)[3])*sin(x(T)[2])*dx(T)[2]-L2*cos(x(T)[2])*sin(x(T)[3])*dx(T)[3]   L2*cos(x(T)[2])*cos(x(T)[3])*dx(T)[3]-L2*sin(x(T)[2])*sin(x(T)[3])*dx(T)[2]   -L2*cos(x(T)[2])*dx(T)[2]   -L2*cos(x(T)[3])*sin(x(T)[2])*dx(T)[2]-L2*cos(x(T)[2])*sin(x(T)[3])*dx(T)[3]   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L2*cos(x(T)[2])*cos(x(T)[3])*dx(T)[2]-L2*sin(x(T)[2])*sin(x(T)[3])*dx(T)[3]   0.0   -L2*cos(x(T)[2])*sin(x(T)[3])*dx(T)[2]-L2*cos(x(T)[3])*sin(x(T)[2])*dx(T)[3]   L2*cos(x(T)[2])*cos(x(T)[3])*dx(T)[2]-L2*sin(x(T)[2])*sin(x(T)[3])*dx(T)[3]   0.0   -L2*cos(x(T)[2])*sin(x(T)[3])*dx(T)[2]-L2*cos(x(T)[3])*sin(x(T)[2])*dx(T)[3]   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -(sqrt(3)*L1*cos(x(T)[4])*dx(T)[4])*0.5   -(L1*cos(x(T)[4])*dx(T)[4])*0.5   L1*sin(x(T)[4])*dx(T)[4]   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x(T)[5])*cos(x(T)[6])*dx(T)[6])*0.5-(sqrt(3)*L2*cos(x(T)[5])*dx(T)[5])*0.5-(L2*sin(x(T)[5])*sin(x(T)[6])*dx(T)[5])*0.5   (sqrt(3)*L2*sin(x(T)[5])*sin(x(T)[6])*dx(T)[5])*0.5-(sqrt(3)*L2*cos(x(T)[5])*cos(x(T)[6])*dx(T)[6])*0.5-(L2*cos(x(T)[5])*dx(T)[5])*0.5   L2*cos(x(T)[6])*sin(x(T)[5])*dx(T)[5]+L2*cos(x(T)[5])*sin(x(T)[6])*dx(T)[6]   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x(T)[5])*cos(x(T)[6])*dx(T)[5])*0.5-(L2*sin(x(T)[5])*sin(x(T)[6])*dx(T)[6])*0.5   (sqrt(3)*L2*sin(x(T)[5])*sin(x(T)[6])*dx(T)[6])*0.5-(sqrt(3)*L2*cos(x(T)[5])*cos(x(T)[6])*dx(T)[5])*0.5   L2*cos(x(T)[5])*sin(x(T)[6])*dx(T)[5]+L2*cos(x(T)[6])*sin(x(T)[5])*dx(T)[6]   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (sqrt(3)*L1*cos(x(T)[7])*dx(T)[7])*0.5   -(L1*cos(x(T)[7])*dx(T)[7])*0.5   L1*sin(x(T)[7])*dx(T)[7]   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x(T)[8])*cos(x(T)[9])*dx(T)[9])*0.5+(sqrt(3)*L2*cos(x(T)[8])*dx(T)[8])*0.5-(L2*sin(x(T)[8])*sin(x(T)[9])*dx(T)[8])*0.5   (sqrt(3)*L2*cos(x(T)[8])*cos(x(T)[9])*dx(T)[9])*0.5-(L2*cos(x(T)[8])*dx(T)[8])*0.5-(sqrt(3)*L2*sin(x(T)[8])*sin(x(T)[9])*dx(T)[8])*0.5   L2*cos(x(T)[9])*sin(x(T)[8])*dx(T)[8]+L2*cos(x(T)[8])*sin(x(T)[9])*dx(T)[9]   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x(T)[8])*cos(x(T)[9])*dx(T)[8])*0.5-(L2*sin(x(T)[8])*sin(x(T)[9])*dx(T)[9])*0.5   (sqrt(3)*L2*cos(x(T)[8])*cos(x(T)[9])*dx(T)[8])*0.5-(sqrt(3)*L2*sin(x(T)[8])*sin(x(T)[9])*dx(T)[9])*0.5   L2*cos(x(T)[8])*sin(x(T)[9])*dx(T)[8]+L2*cos(x(T)[9])*sin(x(T)[8])*dx(T)[9]   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*(L2*M3+LC2*M2)*(cos(x(T)[2])*cos(x(T)[3])*sin(x(T)[1])*dx(T)[1]-cos(x(T)[2])*sin(x(T)[1])*dx(T)[2]-cos(x(T)[1])*sin(x(T)[2])*dx(T)[1]+cos(x(T)[1])*cos(x(T)[3])*sin(x(T)[2])*dx(T)[2]+cos(x(T)[1])*cos(x(T)[2])*sin(x(T)[3])*dx(T)[3])   L1*sin(x(T)[1])*sin(x(T)[2])*sin(x(T)[3])*dx(T)[1]*(L2*M3+LC2*M2)-L1*cos(x(T)[1])*cos(x(T)[3])*sin(x(T)[2])*dx(T)[3]*(L2*M3+LC2*M2)-L1*cos(x(T)[1])*cos(x(T)[2])*sin(x(T)[3])*dx(T)[2]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*cos(x(T)[1])*dx(T)[1]   L1*sin(x(T)[1])*dx(T)[1]   0.0   L1*cos(x(T)[1])*dx(T)[1]   L1*sin(x(T)[1])*dx(T)[1]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*(L2*M3+LC2*M2)*(cos(x(T)[2])*cos(x(T)[3])*sin(x(T)[1])*dx(T)[1]-cos(x(T)[2])*sin(x(T)[1])*dx(T)[2]-cos(x(T)[1])*sin(x(T)[2])*dx(T)[1]+cos(x(T)[1])*cos(x(T)[3])*sin(x(T)[2])*dx(T)[2]+cos(x(T)[1])*cos(x(T)[2])*sin(x(T)[3])*dx(T)[3])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L2*sin(x(T)[2])*sin(x(T)[3])*dx(T)[2]-L2*cos(x(T)[2])*cos(x(T)[3])*dx(T)[3]   L2*cos(x(T)[2])*dx(T)[2]   L2*cos(x(T)[3])*sin(x(T)[2])*dx(T)[2]+L2*cos(x(T)[2])*sin(x(T)[3])*dx(T)[3]   L2*sin(x(T)[2])*sin(x(T)[3])*dx(T)[2]-L2*cos(x(T)[2])*cos(x(T)[3])*dx(T)[3]   L2*cos(x(T)[2])*dx(T)[2]   L2*cos(x(T)[3])*sin(x(T)[2])*dx(T)[2]+L2*cos(x(T)[2])*sin(x(T)[3])*dx(T)[3]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*sin(x(T)[1])*sin(x(T)[2])*sin(x(T)[3])*dx(T)[1]*(L2*M3+LC2*M2)-L1*cos(x(T)[1])*cos(x(T)[3])*sin(x(T)[2])*dx(T)[3]*(L2*M3+LC2*M2)-L1*cos(x(T)[1])*cos(x(T)[2])*sin(x(T)[3])*dx(T)[2]*(L2*M3+LC2*M2)   0.0   2*cos(x(T)[2])*sin(x(T)[2])*dx(T)[2]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   L2*sin(x(T)[2])*sin(x(T)[3])*dx(T)[3]-L2*cos(x(T)[2])*cos(x(T)[3])*dx(T)[2]   0.0   L2*cos(x(T)[2])*sin(x(T)[3])*dx(T)[2]+L2*cos(x(T)[3])*sin(x(T)[2])*dx(T)[3]   L2*sin(x(T)[2])*sin(x(T)[3])*dx(T)[3]-L2*cos(x(T)[2])*cos(x(T)[3])*dx(T)[2]   0.0   L2*cos(x(T)[2])*sin(x(T)[3])*dx(T)[2]+L2*cos(x(T)[3])*sin(x(T)[2])*dx(T)[3]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*(L2*M3+LC2*M2)*(cos(x(T)[5])*cos(x(T)[6])*sin(x(T)[4])*dx(T)[4]-cos(x(T)[5])*sin(x(T)[4])*dx(T)[5]-cos(x(T)[4])*sin(x(T)[5])*dx(T)[4]+cos(x(T)[4])*cos(x(T)[6])*sin(x(T)[5])*dx(T)[5]+cos(x(T)[4])*cos(x(T)[5])*sin(x(T)[6])*dx(T)[6])   L1*sin(x(T)[4])*sin(x(T)[5])*sin(x(T)[6])*dx(T)[4]*(L2*M3+LC2*M2)-L1*cos(x(T)[4])*cos(x(T)[6])*sin(x(T)[5])*dx(T)[6]*(L2*M3+LC2*M2)-L1*cos(x(T)[4])*cos(x(T)[5])*sin(x(T)[6])*dx(T)[5]*(L2*M3+LC2*M2)   0.0   0.0   0.0   (sqrt(3)*L1*cos(x(T)[4])*dx(T)[4])*0.5   (L1*cos(x(T)[4])*dx(T)[4])*0.5   -L1*sin(x(T)[4])*dx(T)[4]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*(L2*M3+LC2*M2)*(cos(x(T)[5])*cos(x(T)[6])*sin(x(T)[4])*dx(T)[4]-cos(x(T)[5])*sin(x(T)[4])*dx(T)[5]-cos(x(T)[4])*sin(x(T)[5])*dx(T)[4]+cos(x(T)[4])*cos(x(T)[6])*sin(x(T)[5])*dx(T)[5]+cos(x(T)[4])*cos(x(T)[5])*sin(x(T)[6])*dx(T)[6])   0.0   0.0   0.0   0.0   0.0   (sqrt(3)*L2*cos(x(T)[5])*dx(T)[5])*0.5-(L2*cos(x(T)[5])*cos(x(T)[6])*dx(T)[6])*0.5+(L2*sin(x(T)[5])*sin(x(T)[6])*dx(T)[5])*0.5   (L2*cos(x(T)[5])*dx(T)[5])*0.5+(sqrt(3)*L2*cos(x(T)[5])*cos(x(T)[6])*dx(T)[6])*0.5-(sqrt(3)*L2*sin(x(T)[5])*sin(x(T)[6])*dx(T)[5])*0.5   -L2*cos(x(T)[6])*sin(x(T)[5])*dx(T)[5]-L2*cos(x(T)[5])*sin(x(T)[6])*dx(T)[6]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*sin(x(T)[4])*sin(x(T)[5])*sin(x(T)[6])*dx(T)[4]*(L2*M3+LC2*M2)-L1*cos(x(T)[4])*cos(x(T)[6])*sin(x(T)[5])*dx(T)[6]*(L2*M3+LC2*M2)-L1*cos(x(T)[4])*cos(x(T)[5])*sin(x(T)[6])*dx(T)[5]*(L2*M3+LC2*M2)   0.0   2*cos(x(T)[5])*sin(x(T)[5])*dx(T)[5]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   (L2*sin(x(T)[5])*sin(x(T)[6])*dx(T)[6])*0.5-(L2*cos(x(T)[5])*cos(x(T)[6])*dx(T)[5])*0.5   (sqrt(3)*L2*cos(x(T)[5])*cos(x(T)[6])*dx(T)[5])*0.5-(sqrt(3)*L2*sin(x(T)[5])*sin(x(T)[6])*dx(T)[6])*0.5   -L2*cos(x(T)[5])*sin(x(T)[6])*dx(T)[5]-L2*cos(x(T)[6])*sin(x(T)[5])*dx(T)[6]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*(L2*M3+LC2*M2)*(cos(x(T)[8])*cos(x(T)[9])*sin(x(T)[7])*dx(T)[7]-cos(x(T)[8])*sin(x(T)[7])*dx(T)[8]-cos(x(T)[7])*sin(x(T)[8])*dx(T)[7]+cos(x(T)[7])*cos(x(T)[9])*sin(x(T)[8])*dx(T)[8]+cos(x(T)[7])*cos(x(T)[8])*sin(x(T)[9])*dx(T)[9])   L1*sin(x(T)[7])*sin(x(T)[8])*sin(x(T)[9])*dx(T)[7]*(L2*M3+LC2*M2)-L1*cos(x(T)[7])*cos(x(T)[9])*sin(x(T)[8])*dx(T)[9]*(L2*M3+LC2*M2)-L1*cos(x(T)[7])*cos(x(T)[8])*sin(x(T)[9])*dx(T)[8]*(L2*M3+LC2*M2)   0.0   0.0   0.0   -(sqrt(3)*L1*cos(x(T)[7])*dx(T)[7])*0.5   (L1*cos(x(T)[7])*dx(T)[7])*0.5   -L1*sin(x(T)[7])*dx(T)[7]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*(L2*M3+LC2*M2)*(cos(x(T)[8])*cos(x(T)[9])*sin(x(T)[7])*dx(T)[7]-cos(x(T)[8])*sin(x(T)[7])*dx(T)[8]-cos(x(T)[7])*sin(x(T)[8])*dx(T)[7]+cos(x(T)[7])*cos(x(T)[9])*sin(x(T)[8])*dx(T)[8]+cos(x(T)[7])*cos(x(T)[8])*sin(x(T)[9])*dx(T)[9])   0.0   0.0   0.0   0.0   0.0   (L2*sin(x(T)[8])*sin(x(T)[9])*dx(T)[8])*0.5-(sqrt(3)*L2*cos(x(T)[8])*dx(T)[8])*0.5-(L2*cos(x(T)[8])*cos(x(T)[9])*dx(T)[9])*0.5   (L2*cos(x(T)[8])*dx(T)[8])*0.5-(sqrt(3)*L2*cos(x(T)[8])*cos(x(T)[9])*dx(T)[9])*0.5+(sqrt(3)*L2*sin(x(T)[8])*sin(x(T)[9])*dx(T)[8])*0.5   -L2*cos(x(T)[9])*sin(x(T)[8])*dx(T)[8]-L2*cos(x(T)[8])*sin(x(T)[9])*dx(T)[9]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*sin(x(T)[7])*sin(x(T)[8])*sin(x(T)[9])*dx(T)[7]*(L2*M3+LC2*M2)-L1*cos(x(T)[7])*cos(x(T)[9])*sin(x(T)[8])*dx(T)[9]*(L2*M3+LC2*M2)-L1*cos(x(T)[7])*cos(x(T)[8])*sin(x(T)[9])*dx(T)[8]*(L2*M3+LC2*M2)   0.0   2*cos(x(T)[8])*sin(x(T)[8])*dx(T)[8]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   (L2*sin(x(T)[8])*sin(x(T)[9])*dx(T)[9])*0.5-(L2*cos(x(T)[8])*cos(x(T)[9])*dx(T)[8])*0.5   (sqrt(3)*L2*sin(x(T)[8])*sin(x(T)[9])*dx(T)[9])*0.5-(sqrt(3)*L2*cos(x(T)[8])*cos(x(T)[9])*dx(T)[8])*0.5   -L2*cos(x(T)[8])*sin(x(T)[9])*dx(T)[8]-L2*cos(x(T)[9])*sin(x(T)[8])*dx(T)[9]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0]
        gₓ = [0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -(2*y(T)[1]-2*x2(T)[31])/T   -(2*y(T)[2]-2*x2(T)[32])/T   -(2*y(T)[3]-2*x2(T)[33])/T]
        gdₓ = [0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -(2*dy(T)[1]-2*dx2(T)[31])/T   -(2*dy(T)[2]-2*dx2(T)[32])/T   -(2*dy(T)[3]-2*dx2(T)[33])/T]
        Fp = [
            cos(x(T)[1])*dx(T)[27]+cos(x(T)[1])*dx(T)[30]-sin(x(T)[1])*dx(T)[26]-sin(x(T)[1])*dx(T)[29]
            0.0
            0.0
            -cos(x(T)[4])*dx(T)[27]-(sin(x(T)[4])*dx(T)[26])*0.5-(sqrt(3)*sin(x(T)[4])*dx(T)[25])*0.5
            0.0
            0.0
            (sqrt(3)*sin(x(T)[7])*dx(T)[28])*0.5-(sin(x(T)[7])*dx(T)[29])*0.5-cos(x(T)[7])*dx(T)[30]
            0.0
            0.0
            sin(x(T)[1])*dx(T)[20]-cos(x(T)[1])*dx(T)[24]-cos(x(T)[1])*dx(T)[21]+sin(x(T)[1])*dx(T)[23]-0.0*cos(x(T)[1])*(M2+M3)+dx(T)[11]*(L2*M3+LC2*M2)*(sin(x(T)[1])*sin(x(T)[2])+cos(x(T)[1])*cos(x(T)[2])*cos(x(T)[3]))+2*L1*dx(T)[10]*(M2+M3)+x(T)[11]^2*(L2*M3+LC2*M2)*(cos(x(T)[2])*sin(x(T)[1])-cos(x(T)[1])*cos(x(T)[3])*sin(x(T)[2]))-cos(x(T)[1])*sin(x(T)[2])*sin(x(T)[3])*dx(T)[12]*(L2*M3+LC2*M2)-cos(x(T)[1])*cos(x(T)[3])*sin(x(T)[2])*x(T)[12]^2*(L2*M3+LC2*M2)-2*cos(x(T)[1])*cos(x(T)[2])*sin(x(T)[3])*x(T)[11]*x(T)[12]*(L2*M3+LC2*M2)
            dx(T)[10]*(L2*M3+LC2*M2)*(sin(x(T)[1])*sin(x(T)[2])+cos(x(T)[1])*cos(x(T)[2])*cos(x(T)[3]))+x(T)[10]^2*(L2*M3+LC2*M2)*(cos(x(T)[1])*sin(x(T)[2])-cos(x(T)[2])*cos(x(T)[3])*sin(x(T)[1]))
            sin(x(T)[1])*sin(x(T)[2])*sin(x(T)[3])*x(T)[10]^2*(L2*M3+LC2*M2)-cos(x(T)[1])*sin(x(T)[2])*sin(x(T)[3])*dx(T)[10]*(L2*M3+LC2*M2)
            cos(x(T)[4])*dx(T)[21]+(sin(x(T)[4])*dx(T)[20])*0.5-0.0*cos(x(T)[4])*(M2+M3)+dx(T)[14]*(L2*M3+LC2*M2)*(sin(x(T)[4])*sin(x(T)[5])+cos(x(T)[4])*cos(x(T)[5])*cos(x(T)[6]))+2*L1*dx(T)[13]*(M2+M3)+x(T)[14]^2*(L2*M3+LC2*M2)*(cos(x(T)[5])*sin(x(T)[4])-cos(x(T)[4])*cos(x(T)[6])*sin(x(T)[5]))+(sqrt(3)*sin(x(T)[4])*dx(T)[19])*0.5-cos(x(T)[4])*sin(x(T)[5])*sin(x(T)[6])*dx(T)[15]*(L2*M3+LC2*M2)-cos(x(T)[4])*cos(x(T)[6])*sin(x(T)[5])*x(T)[15]^2*(L2*M3+LC2*M2)-2*cos(x(T)[4])*cos(x(T)[5])*sin(x(T)[6])*x(T)[14]*x(T)[15]*(L2*M3+LC2*M2)
            dx(T)[13]*(L2*M3+LC2*M2)*(sin(x(T)[4])*sin(x(T)[5])+cos(x(T)[4])*cos(x(T)[5])*cos(x(T)[6]))+x(T)[13]^2*(L2*M3+LC2*M2)*(cos(x(T)[4])*sin(x(T)[5])-cos(x(T)[5])*cos(x(T)[6])*sin(x(T)[4]))
            sin(x(T)[4])*sin(x(T)[5])*sin(x(T)[6])*x(T)[13]^2*(L2*M3+LC2*M2)-cos(x(T)[4])*sin(x(T)[5])*sin(x(T)[6])*dx(T)[13]*(L2*M3+LC2*M2)
            cos(x(T)[7])*dx(T)[24]+(sin(x(T)[7])*dx(T)[23])*0.5-0.0*cos(x(T)[7])*(M2+M3)+dx(T)[17]*(L2*M3+LC2*M2)*(sin(x(T)[7])*sin(x(T)[8])+cos(x(T)[7])*cos(x(T)[8])*cos(x(T)[9]))+2*L1*dx(T)[16]*(M2+M3)+x(T)[17]^2*(L2*M3+LC2*M2)*(cos(x(T)[8])*sin(x(T)[7])-cos(x(T)[7])*cos(x(T)[9])*sin(x(T)[8]))-(sqrt(3)*sin(x(T)[7])*dx(T)[22])*0.5-cos(x(T)[7])*sin(x(T)[8])*sin(x(T)[9])*dx(T)[18]*(L2*M3+LC2*M2)-cos(x(T)[7])*cos(x(T)[9])*sin(x(T)[8])*x(T)[18]^2*(L2*M3+LC2*M2)-2*cos(x(T)[7])*cos(x(T)[8])*sin(x(T)[9])*x(T)[17]*x(T)[18]*(L2*M3+LC2*M2)
            dx(T)[16]*(L2*M3+LC2*M2)*(sin(x(T)[7])*sin(x(T)[8])+cos(x(T)[7])*cos(x(T)[8])*cos(x(T)[9]))+x(T)[16]^2*(L2*M3+LC2*M2)*(cos(x(T)[7])*sin(x(T)[8])-cos(x(T)[8])*cos(x(T)[9])*sin(x(T)[7]))
            sin(x(T)[7])*sin(x(T)[8])*sin(x(T)[9])*x(T)[16]^2*(L2*M3+LC2*M2)-cos(x(T)[7])*sin(x(T)[8])*sin(x(T)[9])*dx(T)[16]*(L2*M3+LC2*M2)
            (sqrt(3)*cos(x(T)[4]))*0.5
            cos(x(T)[1])+cos(x(T)[4])*0.5
            sin(x(T)[1])-sin(x(T)[4])
            -(sqrt(3)*cos(x(T)[7]))*0.5
            cos(x(T)[1])+cos(x(T)[7])*0.5
            sin(x(T)[1])-sin(x(T)[7])
            -(sqrt(3)*sin(x(T)[4])*x(T)[13])*0.5
            -sin(x(T)[1])*x(T)[10]-(sin(x(T)[4])*x(T)[13])*0.5
            cos(x(T)[1])*x(T)[10]-cos(x(T)[4])*x(T)[13]
            (sqrt(3)*sin(x(T)[7])*x(T)[16])*0.5
            -sin(x(T)[1])*x(T)[10]-(sin(x(T)[7])*x(T)[16])*0.5
            cos(x(T)[1])*x(T)[10]-cos(x(T)[7])*x(T)[16]
            0.0
            -cos(x(T)[1])
            -sin(x(T)[1])]


        # NOTE: Convention is used that derivatives wrt to θ stack by adding new cols
        # while derivatives wrt to x stack by adding new rows

        ###################### INITIALIZING ADJOINT SYSTEM ####################
        # Indices 1-18 are differential (d), while 19-33 are algebraic (a)
        dinds = 1:18
        ainds = 19:33
        λT  = zeros(33)
        dλT = zeros(33)
        temp = (-gₓ)/vcat(Fdx(T)[dinds,:], -Fx[ainds,:])
        λT[dinds] = zeros(length(dinds))
        λT[ainds] = temp[ainds]
        dλT[dinds] = temp[dinds]
        temp = (-gdₓ + (dλT[dinds]')*(Fx[dinds,:] - Fddx[dinds,:] - Fdx(T)[dinds,:]) + (λT[ainds]')*Fx[ainds,:])/vcat(Fdx(T)[dinds,:], -Fx[ainds,:])
        dλT[ainds] = temp[ainds]

        function f!(res, dz, z, _, t)
            # ut = u(t)
            # wt = w(t)
            xt = x(t)
            dxt = dx(t)
            yt = y(t)

            # Adjoint equations
            res[1] = dz[1]-z[11]*(L1*dxt[10]*(L2*M3+LC2*M2)*(cos(xt[1])*sin(xt[2])-cos(xt[2])*cos(xt[3])*sin(xt[1]))-L1*xt[10]^2*(L2*M3+LC2*M2)*(sin(xt[1])*sin(xt[2])+cos(xt[1])*cos(xt[2])*cos(xt[3])))-z[10]*(0.0*sin(xt[1])*(L1*(M2+M3)+LC1*M1)+L1*cos(xt[1])*dxt[20]+L1*cos(xt[1])*dxt[23]+L1*sin(xt[1])*dxt[21]+L1*sin(xt[1])*dxt[24]+L1*dxt[11]*(L2*M3+LC2*M2)*(cos(xt[1])*sin(xt[2])-cos(xt[2])*cos(xt[3])*sin(xt[1]))+L1*xt[11]^2*(L2*M3+LC2*M2)*(cos(xt[1])*cos(xt[2])+cos(xt[3])*sin(xt[1])*sin(xt[2]))+L1*cos(xt[3])*sin(xt[1])*sin(xt[2])*xt[12]^2*(L2*M3+LC2*M2)+L1*sin(xt[1])*sin(xt[2])*sin(xt[3])*dxt[12]*(L2*M3+LC2*M2)+2*L1*cos(xt[2])*sin(xt[1])*sin(xt[3])*xt[11]*xt[12]*(L2*M3+LC2*M2))-z[12]*(L1*cos(xt[1])*sin(xt[2])*sin(xt[3])*xt[10]^2*(L2*M3+LC2*M2)+L1*sin(xt[1])*sin(xt[2])*sin(xt[3])*dxt[10]*(L2*M3+LC2*M2))+z[1]*(L1*cos(xt[1])*dxt[26]+L1*cos(xt[1])*dxt[29]+L1*sin(xt[1])*dxt[27]+L1*sin(xt[1])*dxt[30])-L1*cos(xt[1])*z[21]-L1*cos(xt[1])*z[24]+L1*cos(xt[1])*z[33]+L1*sin(xt[1])*z[20]+L1*sin(xt[1])*z[23]-L1*sin(xt[1])*z[32]+L1*cos(xt[1])*z[26]*xt[10]+L1*cos(xt[1])*z[29]*xt[10]+L1*sin(xt[1])*z[27]*xt[10]+L1*sin(xt[1])*z[30]*xt[10]
            res[2] = dz[2]-z[25]*(L2*cos(xt[2])*cos(xt[3])*xt[12]-L2*sin(xt[2])*sin(xt[3])*xt[11])+z[27]*(L2*cos(xt[3])*sin(xt[2])*xt[11]+L2*cos(xt[2])*sin(xt[3])*xt[12])-z[28]*(L2*cos(xt[2])*cos(xt[3])*xt[12]-L2*sin(xt[2])*sin(xt[3])*xt[11])+z[30]*(L2*cos(xt[3])*sin(xt[2])*xt[11]+L2*cos(xt[2])*sin(xt[3])*xt[12])-z[3]*(L2*cos(xt[2])*cos(xt[3])*dxt[25]+L2*cos(xt[2])*cos(xt[3])*dxt[28]-L2*cos(xt[2])*sin(xt[3])*dxt[27]-L2*cos(xt[2])*sin(xt[3])*dxt[30])-z[12]*(L2*cos(xt[2])*sin(xt[3])*dxt[21]-L2*cos(xt[2])*cos(xt[3])*dxt[22]-L2*cos(xt[2])*cos(xt[3])*dxt[19]+L2*cos(xt[2])*sin(xt[3])*dxt[24]+2*cos(xt[2])*sin(xt[2])*dxt[12]*(J2+L2^2*M3+LC2^2*M2)+2*cos(2*xt[2])*xt[11]*xt[12]*(J2+L2^2*M3+LC2^2*M2)+0.0*cos(xt[2])*sin(xt[3])*(L2*M3+LC2*M2)+L1*cos(xt[2])*sin(xt[1])*sin(xt[3])*xt[10]^2*(L2*M3+LC2*M2)-L1*cos(xt[1])*cos(xt[2])*sin(xt[3])*dxt[10]*(L2*M3+LC2*M2))-z[11]*(sin(xt[2])^2*xt[12]^2*(J2+L2^2*M3+LC2^2*M2)-cos(xt[2])^2*xt[12]^2*(J2+L2^2*M3+LC2^2*M2)+L2*cos(xt[2])*dxt[20]+L2*cos(xt[2])*dxt[23]+L2*cos(xt[3])*sin(xt[2])*dxt[21]+L2*cos(xt[3])*sin(xt[2])*dxt[24]+L2*sin(xt[2])*sin(xt[3])*dxt[19]+L2*sin(xt[2])*sin(xt[3])*dxt[22]+L1*dxt[10]*(L2*M3+LC2*M2)*(cos(xt[2])*sin(xt[1])-cos(xt[1])*cos(xt[3])*sin(xt[2]))+0.0*cos(xt[3])*sin(xt[2])*(L2*M3+LC2*M2)+L1*xt[10]^2*(L2*M3+LC2*M2)*(cos(xt[1])*cos(xt[2])+cos(xt[3])*sin(xt[1])*sin(xt[2])))+z[10]*(L1*xt[11]^2*(L2*M3+LC2*M2)*(sin(xt[1])*sin(xt[2])+cos(xt[1])*cos(xt[2])*cos(xt[3]))-L1*dxt[11]*(L2*M3+LC2*M2)*(cos(xt[2])*sin(xt[1])-cos(xt[1])*cos(xt[3])*sin(xt[2]))+L1*cos(xt[1])*cos(xt[2])*cos(xt[3])*xt[12]^2*(L2*M3+LC2*M2)+L1*cos(xt[1])*cos(xt[2])*sin(xt[3])*dxt[12]*(L2*M3+LC2*M2)-2*L1*cos(xt[1])*sin(xt[2])*sin(xt[3])*xt[11]*xt[12]*(L2*M3+LC2*M2))+z[2]*(L2*cos(xt[2])*dxt[26]+L2*cos(xt[2])*dxt[29]+L2*cos(xt[3])*sin(xt[2])*dxt[27]+L2*cos(xt[3])*sin(xt[2])*dxt[30]+L2*sin(xt[2])*sin(xt[3])*dxt[25]+L2*sin(xt[2])*sin(xt[3])*dxt[28])+L2*sin(xt[2])*z[20]+L2*sin(xt[2])*z[23]-L2*sin(xt[2])*z[32]-L2*cos(xt[2])*cos(xt[3])*z[21]-L2*cos(xt[2])*cos(xt[3])*z[24]+L2*cos(xt[2])*cos(xt[3])*z[33]-L2*cos(xt[2])*sin(xt[3])*z[19]-L2*cos(xt[2])*sin(xt[3])*z[22]+L2*cos(xt[2])*sin(xt[3])*z[31]+L2*cos(xt[2])*z[26]*xt[11]+L2*cos(xt[2])*z[29]*xt[11]
            res[3] = dz[3]-z[12]*(L2*cos(xt[3])*sin(xt[2])*dxt[21]+L2*cos(xt[3])*sin(xt[2])*dxt[24]+L2*sin(xt[2])*sin(xt[3])*dxt[19]+L2*sin(xt[2])*sin(xt[3])*dxt[22]+0.0*cos(xt[3])*sin(xt[2])*(L2*M3+LC2*M2)+L1*cos(xt[3])*sin(xt[1])*sin(xt[2])*xt[10]^2*(L2*M3+LC2*M2)-L1*cos(xt[1])*cos(xt[3])*sin(xt[2])*dxt[10]*(L2*M3+LC2*M2))-z[25]*(L2*cos(xt[2])*cos(xt[3])*xt[11]-L2*sin(xt[2])*sin(xt[3])*xt[12])+z[27]*(L2*cos(xt[2])*sin(xt[3])*xt[11]+L2*cos(xt[3])*sin(xt[2])*xt[12])-z[28]*(L2*cos(xt[2])*cos(xt[3])*xt[11]-L2*sin(xt[2])*sin(xt[3])*xt[12])+z[30]*(L2*cos(xt[2])*sin(xt[3])*xt[11]+L2*cos(xt[3])*sin(xt[2])*xt[12])-z[2]*(L2*cos(xt[2])*cos(xt[3])*dxt[25]+L2*cos(xt[2])*cos(xt[3])*dxt[28]-L2*cos(xt[2])*sin(xt[3])*dxt[27]-L2*cos(xt[2])*sin(xt[3])*dxt[30])+z[10]*(L1*cos(xt[1])*cos(xt[2])*sin(xt[3])*dxt[11]*(L2*M3+LC2*M2)-L1*cos(xt[1])*sin(xt[2])*sin(xt[3])*xt[12]^2*(L2*M3+LC2*M2)-L1*cos(xt[1])*sin(xt[2])*sin(xt[3])*xt[11]^2*(L2*M3+LC2*M2)+L1*cos(xt[1])*cos(xt[3])*sin(xt[2])*dxt[12]*(L2*M3+LC2*M2)+2*L1*cos(xt[1])*cos(xt[2])*cos(xt[3])*xt[11]*xt[12]*(L2*M3+LC2*M2))+z[3]*(L2*cos(xt[3])*sin(xt[2])*dxt[27]+L2*cos(xt[3])*sin(xt[2])*dxt[30]+L2*sin(xt[2])*sin(xt[3])*dxt[25]+L2*sin(xt[2])*sin(xt[3])*dxt[28])-z[11]*(L2*cos(xt[2])*sin(xt[3])*dxt[21]-L2*cos(xt[2])*cos(xt[3])*dxt[22]-L2*cos(xt[2])*cos(xt[3])*dxt[19]+L2*cos(xt[2])*sin(xt[3])*dxt[24]+0.0*cos(xt[2])*sin(xt[3])*(L2*M3+LC2*M2)+L1*cos(xt[2])*sin(xt[1])*sin(xt[3])*xt[10]^2*(L2*M3+LC2*M2)-L1*cos(xt[1])*cos(xt[2])*sin(xt[3])*dxt[10]*(L2*M3+LC2*M2))-L2*cos(xt[3])*sin(xt[2])*z[19]-L2*cos(xt[3])*sin(xt[2])*z[22]+L2*cos(xt[3])*sin(xt[2])*z[31]+L2*sin(xt[2])*sin(xt[3])*z[21]+L2*sin(xt[2])*sin(xt[3])*z[24]-L2*sin(xt[2])*sin(xt[3])*z[33]
            res[4] = dz[4]-z[14]*(L1*dxt[13]*(L2*M3+LC2*M2)*(cos(xt[4])*sin(xt[5])-cos(xt[5])*cos(xt[6])*sin(xt[4]))-L1*xt[13]^2*(L2*M3+LC2*M2)*(sin(xt[4])*sin(xt[5])+cos(xt[4])*cos(xt[5])*cos(xt[6])))+z[4]*((L1*cos(xt[4])*dxt[26])*0.5-L1*sin(xt[4])*dxt[27]+(sqrt(3)*L1*cos(xt[4])*dxt[25])*0.5)-z[15]*(L1*cos(xt[4])*sin(xt[5])*sin(xt[6])*xt[13]^2*(L2*M3+LC2*M2)+L1*sin(xt[4])*sin(xt[5])*sin(xt[6])*dxt[13]*(L2*M3+LC2*M2))-z[13]*(0.0*sin(xt[4])*(L1*(M2+M3)+LC1*M1)+(L1*cos(xt[4])*dxt[20])*0.5-L1*sin(xt[4])*dxt[21]+(sqrt(3)*L1*cos(xt[4])*dxt[19])*0.5+L1*dxt[14]*(L2*M3+LC2*M2)*(cos(xt[4])*sin(xt[5])-cos(xt[5])*cos(xt[6])*sin(xt[4]))+L1*xt[14]^2*(L2*M3+LC2*M2)*(cos(xt[4])*cos(xt[5])+cos(xt[6])*sin(xt[4])*sin(xt[5]))+L1*cos(xt[6])*sin(xt[4])*sin(xt[5])*xt[15]^2*(L2*M3+LC2*M2)+L1*sin(xt[4])*sin(xt[5])*sin(xt[6])*dxt[15]*(L2*M3+LC2*M2)+2*L1*cos(xt[5])*sin(xt[4])*sin(xt[6])*xt[14]*xt[15]*(L2*M3+LC2*M2))+L1*cos(xt[4])*z[21]+(L1*sin(xt[4])*z[20])*0.5+(sqrt(3)*L1*sin(xt[4])*z[19])*0.5+(L1*cos(xt[4])*z[26]*xt[13])*0.5-L1*sin(xt[4])*z[27]*xt[13]+(sqrt(3)*L1*cos(xt[4])*z[25]*xt[13])*0.5
            res[5] = dz[5]-z[27]*(L2*cos(xt[6])*sin(xt[5])*xt[14]+L2*cos(xt[5])*sin(xt[6])*xt[15])+z[25]*((sqrt(3)*L2*cos(xt[5])*xt[14])*0.5-(L2*cos(xt[5])*cos(xt[6])*xt[15])*0.5+(L2*sin(xt[5])*sin(xt[6])*xt[14])*0.5)-z[6]*((L2*cos(xt[5])*cos(xt[6])*dxt[25])*0.5+L2*cos(xt[5])*sin(xt[6])*dxt[27]-(sqrt(3)*L2*cos(xt[5])*cos(xt[6])*dxt[26])*0.5)-z[19]*((L2*cos(xt[5])*sin(xt[6]))*0.5-(sqrt(3)*L2*sin(xt[5]))*0.5)+z[20]*((L2*sin(xt[5]))*0.5+(sqrt(3)*L2*cos(xt[5])*sin(xt[6]))*0.5)+z[5]*(dxt[25]*((sqrt(3)*L2*cos(xt[5]))*0.5+(L2*sin(xt[5])*sin(xt[6]))*0.5)+dxt[26]*((L2*cos(xt[5]))*0.5-(sqrt(3)*L2*sin(xt[5])*sin(xt[6]))*0.5)-L2*cos(xt[6])*sin(xt[5])*dxt[27])-z[15]*(2*cos(xt[5])*sin(xt[5])*dxt[15]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(xt[5])*sin(xt[6])*dxt[21]-(L2*cos(xt[5])*cos(xt[6])*dxt[19])*0.5+2*cos(2*xt[5])*xt[14]*xt[15]*(J2+L2^2*M3+LC2^2*M2)+0.0*cos(xt[5])*sin(xt[6])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(xt[5])*cos(xt[6])*dxt[20])*0.5+L1*cos(xt[5])*sin(xt[4])*sin(xt[6])*xt[13]^2*(L2*M3+LC2*M2)-L1*cos(xt[4])*cos(xt[5])*sin(xt[6])*dxt[13]*(L2*M3+LC2*M2))+z[26]*((L2*cos(xt[5])*xt[14])*0.5+(sqrt(3)*L2*cos(xt[5])*cos(xt[6])*xt[15])*0.5-(sqrt(3)*L2*sin(xt[5])*sin(xt[6])*xt[14])*0.5)+z[13]*(L1*xt[14]^2*(L2*M3+LC2*M2)*(sin(xt[4])*sin(xt[5])+cos(xt[4])*cos(xt[5])*cos(xt[6]))-L1*dxt[14]*(L2*M3+LC2*M2)*(cos(xt[5])*sin(xt[4])-cos(xt[4])*cos(xt[6])*sin(xt[5]))+L1*cos(xt[4])*cos(xt[5])*cos(xt[6])*xt[15]^2*(L2*M3+LC2*M2)+L1*cos(xt[4])*cos(xt[5])*sin(xt[6])*dxt[15]*(L2*M3+LC2*M2)-2*L1*cos(xt[4])*sin(xt[5])*sin(xt[6])*xt[14]*xt[15]*(L2*M3+LC2*M2))-z[14]*(dxt[19]*((sqrt(3)*L2*cos(xt[5]))*0.5+(L2*sin(xt[5])*sin(xt[6]))*0.5)+dxt[20]*((L2*cos(xt[5]))*0.5-(sqrt(3)*L2*sin(xt[5])*sin(xt[6]))*0.5)-cos(xt[5])^2*xt[15]^2*(J2+L2^2*M3+LC2^2*M2)+sin(xt[5])^2*xt[15]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(xt[6])*sin(xt[5])*dxt[21]+L1*dxt[13]*(L2*M3+LC2*M2)*(cos(xt[5])*sin(xt[4])-cos(xt[4])*cos(xt[6])*sin(xt[5]))+0.0*cos(xt[6])*sin(xt[5])*(L2*M3+LC2*M2)+L1*xt[13]^2*(L2*M3+LC2*M2)*(cos(xt[4])*cos(xt[5])+cos(xt[6])*sin(xt[4])*sin(xt[5])))+L2*cos(xt[5])*cos(xt[6])*z[21]
            res[6] = dz[6]+z[14]*((L2*cos(xt[5])*cos(xt[6])*dxt[19])*0.5+L2*cos(xt[5])*sin(xt[6])*dxt[21]-0.0*cos(xt[5])*sin(xt[6])*(L2*M3+LC2*M2)-(sqrt(3)*L2*cos(xt[5])*cos(xt[6])*dxt[20])*0.5-L1*cos(xt[5])*sin(xt[4])*sin(xt[6])*xt[13]^2*(L2*M3+LC2*M2)+L1*cos(xt[4])*cos(xt[5])*sin(xt[6])*dxt[13]*(L2*M3+LC2*M2))-z[27]*(L2*cos(xt[5])*sin(xt[6])*xt[14]+L2*cos(xt[6])*sin(xt[5])*xt[15])-z[25]*((L2*cos(xt[5])*cos(xt[6])*xt[14])*0.5-(L2*sin(xt[5])*sin(xt[6])*xt[15])*0.5)+z[15]*(L2*cos(xt[6])*sin(xt[5])*dxt[21]-(L2*sin(xt[5])*sin(xt[6])*dxt[19])*0.5-0.0*cos(xt[6])*sin(xt[5])*(L2*M3+LC2*M2)+(sqrt(3)*L2*sin(xt[5])*sin(xt[6])*dxt[20])*0.5-L1*cos(xt[6])*sin(xt[4])*sin(xt[5])*xt[13]^2*(L2*M3+LC2*M2)+L1*cos(xt[4])*cos(xt[6])*sin(xt[5])*dxt[13]*(L2*M3+LC2*M2))+z[13]*(L1*cos(xt[4])*cos(xt[5])*sin(xt[6])*dxt[14]*(L2*M3+LC2*M2)-L1*cos(xt[4])*sin(xt[5])*sin(xt[6])*xt[15]^2*(L2*M3+LC2*M2)-L1*cos(xt[4])*sin(xt[5])*sin(xt[6])*xt[14]^2*(L2*M3+LC2*M2)+L1*cos(xt[4])*cos(xt[6])*sin(xt[5])*dxt[15]*(L2*M3+LC2*M2)+2*L1*cos(xt[4])*cos(xt[5])*cos(xt[6])*xt[14]*xt[15]*(L2*M3+LC2*M2))-z[5]*((L2*cos(xt[5])*cos(xt[6])*dxt[25])*0.5+L2*cos(xt[5])*sin(xt[6])*dxt[27]-(sqrt(3)*L2*cos(xt[5])*cos(xt[6])*dxt[26])*0.5)-z[6]*(L2*cos(xt[6])*sin(xt[5])*dxt[27]-(L2*sin(xt[5])*sin(xt[6])*dxt[25])*0.5+(sqrt(3)*L2*sin(xt[5])*sin(xt[6])*dxt[26])*0.5)+z[26]*((sqrt(3)*L2*cos(xt[5])*cos(xt[6])*xt[14])*0.5-(sqrt(3)*L2*sin(xt[5])*sin(xt[6])*xt[15])*0.5)-(L2*cos(xt[6])*sin(xt[5])*z[19])*0.5-L2*sin(xt[5])*sin(xt[6])*z[21]+(sqrt(3)*L2*cos(xt[6])*sin(xt[5])*z[20])*0.5
            res[7] = dz[7]-z[17]*(L1*dxt[16]*(L2*M3+LC2*M2)*(cos(xt[7])*sin(xt[8])-cos(xt[8])*cos(xt[9])*sin(xt[7]))-L1*xt[16]^2*(L2*M3+LC2*M2)*(sin(xt[7])*sin(xt[8])+cos(xt[7])*cos(xt[8])*cos(xt[9])))-z[7]*(L1*sin(xt[7])*dxt[30]-(L1*cos(xt[7])*dxt[29])*0.5+(sqrt(3)*L1*cos(xt[7])*dxt[28])*0.5)-z[18]*(L1*cos(xt[7])*sin(xt[8])*sin(xt[9])*xt[16]^2*(L2*M3+LC2*M2)+L1*sin(xt[7])*sin(xt[8])*sin(xt[9])*dxt[16]*(L2*M3+LC2*M2))-z[16]*(0.0*sin(xt[7])*(L1*(M2+M3)+LC1*M1)+(L1*cos(xt[7])*dxt[23])*0.5-L1*sin(xt[7])*dxt[24]-(sqrt(3)*L1*cos(xt[7])*dxt[22])*0.5+L1*dxt[17]*(L2*M3+LC2*M2)*(cos(xt[7])*sin(xt[8])-cos(xt[8])*cos(xt[9])*sin(xt[7]))+L1*xt[17]^2*(L2*M3+LC2*M2)*(cos(xt[7])*cos(xt[8])+cos(xt[9])*sin(xt[7])*sin(xt[8]))+L1*cos(xt[9])*sin(xt[7])*sin(xt[8])*xt[18]^2*(L2*M3+LC2*M2)+L1*sin(xt[7])*sin(xt[8])*sin(xt[9])*dxt[18]*(L2*M3+LC2*M2)+2*L1*cos(xt[8])*sin(xt[7])*sin(xt[9])*xt[17]*xt[18]*(L2*M3+LC2*M2))+L1*cos(xt[7])*z[24]+(L1*sin(xt[7])*z[23])*0.5-(sqrt(3)*L1*sin(xt[7])*z[22])*0.5+(L1*cos(xt[7])*z[29]*xt[16])*0.5-L1*sin(xt[7])*z[30]*xt[16]-(sqrt(3)*L1*cos(xt[7])*z[28]*xt[16])*0.5
            res[8] = dz[8]-z[30]*(L2*cos(xt[9])*sin(xt[8])*xt[17]+L2*cos(xt[8])*sin(xt[9])*xt[18])-z[28]*((L2*cos(xt[8])*cos(xt[9])*xt[18])*0.5+(sqrt(3)*L2*cos(xt[8])*xt[17])*0.5-(L2*sin(xt[8])*sin(xt[9])*xt[17])*0.5)-z[9]*((L2*cos(xt[8])*cos(xt[9])*dxt[28])*0.5+L2*cos(xt[8])*sin(xt[9])*dxt[30]+(sqrt(3)*L2*cos(xt[8])*cos(xt[9])*dxt[29])*0.5)-z[22]*((L2*cos(xt[8])*sin(xt[9]))*0.5+(sqrt(3)*L2*sin(xt[8]))*0.5)+z[23]*((L2*sin(xt[8]))*0.5-(sqrt(3)*L2*cos(xt[8])*sin(xt[9]))*0.5)-z[8]*(dxt[28]*((sqrt(3)*L2*cos(xt[8]))*0.5-(L2*sin(xt[8])*sin(xt[9]))*0.5)-dxt[29]*((L2*cos(xt[8]))*0.5+(sqrt(3)*L2*sin(xt[8])*sin(xt[9]))*0.5)+L2*cos(xt[9])*sin(xt[8])*dxt[30])+z[18]*((L2*cos(xt[8])*cos(xt[9])*dxt[22])*0.5+L2*cos(xt[8])*sin(xt[9])*dxt[24]-2*cos(xt[8])*sin(xt[8])*dxt[18]*(J2+L2^2*M3+LC2^2*M2)-2*cos(2*xt[8])*xt[17]*xt[18]*(J2+L2^2*M3+LC2^2*M2)-0.0*cos(xt[8])*sin(xt[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(xt[8])*cos(xt[9])*dxt[23])*0.5-L1*cos(xt[8])*sin(xt[7])*sin(xt[9])*xt[16]^2*(L2*M3+LC2*M2)+L1*cos(xt[7])*cos(xt[8])*sin(xt[9])*dxt[16]*(L2*M3+LC2*M2))+z[29]*((L2*cos(xt[8])*xt[17])*0.5-(sqrt(3)*L2*cos(xt[8])*cos(xt[9])*xt[18])*0.5+(sqrt(3)*L2*sin(xt[8])*sin(xt[9])*xt[17])*0.5)+z[16]*(L1*xt[17]^2*(L2*M3+LC2*M2)*(sin(xt[7])*sin(xt[8])+cos(xt[7])*cos(xt[8])*cos(xt[9]))-L1*dxt[17]*(L2*M3+LC2*M2)*(cos(xt[8])*sin(xt[7])-cos(xt[7])*cos(xt[9])*sin(xt[8]))+L1*cos(xt[7])*cos(xt[8])*cos(xt[9])*xt[18]^2*(L2*M3+LC2*M2)+L1*cos(xt[7])*cos(xt[8])*sin(xt[9])*dxt[18]*(L2*M3+LC2*M2)-2*L1*cos(xt[7])*sin(xt[8])*sin(xt[9])*xt[17]*xt[18]*(L2*M3+LC2*M2))-z[17]*(dxt[23]*((L2*cos(xt[8]))*0.5+(sqrt(3)*L2*sin(xt[8])*sin(xt[9]))*0.5)-dxt[22]*((sqrt(3)*L2*cos(xt[8]))*0.5-(L2*sin(xt[8])*sin(xt[9]))*0.5)-cos(xt[8])^2*xt[18]^2*(J2+L2^2*M3+LC2^2*M2)+sin(xt[8])^2*xt[18]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(xt[9])*sin(xt[8])*dxt[24]+L1*dxt[16]*(L2*M3+LC2*M2)*(cos(xt[8])*sin(xt[7])-cos(xt[7])*cos(xt[9])*sin(xt[8]))+0.0*cos(xt[9])*sin(xt[8])*(L2*M3+LC2*M2)+L1*xt[16]^2*(L2*M3+LC2*M2)*(cos(xt[7])*cos(xt[8])+cos(xt[9])*sin(xt[7])*sin(xt[8])))+L2*cos(xt[8])*cos(xt[9])*z[24]
            res[9] = dz[9]+z[17]*((L2*cos(xt[8])*cos(xt[9])*dxt[22])*0.5+L2*cos(xt[8])*sin(xt[9])*dxt[24]-0.0*cos(xt[8])*sin(xt[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(xt[8])*cos(xt[9])*dxt[23])*0.5-L1*cos(xt[8])*sin(xt[7])*sin(xt[9])*xt[16]^2*(L2*M3+LC2*M2)+L1*cos(xt[7])*cos(xt[8])*sin(xt[9])*dxt[16]*(L2*M3+LC2*M2))-z[30]*(L2*cos(xt[8])*sin(xt[9])*xt[17]+L2*cos(xt[9])*sin(xt[8])*xt[18])-z[28]*((L2*cos(xt[8])*cos(xt[9])*xt[17])*0.5-(L2*sin(xt[8])*sin(xt[9])*xt[18])*0.5)-z[18]*((L2*sin(xt[8])*sin(xt[9])*dxt[22])*0.5-L2*cos(xt[9])*sin(xt[8])*dxt[24]+0.0*cos(xt[9])*sin(xt[8])*(L2*M3+LC2*M2)+(sqrt(3)*L2*sin(xt[8])*sin(xt[9])*dxt[23])*0.5+L1*cos(xt[9])*sin(xt[7])*sin(xt[8])*xt[16]^2*(L2*M3+LC2*M2)-L1*cos(xt[7])*cos(xt[9])*sin(xt[8])*dxt[16]*(L2*M3+LC2*M2))+z[16]*(L1*cos(xt[7])*cos(xt[8])*sin(xt[9])*dxt[17]*(L2*M3+LC2*M2)-L1*cos(xt[7])*sin(xt[8])*sin(xt[9])*xt[18]^2*(L2*M3+LC2*M2)-L1*cos(xt[7])*sin(xt[8])*sin(xt[9])*xt[17]^2*(L2*M3+LC2*M2)+L1*cos(xt[7])*cos(xt[9])*sin(xt[8])*dxt[18]*(L2*M3+LC2*M2)+2*L1*cos(xt[7])*cos(xt[8])*cos(xt[9])*xt[17]*xt[18]*(L2*M3+LC2*M2))-z[8]*((L2*cos(xt[8])*cos(xt[9])*dxt[28])*0.5+L2*cos(xt[8])*sin(xt[9])*dxt[30]+(sqrt(3)*L2*cos(xt[8])*cos(xt[9])*dxt[29])*0.5)+z[9]*((L2*sin(xt[8])*sin(xt[9])*dxt[28])*0.5-L2*cos(xt[9])*sin(xt[8])*dxt[30]+(sqrt(3)*L2*sin(xt[8])*sin(xt[9])*dxt[29])*0.5)-z[29]*((sqrt(3)*L2*cos(xt[8])*cos(xt[9])*xt[17])*0.5-(sqrt(3)*L2*sin(xt[8])*sin(xt[9])*xt[18])*0.5)-(L2*cos(xt[9])*sin(xt[8])*z[22])*0.5-L2*sin(xt[8])*sin(xt[9])*z[24]-(sqrt(3)*L2*cos(xt[9])*sin(xt[8])*z[23])*0.5
            res[10] = z[1]+dz[10]*(J1+L1^2*(M2+M3)+LC1^2*M1)-z[11]*(L1*(L2*M3+LC2*M2)*(cos(xt[2])*cos(xt[3])*sin(xt[1])*dxt[1]-cos(xt[2])*sin(xt[1])*dxt[2]-cos(xt[1])*sin(xt[2])*dxt[1]+cos(xt[1])*cos(xt[3])*sin(xt[2])*dxt[2]+cos(xt[1])*cos(xt[2])*sin(xt[3])*dxt[3])+2*L1*xt[10]*(L2*M3+LC2*M2)*(cos(xt[1])*sin(xt[2])-cos(xt[2])*cos(xt[3])*sin(xt[1])))-γ*z[10]-z[12]*(L1*cos(xt[1])*cos(xt[2])*sin(xt[3])*dxt[2]*(L2*M3+LC2*M2)+L1*cos(xt[1])*cos(xt[3])*sin(xt[2])*dxt[3]*(L2*M3+LC2*M2)-L1*sin(xt[1])*sin(xt[2])*sin(xt[3])*dxt[1]*(L2*M3+LC2*M2)+2*L1*sin(xt[1])*sin(xt[2])*sin(xt[3])*xt[10]*(L2*M3+LC2*M2))-L1*cos(xt[1])*z[27]-L1*cos(xt[1])*z[30]+L1*sin(xt[1])*z[26]+L1*sin(xt[1])*z[29]+L1*dz[11]*(L2*M3+LC2*M2)*(sin(xt[1])*sin(xt[2])+cos(xt[1])*cos(xt[2])*cos(xt[3]))-L1*cos(xt[1])*sin(xt[2])*sin(xt[3])*dz[12]*(L2*M3+LC2*M2)
            res[11] = z[2]-z[10]*(L1*(L2*M3+LC2*M2)*(cos(xt[2])*cos(xt[3])*sin(xt[1])*dxt[1]-cos(xt[2])*sin(xt[1])*dxt[2]-cos(xt[1])*sin(xt[2])*dxt[1]+cos(xt[1])*cos(xt[3])*sin(xt[2])*dxt[2]+cos(xt[1])*cos(xt[2])*sin(xt[3])*dxt[3])+2*L1*xt[11]*(L2*M3+LC2*M2)*(cos(xt[2])*sin(xt[1])-cos(xt[1])*cos(xt[3])*sin(xt[2]))-2*L1*cos(xt[1])*cos(xt[2])*sin(xt[3])*xt[12]*(L2*M3+LC2*M2))+dz[11]*(J2+L2^2*M3+LC2^2*M2)-γ*z[11]+L2*sin(xt[2])*z[26]+L2*sin(xt[2])*z[29]-L2*cos(xt[2])*cos(xt[3])*z[27]-L2*cos(xt[2])*cos(xt[3])*z[30]-L2*cos(xt[2])*sin(xt[3])*z[25]-L2*cos(xt[2])*sin(xt[3])*z[28]-sin(2*xt[2])*z[12]*xt[12]*(J2+L2^2*M3+LC2^2*M2)+L1*dz[10]*(L2*M3+LC2*M2)*(sin(xt[1])*sin(xt[2])+cos(xt[1])*cos(xt[2])*cos(xt[3]))
            res[12] = z[3]-z[12]*(γ+sin(2*xt[2])*xt[11]*(J2+L2^2*M3+LC2^2*M2)-2*cos(xt[2])*sin(xt[2])*dxt[2]*(J2+L2^2*M3+LC2^2*M2))+z[10]*(2*L1*cos(xt[1])*cos(xt[2])*sin(xt[3])*xt[11]*(L2*M3+LC2*M2)-L1*cos(xt[1])*cos(xt[3])*sin(xt[2])*dxt[3]*(L2*M3+LC2*M2)-L1*cos(xt[1])*cos(xt[2])*sin(xt[3])*dxt[2]*(L2*M3+LC2*M2)+2*L1*cos(xt[1])*cos(xt[3])*sin(xt[2])*xt[12]*(L2*M3+LC2*M2)+L1*sin(xt[1])*sin(xt[2])*sin(xt[3])*dxt[1]*(L2*M3+LC2*M2))+sin(xt[2])^2*dz[12]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(xt[3])*sin(xt[2])*z[25]-L2*cos(xt[3])*sin(xt[2])*z[28]+L2*sin(xt[2])*sin(xt[3])*z[27]+L2*sin(xt[2])*sin(xt[3])*z[30]+2*cos(xt[2])*sin(xt[2])*z[11]*xt[12]*(J2+L2^2*M3+LC2^2*M2)-L1*cos(xt[1])*sin(xt[2])*sin(xt[3])*dz[10]*(L2*M3+LC2*M2)
            res[13] = z[4]+dz[13]*(J1+L1^2*(M2+M3)+LC1^2*M1)-z[14]*(L1*(L2*M3+LC2*M2)*(cos(xt[5])*cos(xt[6])*sin(xt[4])*dxt[4]-cos(xt[5])*sin(xt[4])*dxt[5]-cos(xt[4])*sin(xt[5])*dxt[4]+cos(xt[4])*cos(xt[6])*sin(xt[5])*dxt[5]+cos(xt[4])*cos(xt[5])*sin(xt[6])*dxt[6])+2*L1*xt[13]*(L2*M3+LC2*M2)*(cos(xt[4])*sin(xt[5])-cos(xt[5])*cos(xt[6])*sin(xt[4])))-γ*z[13]-z[15]*(L1*cos(xt[4])*cos(xt[5])*sin(xt[6])*dxt[5]*(L2*M3+LC2*M2)+L1*cos(xt[4])*cos(xt[6])*sin(xt[5])*dxt[6]*(L2*M3+LC2*M2)-L1*sin(xt[4])*sin(xt[5])*sin(xt[6])*dxt[4]*(L2*M3+LC2*M2)+2*L1*sin(xt[4])*sin(xt[5])*sin(xt[6])*xt[13]*(L2*M3+LC2*M2))+L1*cos(xt[4])*z[27]+(L1*sin(xt[4])*z[26])*0.5+(sqrt(3)*L1*sin(xt[4])*z[25])*0.5+L1*dz[14]*(L2*M3+LC2*M2)*(sin(xt[4])*sin(xt[5])+cos(xt[4])*cos(xt[5])*cos(xt[6]))-L1*cos(xt[4])*sin(xt[5])*sin(xt[6])*dz[15]*(L2*M3+LC2*M2)
            res[14] = z[5]-z[25]*((L2*cos(xt[5])*sin(xt[6]))*0.5-(sqrt(3)*L2*sin(xt[5]))*0.5)+z[26]*((L2*sin(xt[5]))*0.5+(sqrt(3)*L2*cos(xt[5])*sin(xt[6]))*0.5)-z[13]*(L1*(L2*M3+LC2*M2)*(cos(xt[5])*cos(xt[6])*sin(xt[4])*dxt[4]-cos(xt[5])*sin(xt[4])*dxt[5]-cos(xt[4])*sin(xt[5])*dxt[4]+cos(xt[4])*cos(xt[6])*sin(xt[5])*dxt[5]+cos(xt[4])*cos(xt[5])*sin(xt[6])*dxt[6])+2*L1*xt[14]*(L2*M3+LC2*M2)*(cos(xt[5])*sin(xt[4])-cos(xt[4])*cos(xt[6])*sin(xt[5]))-2*L1*cos(xt[4])*cos(xt[5])*sin(xt[6])*xt[15]*(L2*M3+LC2*M2))+dz[14]*(J2+L2^2*M3+LC2^2*M2)-γ*z[14]+L2*cos(xt[5])*cos(xt[6])*z[27]-sin(2*xt[5])*z[15]*xt[15]*(J2+L2^2*M3+LC2^2*M2)+L1*dz[13]*(L2*M3+LC2*M2)*(sin(xt[4])*sin(xt[5])+cos(xt[4])*cos(xt[5])*cos(xt[6]))
            res[15] = z[6]-z[15]*(γ+sin(2*xt[5])*xt[14]*(J2+L2^2*M3+LC2^2*M2)-2*cos(xt[5])*sin(xt[5])*dxt[5]*(J2+L2^2*M3+LC2^2*M2))+z[13]*(2*L1*cos(xt[4])*cos(xt[5])*sin(xt[6])*xt[14]*(L2*M3+LC2*M2)-L1*cos(xt[4])*cos(xt[6])*sin(xt[5])*dxt[6]*(L2*M3+LC2*M2)-L1*cos(xt[4])*cos(xt[5])*sin(xt[6])*dxt[5]*(L2*M3+LC2*M2)+2*L1*cos(xt[4])*cos(xt[6])*sin(xt[5])*xt[15]*(L2*M3+LC2*M2)+L1*sin(xt[4])*sin(xt[5])*sin(xt[6])*dxt[4]*(L2*M3+LC2*M2))+sin(xt[5])^2*dz[15]*(J2+L2^2*M3+LC2^2*M2)-(L2*cos(xt[6])*sin(xt[5])*z[25])*0.5-L2*sin(xt[5])*sin(xt[6])*z[27]+2*cos(xt[5])*sin(xt[5])*z[14]*xt[15]*(J2+L2^2*M3+LC2^2*M2)+(sqrt(3)*L2*cos(xt[6])*sin(xt[5])*z[26])*0.5-L1*cos(xt[4])*sin(xt[5])*sin(xt[6])*dz[13]*(L2*M3+LC2*M2)
            res[16] = z[7]+dz[16]*(J1+L1^2*(M2+M3)+LC1^2*M1)-z[17]*(L1*(L2*M3+LC2*M2)*(cos(xt[8])*cos(xt[9])*sin(xt[7])*dxt[7]-cos(xt[8])*sin(xt[7])*dxt[8]-cos(xt[7])*sin(xt[8])*dxt[7]+cos(xt[7])*cos(xt[9])*sin(xt[8])*dxt[8]+cos(xt[7])*cos(xt[8])*sin(xt[9])*dxt[9])+2*L1*xt[16]*(L2*M3+LC2*M2)*(cos(xt[7])*sin(xt[8])-cos(xt[8])*cos(xt[9])*sin(xt[7])))-γ*z[16]-z[18]*(L1*cos(xt[7])*cos(xt[8])*sin(xt[9])*dxt[8]*(L2*M3+LC2*M2)+L1*cos(xt[7])*cos(xt[9])*sin(xt[8])*dxt[9]*(L2*M3+LC2*M2)-L1*sin(xt[7])*sin(xt[8])*sin(xt[9])*dxt[7]*(L2*M3+LC2*M2)+2*L1*sin(xt[7])*sin(xt[8])*sin(xt[9])*xt[16]*(L2*M3+LC2*M2))+L1*cos(xt[7])*z[30]+(L1*sin(xt[7])*z[29])*0.5-(sqrt(3)*L1*sin(xt[7])*z[28])*0.5+L1*dz[17]*(L2*M3+LC2*M2)*(sin(xt[7])*sin(xt[8])+cos(xt[7])*cos(xt[8])*cos(xt[9]))-L1*cos(xt[7])*sin(xt[8])*sin(xt[9])*dz[18]*(L2*M3+LC2*M2)
            res[17] = z[8]-z[28]*((L2*cos(xt[8])*sin(xt[9]))*0.5+(sqrt(3)*L2*sin(xt[8]))*0.5)+z[29]*((L2*sin(xt[8]))*0.5-(sqrt(3)*L2*cos(xt[8])*sin(xt[9]))*0.5)-z[16]*(L1*(L2*M3+LC2*M2)*(cos(xt[8])*cos(xt[9])*sin(xt[7])*dxt[7]-cos(xt[8])*sin(xt[7])*dxt[8]-cos(xt[7])*sin(xt[8])*dxt[7]+cos(xt[7])*cos(xt[9])*sin(xt[8])*dxt[8]+cos(xt[7])*cos(xt[8])*sin(xt[9])*dxt[9])+2*L1*xt[17]*(L2*M3+LC2*M2)*(cos(xt[8])*sin(xt[7])-cos(xt[7])*cos(xt[9])*sin(xt[8]))-2*L1*cos(xt[7])*cos(xt[8])*sin(xt[9])*xt[18]*(L2*M3+LC2*M2))+dz[17]*(J2+L2^2*M3+LC2^2*M2)-γ*z[17]+L2*cos(xt[8])*cos(xt[9])*z[30]-sin(2*xt[8])*z[18]*xt[18]*(J2+L2^2*M3+LC2^2*M2)+L1*dz[16]*(L2*M3+LC2*M2)*(sin(xt[7])*sin(xt[8])+cos(xt[7])*cos(xt[8])*cos(xt[9]))
            res[18] = z[9]-z[18]*(γ+sin(2*xt[8])*xt[17]*(J2+L2^2*M3+LC2^2*M2)-2*cos(xt[8])*sin(xt[8])*dxt[8]*(J2+L2^2*M3+LC2^2*M2))+z[16]*(2*L1*cos(xt[7])*cos(xt[8])*sin(xt[9])*xt[17]*(L2*M3+LC2*M2)-L1*cos(xt[7])*cos(xt[9])*sin(xt[8])*dxt[9]*(L2*M3+LC2*M2)-L1*cos(xt[7])*cos(xt[8])*sin(xt[9])*dxt[8]*(L2*M3+LC2*M2)+2*L1*cos(xt[7])*cos(xt[9])*sin(xt[8])*xt[18]*(L2*M3+LC2*M2)+L1*sin(xt[7])*sin(xt[8])*sin(xt[9])*dxt[7]*(L2*M3+LC2*M2))+sin(xt[8])^2*dz[18]*(J2+L2^2*M3+LC2^2*M2)-(L2*cos(xt[9])*sin(xt[8])*z[28])*0.5-L2*sin(xt[8])*sin(xt[9])*z[30]+2*cos(xt[8])*sin(xt[8])*z[17]*xt[18]*(J2+L2^2*M3+LC2^2*M2)-(sqrt(3)*L2*cos(xt[9])*sin(xt[8])*z[29])*0.5-L1*cos(xt[7])*sin(xt[8])*sin(xt[9])*dz[16]*(L2*M3+LC2*M2)
            res[19] = z[14]*((sqrt(3)*L2*cos(xt[5])*dxt[5])*0.5-(L2*cos(xt[5])*cos(xt[6])*dxt[6])*0.5+(L2*sin(xt[5])*sin(xt[6])*dxt[5])*0.5)-dz[14]*((L2*cos(xt[5])*sin(xt[6]))*0.5-(sqrt(3)*L2*sin(xt[5]))*0.5)-z[11]*(L2*cos(xt[2])*cos(xt[3])*dxt[3]-L2*sin(xt[2])*sin(xt[3])*dxt[2])-z[12]*(L2*cos(xt[2])*cos(xt[3])*dxt[2]-L2*sin(xt[2])*sin(xt[3])*dxt[3])-z[15]*((L2*cos(xt[5])*cos(xt[6])*dxt[5])*0.5-(L2*sin(xt[5])*sin(xt[6])*dxt[6])*0.5)-L2*cos(xt[2])*sin(xt[3])*dz[11]-L2*cos(xt[3])*sin(xt[2])*dz[12]-(L2*cos(xt[6])*sin(xt[5])*dz[15])*0.5+(sqrt(3)*L1*sin(xt[4])*dz[13])*0.5+(sqrt(3)*L1*cos(xt[4])*dxt[4]*z[13])*0.5
            res[20] = dz[14]*((L2*sin(xt[5]))*0.5+(sqrt(3)*L2*cos(xt[5])*sin(xt[6]))*0.5)+z[14]*((L2*cos(xt[5])*dxt[5])*0.5+(sqrt(3)*L2*cos(xt[5])*cos(xt[6])*dxt[6])*0.5-(sqrt(3)*L2*sin(xt[5])*sin(xt[6])*dxt[5])*0.5)+z[15]*((sqrt(3)*L2*cos(xt[5])*cos(xt[6])*dxt[5])*0.5-(sqrt(3)*L2*sin(xt[5])*sin(xt[6])*dxt[6])*0.5)+L1*sin(xt[1])*dz[10]+L2*sin(xt[2])*dz[11]+(L1*sin(xt[4])*dz[13])*0.5+L1*cos(xt[1])*dxt[1]*z[10]+L2*cos(xt[2])*dxt[2]*z[11]+(L1*cos(xt[4])*dxt[4]*z[13])*0.5+(sqrt(3)*L2*cos(xt[6])*sin(xt[5])*dz[15])*0.5
            res[21] = z[11]*(L2*cos(xt[3])*sin(xt[2])*dxt[2]+L2*cos(xt[2])*sin(xt[3])*dxt[3])+z[12]*(L2*cos(xt[2])*sin(xt[3])*dxt[2]+L2*cos(xt[3])*sin(xt[2])*dxt[3])-z[14]*(L2*cos(xt[6])*sin(xt[5])*dxt[5]+L2*cos(xt[5])*sin(xt[6])*dxt[6])-z[15]*(L2*cos(xt[5])*sin(xt[6])*dxt[5]+L2*cos(xt[6])*sin(xt[5])*dxt[6])-L1*cos(xt[1])*dz[10]+L1*cos(xt[4])*dz[13]-L2*cos(xt[2])*cos(xt[3])*dz[11]+L2*cos(xt[5])*cos(xt[6])*dz[14]+L2*sin(xt[2])*sin(xt[3])*dz[12]-L2*sin(xt[5])*sin(xt[6])*dz[15]+L1*sin(xt[1])*dxt[1]*z[10]-L1*sin(xt[4])*dxt[4]*z[13]
            res[22] = -z[17]*((L2*cos(xt[8])*cos(xt[9])*dxt[9])*0.5+(sqrt(3)*L2*cos(xt[8])*dxt[8])*0.5-(L2*sin(xt[8])*sin(xt[9])*dxt[8])*0.5)-dz[17]*((L2*cos(xt[8])*sin(xt[9]))*0.5+(sqrt(3)*L2*sin(xt[8]))*0.5)-z[11]*(L2*cos(xt[2])*cos(xt[3])*dxt[3]-L2*sin(xt[2])*sin(xt[3])*dxt[2])-z[12]*(L2*cos(xt[2])*cos(xt[3])*dxt[2]-L2*sin(xt[2])*sin(xt[3])*dxt[3])-z[18]*((L2*cos(xt[8])*cos(xt[9])*dxt[8])*0.5-(L2*sin(xt[8])*sin(xt[9])*dxt[9])*0.5)-L2*cos(xt[2])*sin(xt[3])*dz[11]-L2*cos(xt[3])*sin(xt[2])*dz[12]-(L2*cos(xt[9])*sin(xt[8])*dz[18])*0.5-(sqrt(3)*L1*sin(xt[7])*dz[16])*0.5-(sqrt(3)*L1*cos(xt[7])*dxt[7]*z[16])*0.5
            res[23] = dz[17]*((L2*sin(xt[8]))*0.5-(sqrt(3)*L2*cos(xt[8])*sin(xt[9]))*0.5)+z[17]*((L2*cos(xt[8])*dxt[8])*0.5-(sqrt(3)*L2*cos(xt[8])*cos(xt[9])*dxt[9])*0.5+(sqrt(3)*L2*sin(xt[8])*sin(xt[9])*dxt[8])*0.5)-z[18]*((sqrt(3)*L2*cos(xt[8])*cos(xt[9])*dxt[8])*0.5-(sqrt(3)*L2*sin(xt[8])*sin(xt[9])*dxt[9])*0.5)+L1*sin(xt[1])*dz[10]+L2*sin(xt[2])*dz[11]+(L1*sin(xt[7])*dz[16])*0.5+L1*cos(xt[1])*dxt[1]*z[10]+L2*cos(xt[2])*dxt[2]*z[11]+(L1*cos(xt[7])*dxt[7]*z[16])*0.5-(sqrt(3)*L2*cos(xt[9])*sin(xt[8])*dz[18])*0.5
            res[24] = z[11]*(L2*cos(xt[3])*sin(xt[2])*dxt[2]+L2*cos(xt[2])*sin(xt[3])*dxt[3])+z[12]*(L2*cos(xt[2])*sin(xt[3])*dxt[2]+L2*cos(xt[3])*sin(xt[2])*dxt[3])-z[17]*(L2*cos(xt[9])*sin(xt[8])*dxt[8]+L2*cos(xt[8])*sin(xt[9])*dxt[9])-z[18]*(L2*cos(xt[8])*sin(xt[9])*dxt[8]+L2*cos(xt[9])*sin(xt[8])*dxt[9])-L1*cos(xt[1])*dz[10]+L1*cos(xt[7])*dz[16]-L2*cos(xt[2])*cos(xt[3])*dz[11]+L2*cos(xt[8])*cos(xt[9])*dz[17]+L2*sin(xt[2])*sin(xt[3])*dz[12]-L2*sin(xt[8])*sin(xt[9])*dz[18]+L1*sin(xt[1])*dxt[1]*z[10]-L1*sin(xt[7])*dxt[7]*z[16]
            res[25] = dz[5]*((L2*cos(xt[5])*sin(xt[6]))*0.5-(sqrt(3)*L2*sin(xt[5]))*0.5)-z[5]*((sqrt(3)*L2*cos(xt[5])*dxt[5])*0.5-(L2*cos(xt[5])*cos(xt[6])*dxt[6])*0.5+(L2*sin(xt[5])*sin(xt[6])*dxt[5])*0.5)+z[2]*(L2*cos(xt[2])*cos(xt[3])*dxt[3]-L2*sin(xt[2])*sin(xt[3])*dxt[2])+z[3]*(L2*cos(xt[2])*cos(xt[3])*dxt[2]-L2*sin(xt[2])*sin(xt[3])*dxt[3])+z[6]*((L2*cos(xt[5])*cos(xt[6])*dxt[5])*0.5-(L2*sin(xt[5])*sin(xt[6])*dxt[6])*0.5)+L2*cos(xt[2])*sin(xt[3])*dz[2]+L2*cos(xt[3])*sin(xt[2])*dz[3]+(L2*cos(xt[6])*sin(xt[5])*dz[6])*0.5-(sqrt(3)*L1*sin(xt[4])*dz[4])*0.5-(sqrt(3)*L1*cos(xt[4])*dxt[4]*z[4])*0.5
            res[26] = -dz[5]*((L2*sin(xt[5]))*0.5+(sqrt(3)*L2*cos(xt[5])*sin(xt[6]))*0.5)-z[5]*((L2*cos(xt[5])*dxt[5])*0.5+(sqrt(3)*L2*cos(xt[5])*cos(xt[6])*dxt[6])*0.5-(sqrt(3)*L2*sin(xt[5])*sin(xt[6])*dxt[5])*0.5)-z[6]*((sqrt(3)*L2*cos(xt[5])*cos(xt[6])*dxt[5])*0.5-(sqrt(3)*L2*sin(xt[5])*sin(xt[6])*dxt[6])*0.5)-L1*sin(xt[1])*dz[1]-L2*sin(xt[2])*dz[2]-(L1*sin(xt[4])*dz[4])*0.5-L1*cos(xt[1])*dxt[1]*z[1]-L2*cos(xt[2])*dxt[2]*z[2]-(L1*cos(xt[4])*dxt[4]*z[4])*0.5-(sqrt(3)*L2*cos(xt[6])*sin(xt[5])*dz[6])*0.5
            res[27] = z[5]*(L2*cos(xt[6])*sin(xt[5])*dxt[5]+L2*cos(xt[5])*sin(xt[6])*dxt[6])-z[3]*(L2*cos(xt[2])*sin(xt[3])*dxt[2]+L2*cos(xt[3])*sin(xt[2])*dxt[3])-z[2]*(L2*cos(xt[3])*sin(xt[2])*dxt[2]+L2*cos(xt[2])*sin(xt[3])*dxt[3])+z[6]*(L2*cos(xt[5])*sin(xt[6])*dxt[5]+L2*cos(xt[6])*sin(xt[5])*dxt[6])+L1*cos(xt[1])*dz[1]-L1*cos(xt[4])*dz[4]+L2*cos(xt[2])*cos(xt[3])*dz[2]-L2*cos(xt[5])*cos(xt[6])*dz[5]-L2*sin(xt[2])*sin(xt[3])*dz[3]+L2*sin(xt[5])*sin(xt[6])*dz[6]-L1*sin(xt[1])*dxt[1]*z[1]+L1*sin(xt[4])*dxt[4]*z[4]
            res[28] = z[8]*((L2*cos(xt[8])*cos(xt[9])*dxt[9])*0.5+(sqrt(3)*L2*cos(xt[8])*dxt[8])*0.5-(L2*sin(xt[8])*sin(xt[9])*dxt[8])*0.5)+dz[8]*((L2*cos(xt[8])*sin(xt[9]))*0.5+(sqrt(3)*L2*sin(xt[8]))*0.5)+z[2]*(L2*cos(xt[2])*cos(xt[3])*dxt[3]-L2*sin(xt[2])*sin(xt[3])*dxt[2])+z[3]*(L2*cos(xt[2])*cos(xt[3])*dxt[2]-L2*sin(xt[2])*sin(xt[3])*dxt[3])+z[9]*((L2*cos(xt[8])*cos(xt[9])*dxt[8])*0.5-(L2*sin(xt[8])*sin(xt[9])*dxt[9])*0.5)+L2*cos(xt[2])*sin(xt[3])*dz[2]+L2*cos(xt[3])*sin(xt[2])*dz[3]+(L2*cos(xt[9])*sin(xt[8])*dz[9])*0.5+(sqrt(3)*L1*sin(xt[7])*dz[7])*0.5+(sqrt(3)*L1*cos(xt[7])*dxt[7]*z[7])*0.5
            res[29] = z[9]*((sqrt(3)*L2*cos(xt[8])*cos(xt[9])*dxt[8])*0.5-(sqrt(3)*L2*sin(xt[8])*sin(xt[9])*dxt[9])*0.5)-z[8]*((L2*cos(xt[8])*dxt[8])*0.5-(sqrt(3)*L2*cos(xt[8])*cos(xt[9])*dxt[9])*0.5+(sqrt(3)*L2*sin(xt[8])*sin(xt[9])*dxt[8])*0.5)-dz[8]*((L2*sin(xt[8]))*0.5-(sqrt(3)*L2*cos(xt[8])*sin(xt[9]))*0.5)-L1*sin(xt[1])*dz[1]-L2*sin(xt[2])*dz[2]-(L1*sin(xt[7])*dz[7])*0.5-L1*cos(xt[1])*dxt[1]*z[1]-L2*cos(xt[2])*dxt[2]*z[2]-(L1*cos(xt[7])*dxt[7]*z[7])*0.5+(sqrt(3)*L2*cos(xt[9])*sin(xt[8])*dz[9])*0.5
            res[30] = z[8]*(L2*cos(xt[9])*sin(xt[8])*dxt[8]+L2*cos(xt[8])*sin(xt[9])*dxt[9])-z[3]*(L2*cos(xt[2])*sin(xt[3])*dxt[2]+L2*cos(xt[3])*sin(xt[2])*dxt[3])-z[2]*(L2*cos(xt[3])*sin(xt[2])*dxt[2]+L2*cos(xt[2])*sin(xt[3])*dxt[3])+z[9]*(L2*cos(xt[8])*sin(xt[9])*dxt[8]+L2*cos(xt[9])*sin(xt[8])*dxt[9])+L1*cos(xt[1])*dz[1]-L1*cos(xt[7])*dz[7]+L2*cos(xt[2])*cos(xt[3])*dz[2]-L2*cos(xt[8])*cos(xt[9])*dz[8]-L2*sin(xt[2])*sin(xt[3])*dz[3]+L2*sin(xt[8])*sin(xt[9])*dz[9]-L1*sin(xt[1])*dxt[1]*z[1]+L1*sin(xt[7])*dxt[7]*z[7]
            # TODO: Do we really need to simulate with these three equations or could we remove them somehow since they are quite trivial?
            res[31] = -z[31]-(2*yt[1]-2*xt[31])/T
            res[32] = -z[32]-(2*yt[2]-2*xt[32])/T
            res[33] = -z[33]-(2*yt[3]-2*xt[33])/T
            # \beta equations (parameter-specific)
            res[34] = dz[34] - (z[12]*(sin(xt[1])*sin(xt[2])*sin(xt[3])*xt[10]^2*(L2*M3+LC2*M2)-cos(xt[1])*sin(xt[2])*sin(xt[3])*dxt[10]*(L2*M3+LC2*M2))+z[15]*(sin(xt[4])*sin(xt[5])*sin(xt[6])*xt[13]^2*(L2*M3+LC2*M2)-cos(xt[4])*sin(xt[5])*sin(xt[6])*dxt[13]*(L2*M3+LC2*M2))+z[18]*(sin(xt[7])*sin(xt[8])*sin(xt[9])*xt[16]^2*(L2*M3+LC2*M2)-cos(xt[7])*sin(xt[8])*sin(xt[9])*dxt[16]*(L2*M3+LC2*M2))+z[11]*(dxt[10]*(L2*M3+LC2*M2)*(sin(xt[1])*sin(xt[2])+cos(xt[1])*cos(xt[2])*cos(xt[3]))+xt[10]^2*(L2*M3+LC2*M2)*(cos(xt[1])*sin(xt[2])-cos(xt[2])*cos(xt[3])*sin(xt[1])))+z[14]*(dxt[13]*(L2*M3+LC2*M2)*(sin(xt[4])*sin(xt[5])+cos(xt[4])*cos(xt[5])*cos(xt[6]))+xt[13]^2*(L2*M3+LC2*M2)*(cos(xt[4])*sin(xt[5])-cos(xt[5])*cos(xt[6])*sin(xt[4])))+z[17]*(dxt[16]*(L2*M3+LC2*M2)*(sin(xt[7])*sin(xt[8])+cos(xt[7])*cos(xt[8])*cos(xt[9]))+xt[16]^2*(L2*M3+LC2*M2)*(cos(xt[7])*sin(xt[8])-cos(xt[8])*cos(xt[9])*sin(xt[7])))-cos(xt[1])*z[32]+z[20]*(cos(xt[1])+cos(xt[4])*0.5)+z[23]*(cos(xt[1])+cos(xt[7])*0.5)-sin(xt[1])*z[33]-z[4]*(cos(xt[4])*dxt[27]+(sin(xt[4])*dxt[26])*0.5+(sqrt(3)*sin(xt[4])*dxt[25])*0.5)-z[7]*(cos(xt[7])*dxt[30]+(sin(xt[7])*dxt[29])*0.5-(sqrt(3)*sin(xt[7])*dxt[28])*0.5)+z[21]*(sin(xt[1])-sin(xt[4]))+z[24]*(sin(xt[1])-sin(xt[7]))+z[1]*(cos(xt[1])*dxt[27]+cos(xt[1])*dxt[30]-sin(xt[1])*dxt[26]-sin(xt[1])*dxt[29])+z[27]*(cos(xt[1])*xt[10]-cos(xt[4])*xt[13])+z[30]*(cos(xt[1])*xt[10]-cos(xt[7])*xt[16])+z[13]*(cos(xt[4])*dxt[21]+(sin(xt[4])*dxt[20])*0.5-0.0*cos(xt[4])*(M2+M3)+dxt[14]*(L2*M3+LC2*M2)*(sin(xt[4])*sin(xt[5])+cos(xt[4])*cos(xt[5])*cos(xt[6]))+2*L1*dxt[13]*(M2+M3)+xt[14]^2*(L2*M3+LC2*M2)*(cos(xt[5])*sin(xt[4])-cos(xt[4])*cos(xt[6])*sin(xt[5]))+(sqrt(3)*sin(xt[4])*dxt[19])*0.5-cos(xt[4])*sin(xt[5])*sin(xt[6])*dxt[15]*(L2*M3+LC2*M2)-cos(xt[4])*cos(xt[6])*sin(xt[5])*xt[15]^2*(L2*M3+LC2*M2)-2*cos(xt[4])*cos(xt[5])*sin(xt[6])*xt[14]*xt[15]*(L2*M3+LC2*M2))+z[16]*(cos(xt[7])*dxt[24]+(sin(xt[7])*dxt[23])*0.5-0.0*cos(xt[7])*(M2+M3)+dxt[17]*(L2*M3+LC2*M2)*(sin(xt[7])*sin(xt[8])+cos(xt[7])*cos(xt[8])*cos(xt[9]))+2*L1*dxt[16]*(M2+M3)+xt[17]^2*(L2*M3+LC2*M2)*(cos(xt[8])*sin(xt[7])-cos(xt[7])*cos(xt[9])*sin(xt[8]))-(sqrt(3)*sin(xt[7])*dxt[22])*0.5-cos(xt[7])*sin(xt[8])*sin(xt[9])*dxt[18]*(L2*M3+LC2*M2)-cos(xt[7])*cos(xt[9])*sin(xt[8])*xt[18]^2*(L2*M3+LC2*M2)-2*cos(xt[7])*cos(xt[8])*sin(xt[9])*xt[17]*xt[18]*(L2*M3+LC2*M2))-z[26]*(sin(xt[1])*xt[10]+(sin(xt[4])*xt[13])*0.5)-z[29]*(sin(xt[1])*xt[10]+(sin(xt[7])*xt[16])*0.5)-z[10]*(cos(xt[1])*dxt[21]+cos(xt[1])*dxt[24]-sin(xt[1])*dxt[20]-sin(xt[1])*dxt[23]+0.0*cos(xt[1])*(M2+M3)-dxt[11]*(L2*M3+LC2*M2)*(sin(xt[1])*sin(xt[2])+cos(xt[1])*cos(xt[2])*cos(xt[3]))-2*L1*dxt[10]*(M2+M3)-xt[11]^2*(L2*M3+LC2*M2)*(cos(xt[2])*sin(xt[1])-cos(xt[1])*cos(xt[3])*sin(xt[2]))+cos(xt[1])*sin(xt[2])*sin(xt[3])*dxt[12]*(L2*M3+LC2*M2)+cos(xt[1])*cos(xt[3])*sin(xt[2])*xt[12]^2*(L2*M3+LC2*M2)+2*cos(xt[1])*cos(xt[2])*sin(xt[3])*xt[11]*xt[12]*(L2*M3+LC2*M2))+(sqrt(3)*cos(xt[4])*z[19])*0.5-(sqrt(3)*cos(xt[7])*z[22])*0.5-(sqrt(3)*sin(xt[4])*z[25]*xt[13])*0.5+(sqrt(3)*sin(xt[7])*z[28]*xt[16])*0.5)

            nothing
        end

        z0  = vcat(λT[:], zeros(np))
        dz0 = vcat(dλT[:], (λT')*Fp)   # For some reason (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired

        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            # NOTE: Changes signs to match what I had in my manual calculations, seems correct now
            # Gp = adj_sol.u[end][end-np+1:end] + (((adj_sol.u[end][1:end-np]')*Fdx(0.0))*xp0)[:]
            # Gp = adj_sol.u[end-N_trans][nx+1:nx+np] + (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)[:]
            Gp = adj_sol.u[end][nx+1:nx+np] .+ (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)
            # return 0.0
        end

        function get_Gp_debug(adj_sol::DAESolution)
            integral = adj_sol.u[end][nx+1:nx+np]
            term = (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)
            Gp = integral .+ term
            return Gp, integral, term
        end

        function get_term_debug(adj_sol::DAESolution, xps::Matrix{Float64}, times::AbstractVector{Float64})
            term = zeros(length(adj_sol.u))
            for ind=eachindex(adj_sol.u)
                term[ind] = ((adj_sol.u[end+1-ind][1:nx]')*Fdx(times[ind]))*xps[:,ind]
            end
            return term
        end

        # The only adjoint variables whose derivatives occur in the adjoint equations are those adjoint variables corresponding to the differential equations of the nominal system
        dvars = vcat(fill(true, 18), fill(false, 15), fill(true, np))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 for adjoint delta robot is: $r0"

        debugs = (get_Gp_debug, get_term_debug)

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp, debugs
    end
end

# gravity-compensated model
function delta_adj_stepbystep_NEW(Φ::Float64, u::Function, w::Function, θ::Vector{Float64}, y::Function, x_func::Function, dx_func::Function, λ::Function, dλ::Function, Fp::Function, T::Float64)::Model
    let L0 = θ[1], L1 = θ[2], L2 = θ[3], L3 = θ[4], LC1 = θ[5], LC2 = θ[6], M1 = θ[7], M2 = θ[8], M3 = θ[9], J1 = θ[10], J2 = θ[11], g = 0.0, γ = θ[13]

        Fx = (x,dx) -> [-L1*cos(x[1])*dx[26]-L1*cos(x[1])*dx[29]-L1*sin(x[1])*dx[27]-L1*sin(x[1])*dx[30]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   -L2*cos(x[2])*dx[26]-L2*cos(x[2])*dx[29]-L2*cos(x[3])*sin(x[2])*dx[27]-L2*cos(x[3])*sin(x[2])*dx[30]-L2*sin(x[2])*sin(x[3])*dx[25]-L2*sin(x[2])*sin(x[3])*dx[28]   L2*cos(x[2])*cos(x[3])*dx[25]+L2*cos(x[2])*cos(x[3])*dx[28]-L2*cos(x[2])*sin(x[3])*dx[27]-L2*cos(x[2])*sin(x[3])*dx[30]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   L2*cos(x[2])*cos(x[3])*dx[25]+L2*cos(x[2])*cos(x[3])*dx[28]-L2*cos(x[2])*sin(x[3])*dx[27]-L2*cos(x[2])*sin(x[3])*dx[30]   -L2*cos(x[3])*sin(x[2])*dx[27]-L2*cos(x[3])*sin(x[2])*dx[30]-L2*sin(x[2])*sin(x[3])*dx[25]-L2*sin(x[2])*sin(x[3])*dx[28]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   L1*sin(x[4])*dx[27]-(L1*cos(x[4])*dx[26])*0.5-(sqrt(3)*L1*cos(x[4])*dx[25])*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   L2*cos(x[6])*sin(x[5])*dx[27]-dx[26]*((L2*cos(x[5]))*0.5-(sqrt(3)*L2*sin(x[5])*sin(x[6]))*0.5)-dx[25]*((sqrt(3)*L2*cos(x[5]))*0.5+(L2*sin(x[5])*sin(x[6]))*0.5)   (L2*cos(x[5])*cos(x[6])*dx[25])*0.5+L2*cos(x[5])*sin(x[6])*dx[27]-(sqrt(3)*L2*cos(x[5])*cos(x[6])*dx[26])*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   (L2*cos(x[5])*cos(x[6])*dx[25])*0.5+L2*cos(x[5])*sin(x[6])*dx[27]-(sqrt(3)*L2*cos(x[5])*cos(x[6])*dx[26])*0.5   L2*cos(x[6])*sin(x[5])*dx[27]-(L2*sin(x[5])*sin(x[6])*dx[25])*0.5+(sqrt(3)*L2*sin(x[5])*sin(x[6])*dx[26])*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   L1*sin(x[7])*dx[30]-(L1*cos(x[7])*dx[29])*0.5+(sqrt(3)*L1*cos(x[7])*dx[28])*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   dx[28]*((sqrt(3)*L2*cos(x[8]))*0.5-(L2*sin(x[8])*sin(x[9]))*0.5)-dx[29]*((L2*cos(x[8]))*0.5+(sqrt(3)*L2*sin(x[8])*sin(x[9]))*0.5)+L2*cos(x[9])*sin(x[8])*dx[30]   (L2*cos(x[8])*cos(x[9])*dx[28])*0.5+L2*cos(x[8])*sin(x[9])*dx[30]+(sqrt(3)*L2*cos(x[8])*cos(x[9])*dx[29])*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x[8])*cos(x[9])*dx[28])*0.5+L2*cos(x[8])*sin(x[9])*dx[30]+(sqrt(3)*L2*cos(x[8])*cos(x[9])*dx[29])*0.5   L2*cos(x[9])*sin(x[8])*dx[30]-(L2*sin(x[8])*sin(x[9])*dx[28])*0.5-(sqrt(3)*L2*sin(x[8])*sin(x[9])*dx[29])*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0*sin(x[1])*(L1*(M2+M3)+LC1*M1)+L1*cos(x[1])*dx[20]+L1*cos(x[1])*dx[23]+L1*sin(x[1])*dx[21]+L1*sin(x[1])*dx[24]+L1*dx[11]*(L2*M3+LC2*M2)*(cos(x[1])*sin(x[2])-cos(x[2])*cos(x[3])*sin(x[1]))+L1*x[11]^2*(L2*M3+LC2*M2)*(cos(x[1])*cos(x[2])+cos(x[3])*sin(x[1])*sin(x[2]))+L1*cos(x[3])*sin(x[1])*sin(x[2])*x[12]^2*(L2*M3+LC2*M2)+L1*sin(x[1])*sin(x[2])*sin(x[3])*dx[12]*(L2*M3+LC2*M2)+2*L1*cos(x[2])*sin(x[1])*sin(x[3])*x[11]*x[12]*(L2*M3+LC2*M2)   L1*dx[11]*(L2*M3+LC2*M2)*(cos(x[2])*sin(x[1])-cos(x[1])*cos(x[3])*sin(x[2]))-L1*x[11]^2*(L2*M3+LC2*M2)*(sin(x[1])*sin(x[2])+cos(x[1])*cos(x[2])*cos(x[3]))-L1*cos(x[1])*cos(x[2])*cos(x[3])*x[12]^2*(L2*M3+LC2*M2)-L1*cos(x[1])*cos(x[2])*sin(x[3])*dx[12]*(L2*M3+LC2*M2)+2*L1*cos(x[1])*sin(x[2])*sin(x[3])*x[11]*x[12]*(L2*M3+LC2*M2)   L1*cos(x[1])*sin(x[2])*sin(x[3])*x[11]^2*(L2*M3+LC2*M2)+L1*cos(x[1])*sin(x[2])*sin(x[3])*x[12]^2*(L2*M3+LC2*M2)-L1*cos(x[1])*cos(x[2])*sin(x[3])*dx[11]*(L2*M3+LC2*M2)-L1*cos(x[1])*cos(x[3])*sin(x[2])*dx[12]*(L2*M3+LC2*M2)-2*L1*cos(x[1])*cos(x[2])*cos(x[3])*x[11]*x[12]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   γ   2*L1*x[11]*(L2*M3+LC2*M2)*(cos(x[2])*sin(x[1])-cos(x[1])*cos(x[3])*sin(x[2]))-2*L1*cos(x[1])*cos(x[2])*sin(x[3])*x[12]*(L2*M3+LC2*M2)   -2*L1*cos(x[1])*cos(x[2])*sin(x[3])*x[11]*(L2*M3+LC2*M2)-2*L1*cos(x[1])*cos(x[3])*sin(x[2])*x[12]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            L1*dx[10]*(L2*M3+LC2*M2)*(cos(x[1])*sin(x[2])-cos(x[2])*cos(x[3])*sin(x[1]))-L1*x[10]^2*(L2*M3+LC2*M2)*(sin(x[1])*sin(x[2])+cos(x[1])*cos(x[2])*cos(x[3]))   sin(x[2])^2*x[12]^2*(J2+L2^2*M3+LC2^2*M2)-cos(x[2])^2*x[12]^2*(J2+L2^2*M3+LC2^2*M2)+L2*cos(x[2])*dx[20]+L2*cos(x[2])*dx[23]+L2*cos(x[3])*sin(x[2])*dx[21]+L2*cos(x[3])*sin(x[2])*dx[24]+L2*sin(x[2])*sin(x[3])*dx[19]+L2*sin(x[2])*sin(x[3])*dx[22]+L1*dx[10]*(L2*M3+LC2*M2)*(cos(x[2])*sin(x[1])-cos(x[1])*cos(x[3])*sin(x[2]))+0.0*cos(x[3])*sin(x[2])*(L2*M3+LC2*M2)+L1*x[10]^2*(L2*M3+LC2*M2)*(cos(x[1])*cos(x[2])+cos(x[3])*sin(x[1])*sin(x[2]))   L2*cos(x[2])*sin(x[3])*dx[21]-L2*cos(x[2])*cos(x[3])*dx[22]-L2*cos(x[2])*cos(x[3])*dx[19]+L2*cos(x[2])*sin(x[3])*dx[24]+0.0*cos(x[2])*sin(x[3])*(L2*M3+LC2*M2)+L1*cos(x[2])*sin(x[1])*sin(x[3])*x[10]^2*(L2*M3+LC2*M2)-L1*cos(x[1])*cos(x[2])*sin(x[3])*dx[10]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   2*L1*x[10]*(L2*M3+LC2*M2)*(cos(x[1])*sin(x[2])-cos(x[2])*cos(x[3])*sin(x[1]))   γ   -2*cos(x[2])*sin(x[2])*x[12]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            L1*cos(x[1])*sin(x[2])*sin(x[3])*x[10]^2*(L2*M3+LC2*M2)+L1*sin(x[1])*sin(x[2])*sin(x[3])*dx[10]*(L2*M3+LC2*M2)   L2*cos(x[2])*sin(x[3])*dx[21]-L2*cos(x[2])*cos(x[3])*dx[22]-L2*cos(x[2])*cos(x[3])*dx[19]+L2*cos(x[2])*sin(x[3])*dx[24]+2*cos(x[2])*sin(x[2])*dx[12]*(J2+L2^2*M3+LC2^2*M2)+2*cos(2*x[2])*x[11]*x[12]*(J2+L2^2*M3+LC2^2*M2)+0.0*cos(x[2])*sin(x[3])*(L2*M3+LC2*M2)+L1*cos(x[2])*sin(x[1])*sin(x[3])*x[10]^2*(L2*M3+LC2*M2)-L1*cos(x[1])*cos(x[2])*sin(x[3])*dx[10]*(L2*M3+LC2*M2)   L2*cos(x[3])*sin(x[2])*dx[21]+L2*cos(x[3])*sin(x[2])*dx[24]+L2*sin(x[2])*sin(x[3])*dx[19]+L2*sin(x[2])*sin(x[3])*dx[22]+0.0*cos(x[3])*sin(x[2])*(L2*M3+LC2*M2)+L1*cos(x[3])*sin(x[1])*sin(x[2])*x[10]^2*(L2*M3+LC2*M2)-L1*cos(x[1])*cos(x[3])*sin(x[2])*dx[10]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   2*L1*sin(x[1])*sin(x[2])*sin(x[3])*x[10]*(L2*M3+LC2*M2)   sin(2*x[2])*x[12]*(J2+L2^2*M3+LC2^2*M2)   γ+sin(2*x[2])*x[11]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0*sin(x[4])*(L1*(M2+M3)+LC1*M1)+(L1*cos(x[4])*dx[20])*0.5-L1*sin(x[4])*dx[21]+(sqrt(3)*L1*cos(x[4])*dx[19])*0.5+L1*dx[14]*(L2*M3+LC2*M2)*(cos(x[4])*sin(x[5])-cos(x[5])*cos(x[6])*sin(x[4]))+L1*x[14]^2*(L2*M3+LC2*M2)*(cos(x[4])*cos(x[5])+cos(x[6])*sin(x[4])*sin(x[5]))+L1*cos(x[6])*sin(x[4])*sin(x[5])*x[15]^2*(L2*M3+LC2*M2)+L1*sin(x[4])*sin(x[5])*sin(x[6])*dx[15]*(L2*M3+LC2*M2)+2*L1*cos(x[5])*sin(x[4])*sin(x[6])*x[14]*x[15]*(L2*M3+LC2*M2)   L1*dx[14]*(L2*M3+LC2*M2)*(cos(x[5])*sin(x[4])-cos(x[4])*cos(x[6])*sin(x[5]))-L1*x[14]^2*(L2*M3+LC2*M2)*(sin(x[4])*sin(x[5])+cos(x[4])*cos(x[5])*cos(x[6]))-L1*cos(x[4])*cos(x[5])*cos(x[6])*x[15]^2*(L2*M3+LC2*M2)-L1*cos(x[4])*cos(x[5])*sin(x[6])*dx[15]*(L2*M3+LC2*M2)+2*L1*cos(x[4])*sin(x[5])*sin(x[6])*x[14]*x[15]*(L2*M3+LC2*M2)   L1*cos(x[4])*sin(x[5])*sin(x[6])*x[14]^2*(L2*M3+LC2*M2)+L1*cos(x[4])*sin(x[5])*sin(x[6])*x[15]^2*(L2*M3+LC2*M2)-L1*cos(x[4])*cos(x[5])*sin(x[6])*dx[14]*(L2*M3+LC2*M2)-L1*cos(x[4])*cos(x[6])*sin(x[5])*dx[15]*(L2*M3+LC2*M2)-2*L1*cos(x[4])*cos(x[5])*cos(x[6])*x[14]*x[15]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   γ   2*L1*x[14]*(L2*M3+LC2*M2)*(cos(x[5])*sin(x[4])-cos(x[4])*cos(x[6])*sin(x[5]))-2*L1*cos(x[4])*cos(x[5])*sin(x[6])*x[15]*(L2*M3+LC2*M2)   -2*L1*cos(x[4])*cos(x[5])*sin(x[6])*x[14]*(L2*M3+LC2*M2)-2*L1*cos(x[4])*cos(x[6])*sin(x[5])*x[15]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   L1*dx[13]*(L2*M3+LC2*M2)*(cos(x[4])*sin(x[5])-cos(x[5])*cos(x[6])*sin(x[4]))-L1*x[13]^2*(L2*M3+LC2*M2)*(sin(x[4])*sin(x[5])+cos(x[4])*cos(x[5])*cos(x[6]))   dx[19]*((sqrt(3)*L2*cos(x[5]))*0.5+(L2*sin(x[5])*sin(x[6]))*0.5)+dx[20]*((L2*cos(x[5]))*0.5-(sqrt(3)*L2*sin(x[5])*sin(x[6]))*0.5)-cos(x[5])^2*x[15]^2*(J2+L2^2*M3+LC2^2*M2)+sin(x[5])^2*x[15]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(x[6])*sin(x[5])*dx[21]+L1*dx[13]*(L2*M3+LC2*M2)*(cos(x[5])*sin(x[4])-cos(x[4])*cos(x[6])*sin(x[5]))+0.0*cos(x[6])*sin(x[5])*(L2*M3+LC2*M2)+L1*x[13]^2*(L2*M3+LC2*M2)*(cos(x[4])*cos(x[5])+cos(x[6])*sin(x[4])*sin(x[5]))   0.0*cos(x[5])*sin(x[6])*(L2*M3+LC2*M2)-L2*cos(x[5])*sin(x[6])*dx[21]-(L2*cos(x[5])*cos(x[6])*dx[19])*0.5+(sqrt(3)*L2*cos(x[5])*cos(x[6])*dx[20])*0.5+L1*cos(x[5])*sin(x[4])*sin(x[6])*x[13]^2*(L2*M3+LC2*M2)-L1*cos(x[4])*cos(x[5])*sin(x[6])*dx[13]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   2*L1*x[13]*(L2*M3+LC2*M2)*(cos(x[4])*sin(x[5])-cos(x[5])*cos(x[6])*sin(x[4]))   γ   -2*cos(x[5])*sin(x[5])*x[15]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   L1*cos(x[4])*sin(x[5])*sin(x[6])*x[13]^2*(L2*M3+LC2*M2)+L1*sin(x[4])*sin(x[5])*sin(x[6])*dx[13]*(L2*M3+LC2*M2)   2*cos(x[5])*sin(x[5])*dx[15]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(x[5])*sin(x[6])*dx[21]-(L2*cos(x[5])*cos(x[6])*dx[19])*0.5+2*cos(2*x[5])*x[14]*x[15]*(J2+L2^2*M3+LC2^2*M2)+0.0*cos(x[5])*sin(x[6])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(x[5])*cos(x[6])*dx[20])*0.5+L1*cos(x[5])*sin(x[4])*sin(x[6])*x[13]^2*(L2*M3+LC2*M2)-L1*cos(x[4])*cos(x[5])*sin(x[6])*dx[13]*(L2*M3+LC2*M2)   (L2*sin(x[5])*sin(x[6])*dx[19])*0.5-L2*cos(x[6])*sin(x[5])*dx[21]+0.0*cos(x[6])*sin(x[5])*(L2*M3+LC2*M2)-(sqrt(3)*L2*sin(x[5])*sin(x[6])*dx[20])*0.5+L1*cos(x[6])*sin(x[4])*sin(x[5])*x[13]^2*(L2*M3+LC2*M2)-L1*cos(x[4])*cos(x[6])*sin(x[5])*dx[13]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   2*L1*sin(x[4])*sin(x[5])*sin(x[6])*x[13]*(L2*M3+LC2*M2)   sin(2*x[5])*x[15]*(J2+L2^2*M3+LC2^2*M2)   γ+sin(2*x[5])*x[14]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0*sin(x[7])*(L1*(M2+M3)+LC1*M1)+(L1*cos(x[7])*dx[23])*0.5-L1*sin(x[7])*dx[24]-(sqrt(3)*L1*cos(x[7])*dx[22])*0.5+L1*dx[17]*(L2*M3+LC2*M2)*(cos(x[7])*sin(x[8])-cos(x[8])*cos(x[9])*sin(x[7]))+L1*x[17]^2*(L2*M3+LC2*M2)*(cos(x[7])*cos(x[8])+cos(x[9])*sin(x[7])*sin(x[8]))+L1*cos(x[9])*sin(x[7])*sin(x[8])*x[18]^2*(L2*M3+LC2*M2)+L1*sin(x[7])*sin(x[8])*sin(x[9])*dx[18]*(L2*M3+LC2*M2)+2*L1*cos(x[8])*sin(x[7])*sin(x[9])*x[17]*x[18]*(L2*M3+LC2*M2)   L1*dx[17]*(L2*M3+LC2*M2)*(cos(x[8])*sin(x[7])-cos(x[7])*cos(x[9])*sin(x[8]))-L1*x[17]^2*(L2*M3+LC2*M2)*(sin(x[7])*sin(x[8])+cos(x[7])*cos(x[8])*cos(x[9]))-L1*cos(x[7])*cos(x[8])*cos(x[9])*x[18]^2*(L2*M3+LC2*M2)-L1*cos(x[7])*cos(x[8])*sin(x[9])*dx[18]*(L2*M3+LC2*M2)+2*L1*cos(x[7])*sin(x[8])*sin(x[9])*x[17]*x[18]*(L2*M3+LC2*M2)   L1*cos(x[7])*sin(x[8])*sin(x[9])*x[17]^2*(L2*M3+LC2*M2)+L1*cos(x[7])*sin(x[8])*sin(x[9])*x[18]^2*(L2*M3+LC2*M2)-L1*cos(x[7])*cos(x[8])*sin(x[9])*dx[17]*(L2*M3+LC2*M2)-L1*cos(x[7])*cos(x[9])*sin(x[8])*dx[18]*(L2*M3+LC2*M2)-2*L1*cos(x[7])*cos(x[8])*cos(x[9])*x[17]*x[18]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   γ   2*L1*x[17]*(L2*M3+LC2*M2)*(cos(x[8])*sin(x[7])-cos(x[7])*cos(x[9])*sin(x[8]))-2*L1*cos(x[7])*cos(x[8])*sin(x[9])*x[18]*(L2*M3+LC2*M2)   -2*L1*cos(x[7])*cos(x[8])*sin(x[9])*x[17]*(L2*M3+LC2*M2)-2*L1*cos(x[7])*cos(x[9])*sin(x[8])*x[18]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   L1*dx[16]*(L2*M3+LC2*M2)*(cos(x[7])*sin(x[8])-cos(x[8])*cos(x[9])*sin(x[7]))-L1*x[16]^2*(L2*M3+LC2*M2)*(sin(x[7])*sin(x[8])+cos(x[7])*cos(x[8])*cos(x[9]))   dx[23]*((L2*cos(x[8]))*0.5+(sqrt(3)*L2*sin(x[8])*sin(x[9]))*0.5)-dx[22]*((sqrt(3)*L2*cos(x[8]))*0.5-(L2*sin(x[8])*sin(x[9]))*0.5)-cos(x[8])^2*x[18]^2*(J2+L2^2*M3+LC2^2*M2)+sin(x[8])^2*x[18]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(x[9])*sin(x[8])*dx[24]+L1*dx[16]*(L2*M3+LC2*M2)*(cos(x[8])*sin(x[7])-cos(x[7])*cos(x[9])*sin(x[8]))+0.0*cos(x[9])*sin(x[8])*(L2*M3+LC2*M2)+L1*x[16]^2*(L2*M3+LC2*M2)*(cos(x[7])*cos(x[8])+cos(x[9])*sin(x[7])*sin(x[8]))   0.0*cos(x[8])*sin(x[9])*(L2*M3+LC2*M2)-L2*cos(x[8])*sin(x[9])*dx[24]-(L2*cos(x[8])*cos(x[9])*dx[22])*0.5-(sqrt(3)*L2*cos(x[8])*cos(x[9])*dx[23])*0.5+L1*cos(x[8])*sin(x[7])*sin(x[9])*x[16]^2*(L2*M3+LC2*M2)-L1*cos(x[7])*cos(x[8])*sin(x[9])*dx[16]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   2*L1*x[16]*(L2*M3+LC2*M2)*(cos(x[7])*sin(x[8])-cos(x[8])*cos(x[9])*sin(x[7]))   γ   -2*cos(x[8])*sin(x[8])*x[18]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   L1*cos(x[7])*sin(x[8])*sin(x[9])*x[16]^2*(L2*M3+LC2*M2)+L1*sin(x[7])*sin(x[8])*sin(x[9])*dx[16]*(L2*M3+LC2*M2)   2*cos(x[8])*sin(x[8])*dx[18]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(x[8])*sin(x[9])*dx[24]-(L2*cos(x[8])*cos(x[9])*dx[22])*0.5+2*cos(2*x[8])*x[17]*x[18]*(J2+L2^2*M3+LC2^2*M2)+0.0*cos(x[8])*sin(x[9])*(L2*M3+LC2*M2)-(sqrt(3)*L2*cos(x[8])*cos(x[9])*dx[23])*0.5+L1*cos(x[8])*sin(x[7])*sin(x[9])*x[16]^2*(L2*M3+LC2*M2)-L1*cos(x[7])*cos(x[8])*sin(x[9])*dx[16]*(L2*M3+LC2*M2)   (L2*sin(x[8])*sin(x[9])*dx[22])*0.5-L2*cos(x[9])*sin(x[8])*dx[24]+0.0*cos(x[9])*sin(x[8])*(L2*M3+LC2*M2)+(sqrt(3)*L2*sin(x[8])*sin(x[9])*dx[23])*0.5+L1*cos(x[9])*sin(x[7])*sin(x[8])*x[16]^2*(L2*M3+LC2*M2)-L1*cos(x[7])*cos(x[9])*sin(x[8])*dx[16]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   2*L1*sin(x[7])*sin(x[8])*sin(x[9])*x[16]*(L2*M3+LC2*M2)   sin(2*x[8])*x[18]*(J2+L2^2*M3+LC2^2*M2)   γ+sin(2*x[8])*x[17]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   L2*cos(x[2])*sin(x[3])   L2*cos(x[3])*sin(x[2])   -(sqrt(3)*L1*sin(x[4]))*0.5   (L2*cos(x[5])*sin(x[6]))*0.5-(sqrt(3)*L2*sin(x[5]))*0.5   (L2*cos(x[6])*sin(x[5]))*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            -L1*sin(x[1])   -L2*sin(x[2])   0.0   -(L1*sin(x[4]))*0.5   -(L2*sin(x[5]))*0.5-(sqrt(3)*L2*cos(x[5])*sin(x[6]))*0.5   -(sqrt(3)*L2*cos(x[6])*sin(x[5]))*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            L1*cos(x[1])   L2*cos(x[2])*cos(x[3])   -L2*sin(x[2])*sin(x[3])   -L1*cos(x[4])   -L2*cos(x[5])*cos(x[6])   L2*sin(x[5])*sin(x[6])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   L2*cos(x[2])*sin(x[3])   L2*cos(x[3])*sin(x[2])   0.0   0.0   0.0   (sqrt(3)*L1*sin(x[7]))*0.5   (L2*cos(x[8])*sin(x[9]))*0.5+(sqrt(3)*L2*sin(x[8]))*0.5   (L2*cos(x[9])*sin(x[8]))*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            -L1*sin(x[1])   -L2*sin(x[2])   0.0   0.0   0.0   0.0   -(L1*sin(x[7]))*0.5   (sqrt(3)*L2*cos(x[8])*sin(x[9]))*0.5-(L2*sin(x[8]))*0.5   (sqrt(3)*L2*cos(x[9])*sin(x[8]))*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            L1*cos(x[1])   L2*cos(x[2])*cos(x[3])   -L2*sin(x[2])*sin(x[3])   0.0   0.0   0.0   -L1*cos(x[7])   -L2*cos(x[8])*cos(x[9])   L2*sin(x[8])*sin(x[9])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   L2*cos(x[2])*cos(x[3])*x[12]-L2*sin(x[2])*sin(x[3])*x[11]   L2*cos(x[2])*cos(x[3])*x[11]-L2*sin(x[2])*sin(x[3])*x[12]   -(sqrt(3)*L1*cos(x[4])*x[13])*0.5   (L2*cos(x[5])*cos(x[6])*x[15])*0.5-(sqrt(3)*L2*cos(x[5])*x[14])*0.5-(L2*sin(x[5])*sin(x[6])*x[14])*0.5   (L2*cos(x[5])*cos(x[6])*x[14])*0.5-(L2*sin(x[5])*sin(x[6])*x[15])*0.5   0.0   0.0   0.0   0.0   L2*cos(x[2])*sin(x[3])   L2*cos(x[3])*sin(x[2])   -(sqrt(3)*L1*sin(x[4]))*0.5   (L2*cos(x[5])*sin(x[6]))*0.5-(sqrt(3)*L2*sin(x[5]))*0.5   (L2*cos(x[6])*sin(x[5]))*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            -L1*cos(x[1])*x[10]   -L2*cos(x[2])*x[11]   0.0   -(L1*cos(x[4])*x[13])*0.5   (sqrt(3)*L2*sin(x[5])*sin(x[6])*x[14])*0.5-(sqrt(3)*L2*cos(x[5])*cos(x[6])*x[15])*0.5-(L2*cos(x[5])*x[14])*0.5   (sqrt(3)*L2*sin(x[5])*sin(x[6])*x[15])*0.5-(sqrt(3)*L2*cos(x[5])*cos(x[6])*x[14])*0.5   0.0   0.0   0.0   -L1*sin(x[1])   -L2*sin(x[2])   0.0   -(L1*sin(x[4]))*0.5   -(L2*sin(x[5]))*0.5-(sqrt(3)*L2*cos(x[5])*sin(x[6]))*0.5   -(sqrt(3)*L2*cos(x[6])*sin(x[5]))*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            -L1*sin(x[1])*x[10]   -L2*cos(x[3])*sin(x[2])*x[11]-L2*cos(x[2])*sin(x[3])*x[12]   -L2*cos(x[2])*sin(x[3])*x[11]-L2*cos(x[3])*sin(x[2])*x[12]   L1*sin(x[4])*x[13]   L2*cos(x[6])*sin(x[5])*x[14]+L2*cos(x[5])*sin(x[6])*x[15]   L2*cos(x[5])*sin(x[6])*x[14]+L2*cos(x[6])*sin(x[5])*x[15]   0.0   0.0   0.0   L1*cos(x[1])   L2*cos(x[2])*cos(x[3])   -L2*sin(x[2])*sin(x[3])   -L1*cos(x[4])   -L2*cos(x[5])*cos(x[6])   L2*sin(x[5])*sin(x[6])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   L2*cos(x[2])*cos(x[3])*x[12]-L2*sin(x[2])*sin(x[3])*x[11]   L2*cos(x[2])*cos(x[3])*x[11]-L2*sin(x[2])*sin(x[3])*x[12]   0.0   0.0   0.0   (sqrt(3)*L1*cos(x[7])*x[16])*0.5   (L2*cos(x[8])*cos(x[9])*x[18])*0.5+(sqrt(3)*L2*cos(x[8])*x[17])*0.5-(L2*sin(x[8])*sin(x[9])*x[17])*0.5   (L2*cos(x[8])*cos(x[9])*x[17])*0.5-(L2*sin(x[8])*sin(x[9])*x[18])*0.5   0.0   L2*cos(x[2])*sin(x[3])   L2*cos(x[3])*sin(x[2])   0.0   0.0   0.0   (sqrt(3)*L1*sin(x[7]))*0.5   (L2*cos(x[8])*sin(x[9]))*0.5+(sqrt(3)*L2*sin(x[8]))*0.5   (L2*cos(x[9])*sin(x[8]))*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            -L1*cos(x[1])*x[10]   -L2*cos(x[2])*x[11]   0.0   0.0   0.0   0.0   -(L1*cos(x[7])*x[16])*0.5   (sqrt(3)*L2*cos(x[8])*cos(x[9])*x[18])*0.5-(L2*cos(x[8])*x[17])*0.5-(sqrt(3)*L2*sin(x[8])*sin(x[9])*x[17])*0.5   (sqrt(3)*L2*cos(x[8])*cos(x[9])*x[17])*0.5-(sqrt(3)*L2*sin(x[8])*sin(x[9])*x[18])*0.5   -L1*sin(x[1])   -L2*sin(x[2])   0.0   0.0   0.0   0.0   -(L1*sin(x[7]))*0.5   (sqrt(3)*L2*cos(x[8])*sin(x[9]))*0.5-(L2*sin(x[8]))*0.5   (sqrt(3)*L2*cos(x[9])*sin(x[8]))*0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            -L1*sin(x[1])*x[10]   -L2*cos(x[3])*sin(x[2])*x[11]-L2*cos(x[2])*sin(x[3])*x[12]   -L2*cos(x[2])*sin(x[3])*x[11]-L2*cos(x[3])*sin(x[2])*x[12]   0.0   0.0   0.0   L1*sin(x[7])*x[16]   L2*cos(x[9])*sin(x[8])*x[17]+L2*cos(x[8])*sin(x[9])*x[18]   L2*cos(x[8])*sin(x[9])*x[17]+L2*cos(x[9])*sin(x[8])*x[18]   L1*cos(x[1])   L2*cos(x[2])*cos(x[3])   -L2*sin(x[2])*sin(x[3])   0.0   0.0   0.0   -L1*cos(x[7])   -L2*cos(x[8])*cos(x[9])   L2*sin(x[8])*sin(x[9])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   -L2*cos(x[2])*sin(x[3])   -L2*cos(x[3])*sin(x[2])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0
            L1*sin(x[1])   L2*sin(x[2])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0
            -L1*cos(x[1])   -L2*cos(x[2])*cos(x[3])   L2*sin(x[2])*sin(x[3])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0]
        Fdx = (x,dx) -> [ # TODO: Should we use sparse matrices? Is that worthwile? Might improve efficiency
            1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*sin(x[1])   L1*cos(x[1])   0.0   -L1*sin(x[1])   L1*cos(x[1])   0.0   0.0   0.0
            0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L2*cos(x[2])*sin(x[3])   -L2*sin(x[2])   L2*cos(x[2])*cos(x[3])   L2*cos(x[2])*sin(x[3])   -L2*sin(x[2])   L2*cos(x[2])*cos(x[3])   0.0   0.0   0.0
            0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L2*cos(x[3])*sin(x[2])   0.0   -L2*sin(x[2])*sin(x[3])   L2*cos(x[3])*sin(x[2])   0.0   -L2*sin(x[2])*sin(x[3])   0.0   0.0   0.0
            0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -(sqrt(3)*L1*sin(x[4]))*0.5   -(L1*sin(x[4]))*0.5   -L1*cos(x[4])   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x[5])*sin(x[6]))*0.5-(sqrt(3)*L2*sin(x[5]))*0.5   -(L2*sin(x[5]))*0.5-(sqrt(3)*L2*cos(x[5])*sin(x[6]))*0.5   -L2*cos(x[5])*cos(x[6])   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x[6])*sin(x[5]))*0.5   -(sqrt(3)*L2*cos(x[6])*sin(x[5]))*0.5   L2*sin(x[5])*sin(x[6])   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (sqrt(3)*L1*sin(x[7]))*0.5   -(L1*sin(x[7]))*0.5   -L1*cos(x[7])   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x[8])*sin(x[9]))*0.5+(sqrt(3)*L2*sin(x[8]))*0.5   (sqrt(3)*L2*cos(x[8])*sin(x[9]))*0.5-(L2*sin(x[8]))*0.5   -L2*cos(x[8])*cos(x[9])   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x[9])*sin(x[8]))*0.5   (sqrt(3)*L2*cos(x[9])*sin(x[8]))*0.5   L2*sin(x[8])*sin(x[9])   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(x[1])*sin(x[2])+cos(x[1])*cos(x[2])*cos(x[3]))   -L1*cos(x[1])*sin(x[2])*sin(x[3])*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*sin(x[1])   -L1*cos(x[1])   0.0   L1*sin(x[1])   -L1*cos(x[1])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*(L2*M3+LC2*M2)*(sin(x[1])*sin(x[2])+cos(x[1])*cos(x[2])*cos(x[3]))   J2+L2^2*M3+LC2^2*M2   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L2*cos(x[2])*sin(x[3])   L2*sin(x[2])   -L2*cos(x[2])*cos(x[3])   -L2*cos(x[2])*sin(x[3])   L2*sin(x[2])   -L2*cos(x[2])*cos(x[3])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*cos(x[1])*sin(x[2])*sin(x[3])*(L2*M3+LC2*M2)   0.0   sin(x[2])^2*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   -L2*cos(x[3])*sin(x[2])   0.0   L2*sin(x[2])*sin(x[3])   -L2*cos(x[3])*sin(x[2])   0.0   L2*sin(x[2])*sin(x[3])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(x[4])*sin(x[5])+cos(x[4])*cos(x[5])*cos(x[6]))   -L1*cos(x[4])*sin(x[5])*sin(x[6])*(L2*M3+LC2*M2)   0.0   0.0   0.0   (sqrt(3)*L1*sin(x[4]))*0.5   (L1*sin(x[4]))*0.5   L1*cos(x[4])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*(L2*M3+LC2*M2)*(sin(x[4])*sin(x[5])+cos(x[4])*cos(x[5])*cos(x[6]))   J2+L2^2*M3+LC2^2*M2   0.0   0.0   0.0   0.0   (sqrt(3)*L2*sin(x[5]))*0.5-(L2*cos(x[5])*sin(x[6]))*0.5   (L2*sin(x[5]))*0.5+(sqrt(3)*L2*cos(x[5])*sin(x[6]))*0.5   L2*cos(x[5])*cos(x[6])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*cos(x[4])*sin(x[5])*sin(x[6])*(L2*M3+LC2*M2)   0.0   sin(x[5])^2*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   -(L2*cos(x[6])*sin(x[5]))*0.5   (sqrt(3)*L2*cos(x[6])*sin(x[5]))*0.5   -L2*sin(x[5])*sin(x[6])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(x[7])*sin(x[8])+cos(x[7])*cos(x[8])*cos(x[9]))   -L1*cos(x[7])*sin(x[8])*sin(x[9])*(L2*M3+LC2*M2)   0.0   0.0   0.0   -(sqrt(3)*L1*sin(x[7]))*0.5   (L1*sin(x[7]))*0.5   L1*cos(x[7])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*(L2*M3+LC2*M2)*(sin(x[7])*sin(x[8])+cos(x[7])*cos(x[8])*cos(x[9]))   J2+L2^2*M3+LC2^2*M2   0.0   0.0   0.0   0.0   -(L2*cos(x[8])*sin(x[9]))*0.5-(sqrt(3)*L2*sin(x[8]))*0.5   (L2*sin(x[8]))*0.5-(sqrt(3)*L2*cos(x[8])*sin(x[9]))*0.5   L2*cos(x[8])*cos(x[9])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*cos(x[7])*sin(x[8])*sin(x[9])*(L2*M3+LC2*M2)   0.0   sin(x[8])^2*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   -(L2*cos(x[9])*sin(x[8]))*0.5   -(sqrt(3)*L2*cos(x[9])*sin(x[8]))*0.5   -L2*sin(x[8])*sin(x[9])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0]
        Fddx = (x,dx) -> [
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*cos(x[1])*dx[1]   -L1*sin(x[1])*dx[1]   0.0   -L1*cos(x[1])*dx[1]   -L1*sin(x[1])*dx[1]   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L2*cos(x[2])*cos(x[3])*dx[3]-L2*sin(x[2])*sin(x[3])*dx[2]   -L2*cos(x[2])*dx[2]   -L2*cos(x[3])*sin(x[2])*dx[2]-L2*cos(x[2])*sin(x[3])*dx[3]   L2*cos(x[2])*cos(x[3])*dx[3]-L2*sin(x[2])*sin(x[3])*dx[2]   -L2*cos(x[2])*dx[2]   -L2*cos(x[3])*sin(x[2])*dx[2]-L2*cos(x[2])*sin(x[3])*dx[3]   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L2*cos(x[2])*cos(x[3])*dx[2]-L2*sin(x[2])*sin(x[3])*dx[3]   0.0   -L2*cos(x[2])*sin(x[3])*dx[2]-L2*cos(x[3])*sin(x[2])*dx[3]   L2*cos(x[2])*cos(x[3])*dx[2]-L2*sin(x[2])*sin(x[3])*dx[3]   0.0   -L2*cos(x[2])*sin(x[3])*dx[2]-L2*cos(x[3])*sin(x[2])*dx[3]   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -(sqrt(3)*L1*cos(x[4])*dx[4])*0.5   -(L1*cos(x[4])*dx[4])*0.5   L1*sin(x[4])*dx[4]   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x[5])*cos(x[6])*dx[6])*0.5-(sqrt(3)*L2*cos(x[5])*dx[5])*0.5-(L2*sin(x[5])*sin(x[6])*dx[5])*0.5   (sqrt(3)*L2*sin(x[5])*sin(x[6])*dx[5])*0.5-(sqrt(3)*L2*cos(x[5])*cos(x[6])*dx[6])*0.5-(L2*cos(x[5])*dx[5])*0.5   L2*cos(x[6])*sin(x[5])*dx[5]+L2*cos(x[5])*sin(x[6])*dx[6]   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x[5])*cos(x[6])*dx[5])*0.5-(L2*sin(x[5])*sin(x[6])*dx[6])*0.5   (sqrt(3)*L2*sin(x[5])*sin(x[6])*dx[6])*0.5-(sqrt(3)*L2*cos(x[5])*cos(x[6])*dx[5])*0.5   L2*cos(x[5])*sin(x[6])*dx[5]+L2*cos(x[6])*sin(x[5])*dx[6]   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (sqrt(3)*L1*cos(x[7])*dx[7])*0.5   -(L1*cos(x[7])*dx[7])*0.5   L1*sin(x[7])*dx[7]   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x[8])*cos(x[9])*dx[9])*0.5+(sqrt(3)*L2*cos(x[8])*dx[8])*0.5-(L2*sin(x[8])*sin(x[9])*dx[8])*0.5   (sqrt(3)*L2*cos(x[8])*cos(x[9])*dx[9])*0.5-(L2*cos(x[8])*dx[8])*0.5-(sqrt(3)*L2*sin(x[8])*sin(x[9])*dx[8])*0.5   L2*cos(x[9])*sin(x[8])*dx[8]+L2*cos(x[8])*sin(x[9])*dx[9]   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x[8])*cos(x[9])*dx[8])*0.5-(L2*sin(x[8])*sin(x[9])*dx[9])*0.5   (sqrt(3)*L2*cos(x[8])*cos(x[9])*dx[8])*0.5-(sqrt(3)*L2*sin(x[8])*sin(x[9])*dx[9])*0.5   L2*cos(x[8])*sin(x[9])*dx[8]+L2*cos(x[9])*sin(x[8])*dx[9]   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*(L2*M3+LC2*M2)*(cos(x[2])*cos(x[3])*sin(x[1])*dx[1]-cos(x[2])*sin(x[1])*dx[2]-cos(x[1])*sin(x[2])*dx[1]+cos(x[1])*cos(x[3])*sin(x[2])*dx[2]+cos(x[1])*cos(x[2])*sin(x[3])*dx[3])   L1*sin(x[1])*sin(x[2])*sin(x[3])*dx[1]*(L2*M3+LC2*M2)-L1*cos(x[1])*cos(x[3])*sin(x[2])*dx[3]*(L2*M3+LC2*M2)-L1*cos(x[1])*cos(x[2])*sin(x[3])*dx[2]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*cos(x[1])*dx[1]   L1*sin(x[1])*dx[1]   0.0   L1*cos(x[1])*dx[1]   L1*sin(x[1])*dx[1]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*(L2*M3+LC2*M2)*(cos(x[2])*cos(x[3])*sin(x[1])*dx[1]-cos(x[2])*sin(x[1])*dx[2]-cos(x[1])*sin(x[2])*dx[1]+cos(x[1])*cos(x[3])*sin(x[2])*dx[2]+cos(x[1])*cos(x[2])*sin(x[3])*dx[3])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L2*sin(x[2])*sin(x[3])*dx[2]-L2*cos(x[2])*cos(x[3])*dx[3]   L2*cos(x[2])*dx[2]   L2*cos(x[3])*sin(x[2])*dx[2]+L2*cos(x[2])*sin(x[3])*dx[3]   L2*sin(x[2])*sin(x[3])*dx[2]-L2*cos(x[2])*cos(x[3])*dx[3]   L2*cos(x[2])*dx[2]   L2*cos(x[3])*sin(x[2])*dx[2]+L2*cos(x[2])*sin(x[3])*dx[3]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*sin(x[1])*sin(x[2])*sin(x[3])*dx[1]*(L2*M3+LC2*M2)-L1*cos(x[1])*cos(x[3])*sin(x[2])*dx[3]*(L2*M3+LC2*M2)-L1*cos(x[1])*cos(x[2])*sin(x[3])*dx[2]*(L2*M3+LC2*M2)   0.0   2*cos(x[2])*sin(x[2])*dx[2]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   L2*sin(x[2])*sin(x[3])*dx[3]-L2*cos(x[2])*cos(x[3])*dx[2]   0.0   L2*cos(x[2])*sin(x[3])*dx[2]+L2*cos(x[3])*sin(x[2])*dx[3]   L2*sin(x[2])*sin(x[3])*dx[3]-L2*cos(x[2])*cos(x[3])*dx[2]   0.0   L2*cos(x[2])*sin(x[3])*dx[2]+L2*cos(x[3])*sin(x[2])*dx[3]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*(L2*M3+LC2*M2)*(cos(x[5])*cos(x[6])*sin(x[4])*dx[4]-cos(x[5])*sin(x[4])*dx[5]-cos(x[4])*sin(x[5])*dx[4]+cos(x[4])*cos(x[6])*sin(x[5])*dx[5]+cos(x[4])*cos(x[5])*sin(x[6])*dx[6])   L1*sin(x[4])*sin(x[5])*sin(x[6])*dx[4]*(L2*M3+LC2*M2)-L1*cos(x[4])*cos(x[6])*sin(x[5])*dx[6]*(L2*M3+LC2*M2)-L1*cos(x[4])*cos(x[5])*sin(x[6])*dx[5]*(L2*M3+LC2*M2)   0.0   0.0   0.0   (sqrt(3)*L1*cos(x[4])*dx[4])*0.5   (L1*cos(x[4])*dx[4])*0.5   -L1*sin(x[4])*dx[4]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*(L2*M3+LC2*M2)*(cos(x[5])*cos(x[6])*sin(x[4])*dx[4]-cos(x[5])*sin(x[4])*dx[5]-cos(x[4])*sin(x[5])*dx[4]+cos(x[4])*cos(x[6])*sin(x[5])*dx[5]+cos(x[4])*cos(x[5])*sin(x[6])*dx[6])   0.0   0.0   0.0   0.0   0.0   (sqrt(3)*L2*cos(x[5])*dx[5])*0.5-(L2*cos(x[5])*cos(x[6])*dx[6])*0.5+(L2*sin(x[5])*sin(x[6])*dx[5])*0.5   (L2*cos(x[5])*dx[5])*0.5+(sqrt(3)*L2*cos(x[5])*cos(x[6])*dx[6])*0.5-(sqrt(3)*L2*sin(x[5])*sin(x[6])*dx[5])*0.5   -L2*cos(x[6])*sin(x[5])*dx[5]-L2*cos(x[5])*sin(x[6])*dx[6]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*sin(x[4])*sin(x[5])*sin(x[6])*dx[4]*(L2*M3+LC2*M2)-L1*cos(x[4])*cos(x[6])*sin(x[5])*dx[6]*(L2*M3+LC2*M2)-L1*cos(x[4])*cos(x[5])*sin(x[6])*dx[5]*(L2*M3+LC2*M2)   0.0   2*cos(x[5])*sin(x[5])*dx[5]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   (L2*sin(x[5])*sin(x[6])*dx[6])*0.5-(L2*cos(x[5])*cos(x[6])*dx[5])*0.5   (sqrt(3)*L2*cos(x[5])*cos(x[6])*dx[5])*0.5-(sqrt(3)*L2*sin(x[5])*sin(x[6])*dx[6])*0.5   -L2*cos(x[5])*sin(x[6])*dx[5]-L2*cos(x[6])*sin(x[5])*dx[6]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*(L2*M3+LC2*M2)*(cos(x[8])*cos(x[9])*sin(x[7])*dx[7]-cos(x[8])*sin(x[7])*dx[8]-cos(x[7])*sin(x[8])*dx[7]+cos(x[7])*cos(x[9])*sin(x[8])*dx[8]+cos(x[7])*cos(x[8])*sin(x[9])*dx[9])   L1*sin(x[7])*sin(x[8])*sin(x[9])*dx[7]*(L2*M3+LC2*M2)-L1*cos(x[7])*cos(x[9])*sin(x[8])*dx[9]*(L2*M3+LC2*M2)-L1*cos(x[7])*cos(x[8])*sin(x[9])*dx[8]*(L2*M3+LC2*M2)   0.0   0.0   0.0   -(sqrt(3)*L1*cos(x[7])*dx[7])*0.5   (L1*cos(x[7])*dx[7])*0.5   -L1*sin(x[7])*dx[7]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*(L2*M3+LC2*M2)*(cos(x[8])*cos(x[9])*sin(x[7])*dx[7]-cos(x[8])*sin(x[7])*dx[8]-cos(x[7])*sin(x[8])*dx[7]+cos(x[7])*cos(x[9])*sin(x[8])*dx[8]+cos(x[7])*cos(x[8])*sin(x[9])*dx[9])   0.0   0.0   0.0   0.0   0.0   (L2*sin(x[8])*sin(x[9])*dx[8])*0.5-(sqrt(3)*L2*cos(x[8])*dx[8])*0.5-(L2*cos(x[8])*cos(x[9])*dx[9])*0.5   (L2*cos(x[8])*dx[8])*0.5-(sqrt(3)*L2*cos(x[8])*cos(x[9])*dx[9])*0.5+(sqrt(3)*L2*sin(x[8])*sin(x[9])*dx[8])*0.5   -L2*cos(x[9])*sin(x[8])*dx[8]-L2*cos(x[8])*sin(x[9])*dx[9]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*sin(x[7])*sin(x[8])*sin(x[9])*dx[7]*(L2*M3+LC2*M2)-L1*cos(x[7])*cos(x[9])*sin(x[8])*dx[9]*(L2*M3+LC2*M2)-L1*cos(x[7])*cos(x[8])*sin(x[9])*dx[8]*(L2*M3+LC2*M2)   0.0   2*cos(x[8])*sin(x[8])*dx[8]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   (L2*sin(x[8])*sin(x[9])*dx[9])*0.5-(L2*cos(x[8])*cos(x[9])*dx[8])*0.5   (sqrt(3)*L2*sin(x[8])*sin(x[9])*dx[9])*0.5-(sqrt(3)*L2*cos(x[8])*cos(x[9])*dx[8])*0.5   -L2*cos(x[8])*sin(x[9])*dx[8]-L2*cos(x[9])*sin(x[8])*dx[9]   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0]
        gₓ = (x,dx,t) -> [0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; -(2*y(t)[1]-2*x[31])/T; -(2*y(t)[2]-2*x[32])/T; -(2*y(t)[3]-2*x[33])/T]

        # the residual function
        function f!(res, dz, z, θ, t)
            wt = w(t)
            ut = u(t)
            xt = x_func(t)
            dxt = dx_func(t)
            x = xt[1:33]
            dx = dxt[1:33]
            xθ = xt[34:end]
            dxθ = dxt[34:end]

            res[1] = dz[1] - (gₓ(x,dx,t)')*xθ
            res[2] = dz[2] - ( (gₓ(x,dx,t)')*xθ - first((λ(t)')*(Fp(x,dx) + Fx(x,dx)*xθ + Fdx(x,dx)*dxθ)) )
            res[3] = dz[3] - ( -first((λ(t)')*Fp(x,dx)) - ( -gₓ(x,dx,t)' + (λ(t)')*(Fx(x,dx)-Fddx(x,dx)) - (dλ(t)')*Fdx(x,dx) )*xθ )
            res[4] = dz[4] - first(( -(λ(t)')*Fp(x,dx) ))
            # SOME BONUS DEBUG TERMS
            res[5] = dz[5] - ((x[31]-y(t)[1])^2 + (x[32]-y(t)[2])^2 + (x[33]-y(t)[3])^2)/T 
            res[6] = dz[6] + first((λ(t)')*(Fp(x,dx) + Fx(x,dx)*xθ + Fdx(x,dx)*dxθ))    # This should be approximately zero, otherwise forward solution isn't right

            nothing
        end

        # Finding consistent initial conditions
        x_func0 = x_func(0.)
        dx_func0 = dx_func(0.)
        x0 = x_func0[1:33]
        dx0 = dx_func0[1:33]
        xθ0 = x_func0[34:end]
        dxθ0 = dx_func0[34:end]

        dz10 = (gₓ(x0,dx0,0.)')*xθ0
        dz20 = (gₓ(x0,dx0,0.)')*xθ0 - first((λ(0.)')*(Fp(x0,dx0) + Fx(x0,dx0)*xθ0 + Fdx(x0,dx0)*dxθ0))
        dz30 = -first((λ(0.)')*Fp(x0,dx0)) - ( -gₓ(x0,dx0,0.)' + (λ(0.)')*(Fx(x0,dx0)-Fddx(x0,dx0)) - (dλ(0.)')*Fdx(x0,dx0) )*xθ0
        dz40 = -first((λ(0.)')*Fp(x0,dx0))
        dz50 = ((x0[31]-y(0.)[1])^2 + (x0[32]-y(0.)[2])^2 + (x0[33]-y(0.)[3])^2)/T
        dz60 = -first((λ(0.)')*(Fp(x0,dx0) + Fx(x0,dx0)*xθ0 + Fdx(x0,dx0)*dxθ0))

        z0   = zeros(6)
        dz0  = [dz10, dz20, dz30, dz40, dz50, dz60]

        dvars = fill(true,6)

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, z0, dz0, dvars, r0)
    end
end

# Robert product, very inefficient implementation (actually is it that inefficient?). TODO: I don't think this is used anymore, consider removing
function rp(A::AbstractMatrix, B::AbstractMatrix)
    return [A[:,1:9]*B A[:,10:18]*B A[:,19:27]*B A[:,28:36]*B A[:,37:45]*B A[:,46:54]*B A[:,55:63]*B A[:,64:72]*B A[:,73:81]*B ]
end

function rp(A::AbstractMatrix, B::AbstractVector)
    return [A[:,1:9]*B A[:,10:18]*B A[:,19:27]*B A[:,28:36]*B A[:,37:45]*B A[:,46:54]*B A[:,55:63]*B A[:,64:72]*B A[:,73:81]*B ]
end

function get_delta_initial_J1sens_comp(θ, u0, w0)
    let L0 = θ[1], L1 = θ[2], L2 = θ[3], L3 = θ[4], LC1 = θ[5], LC2 = θ[6], M1 = θ[7], M2 = θ[8], M3 = θ[9], J1 = θ[10], J2 = θ[11], g = 0.0, γ = θ[13]

        z  = zeros(60)
        dz = zeros(60)
        z[1:3] = [π/6; acos(-0.5*sqrt(3)*L1/L2 + (L3-L0)/L2); 0.0]
        z[4:6] = [π/6; acos(-0.5*sqrt(3)*L1/L2 + (L3-L0)/L2); 0.0]
        z[7:9] = [π/6; acos(-0.5*sqrt(3)*L1/L2 + (L3-L0)/L2); 0.0]

        # Sensitivity of initila value is zero wrt to J1, so no need to compute it in this case
        # z[26] = 0.0
        # z[29] = 0.0
        # z[32] = 0.0


        ################### INITIALIZING NOMINAL MODEL PART ########################

        H = [
            0.0   L2*cos(z[2])*sin(z[3])   L2*cos(z[3])*sin(z[2])   -(sqrt(3)*L1*sin(z[4]))*0.5   (L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5   (L2*cos(z[6])*sin(z[5]))*0.5   0.0   0.0   0
            -L1*sin(z[1])   -L2*sin(z[2])   0.0   -(L1*sin(z[4]))*0.5   -(L2*sin(z[5]))*0.5-(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5   -(sqrt(3)*L2*cos(z[6])*sin(z[5]))*0.5   0.0   0.0   0
            L1*cos(z[1])   L2*cos(z[2])*cos(z[3])   -L2*sin(z[2])*sin(z[3])   -L1*cos(z[4])   -L2*cos(z[5])*cos(z[6])   L2*sin(z[5])*sin(z[6])   0.0   0.0   0
            0.0   L2*cos(z[2])*sin(z[3])   L2*cos(z[3])*sin(z[2])   0.0   0.0   0.0   (sqrt(3)*L1*sin(z[7]))*0.5   (L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5   (L2*cos(z[9])*sin(z[8]))*0.5
            -L1*sin(z[1])   -L2*sin(z[2])   0.0   0.0   0.0   0.0   -(L1*sin(z[7]))*0.5   (sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5-(L2*sin(z[8]))*0.5   (sqrt(3)*L2*cos(z[9])*sin(z[8]))*0.5
            L1*cos(z[1])   L2*cos(z[2])*cos(z[3])   -L2*sin(z[2])*sin(z[3])   0.0   0.0   0.0   -L1*cos(z[7])   -L2*cos(z[8])*cos(z[9])   L2*sin(z[8])*sin(z[9])
        ]

        m1 = [
            J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))   -L1*cos(z[1])*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)
            L1*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))   J2+L2^2*M3+LC2^2*M2   0
            -L1*cos(z[1])*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)   0.0   sin(z[2])^2*(J2+L2^2*M3+LC2^2*M2)
        ]
        m2 = [
            J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))   -L1*cos(z[4])*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)
            L1*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))   J2+L2^2*M3+LC2^2*M2   0
            -L1*cos(z[4])*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)   0.0   sin(z[5])^2*(J2+L2^2*M3+LC2^2*M2)
        ]
        m3 = [
            J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))   -L1*cos(z[7])*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)
            L1*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))   J2+L2^2*M3+LC2^2*M2   0
            -L1*cos(z[7])*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)   0.0   sin(z[8])^2*(J2+L2^2*M3+LC2^2*M2)
        ]

        HinvM = [
            H[:,1:3]/m1   H[:,4:6]/m2   H[:,7:9]/m3
        ]

        cgBu = [
            γ*z[10]-u0[1]+w0[1]^2-g*cos(z[1])*(L1*(M2+M3)+LC1*M1)+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2)
            γ*z[11]-cos(z[2])*sin(z[2])*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-g*cos(z[2])*cos(z[3])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))
            γ*z[12]+sin(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+g*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*sin(z[1])*sin(z[2])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)
            γ*z[13]-u0[2]+w0[2]^2-g*cos(z[4])*(L1*(M2+M3)+LC1*M1)+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2)
            γ*z[14]-cos(z[5])*sin(z[5])*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-g*cos(z[5])*cos(z[6])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))
            γ*z[15]+sin(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)+g*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)+L1*sin(z[4])*sin(z[5])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)
            γ*z[16]-u0[3]+w0[3]^2-g*cos(z[7])*(L1*(M2+M3)+LC1*M1)+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2)
            γ*z[17]-cos(z[8])*sin(z[8])*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-g*cos(z[8])*cos(z[9])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))
            γ*z[18]+sin(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)+g*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)+L1*sin(z[7])*sin(z[8])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)
        ]

        # dκ = (HinvM*transpose(H))\(HinvM*cgBu)
        dz[19:24] = (HinvM*transpose(H))\(HinvM*cgBu)

        # dv = inv(M)*Mvterm
        dz[10:18] = [
            m1\[
                u0[1]+w0[1]^2-γ*z[10]+g*cos(z[1])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[1])*dz[21]+L1*cos(z[1])*dz[24]-L1*sin(z[1])*dz[20]-L1*sin(z[1])*dz[23]-L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))+L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2)
                L2*cos(z[2])*cos(z[3])*dz[21]-L2*sin(z[2])*dz[20]-L2*sin(z[2])*dz[23]-γ*z[11]+L2*cos(z[2])*cos(z[3])*dz[24]+L2*cos(z[2])*sin(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[22]+cos(z[2])*sin(z[2])*z[12]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[2])*cos(z[3])*(L2*M3+LC2*M2)-L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))
                L2*cos(z[3])*sin(z[2])*dz[19]-γ*z[12]+L2*cos(z[3])*sin(z[2])*dz[22]-L2*sin(z[2])*sin(z[3])*dz[21]-L2*sin(z[2])*sin(z[3])*dz[24]-sin(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)-g*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)-L1*sin(z[1])*sin(z[2])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)
            ]
            m2\[
                u0[2]+w0[2]^2-γ*z[13]+g*cos(z[4])*(L1*(M2+M3)+LC1*M1)-L1*cos(z[4])*dz[21]-(L1*sin(z[4])*dz[20])*0.5-(sqrt(3)*L1*sin(z[4])*dz[19])*0.5-L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))+L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2)
                dz[19]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)-dz[20]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-γ*z[14]-L2*cos(z[5])*cos(z[6])*dz[21]+cos(z[5])*sin(z[5])*z[15]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[5])*cos(z[6])*(L2*M3+LC2*M2)-L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))
                (L2*cos(z[6])*sin(z[5])*dz[19])*0.5-γ*z[15]+L2*sin(z[5])*sin(z[6])*dz[21]-sin(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)-g*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)-(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[20])*0.5-L1*sin(z[4])*sin(z[5])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)
            ]
            m3\[
                u0[3]+w0[3]^2-γ*z[16]+g*cos(z[7])*(L1*(M2+M3)+LC1*M1)-L1*cos(z[7])*dz[24]-(L1*sin(z[7])*dz[23])*0.5+(sqrt(3)*L1*sin(z[7])*dz[22])*0.5-L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))+L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2)
                dz[22]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)-dz[23]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)-γ*z[17]-L2*cos(z[8])*cos(z[9])*dz[24]+cos(z[8])*sin(z[8])*z[18]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[8])*cos(z[9])*(L2*M3+LC2*M2)-L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))
                (L2*cos(z[9])*sin(z[8])*dz[22])*0.5-γ*z[18]+L2*sin(z[8])*sin(z[9])*dz[24]-sin(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)-g*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[23])*0.5-L1*sin(z[7])*sin(z[8])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)
            ]
        ]

        ############################# INITIALIZING SENSITIVITY PART ####################################

        # Block matrix product behaviour reminder
        # [a11 a12 a13] [inv(m1) 0 0]   [a11*inv(m1) a12*inv(m2) a13*inv(m3)]
        # [a21 a22 a23]*[0 inv(m2) 0] = [a21*inv(m1) a22*inv(m2) a23*inv(m3)] = [a_1*inv(m1) a_2*inv(m2) a_3*inv(m3)]
        # [a31 a32 a33] [0 0 inv(m3)]   [a31*inv(m1) a32*inv(m2) a33*inv(m3)]
        # and
        # [inv(m1) 0 0] [a11 a12 a13]   [inv(m1)*a11 inv(m2)*a12 inv(m3)*a13]   [inv(m1)*a1_]
        # [0 inv(m2) 0]*[a21 a22 a23] = [inv(m1)*a21 inv(m2)*a22 inv(m3)*a23] = [inv(m2)*a2_]
        # [0 0 inv(m3)] [a31 a32 a33]   [inv(m1)*a31 inv(m2)*a32 inv(m3)*a33]   [inv(m3)*a3_]

        dH_dp = [   # only sans_p-part, partial derivative wrt J1 is zero
            0.0   L2*cos(z[2])*cos(z[3])*z[33]-L2*sin(z[2])*sin(z[3])*z[32]   L2*cos(z[2])*cos(z[3])*z[32]-L2*sin(z[2])*sin(z[3])*z[33]   -(sqrt(3)*L1*cos(z[4])*z[34])*0.5   (L2*cos(z[5])*cos(z[6])*z[36])*0.5-z[35]*((sqrt(3)*L2*cos(z[5]))*0.5+(L2*sin(z[5])*sin(z[6]))*0.5)   (L2*cos(z[5])*cos(z[6])*z[35])*0.5-(L2*sin(z[5])*sin(z[6])*z[36])*0.5   0.0   0.0   0
            -2*L1*cos(z[1])*z[31]   -L2*cos(z[2])*z[32]   0.0   -(L1*cos(z[4])*z[34])*0.5   -z[35]*((L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5)-(sqrt(3)*L2*cos(z[5])*cos(z[6])*z[36])*0.5   (sqrt(3)*L2*sin(z[5])*sin(z[6])*z[36])*0.5-(sqrt(3)*L2*cos(z[5])*cos(z[6])*z[35])*0.5   0.0   0.0   0
            -2*L1*sin(z[1])*z[31]   -L2*cos(z[3])*sin(z[2])*z[32]-L2*cos(z[2])*sin(z[3])*z[33]   -L2*cos(z[2])*sin(z[3])*z[32]-L2*cos(z[3])*sin(z[2])*z[33]   L1*sin(z[4])*z[34]   L2*cos(z[6])*sin(z[5])*z[35]+L2*cos(z[5])*sin(z[6])*z[36]   L2*cos(z[5])*sin(z[6])*z[35]+L2*cos(z[6])*sin(z[5])*z[36]   0.0   0.0   0
            0.0   L2*cos(z[2])*cos(z[3])*z[33]-L2*sin(z[2])*sin(z[3])*z[32]   L2*cos(z[2])*cos(z[3])*z[32]-L2*sin(z[2])*sin(z[3])*z[33]   0.0   0.0   0.0   (sqrt(3)*L1*cos(z[7])*z[37])*0.5   z[38]*((sqrt(3)*L2*cos(z[8]))*0.5-(L2*sin(z[8])*sin(z[9]))*0.5)+(L2*cos(z[8])*cos(z[9])*z[39])*0.5   (L2*cos(z[8])*cos(z[9])*z[38])*0.5-(L2*sin(z[8])*sin(z[9])*z[39])*0.5
            -2*L1*cos(z[1])*z[31]   -L2*cos(z[2])*z[32]   0.0   0.0   0.0   0.0   -(L1*cos(z[7])*z[37])*0.5   (sqrt(3)*L2*cos(z[8])*cos(z[9])*z[39])*0.5-z[38]*((L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5)   (sqrt(3)*L2*cos(z[8])*cos(z[9])*z[38])*0.5-(sqrt(3)*L2*sin(z[8])*sin(z[9])*z[39])*0.5
            -2*L1*sin(z[1])*z[31]   -L2*cos(z[3])*sin(z[2])*z[32]-L2*cos(z[2])*sin(z[3])*z[33]   -L2*cos(z[2])*sin(z[3])*z[32]-L2*cos(z[3])*sin(z[2])*z[33]   0.0   0.0   0.0   L1*sin(z[7])*z[37]   L2*cos(z[9])*sin(z[8])*z[38]+L2*cos(z[8])*sin(z[9])*z[39]   L2*cos(z[8])*sin(z[9])*z[38]+L2*cos(z[9])*sin(z[8])*z[39]        
        ]

        dM_dp = [   # Summed together sans_p- and p-term, since p-term was so simple (just a couple of ones)
            1.0   2*L1*z[31]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))+L1*z[32]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-L1*cos(z[1])*cos(z[2])*sin(z[3])*z[33]*(L2*M3+LC2*M2)   2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[31]*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[3])*sin(z[2])*z[33]*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[2])*sin(z[3])*z[32]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0
            2*L1*z[31]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))+L1*z[32]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-L1*cos(z[1])*cos(z[2])*sin(z[3])*z[33]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0
            2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[31]*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[3])*sin(z[2])*z[33]*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[2])*sin(z[3])*z[32]*(L2*M3+LC2*M2)   0.0   2*cos(z[2])*sin(z[2])*z[32]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0
            0.0   0.0   0.0   1.0   L1*z[34]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))+L1*z[35]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-L1*cos(z[4])*cos(z[5])*sin(z[6])*z[36]*(L2*M3+LC2*M2)   L1*sin(z[4])*sin(z[5])*sin(z[6])*z[34]*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[6])*sin(z[5])*z[36]*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[5])*sin(z[6])*z[35]*(L2*M3+LC2*M2)   0.0   0.0   0
            0.0   0.0   0.0   L1*z[34]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))+L1*z[35]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-L1*cos(z[4])*cos(z[5])*sin(z[6])*z[36]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0
            0.0   0.0   0.0   L1*sin(z[4])*sin(z[5])*sin(z[6])*z[34]*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[6])*sin(z[5])*z[36]*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[5])*sin(z[6])*z[35]*(L2*M3+LC2*M2)   0.0   2*cos(z[5])*sin(z[5])*z[35]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0
            0.0   0.0   0.0   0.0   0.0   0.0   1.0   L1*z[37]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))+L1*z[38]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-L1*cos(z[7])*cos(z[8])*sin(z[9])*z[39]*(L2*M3+LC2*M2)   L1*sin(z[7])*sin(z[8])*sin(z[9])*z[37]*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[9])*sin(z[8])*z[39]*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[8])*sin(z[9])*z[38]*(L2*M3+LC2*M2)
            0.0   0.0   0.0   0.0   0.0   0.0   L1*z[37]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))+L1*z[38]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-L1*cos(z[7])*cos(z[8])*sin(z[9])*z[39]*(L2*M3+LC2*M2)   0.0   0
            0.0   0.0   0.0   0.0   0.0   0.0   L1*sin(z[7])*sin(z[8])*sin(z[9])*z[37]*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[9])*sin(z[8])*z[39]*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[8])*sin(z[9])*z[38]*(L2*M3+LC2*M2)   0.0   2*cos(z[8])*sin(z[8])*z[38]*(J2+L2^2*M3+LC2^2*M2)        
        ]

        dM_dpinvM = [dM_dp[:,1:3]/m1   dM_dp[:,4:6]/m2   dM_dp[:,7:9]/m3]
        dH_dpinvM = [dH_dp[:,1:3]/m1   dH_dp[:,4:6]/m2   dH_dp[:,7:9]/m3]

        orange_term = dH_dpinvM - HinvM*dM_dpinvM
        d_HinvMHT_dp = orange_term*transpose(H) + HinvM*transpose(dH_dp)

        # dκp = [H*inv(M)*transpose(H)]\{orange_term*cgBu + H*inv(M)*dcgBu/dp) - [orange_term*transpose(H)+H*inv(M)*transpose(dH/dp)] }
        dz[49:54] = (HinvM*transpose(H))\(orange_term*cgBu + HinvM*[
            # This matrix is dcgBu_p_sansp, p-part it zero for p=J1
            z[41]*(2*L1*z[11]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[12]*(L2*M3+LC2*M2))-z[42]*(2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]*(L2*M3+LC2*M2))-z[32]*(L1*z[11]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))+L1*cos(z[1])*cos(z[2])*cos(z[3])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+γ*z[40]+z[31]*(g*sin(z[1])*(L1*(M2+M3)+LC1*M1)+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2]))+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)+2*L1*cos(z[2])*sin(z[1])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+z[33]*(L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]^2*(L2*M3+LC2*M2)+L1*cos(z[1])*sin(z[2])*sin(z[3])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*cos(z[2])*cos(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))
            z[33]*(g*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2))+γ*z[41]+z[32]*(sin(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2])))-2*cos(z[2])*sin(z[2])*z[42]*z[12]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[40]*z[10]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))-L1*z[31]*z[10]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))
            z[42]*(γ+sin(2*z[2])*z[11]*(J2+L2^2*M3+LC2^2*M2))+z[33]*(g*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[10]^2*(L2*M3+LC2*M2))+z[32]*(2*cos(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2))+sin(2*z[2])*z[41]*z[12]*(J2+L2^2*M3+LC2^2*M2)+L1*cos(z[1])*sin(z[2])*sin(z[3])*z[31]*z[10]^2*(L2*M3+LC2*M2)+2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[40]*z[10]*(L2*M3+LC2*M2)
            z[44]*(2*L1*z[14]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[15]*(L2*M3+LC2*M2))-z[45]*(2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]*(L2*M3+LC2*M2))-z[35]*(L1*z[14]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))+L1*cos(z[4])*cos(z[5])*cos(z[6])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+γ*z[43]+z[34]*(g*sin(z[4])*(L1*(M2+M3)+LC1*M1)+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5]))+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)+2*L1*cos(z[5])*sin(z[4])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+z[36]*(L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*sin(z[5])*sin(z[6])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*cos(z[5])*cos(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))
            z[36]*(g*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2))+γ*z[44]+z[35]*(sin(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5])))-2*cos(z[5])*sin(z[5])*z[45]*z[15]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[43]*z[13]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))-L1*z[34]*z[13]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))
            z[45]*(γ+sin(2*z[5])*z[14]*(J2+L2^2*M3+LC2^2*M2))+z[36]*(g*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[13]^2*(L2*M3+LC2*M2))+z[35]*(2*cos(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2))+sin(2*z[5])*z[44]*z[15]*(J2+L2^2*M3+LC2^2*M2)+L1*cos(z[4])*sin(z[5])*sin(z[6])*z[34]*z[13]^2*(L2*M3+LC2*M2)+2*L1*sin(z[4])*sin(z[5])*sin(z[6])*z[43]*z[13]*(L2*M3+LC2*M2)
            z[47]*(2*L1*z[17]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[18]*(L2*M3+LC2*M2))-z[48]*(2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]*(L2*M3+LC2*M2))-z[38]*(L1*z[17]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))+L1*cos(z[7])*cos(z[8])*cos(z[9])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+γ*z[46]+z[37]*(g*sin(z[7])*(L1*(M2+M3)+LC1*M1)+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8]))+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)+2*L1*cos(z[8])*sin(z[7])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+z[39]*(L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*sin(z[8])*sin(z[9])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*cos(z[8])*cos(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))
            z[39]*(g*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2))+γ*z[47]+z[38]*(sin(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8])))-2*cos(z[8])*sin(z[8])*z[48]*z[18]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[46]*z[16]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))-L1*z[37]*z[16]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))
            z[48]*(γ+sin(2*z[8])*z[17]*(J2+L2^2*M3+LC2^2*M2))+z[39]*(g*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[16]^2*(L2*M3+LC2*M2))+z[38]*(2*cos(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2))+sin(2*z[8])*z[47]*z[18]*(J2+L2^2*M3+LC2^2*M2)+L1*cos(z[7])*sin(z[8])*sin(z[9])*z[37]*z[16]^2*(L2*M3+LC2*M2)+2*L1*sin(z[7])*sin(z[8])*sin(z[9])*z[46]*z[16]*(L2*M3+LC2*M2)
        ] - d_HinvMHT_dp*z[19:24])

        dMvterm_dp = [
            z[42]*(2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]*(L2*M3+LC2*M2))-z[31]*(g*sin(z[1])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[1])*dz[20]+L1*cos(z[1])*dz[23]+L1*sin(z[1])*dz[21]+L1*sin(z[1])*dz[24]+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2]))+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)+2*L1*cos(z[2])*sin(z[1])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))-z[41]*(2*L1*z[11]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[12]*(L2*M3+LC2*M2))+z[32]*(L1*z[11]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))+L1*cos(z[1])*cos(z[2])*cos(z[3])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))-γ*z[40]-z[33]*(L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]^2*(L2*M3+LC2*M2)+L1*cos(z[1])*sin(z[2])*sin(z[3])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*cos(z[2])*cos(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+L1*cos(z[1])*dz[51]+L1*cos(z[1])*dz[54]-L1*sin(z[1])*dz[50]-L1*sin(z[1])*dz[53]
            L2*cos(z[2])*cos(z[3])*dz[51]-γ*z[41]-z[32]*(sin(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)+L2*cos(z[2])*dz[20]+L2*cos(z[2])*dz[23]+L2*cos(z[3])*sin(z[2])*dz[21]+L2*cos(z[3])*sin(z[2])*dz[24]+L2*sin(z[2])*sin(z[3])*dz[19]+L2*sin(z[2])*sin(z[3])*dz[22]+g*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2])))-L2*sin(z[2])*dz[50]-L2*sin(z[2])*dz[53]-z[33]*(L2*cos(z[2])*sin(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[22]-L2*cos(z[2])*cos(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[24]+g*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2))+L2*cos(z[2])*cos(z[3])*dz[54]+L2*cos(z[2])*sin(z[3])*dz[49]+L2*cos(z[2])*sin(z[3])*dz[52]+2*cos(z[2])*sin(z[2])*z[42]*z[12]*(J2+L2^2*M3+LC2^2*M2)-2*L1*z[40]*z[10]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))+L1*z[31]*z[10]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))
            L2*cos(z[3])*sin(z[2])*dz[49]-z[33]*(L2*cos(z[3])*sin(z[2])*dz[21]+L2*cos(z[3])*sin(z[2])*dz[24]+L2*sin(z[2])*sin(z[3])*dz[19]+L2*sin(z[2])*sin(z[3])*dz[22]+g*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[10]^2*(L2*M3+LC2*M2))-z[42]*(γ+sin(2*z[2])*z[11]*(J2+L2^2*M3+LC2^2*M2))-z[32]*(L2*cos(z[2])*sin(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[22]-L2*cos(z[2])*cos(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[24]+2*cos(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2))+L2*cos(z[3])*sin(z[2])*dz[52]-L2*sin(z[2])*sin(z[3])*dz[51]-L2*sin(z[2])*sin(z[3])*dz[54]-sin(2*z[2])*z[41]*z[12]*(J2+L2^2*M3+LC2^2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*z[31]*z[10]^2*(L2*M3+LC2*M2)-2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[40]*z[10]*(L2*M3+LC2*M2)
            z[45]*(2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]*(L2*M3+LC2*M2))-z[44]*(2*L1*z[14]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[15]*(L2*M3+LC2*M2))+z[35]*(L1*z[14]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))+L1*cos(z[4])*cos(z[5])*cos(z[6])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))-z[34]*(g*sin(z[4])*(L1*(M2+M3)+LC1*M1)+(L1*cos(z[4])*dz[20])*0.5-L1*sin(z[4])*dz[21]+(sqrt(3)*L1*cos(z[4])*dz[19])*0.5+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5]))+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)+2*L1*cos(z[5])*sin(z[4])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))-γ*z[43]-z[36]*(L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*sin(z[5])*sin(z[6])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*cos(z[5])*cos(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))-L1*cos(z[4])*dz[51]-(L1*sin(z[4])*dz[50])*0.5-(sqrt(3)*L1*sin(z[4])*dz[49])*0.5
            dz[49]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)-dz[50]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-z[36]*(g*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)-L2*cos(z[5])*sin(z[6])*dz[21]-(L2*cos(z[5])*cos(z[6])*dz[19])*0.5+(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[20])*0.5+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2))-z[35]*(dz[19]*((sqrt(3)*L2*cos(z[5]))*0.5+(L2*sin(z[5])*sin(z[6]))*0.5)+dz[20]*((L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5)-cos(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)+sin(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[6])*sin(z[5])*dz[21]+g*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5])))-γ*z[44]-L2*cos(z[5])*cos(z[6])*dz[51]+2*cos(z[5])*sin(z[5])*z[45]*z[15]*(J2+L2^2*M3+LC2^2*M2)-2*L1*z[43]*z[13]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))+L1*z[34]*z[13]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))
            (L2*cos(z[6])*sin(z[5])*dz[49])*0.5-z[35]*(2*cos(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[5])*sin(z[6])*dz[21]-(L2*cos(z[5])*cos(z[6])*dz[19])*0.5+g*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[20])*0.5+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2))-z[36]*((L2*sin(z[5])*sin(z[6])*dz[19])*0.5-L2*cos(z[6])*sin(z[5])*dz[21]+g*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)-(sqrt(3)*L2*sin(z[5])*sin(z[6])*dz[20])*0.5+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[13]^2*(L2*M3+LC2*M2))-z[45]*(γ+sin(2*z[5])*z[14]*(J2+L2^2*M3+LC2^2*M2))+L2*sin(z[5])*sin(z[6])*dz[51]-sin(2*z[5])*z[44]*z[15]*(J2+L2^2*M3+LC2^2*M2)-(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[50])*0.5-L1*cos(z[4])*sin(z[5])*sin(z[6])*z[34]*z[13]^2*(L2*M3+LC2*M2)-2*L1*sin(z[4])*sin(z[5])*sin(z[6])*z[43]*z[13]*(L2*M3+LC2*M2)
            z[48]*(2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]*(L2*M3+LC2*M2))-z[47]*(2*L1*z[17]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[18]*(L2*M3+LC2*M2))+z[38]*(L1*z[17]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))+L1*cos(z[7])*cos(z[8])*cos(z[9])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))-z[37]*(g*sin(z[7])*(L1*(M2+M3)+LC1*M1)+(L1*cos(z[7])*dz[23])*0.5-L1*sin(z[7])*dz[24]-(sqrt(3)*L1*cos(z[7])*dz[22])*0.5+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8]))+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)+2*L1*cos(z[8])*sin(z[7])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))-γ*z[46]-z[39]*(L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*sin(z[8])*sin(z[9])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*cos(z[8])*cos(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))-L1*cos(z[7])*dz[54]-(L1*sin(z[7])*dz[53])*0.5+(sqrt(3)*L1*sin(z[7])*dz[52])*0.5
            dz[52]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)-dz[53]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)+z[39]*((L2*cos(z[8])*cos(z[9])*dz[22])*0.5+L2*cos(z[8])*sin(z[9])*dz[24]-g*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[23])*0.5-L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2))-z[38]*(dz[23]*((L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5)-dz[22]*((sqrt(3)*L2*cos(z[8]))*0.5-(L2*sin(z[8])*sin(z[9]))*0.5)-cos(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)+sin(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[9])*sin(z[8])*dz[24]+g*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8])))-γ*z[47]-L2*cos(z[8])*cos(z[9])*dz[54]+2*cos(z[8])*sin(z[8])*z[48]*z[18]*(J2+L2^2*M3+LC2^2*M2)-2*L1*z[46]*z[16]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))+L1*z[37]*z[16]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))
            z[38]*((L2*cos(z[8])*cos(z[9])*dz[22])*0.5+L2*cos(z[8])*sin(z[9])*dz[24]-2*cos(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)-g*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[23])*0.5-L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2))-z[48]*(γ+sin(2*z[8])*z[17]*(J2+L2^2*M3+LC2^2*M2))-z[39]*((L2*sin(z[8])*sin(z[9])*dz[22])*0.5-L2*cos(z[9])*sin(z[8])*dz[24]+g*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+(sqrt(3)*L2*sin(z[8])*sin(z[9])*dz[23])*0.5+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[16]^2*(L2*M3+LC2*M2))+(L2*cos(z[9])*sin(z[8])*dz[52])*0.5+L2*sin(z[8])*sin(z[9])*dz[54]-sin(2*z[8])*z[47]*z[18]*(J2+L2^2*M3+LC2^2*M2)+(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[53])*0.5-L1*cos(z[7])*sin(z[8])*sin(z[9])*z[37]*z[16]^2*(L2*M3+LC2*M2)-2*L1*sin(z[7])*sin(z[8])*sin(z[9])*z[46]*z[16]*(L2*M3+LC2*M2)
        ]   # only sans_p-term, partial derivative wrt to J1 is zero

        # dvp = inv(M)*(dMvterm_dp - rp(dM/dp, dv)), rp is my special product
        dz[40:48] = [m1\(dMvterm_dp[1:3,:] - dM_dp[1:3,:]*dz[10:18])
            m2\(dMvterm_dp[4:6,:] - dM_dp[4:6,:]*dz[10:18])
            m3\(dMvterm_dp[7:9,:] - dM_dp[7:9,:]*dz[10:18])]

        return z, dz
    end
end

function get_delta_initial_L1sens_comp(θ, u0, w0)
    let L0 = θ[1], L1 = θ[2], L2 = θ[3], L3 = θ[4], LC1 = θ[5], LC2 = θ[6], M1 = θ[7], M2 = θ[8], M3 = θ[9], J1 = θ[10], J2 = θ[11], g = 0.0, γ = θ[13]

        z  = zeros(60)
        dz = zeros(60)
        z[1:3] = [π/6; acos(-0.5*sqrt(3)*L1/L2 + (L3-L0)/L2); 0.0]
        z[4:6] = [π/6; acos(-0.5*sqrt(3)*L1/L2 + (L3-L0)/L2); 0.0]
        z[7:9] = [π/6; acos(-0.5*sqrt(3)*L1/L2 + (L3-L0)/L2); 0.0]

        # Sensitivity of initial value wrt to L1
        z[32] = sqrt(3)/(2*L2*(1-(L0-L3+(sqrt(3)*L1)*0.5)^2/L2^2)^(0.5))
        z[35] = sqrt(3)/(2*L2*(1-(L0-L3+(sqrt(3)*L1)*0.5)^2/L2^2)^(0.5))
        z[38] = sqrt(3)/(2*L2*(1-(L0-L3+(sqrt(3)*L1)*0.5)^2/L2^2)^(0.5))


        ################### INITIALIZING NOMINAL MODEL PART ########################

        H = [
            0.0   L2*cos(z[2])*sin(z[3])   L2*cos(z[3])*sin(z[2])   -(sqrt(3)*L1*sin(z[4]))*0.5   (L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5   (L2*cos(z[6])*sin(z[5]))*0.5   0.0   0.0   0
            -L1*sin(z[1])   -L2*sin(z[2])   0.0   -(L1*sin(z[4]))*0.5   -(L2*sin(z[5]))*0.5-(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5   -(sqrt(3)*L2*cos(z[6])*sin(z[5]))*0.5   0.0   0.0   0
            L1*cos(z[1])   L2*cos(z[2])*cos(z[3])   -L2*sin(z[2])*sin(z[3])   -L1*cos(z[4])   -L2*cos(z[5])*cos(z[6])   L2*sin(z[5])*sin(z[6])   0.0   0.0   0
            0.0   L2*cos(z[2])*sin(z[3])   L2*cos(z[3])*sin(z[2])   0.0   0.0   0.0   (sqrt(3)*L1*sin(z[7]))*0.5   (L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5   (L2*cos(z[9])*sin(z[8]))*0.5
            -L1*sin(z[1])   -L2*sin(z[2])   0.0   0.0   0.0   0.0   -(L1*sin(z[7]))*0.5   (sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5-(L2*sin(z[8]))*0.5   (sqrt(3)*L2*cos(z[9])*sin(z[8]))*0.5
            L1*cos(z[1])   L2*cos(z[2])*cos(z[3])   -L2*sin(z[2])*sin(z[3])   0.0   0.0   0.0   -L1*cos(z[7])   -L2*cos(z[8])*cos(z[9])   L2*sin(z[8])*sin(z[9])
        ]

        m1 = [
            J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))   -L1*cos(z[1])*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)
            L1*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))   J2+L2^2*M3+LC2^2*M2   0
            -L1*cos(z[1])*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)   0.0   sin(z[2])^2*(J2+L2^2*M3+LC2^2*M2)
        ]
        m2 = [
            J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))   -L1*cos(z[4])*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)
            L1*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))   J2+L2^2*M3+LC2^2*M2   0
            -L1*cos(z[4])*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)   0.0   sin(z[5])^2*(J2+L2^2*M3+LC2^2*M2)
        ]
        m3 = [
            J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))   -L1*cos(z[7])*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)
            L1*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))   J2+L2^2*M3+LC2^2*M2   0
            -L1*cos(z[7])*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)   0.0   sin(z[8])^2*(J2+L2^2*M3+LC2^2*M2)
        ]

        HinvM = [
            H[:,1:3]/m1   H[:,4:6]/m2   H[:,7:9]/m3
        ]

        cgBu = [
            γ*z[10]-u0[1]+w0[1]^2-g*cos(z[1])*(L1*(M2+M3)+LC1*M1)+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2)
            γ*z[11]-cos(z[2])*sin(z[2])*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-g*cos(z[2])*cos(z[3])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))
            γ*z[12]+sin(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+g*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*sin(z[1])*sin(z[2])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)
            γ*z[13]-u0[2]+w0[2]^2-g*cos(z[4])*(L1*(M2+M3)+LC1*M1)+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2)
            γ*z[14]-cos(z[5])*sin(z[5])*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-g*cos(z[5])*cos(z[6])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))
            γ*z[15]+sin(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)+g*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)+L1*sin(z[4])*sin(z[5])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)
            γ*z[16]-u0[3]+w0[3]^2-g*cos(z[7])*(L1*(M2+M3)+LC1*M1)+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2)
            γ*z[17]-cos(z[8])*sin(z[8])*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-g*cos(z[8])*cos(z[9])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))
            γ*z[18]+sin(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)+g*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)+L1*sin(z[7])*sin(z[8])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)
        ]

        # dκ = (HinvM*transpose(H))\(HinvM*cgBu)
        dz[19:24] = (HinvM*transpose(H))\(HinvM*cgBu)

        # dv = inv(M)*Mvterm
        dz[10:18] = [
            m1\[
                u0[1]+w0[1]^2-γ*z[10]+g*cos(z[1])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[1])*dz[21]+L1*cos(z[1])*dz[24]-L1*sin(z[1])*dz[20]-L1*sin(z[1])*dz[23]-L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))+L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2)
                L2*cos(z[2])*cos(z[3])*dz[21]-L2*sin(z[2])*dz[20]-L2*sin(z[2])*dz[23]-γ*z[11]+L2*cos(z[2])*cos(z[3])*dz[24]+L2*cos(z[2])*sin(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[22]+cos(z[2])*sin(z[2])*z[12]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[2])*cos(z[3])*(L2*M3+LC2*M2)-L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))
                L2*cos(z[3])*sin(z[2])*dz[19]-γ*z[12]+L2*cos(z[3])*sin(z[2])*dz[22]-L2*sin(z[2])*sin(z[3])*dz[21]-L2*sin(z[2])*sin(z[3])*dz[24]-sin(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)-g*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)-L1*sin(z[1])*sin(z[2])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)
            ]
            m2\[
                u0[2]+w0[2]^2-γ*z[13]+g*cos(z[4])*(L1*(M2+M3)+LC1*M1)-L1*cos(z[4])*dz[21]-(L1*sin(z[4])*dz[20])*0.5-(sqrt(3)*L1*sin(z[4])*dz[19])*0.5-L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))+L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2)
                dz[19]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)-dz[20]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-γ*z[14]-L2*cos(z[5])*cos(z[6])*dz[21]+cos(z[5])*sin(z[5])*z[15]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[5])*cos(z[6])*(L2*M3+LC2*M2)-L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))
                (L2*cos(z[6])*sin(z[5])*dz[19])*0.5-γ*z[15]+L2*sin(z[5])*sin(z[6])*dz[21]-sin(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)-g*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)-(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[20])*0.5-L1*sin(z[4])*sin(z[5])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)
            ]
            m3\[
                u0[3]+w0[3]^2-γ*z[16]+g*cos(z[7])*(L1*(M2+M3)+LC1*M1)-L1*cos(z[7])*dz[24]-(L1*sin(z[7])*dz[23])*0.5+(sqrt(3)*L1*sin(z[7])*dz[22])*0.5-L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))+L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2)
                dz[22]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)-dz[23]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)-γ*z[17]-L2*cos(z[8])*cos(z[9])*dz[24]+cos(z[8])*sin(z[8])*z[18]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[8])*cos(z[9])*(L2*M3+LC2*M2)-L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))
                (L2*cos(z[9])*sin(z[8])*dz[22])*0.5-γ*z[18]+L2*sin(z[8])*sin(z[9])*dz[24]-sin(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)-g*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[23])*0.5-L1*sin(z[7])*sin(z[8])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)
            ]
        ]

        ############################# INITIALIZING SENSITIVITY PART ####################################

        # Block matrix product behaviour reminder
        # [a11 a12 a13] [inv(m1) 0 0]   [a11*inv(m1) a12*inv(m2) a13*inv(m3)]
        # [a21 a22 a23]*[0 inv(m2) 0] = [a21*inv(m1) a22*inv(m2) a23*inv(m3)] = [a_1*inv(m1) a_2*inv(m2) a_3*inv(m3)]
        # [a31 a32 a33] [0 0 inv(m3)]   [a31*inv(m1) a32*inv(m2) a33*inv(m3)]
        # and
        # [inv(m1) 0 0] [a11 a12 a13]   [inv(m1)*a11 inv(m2)*a12 inv(m3)*a13]   [inv(m1)*a1_]
        # [0 inv(m2) 0]*[a21 a22 a23] = [inv(m1)*a21 inv(m2)*a22 inv(m3)*a23] = [inv(m2)*a2_]
        # [0 0 inv(m3)] [a31 a32 a33]   [inv(m1)*a31 inv(m2)*a32 inv(m3)*a33]   [inv(m3)*a3_]

        dH_dp = [   # Hp_sansp
            0.0   L2*cos(z[2])*cos(z[3])*z[33]-L2*sin(z[2])*sin(z[3])*z[32]   L2*cos(z[2])*cos(z[3])*z[32]-L2*sin(z[2])*sin(z[3])*z[33]   -(sqrt(3)*L1*cos(z[4])*z[34])*0.5   (L2*cos(z[5])*cos(z[6])*z[36])*0.5-z[35]*((sqrt(3)*L2*cos(z[5]))*0.5+(L2*sin(z[5])*sin(z[6]))*0.5)   (L2*cos(z[5])*cos(z[6])*z[35])*0.5-(L2*sin(z[5])*sin(z[6])*z[36])*0.5   0.0   0.0   0
            -2*L1*cos(z[1])*z[31]   -L2*cos(z[2])*z[32]   0.0   -(L1*cos(z[4])*z[34])*0.5   -z[35]*((L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5)-(sqrt(3)*L2*cos(z[5])*cos(z[6])*z[36])*0.5   (sqrt(3)*L2*sin(z[5])*sin(z[6])*z[36])*0.5-(sqrt(3)*L2*cos(z[5])*cos(z[6])*z[35])*0.5   0.0   0.0   0
            -2*L1*sin(z[1])*z[31]   -L2*cos(z[3])*sin(z[2])*z[32]-L2*cos(z[2])*sin(z[3])*z[33]   -L2*cos(z[2])*sin(z[3])*z[32]-L2*cos(z[3])*sin(z[2])*z[33]   L1*sin(z[4])*z[34]   L2*cos(z[6])*sin(z[5])*z[35]+L2*cos(z[5])*sin(z[6])*z[36]   L2*cos(z[5])*sin(z[6])*z[35]+L2*cos(z[6])*sin(z[5])*z[36]   0.0   0.0   0
            0.0   L2*cos(z[2])*cos(z[3])*z[33]-L2*sin(z[2])*sin(z[3])*z[32]   L2*cos(z[2])*cos(z[3])*z[32]-L2*sin(z[2])*sin(z[3])*z[33]   0.0   0.0   0.0   (sqrt(3)*L1*cos(z[7])*z[37])*0.5   z[38]*((sqrt(3)*L2*cos(z[8]))*0.5-(L2*sin(z[8])*sin(z[9]))*0.5)+(L2*cos(z[8])*cos(z[9])*z[39])*0.5   (L2*cos(z[8])*cos(z[9])*z[38])*0.5-(L2*sin(z[8])*sin(z[9])*z[39])*0.5
            -2*L1*cos(z[1])*z[31]   -L2*cos(z[2])*z[32]   0.0   0.0   0.0   0.0   -(L1*cos(z[7])*z[37])*0.5   (sqrt(3)*L2*cos(z[8])*cos(z[9])*z[39])*0.5-z[38]*((L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5)   (sqrt(3)*L2*cos(z[8])*cos(z[9])*z[38])*0.5-(sqrt(3)*L2*sin(z[8])*sin(z[9])*z[39])*0.5
            -2*L1*sin(z[1])*z[31]   -L2*cos(z[3])*sin(z[2])*z[32]-L2*cos(z[2])*sin(z[3])*z[33]   -L2*cos(z[2])*sin(z[3])*z[32]-L2*cos(z[3])*sin(z[2])*z[33]   0.0   0.0   0.0   L1*sin(z[7])*z[37]   L2*cos(z[9])*sin(z[8])*z[38]+L2*cos(z[8])*sin(z[9])*z[39]   L2*cos(z[8])*sin(z[9])*z[38]+L2*cos(z[9])*sin(z[8])*z[39]        
        ] + [   # Hp for p=L1
            0	0	0	-(sqrt(3)*sin(z[4]))*0.5	0	0	0	0	0
            -sin(z[1])	0	0	-sin(z[4])*0.5	0	0	0	0	0
            cos(z[1])	0	0	-cos(z[4])	0	0	0	0	0
            0	0	0	0	0	0	(sqrt(3)*sin(z[7]))*0.5	0	0
            -sin(z[1])	0	0	0	0	0	-sin(z[7])*0.5	0	0
            cos(z[1])	0	0	0	0	0	-cos(z[7])	0	0
        ]

        dM_dp = [   # Mp_sansp
            0.0   2*L1*z[31]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))+L1*z[32]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-L1*cos(z[1])*cos(z[2])*sin(z[3])*z[33]*(L2*M3+LC2*M2)   2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[31]*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[3])*sin(z[2])*z[33]*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[2])*sin(z[3])*z[32]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0
            2*L1*z[31]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))+L1*z[32]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-L1*cos(z[1])*cos(z[2])*sin(z[3])*z[33]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[31]*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[3])*sin(z[2])*z[33]*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[2])*sin(z[3])*z[32]*(L2*M3+LC2*M2)   0.0   2*cos(z[2])*sin(z[2])*z[32]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   L1*z[34]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))+L1*z[35]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-L1*cos(z[4])*cos(z[5])*sin(z[6])*z[36]*(L2*M3+LC2*M2)   L1*sin(z[4])*sin(z[5])*sin(z[6])*z[34]*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[6])*sin(z[5])*z[36]*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[5])*sin(z[6])*z[35]*(L2*M3+LC2*M2)   0.0   0.0   0.0
            0.0   0.0   0.0   L1*z[34]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))+L1*z[35]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-L1*cos(z[4])*cos(z[5])*sin(z[6])*z[36]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   L1*sin(z[4])*sin(z[5])*sin(z[6])*z[34]*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[6])*sin(z[5])*z[36]*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[5])*sin(z[6])*z[35]*(L2*M3+LC2*M2)   0.0   2*cos(z[5])*sin(z[5])*z[35]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*z[37]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))+L1*z[38]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-L1*cos(z[7])*cos(z[8])*sin(z[9])*z[39]*(L2*M3+LC2*M2)   L1*sin(z[7])*sin(z[8])*sin(z[9])*z[37]*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[9])*sin(z[8])*z[39]*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[8])*sin(z[9])*z[38]*(L2*M3+LC2*M2)
            0.0   0.0   0.0   0.0   0.0   0.0   L1*z[37]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))+L1*z[38]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-L1*cos(z[7])*cos(z[8])*sin(z[9])*z[39]*(L2*M3+LC2*M2)   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   L1*sin(z[7])*sin(z[8])*sin(z[9])*z[37]*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[9])*sin(z[8])*z[39]*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[8])*sin(z[9])*z[38]*(L2*M3+LC2*M2)   0.0   2*cos(z[8])*sin(z[8])*z[38]*(J2+L2^2*M3+LC2^2*M2)        
        ] + [   # Mp for p=L1
            2*L1*(M2+M3)	(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))	-cos(z[1])*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)	0	0	0	0	0	0
            (L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))	0	0	0	0	0	0	0	0
            -cos(z[1])*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)	0	0	0	0	0	0	0	0
            0	0	0	2*L1*(M2+M3)	(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))	-cos(z[4])*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)	0	0	0
            0	0	0	(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))	0	0	0	0	0
            0	0	0	-cos(z[4])*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)	0	0	0	0	0
            0	0	0	0	0	0	2*L1*(M2+M3)	(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))	-cos(z[7])*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)
            0	0	0	0	0	0	(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))	0	0
            0	0	0	0	0	0	-cos(z[7])*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)	0	0
        ]

        dM_dpinvM = [dM_dp[:,1:3]/m1   dM_dp[:,4:6]/m2   dM_dp[:,7:9]/m3]
        dH_dpinvM = [dH_dp[:,1:3]/m1   dH_dp[:,4:6]/m2   dH_dp[:,7:9]/m3]

        orange_term = dH_dpinvM - HinvM*dM_dpinvM
        d_HinvMHT_dp = orange_term*transpose(H) + HinvM*transpose(dH_dp)

        # dκp = [H*inv(M)*transpose(H)]\{orange_term*cgBu + H*inv(M)*dcgBu/dp) - [orange_term*transpose(H)+H*inv(M)*transpose(dH/dp)] }
        dz[49:54] = (HinvM*transpose(H))\(orange_term*cgBu + HinvM*([
            # cgBu_p_sansp
            z[41]*(2*L1*z[11]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[12]*(L2*M3+LC2*M2))-z[42]*(2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]*(L2*M3+LC2*M2))-z[32]*(L1*z[11]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))+L1*cos(z[1])*cos(z[2])*cos(z[3])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+γ*z[40]+z[31]*(g*sin(z[1])*(L1*(M2+M3)+LC1*M1)+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2]))+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)+2*L1*cos(z[2])*sin(z[1])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+z[33]*(L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]^2*(L2*M3+LC2*M2)+L1*cos(z[1])*sin(z[2])*sin(z[3])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*cos(z[2])*cos(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))
            z[33]*(g*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2))+γ*z[41]+z[32]*(sin(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2])))-2*cos(z[2])*sin(z[2])*z[42]*z[12]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[40]*z[10]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))-L1*z[31]*z[10]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))
            z[42]*(γ+sin(2*z[2])*z[11]*(J2+L2^2*M3+LC2^2*M2))+z[33]*(g*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[10]^2*(L2*M3+LC2*M2))+z[32]*(2*cos(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2))+sin(2*z[2])*z[41]*z[12]*(J2+L2^2*M3+LC2^2*M2)+L1*cos(z[1])*sin(z[2])*sin(z[3])*z[31]*z[10]^2*(L2*M3+LC2*M2)+2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[40]*z[10]*(L2*M3+LC2*M2)
            z[44]*(2*L1*z[14]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[15]*(L2*M3+LC2*M2))-z[45]*(2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]*(L2*M3+LC2*M2))-z[35]*(L1*z[14]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))+L1*cos(z[4])*cos(z[5])*cos(z[6])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+γ*z[43]+z[34]*(g*sin(z[4])*(L1*(M2+M3)+LC1*M1)+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5]))+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)+2*L1*cos(z[5])*sin(z[4])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+z[36]*(L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*sin(z[5])*sin(z[6])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*cos(z[5])*cos(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))
            z[36]*(g*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2))+γ*z[44]+z[35]*(sin(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5])))-2*cos(z[5])*sin(z[5])*z[45]*z[15]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[43]*z[13]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))-L1*z[34]*z[13]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))
            z[45]*(γ+sin(2*z[5])*z[14]*(J2+L2^2*M3+LC2^2*M2))+z[36]*(g*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[13]^2*(L2*M3+LC2*M2))+z[35]*(2*cos(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2))+sin(2*z[5])*z[44]*z[15]*(J2+L2^2*M3+LC2^2*M2)+L1*cos(z[4])*sin(z[5])*sin(z[6])*z[34]*z[13]^2*(L2*M3+LC2*M2)+2*L1*sin(z[4])*sin(z[5])*sin(z[6])*z[43]*z[13]*(L2*M3+LC2*M2)
            z[47]*(2*L1*z[17]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[18]*(L2*M3+LC2*M2))-z[48]*(2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]*(L2*M3+LC2*M2))-z[38]*(L1*z[17]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))+L1*cos(z[7])*cos(z[8])*cos(z[9])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+γ*z[46]+z[37]*(g*sin(z[7])*(L1*(M2+M3)+LC1*M1)+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8]))+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)+2*L1*cos(z[8])*sin(z[7])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+z[39]*(L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*sin(z[8])*sin(z[9])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*cos(z[8])*cos(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))
            z[39]*(g*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2))+γ*z[47]+z[38]*(sin(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8])))-2*cos(z[8])*sin(z[8])*z[48]*z[18]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[46]*z[16]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))-L1*z[37]*z[16]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))
            z[48]*(γ+sin(2*z[8])*z[17]*(J2+L2^2*M3+LC2^2*M2))+z[39]*(g*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[16]^2*(L2*M3+LC2*M2))+z[38]*(2*cos(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2))+sin(2*z[8])*z[47]*z[18]*(J2+L2^2*M3+LC2^2*M2)+L1*cos(z[7])*sin(z[8])*sin(z[9])*z[37]*z[16]^2*(L2*M3+LC2*M2)+2*L1*sin(z[7])*sin(z[8])*sin(z[9])*z[46]*z[16]*(L2*M3+LC2*M2)
        ] + [   #cgBu_p for p=L1
            z[11]^2*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-g*cos(z[1])*(M2+M3)-cos(z[1])*cos(z[3])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)-2*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2)
            z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))
            sin(z[1])*sin(z[2])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)
            z[14]^2*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-g*cos(z[4])*(M2+M3)-cos(z[4])*cos(z[6])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)-2*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2)
            z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))
            sin(z[4])*sin(z[5])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)
            z[17]^2*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-g*cos(z[7])*(M2+M3)-cos(z[7])*cos(z[9])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)-2*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2)
            z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))
            sin(z[7])*sin(z[8])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)        
        ]) - d_HinvMHT_dp*z[19:24])     # d_HinvMHT_dp = orange_term*transpose(H) + HinvM*transpose(dH_dp)

        dMvterm_dp = [  # Mvterm_p_sansp
            z[42]*(2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]*(L2*M3+LC2*M2))-z[31]*(g*sin(z[1])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[1])*dz[20]+L1*cos(z[1])*dz[23]+L1*sin(z[1])*dz[21]+L1*sin(z[1])*dz[24]+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2]))+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)+2*L1*cos(z[2])*sin(z[1])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))-z[41]*(2*L1*z[11]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[12]*(L2*M3+LC2*M2))+z[32]*(L1*z[11]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))+L1*cos(z[1])*cos(z[2])*cos(z[3])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))-γ*z[40]-z[33]*(L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]^2*(L2*M3+LC2*M2)+L1*cos(z[1])*sin(z[2])*sin(z[3])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*cos(z[2])*cos(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+L1*cos(z[1])*dz[51]+L1*cos(z[1])*dz[54]-L1*sin(z[1])*dz[50]-L1*sin(z[1])*dz[53]
            L2*cos(z[2])*cos(z[3])*dz[51]-γ*z[41]-z[32]*(sin(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)+L2*cos(z[2])*dz[20]+L2*cos(z[2])*dz[23]+L2*cos(z[3])*sin(z[2])*dz[21]+L2*cos(z[3])*sin(z[2])*dz[24]+L2*sin(z[2])*sin(z[3])*dz[19]+L2*sin(z[2])*sin(z[3])*dz[22]+g*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2])))-L2*sin(z[2])*dz[50]-L2*sin(z[2])*dz[53]-z[33]*(L2*cos(z[2])*sin(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[22]-L2*cos(z[2])*cos(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[24]+g*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2))+L2*cos(z[2])*cos(z[3])*dz[54]+L2*cos(z[2])*sin(z[3])*dz[49]+L2*cos(z[2])*sin(z[3])*dz[52]+2*cos(z[2])*sin(z[2])*z[42]*z[12]*(J2+L2^2*M3+LC2^2*M2)-2*L1*z[40]*z[10]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))+L1*z[31]*z[10]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))
            L2*cos(z[3])*sin(z[2])*dz[49]-z[33]*(L2*cos(z[3])*sin(z[2])*dz[21]+L2*cos(z[3])*sin(z[2])*dz[24]+L2*sin(z[2])*sin(z[3])*dz[19]+L2*sin(z[2])*sin(z[3])*dz[22]+g*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[10]^2*(L2*M3+LC2*M2))-z[42]*(γ+sin(2*z[2])*z[11]*(J2+L2^2*M3+LC2^2*M2))-z[32]*(L2*cos(z[2])*sin(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[22]-L2*cos(z[2])*cos(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[24]+2*cos(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2))+L2*cos(z[3])*sin(z[2])*dz[52]-L2*sin(z[2])*sin(z[3])*dz[51]-L2*sin(z[2])*sin(z[3])*dz[54]-sin(2*z[2])*z[41]*z[12]*(J2+L2^2*M3+LC2^2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*z[31]*z[10]^2*(L2*M3+LC2*M2)-2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[40]*z[10]*(L2*M3+LC2*M2)
            z[45]*(2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]*(L2*M3+LC2*M2))-z[44]*(2*L1*z[14]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[15]*(L2*M3+LC2*M2))+z[35]*(L1*z[14]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))+L1*cos(z[4])*cos(z[5])*cos(z[6])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))-z[34]*(g*sin(z[4])*(L1*(M2+M3)+LC1*M1)+(L1*cos(z[4])*dz[20])*0.5-L1*sin(z[4])*dz[21]+(sqrt(3)*L1*cos(z[4])*dz[19])*0.5+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5]))+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)+2*L1*cos(z[5])*sin(z[4])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))-γ*z[43]-z[36]*(L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*sin(z[5])*sin(z[6])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*cos(z[5])*cos(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))-L1*cos(z[4])*dz[51]-(L1*sin(z[4])*dz[50])*0.5-(sqrt(3)*L1*sin(z[4])*dz[49])*0.5
            dz[49]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)-dz[50]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-z[36]*(g*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)-L2*cos(z[5])*sin(z[6])*dz[21]-(L2*cos(z[5])*cos(z[6])*dz[19])*0.5+(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[20])*0.5+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2))-z[35]*(dz[19]*((sqrt(3)*L2*cos(z[5]))*0.5+(L2*sin(z[5])*sin(z[6]))*0.5)+dz[20]*((L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5)-cos(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)+sin(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[6])*sin(z[5])*dz[21]+g*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5])))-γ*z[44]-L2*cos(z[5])*cos(z[6])*dz[51]+2*cos(z[5])*sin(z[5])*z[45]*z[15]*(J2+L2^2*M3+LC2^2*M2)-2*L1*z[43]*z[13]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))+L1*z[34]*z[13]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))
            (L2*cos(z[6])*sin(z[5])*dz[49])*0.5-z[35]*(2*cos(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[5])*sin(z[6])*dz[21]-(L2*cos(z[5])*cos(z[6])*dz[19])*0.5+g*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[20])*0.5+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2))-z[36]*((L2*sin(z[5])*sin(z[6])*dz[19])*0.5-L2*cos(z[6])*sin(z[5])*dz[21]+g*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)-(sqrt(3)*L2*sin(z[5])*sin(z[6])*dz[20])*0.5+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[13]^2*(L2*M3+LC2*M2))-z[45]*(γ+sin(2*z[5])*z[14]*(J2+L2^2*M3+LC2^2*M2))+L2*sin(z[5])*sin(z[6])*dz[51]-sin(2*z[5])*z[44]*z[15]*(J2+L2^2*M3+LC2^2*M2)-(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[50])*0.5-L1*cos(z[4])*sin(z[5])*sin(z[6])*z[34]*z[13]^2*(L2*M3+LC2*M2)-2*L1*sin(z[4])*sin(z[5])*sin(z[6])*z[43]*z[13]*(L2*M3+LC2*M2)
            z[48]*(2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]*(L2*M3+LC2*M2))-z[47]*(2*L1*z[17]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[18]*(L2*M3+LC2*M2))+z[38]*(L1*z[17]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))+L1*cos(z[7])*cos(z[8])*cos(z[9])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))-z[37]*(g*sin(z[7])*(L1*(M2+M3)+LC1*M1)+(L1*cos(z[7])*dz[23])*0.5-L1*sin(z[7])*dz[24]-(sqrt(3)*L1*cos(z[7])*dz[22])*0.5+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8]))+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)+2*L1*cos(z[8])*sin(z[7])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))-γ*z[46]-z[39]*(L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*sin(z[8])*sin(z[9])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*cos(z[8])*cos(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))-L1*cos(z[7])*dz[54]-(L1*sin(z[7])*dz[53])*0.5+(sqrt(3)*L1*sin(z[7])*dz[52])*0.5
            dz[52]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)-dz[53]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)+z[39]*((L2*cos(z[8])*cos(z[9])*dz[22])*0.5+L2*cos(z[8])*sin(z[9])*dz[24]-g*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[23])*0.5-L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2))-z[38]*(dz[23]*((L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5)-dz[22]*((sqrt(3)*L2*cos(z[8]))*0.5-(L2*sin(z[8])*sin(z[9]))*0.5)-cos(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)+sin(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[9])*sin(z[8])*dz[24]+g*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8])))-γ*z[47]-L2*cos(z[8])*cos(z[9])*dz[54]+2*cos(z[8])*sin(z[8])*z[48]*z[18]*(J2+L2^2*M3+LC2^2*M2)-2*L1*z[46]*z[16]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))+L1*z[37]*z[16]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))
            z[38]*((L2*cos(z[8])*cos(z[9])*dz[22])*0.5+L2*cos(z[8])*sin(z[9])*dz[24]-2*cos(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)-g*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[23])*0.5-L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2))-z[48]*(γ+sin(2*z[8])*z[17]*(J2+L2^2*M3+LC2^2*M2))-z[39]*((L2*sin(z[8])*sin(z[9])*dz[22])*0.5-L2*cos(z[9])*sin(z[8])*dz[24]+g*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+(sqrt(3)*L2*sin(z[8])*sin(z[9])*dz[23])*0.5+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[16]^2*(L2*M3+LC2*M2))+(L2*cos(z[9])*sin(z[8])*dz[52])*0.5+L2*sin(z[8])*sin(z[9])*dz[54]-sin(2*z[8])*z[47]*z[18]*(J2+L2^2*M3+LC2^2*M2)+(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[53])*0.5-L1*cos(z[7])*sin(z[8])*sin(z[9])*z[37]*z[16]^2*(L2*M3+LC2*M2)-2*L1*sin(z[7])*sin(z[8])*sin(z[9])*z[46]*z[16]*(L2*M3+LC2*M2)
        ] + [   # Mvterm_p for p=L1
            cos(z[1])*dz[21]+cos(z[1])*dz[24]-sin(z[1])*dz[20]-sin(z[1])*dz[23]+g*cos(z[1])*(M2+M3)-z[11]^2*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))+cos(z[1])*cos(z[3])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)+2*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2)
            -z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))
            -sin(z[1])*sin(z[2])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)
            g*cos(z[4])*(M2+M3)-(sin(z[4])*dz[20])*0.5-cos(z[4])*dz[21]-z[14]^2*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-(sqrt(3)*sin(z[4])*dz[19])*0.5+cos(z[4])*cos(z[6])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)+2*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2)
            -z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))
            -sin(z[4])*sin(z[5])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)
            g*cos(z[7])*(M2+M3)-(sin(z[7])*dz[23])*0.5-cos(z[7])*dz[24]-z[17]^2*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))+(sqrt(3)*sin(z[7])*dz[22])*0.5+cos(z[7])*cos(z[9])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)+2*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2)
            -z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))
            -sin(z[7])*sin(z[8])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)        
        ]

        # dvp = inv(M)*(dMvterm_dp - rp(dM/dp, dv)), rp is my special product
        dz[40:48] = [m1\(dMvterm_dp[1:3,:] - dM_dp[1:3,:]*dz[10:18])
            m2\(dMvterm_dp[4:6,:] - dM_dp[4:6,:]*dz[10:18])
            m3\(dMvterm_dp[7:9,:] - dM_dp[7:9,:]*dz[10:18])]

        return z, dz
    end
end

function get_delta_initial_γsens_comp(θ, u0, w0)
    let L0 = θ[1], L1 = θ[2], L2 = θ[3], L3 = θ[4], LC1 = θ[5], LC2 = θ[6], M1 = θ[7], M2 = θ[8], M3 = θ[9], J1 = θ[10], J2 = θ[11], g = 0.0, γ = θ[13]

        z  = zeros(60)
        dz = zeros(60)
        z[1:3] = [π/6; acos(-0.5*sqrt(3)*L1/L2 + (L3-L0)/L2); 0.0]
        z[4:6] = [π/6; acos(-0.5*sqrt(3)*L1/L2 + (L3-L0)/L2); 0.0]
        z[7:9] = [π/6; acos(-0.5*sqrt(3)*L1/L2 + (L3-L0)/L2); 0.0]

        # Sensitivity of initila value is zero wrt to γ, so no need to compute it in this case
        # z[26] = 0.0
        # z[29] = 0.0
        # z[32] = 0.0


        ################### INITIALIZING NOMINAL MODEL PART ########################

        H = [
            0.0   L2*cos(z[2])*sin(z[3])   L2*cos(z[3])*sin(z[2])   -(sqrt(3)*L1*sin(z[4]))*0.5   (L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5   (L2*cos(z[6])*sin(z[5]))*0.5   0.0   0.0   0.0
            -L1*sin(z[1])   -L2*sin(z[2])   0.0   -(L1*sin(z[4]))*0.5   -(L2*sin(z[5]))*0.5-(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5   -(sqrt(3)*L2*cos(z[6])*sin(z[5]))*0.5   0.0   0.0   0.0
            L1*cos(z[1])   L2*cos(z[2])*cos(z[3])   -L2*sin(z[2])*sin(z[3])   -L1*cos(z[4])   -L2*cos(z[5])*cos(z[6])   L2*sin(z[5])*sin(z[6])   0.0   0.0   0.0
            0.0   L2*cos(z[2])*sin(z[3])   L2*cos(z[3])*sin(z[2])   0.0   0.0   0.0   (sqrt(3)*L1*sin(z[7]))*0.5   (L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5   (L2*cos(z[9])*sin(z[8]))*0.5
            -L1*sin(z[1])   -L2*sin(z[2])   0.0   0.0   0.0   0.0   -(L1*sin(z[7]))*0.5   (sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5-(L2*sin(z[8]))*0.5   (sqrt(3)*L2*cos(z[9])*sin(z[8]))*0.5
            L1*cos(z[1])   L2*cos(z[2])*cos(z[3])   -L2*sin(z[2])*sin(z[3])   0.0   0.0   0.0   -L1*cos(z[7])   -L2*cos(z[8])*cos(z[9])   L2*sin(z[8])*sin(z[9])
        ]

        m1 = [
            J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))   -L1*cos(z[1])*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)
            L1*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))   J2+L2^2*M3+LC2^2*M2   0.0
            -L1*cos(z[1])*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)   0.0   sin(z[2])^2*(J2+L2^2*M3+LC2^2*M2)
        ]
        m2 = [
            J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))   -L1*cos(z[4])*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)
            L1*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))   J2+L2^2*M3+LC2^2*M2   0.0
            -L1*cos(z[4])*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)   0.0   sin(z[5])^2*(J2+L2^2*M3+LC2^2*M2)
        ]
        m3 = [
            J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))   -L1*cos(z[7])*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)
            L1*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))   J2+L2^2*M3+LC2^2*M2   0.0
            -L1*cos(z[7])*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)   0.0   sin(z[8])^2*(J2+L2^2*M3+LC2^2*M2)
        ]

        HinvM = [
            H[:,1:3]/m1   H[:,4:6]/m2   H[:,7:9]/m3
        ]

        cgBu = [
            γ*z[10]-u0[1]+w0[1]^2-g*cos(z[1])*(L1*(M2+M3)+LC1*M1)+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2)
            γ*z[11]-cos(z[2])*sin(z[2])*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-g*cos(z[2])*cos(z[3])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))
            γ*z[12]+sin(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+g*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*sin(z[1])*sin(z[2])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)
            γ*z[13]-u0[2]+w0[2]^2-g*cos(z[4])*(L1*(M2+M3)+LC1*M1)+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2)
            γ*z[14]-cos(z[5])*sin(z[5])*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-g*cos(z[5])*cos(z[6])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))
            γ*z[15]+sin(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)+g*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)+L1*sin(z[4])*sin(z[5])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)
            γ*z[16]-u0[3]+w0[3]^2-g*cos(z[7])*(L1*(M2+M3)+LC1*M1)+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2)
            γ*z[17]-cos(z[8])*sin(z[8])*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-g*cos(z[8])*cos(z[9])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))
            γ*z[18]+sin(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)+g*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)+L1*sin(z[7])*sin(z[8])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)
        ]

        # dκ = (HinvM*transpose(H))\(HinvM*cgBu)
        dz[19:24] = (HinvM*transpose(H))\(HinvM*cgBu)

        # dv = inv(M)*Mvterm
        dz[10:18] = [
            m1\[
                u0[1]+w0[1]^2-γ*z[10]+g*cos(z[1])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[1])*dz[21]+L1*cos(z[1])*dz[24]-L1*sin(z[1])*dz[20]-L1*sin(z[1])*dz[23]-L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))+L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2)
                L2*cos(z[2])*cos(z[3])*dz[21]-L2*sin(z[2])*dz[20]-L2*sin(z[2])*dz[23]-γ*z[11]+L2*cos(z[2])*cos(z[3])*dz[24]+L2*cos(z[2])*sin(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[22]+cos(z[2])*sin(z[2])*z[12]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[2])*cos(z[3])*(L2*M3+LC2*M2)-L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))
                L2*cos(z[3])*sin(z[2])*dz[19]-γ*z[12]+L2*cos(z[3])*sin(z[2])*dz[22]-L2*sin(z[2])*sin(z[3])*dz[21]-L2*sin(z[2])*sin(z[3])*dz[24]-sin(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)-g*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)-L1*sin(z[1])*sin(z[2])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)
            ]
            m2\[
                u0[2]+w0[2]^2-γ*z[13]+g*cos(z[4])*(L1*(M2+M3)+LC1*M1)-L1*cos(z[4])*dz[21]-(L1*sin(z[4])*dz[20])*0.5-(sqrt(3)*L1*sin(z[4])*dz[19])*0.5-L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))+L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2)
                dz[19]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)-dz[20]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-γ*z[14]-L2*cos(z[5])*cos(z[6])*dz[21]+cos(z[5])*sin(z[5])*z[15]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[5])*cos(z[6])*(L2*M3+LC2*M2)-L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))
                (L2*cos(z[6])*sin(z[5])*dz[19])*0.5-γ*z[15]+L2*sin(z[5])*sin(z[6])*dz[21]-sin(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)-g*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)-(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[20])*0.5-L1*sin(z[4])*sin(z[5])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)
            ]
            m3\[
                u0[3]+w0[3]^2-γ*z[16]+g*cos(z[7])*(L1*(M2+M3)+LC1*M1)-L1*cos(z[7])*dz[24]-(L1*sin(z[7])*dz[23])*0.5+(sqrt(3)*L1*sin(z[7])*dz[22])*0.5-L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))+L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2)
                dz[22]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)-dz[23]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)-γ*z[17]-L2*cos(z[8])*cos(z[9])*dz[24]+cos(z[8])*sin(z[8])*z[18]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[8])*cos(z[9])*(L2*M3+LC2*M2)-L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))
                (L2*cos(z[9])*sin(z[8])*dz[22])*0.5-γ*z[18]+L2*sin(z[8])*sin(z[9])*dz[24]-sin(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)-g*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[23])*0.5-L1*sin(z[7])*sin(z[8])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)
            ]
        ]

        ############################# INITIALIZING SENSITIVITY PART ####################################

        # Block matrix product behaviour reminder
        # [a11 a12 a13] [inv(m1) 0 0]   [a11*inv(m1) a12*inv(m2) a13*inv(m3)]
        # [a21 a22 a23]*[0 inv(m2) 0] = [a21*inv(m1) a22*inv(m2) a23*inv(m3)] = [a_1*inv(m1) a_2*inv(m2) a_3*inv(m3)]
        # [a31 a32 a33] [0 0 inv(m3)]   [a31*inv(m1) a32*inv(m2) a33*inv(m3)]
        # and
        # [inv(m1) 0 0] [a11 a12 a13]   [inv(m1)*a11 inv(m2)*a12 inv(m3)*a13]   [inv(m1)*a1_]
        # [0 inv(m2) 0]*[a21 a22 a23] = [inv(m1)*a21 inv(m2)*a22 inv(m3)*a23] = [inv(m2)*a2_]
        # [0 0 inv(m3)] [a31 a32 a33]   [inv(m1)*a31 inv(m2)*a32 inv(m3)*a33]   [inv(m3)*a3_]

        dH_dp = [   # Hp_sansp
            0.0   L2*cos(z[2])*cos(z[3])*z[33]-L2*sin(z[2])*sin(z[3])*z[32]   L2*cos(z[2])*cos(z[3])*z[32]-L2*sin(z[2])*sin(z[3])*z[33]   -(sqrt(3)*L1*cos(z[4])*z[34])*0.5   (L2*cos(z[5])*cos(z[6])*z[36])*0.5-z[35]*((sqrt(3)*L2*cos(z[5]))*0.5+(L2*sin(z[5])*sin(z[6]))*0.5)   (L2*cos(z[5])*cos(z[6])*z[35])*0.5-(L2*sin(z[5])*sin(z[6])*z[36])*0.5   0.0   0.0   0.0
            -2*L1*cos(z[1])*z[31]   -L2*cos(z[2])*z[32]   0.0   -(L1*cos(z[4])*z[34])*0.5   -z[35]*((L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5)-(sqrt(3)*L2*cos(z[5])*cos(z[6])*z[36])*0.5   (sqrt(3)*L2*sin(z[5])*sin(z[6])*z[36])*0.5-(sqrt(3)*L2*cos(z[5])*cos(z[6])*z[35])*0.5   0.0   0.0   0.0
            -2*L1*sin(z[1])*z[31]   -L2*cos(z[3])*sin(z[2])*z[32]-L2*cos(z[2])*sin(z[3])*z[33]   -L2*cos(z[2])*sin(z[3])*z[32]-L2*cos(z[3])*sin(z[2])*z[33]   L1*sin(z[4])*z[34]   L2*cos(z[6])*sin(z[5])*z[35]+L2*cos(z[5])*sin(z[6])*z[36]   L2*cos(z[5])*sin(z[6])*z[35]+L2*cos(z[6])*sin(z[5])*z[36]   0.0   0.0   0.0
            0.0   L2*cos(z[2])*cos(z[3])*z[33]-L2*sin(z[2])*sin(z[3])*z[32]   L2*cos(z[2])*cos(z[3])*z[32]-L2*sin(z[2])*sin(z[3])*z[33]   0.0   0.0   0.0   (sqrt(3)*L1*cos(z[7])*z[37])*0.5   z[38]*((sqrt(3)*L2*cos(z[8]))*0.5-(L2*sin(z[8])*sin(z[9]))*0.5)+(L2*cos(z[8])*cos(z[9])*z[39])*0.5   (L2*cos(z[8])*cos(z[9])*z[38])*0.5-(L2*sin(z[8])*sin(z[9])*z[39])*0.5
            -2*L1*cos(z[1])*z[31]   -L2*cos(z[2])*z[32]   0.0   0.0   0.0   0.0   -(L1*cos(z[7])*z[37])*0.5   (sqrt(3)*L2*cos(z[8])*cos(z[9])*z[39])*0.5-z[38]*((L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5)   (sqrt(3)*L2*cos(z[8])*cos(z[9])*z[38])*0.5-(sqrt(3)*L2*sin(z[8])*sin(z[9])*z[39])*0.5
            -2*L1*sin(z[1])*z[31]   -L2*cos(z[3])*sin(z[2])*z[32]-L2*cos(z[2])*sin(z[3])*z[33]   -L2*cos(z[2])*sin(z[3])*z[32]-L2*cos(z[3])*sin(z[2])*z[33]   0.0   0.0   0.0   L1*sin(z[7])*z[37]   L2*cos(z[9])*sin(z[8])*z[38]+L2*cos(z[8])*sin(z[9])*z[39]   L2*cos(z[8])*sin(z[9])*z[38]+L2*cos(z[9])*sin(z[8])*z[39]        
        ] # + 0 since Hp is zero for γ 

        dM_dp = [   # Mp_sansp
            0.0   2*L1*z[31]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))+L1*z[32]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-L1*cos(z[1])*cos(z[2])*sin(z[3])*z[33]*(L2*M3+LC2*M2)   2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[31]*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[3])*sin(z[2])*z[33]*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[2])*sin(z[3])*z[32]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0
            2*L1*z[31]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))+L1*z[32]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-L1*cos(z[1])*cos(z[2])*sin(z[3])*z[33]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[31]*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[3])*sin(z[2])*z[33]*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[2])*sin(z[3])*z[32]*(L2*M3+LC2*M2)   0.0   2*cos(z[2])*sin(z[2])*z[32]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   L1*z[34]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))+L1*z[35]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-L1*cos(z[4])*cos(z[5])*sin(z[6])*z[36]*(L2*M3+LC2*M2)   L1*sin(z[4])*sin(z[5])*sin(z[6])*z[34]*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[6])*sin(z[5])*z[36]*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[5])*sin(z[6])*z[35]*(L2*M3+LC2*M2)   0.0   0.0   0.0
            0.0   0.0   0.0   L1*z[34]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))+L1*z[35]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-L1*cos(z[4])*cos(z[5])*sin(z[6])*z[36]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   L1*sin(z[4])*sin(z[5])*sin(z[6])*z[34]*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[6])*sin(z[5])*z[36]*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[5])*sin(z[6])*z[35]*(L2*M3+LC2*M2)   0.0   2*cos(z[5])*sin(z[5])*z[35]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*z[37]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))+L1*z[38]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-L1*cos(z[7])*cos(z[8])*sin(z[9])*z[39]*(L2*M3+LC2*M2)   L1*sin(z[7])*sin(z[8])*sin(z[9])*z[37]*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[9])*sin(z[8])*z[39]*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[8])*sin(z[9])*z[38]*(L2*M3+LC2*M2)
            0.0   0.0   0.0   0.0   0.0   0.0   L1*z[37]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))+L1*z[38]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-L1*cos(z[7])*cos(z[8])*sin(z[9])*z[39]*(L2*M3+LC2*M2)   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   L1*sin(z[7])*sin(z[8])*sin(z[9])*z[37]*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[9])*sin(z[8])*z[39]*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[8])*sin(z[9])*z[38]*(L2*M3+LC2*M2)   0.0   2*cos(z[8])*sin(z[8])*z[38]*(J2+L2^2*M3+LC2^2*M2)        
        ] # + 0 since Mp is zero for γ

        dM_dpinvM = [dM_dp[:,1:3]/m1   dM_dp[:,4:6]/m2   dM_dp[:,7:9]/m3]
        dH_dpinvM = [dH_dp[:,1:3]/m1   dH_dp[:,4:6]/m2   dH_dp[:,7:9]/m3]

        orange_term = dH_dpinvM - HinvM*dM_dpinvM
        d_HinvMHT_dp = orange_term*transpose(H) + HinvM*transpose(dH_dp)

        # dκp = [H*inv(M)*transpose(H)]\{orange_term*cgBu + H*inv(M)*dcgBu/dp) - [orange_term*transpose(H)+H*inv(M)*transpose(dH/dp)] }
        dz[49:54] = (HinvM*transpose(H))\(orange_term*cgBu + HinvM*([
            # cgBu_p_sansp
            z[41]*(2*L1*z[11]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[12]*(L2*M3+LC2*M2))-z[42]*(2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]*(L2*M3+LC2*M2))-z[32]*(L1*z[11]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))+L1*cos(z[1])*cos(z[2])*cos(z[3])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+γ*z[40]+z[31]*(g*sin(z[1])*(L1*(M2+M3)+LC1*M1)+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2]))+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)+2*L1*cos(z[2])*sin(z[1])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+z[33]*(L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]^2*(L2*M3+LC2*M2)+L1*cos(z[1])*sin(z[2])*sin(z[3])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*cos(z[2])*cos(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))
            z[33]*(g*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2))+γ*z[41]+z[32]*(sin(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2])))-2*cos(z[2])*sin(z[2])*z[42]*z[12]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[40]*z[10]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))-L1*z[31]*z[10]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))
            z[42]*(γ+sin(2*z[2])*z[11]*(J2+L2^2*M3+LC2^2*M2))+z[33]*(g*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[10]^2*(L2*M3+LC2*M2))+z[32]*(2*cos(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2))+sin(2*z[2])*z[41]*z[12]*(J2+L2^2*M3+LC2^2*M2)+L1*cos(z[1])*sin(z[2])*sin(z[3])*z[31]*z[10]^2*(L2*M3+LC2*M2)+2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[40]*z[10]*(L2*M3+LC2*M2)
            z[44]*(2*L1*z[14]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[15]*(L2*M3+LC2*M2))-z[45]*(2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]*(L2*M3+LC2*M2))-z[35]*(L1*z[14]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))+L1*cos(z[4])*cos(z[5])*cos(z[6])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+γ*z[43]+z[34]*(g*sin(z[4])*(L1*(M2+M3)+LC1*M1)+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5]))+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)+2*L1*cos(z[5])*sin(z[4])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+z[36]*(L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*sin(z[5])*sin(z[6])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*cos(z[5])*cos(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))
            z[36]*(g*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2))+γ*z[44]+z[35]*(sin(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5])))-2*cos(z[5])*sin(z[5])*z[45]*z[15]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[43]*z[13]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))-L1*z[34]*z[13]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))
            z[45]*(γ+sin(2*z[5])*z[14]*(J2+L2^2*M3+LC2^2*M2))+z[36]*(g*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[13]^2*(L2*M3+LC2*M2))+z[35]*(2*cos(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2))+sin(2*z[5])*z[44]*z[15]*(J2+L2^2*M3+LC2^2*M2)+L1*cos(z[4])*sin(z[5])*sin(z[6])*z[34]*z[13]^2*(L2*M3+LC2*M2)+2*L1*sin(z[4])*sin(z[5])*sin(z[6])*z[43]*z[13]*(L2*M3+LC2*M2)
            z[47]*(2*L1*z[17]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[18]*(L2*M3+LC2*M2))-z[48]*(2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]*(L2*M3+LC2*M2))-z[38]*(L1*z[17]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))+L1*cos(z[7])*cos(z[8])*cos(z[9])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+γ*z[46]+z[37]*(g*sin(z[7])*(L1*(M2+M3)+LC1*M1)+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8]))+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)+2*L1*cos(z[8])*sin(z[7])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+z[39]*(L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*sin(z[8])*sin(z[9])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*cos(z[8])*cos(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))
            z[39]*(g*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2))+γ*z[47]+z[38]*(sin(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8])))-2*cos(z[8])*sin(z[8])*z[48]*z[18]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[46]*z[16]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))-L1*z[37]*z[16]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))
            z[48]*(γ+sin(2*z[8])*z[17]*(J2+L2^2*M3+LC2^2*M2))+z[39]*(g*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[16]^2*(L2*M3+LC2*M2))+z[38]*(2*cos(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2))+sin(2*z[8])*z[47]*z[18]*(J2+L2^2*M3+LC2^2*M2)+L1*cos(z[7])*sin(z[8])*sin(z[9])*z[37]*z[16]^2*(L2*M3+LC2*M2)+2*L1*sin(z[7])*sin(z[8])*sin(z[9])*z[46]*z[16]*(L2*M3+LC2*M2)
        ] + [   #cgBu_p for p=γ
            z[10]
            z[11]
            z[12]
            z[13]
            z[14]
            z[15]
            z[16]
            z[17]
            z[18]               
        ]) - d_HinvMHT_dp*z[19:24])     # d_HinvMHT_dp = orange_term*transpose(H) + HinvM*transpose(dH_dp)

        dMvterm_dp = [  # Mvterm_p_sansp
            z[42]*(2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]*(L2*M3+LC2*M2))-z[31]*(g*sin(z[1])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[1])*dz[20]+L1*cos(z[1])*dz[23]+L1*sin(z[1])*dz[21]+L1*sin(z[1])*dz[24]+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2]))+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)+2*L1*cos(z[2])*sin(z[1])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))-z[41]*(2*L1*z[11]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[12]*(L2*M3+LC2*M2))+z[32]*(L1*z[11]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))+L1*cos(z[1])*cos(z[2])*cos(z[3])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))-γ*z[40]-z[33]*(L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]^2*(L2*M3+LC2*M2)+L1*cos(z[1])*sin(z[2])*sin(z[3])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*cos(z[2])*cos(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+L1*cos(z[1])*dz[51]+L1*cos(z[1])*dz[54]-L1*sin(z[1])*dz[50]-L1*sin(z[1])*dz[53]
            L2*cos(z[2])*cos(z[3])*dz[51]-γ*z[41]-z[32]*(sin(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)+L2*cos(z[2])*dz[20]+L2*cos(z[2])*dz[23]+L2*cos(z[3])*sin(z[2])*dz[21]+L2*cos(z[3])*sin(z[2])*dz[24]+L2*sin(z[2])*sin(z[3])*dz[19]+L2*sin(z[2])*sin(z[3])*dz[22]+g*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2])))-L2*sin(z[2])*dz[50]-L2*sin(z[2])*dz[53]-z[33]*(L2*cos(z[2])*sin(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[22]-L2*cos(z[2])*cos(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[24]+g*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2))+L2*cos(z[2])*cos(z[3])*dz[54]+L2*cos(z[2])*sin(z[3])*dz[49]+L2*cos(z[2])*sin(z[3])*dz[52]+2*cos(z[2])*sin(z[2])*z[42]*z[12]*(J2+L2^2*M3+LC2^2*M2)-2*L1*z[40]*z[10]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))+L1*z[31]*z[10]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))
            L2*cos(z[3])*sin(z[2])*dz[49]-z[33]*(L2*cos(z[3])*sin(z[2])*dz[21]+L2*cos(z[3])*sin(z[2])*dz[24]+L2*sin(z[2])*sin(z[3])*dz[19]+L2*sin(z[2])*sin(z[3])*dz[22]+g*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[10]^2*(L2*M3+LC2*M2))-z[42]*(γ+sin(2*z[2])*z[11]*(J2+L2^2*M3+LC2^2*M2))-z[32]*(L2*cos(z[2])*sin(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[22]-L2*cos(z[2])*cos(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[24]+2*cos(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2))+L2*cos(z[3])*sin(z[2])*dz[52]-L2*sin(z[2])*sin(z[3])*dz[51]-L2*sin(z[2])*sin(z[3])*dz[54]-sin(2*z[2])*z[41]*z[12]*(J2+L2^2*M3+LC2^2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*z[31]*z[10]^2*(L2*M3+LC2*M2)-2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[40]*z[10]*(L2*M3+LC2*M2)
            z[45]*(2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]*(L2*M3+LC2*M2))-z[44]*(2*L1*z[14]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[15]*(L2*M3+LC2*M2))+z[35]*(L1*z[14]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))+L1*cos(z[4])*cos(z[5])*cos(z[6])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))-z[34]*(g*sin(z[4])*(L1*(M2+M3)+LC1*M1)+(L1*cos(z[4])*dz[20])*0.5-L1*sin(z[4])*dz[21]+(sqrt(3)*L1*cos(z[4])*dz[19])*0.5+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5]))+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)+2*L1*cos(z[5])*sin(z[4])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))-γ*z[43]-z[36]*(L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*sin(z[5])*sin(z[6])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*cos(z[5])*cos(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))-L1*cos(z[4])*dz[51]-(L1*sin(z[4])*dz[50])*0.5-(sqrt(3)*L1*sin(z[4])*dz[49])*0.5
            dz[49]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)-dz[50]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-z[36]*(g*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)-L2*cos(z[5])*sin(z[6])*dz[21]-(L2*cos(z[5])*cos(z[6])*dz[19])*0.5+(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[20])*0.5+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2))-z[35]*(dz[19]*((sqrt(3)*L2*cos(z[5]))*0.5+(L2*sin(z[5])*sin(z[6]))*0.5)+dz[20]*((L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5)-cos(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)+sin(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[6])*sin(z[5])*dz[21]+g*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5])))-γ*z[44]-L2*cos(z[5])*cos(z[6])*dz[51]+2*cos(z[5])*sin(z[5])*z[45]*z[15]*(J2+L2^2*M3+LC2^2*M2)-2*L1*z[43]*z[13]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))+L1*z[34]*z[13]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))
            (L2*cos(z[6])*sin(z[5])*dz[49])*0.5-z[35]*(2*cos(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[5])*sin(z[6])*dz[21]-(L2*cos(z[5])*cos(z[6])*dz[19])*0.5+g*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[20])*0.5+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2))-z[36]*((L2*sin(z[5])*sin(z[6])*dz[19])*0.5-L2*cos(z[6])*sin(z[5])*dz[21]+g*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)-(sqrt(3)*L2*sin(z[5])*sin(z[6])*dz[20])*0.5+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[13]^2*(L2*M3+LC2*M2))-z[45]*(γ+sin(2*z[5])*z[14]*(J2+L2^2*M3+LC2^2*M2))+L2*sin(z[5])*sin(z[6])*dz[51]-sin(2*z[5])*z[44]*z[15]*(J2+L2^2*M3+LC2^2*M2)-(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[50])*0.5-L1*cos(z[4])*sin(z[5])*sin(z[6])*z[34]*z[13]^2*(L2*M3+LC2*M2)-2*L1*sin(z[4])*sin(z[5])*sin(z[6])*z[43]*z[13]*(L2*M3+LC2*M2)
            z[48]*(2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]*(L2*M3+LC2*M2))-z[47]*(2*L1*z[17]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[18]*(L2*M3+LC2*M2))+z[38]*(L1*z[17]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))+L1*cos(z[7])*cos(z[8])*cos(z[9])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))-z[37]*(g*sin(z[7])*(L1*(M2+M3)+LC1*M1)+(L1*cos(z[7])*dz[23])*0.5-L1*sin(z[7])*dz[24]-(sqrt(3)*L1*cos(z[7])*dz[22])*0.5+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8]))+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)+2*L1*cos(z[8])*sin(z[7])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))-γ*z[46]-z[39]*(L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*sin(z[8])*sin(z[9])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*cos(z[8])*cos(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))-L1*cos(z[7])*dz[54]-(L1*sin(z[7])*dz[53])*0.5+(sqrt(3)*L1*sin(z[7])*dz[52])*0.5
            dz[52]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)-dz[53]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)+z[39]*((L2*cos(z[8])*cos(z[9])*dz[22])*0.5+L2*cos(z[8])*sin(z[9])*dz[24]-g*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[23])*0.5-L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2))-z[38]*(dz[23]*((L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5)-dz[22]*((sqrt(3)*L2*cos(z[8]))*0.5-(L2*sin(z[8])*sin(z[9]))*0.5)-cos(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)+sin(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[9])*sin(z[8])*dz[24]+g*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8])))-γ*z[47]-L2*cos(z[8])*cos(z[9])*dz[54]+2*cos(z[8])*sin(z[8])*z[48]*z[18]*(J2+L2^2*M3+LC2^2*M2)-2*L1*z[46]*z[16]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))+L1*z[37]*z[16]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))
            z[38]*((L2*cos(z[8])*cos(z[9])*dz[22])*0.5+L2*cos(z[8])*sin(z[9])*dz[24]-2*cos(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)-g*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[23])*0.5-L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2))-z[48]*(γ+sin(2*z[8])*z[17]*(J2+L2^2*M3+LC2^2*M2))-z[39]*((L2*sin(z[8])*sin(z[9])*dz[22])*0.5-L2*cos(z[9])*sin(z[8])*dz[24]+g*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+(sqrt(3)*L2*sin(z[8])*sin(z[9])*dz[23])*0.5+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[16]^2*(L2*M3+LC2*M2))+(L2*cos(z[9])*sin(z[8])*dz[52])*0.5+L2*sin(z[8])*sin(z[9])*dz[54]-sin(2*z[8])*z[47]*z[18]*(J2+L2^2*M3+LC2^2*M2)+(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[53])*0.5-L1*cos(z[7])*sin(z[8])*sin(z[9])*z[37]*z[16]^2*(L2*M3+LC2*M2)-2*L1*sin(z[7])*sin(z[8])*sin(z[9])*z[46]*z[16]*(L2*M3+LC2*M2)
        ] + [   # Mvterm_p for p=γ
            -z[10]
            -z[11]
            -z[12]
            -z[13]
            -z[14]
            -z[15]
            -z[16]
            -z[17]
            -z[18]            
        ]

        # dvp = inv(M)*(dMvterm_dp - rp(dM/dp, dv)), rp is my special product
        dz[40:48] = [m1\(dMvterm_dp[1:3,:] - dM_dp[1:3,:]*dz[10:18])
            m2\(dMvterm_dp[4:6,:] - dM_dp[4:6,:]*dz[10:18])
            m3\(dMvterm_dp[7:9,:] - dM_dp[7:9,:]*dz[10:18])]

        return z, dz
    end
end

function get_delta_initial_M1sens_comp(θ, u0, w0)
    let L0 = θ[1], L1 = θ[2], L2 = θ[3], L3 = θ[4], LC1 = θ[5], LC2 = θ[6], M1 = θ[7], M2 = θ[8], M3 = θ[9], J1 = θ[10], J2 = θ[11], g = 0.0, γ = θ[13]

        z  = zeros(60)
        dz = zeros(60)
        z[1:3] = [π/6; acos(-0.5*sqrt(3)*L1/L2 + (L3-L0)/L2); 0.0]
        z[4:6] = [π/6; acos(-0.5*sqrt(3)*L1/L2 + (L3-L0)/L2); 0.0]
        z[7:9] = [π/6; acos(-0.5*sqrt(3)*L1/L2 + (L3-L0)/L2); 0.0]

        # Sensitivity of initial value wrt to M1 is zero
        # z[32] = 0
        # z[35] = 0
        # z[38] = 0


        ################### INITIALIZING NOMINAL MODEL PART ########################

        H = [
            0.0   L2*cos(z[2])*sin(z[3])   L2*cos(z[3])*sin(z[2])   -(sqrt(3)*L1*sin(z[4]))*0.5   (L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5   (L2*cos(z[6])*sin(z[5]))*0.5   0.0   0.0   0.0
            -L1*sin(z[1])   -L2*sin(z[2])   0.0   -(L1*sin(z[4]))*0.5   -(L2*sin(z[5]))*0.5-(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5   -(sqrt(3)*L2*cos(z[6])*sin(z[5]))*0.5   0.0   0.0   0.0
            L1*cos(z[1])   L2*cos(z[2])*cos(z[3])   -L2*sin(z[2])*sin(z[3])   -L1*cos(z[4])   -L2*cos(z[5])*cos(z[6])   L2*sin(z[5])*sin(z[6])   0.0   0.0   0.0
            0.0   L2*cos(z[2])*sin(z[3])   L2*cos(z[3])*sin(z[2])   0.0   0.0   0.0   (sqrt(3)*L1*sin(z[7]))*0.5   (L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5   (L2*cos(z[9])*sin(z[8]))*0.5
            -L1*sin(z[1])   -L2*sin(z[2])   0.0   0.0   0.0   0.0   -(L1*sin(z[7]))*0.5   (sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5-(L2*sin(z[8]))*0.5   (sqrt(3)*L2*cos(z[9])*sin(z[8]))*0.5
            L1*cos(z[1])   L2*cos(z[2])*cos(z[3])   -L2*sin(z[2])*sin(z[3])   0.0   0.0   0.0   -L1*cos(z[7])   -L2*cos(z[8])*cos(z[9])   L2*sin(z[8])*sin(z[9])
        ]

        m1 = [
            J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))   -L1*cos(z[1])*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)
            L1*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))   J2+L2^2*M3+LC2^2*M2   0.0
            -L1*cos(z[1])*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)   0.0   sin(z[2])^2*(J2+L2^2*M3+LC2^2*M2)
        ]
        m2 = [
            J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))   -L1*cos(z[4])*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)
            L1*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))   J2+L2^2*M3+LC2^2*M2   0.0
            -L1*cos(z[4])*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)   0.0   sin(z[5])^2*(J2+L2^2*M3+LC2^2*M2)
        ]
        m3 = [
            J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))   -L1*cos(z[7])*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)
            L1*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))   J2+L2^2*M3+LC2^2*M2   0.0
            -L1*cos(z[7])*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)   0.0   sin(z[8])^2*(J2+L2^2*M3+LC2^2*M2)
        ]

        HinvM = [
            H[:,1:3]/m1   H[:,4:6]/m2   H[:,7:9]/m3
        ]

        cgBu = [
            γ*z[10]-u0[1]+w0[1]^2-g*cos(z[1])*(L1*(M2+M3)+LC1*M1)+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2)
            γ*z[11]-cos(z[2])*sin(z[2])*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-g*cos(z[2])*cos(z[3])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))
            γ*z[12]+sin(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+g*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*sin(z[1])*sin(z[2])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)
            γ*z[13]-u0[2]+w0[2]^2-g*cos(z[4])*(L1*(M2+M3)+LC1*M1)+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2)
            γ*z[14]-cos(z[5])*sin(z[5])*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-g*cos(z[5])*cos(z[6])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))
            γ*z[15]+sin(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)+g*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)+L1*sin(z[4])*sin(z[5])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)
            γ*z[16]-u0[3]+w0[3]^2-g*cos(z[7])*(L1*(M2+M3)+LC1*M1)+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2)
            γ*z[17]-cos(z[8])*sin(z[8])*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-g*cos(z[8])*cos(z[9])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))
            γ*z[18]+sin(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)+g*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)+L1*sin(z[7])*sin(z[8])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)
        ]

        # dκ = (HinvM*transpose(H))\(HinvM*cgBu)
        dz[19:24] = (HinvM*transpose(H))\(HinvM*cgBu)

        # dv = inv(M)*Mvterm
        dz[10:18] = [
            m1\[
                u0[1]+w0[1]^2-γ*z[10]+g*cos(z[1])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[1])*dz[21]+L1*cos(z[1])*dz[24]-L1*sin(z[1])*dz[20]-L1*sin(z[1])*dz[23]-L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))+L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2)
                L2*cos(z[2])*cos(z[3])*dz[21]-L2*sin(z[2])*dz[20]-L2*sin(z[2])*dz[23]-γ*z[11]+L2*cos(z[2])*cos(z[3])*dz[24]+L2*cos(z[2])*sin(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[22]+cos(z[2])*sin(z[2])*z[12]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[2])*cos(z[3])*(L2*M3+LC2*M2)-L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))
                L2*cos(z[3])*sin(z[2])*dz[19]-γ*z[12]+L2*cos(z[3])*sin(z[2])*dz[22]-L2*sin(z[2])*sin(z[3])*dz[21]-L2*sin(z[2])*sin(z[3])*dz[24]-sin(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)-g*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)-L1*sin(z[1])*sin(z[2])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)
            ]
            m2\[
                u0[2]+w0[2]^2-γ*z[13]+g*cos(z[4])*(L1*(M2+M3)+LC1*M1)-L1*cos(z[4])*dz[21]-(L1*sin(z[4])*dz[20])*0.5-(sqrt(3)*L1*sin(z[4])*dz[19])*0.5-L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))+L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2)
                dz[19]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)-dz[20]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-γ*z[14]-L2*cos(z[5])*cos(z[6])*dz[21]+cos(z[5])*sin(z[5])*z[15]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[5])*cos(z[6])*(L2*M3+LC2*M2)-L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))
                (L2*cos(z[6])*sin(z[5])*dz[19])*0.5-γ*z[15]+L2*sin(z[5])*sin(z[6])*dz[21]-sin(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)-g*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)-(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[20])*0.5-L1*sin(z[4])*sin(z[5])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)
            ]
            m3\[
                u0[3]+w0[3]^2-γ*z[16]+g*cos(z[7])*(L1*(M2+M3)+LC1*M1)-L1*cos(z[7])*dz[24]-(L1*sin(z[7])*dz[23])*0.5+(sqrt(3)*L1*sin(z[7])*dz[22])*0.5-L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))+L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2)
                dz[22]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)-dz[23]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)-γ*z[17]-L2*cos(z[8])*cos(z[9])*dz[24]+cos(z[8])*sin(z[8])*z[18]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[8])*cos(z[9])*(L2*M3+LC2*M2)-L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))
                (L2*cos(z[9])*sin(z[8])*dz[22])*0.5-γ*z[18]+L2*sin(z[8])*sin(z[9])*dz[24]-sin(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)-g*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[23])*0.5-L1*sin(z[7])*sin(z[8])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)
            ]
        ]

        ############################# INITIALIZING SENSITIVITY PART ####################################

        # Block matrix product behaviour reminder
        # [a11 a12 a13] [inv(m1) 0 0]   [a11*inv(m1) a12*inv(m2) a13*inv(m3)]
        # [a21 a22 a23]*[0 inv(m2) 0] = [a21*inv(m1) a22*inv(m2) a23*inv(m3)] = [a_1*inv(m1) a_2*inv(m2) a_3*inv(m3)]
        # [a31 a32 a33] [0 0 inv(m3)]   [a31*inv(m1) a32*inv(m2) a33*inv(m3)]
        # and
        # [inv(m1) 0 0] [a11 a12 a13]   [inv(m1)*a11 inv(m2)*a12 inv(m3)*a13]   [inv(m1)*a1_]
        # [0 inv(m2) 0]*[a21 a22 a23] = [inv(m1)*a21 inv(m2)*a22 inv(m3)*a23] = [inv(m2)*a2_]
        # [0 0 inv(m3)] [a31 a32 a33]   [inv(m1)*a31 inv(m2)*a32 inv(m3)*a33]   [inv(m3)*a3_]

        dH_dp = [   # Hp_sansp
            0.0   L2*cos(z[2])*cos(z[3])*z[33]-L2*sin(z[2])*sin(z[3])*z[32]   L2*cos(z[2])*cos(z[3])*z[32]-L2*sin(z[2])*sin(z[3])*z[33]   -(sqrt(3)*L1*cos(z[4])*z[34])*0.5   (L2*cos(z[5])*cos(z[6])*z[36])*0.5-z[35]*((sqrt(3)*L2*cos(z[5]))*0.5+(L2*sin(z[5])*sin(z[6]))*0.5)   (L2*cos(z[5])*cos(z[6])*z[35])*0.5-(L2*sin(z[5])*sin(z[6])*z[36])*0.5   0.0   0.0   0.0
            -2*L1*cos(z[1])*z[31]   -L2*cos(z[2])*z[32]   0.0   -(L1*cos(z[4])*z[34])*0.5   -z[35]*((L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5)-(sqrt(3)*L2*cos(z[5])*cos(z[6])*z[36])*0.5   (sqrt(3)*L2*sin(z[5])*sin(z[6])*z[36])*0.5-(sqrt(3)*L2*cos(z[5])*cos(z[6])*z[35])*0.5   0.0   0.0   0.0
            -2*L1*sin(z[1])*z[31]   -L2*cos(z[3])*sin(z[2])*z[32]-L2*cos(z[2])*sin(z[3])*z[33]   -L2*cos(z[2])*sin(z[3])*z[32]-L2*cos(z[3])*sin(z[2])*z[33]   L1*sin(z[4])*z[34]   L2*cos(z[6])*sin(z[5])*z[35]+L2*cos(z[5])*sin(z[6])*z[36]   L2*cos(z[5])*sin(z[6])*z[35]+L2*cos(z[6])*sin(z[5])*z[36]   0.0   0.0   0.0
            0.0   L2*cos(z[2])*cos(z[3])*z[33]-L2*sin(z[2])*sin(z[3])*z[32]   L2*cos(z[2])*cos(z[3])*z[32]-L2*sin(z[2])*sin(z[3])*z[33]   0.0   0.0   0.0   (sqrt(3)*L1*cos(z[7])*z[37])*0.5   z[38]*((sqrt(3)*L2*cos(z[8]))*0.5-(L2*sin(z[8])*sin(z[9]))*0.5)+(L2*cos(z[8])*cos(z[9])*z[39])*0.5   (L2*cos(z[8])*cos(z[9])*z[38])*0.5-(L2*sin(z[8])*sin(z[9])*z[39])*0.5
            -2*L1*cos(z[1])*z[31]   -L2*cos(z[2])*z[32]   0.0   0.0   0.0   0.0   -(L1*cos(z[7])*z[37])*0.5   (sqrt(3)*L2*cos(z[8])*cos(z[9])*z[39])*0.5-z[38]*((L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5)   (sqrt(3)*L2*cos(z[8])*cos(z[9])*z[38])*0.5-(sqrt(3)*L2*sin(z[8])*sin(z[9])*z[39])*0.5
            -2*L1*sin(z[1])*z[31]   -L2*cos(z[3])*sin(z[2])*z[32]-L2*cos(z[2])*sin(z[3])*z[33]   -L2*cos(z[2])*sin(z[3])*z[32]-L2*cos(z[3])*sin(z[2])*z[33]   0.0   0.0   0.0   L1*sin(z[7])*z[37]   L2*cos(z[9])*sin(z[8])*z[38]+L2*cos(z[8])*sin(z[9])*z[39]   L2*cos(z[8])*sin(z[9])*z[38]+L2*cos(z[9])*sin(z[8])*z[39]        
        ] # Hp=0 for p=M1

        dM_dp = [   # Mp_sansp
            0.0   2*L1*z[31]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))+L1*z[32]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-L1*cos(z[1])*cos(z[2])*sin(z[3])*z[33]*(L2*M3+LC2*M2)   2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[31]*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[3])*sin(z[2])*z[33]*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[2])*sin(z[3])*z[32]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0
            2*L1*z[31]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))+L1*z[32]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-L1*cos(z[1])*cos(z[2])*sin(z[3])*z[33]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[31]*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[3])*sin(z[2])*z[33]*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[2])*sin(z[3])*z[32]*(L2*M3+LC2*M2)   0.0   2*cos(z[2])*sin(z[2])*z[32]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   L1*z[34]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))+L1*z[35]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-L1*cos(z[4])*cos(z[5])*sin(z[6])*z[36]*(L2*M3+LC2*M2)   L1*sin(z[4])*sin(z[5])*sin(z[6])*z[34]*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[6])*sin(z[5])*z[36]*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[5])*sin(z[6])*z[35]*(L2*M3+LC2*M2)   0.0   0.0   0.0
            0.0   0.0   0.0   L1*z[34]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))+L1*z[35]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-L1*cos(z[4])*cos(z[5])*sin(z[6])*z[36]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   L1*sin(z[4])*sin(z[5])*sin(z[6])*z[34]*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[6])*sin(z[5])*z[36]*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[5])*sin(z[6])*z[35]*(L2*M3+LC2*M2)   0.0   2*cos(z[5])*sin(z[5])*z[35]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*z[37]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))+L1*z[38]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-L1*cos(z[7])*cos(z[8])*sin(z[9])*z[39]*(L2*M3+LC2*M2)   L1*sin(z[7])*sin(z[8])*sin(z[9])*z[37]*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[9])*sin(z[8])*z[39]*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[8])*sin(z[9])*z[38]*(L2*M3+LC2*M2)
            0.0   0.0   0.0   0.0   0.0   0.0   L1*z[37]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))+L1*z[38]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-L1*cos(z[7])*cos(z[8])*sin(z[9])*z[39]*(L2*M3+LC2*M2)   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   L1*sin(z[7])*sin(z[8])*sin(z[9])*z[37]*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[9])*sin(z[8])*z[39]*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[8])*sin(z[9])*z[38]*(L2*M3+LC2*M2)   0.0   2*cos(z[8])*sin(z[8])*z[38]*(J2+L2^2*M3+LC2^2*M2)        
        ] + [   # Mp for p=M1
            LC1^2	0	0	0	    0	0	0	    0	0.0
            0	    0	0	0	    0	0	0	    0	0.0
            0	    0	0	0	    0	0	0	    0	0.0
            0	    0	0	LC1^2	0	0	0	    0	0.0
            0	    0	0	0	    0	0	0	    0	0.0
            0	    0	0	0	    0	0	0	    0	0.0
            0	    0	0	0	    0	0	LC1^2	0	0.0
            0	    0	0	0	    0	0	0	    0	0.0
            0	    0	0	0	    0	0	0	    0	0.0
        ]

        dM_dpinvM = [dM_dp[:,1:3]/m1   dM_dp[:,4:6]/m2   dM_dp[:,7:9]/m3]
        dH_dpinvM = [dH_dp[:,1:3]/m1   dH_dp[:,4:6]/m2   dH_dp[:,7:9]/m3]

        orange_term = dH_dpinvM - HinvM*dM_dpinvM
        d_HinvMHT_dp = orange_term*transpose(H) + HinvM*transpose(dH_dp)

        # dκp = [H*inv(M)*transpose(H)]\{orange_term*cgBu + H*inv(M)*dcgBu/dp) - [orange_term*transpose(H)+H*inv(M)*transpose(dH/dp)] }
        dz[49:54] = (HinvM*transpose(H))\(orange_term*cgBu + HinvM*([
            # cgBu_p_sansp
            z[41]*(2*L1*z[11]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[12]*(L2*M3+LC2*M2))-z[42]*(2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]*(L2*M3+LC2*M2))-z[32]*(L1*z[11]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))+L1*cos(z[1])*cos(z[2])*cos(z[3])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+γ*z[40]+z[31]*(g*sin(z[1])*(L1*(M2+M3)+LC1*M1)+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2]))+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)+2*L1*cos(z[2])*sin(z[1])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+z[33]*(L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]^2*(L2*M3+LC2*M2)+L1*cos(z[1])*sin(z[2])*sin(z[3])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*cos(z[2])*cos(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))
            z[33]*(g*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2))+γ*z[41]+z[32]*(sin(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2])))-2*cos(z[2])*sin(z[2])*z[42]*z[12]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[40]*z[10]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))-L1*z[31]*z[10]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))
            z[42]*(γ+sin(2*z[2])*z[11]*(J2+L2^2*M3+LC2^2*M2))+z[33]*(g*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[10]^2*(L2*M3+LC2*M2))+z[32]*(2*cos(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2))+sin(2*z[2])*z[41]*z[12]*(J2+L2^2*M3+LC2^2*M2)+L1*cos(z[1])*sin(z[2])*sin(z[3])*z[31]*z[10]^2*(L2*M3+LC2*M2)+2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[40]*z[10]*(L2*M3+LC2*M2)
            z[44]*(2*L1*z[14]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[15]*(L2*M3+LC2*M2))-z[45]*(2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]*(L2*M3+LC2*M2))-z[35]*(L1*z[14]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))+L1*cos(z[4])*cos(z[5])*cos(z[6])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+γ*z[43]+z[34]*(g*sin(z[4])*(L1*(M2+M3)+LC1*M1)+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5]))+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)+2*L1*cos(z[5])*sin(z[4])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+z[36]*(L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*sin(z[5])*sin(z[6])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*cos(z[5])*cos(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))
            z[36]*(g*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2))+γ*z[44]+z[35]*(sin(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5])))-2*cos(z[5])*sin(z[5])*z[45]*z[15]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[43]*z[13]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))-L1*z[34]*z[13]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))
            z[45]*(γ+sin(2*z[5])*z[14]*(J2+L2^2*M3+LC2^2*M2))+z[36]*(g*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[13]^2*(L2*M3+LC2*M2))+z[35]*(2*cos(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2))+sin(2*z[5])*z[44]*z[15]*(J2+L2^2*M3+LC2^2*M2)+L1*cos(z[4])*sin(z[5])*sin(z[6])*z[34]*z[13]^2*(L2*M3+LC2*M2)+2*L1*sin(z[4])*sin(z[5])*sin(z[6])*z[43]*z[13]*(L2*M3+LC2*M2)
            z[47]*(2*L1*z[17]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[18]*(L2*M3+LC2*M2))-z[48]*(2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]*(L2*M3+LC2*M2))-z[38]*(L1*z[17]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))+L1*cos(z[7])*cos(z[8])*cos(z[9])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+γ*z[46]+z[37]*(g*sin(z[7])*(L1*(M2+M3)+LC1*M1)+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8]))+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)+2*L1*cos(z[8])*sin(z[7])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+z[39]*(L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*sin(z[8])*sin(z[9])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*cos(z[8])*cos(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))
            z[39]*(g*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2))+γ*z[47]+z[38]*(sin(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8])))-2*cos(z[8])*sin(z[8])*z[48]*z[18]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[46]*z[16]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))-L1*z[37]*z[16]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))
            z[48]*(γ+sin(2*z[8])*z[17]*(J2+L2^2*M3+LC2^2*M2))+z[39]*(g*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[16]^2*(L2*M3+LC2*M2))+z[38]*(2*cos(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2))+sin(2*z[8])*z[47]*z[18]*(J2+L2^2*M3+LC2^2*M2)+L1*cos(z[7])*sin(z[8])*sin(z[9])*z[37]*z[16]^2*(L2*M3+LC2*M2)+2*L1*sin(z[7])*sin(z[8])*sin(z[9])*z[46]*z[16]*(L2*M3+LC2*M2)
        ] + [   #cgBu_p for p=M1
            -LC1*g*cos(z[1])
            0.0
            0.0
            -LC1*g*cos(z[4])
            0.0
            0.0
            -LC1*g*cos(z[7])
            0.0
            0.0
        ]) - d_HinvMHT_dp*z[19:24])     # d_HinvMHT_dp = orange_term*transpose(H) + HinvM*transpose(dH_dp)

        dMvterm_dp = [  # Mvterm_p_sansp
            z[42]*(2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]*(L2*M3+LC2*M2))-z[31]*(g*sin(z[1])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[1])*dz[20]+L1*cos(z[1])*dz[23]+L1*sin(z[1])*dz[21]+L1*sin(z[1])*dz[24]+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2]))+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)+2*L1*cos(z[2])*sin(z[1])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))-z[41]*(2*L1*z[11]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[12]*(L2*M3+LC2*M2))+z[32]*(L1*z[11]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))+L1*cos(z[1])*cos(z[2])*cos(z[3])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))-γ*z[40]-z[33]*(L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]^2*(L2*M3+LC2*M2)+L1*cos(z[1])*sin(z[2])*sin(z[3])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*cos(z[2])*cos(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+L1*cos(z[1])*dz[51]+L1*cos(z[1])*dz[54]-L1*sin(z[1])*dz[50]-L1*sin(z[1])*dz[53]
            L2*cos(z[2])*cos(z[3])*dz[51]-γ*z[41]-z[32]*(sin(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)+L2*cos(z[2])*dz[20]+L2*cos(z[2])*dz[23]+L2*cos(z[3])*sin(z[2])*dz[21]+L2*cos(z[3])*sin(z[2])*dz[24]+L2*sin(z[2])*sin(z[3])*dz[19]+L2*sin(z[2])*sin(z[3])*dz[22]+g*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2])))-L2*sin(z[2])*dz[50]-L2*sin(z[2])*dz[53]-z[33]*(L2*cos(z[2])*sin(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[22]-L2*cos(z[2])*cos(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[24]+g*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2))+L2*cos(z[2])*cos(z[3])*dz[54]+L2*cos(z[2])*sin(z[3])*dz[49]+L2*cos(z[2])*sin(z[3])*dz[52]+2*cos(z[2])*sin(z[2])*z[42]*z[12]*(J2+L2^2*M3+LC2^2*M2)-2*L1*z[40]*z[10]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))+L1*z[31]*z[10]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))
            L2*cos(z[3])*sin(z[2])*dz[49]-z[33]*(L2*cos(z[3])*sin(z[2])*dz[21]+L2*cos(z[3])*sin(z[2])*dz[24]+L2*sin(z[2])*sin(z[3])*dz[19]+L2*sin(z[2])*sin(z[3])*dz[22]+g*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[10]^2*(L2*M3+LC2*M2))-z[42]*(γ+sin(2*z[2])*z[11]*(J2+L2^2*M3+LC2^2*M2))-z[32]*(L2*cos(z[2])*sin(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[22]-L2*cos(z[2])*cos(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[24]+2*cos(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2))+L2*cos(z[3])*sin(z[2])*dz[52]-L2*sin(z[2])*sin(z[3])*dz[51]-L2*sin(z[2])*sin(z[3])*dz[54]-sin(2*z[2])*z[41]*z[12]*(J2+L2^2*M3+LC2^2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*z[31]*z[10]^2*(L2*M3+LC2*M2)-2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[40]*z[10]*(L2*M3+LC2*M2)
            z[45]*(2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]*(L2*M3+LC2*M2))-z[44]*(2*L1*z[14]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[15]*(L2*M3+LC2*M2))+z[35]*(L1*z[14]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))+L1*cos(z[4])*cos(z[5])*cos(z[6])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))-z[34]*(g*sin(z[4])*(L1*(M2+M3)+LC1*M1)+(L1*cos(z[4])*dz[20])*0.5-L1*sin(z[4])*dz[21]+(sqrt(3)*L1*cos(z[4])*dz[19])*0.5+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5]))+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)+2*L1*cos(z[5])*sin(z[4])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))-γ*z[43]-z[36]*(L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*sin(z[5])*sin(z[6])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*cos(z[5])*cos(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))-L1*cos(z[4])*dz[51]-(L1*sin(z[4])*dz[50])*0.5-(sqrt(3)*L1*sin(z[4])*dz[49])*0.5
            dz[49]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)-dz[50]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-z[36]*(g*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)-L2*cos(z[5])*sin(z[6])*dz[21]-(L2*cos(z[5])*cos(z[6])*dz[19])*0.5+(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[20])*0.5+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2))-z[35]*(dz[19]*((sqrt(3)*L2*cos(z[5]))*0.5+(L2*sin(z[5])*sin(z[6]))*0.5)+dz[20]*((L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5)-cos(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)+sin(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[6])*sin(z[5])*dz[21]+g*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5])))-γ*z[44]-L2*cos(z[5])*cos(z[6])*dz[51]+2*cos(z[5])*sin(z[5])*z[45]*z[15]*(J2+L2^2*M3+LC2^2*M2)-2*L1*z[43]*z[13]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))+L1*z[34]*z[13]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))
            (L2*cos(z[6])*sin(z[5])*dz[49])*0.5-z[35]*(2*cos(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[5])*sin(z[6])*dz[21]-(L2*cos(z[5])*cos(z[6])*dz[19])*0.5+g*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[20])*0.5+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2))-z[36]*((L2*sin(z[5])*sin(z[6])*dz[19])*0.5-L2*cos(z[6])*sin(z[5])*dz[21]+g*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)-(sqrt(3)*L2*sin(z[5])*sin(z[6])*dz[20])*0.5+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[13]^2*(L2*M3+LC2*M2))-z[45]*(γ+sin(2*z[5])*z[14]*(J2+L2^2*M3+LC2^2*M2))+L2*sin(z[5])*sin(z[6])*dz[51]-sin(2*z[5])*z[44]*z[15]*(J2+L2^2*M3+LC2^2*M2)-(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[50])*0.5-L1*cos(z[4])*sin(z[5])*sin(z[6])*z[34]*z[13]^2*(L2*M3+LC2*M2)-2*L1*sin(z[4])*sin(z[5])*sin(z[6])*z[43]*z[13]*(L2*M3+LC2*M2)
            z[48]*(2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]*(L2*M3+LC2*M2))-z[47]*(2*L1*z[17]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[18]*(L2*M3+LC2*M2))+z[38]*(L1*z[17]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))+L1*cos(z[7])*cos(z[8])*cos(z[9])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))-z[37]*(g*sin(z[7])*(L1*(M2+M3)+LC1*M1)+(L1*cos(z[7])*dz[23])*0.5-L1*sin(z[7])*dz[24]-(sqrt(3)*L1*cos(z[7])*dz[22])*0.5+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8]))+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)+2*L1*cos(z[8])*sin(z[7])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))-γ*z[46]-z[39]*(L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*sin(z[8])*sin(z[9])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*cos(z[8])*cos(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))-L1*cos(z[7])*dz[54]-(L1*sin(z[7])*dz[53])*0.5+(sqrt(3)*L1*sin(z[7])*dz[52])*0.5
            dz[52]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)-dz[53]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)+z[39]*((L2*cos(z[8])*cos(z[9])*dz[22])*0.5+L2*cos(z[8])*sin(z[9])*dz[24]-g*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[23])*0.5-L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2))-z[38]*(dz[23]*((L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5)-dz[22]*((sqrt(3)*L2*cos(z[8]))*0.5-(L2*sin(z[8])*sin(z[9]))*0.5)-cos(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)+sin(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[9])*sin(z[8])*dz[24]+g*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8])))-γ*z[47]-L2*cos(z[8])*cos(z[9])*dz[54]+2*cos(z[8])*sin(z[8])*z[48]*z[18]*(J2+L2^2*M3+LC2^2*M2)-2*L1*z[46]*z[16]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))+L1*z[37]*z[16]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))
            z[38]*((L2*cos(z[8])*cos(z[9])*dz[22])*0.5+L2*cos(z[8])*sin(z[9])*dz[24]-2*cos(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)-g*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[23])*0.5-L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2))-z[48]*(γ+sin(2*z[8])*z[17]*(J2+L2^2*M3+LC2^2*M2))-z[39]*((L2*sin(z[8])*sin(z[9])*dz[22])*0.5-L2*cos(z[9])*sin(z[8])*dz[24]+g*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+(sqrt(3)*L2*sin(z[8])*sin(z[9])*dz[23])*0.5+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[16]^2*(L2*M3+LC2*M2))+(L2*cos(z[9])*sin(z[8])*dz[52])*0.5+L2*sin(z[8])*sin(z[9])*dz[54]-sin(2*z[8])*z[47]*z[18]*(J2+L2^2*M3+LC2^2*M2)+(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[53])*0.5-L1*cos(z[7])*sin(z[8])*sin(z[9])*z[37]*z[16]^2*(L2*M3+LC2*M2)-2*L1*sin(z[7])*sin(z[8])*sin(z[9])*z[46]*z[16]*(L2*M3+LC2*M2)
        ] + [   # Mvterm_p for p=M1
            LC1*g*cos(z[1])
            0.0
            0.0
            LC1*g*cos(z[4])
            0.0
            0.0
            LC1*g*cos(z[7])
            0.0
            0.0
        ]

        # dvp = inv(M)*(dMvterm_dp - rp(dM/dp, dv)), rp is my special product
        dz[40:48] = [m1\(dMvterm_dp[1:3,:] - dM_dp[1:3,:]*dz[10:18])
            m2\(dMvterm_dp[4:6,:] - dM_dp[4:6,:]*dz[10:18])
            m3\(dMvterm_dp[7:9,:] - dM_dp[7:9,:]*dz[10:18])]

        return z, dz
    end
end

function get_delta_initial_L1M1J1sens_comp(θ, u0, w0)
    let L0 = θ[1], L1 = θ[2], L2 = θ[3], L3 = θ[4], LC1 = θ[5], LC2 = θ[6], M1 = θ[7], M2 = θ[8], M3 = θ[9], J1 = θ[10], J2 = θ[11], g = 0.0, γ = θ[13]

        z  = zeros(60)
        dz = zeros(60)
        z[1:3] = [π/6; acos(-0.5*sqrt(3)*L1/L2 + (L3-L0)/L2); 0.0]
        z[4:6] = [π/6; acos(-0.5*sqrt(3)*L1/L2 + (L3-L0)/L2); 0.0]
        z[7:9] = [π/6; acos(-0.5*sqrt(3)*L1/L2 + (L3-L0)/L2); 0.0]

        # Sensitivity of initial value wrt to L1
        z[32] = sqrt(3)/(2*L2*(1-(L0-L3+(sqrt(3)*L1)*0.5)^2/L2^2)^(0.5))
        z[35] = sqrt(3)/(2*L2*(1-(L0-L3+(sqrt(3)*L1)*0.5)^2/L2^2)^(0.5))
        z[38] = sqrt(3)/(2*L2*(1-(L0-L3+(sqrt(3)*L1)*0.5)^2/L2^2)^(0.5))
        # Sensitivity of initial value wrt to M1
        # z[62] = 0
        # z[65] = 0
        # z[68] = 0
        # Sensitivity of initial value wrt to J1
        # z[92] = 0
        # z[95] = 0
        # z[98] = 0


        ################### INITIALIZING NOMINAL MODEL PART ########################

        H = [
            0.0   L2*cos(z[2])*sin(z[3])   L2*cos(z[3])*sin(z[2])   -(sqrt(3)*L1*sin(z[4]))*0.5   (L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5   (L2*cos(z[6])*sin(z[5]))*0.5   0.0   0.0   0
            -L1*sin(z[1])   -L2*sin(z[2])   0.0   -(L1*sin(z[4]))*0.5   -(L2*sin(z[5]))*0.5-(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5   -(sqrt(3)*L2*cos(z[6])*sin(z[5]))*0.5   0.0   0.0   0
            L1*cos(z[1])   L2*cos(z[2])*cos(z[3])   -L2*sin(z[2])*sin(z[3])   -L1*cos(z[4])   -L2*cos(z[5])*cos(z[6])   L2*sin(z[5])*sin(z[6])   0.0   0.0   0
            0.0   L2*cos(z[2])*sin(z[3])   L2*cos(z[3])*sin(z[2])   0.0   0.0   0.0   (sqrt(3)*L1*sin(z[7]))*0.5   (L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5   (L2*cos(z[9])*sin(z[8]))*0.5
            -L1*sin(z[1])   -L2*sin(z[2])   0.0   0.0   0.0   0.0   -(L1*sin(z[7]))*0.5   (sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5-(L2*sin(z[8]))*0.5   (sqrt(3)*L2*cos(z[9])*sin(z[8]))*0.5
            L1*cos(z[1])   L2*cos(z[2])*cos(z[3])   -L2*sin(z[2])*sin(z[3])   0.0   0.0   0.0   -L1*cos(z[7])   -L2*cos(z[8])*cos(z[9])   L2*sin(z[8])*sin(z[9])
        ]

        m1 = [
            J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))   -L1*cos(z[1])*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)
            L1*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))   J2+L2^2*M3+LC2^2*M2   0
            -L1*cos(z[1])*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)   0.0   sin(z[2])^2*(J2+L2^2*M3+LC2^2*M2)
        ]
        m2 = [
            J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))   -L1*cos(z[4])*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)
            L1*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))   J2+L2^2*M3+LC2^2*M2   0
            -L1*cos(z[4])*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)   0.0   sin(z[5])^2*(J2+L2^2*M3+LC2^2*M2)
        ]
        m3 = [
            J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))   -L1*cos(z[7])*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)
            L1*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))   J2+L2^2*M3+LC2^2*M2   0
            -L1*cos(z[7])*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)   0.0   sin(z[8])^2*(J2+L2^2*M3+LC2^2*M2)
        ]

        HinvM = [
            H[:,1:3]/m1   H[:,4:6]/m2   H[:,7:9]/m3
        ]

        cgBu = [
            γ*z[10]-u0[1]+w0[1]^2-g*cos(z[1])*(L1*(M2+M3)+LC1*M1)+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2)
            γ*z[11]-cos(z[2])*sin(z[2])*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-g*cos(z[2])*cos(z[3])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))
            γ*z[12]+sin(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+g*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*sin(z[1])*sin(z[2])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)
            γ*z[13]-u0[2]+w0[2]^2-g*cos(z[4])*(L1*(M2+M3)+LC1*M1)+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2)
            γ*z[14]-cos(z[5])*sin(z[5])*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-g*cos(z[5])*cos(z[6])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))
            γ*z[15]+sin(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)+g*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)+L1*sin(z[4])*sin(z[5])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)
            γ*z[16]-u0[3]+w0[3]^2-g*cos(z[7])*(L1*(M2+M3)+LC1*M1)+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2)
            γ*z[17]-cos(z[8])*sin(z[8])*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-g*cos(z[8])*cos(z[9])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))
            γ*z[18]+sin(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)+g*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)+L1*sin(z[7])*sin(z[8])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)
        ]

        # dκ = (HinvM*transpose(H))\(HinvM*cgBu)
        dz[19:24] = (HinvM*transpose(H))\(HinvM*cgBu)

        # dv = inv(M)*Mvterm
        dz[10:18] = [
            m1\[
                u0[1]+w0[1]^2-γ*z[10]+g*cos(z[1])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[1])*dz[21]+L1*cos(z[1])*dz[24]-L1*sin(z[1])*dz[20]-L1*sin(z[1])*dz[23]-L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))+L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2)
                L2*cos(z[2])*cos(z[3])*dz[21]-L2*sin(z[2])*dz[20]-L2*sin(z[2])*dz[23]-γ*z[11]+L2*cos(z[2])*cos(z[3])*dz[24]+L2*cos(z[2])*sin(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[22]+cos(z[2])*sin(z[2])*z[12]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[2])*cos(z[3])*(L2*M3+LC2*M2)-L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))
                L2*cos(z[3])*sin(z[2])*dz[19]-γ*z[12]+L2*cos(z[3])*sin(z[2])*dz[22]-L2*sin(z[2])*sin(z[3])*dz[21]-L2*sin(z[2])*sin(z[3])*dz[24]-sin(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)-g*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)-L1*sin(z[1])*sin(z[2])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)
            ]
            m2\[
                u0[2]+w0[2]^2-γ*z[13]+g*cos(z[4])*(L1*(M2+M3)+LC1*M1)-L1*cos(z[4])*dz[21]-(L1*sin(z[4])*dz[20])*0.5-(sqrt(3)*L1*sin(z[4])*dz[19])*0.5-L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))+L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2)
                dz[19]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)-dz[20]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-γ*z[14]-L2*cos(z[5])*cos(z[6])*dz[21]+cos(z[5])*sin(z[5])*z[15]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[5])*cos(z[6])*(L2*M3+LC2*M2)-L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))
                (L2*cos(z[6])*sin(z[5])*dz[19])*0.5-γ*z[15]+L2*sin(z[5])*sin(z[6])*dz[21]-sin(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)-g*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)-(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[20])*0.5-L1*sin(z[4])*sin(z[5])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)
            ]
            m3\[
                u0[3]+w0[3]^2-γ*z[16]+g*cos(z[7])*(L1*(M2+M3)+LC1*M1)-L1*cos(z[7])*dz[24]-(L1*sin(z[7])*dz[23])*0.5+(sqrt(3)*L1*sin(z[7])*dz[22])*0.5-L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))+L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2)
                dz[22]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)-dz[23]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)-γ*z[17]-L2*cos(z[8])*cos(z[9])*dz[24]+cos(z[8])*sin(z[8])*z[18]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[8])*cos(z[9])*(L2*M3+LC2*M2)-L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))
                (L2*cos(z[9])*sin(z[8])*dz[22])*0.5-γ*z[18]+L2*sin(z[8])*sin(z[9])*dz[24]-sin(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)-g*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[23])*0.5-L1*sin(z[7])*sin(z[8])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)
            ]
        ]

        ############################# INITIALIZING SENSITIVITY PART FOR L1 ####################################

        # Block matrix product behaviour reminder
        # [a11 a12 a13] [inv(m1) 0 0]   [a11*inv(m1) a12*inv(m2) a13*inv(m3)]
        # [a21 a22 a23]*[0 inv(m2) 0] = [a21*inv(m1) a22*inv(m2) a23*inv(m3)] = [a_1*inv(m1) a_2*inv(m2) a_3*inv(m3)]
        # [a31 a32 a33] [0 0 inv(m3)]   [a31*inv(m1) a32*inv(m2) a33*inv(m3)]
        # and
        # [inv(m1) 0 0] [a11 a12 a13]   [inv(m1)*a11 inv(m2)*a12 inv(m3)*a13]   [inv(m1)*a1_]
        # [0 inv(m2) 0]*[a21 a22 a23] = [inv(m1)*a21 inv(m2)*a22 inv(m3)*a23] = [inv(m2)*a2_]
        # [0 0 inv(m3)] [a31 a32 a33]   [inv(m1)*a31 inv(m2)*a32 inv(m3)*a33]   [inv(m3)*a3_]

        dH_dL1 = [   # Hp_sansp
            0.0   L2*cos(z[2])*cos(z[3])*z[33]-L2*sin(z[2])*sin(z[3])*z[32]   L2*cos(z[2])*cos(z[3])*z[32]-L2*sin(z[2])*sin(z[3])*z[33]   -(sqrt(3)*L1*cos(z[4])*z[34])*0.5   (L2*cos(z[5])*cos(z[6])*z[36])*0.5-z[35]*((sqrt(3)*L2*cos(z[5]))*0.5+(L2*sin(z[5])*sin(z[6]))*0.5)   (L2*cos(z[5])*cos(z[6])*z[35])*0.5-(L2*sin(z[5])*sin(z[6])*z[36])*0.5   0.0   0.0   0.0
            -2*L1*cos(z[1])*z[31]   -L2*cos(z[2])*z[32]   0.0   -(L1*cos(z[4])*z[34])*0.5   -z[35]*((L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5)-(sqrt(3)*L2*cos(z[5])*cos(z[6])*z[36])*0.5   (sqrt(3)*L2*sin(z[5])*sin(z[6])*z[36])*0.5-(sqrt(3)*L2*cos(z[5])*cos(z[6])*z[35])*0.5   0.0   0.0   0.0
            -2*L1*sin(z[1])*z[31]   -L2*cos(z[3])*sin(z[2])*z[32]-L2*cos(z[2])*sin(z[3])*z[33]   -L2*cos(z[2])*sin(z[3])*z[32]-L2*cos(z[3])*sin(z[2])*z[33]   L1*sin(z[4])*z[34]   L2*cos(z[6])*sin(z[5])*z[35]+L2*cos(z[5])*sin(z[6])*z[36]   L2*cos(z[5])*sin(z[6])*z[35]+L2*cos(z[6])*sin(z[5])*z[36]   0.0   0.0   0.0
            0.0   L2*cos(z[2])*cos(z[3])*z[33]-L2*sin(z[2])*sin(z[3])*z[32]   L2*cos(z[2])*cos(z[3])*z[32]-L2*sin(z[2])*sin(z[3])*z[33]   0.0   0.0   0.0   (sqrt(3)*L1*cos(z[7])*z[37])*0.5   z[38]*((sqrt(3)*L2*cos(z[8]))*0.5-(L2*sin(z[8])*sin(z[9]))*0.5)+(L2*cos(z[8])*cos(z[9])*z[39])*0.5   (L2*cos(z[8])*cos(z[9])*z[38])*0.5-(L2*sin(z[8])*sin(z[9])*z[39])*0.5
            -2*L1*cos(z[1])*z[31]   -L2*cos(z[2])*z[32]   0.0   0.0   0.0   0.0   -(L1*cos(z[7])*z[37])*0.5   (sqrt(3)*L2*cos(z[8])*cos(z[9])*z[39])*0.5-z[38]*((L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5)   (sqrt(3)*L2*cos(z[8])*cos(z[9])*z[38])*0.5-(sqrt(3)*L2*sin(z[8])*sin(z[9])*z[39])*0.5
            -2*L1*sin(z[1])*z[31]   -L2*cos(z[3])*sin(z[2])*z[32]-L2*cos(z[2])*sin(z[3])*z[33]   -L2*cos(z[2])*sin(z[3])*z[32]-L2*cos(z[3])*sin(z[2])*z[33]   0.0   0.0   0.0   L1*sin(z[7])*z[37]   L2*cos(z[9])*sin(z[8])*z[38]+L2*cos(z[8])*sin(z[9])*z[39]   L2*cos(z[8])*sin(z[9])*z[38]+L2*cos(z[9])*sin(z[8])*z[39]        
        ] + [   # Hp for p=L1
            0	0	0	-(sqrt(3)*sin(z[4]))*0.5	0	0	0	0	0
            -sin(z[1])	0	0	-sin(z[4])*0.5	0	0	0	0	0
            cos(z[1])	0	0	-cos(z[4])	0	0	0	0	0
            0	0	0	0	0	0	(sqrt(3)*sin(z[7]))*0.5	0	0
            -sin(z[1])	0	0	0	0	0	-sin(z[7])*0.5	0	0
            cos(z[1])	0	0	0	0	0	-cos(z[7])	0	0
        ]

        dM_dL1 = [   # Mp_sansp
            0.0   2*L1*z[31]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))+L1*z[32]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-L1*cos(z[1])*cos(z[2])*sin(z[3])*z[33]*(L2*M3+LC2*M2)   2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[31]*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[3])*sin(z[2])*z[33]*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[2])*sin(z[3])*z[32]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0
            2*L1*z[31]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))+L1*z[32]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-L1*cos(z[1])*cos(z[2])*sin(z[3])*z[33]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[31]*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[3])*sin(z[2])*z[33]*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[2])*sin(z[3])*z[32]*(L2*M3+LC2*M2)   0.0   2*cos(z[2])*sin(z[2])*z[32]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   L1*z[34]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))+L1*z[35]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-L1*cos(z[4])*cos(z[5])*sin(z[6])*z[36]*(L2*M3+LC2*M2)   L1*sin(z[4])*sin(z[5])*sin(z[6])*z[34]*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[6])*sin(z[5])*z[36]*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[5])*sin(z[6])*z[35]*(L2*M3+LC2*M2)   0.0   0.0   0.0
            0.0   0.0   0.0   L1*z[34]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))+L1*z[35]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-L1*cos(z[4])*cos(z[5])*sin(z[6])*z[36]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   L1*sin(z[4])*sin(z[5])*sin(z[6])*z[34]*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[6])*sin(z[5])*z[36]*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[5])*sin(z[6])*z[35]*(L2*M3+LC2*M2)   0.0   2*cos(z[5])*sin(z[5])*z[35]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*z[37]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))+L1*z[38]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-L1*cos(z[7])*cos(z[8])*sin(z[9])*z[39]*(L2*M3+LC2*M2)   L1*sin(z[7])*sin(z[8])*sin(z[9])*z[37]*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[9])*sin(z[8])*z[39]*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[8])*sin(z[9])*z[38]*(L2*M3+LC2*M2)
            0.0   0.0   0.0   0.0   0.0   0.0   L1*z[37]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))+L1*z[38]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-L1*cos(z[7])*cos(z[8])*sin(z[9])*z[39]*(L2*M3+LC2*M2)   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   L1*sin(z[7])*sin(z[8])*sin(z[9])*z[37]*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[9])*sin(z[8])*z[39]*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[8])*sin(z[9])*z[38]*(L2*M3+LC2*M2)   0.0   2*cos(z[8])*sin(z[8])*z[38]*(J2+L2^2*M3+LC2^2*M2)        
        ] + [   # Mp for p=L1
            2*L1*(M2+M3)	(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))	-cos(z[1])*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)	0	0	0	0	0	0
            (L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))	0	0	0	0	0	0	0	0
            -cos(z[1])*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)	0	0	0	0	0	0	0	0
            0	0	0	2*L1*(M2+M3)	(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))	-cos(z[4])*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)	0	0	0
            0	0	0	(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))	0	0	0	0	0
            0	0	0	-cos(z[4])*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)	0	0	0	0	0
            0	0	0	0	0	0	2*L1*(M2+M3)	(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))	-cos(z[7])*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)
            0	0	0	0	0	0	(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))	0	0
            0	0	0	0	0	0	-cos(z[7])*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)	0	0
        ]

        dM_dL1invM = [dM_dL1[:,1:3]/m1   dM_dL1[:,4:6]/m2   dM_dL1[:,7:9]/m3]
        dH_dL1invM = [dH_dL1[:,1:3]/m1   dH_dL1[:,4:6]/m2   dH_dL1[:,7:9]/m3]

        orange_term_L1 = dH_dL1invM - HinvM*dM_dL1invM
        d_HinvMHT_dL1 = orange_term_L1*transpose(H) + HinvM*transpose(dH_dL1)

        # dκp = [H*inv(M)*transpose(H)]\{orange_term*cgBu + H*inv(M)*dcgBu/dp) - [orange_term*transpose(H)+H*inv(M)*transpose(dH/dp)] }
        dz[49:54] = (HinvM*transpose(H))\(orange_term_L1*cgBu + HinvM*([
            # cgBu_p_sansp for p=L1
            z[41]*(2*L1*z[11]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[12]*(L2*M3+LC2*M2))-z[42]*(2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]*(L2*M3+LC2*M2))-z[32]*(L1*z[11]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))+L1*cos(z[1])*cos(z[2])*cos(z[3])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+γ*z[40]+z[31]*(g*sin(z[1])*(L1*(M2+M3)+LC1*M1)+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2]))+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)+2*L1*cos(z[2])*sin(z[1])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+z[33]*(L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]^2*(L2*M3+LC2*M2)+L1*cos(z[1])*sin(z[2])*sin(z[3])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*cos(z[2])*cos(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))
            z[33]*(g*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2))+γ*z[41]+z[32]*(sin(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2])))-2*cos(z[2])*sin(z[2])*z[42]*z[12]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[40]*z[10]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))-L1*z[31]*z[10]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))
            z[42]*(γ+sin(2*z[2])*z[11]*(J2+L2^2*M3+LC2^2*M2))+z[33]*(g*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[10]^2*(L2*M3+LC2*M2))+z[32]*(2*cos(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2))+sin(2*z[2])*z[41]*z[12]*(J2+L2^2*M3+LC2^2*M2)+L1*cos(z[1])*sin(z[2])*sin(z[3])*z[31]*z[10]^2*(L2*M3+LC2*M2)+2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[40]*z[10]*(L2*M3+LC2*M2)
            z[44]*(2*L1*z[14]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[15]*(L2*M3+LC2*M2))-z[45]*(2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]*(L2*M3+LC2*M2))-z[35]*(L1*z[14]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))+L1*cos(z[4])*cos(z[5])*cos(z[6])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+γ*z[43]+z[34]*(g*sin(z[4])*(L1*(M2+M3)+LC1*M1)+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5]))+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)+2*L1*cos(z[5])*sin(z[4])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+z[36]*(L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*sin(z[5])*sin(z[6])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*cos(z[5])*cos(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))
            z[36]*(g*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2))+γ*z[44]+z[35]*(sin(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5])))-2*cos(z[5])*sin(z[5])*z[45]*z[15]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[43]*z[13]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))-L1*z[34]*z[13]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))
            z[45]*(γ+sin(2*z[5])*z[14]*(J2+L2^2*M3+LC2^2*M2))+z[36]*(g*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[13]^2*(L2*M3+LC2*M2))+z[35]*(2*cos(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2))+sin(2*z[5])*z[44]*z[15]*(J2+L2^2*M3+LC2^2*M2)+L1*cos(z[4])*sin(z[5])*sin(z[6])*z[34]*z[13]^2*(L2*M3+LC2*M2)+2*L1*sin(z[4])*sin(z[5])*sin(z[6])*z[43]*z[13]*(L2*M3+LC2*M2)
            z[47]*(2*L1*z[17]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[18]*(L2*M3+LC2*M2))-z[48]*(2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]*(L2*M3+LC2*M2))-z[38]*(L1*z[17]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))+L1*cos(z[7])*cos(z[8])*cos(z[9])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+γ*z[46]+z[37]*(g*sin(z[7])*(L1*(M2+M3)+LC1*M1)+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8]))+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)+2*L1*cos(z[8])*sin(z[7])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+z[39]*(L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*sin(z[8])*sin(z[9])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*cos(z[8])*cos(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))
            z[39]*(g*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2))+γ*z[47]+z[38]*(sin(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8])))-2*cos(z[8])*sin(z[8])*z[48]*z[18]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[46]*z[16]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))-L1*z[37]*z[16]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))
            z[48]*(γ+sin(2*z[8])*z[17]*(J2+L2^2*M3+LC2^2*M2))+z[39]*(g*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[16]^2*(L2*M3+LC2*M2))+z[38]*(2*cos(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2))+sin(2*z[8])*z[47]*z[18]*(J2+L2^2*M3+LC2^2*M2)+L1*cos(z[7])*sin(z[8])*sin(z[9])*z[37]*z[16]^2*(L2*M3+LC2*M2)+2*L1*sin(z[7])*sin(z[8])*sin(z[9])*z[46]*z[16]*(L2*M3+LC2*M2)
        ] + [   #cgBu_p for p=L1
            z[11]^2*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-g*cos(z[1])*(M2+M3)-cos(z[1])*cos(z[3])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)-2*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2)
            z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))
            sin(z[1])*sin(z[2])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)
            z[14]^2*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-g*cos(z[4])*(M2+M3)-cos(z[4])*cos(z[6])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)-2*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2)
            z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))
            sin(z[4])*sin(z[5])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)
            z[17]^2*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-g*cos(z[7])*(M2+M3)-cos(z[7])*cos(z[9])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)-2*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2)
            z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))
            sin(z[7])*sin(z[8])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)        
        ]) - d_HinvMHT_dL1*z[19:24])     # d_HinvMHT_dL1 = orange_term_L1*transpose(H) + HinvM*transpose(dH_dL1)

        dMvterm_dL1 = [  # Mvterm_p_sansp for p=L1
            z[42]*(2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]*(L2*M3+LC2*M2))-z[31]*(g*sin(z[1])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[1])*dz[20]+L1*cos(z[1])*dz[23]+L1*sin(z[1])*dz[21]+L1*sin(z[1])*dz[24]+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2]))+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)+2*L1*cos(z[2])*sin(z[1])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))-z[41]*(2*L1*z[11]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[12]*(L2*M3+LC2*M2))+z[32]*(L1*z[11]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))+L1*cos(z[1])*cos(z[2])*cos(z[3])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))-γ*z[40]-z[33]*(L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]^2*(L2*M3+LC2*M2)+L1*cos(z[1])*sin(z[2])*sin(z[3])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*cos(z[2])*cos(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+L1*cos(z[1])*dz[51]+L1*cos(z[1])*dz[54]-L1*sin(z[1])*dz[50]-L1*sin(z[1])*dz[53]
            L2*cos(z[2])*cos(z[3])*dz[51]-γ*z[41]-z[32]*(sin(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)+L2*cos(z[2])*dz[20]+L2*cos(z[2])*dz[23]+L2*cos(z[3])*sin(z[2])*dz[21]+L2*cos(z[3])*sin(z[2])*dz[24]+L2*sin(z[2])*sin(z[3])*dz[19]+L2*sin(z[2])*sin(z[3])*dz[22]+g*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2])))-L2*sin(z[2])*dz[50]-L2*sin(z[2])*dz[53]-z[33]*(L2*cos(z[2])*sin(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[22]-L2*cos(z[2])*cos(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[24]+g*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2))+L2*cos(z[2])*cos(z[3])*dz[54]+L2*cos(z[2])*sin(z[3])*dz[49]+L2*cos(z[2])*sin(z[3])*dz[52]+2*cos(z[2])*sin(z[2])*z[42]*z[12]*(J2+L2^2*M3+LC2^2*M2)-2*L1*z[40]*z[10]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))+L1*z[31]*z[10]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))
            L2*cos(z[3])*sin(z[2])*dz[49]-z[33]*(L2*cos(z[3])*sin(z[2])*dz[21]+L2*cos(z[3])*sin(z[2])*dz[24]+L2*sin(z[2])*sin(z[3])*dz[19]+L2*sin(z[2])*sin(z[3])*dz[22]+g*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[10]^2*(L2*M3+LC2*M2))-z[42]*(γ+sin(2*z[2])*z[11]*(J2+L2^2*M3+LC2^2*M2))-z[32]*(L2*cos(z[2])*sin(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[22]-L2*cos(z[2])*cos(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[24]+2*cos(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2))+L2*cos(z[3])*sin(z[2])*dz[52]-L2*sin(z[2])*sin(z[3])*dz[51]-L2*sin(z[2])*sin(z[3])*dz[54]-sin(2*z[2])*z[41]*z[12]*(J2+L2^2*M3+LC2^2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*z[31]*z[10]^2*(L2*M3+LC2*M2)-2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[40]*z[10]*(L2*M3+LC2*M2)
            z[45]*(2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]*(L2*M3+LC2*M2))-z[44]*(2*L1*z[14]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[15]*(L2*M3+LC2*M2))+z[35]*(L1*z[14]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))+L1*cos(z[4])*cos(z[5])*cos(z[6])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))-z[34]*(g*sin(z[4])*(L1*(M2+M3)+LC1*M1)+(L1*cos(z[4])*dz[20])*0.5-L1*sin(z[4])*dz[21]+(sqrt(3)*L1*cos(z[4])*dz[19])*0.5+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5]))+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)+2*L1*cos(z[5])*sin(z[4])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))-γ*z[43]-z[36]*(L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*sin(z[5])*sin(z[6])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*cos(z[5])*cos(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))-L1*cos(z[4])*dz[51]-(L1*sin(z[4])*dz[50])*0.5-(sqrt(3)*L1*sin(z[4])*dz[49])*0.5
            dz[49]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)-dz[50]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-z[36]*(g*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)-L2*cos(z[5])*sin(z[6])*dz[21]-(L2*cos(z[5])*cos(z[6])*dz[19])*0.5+(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[20])*0.5+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2))-z[35]*(dz[19]*((sqrt(3)*L2*cos(z[5]))*0.5+(L2*sin(z[5])*sin(z[6]))*0.5)+dz[20]*((L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5)-cos(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)+sin(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[6])*sin(z[5])*dz[21]+g*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5])))-γ*z[44]-L2*cos(z[5])*cos(z[6])*dz[51]+2*cos(z[5])*sin(z[5])*z[45]*z[15]*(J2+L2^2*M3+LC2^2*M2)-2*L1*z[43]*z[13]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))+L1*z[34]*z[13]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))
            (L2*cos(z[6])*sin(z[5])*dz[49])*0.5-z[35]*(2*cos(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[5])*sin(z[6])*dz[21]-(L2*cos(z[5])*cos(z[6])*dz[19])*0.5+g*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[20])*0.5+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2))-z[36]*((L2*sin(z[5])*sin(z[6])*dz[19])*0.5-L2*cos(z[6])*sin(z[5])*dz[21]+g*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)-(sqrt(3)*L2*sin(z[5])*sin(z[6])*dz[20])*0.5+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[13]^2*(L2*M3+LC2*M2))-z[45]*(γ+sin(2*z[5])*z[14]*(J2+L2^2*M3+LC2^2*M2))+L2*sin(z[5])*sin(z[6])*dz[51]-sin(2*z[5])*z[44]*z[15]*(J2+L2^2*M3+LC2^2*M2)-(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[50])*0.5-L1*cos(z[4])*sin(z[5])*sin(z[6])*z[34]*z[13]^2*(L2*M3+LC2*M2)-2*L1*sin(z[4])*sin(z[5])*sin(z[6])*z[43]*z[13]*(L2*M3+LC2*M2)
            z[48]*(2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]*(L2*M3+LC2*M2))-z[47]*(2*L1*z[17]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[18]*(L2*M3+LC2*M2))+z[38]*(L1*z[17]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))+L1*cos(z[7])*cos(z[8])*cos(z[9])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))-z[37]*(g*sin(z[7])*(L1*(M2+M3)+LC1*M1)+(L1*cos(z[7])*dz[23])*0.5-L1*sin(z[7])*dz[24]-(sqrt(3)*L1*cos(z[7])*dz[22])*0.5+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8]))+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)+2*L1*cos(z[8])*sin(z[7])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))-γ*z[46]-z[39]*(L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*sin(z[8])*sin(z[9])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*cos(z[8])*cos(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))-L1*cos(z[7])*dz[54]-(L1*sin(z[7])*dz[53])*0.5+(sqrt(3)*L1*sin(z[7])*dz[52])*0.5
            dz[52]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)-dz[53]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)+z[39]*((L2*cos(z[8])*cos(z[9])*dz[22])*0.5+L2*cos(z[8])*sin(z[9])*dz[24]-g*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[23])*0.5-L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2))-z[38]*(dz[23]*((L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5)-dz[22]*((sqrt(3)*L2*cos(z[8]))*0.5-(L2*sin(z[8])*sin(z[9]))*0.5)-cos(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)+sin(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[9])*sin(z[8])*dz[24]+g*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8])))-γ*z[47]-L2*cos(z[8])*cos(z[9])*dz[54]+2*cos(z[8])*sin(z[8])*z[48]*z[18]*(J2+L2^2*M3+LC2^2*M2)-2*L1*z[46]*z[16]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))+L1*z[37]*z[16]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))
            z[38]*((L2*cos(z[8])*cos(z[9])*dz[22])*0.5+L2*cos(z[8])*sin(z[9])*dz[24]-2*cos(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)-g*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[23])*0.5-L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2))-z[48]*(γ+sin(2*z[8])*z[17]*(J2+L2^2*M3+LC2^2*M2))-z[39]*((L2*sin(z[8])*sin(z[9])*dz[22])*0.5-L2*cos(z[9])*sin(z[8])*dz[24]+g*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+(sqrt(3)*L2*sin(z[8])*sin(z[9])*dz[23])*0.5+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[16]^2*(L2*M3+LC2*M2))+(L2*cos(z[9])*sin(z[8])*dz[52])*0.5+L2*sin(z[8])*sin(z[9])*dz[54]-sin(2*z[8])*z[47]*z[18]*(J2+L2^2*M3+LC2^2*M2)+(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[53])*0.5-L1*cos(z[7])*sin(z[8])*sin(z[9])*z[37]*z[16]^2*(L2*M3+LC2*M2)-2*L1*sin(z[7])*sin(z[8])*sin(z[9])*z[46]*z[16]*(L2*M3+LC2*M2)
        ] + [   # Mvterm_p for p=L1
            cos(z[1])*dz[21]+cos(z[1])*dz[24]-sin(z[1])*dz[20]-sin(z[1])*dz[23]+g*cos(z[1])*(M2+M3)-z[11]^2*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))+cos(z[1])*cos(z[3])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)+2*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2)
            -z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))
            -sin(z[1])*sin(z[2])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)
            g*cos(z[4])*(M2+M3)-(sin(z[4])*dz[20])*0.5-cos(z[4])*dz[21]-z[14]^2*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-(sqrt(3)*sin(z[4])*dz[19])*0.5+cos(z[4])*cos(z[6])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)+2*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2)
            -z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))
            -sin(z[4])*sin(z[5])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)
            g*cos(z[7])*(M2+M3)-(sin(z[7])*dz[23])*0.5-cos(z[7])*dz[24]-z[17]^2*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))+(sqrt(3)*sin(z[7])*dz[22])*0.5+cos(z[7])*cos(z[9])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)+2*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2)
            -z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))
            -sin(z[7])*sin(z[8])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)        
        ]

        # dvp = inv(M)*(dMvterm_dp - rp(dM/dp, dv)), rp is my special product
        dz[40:48] = [m1\(dMvterm_dL1[1:3,:] - dM_dL1[1:3,:]*dz[10:18])
            m2\(dMvterm_dL1[4:6,:] - dM_dL1[4:6,:]*dz[10:18])
            m3\(dMvterm_dL1[7:9,:] - dM_dL1[7:9,:]*dz[10:18])]

        ############################# INITIALIZING SENSITIVITY PART FOR M1 ####################################

        dH_dM1 = [   # Hp_sansp
            0.0   L2*cos(z[2])*cos(z[3])*z[33]-L2*sin(z[2])*sin(z[3])*z[32]   L2*cos(z[2])*cos(z[3])*z[32]-L2*sin(z[2])*sin(z[3])*z[33]   -(sqrt(3)*L1*cos(z[4])*z[34])*0.5   (L2*cos(z[5])*cos(z[6])*z[36])*0.5-z[35]*((sqrt(3)*L2*cos(z[5]))*0.5+(L2*sin(z[5])*sin(z[6]))*0.5)   (L2*cos(z[5])*cos(z[6])*z[35])*0.5-(L2*sin(z[5])*sin(z[6])*z[36])*0.5   0.0   0.0   0
            -2*L1*cos(z[1])*z[31]   -L2*cos(z[2])*z[32]   0.0   -(L1*cos(z[4])*z[34])*0.5   -z[35]*((L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5)-(sqrt(3)*L2*cos(z[5])*cos(z[6])*z[36])*0.5   (sqrt(3)*L2*sin(z[5])*sin(z[6])*z[36])*0.5-(sqrt(3)*L2*cos(z[5])*cos(z[6])*z[35])*0.5   0.0   0.0   0
            -2*L1*sin(z[1])*z[31]   -L2*cos(z[3])*sin(z[2])*z[32]-L2*cos(z[2])*sin(z[3])*z[33]   -L2*cos(z[2])*sin(z[3])*z[32]-L2*cos(z[3])*sin(z[2])*z[33]   L1*sin(z[4])*z[34]   L2*cos(z[6])*sin(z[5])*z[35]+L2*cos(z[5])*sin(z[6])*z[36]   L2*cos(z[5])*sin(z[6])*z[35]+L2*cos(z[6])*sin(z[5])*z[36]   0.0   0.0   0
            0.0   L2*cos(z[2])*cos(z[3])*z[33]-L2*sin(z[2])*sin(z[3])*z[32]   L2*cos(z[2])*cos(z[3])*z[32]-L2*sin(z[2])*sin(z[3])*z[33]   0.0   0.0   0.0   (sqrt(3)*L1*cos(z[7])*z[37])*0.5   z[38]*((sqrt(3)*L2*cos(z[8]))*0.5-(L2*sin(z[8])*sin(z[9]))*0.5)+(L2*cos(z[8])*cos(z[9])*z[39])*0.5   (L2*cos(z[8])*cos(z[9])*z[38])*0.5-(L2*sin(z[8])*sin(z[9])*z[39])*0.5
            -2*L1*cos(z[1])*z[31]   -L2*cos(z[2])*z[32]   0.0   0.0   0.0   0.0   -(L1*cos(z[7])*z[37])*0.5   (sqrt(3)*L2*cos(z[8])*cos(z[9])*z[39])*0.5-z[38]*((L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5)   (sqrt(3)*L2*cos(z[8])*cos(z[9])*z[38])*0.5-(sqrt(3)*L2*sin(z[8])*sin(z[9])*z[39])*0.5
            -2*L1*sin(z[1])*z[31]   -L2*cos(z[3])*sin(z[2])*z[32]-L2*cos(z[2])*sin(z[3])*z[33]   -L2*cos(z[2])*sin(z[3])*z[32]-L2*cos(z[3])*sin(z[2])*z[33]   0.0   0.0   0.0   L1*sin(z[7])*z[37]   L2*cos(z[9])*sin(z[8])*z[38]+L2*cos(z[8])*sin(z[9])*z[39]   L2*cos(z[8])*sin(z[9])*z[38]+L2*cos(z[9])*sin(z[8])*z[39]        
        ] # Hp for p=M1 is zero

        dM_dM1 = [   # Mp_sansp
            0.0   2*L1*z[31]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))+L1*z[32]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-L1*cos(z[1])*cos(z[2])*sin(z[3])*z[33]*(L2*M3+LC2*M2)   2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[31]*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[3])*sin(z[2])*z[33]*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[2])*sin(z[3])*z[32]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0
            2*L1*z[31]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))+L1*z[32]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-L1*cos(z[1])*cos(z[2])*sin(z[3])*z[33]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0
            2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[31]*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[3])*sin(z[2])*z[33]*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[2])*sin(z[3])*z[32]*(L2*M3+LC2*M2)   0.0   2*cos(z[2])*sin(z[2])*z[32]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0
            0.0   0.0   0.0   0.0   L1*z[34]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))+L1*z[35]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-L1*cos(z[4])*cos(z[5])*sin(z[6])*z[36]*(L2*M3+LC2*M2)   L1*sin(z[4])*sin(z[5])*sin(z[6])*z[34]*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[6])*sin(z[5])*z[36]*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[5])*sin(z[6])*z[35]*(L2*M3+LC2*M2)   0.0   0.0   0
            0.0   0.0   0.0   L1*z[34]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))+L1*z[35]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-L1*cos(z[4])*cos(z[5])*sin(z[6])*z[36]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0
            0.0   0.0   0.0   L1*sin(z[4])*sin(z[5])*sin(z[6])*z[34]*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[6])*sin(z[5])*z[36]*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[5])*sin(z[6])*z[35]*(L2*M3+LC2*M2)   0.0   2*cos(z[5])*sin(z[5])*z[35]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*z[37]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))+L1*z[38]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-L1*cos(z[7])*cos(z[8])*sin(z[9])*z[39]*(L2*M3+LC2*M2)   L1*sin(z[7])*sin(z[8])*sin(z[9])*z[37]*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[9])*sin(z[8])*z[39]*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[8])*sin(z[9])*z[38]*(L2*M3+LC2*M2)
            0.0   0.0   0.0   0.0   0.0   0.0   L1*z[37]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))+L1*z[38]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-L1*cos(z[7])*cos(z[8])*sin(z[9])*z[39]*(L2*M3+LC2*M2)   0.0   0
            0.0   0.0   0.0   0.0   0.0   0.0   L1*sin(z[7])*sin(z[8])*sin(z[9])*z[37]*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[9])*sin(z[8])*z[39]*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[8])*sin(z[9])*z[38]*(L2*M3+LC2*M2)   0.0   2*cos(z[8])*sin(z[8])*z[38]*(J2+L2^2*M3+LC2^2*M2)        
        ] + [   # Mp for p=M1
            LC1^2	0.0   0.0   0.0       0.0   0.0   0.0       0.0   0.0
            0.0       0.0   0.0   0.0       0.0   0.0   0.0       0.0   0.0
            0.0       0.0   0.0   0.0       0.0   0.0   0.0       0.0   0.0
            0.0       0.0   0.0   LC1^2	0.0   0.0   0.0       0.0   0.0
            0.0       0.0   0.0   0.0       0.0   0.0   0.0       0.0   0.0
            0.0       0.0   0.0   0.0       0.0   0.0   0.0       0.0   0.0
            0.0       0.0   0.0   0.0       0.0   0.0   LC1^2	0.0   0.0
            0.0       0.0   0.0   0.0       0.0   0.0   0.0       0.0   0.0
            0.0       0.0   0.0   0.0       0.0   0.0   0.0       0.0   0.0
        ]

        dM_dM1invM = [dM_dM1[:,1:3]/m1   dM_dM1[:,4:6]/m2   dM_dM1[:,7:9]/m3]
        dH_dM1invM = [dH_dM1[:,1:3]/m1   dH_dM1[:,4:6]/m2   dH_dM1[:,7:9]/m3]

        orange_term_M1 = dH_dM1invM - HinvM*dM_dM1invM
        d_HinvMHT_dM1 = orange_term_M1*transpose(H) + HinvM*transpose(dH_dM1)

        # dκp = [H*inv(M)*transpose(H)]\{orange_term*cgBu + H*inv(M)*dcgBu/dp) - [orange_term*transpose(H)+H*inv(M)*transpose(dH/dp)] }
        dz[79:84] = (HinvM*transpose(H))\(orange_term_M1*cgBu + HinvM*([
            # cgBu_p_sansp for p=M1
            z[41]*(2*L1*z[11]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[12]*(L2*M3+LC2*M2))-z[42]*(2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]*(L2*M3+LC2*M2))-z[32]*(L1*z[11]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))+L1*cos(z[1])*cos(z[2])*cos(z[3])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+γ*z[40]+z[31]*(g*sin(z[1])*(L1*(M2+M3)+LC1*M1)+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2]))+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)+2*L1*cos(z[2])*sin(z[1])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+z[33]*(L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]^2*(L2*M3+LC2*M2)+L1*cos(z[1])*sin(z[2])*sin(z[3])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*cos(z[2])*cos(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))
            z[33]*(g*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2))+γ*z[41]+z[32]*(sin(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2])))-2*cos(z[2])*sin(z[2])*z[42]*z[12]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[40]*z[10]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))-L1*z[31]*z[10]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))
            z[42]*(γ+sin(2*z[2])*z[11]*(J2+L2^2*M3+LC2^2*M2))+z[33]*(g*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[10]^2*(L2*M3+LC2*M2))+z[32]*(2*cos(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2))+sin(2*z[2])*z[41]*z[12]*(J2+L2^2*M3+LC2^2*M2)+L1*cos(z[1])*sin(z[2])*sin(z[3])*z[31]*z[10]^2*(L2*M3+LC2*M2)+2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[40]*z[10]*(L2*M3+LC2*M2)
            z[44]*(2*L1*z[14]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[15]*(L2*M3+LC2*M2))-z[45]*(2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]*(L2*M3+LC2*M2))-z[35]*(L1*z[14]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))+L1*cos(z[4])*cos(z[5])*cos(z[6])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+γ*z[43]+z[34]*(g*sin(z[4])*(L1*(M2+M3)+LC1*M1)+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5]))+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)+2*L1*cos(z[5])*sin(z[4])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+z[36]*(L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*sin(z[5])*sin(z[6])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*cos(z[5])*cos(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))
            z[36]*(g*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2))+γ*z[44]+z[35]*(sin(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5])))-2*cos(z[5])*sin(z[5])*z[45]*z[15]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[43]*z[13]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))-L1*z[34]*z[13]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))
            z[45]*(γ+sin(2*z[5])*z[14]*(J2+L2^2*M3+LC2^2*M2))+z[36]*(g*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[13]^2*(L2*M3+LC2*M2))+z[35]*(2*cos(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2))+sin(2*z[5])*z[44]*z[15]*(J2+L2^2*M3+LC2^2*M2)+L1*cos(z[4])*sin(z[5])*sin(z[6])*z[34]*z[13]^2*(L2*M3+LC2*M2)+2*L1*sin(z[4])*sin(z[5])*sin(z[6])*z[43]*z[13]*(L2*M3+LC2*M2)
            z[47]*(2*L1*z[17]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[18]*(L2*M3+LC2*M2))-z[48]*(2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]*(L2*M3+LC2*M2))-z[38]*(L1*z[17]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))+L1*cos(z[7])*cos(z[8])*cos(z[9])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+γ*z[46]+z[37]*(g*sin(z[7])*(L1*(M2+M3)+LC1*M1)+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8]))+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)+2*L1*cos(z[8])*sin(z[7])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+z[39]*(L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*sin(z[8])*sin(z[9])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*cos(z[8])*cos(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))
            z[39]*(g*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2))+γ*z[47]+z[38]*(sin(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8])))-2*cos(z[8])*sin(z[8])*z[48]*z[18]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[46]*z[16]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))-L1*z[37]*z[16]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))
            z[48]*(γ+sin(2*z[8])*z[17]*(J2+L2^2*M3+LC2^2*M2))+z[39]*(g*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[16]^2*(L2*M3+LC2*M2))+z[38]*(2*cos(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2))+sin(2*z[8])*z[47]*z[18]*(J2+L2^2*M3+LC2^2*M2)+L1*cos(z[7])*sin(z[8])*sin(z[9])*z[37]*z[16]^2*(L2*M3+LC2*M2)+2*L1*sin(z[7])*sin(z[8])*sin(z[9])*z[46]*z[16]*(L2*M3+LC2*M2)
        ] + [   #cgBu_p for p=M1
            -LC1*g*cos(z[1])
            0.0
            0.0
            -LC1*g*cos(z[4])
            0.0
            0.0
            -LC1*g*cos(z[7])
            0.0
            0.0
        ]) - d_HinvMHT_dM1*z[19:24])     # d_HinvMHT_dM1 = orange_term_M1*transpose(H) + HinvM*transpose(dH_dM1)

        dMvterm_dM1 = [  # Mvterm_p_sansp for p=M1
            z[42]*(2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]*(L2*M3+LC2*M2))-z[31]*(g*sin(z[1])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[1])*dz[20]+L1*cos(z[1])*dz[23]+L1*sin(z[1])*dz[21]+L1*sin(z[1])*dz[24]+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2]))+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)+2*L1*cos(z[2])*sin(z[1])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))-z[41]*(2*L1*z[11]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[12]*(L2*M3+LC2*M2))+z[32]*(L1*z[11]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))+L1*cos(z[1])*cos(z[2])*cos(z[3])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))-γ*z[40]-z[33]*(L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]^2*(L2*M3+LC2*M2)+L1*cos(z[1])*sin(z[2])*sin(z[3])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*cos(z[2])*cos(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+L1*cos(z[1])*dz[51]+L1*cos(z[1])*dz[54]-L1*sin(z[1])*dz[50]-L1*sin(z[1])*dz[53]
            L2*cos(z[2])*cos(z[3])*dz[51]-γ*z[41]-z[32]*(sin(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)+L2*cos(z[2])*dz[20]+L2*cos(z[2])*dz[23]+L2*cos(z[3])*sin(z[2])*dz[21]+L2*cos(z[3])*sin(z[2])*dz[24]+L2*sin(z[2])*sin(z[3])*dz[19]+L2*sin(z[2])*sin(z[3])*dz[22]+g*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2])))-L2*sin(z[2])*dz[50]-L2*sin(z[2])*dz[53]-z[33]*(L2*cos(z[2])*sin(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[22]-L2*cos(z[2])*cos(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[24]+g*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2))+L2*cos(z[2])*cos(z[3])*dz[54]+L2*cos(z[2])*sin(z[3])*dz[49]+L2*cos(z[2])*sin(z[3])*dz[52]+2*cos(z[2])*sin(z[2])*z[42]*z[12]*(J2+L2^2*M3+LC2^2*M2)-2*L1*z[40]*z[10]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))+L1*z[31]*z[10]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))
            L2*cos(z[3])*sin(z[2])*dz[49]-z[33]*(L2*cos(z[3])*sin(z[2])*dz[21]+L2*cos(z[3])*sin(z[2])*dz[24]+L2*sin(z[2])*sin(z[3])*dz[19]+L2*sin(z[2])*sin(z[3])*dz[22]+g*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[10]^2*(L2*M3+LC2*M2))-z[42]*(γ+sin(2*z[2])*z[11]*(J2+L2^2*M3+LC2^2*M2))-z[32]*(L2*cos(z[2])*sin(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[22]-L2*cos(z[2])*cos(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[24]+2*cos(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2))+L2*cos(z[3])*sin(z[2])*dz[52]-L2*sin(z[2])*sin(z[3])*dz[51]-L2*sin(z[2])*sin(z[3])*dz[54]-sin(2*z[2])*z[41]*z[12]*(J2+L2^2*M3+LC2^2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*z[31]*z[10]^2*(L2*M3+LC2*M2)-2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[40]*z[10]*(L2*M3+LC2*M2)
            z[45]*(2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]*(L2*M3+LC2*M2))-z[44]*(2*L1*z[14]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[15]*(L2*M3+LC2*M2))+z[35]*(L1*z[14]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))+L1*cos(z[4])*cos(z[5])*cos(z[6])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))-z[34]*(g*sin(z[4])*(L1*(M2+M3)+LC1*M1)+(L1*cos(z[4])*dz[20])*0.5-L1*sin(z[4])*dz[21]+(sqrt(3)*L1*cos(z[4])*dz[19])*0.5+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5]))+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)+2*L1*cos(z[5])*sin(z[4])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))-γ*z[43]-z[36]*(L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*sin(z[5])*sin(z[6])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*cos(z[5])*cos(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))-L1*cos(z[4])*dz[51]-(L1*sin(z[4])*dz[50])*0.5-(sqrt(3)*L1*sin(z[4])*dz[49])*0.5
            dz[49]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)-dz[50]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-z[36]*(g*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)-L2*cos(z[5])*sin(z[6])*dz[21]-(L2*cos(z[5])*cos(z[6])*dz[19])*0.5+(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[20])*0.5+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2))-z[35]*(dz[19]*((sqrt(3)*L2*cos(z[5]))*0.5+(L2*sin(z[5])*sin(z[6]))*0.5)+dz[20]*((L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5)-cos(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)+sin(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[6])*sin(z[5])*dz[21]+g*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5])))-γ*z[44]-L2*cos(z[5])*cos(z[6])*dz[51]+2*cos(z[5])*sin(z[5])*z[45]*z[15]*(J2+L2^2*M3+LC2^2*M2)-2*L1*z[43]*z[13]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))+L1*z[34]*z[13]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))
            (L2*cos(z[6])*sin(z[5])*dz[49])*0.5-z[35]*(2*cos(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[5])*sin(z[6])*dz[21]-(L2*cos(z[5])*cos(z[6])*dz[19])*0.5+g*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[20])*0.5+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2))-z[36]*((L2*sin(z[5])*sin(z[6])*dz[19])*0.5-L2*cos(z[6])*sin(z[5])*dz[21]+g*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)-(sqrt(3)*L2*sin(z[5])*sin(z[6])*dz[20])*0.5+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[13]^2*(L2*M3+LC2*M2))-z[45]*(γ+sin(2*z[5])*z[14]*(J2+L2^2*M3+LC2^2*M2))+L2*sin(z[5])*sin(z[6])*dz[51]-sin(2*z[5])*z[44]*z[15]*(J2+L2^2*M3+LC2^2*M2)-(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[50])*0.5-L1*cos(z[4])*sin(z[5])*sin(z[6])*z[34]*z[13]^2*(L2*M3+LC2*M2)-2*L1*sin(z[4])*sin(z[5])*sin(z[6])*z[43]*z[13]*(L2*M3+LC2*M2)
            z[48]*(2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]*(L2*M3+LC2*M2))-z[47]*(2*L1*z[17]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[18]*(L2*M3+LC2*M2))+z[38]*(L1*z[17]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))+L1*cos(z[7])*cos(z[8])*cos(z[9])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))-z[37]*(g*sin(z[7])*(L1*(M2+M3)+LC1*M1)+(L1*cos(z[7])*dz[23])*0.5-L1*sin(z[7])*dz[24]-(sqrt(3)*L1*cos(z[7])*dz[22])*0.5+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8]))+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)+2*L1*cos(z[8])*sin(z[7])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))-γ*z[46]-z[39]*(L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*sin(z[8])*sin(z[9])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*cos(z[8])*cos(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))-L1*cos(z[7])*dz[54]-(L1*sin(z[7])*dz[53])*0.5+(sqrt(3)*L1*sin(z[7])*dz[52])*0.5
            dz[52]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)-dz[53]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)+z[39]*((L2*cos(z[8])*cos(z[9])*dz[22])*0.5+L2*cos(z[8])*sin(z[9])*dz[24]-g*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[23])*0.5-L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2))-z[38]*(dz[23]*((L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5)-dz[22]*((sqrt(3)*L2*cos(z[8]))*0.5-(L2*sin(z[8])*sin(z[9]))*0.5)-cos(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)+sin(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[9])*sin(z[8])*dz[24]+g*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8])))-γ*z[47]-L2*cos(z[8])*cos(z[9])*dz[54]+2*cos(z[8])*sin(z[8])*z[48]*z[18]*(J2+L2^2*M3+LC2^2*M2)-2*L1*z[46]*z[16]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))+L1*z[37]*z[16]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))
            z[38]*((L2*cos(z[8])*cos(z[9])*dz[22])*0.5+L2*cos(z[8])*sin(z[9])*dz[24]-2*cos(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)-g*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[23])*0.5-L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2))-z[48]*(γ+sin(2*z[8])*z[17]*(J2+L2^2*M3+LC2^2*M2))-z[39]*((L2*sin(z[8])*sin(z[9])*dz[22])*0.5-L2*cos(z[9])*sin(z[8])*dz[24]+g*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+(sqrt(3)*L2*sin(z[8])*sin(z[9])*dz[23])*0.5+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[16]^2*(L2*M3+LC2*M2))+(L2*cos(z[9])*sin(z[8])*dz[52])*0.5+L2*sin(z[8])*sin(z[9])*dz[54]-sin(2*z[8])*z[47]*z[18]*(J2+L2^2*M3+LC2^2*M2)+(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[53])*0.5-L1*cos(z[7])*sin(z[8])*sin(z[9])*z[37]*z[16]^2*(L2*M3+LC2*M2)-2*L1*sin(z[7])*sin(z[8])*sin(z[9])*z[46]*z[16]*(L2*M3+LC2*M2)
        ] + [   # Mvterm_p for p=M1
            LC1*g*cos(z[1])
            0.0
            0.0
            LC1*g*cos(z[4])
            0.0
            0.0
            LC1*g*cos(z[7])
            0.0
            0.0
        ]

        # dvp = inv(M)*(dMvterm_dp - rp(dM/dp, dv)), rp is my special product
        dz[70:78] = [m1\(dMvterm_dM1[1:3,:] - dM_dM1[1:3,:]*dz[10:18])
            m2\(dMvterm_dM1[4:6,:] - dM_dM1[4:6,:]*dz[10:18])
            m3\(dMvterm_dM1[7:9,:] - dM_dM1[7:9,:]*dz[10:18])]

        ############################# INITIALIZING SENSITIVITY PART FOR J1 ####################################

        dH_dJ1 = [   # Hp_sansp
            0.0   L2*cos(z[2])*cos(z[3])*z[33]-L2*sin(z[2])*sin(z[3])*z[32]   L2*cos(z[2])*cos(z[3])*z[32]-L2*sin(z[2])*sin(z[3])*z[33]   -(sqrt(3)*L1*cos(z[4])*z[34])*0.5   (L2*cos(z[5])*cos(z[6])*z[36])*0.5-z[35]*((sqrt(3)*L2*cos(z[5]))*0.5+(L2*sin(z[5])*sin(z[6]))*0.5)   (L2*cos(z[5])*cos(z[6])*z[35])*0.5-(L2*sin(z[5])*sin(z[6])*z[36])*0.5   0.0   0.0   0.0
            -2*L1*cos(z[1])*z[31]   -L2*cos(z[2])*z[32]   0.0   -(L1*cos(z[4])*z[34])*0.5   -z[35]*((L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5)-(sqrt(3)*L2*cos(z[5])*cos(z[6])*z[36])*0.5   (sqrt(3)*L2*sin(z[5])*sin(z[6])*z[36])*0.5-(sqrt(3)*L2*cos(z[5])*cos(z[6])*z[35])*0.5   0.0   0.0   0.0
            -2*L1*sin(z[1])*z[31]   -L2*cos(z[3])*sin(z[2])*z[32]-L2*cos(z[2])*sin(z[3])*z[33]   -L2*cos(z[2])*sin(z[3])*z[32]-L2*cos(z[3])*sin(z[2])*z[33]   L1*sin(z[4])*z[34]   L2*cos(z[6])*sin(z[5])*z[35]+L2*cos(z[5])*sin(z[6])*z[36]   L2*cos(z[5])*sin(z[6])*z[35]+L2*cos(z[6])*sin(z[5])*z[36]   0.0   0.0   0.0
            0.0   L2*cos(z[2])*cos(z[3])*z[33]-L2*sin(z[2])*sin(z[3])*z[32]   L2*cos(z[2])*cos(z[3])*z[32]-L2*sin(z[2])*sin(z[3])*z[33]   0.0   0.0   0.0   (sqrt(3)*L1*cos(z[7])*z[37])*0.5   z[38]*((sqrt(3)*L2*cos(z[8]))*0.5-(L2*sin(z[8])*sin(z[9]))*0.5)+(L2*cos(z[8])*cos(z[9])*z[39])*0.5   (L2*cos(z[8])*cos(z[9])*z[38])*0.5-(L2*sin(z[8])*sin(z[9])*z[39])*0.5
            -2*L1*cos(z[1])*z[31]   -L2*cos(z[2])*z[32]   0.0   0.0   0.0   0.0   -(L1*cos(z[7])*z[37])*0.5   (sqrt(3)*L2*cos(z[8])*cos(z[9])*z[39])*0.5-z[38]*((L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5)   (sqrt(3)*L2*cos(z[8])*cos(z[9])*z[38])*0.5-(sqrt(3)*L2*sin(z[8])*sin(z[9])*z[39])*0.5
            -2*L1*sin(z[1])*z[31]   -L2*cos(z[3])*sin(z[2])*z[32]-L2*cos(z[2])*sin(z[3])*z[33]   -L2*cos(z[2])*sin(z[3])*z[32]-L2*cos(z[3])*sin(z[2])*z[33]   0.0   0.0   0.0   L1*sin(z[7])*z[37]   L2*cos(z[9])*sin(z[8])*z[38]+L2*cos(z[8])*sin(z[9])*z[39]   L2*cos(z[8])*sin(z[9])*z[38]+L2*cos(z[9])*sin(z[8])*z[39]        
        ] # Hp for p=J1 is zero

        dM_dJ1 = [   # Mp_sansp
            0.0   2*L1*z[31]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))+L1*z[32]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-L1*cos(z[1])*cos(z[2])*sin(z[3])*z[33]*(L2*M3+LC2*M2)   2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[31]*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[3])*sin(z[2])*z[33]*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[2])*sin(z[3])*z[32]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0
            2*L1*z[31]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))+L1*z[32]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-L1*cos(z[1])*cos(z[2])*sin(z[3])*z[33]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[31]*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[3])*sin(z[2])*z[33]*(L2*M3+LC2*M2)-L1*cos(z[1])*cos(z[2])*sin(z[3])*z[32]*(L2*M3+LC2*M2)   0.0   2*cos(z[2])*sin(z[2])*z[32]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   L1*z[34]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))+L1*z[35]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-L1*cos(z[4])*cos(z[5])*sin(z[6])*z[36]*(L2*M3+LC2*M2)   L1*sin(z[4])*sin(z[5])*sin(z[6])*z[34]*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[6])*sin(z[5])*z[36]*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[5])*sin(z[6])*z[35]*(L2*M3+LC2*M2)   0.0   0.0   0.0
            0.0   0.0   0.0   L1*z[34]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))+L1*z[35]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-L1*cos(z[4])*cos(z[5])*sin(z[6])*z[36]*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   L1*sin(z[4])*sin(z[5])*sin(z[6])*z[34]*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[6])*sin(z[5])*z[36]*(L2*M3+LC2*M2)-L1*cos(z[4])*cos(z[5])*sin(z[6])*z[35]*(L2*M3+LC2*M2)   0.0   2*cos(z[5])*sin(z[5])*z[35]*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*z[37]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))+L1*z[38]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-L1*cos(z[7])*cos(z[8])*sin(z[9])*z[39]*(L2*M3+LC2*M2)   L1*sin(z[7])*sin(z[8])*sin(z[9])*z[37]*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[9])*sin(z[8])*z[39]*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[8])*sin(z[9])*z[38]*(L2*M3+LC2*M2)
            0.0   0.0   0.0   0.0   0.0   0.0   L1*z[37]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))+L1*z[38]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-L1*cos(z[7])*cos(z[8])*sin(z[9])*z[39]*(L2*M3+LC2*M2)   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   L1*sin(z[7])*sin(z[8])*sin(z[9])*z[37]*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[9])*sin(z[8])*z[39]*(L2*M3+LC2*M2)-L1*cos(z[7])*cos(z[8])*sin(z[9])*z[38]*(L2*M3+LC2*M2)   0.0   2*cos(z[8])*sin(z[8])*z[38]*(J2+L2^2*M3+LC2^2*M2)        
        ] + [   # Mp for p=J1
            1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   1.0	0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   1.0	0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0	0.0
        ]

        dM_dJ1invM = [dM_dJ1[:,1:3]/m1   dM_dJ1[:,4:6]/m2   dM_dJ1[:,7:9]/m3]
        dH_dJ1invM = [dH_dJ1[:,1:3]/m1   dH_dJ1[:,4:6]/m2   dH_dJ1[:,7:9]/m3]

        orange_term_J1 = dH_dJ1invM - HinvM*dM_dJ1invM
        d_HinvMHT_dJ1 = orange_term_J1*transpose(H) + HinvM*transpose(dH_dJ1)

        # dκp = [H*inv(M)*transpose(H)]\{orange_term*cgBu + H*inv(M)*dcgBu/dp) - [orange_term*transpose(H)+H*inv(M)*transpose(dH/dp)] }
        dz[109:114] = (HinvM*transpose(H))\(orange_term_J1*cgBu + HinvM*[
            # cgBu_p_sansp for p=J1
            z[41]*(2*L1*z[11]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[12]*(L2*M3+LC2*M2))-z[42]*(2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]*(L2*M3+LC2*M2))-z[32]*(L1*z[11]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))+L1*cos(z[1])*cos(z[2])*cos(z[3])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+γ*z[40]+z[31]*(g*sin(z[1])*(L1*(M2+M3)+LC1*M1)+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2]))+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)+2*L1*cos(z[2])*sin(z[1])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+z[33]*(L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]^2*(L2*M3+LC2*M2)+L1*cos(z[1])*sin(z[2])*sin(z[3])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*cos(z[2])*cos(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))
            z[33]*(g*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2))+γ*z[41]+z[32]*(sin(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2])))-2*cos(z[2])*sin(z[2])*z[42]*z[12]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[40]*z[10]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))-L1*z[31]*z[10]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))
            z[42]*(γ+sin(2*z[2])*z[11]*(J2+L2^2*M3+LC2^2*M2))+z[33]*(g*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[10]^2*(L2*M3+LC2*M2))+z[32]*(2*cos(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2))+sin(2*z[2])*z[41]*z[12]*(J2+L2^2*M3+LC2^2*M2)+L1*cos(z[1])*sin(z[2])*sin(z[3])*z[31]*z[10]^2*(L2*M3+LC2*M2)+2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[40]*z[10]*(L2*M3+LC2*M2)
            z[44]*(2*L1*z[14]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[15]*(L2*M3+LC2*M2))-z[45]*(2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]*(L2*M3+LC2*M2))-z[35]*(L1*z[14]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))+L1*cos(z[4])*cos(z[5])*cos(z[6])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+γ*z[43]+z[34]*(g*sin(z[4])*(L1*(M2+M3)+LC1*M1)+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5]))+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)+2*L1*cos(z[5])*sin(z[4])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))+z[36]*(L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*sin(z[5])*sin(z[6])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*cos(z[5])*cos(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))
            z[36]*(g*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2))+γ*z[44]+z[35]*(sin(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5])))-2*cos(z[5])*sin(z[5])*z[45]*z[15]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[43]*z[13]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))-L1*z[34]*z[13]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))
            z[45]*(γ+sin(2*z[5])*z[14]*(J2+L2^2*M3+LC2^2*M2))+z[36]*(g*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[13]^2*(L2*M3+LC2*M2))+z[35]*(2*cos(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2))+sin(2*z[5])*z[44]*z[15]*(J2+L2^2*M3+LC2^2*M2)+L1*cos(z[4])*sin(z[5])*sin(z[6])*z[34]*z[13]^2*(L2*M3+LC2*M2)+2*L1*sin(z[4])*sin(z[5])*sin(z[6])*z[43]*z[13]*(L2*M3+LC2*M2)
            z[47]*(2*L1*z[17]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[18]*(L2*M3+LC2*M2))-z[48]*(2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]*(L2*M3+LC2*M2))-z[38]*(L1*z[17]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))+L1*cos(z[7])*cos(z[8])*cos(z[9])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+γ*z[46]+z[37]*(g*sin(z[7])*(L1*(M2+M3)+LC1*M1)+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8]))+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)+2*L1*cos(z[8])*sin(z[7])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))+z[39]*(L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*sin(z[8])*sin(z[9])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*cos(z[8])*cos(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))
            z[39]*(g*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2))+γ*z[47]+z[38]*(sin(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8])))-2*cos(z[8])*sin(z[8])*z[48]*z[18]*(J2+L2^2*M3+LC2^2*M2)+2*L1*z[46]*z[16]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))-L1*z[37]*z[16]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))
            z[48]*(γ+sin(2*z[8])*z[17]*(J2+L2^2*M3+LC2^2*M2))+z[39]*(g*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[16]^2*(L2*M3+LC2*M2))+z[38]*(2*cos(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2))+sin(2*z[8])*z[47]*z[18]*(J2+L2^2*M3+LC2^2*M2)+L1*cos(z[7])*sin(z[8])*sin(z[9])*z[37]*z[16]^2*(L2*M3+LC2*M2)+2*L1*sin(z[7])*sin(z[8])*sin(z[9])*z[46]*z[16]*(L2*M3+LC2*M2)
            #cgBu_p for p=J1 is zero
        ] - d_HinvMHT_dJ1*z[19:24])     # d_HinvMHT_dJ1 = orange_term_J1*transpose(H) + HinvM*transpose(dH_dJ1)

        dMvterm_dJ1 = [  # Mvterm_p_sansp for p=J1
            z[42]*(2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]*(L2*M3+LC2*M2))-z[31]*(g*sin(z[1])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[1])*dz[20]+L1*cos(z[1])*dz[23]+L1*sin(z[1])*dz[21]+L1*sin(z[1])*dz[24]+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2]))+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)+2*L1*cos(z[2])*sin(z[1])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))-z[41]*(2*L1*z[11]*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[12]*(L2*M3+LC2*M2))+z[32]*(L1*z[11]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))+L1*cos(z[1])*cos(z[2])*cos(z[3])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))-γ*z[40]-z[33]*(L1*cos(z[1])*sin(z[2])*sin(z[3])*z[11]^2*(L2*M3+LC2*M2)+L1*cos(z[1])*sin(z[2])*sin(z[3])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*cos(z[2])*cos(z[3])*z[11]*z[12]*(L2*M3+LC2*M2))+L1*cos(z[1])*dz[51]+L1*cos(z[1])*dz[54]-L1*sin(z[1])*dz[50]-L1*sin(z[1])*dz[53]
            L2*cos(z[2])*cos(z[3])*dz[51]-γ*z[41]-z[32]*(sin(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-cos(z[2])^2*z[12]^2*(J2+L2^2*M3+LC2^2*M2)+L2*cos(z[2])*dz[20]+L2*cos(z[2])*dz[23]+L2*cos(z[3])*sin(z[2])*dz[21]+L2*cos(z[3])*sin(z[2])*dz[24]+L2*sin(z[2])*sin(z[3])*dz[19]+L2*sin(z[2])*sin(z[3])*dz[22]+g*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*cos(z[2])+cos(z[3])*sin(z[1])*sin(z[2])))-L2*sin(z[2])*dz[50]-L2*sin(z[2])*dz[53]-z[33]*(L2*cos(z[2])*sin(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[22]-L2*cos(z[2])*cos(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[24]+g*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2))+L2*cos(z[2])*cos(z[3])*dz[54]+L2*cos(z[2])*sin(z[3])*dz[49]+L2*cos(z[2])*sin(z[3])*dz[52]+2*cos(z[2])*sin(z[2])*z[42]*z[12]*(J2+L2^2*M3+LC2^2*M2)-2*L1*z[40]*z[10]*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))+L1*z[31]*z[10]^2*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))
            L2*cos(z[3])*sin(z[2])*dz[49]-z[33]*(L2*cos(z[3])*sin(z[2])*dz[21]+L2*cos(z[3])*sin(z[2])*dz[24]+L2*sin(z[2])*sin(z[3])*dz[19]+L2*sin(z[2])*sin(z[3])*dz[22]+g*cos(z[3])*sin(z[2])*(L2*M3+LC2*M2)+L1*cos(z[3])*sin(z[1])*sin(z[2])*z[10]^2*(L2*M3+LC2*M2))-z[42]*(γ+sin(2*z[2])*z[11]*(J2+L2^2*M3+LC2^2*M2))-z[32]*(L2*cos(z[2])*sin(z[3])*dz[21]-L2*cos(z[2])*cos(z[3])*dz[22]-L2*cos(z[2])*cos(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[24]+2*cos(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*cos(z[2])*sin(z[1])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2))+L2*cos(z[3])*sin(z[2])*dz[52]-L2*sin(z[2])*sin(z[3])*dz[51]-L2*sin(z[2])*sin(z[3])*dz[54]-sin(2*z[2])*z[41]*z[12]*(J2+L2^2*M3+LC2^2*M2)-L1*cos(z[1])*sin(z[2])*sin(z[3])*z[31]*z[10]^2*(L2*M3+LC2*M2)-2*L1*sin(z[1])*sin(z[2])*sin(z[3])*z[40]*z[10]*(L2*M3+LC2*M2)
            z[45]*(2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]*(L2*M3+LC2*M2))-z[44]*(2*L1*z[14]*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[15]*(L2*M3+LC2*M2))+z[35]*(L1*z[14]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))+L1*cos(z[4])*cos(z[5])*cos(z[6])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))-z[34]*(g*sin(z[4])*(L1*(M2+M3)+LC1*M1)+(L1*cos(z[4])*dz[20])*0.5-L1*sin(z[4])*dz[21]+(sqrt(3)*L1*cos(z[4])*dz[19])*0.5+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5]))+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)+2*L1*cos(z[5])*sin(z[4])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))-γ*z[43]-z[36]*(L1*cos(z[4])*sin(z[5])*sin(z[6])*z[14]^2*(L2*M3+LC2*M2)+L1*cos(z[4])*sin(z[5])*sin(z[6])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*cos(z[5])*cos(z[6])*z[14]*z[15]*(L2*M3+LC2*M2))-L1*cos(z[4])*dz[51]-(L1*sin(z[4])*dz[50])*0.5-(sqrt(3)*L1*sin(z[4])*dz[49])*0.5
            dz[49]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)-dz[50]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-z[36]*(g*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)-L2*cos(z[5])*sin(z[6])*dz[21]-(L2*cos(z[5])*cos(z[6])*dz[19])*0.5+(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[20])*0.5+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2))-z[35]*(dz[19]*((sqrt(3)*L2*cos(z[5]))*0.5+(L2*sin(z[5])*sin(z[6]))*0.5)+dz[20]*((L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5)-cos(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)+sin(z[5])^2*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[6])*sin(z[5])*dz[21]+g*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*cos(z[5])+cos(z[6])*sin(z[4])*sin(z[5])))-γ*z[44]-L2*cos(z[5])*cos(z[6])*dz[51]+2*cos(z[5])*sin(z[5])*z[45]*z[15]*(J2+L2^2*M3+LC2^2*M2)-2*L1*z[43]*z[13]*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))+L1*z[34]*z[13]^2*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))
            (L2*cos(z[6])*sin(z[5])*dz[49])*0.5-z[35]*(2*cos(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[5])*sin(z[6])*dz[21]-(L2*cos(z[5])*cos(z[6])*dz[19])*0.5+g*cos(z[5])*sin(z[6])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[5])*cos(z[6])*dz[20])*0.5+L1*cos(z[5])*sin(z[4])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2))-z[36]*((L2*sin(z[5])*sin(z[6])*dz[19])*0.5-L2*cos(z[6])*sin(z[5])*dz[21]+g*cos(z[6])*sin(z[5])*(L2*M3+LC2*M2)-(sqrt(3)*L2*sin(z[5])*sin(z[6])*dz[20])*0.5+L1*cos(z[6])*sin(z[4])*sin(z[5])*z[13]^2*(L2*M3+LC2*M2))-z[45]*(γ+sin(2*z[5])*z[14]*(J2+L2^2*M3+LC2^2*M2))+L2*sin(z[5])*sin(z[6])*dz[51]-sin(2*z[5])*z[44]*z[15]*(J2+L2^2*M3+LC2^2*M2)-(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[50])*0.5-L1*cos(z[4])*sin(z[5])*sin(z[6])*z[34]*z[13]^2*(L2*M3+LC2*M2)-2*L1*sin(z[4])*sin(z[5])*sin(z[6])*z[43]*z[13]*(L2*M3+LC2*M2)
            z[48]*(2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]*(L2*M3+LC2*M2))-z[47]*(2*L1*z[17]*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[18]*(L2*M3+LC2*M2))+z[38]*(L1*z[17]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))+L1*cos(z[7])*cos(z[8])*cos(z[9])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))-z[37]*(g*sin(z[7])*(L1*(M2+M3)+LC1*M1)+(L1*cos(z[7])*dz[23])*0.5-L1*sin(z[7])*dz[24]-(sqrt(3)*L1*cos(z[7])*dz[22])*0.5+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8]))+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)+2*L1*cos(z[8])*sin(z[7])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))-γ*z[46]-z[39]*(L1*cos(z[7])*sin(z[8])*sin(z[9])*z[17]^2*(L2*M3+LC2*M2)+L1*cos(z[7])*sin(z[8])*sin(z[9])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*cos(z[8])*cos(z[9])*z[17]*z[18]*(L2*M3+LC2*M2))-L1*cos(z[7])*dz[54]-(L1*sin(z[7])*dz[53])*0.5+(sqrt(3)*L1*sin(z[7])*dz[52])*0.5
            dz[52]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)-dz[53]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)+z[39]*((L2*cos(z[8])*cos(z[9])*dz[22])*0.5+L2*cos(z[8])*sin(z[9])*dz[24]-g*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[23])*0.5-L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2))-z[38]*(dz[23]*((L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5)-dz[22]*((sqrt(3)*L2*cos(z[8]))*0.5-(L2*sin(z[8])*sin(z[9]))*0.5)-cos(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)+sin(z[8])^2*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-L2*cos(z[9])*sin(z[8])*dz[24]+g*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*cos(z[8])+cos(z[9])*sin(z[7])*sin(z[8])))-γ*z[47]-L2*cos(z[8])*cos(z[9])*dz[54]+2*cos(z[8])*sin(z[8])*z[48]*z[18]*(J2+L2^2*M3+LC2^2*M2)-2*L1*z[46]*z[16]*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))+L1*z[37]*z[16]^2*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))
            z[38]*((L2*cos(z[8])*cos(z[9])*dz[22])*0.5+L2*cos(z[8])*sin(z[9])*dz[24]-2*cos(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)-g*cos(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[8])*cos(z[9])*dz[23])*0.5-L1*cos(z[8])*sin(z[7])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2))-z[48]*(γ+sin(2*z[8])*z[17]*(J2+L2^2*M3+LC2^2*M2))-z[39]*((L2*sin(z[8])*sin(z[9])*dz[22])*0.5-L2*cos(z[9])*sin(z[8])*dz[24]+g*cos(z[9])*sin(z[8])*(L2*M3+LC2*M2)+(sqrt(3)*L2*sin(z[8])*sin(z[9])*dz[23])*0.5+L1*cos(z[9])*sin(z[7])*sin(z[8])*z[16]^2*(L2*M3+LC2*M2))+(L2*cos(z[9])*sin(z[8])*dz[52])*0.5+L2*sin(z[8])*sin(z[9])*dz[54]-sin(2*z[8])*z[47]*z[18]*(J2+L2^2*M3+LC2^2*M2)+(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[53])*0.5-L1*cos(z[7])*sin(z[8])*sin(z[9])*z[37]*z[16]^2*(L2*M3+LC2*M2)-2*L1*sin(z[7])*sin(z[8])*sin(z[9])*z[46]*z[16]*(L2*M3+LC2*M2)
        ]   # Mvterm_p for p=J1 is zero

        # dvp = inv(M)*(dMvterm_dp - rp(dM/dp, dv)), rp is my special product
        dz[100:108] = [m1\(dMvterm_dJ1[1:3,:] - dM_dJ1[1:3,:]*dz[10:18])
            m2\(dMvterm_dJ1[4:6,:] - dM_dJ1[4:6,:]*dz[10:18])
            m3\(dMvterm_dJ1[7:9,:] - dM_dJ1[7:9,:]*dz[10:18])]

        return z, dz
    end
end

function get_delta_initial_comp(θ, u0, w0)
    let L0 = θ[1], L1 = θ[2], L2 = θ[3], L3 = θ[4], LC1 = θ[5], LC2 = θ[6], M1 = θ[7], M2 = θ[8], M3 = θ[9], J1 = θ[10], J2 = θ[11], g = 0.0, γ = θ[13]

        z  = zeros(30)
        dz = zeros(30)
        z[1:3] = [π/6; acos(-0.5*sqrt(3)*L1/L2 + (L3-L0)/L2); 0.0]
        z[4:6] = [π/6; acos(-0.5*sqrt(3)*L1/L2 + (L3-L0)/L2); 0.0]
        z[7:9] = [π/6; acos(-0.5*sqrt(3)*L1/L2 + (L3-L0)/L2); 0.0]

        H = [
            0.0   L2*cos(z[2])*sin(z[3])   L2*cos(z[3])*sin(z[2])   -(sqrt(3)*L1*sin(z[4]))*0.5   (L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5   (L2*cos(z[6])*sin(z[5]))*0.5   0.0   0.0   0.0
            -L1*sin(z[1])   -L2*sin(z[2])   0.0   -(L1*sin(z[4]))*0.5   -(L2*sin(z[5]))*0.5-(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5   -(sqrt(3)*L2*cos(z[6])*sin(z[5]))*0.5   0.0   0.0   0.0
            L1*cos(z[1])   L2*cos(z[2])*cos(z[3])   -L2*sin(z[2])*sin(z[3])   -L1*cos(z[4])   -L2*cos(z[5])*cos(z[6])   L2*sin(z[5])*sin(z[6])   0.0   0.0   0.0
            0.0   L2*cos(z[2])*sin(z[3])   L2*cos(z[3])*sin(z[2])   0.0   0.0   0.0   (sqrt(3)*L1*sin(z[7]))*0.5   (L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5   (L2*cos(z[9])*sin(z[8]))*0.5
            -L1*sin(z[1])   -L2*sin(z[2])   0.0   0.0   0.0   0.0   -(L1*sin(z[7]))*0.5   (sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5-(L2*sin(z[8]))*0.5   (sqrt(3)*L2*cos(z[9])*sin(z[8]))*0.5
            L1*cos(z[1])   L2*cos(z[2])*cos(z[3])   -L2*sin(z[2])*sin(z[3])   0.0   0.0   0.0   -L1*cos(z[7])   -L2*cos(z[8])*cos(z[9])   L2*sin(z[8])*sin(z[9])
        ]

        m1 = [
            J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))   -L1*cos(z[1])*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)
            L1*(L2*M3+LC2*M2)*(sin(z[1])*sin(z[2])+cos(z[1])*cos(z[2])*cos(z[3]))   J2+L2^2*M3+LC2^2*M2   0
            -L1*cos(z[1])*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)   0.0   sin(z[2])^2*(J2+L2^2*M3+LC2^2*M2)
        ]
        m2 = [
            J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))   -L1*cos(z[4])*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)
            L1*(L2*M3+LC2*M2)*(sin(z[4])*sin(z[5])+cos(z[4])*cos(z[5])*cos(z[6]))   J2+L2^2*M3+LC2^2*M2   0
            -L1*cos(z[4])*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)   0.0   sin(z[5])^2*(J2+L2^2*M3+LC2^2*M2)
        ]
        m3 = [
            J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))   -L1*cos(z[7])*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)
            L1*(L2*M3+LC2*M2)*(sin(z[7])*sin(z[8])+cos(z[7])*cos(z[8])*cos(z[9]))   J2+L2^2*M3+LC2^2*M2   0
            -L1*cos(z[7])*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)   0.0   sin(z[8])^2*(J2+L2^2*M3+LC2^2*M2)
        ]

        HinvM = [
            H[:,1:3]/m1   H[:,4:6]/m2   H[:,7:9]/m3
        ]

        cgBu = [
            γ*z[10]-u0[1]+w0[1]^2-g*cos(z[1])*(L1*(M2+M3)+LC1*M1)+L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))-L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)-2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2)
            γ*z[11]-cos(z[2])*sin(z[2])*z[12]^2*(J2+L2^2*M3+LC2^2*M2)-g*cos(z[2])*cos(z[3])*(L2*M3+LC2*M2)+L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))
            γ*z[12]+sin(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)+g*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)+L1*sin(z[1])*sin(z[2])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)
            γ*z[13]-u0[2]+w0[2]^2-g*cos(z[4])*(L1*(M2+M3)+LC1*M1)+L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))-L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)-2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2)
            γ*z[14]-cos(z[5])*sin(z[5])*z[15]^2*(J2+L2^2*M3+LC2^2*M2)-g*cos(z[5])*cos(z[6])*(L2*M3+LC2*M2)+L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))
            γ*z[15]+sin(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)+g*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)+L1*sin(z[4])*sin(z[5])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)
            γ*z[16]-u0[3]+w0[3]^2-g*cos(z[7])*(L1*(M2+M3)+LC1*M1)+L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))-L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)-2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2)
            γ*z[17]-cos(z[8])*sin(z[8])*z[18]^2*(J2+L2^2*M3+LC2^2*M2)-g*cos(z[8])*cos(z[9])*(L2*M3+LC2*M2)+L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))
            γ*z[18]+sin(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)+g*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)+L1*sin(z[7])*sin(z[8])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)
        ]

        # dκ = (HinvM*transpose(H))\(HinvM*cgBu)
        dz[19:24] = (HinvM*transpose(H))\(HinvM*cgBu)

        # dv = inv(M)*Mvterm
        dz[10:18] = [
            m1\[
                u0[1]+w0[1]^2-γ*z[10]+g*cos(z[1])*(L1*(M2+M3)+LC1*M1)+L1*cos(z[1])*dz[21]+L1*cos(z[1])*dz[24]-L1*sin(z[1])*dz[20]-L1*sin(z[1])*dz[23]-L1*z[11]^2*(L2*M3+LC2*M2)*(cos(z[2])*sin(z[1])-cos(z[1])*cos(z[3])*sin(z[2]))+L1*cos(z[1])*cos(z[3])*sin(z[2])*z[12]^2*(L2*M3+LC2*M2)+2*L1*cos(z[1])*cos(z[2])*sin(z[3])*z[11]*z[12]*(L2*M3+LC2*M2)
                L2*cos(z[2])*cos(z[3])*dz[21]-L2*sin(z[2])*dz[20]-L2*sin(z[2])*dz[23]-γ*z[11]+L2*cos(z[2])*cos(z[3])*dz[24]+L2*cos(z[2])*sin(z[3])*dz[19]+L2*cos(z[2])*sin(z[3])*dz[22]+cos(z[2])*sin(z[2])*z[12]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[2])*cos(z[3])*(L2*M3+LC2*M2)-L1*z[10]^2*(L2*M3+LC2*M2)*(cos(z[1])*sin(z[2])-cos(z[2])*cos(z[3])*sin(z[1]))
                L2*cos(z[3])*sin(z[2])*dz[19]-γ*z[12]+L2*cos(z[3])*sin(z[2])*dz[22]-L2*sin(z[2])*sin(z[3])*dz[21]-L2*sin(z[2])*sin(z[3])*dz[24]-sin(2*z[2])*z[11]*z[12]*(J2+L2^2*M3+LC2^2*M2)-g*sin(z[2])*sin(z[3])*(L2*M3+LC2*M2)-L1*sin(z[1])*sin(z[2])*sin(z[3])*z[10]^2*(L2*M3+LC2*M2)
            ]
            m2\[
                u0[2]+w0[2]^2-γ*z[13]+g*cos(z[4])*(L1*(M2+M3)+LC1*M1)-L1*cos(z[4])*dz[21]-(L1*sin(z[4])*dz[20])*0.5-(sqrt(3)*L1*sin(z[4])*dz[19])*0.5-L1*z[14]^2*(L2*M3+LC2*M2)*(cos(z[5])*sin(z[4])-cos(z[4])*cos(z[6])*sin(z[5]))+L1*cos(z[4])*cos(z[6])*sin(z[5])*z[15]^2*(L2*M3+LC2*M2)+2*L1*cos(z[4])*cos(z[5])*sin(z[6])*z[14]*z[15]*(L2*M3+LC2*M2)
                dz[19]*((L2*cos(z[5])*sin(z[6]))*0.5-(sqrt(3)*L2*sin(z[5]))*0.5)-dz[20]*((L2*sin(z[5]))*0.5+(sqrt(3)*L2*cos(z[5])*sin(z[6]))*0.5)-γ*z[14]-L2*cos(z[5])*cos(z[6])*dz[21]+cos(z[5])*sin(z[5])*z[15]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[5])*cos(z[6])*(L2*M3+LC2*M2)-L1*z[13]^2*(L2*M3+LC2*M2)*(cos(z[4])*sin(z[5])-cos(z[5])*cos(z[6])*sin(z[4]))
                (L2*cos(z[6])*sin(z[5])*dz[19])*0.5-γ*z[15]+L2*sin(z[5])*sin(z[6])*dz[21]-sin(2*z[5])*z[14]*z[15]*(J2+L2^2*M3+LC2^2*M2)-g*sin(z[5])*sin(z[6])*(L2*M3+LC2*M2)-(sqrt(3)*L2*cos(z[6])*sin(z[5])*dz[20])*0.5-L1*sin(z[4])*sin(z[5])*sin(z[6])*z[13]^2*(L2*M3+LC2*M2)
            ]
            m3\[
                u0[3]+w0[3]^2-γ*z[16]+g*cos(z[7])*(L1*(M2+M3)+LC1*M1)-L1*cos(z[7])*dz[24]-(L1*sin(z[7])*dz[23])*0.5+(sqrt(3)*L1*sin(z[7])*dz[22])*0.5-L1*z[17]^2*(L2*M3+LC2*M2)*(cos(z[8])*sin(z[7])-cos(z[7])*cos(z[9])*sin(z[8]))+L1*cos(z[7])*cos(z[9])*sin(z[8])*z[18]^2*(L2*M3+LC2*M2)+2*L1*cos(z[7])*cos(z[8])*sin(z[9])*z[17]*z[18]*(L2*M3+LC2*M2)
                dz[22]*((L2*cos(z[8])*sin(z[9]))*0.5+(sqrt(3)*L2*sin(z[8]))*0.5)-dz[23]*((L2*sin(z[8]))*0.5-(sqrt(3)*L2*cos(z[8])*sin(z[9]))*0.5)-γ*z[17]-L2*cos(z[8])*cos(z[9])*dz[24]+cos(z[8])*sin(z[8])*z[18]^2*(J2+L2^2*M3+LC2^2*M2)+g*cos(z[8])*cos(z[9])*(L2*M3+LC2*M2)-L1*z[16]^2*(L2*M3+LC2*M2)*(cos(z[7])*sin(z[8])-cos(z[8])*cos(z[9])*sin(z[7]))
                (L2*cos(z[9])*sin(z[8])*dz[22])*0.5-γ*z[18]+L2*sin(z[8])*sin(z[9])*dz[24]-sin(2*z[8])*z[17]*z[18]*(J2+L2^2*M3+LC2^2*M2)-g*sin(z[8])*sin(z[9])*(L2*M3+LC2*M2)+(sqrt(3)*L2*cos(z[9])*sin(z[8])*dz[23])*0.5-L1*sin(z[7])*sin(z[8])*sin(z[9])*z[16]^2*(L2*M3+LC2*M2)
            ]
        ]

        return z, dz
    end
end

function pendulum_new(Φ::Float64, u::Function, w::Function, θ::Vector{Float64})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - x[4] + 2dx[6]*x[1]
            res[2] = dx[2] - x[5] + 2dx[6]*x[2]
            res[3] = m*dx[4] - dx[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
            res[4] = m*dx[5] - dx[3]*x[2] + k*abs(x[5])*x[5] + m*g
            res[5] = x[1]^2 + x[2]^2 - L^2
            res[6] = x[4]*x[1] + x[5]*x[2]
            # Equation for obtaining angle
            res[7] = x[7] - atan(x[1] / -x[2])
            nothing
        end

        # Finding consistent initial conditions
        # Initial values, the pendulum starts at rest
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        x1_0 = L * sin(Φ)
        x2_0 = -L * cos(Φ)
        dx3_0 = m*g/x2_0
        dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m

        x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))

        dvars = vcat(fill(true, 6), [false])

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

# Sensitivity only with respect to disturbance parameter, no dynamic parameter
function pendulum_dist_sens_1(Φ::Float64, u::Function, w::Function, θ::Vector{Float64})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - x[4] + 2dx[6]*x[1]
            res[2] = dx[2] - x[5] + 2dx[6]*x[2]
            res[3] = m*dx[4] - dx[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
            res[4] = m*dx[5] - dx[3]*x[2] + k*abs(x[5])*x[5] + m*g
            res[5] = x[1]^2 + x[2]^2 - L^2
            res[6] = x[4]*x[1] + x[5]*x[2]
            # Angle of pendulum
            res[7] = x[7] - atan(x[1] / -x[2])

            # Sensitivty wrt first disturbance parameter
            # (i.e. parameter corresponding to wt[2])
            res[8] = 2dx[6]x[8] - x[11] + dx[8] + 2x[1]dx[13]
            res[9] = 2dx[6]x[9] - x[12] + dx[9] + 2x[2]dx[13]
            res[10] = -dx[3]x[8] + 2k*abs(x[4])*x[11] - x[1]*dx[10] + m*dx[11] - 2*wt[1]*wt[2]
            res[11] = -dx[3]x[9] + 2k*abs(x[5])*x[12] - x[2]*dx[10] + m*dx[12]
            res[12] = 2x[1]x[8] + 2x[2]x[9]
            res[13] = x[4]x[8] + x[5]x[9] + x[1]x[11] + x[2]x[12]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[2]*x[8])/(L^2)

            nothing
        end

        # Finding consistent initial conditions
        # Initial values, the pendulum starts at rest
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        x1_0 = L * sin(Φ)
        x2_0 = -L * cos(Φ)
        dx3_0 = m*g/x2_0
        dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m

        pend0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        r0 = zeros(7)
        rp0 = zeros(7)

        x0  = vcat(pend0, r0)
        dx0 = vcat(dpend0, rp0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = repeat(vcat(fill(true, 6), [false]), 2)

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

# Sensitivity with respect to two disturbance parameters, no dynamic parameter
function pendulum_dist_sens_2(Φ::Float64, u::Function, w::Function, θ::Vector{Float64})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - x[4] + 2dx[6]*x[1]
            res[2] = dx[2] - x[5] + 2dx[6]*x[2]
            res[3] = m*dx[4] - dx[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
            res[4] = m*dx[5] - dx[3]*x[2] + k*abs(x[5])*x[5] + m*g
            res[5] = x[1]^2 + x[2]^2 - L^2
            res[6] = x[4]*x[1] + x[5]*x[2]
            # Angle of pendulum
            res[7] = x[7] - atan(x[1] / -x[2])

            # Sensitivty wrt first disturbance parameter
            # (i.e. parameter corresponding to wt[2])
            res[8] = 2dx[6]x[8] - x[11] + dx[8] + 2x[1]dx[13]
            res[9] = 2dx[6]x[9] - x[12] + dx[9] + 2x[2]dx[13]
            res[10] = -dx[3]x[8] + 2k*abs(x[4])*x[11] - x[1]*dx[10] + m*dx[11] - 2*wt[1]*wt[2]
            res[11] = -dx[3]x[9] + 2k*abs(x[5])*x[12] - x[2]*dx[10] + m*dx[12]
            res[12] = 2x[1]x[8] + 2x[2]x[9]
            res[13] = x[4]x[8] + x[5]x[9] + x[1]x[11] + x[2]x[12]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[2]*x[8])/(L^2)

            # Sensitivty wrt second disturbance parameter
            # (i.e. parameter corresponding to wt[2])
            res[15] = 2dx[6]x[15] - x[18] + dx[15] + 2x[1]dx[20]
            res[16] = 2dx[6]x[16] - x[19] + dx[16] + 2x[2]dx[20]
            res[17] = -dx[3]x[15] + 2k*abs(x[4])*x[18] - x[1]*dx[17] + m*dx[18] - 2*wt[1]*wt[3]
            res[18] = -dx[3]x[16] + 2k*abs(x[5])*x[19] - x[2]*dx[17] + m*dx[19]
            res[19] = 2x[1]x[15] + 2x[2]x[16]
            res[20] = x[4]x[15] + x[5]x[16] + x[1]x[18] + x[2]x[19]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[21] = x[21] - (x[1]*x[16] - x[2]*x[15])/(L^2)

            nothing
        end

        # Finding consistent initial conditions
        # Initial values, the pendulum starts at rest
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        x1_0 = L * sin(Φ)
        x2_0 = -L * cos(Φ)
        dx3_0 = m*g/x2_0
        dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m

        pend0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        r0 = zeros(7)
        rp0 = zeros(7)

        x0  = vcat(pend0, r0, r0)
        dx0 = vcat(dpend0, rp0, rp0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = repeat(vcat(fill(true, 6), [false]), 3)

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

# Sensitivity with respect to "all" disturbance parameters, no dynamic parameter
function pendulum_dist_sens_3(Φ::Float64, u::Function, w::Function, θ::Vector{Float64})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - x[4] + 2dx[6]*x[1]
            res[2] = dx[2] - x[5] + 2dx[6]*x[2]
            res[3] = m*dx[4] - dx[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
            res[4] = m*dx[5] - dx[3]*x[2] + k*abs(x[5])*x[5] + m*g
            res[5] = x[1]^2 + x[2]^2 - L^2
            res[6] = x[4]*x[1] + x[5]*x[2]
            # Angle of pendulum
            res[7] = x[7] - atan(x[1] / -x[2])

            # Sensitivty wrt first disturbance parameter
            # (i.e. parameter corresponding to wt[2])
            res[8] = 2dx[6]x[8] - x[11] + dx[8] + 2x[1]dx[13]
            res[9] = 2dx[6]x[9] - x[12] + dx[9] + 2x[2]dx[13]
            res[10] = -dx[3]x[8] + 2k*abs(x[4])*x[11] - x[1]*dx[10] + m*dx[11] - 2*wt[1]*wt[2]
            res[11] = -dx[3]x[9] + 2k*abs(x[5])*x[12] - x[2]*dx[10] + m*dx[12]
            res[12] = 2x[1]x[8] + 2x[2]x[9]
            res[13] = x[4]x[8] + x[5]x[9] + x[1]x[11] + x[2]x[12]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[2]*x[8])/(L^2)

            # Sensitivty wrt second disturbance parameter
            # (i.e. parameter corresponding to wt[2])
            res[15] = 2dx[6]x[15] - x[18] + dx[15] + 2x[1]dx[20]
            res[16] = 2dx[6]x[16] - x[19] + dx[16] + 2x[2]dx[20]
            res[17] = -dx[3]x[15] + 2k*abs(x[4])*x[18] - x[1]*dx[17] + m*dx[18] - 2*wt[1]*wt[3]
            res[18] = -dx[3]x[16] + 2k*abs(x[5])*x[19] - x[2]*dx[17] + m*dx[19]
            res[19] = 2x[1]x[15] + 2x[2]x[16]
            res[20] = x[4]x[15] + x[5]x[16] + x[1]x[18] + x[2]x[19]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[21] = x[21] - (x[1]*x[16] - x[2]*x[15])/(L^2)

            # Sensitivty wrt third disturbance parameter
            # (i.e. parameter corresponding to wt[2])
            res[22] = 2dx[6]x[22] - x[25] + dx[22] + 2x[1]dx[27]
            res[23] = 2dx[6]x[23] - x[26] + dx[23] + 2x[2]dx[27]
            res[24] = -dx[3]x[22] + 2k*abs(x[4])*x[25] - x[1]*dx[24] + m*dx[25] - 2*wt[1]*wt[4]
            res[25] = -dx[3]x[23] + 2k*abs(x[5])*x[26] - x[2]*dx[24] + m*dx[26]
            res[26] = 2x[1]x[22] + 2x[2]x[23]
            res[27] = x[4]x[22] + x[5]x[23] + x[1]x[25] + x[2]x[26]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[28] = x[28] - (x[1]*x[23] - x[2]*x[22])/(L^2)

            nothing
        end

        # Finding consistent initial conditions
        # Initial values, the pendulum starts at rest
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        x1_0 = L * sin(Φ)
        x2_0 = -L * cos(Φ)
        dx3_0 = m*g/x2_0
        dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m

        pend0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        r0 = zeros(7)
        rp0 = zeros(7)

        x0  = vcat(pend0, r0, r0, r0)
        dx0 = vcat(dpend0, rp0, rp0, rp0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = repeat(vcat(fill(true, 6), [false]), 4)

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_sensitivity_k(Φ::Float64, u::Function, w::Function, θ::Vector{Float64})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - x[4] + 2dx[6]*x[1]
            res[2] = dx[2] - x[5] + 2dx[6]*x[2]
            res[3] = m*dx[4] - dx[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
            res[4] = m*dx[5] - dx[3]*x[2] + k*abs(x[5])*x[5] + m*g
            res[5] = x[1]^2 + x[2]^2 - L^2
            res[6] = x[4]*x[1] + x[5]*x[2]
            # Angle of pendulum
            res[7] = x[7] - atan(x[1] / -x[2])
            # Sensitivity Equations (wrt k)
            res[8]  = -x[11] + dx[8] + 2x[1]*dx[13] + 2x[8]*dx[6]
            res[9]  = -x[12] + dx[9] + 2x[2]*dx[13] + 2x[9]*dx[6]
            res[10] = abs(x[4])*x[4] + 2k*x[11]*abs(x[4]) - x[1]*dx[10] + m*dx[11] - x[8]*dx[3]
            res[11] = abs(x[5])*x[5] + 2k*x[12]*abs(x[5]) - x[2]*dx[10] + m*dx[12] - x[9]*dx[3]
            res[12] = -2x[8]*x[1] - 2x[9]*x[2]
            res[13] = x[11]*x[1] + x[12]*x[2] + x[8]*x[4] + x[9]*x[5]
            # Sensitivity of angle of pendulum
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[8]*x[2])/(L^2)

            # These equations sensitivity are written on the intuitive form. To
            # obtain form correct for this funciton, do the following replacements:
            # s[1] -> x[8],  s[2] -> x[9],  s[3] -> x[10]
            # s[4] -> x[11], s[5] -> x[12], s[6] -> x[13]
            # and the corresponding replacements for sp and dx

            # res[8]  = -s[4] + sp[1] + 2x[1]*sp[6] + 2s[1]*dx[6]
            # res[9]  = -s[5] + sp[2] + 2x[2]*sp[6] + 2s[2]*dx[6]
            # res[10]  = abs(x[4])*x[4] + 2k*s[4]*abs(x[4]) - x[1]*sp[3] + m*sp[4] - s[1]*dx[3]
            # res[11] = abs(x[5])*x[5] + 2k*s[5]*abs(x[5]) - x[2]*sp[3] + m*sp[5] - s[2]*dx[3]
            # res[12] = -2s[1]*x[1] - 2s[2]*x[2]
            # res[13] = s[4]*x[1] + s[5]*x[2] + s[1]*x[4] + s[2]*x[5]

            nothing
        end

        # Finding consistent initial conditions
        # Initial values, the pendulum starts at rest
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        x1_0 = L * sin(Φ)
        x2_0 = -L * cos(Φ)
        dx3_0 = m*g/x2_0
        dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m

        x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(7))
        dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(10))

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false])

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_sensitivity_L(Φ::Float64, u::Function, w::Function, θ::Vector{Float64})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - x[4] + 2dx[6]*x[1]
            res[2] = dx[2] - x[5] + 2dx[6]*x[2]
            res[3] = m*dx[4] - dx[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
            res[4] = m*dx[5] - dx[3]*x[2] + k*abs(x[5])*x[5] + m*g
            res[5] = x[1]^2 + x[2]^2 - L^2
            res[6] = x[4]*x[1] + x[5]*x[2]
            # Angle of pendulum
            res[7] = x[7] - atan(x[1] / -x[2])
            # Sensitivity equations for L
            res[8]  = dx[8] - x[11] + 2x[8]dx[6] + 2dx[13]x[1]
            res[9]  = dx[9] - x[12] + 2x[9]dx[6] + 2dx[13]x[2]
            res[10] = 2k*x[11]*abs(x[4]) + m*dx[11] - x[8]dx[3] - dx[10]x[1]
            res[11] = 2k*x[12]*abs(x[5]) + m*dx[12] - x[9]dx[3] - dx[10]x[2]
            res[12] = 2x[8]x[1] + 2x[9]x[2] - 2L
            res[13] = x[8]x[4] + x[11]x[1] + x[9]x[5] + x[12]x[2]
            # Sensitivity of angle of pendulum to L
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[8]*x[2])/(L^2)

            # These equations sensitivity are written on the intuitive form. To
            # obtain form correct for this funciton, do the following replacements:
            # s[1] -> x[8],  s[2] -> x[9],  s[3] -> x[10]
            # s[4] -> x[11], s[5] -> x[12], s[6] -> x[13]
            # t[1] -> x[15],  t[2] -> x[16],  t[3] -> x[17]
            # t[4] -> x[18], t[5] -> x[19], t[6] -> x[20]
            # and the corresponding replacements for sp/dx and tp/dx

            # res[8]  = -s[4] + sp[1] + 2x[1]*sp[6] + 2s[1]*dx[6]
            # res[9]  = -s[5] + sp[2] + 2x[2]*sp[6] + 2s[2]*dx[6]
            # res[10]  = abs(x[4])*x[4] + 2k*s[4]*abs(x[4]) - x[1]*sp[3] + m*sp[4] - s[1]*dx[3]
            # res[11] = abs(x[5])*x[5] + 2k*s[5]*abs(x[5]) - x[2]*sp[3] + m*sp[5] - s[2]*dx[3]
            # res[12] = -2s[1]*x[1] - 2s[2]*x[2]
            # res[13] = s[4]*x[1] + s[5]*x[2] + s[1]*x[4] + s[2]*x[5]
            # res[14] = x[14] - (x[1]*s[2] - x[2]*s[1])/(L^2)

            # MATLAB form
            # res[15]  = tp[1] - t[4] + 2t[1]dx[6] + 2tp[6]x[1]
            # res[16]  = tp[2] - t[5] + 2t[2]dx[6] + 2tp[6]x[2]
            # res[17] = 2k*t[4]*abs(x[4]) + m*tp[4] - t[1]dx[3] - tp[3]x[1]
            # res[18] = 2k*t[5]*abs(x[5]) + m*tp[5] - t[2]dx[3] - tp[3]x[2]
            # res[19] = 2L -2t[1]x[1] - 2t[2]x[2]
            # res[20] = t[1]x[4] + t[4]x[1] + t[2]x[5] + t[5]x[2]
            # res[21] = (x[1]*t[2] - x[2]*t[1])/(L^2)

            # Hand-derived form
            # res[15]  = -t[4] + tp[1] + 2x[1]*tp[6] + 2t[1]*dx[6]
            # res[16]  = -t[5] + tp[2] + 2x[2]*tp[6] + 2t[2]*dx[6]
            # res[17] = 2k*t[4]*abs(x[4]) - x[1]*tp[3] + m*tp[4] - t[1]*dx[3]
            # res[18] = 2k*t[5]*abs(x[5]) - x[2]*tp[3] + m*tp[5] - t[2]*dx[3]
            # res[19] = -2t[1]*x[1] - 2t[2]*x[2] + 2*L
            # res[20] = t[4]*x[1] + t[5]*x[2] + t[1]*x[4] + t[2]*x[5]

            nothing
        end

        # Finding consistent initial conditions
        # Initial values, the pendulum starts at rest
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        x1_0 = L * sin(Φ)
        x2_0 = -L * cos(Φ)
        dx3_0 = m*g/x2_0
        dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m

        pend0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        # pend_dvar = vcat(fill(true, 6))
        sL0  = vcat([x1_0/L, x2_0/L], zeros(5))
        dsL0 = vcat([0.,0., -dx3_0/L], zeros(4))
        # if x1_0 != 0.0
        #     t0  = vcat([L/(2x1_0), L/(2x2_0)], zeros(4), [(x1_0^2-x2_0^2)/(2L*x1_0*x2_0)])
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(2x1_0*x2_0)], zeros(4))
        # else
        #     t0  = vcat([0.0, L/x2_0], zeros(5))
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(x2_0^2)], zeros(4))
        # end

        x0  = vcat(pend0, sL0)
        dx0 = vcat(dpend0, dsL0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false])

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)
        # @info "r0 for pendulum L1 sensitivity is $r0"

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_sensitivity_m(Φ::Float64, u::Function, w::Function, θ::Vector{Float64})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - x[4] + 2dx[6]*x[1]
            res[2] = dx[2] - x[5] + 2dx[6]*x[2]
            res[3] = m*dx[4] - dx[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
            res[4] = m*dx[5] - dx[3]*x[2] + k*abs(x[5])*x[5] + m*g
            res[5] = x[1]^2 + x[2]^2 - L^2
            res[6] = x[4]*x[1] + x[5]*x[2]
            # Angle of pendulum
            res[7] = x[7] - atan(x[1] / -x[2])
            # Sensitivity with respect to m
            res[8]  = -x[11] + dx[8] + 2x[1]*dx[13] + 2x[8]*dx[6]
            res[9]  = -x[12] + dx[9] + 2x[2]*dx[13] + 2x[9]*dx[6]
            res[10] = 2k*x[11]*abs(x[4]) - x[1]*dx[10] + m*dx[11] - x[8]*dx[3] + dx[4]
            res[11] = 2k*x[12]*abs(x[5]) - x[2]*dx[10] + m*dx[12] - x[9]*dx[3] + g + dx[5]
            res[12] = -2x[8]*x[1] - 2x[9]*x[2]
            res[13] = x[11]*x[1] + x[12]*x[2] + x[8]*x[4] + x[9]*x[5]
            res[14] = x[14] - (x[1]*x[9] - x[2]*x[8])/(L^2)

            # These equations sensitivity are written on the intuitive form. To
            # obtain form correct for this funciton, do the following replacements:
            # s[1] -> x[8],  s[2] -> x[9],  s[3] -> x[10]
            # s[4] -> x[11], s[5] -> x[12], s[6] -> x[13]
            # t[1] -> x[15],  t[2] -> x[16],  t[3] -> x[17]
            # t[4] -> x[18], t[5] -> x[19], t[6] -> x[20]
            # and the corresponding replacements for sp/dx and tp/dx

            # res[8]  = -s[4] + sp[1] + 2x[1]*sp[6] + 2s[1]*dx[6]
            # res[9]  = -s[5] + sp[2] + 2x[2]*sp[6] + 2s[2]*dx[6]
            # res[10]  = abs(x[4])*x[4] + 2k*s[4]*abs(x[4]) - x[1]*sp[3] + m*sp[4] - s[1]*dx[3]
            # res[11] = abs(x[5])*x[5] + 2k*s[5]*abs(x[5]) - x[2]*sp[3] + m*sp[5] - s[2]*dx[3]
            # res[12] = -2s[1]*x[1] - 2s[2]*x[2]
            # res[13] = s[4]*x[1] + s[5]*x[2] + s[1]*x[4] + s[2]*x[5]
            # res[14] = x[14] - (x[1]*s[2] - x[2]*s[1])/(L^2)

            # MATLAB form
            # res[15]  = tp[1] - t[4] + 2t[1]dx[6] + 2tp[6]x[1]
            # res[16]  = tp[2] - t[5] + 2t[2]dx[6] + 2tp[6]x[2]
            # res[17] = 2k*t[4]*abs(x[4]) + m*tp[4] - t[1]dx[3] - tp[3]x[1]
            # res[18] = 2k*t[5]*abs(x[5]) + m*tp[5] - t[2]dx[3] - tp[3]x[2]
            # res[19] = 2L -2t[1]x[1] - 2t[2]x[2]
            # res[20] = t[1]x[4] + t[4]x[1] + t[2]x[5] + t[5]x[2]
            # res[21] = (x[1]*t[2] - x[2]*t[1])/(L^2)

            # Hand-derived form
            # res[15]  = -t[4] + tp[1] + 2x[1]*tp[6] + 2t[1]*dx[6]
            # res[16]  = -t[5] + tp[2] + 2x[2]*tp[6] + 2t[2]*dx[6]
            # res[17] = 2k*t[4]*abs(x[4]) - x[1]*tp[3] + m*tp[4] - t[1]*dx[3]
            # res[18] = 2k*t[5]*abs(x[5]) - x[2]*tp[3] + m*tp[5] - t[2]*dx[3]
            # res[19] = -2t[1]*x[1] - 2t[2]*x[2] + 2*L
            # res[20] = t[4]*x[1] + t[5]*x[2] + t[1]*x[4] + t[2]*x[5]

            nothing
        end

        # Finding consistent initial conditions
        # Initial values, the pendulum starts at rest
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        x1_0 = L * sin(Φ)
        x2_0 = -L * cos(Φ)
        dx3_0 = m*g/x2_0
        dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m

        pend0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        # pend_dvar = vcat(fill(true, 6))
        sm0  = zeros(7)
        dsm0 = vcat(zeros(3), [-dx4_0/m, -g/m, 0., 0.])
        # if x1_0 != 0.0
        #     t0  = vcat([L/(2x1_0), L/(2x2_0)], zeros(4), [(x1_0^2-x2_0^2)/(2L*x1_0*x2_0)])
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(2x1_0*x2_0)], zeros(4))
        # else
        #     t0  = vcat([0.0, L/x2_0], zeros(5))
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(x2_0^2)], zeros(4))
        # end

        x0  = vcat(pend0, sm0)
        dx0 = vcat(dpend0, dsm0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false])

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_sensitivity_g(Φ::Float64, u::Function, w::Function, θ::Vector{Float64})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - x[4] + 2dx[6]*x[1]
            res[2] = dx[2] - x[5] + 2dx[6]*x[2]
            res[3] = m*dx[4] - dx[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
            res[4] = m*dx[5] - dx[3]*x[2] + k*abs(x[5])*x[5] + m*g
            res[5] = x[1]^2 + x[2]^2 - L^2
            res[6] = x[4]*x[1] + x[5]*x[2]
            # Angle of pendulum
            res[7] = x[7] - atan(x[1] / -x[2])
            # Sensitivity equations for g
            res[8]  = dx[8] - x[11] + 2x[8]dx[6] + 2dx[13]x[1]
            res[9]  = dx[9] - x[12] + 2x[9]dx[6] + 2dx[13]x[2]
            res[10] = 2k*x[11]*abs(x[4]) + m*dx[11] - x[8]dx[3] - dx[10]x[1]
            res[11] = 2k*x[12]*abs(x[5]) + m*dx[12] - x[9]dx[3] - dx[10]x[2] + m
            res[12] = 2x[8]x[1] + 2x[9]x[2]
            res[13] = x[8]x[4] + x[11]x[1] + x[9]x[5] + x[12]x[2]
            # Sensitivity of angle of pendulum to L
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[8]*x[2])/(L^2)

            nothing
        end

        # Finding consistent initial conditions
        # Initial values, the pendulum starts at rest
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        x1_0 = L * sin(Φ)
        x2_0 = -L * cos(Φ)
        dx3_0 = m*g/x2_0
        dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m

        pend0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        # pend_dvar = vcat(fill(true, 6))
        sg0  = zeros(7)
        dsg0 = vcat(zeros(5), [-1., 0.])
        # if x1_0 != 0.0
        #     t0  = vcat([L/(2x1_0), L/(2x2_0)], zeros(4), [(x1_0^2-x2_0^2)/(2L*x1_0*x2_0)])
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(2x1_0*x2_0)], zeros(4))
        # else
        #     t0  = vcat([0.0, L/x2_0], zeros(5))
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(x2_0^2)], zeros(4))
        # end

        x0  = vcat(pend0, sg0)
        dx0 = vcat(dpend0, dsg0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false])

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

# NOTE: Doesn't actually support changing the value of pi
function pendulum_sensitivity_deb(Φ::Float64, u::Function, w::Function, θ::Vector{Float64})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        my_ind = 2
        i = my_ind

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - x[4] + 2dx[6]*x[1]
            res[2] = dx[2] - x[5] + 2dx[6]*x[2]
            res[3] = m*dx[4] - dx[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
            res[4] = m*dx[5] - dx[3]*x[2] + k*abs(x[5])*x[5] + m*g
            res[5] = x[1]^2 + x[2]^2 - L^2
            res[6] = x[4]*x[1] + x[5]*x[2]
            # Angle of pendulum
            res[7] = x[7] - atan(x[1] / -x[2])
            # Sensitivity equations for p1                      # here
            res[8]  = dx[8] - x[11] + 2x[8]dx[6] + 2dx[13]x[1]  + Int(i==1)
            res[9]  = dx[9] - x[12] + 2x[9]dx[6] + 2dx[13]x[2] + Int(i==2)
            res[10] = 2k*x[11]*abs(x[4]) + m*dx[11] - x[8]dx[3] - dx[10]x[1] + Int(i==3)
            res[11] = 2k*x[12]*abs(x[5]) + m*dx[12] - x[9]dx[3] - dx[10]x[2] + Int(i==4)
            res[12] = 2x[8]x[1] + 2x[9]x[2] + Int(i==5)
            res[13] = x[8]x[4] + x[11]x[1] + x[9]x[5] + x[12]x[2] + Int(i==6)
            # Sensitivity of angle of pendulum to L
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[8]*x[2])/(L^2) + Int(i==7)

            nothing
        end

        # # ORIGINAL
        # # Finding consistent initial conditions
        # # Initial values, the pendulum starts at rest
        # u0 = u(0.0)[1]
        # w0 = w(0.0)[1]
        # x1_0 = L * sin(Φ)
        # x2_0 = -L * cos(Φ)
        # dx3_0 = m*g/x2_0
        # dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m
        # pend0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        # dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        # ALLOWING NON-ZERO P
        pvec = zeros(7); pvec[i] = 0.
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        L_new = sqrt(L^2 - pvec[5])
        x1_0 = L_new * sin(Φ)
        x2_0 = -L_new * cos(Φ)
        x5_0 = -pvec[6]/x2_0
        pend0 = vcat([x1_0, x2_0], zeros(2), [x5_0, 0.0, atan(x1_0 / -x2_0)-pvec[7]])
        dx2_0 = x5_0 - pvec[2]
        dx4_0 = (u0 + w0^2 - pvec[3])/m
        dx5_0 = (-k*abs(x5_0)*x5_0 - m*g - pvec[4])/m
        dpend0 = [-pvec[1], dx2_0, 0.0, dx4_0, dx5_0, 0.0, 0.0]

        s1_0 = 0.0#equivalent -Int(i==5)*x1_0/(2*L^2)
        s2_0 = -Int(i==5)/(2x2_0)  # Very different, is it right?
        s5_0 = -Int(i==6)/(x2_0)
        s0  = [s1_0, s2_0, 0.0, 0.0, s5_0, 0.0, -Int(i==7)]
        ds1_0 = -Int(i==1)
        ds2_0 = -Int(i==2) + s5_0
        ds4_0 = -Int(i==3)/m#Almost equivalent : (Int(i==3)+dx3_0*s1_0)/m
        # ds5_0 = (-Int(i==4)+dx3_0*s2_0)/m
        ds5_0 = (-Int(i==4)+0.0*s2_0)/m
        ds7_0 = Int(i==1)*x2_0/(L^2)#Not even close: (x1_0*(Int(i==1)+s4_0) - x2_0*(Int(i==2)+s5_0))/(L^2)
        ds0 = [ds1_0, ds2_0, 0.0, ds4_0, ds5_0, 0.0, ds7_0]

        # res should be:
        x = vcat(pend0, s0)
        dx = vcat(dpend0, ds0)
        my_res = dx[8] - x[11] + 2x[8]dx[6] + 2dx[13]x[1]  + dx[1] - x[4] + 2dx[6]*x[1]

        # if x1_0 != 0.0
        #     t0  = vcat([L/(2x1_0), L/(2x2_0)], zeros(4), [(x1_0^2-x2_0^2)/(2L*x1_0*x2_0)])
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(2x1_0*x2_0)], zeros(4))
        # else
        #     t0  = vcat([0.0, L/x2_0], zeros(5))
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(x2_0^2)], zeros(4))
        # end

        x0  = vcat(pend0, s0)
        dx0 = vcat(dpend0, ds0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false])

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)
        # @info "r0 is $r0 ! debsens"

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

# NOTE: Doesn't actually support changing the value of pi
function pendulum_sensitivity_deb_0p01(Φ::Float64, u::Function, w::Function, θ::Vector{Float64})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        my_ind = 2
        i = my_ind

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - x[4] + 2dx[6]*x[1] + Int(i==1)*0.01
            res[2] = dx[2] - x[5] + 2dx[6]*x[2] + Int(i==2)*0.01
            res[3] = m*dx[4] - dx[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2 + Int(i==3)*0.01
            res[4] = m*dx[5] - dx[3]*x[2] + k*abs(x[5])*x[5] + m*g + Int(i==4)*0.01
            res[5] = x[1]^2 + x[2]^2 - L^2 + Int(i==5)*0.01
            res[6] = x[4]*x[1] + x[5]*x[2] + Int(i==6)*0.01
            # Angle of pendulum
            res[7] = x[7] - atan(x[1] / -x[2]) + Int(i==7)*0.01
            # Sensitivity equations for p1                      # here
            res[8]  = dx[8] - x[11] + 2x[8]dx[6] + 2dx[13]x[1]  + Int(i==1)
            res[9]  = dx[9] - x[12] + 2x[9]dx[6] + 2dx[13]x[2] + Int(i==2)
            res[10] = 2k*x[11]*abs(x[4]) + m*dx[11] - x[8]dx[3] - dx[10]x[1] + Int(i==3)
            res[11] = 2k*x[12]*abs(x[5]) + m*dx[12] - x[9]dx[3] - dx[10]x[2] + Int(i==4)
            res[12] = 2x[8]x[1] + 2x[9]x[2] + Int(i==5)
            res[13] = x[8]x[4] + x[11]x[1] + x[9]x[5] + x[12]x[2] + Int(i==6)
            # Sensitivity of angle of pendulum to L
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[8]*x[2])/(L^2) + Int(i==7)

            nothing
        end

        # # ORIGINAL
        # # Finding consistent initial conditions
        # # Initial values, the pendulum starts at rest
        # u0 = u(0.0)[1]
        # w0 = w(0.0)[1]
        # x1_0 = L * sin(Φ)
        # x2_0 = -L * cos(Φ)
        # dx3_0 = m*g/x2_0
        # dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m
        # pend0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        # dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        # ALLOWING NON-ZERO P
        pvec = zeros(7); pvec[i] = 0.01
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        L_new = sqrt(L^2 - pvec[5])
        x1_0 = L_new * sin(Φ)
        x2_0 = -L_new * cos(Φ)
        x5_0 = -pvec[6]/x2_0
        pend0 = vcat([x1_0, x2_0], zeros(2), [x5_0, 0.0, atan(x1_0 / -x2_0)-pvec[7]])
        dx2_0 = x5_0 - pvec[2]
        dx4_0 = (u0 + w0^2 - pvec[3])/m
        dx5_0 = (-k*abs(x5_0)*x5_0 - m*g - pvec[4])/m
        dpend0 = [-pvec[1], dx2_0, 0.0, dx4_0, dx5_0, 0.0, 0.0]

        s1_0 = 0.0#equivalent -Int(i==5)*x1_0/(2*L^2)
        s2_0 = -Int(i==5)/(2x2_0)  # Very different, is it right?
        s5_0 = -Int(i==6)/(x2_0)
        s0  = [s1_0, s2_0, 0.0, 0.0, s5_0, 0.0, -Int(i==7)]
        ds1_0 = -Int(i==1)
        ds2_0 = -Int(i==2) + s5_0
        ds4_0 = -Int(i==3)/m#Almost equivalent : (Int(i==3)+dx3_0*s1_0)/m
        # ds5_0 = (-Int(i==4)+dx3_0*s2_0)/m
        ds5_0 = (-Int(i==4)+0.0*s2_0)/m
        ds7_0 = Int(i==1)*x2_0/(L^2)#Not even close: (x1_0*(Int(i==1)+s4_0) - x2_0*(Int(i==2)+s5_0))/(L^2)
        ds0 = [ds1_0, ds2_0, 0.0, ds4_0, ds5_0, 0.0, ds7_0]
        # s1_0 = -Int(i==5)*x1_0/(2*L^2)
        # s2_0 = Int(i==5)*x2_0/(2*L^2)
        # s4_0 = Int(i==6)/(2x1_0)
        # s5_0 = Int(i==6)/(2x2_0)
        # s0  = [s1_0, s2_0, 0.0, s4_0, s5_0, 0.0, -Int(i==7)]
        # ds1_0 = Int(i==1) + s4_0
        # ds2_0 = Int(i==2) + s5_0
        # ds4_0 = (Int(i==3)+dx3_0*s1_0)/m
        # ds5_0 = (Int(i==4)+dx3_0*s2_0)/m
        # ds7_0 = (x1_0*(Int(i==1)+s4_0) - x2_0*(Int(i==2)+s5_0))/(L^2)
        # ds0 = [ds1_0, ds2_0, 0.0, ds4_0, ds5_0, 0.0, ds7_0]

        # res should be:
        x = vcat(pend0, s0)
        dx = vcat(dpend0, ds0)
        my_res = dx[8] - x[11] + 2x[8]dx[6] + 2dx[13]x[1]  + dx[1] - x[4] + 2dx[6]*x[1]

        # if x1_0 != 0.0
        #     t0  = vcat([L/(2x1_0), L/(2x2_0)], zeros(4), [(x1_0^2-x2_0^2)/(2L*x1_0*x2_0)])
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(2x1_0*x2_0)], zeros(4))
        # else
        #     t0  = vcat([0.0, L/x2_0], zeros(5))
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(x2_0^2)], zeros(4))
        # end

        x0  = vcat(pend0, s0)
        dx0 = vcat(dpend0, ds0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false])

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)
        # @info "r0 is $r0, for this deb one"

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_sensitivity_Lk(Φ::Float64, u::Function, w::Function, θ::Vector{Float64})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - x[4] + 2dx[6]*x[1]
            res[2] = dx[2] - x[5] + 2dx[6]*x[2]
            res[3] = m*dx[4] - dx[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
            res[4] = m*dx[5] - dx[3]*x[2] + k*abs(x[5])*x[5] + m*g
            res[5] = x[1]^2 + x[2]^2 - L^2
            res[6] = x[4]*x[1] + x[5]*x[2]
            # Angle of pendulum
            res[7] = x[7] - atan(x[1] / -x[2])
            # Sensitivity equations for L
            res[8]  = dx[8] - x[11] + 2x[8]dx[6] + 2dx[13]x[1]
            res[9]  = dx[9] - x[12] + 2x[9]dx[6] + 2dx[13]x[2]
            res[10] = 2k*x[11]*abs(x[4]) + m*dx[11] - x[8]dx[3] - dx[10]x[1]
            res[11] = 2k*x[12]*abs(x[5]) + m*dx[12] - x[9]dx[3] - dx[10]x[2]
            res[12] = 2x[8]x[1] + 2x[9]x[2] - 2L
            res[13] = x[8]x[4] + x[11]x[1] + x[9]x[5] + x[12]x[2]
            # Sensitivity of angle of pendulum to L
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[8]*x[2])/(L^2)
            # Sensitivity equations for k
            res[15]  = -x[18] + dx[15] + 2x[1]*dx[20] + 2x[15]*dx[6]
            res[16]  = -x[19] + dx[16] + 2x[2]*dx[20] + 2x[16]*dx[6]
            res[17] = abs(x[4])*x[4] + 2k*x[18]*abs(x[4]) - x[1]*dx[17] + m*dx[18] - x[15]*dx[3]
            res[18] = abs(x[5])*x[5] + 2k*x[19]*abs(x[5]) - x[2]*dx[17] + m*dx[19] - x[16]*dx[3]
            res[19] = 2x[15]*x[1] + 2x[16]*x[2]
            res[20] = x[18]*x[1] + x[19]*x[2] + x[15]*x[4] + x[16]*x[5]
            # Sensitivity of angle of pendulum to k
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[21] = x[21] - (x[1]*x[16] - x[15]*x[2])/(L^2)

            # These equations sensitivity are written on the intuitive form. To
            # obtain form correct for this funciton, do the following replacements:
            # s[1] -> x[8],  s[2] -> x[9],  s[3] -> x[10]
            # s[4] -> x[11], s[5] -> x[12], s[6] -> x[13]
            # t[1] -> x[15],  t[2] -> x[16],  t[3] -> x[17]
            # t[4] -> x[18], t[5] -> x[19], t[6] -> x[20]
            # and the corresponding replacements for sp/dx and tp/dx

            # res[8]  = -s[4] + sp[1] + 2x[1]*sp[6] + 2s[1]*dx[6]
            # res[9]  = -s[5] + sp[2] + 2x[2]*sp[6] + 2s[2]*dx[6]
            # res[10]  = abs(x[4])*x[4] + 2k*s[4]*abs(x[4]) - x[1]*sp[3] + m*sp[4] - s[1]*dx[3]
            # res[11] = abs(x[5])*x[5] + 2k*s[5]*abs(x[5]) - x[2]*sp[3] + m*sp[5] - s[2]*dx[3]
            # res[12] = -2s[1]*x[1] - 2s[2]*x[2]
            # res[13] = s[4]*x[1] + s[5]*x[2] + s[1]*x[4] + s[2]*x[5]
            # res[14] = x[14] - (x[1]*s[2] - x[2]*s[1])/(L^2)

            # MATLAB form
            # res[15]  = tp[1] - t[4] + 2t[1]dx[6] + 2tp[6]x[1]
            # res[16]  = tp[2] - t[5] + 2t[2]dx[6] + 2tp[6]x[2]
            # res[17] = 2k*t[4]*abs(x[4]) + m*tp[4] - t[1]dx[3] - tp[3]x[1]
            # res[18] = 2k*t[5]*abs(x[5]) + m*tp[5] - t[2]dx[3] - tp[3]x[2]
            # res[19] = 2L -2t[1]x[1] - 2t[2]x[2]
            # res[20] = t[1]x[4] + t[4]x[1] + t[2]x[5] + t[5]x[2]
            # res[21] = (x[1]*t[2] - x[2]*t[1])/(L^2)

            # Hand-derived form
            # res[15]  = -t[4] + tp[1] + 2x[1]*tp[6] + 2t[1]*dx[6]
            # res[16]  = -t[5] + tp[2] + 2x[2]*tp[6] + 2t[2]*dx[6]
            # res[17] = 2k*t[4]*abs(x[4]) - x[1]*tp[3] + m*tp[4] - t[1]*dx[3]
            # res[18] = 2k*t[5]*abs(x[5]) - x[2]*tp[3] + m*tp[5] - t[2]*dx[3]
            # res[19] = -2t[1]*x[1] - 2t[2]*x[2] + 2*L
            # res[20] = t[4]*x[1] + t[5]*x[2] + t[1]*x[4] + t[2]*x[5]

            nothing
        end

        # Finding consistent initial conditions
        # Initial values, the pendulum starts at rest
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        x1_0 = L * sin(Φ)
        x2_0 = -L * cos(Φ)
        dx3_0 = m*g/x2_0
        dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m

        pend0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        # pend_dvar = vcat(fill(true, 6))
        s0  = zeros(7)
        sp0 = zeros(7)
        t0 = vcat([sin(Φ), -cos(Φ)], zeros(5))
        tp0 = vcat([0.,0.], [-dx3_0/L], zeros(4))
        # if x1_0 != 0.0
        #     t0  = vcat([L/(2x1_0), L/(2x2_0)], zeros(4), [(x1_0^2-x2_0^2)/(2L*x1_0*x2_0)])
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(2x1_0*x2_0)], zeros(4))
        # else
        #     t0  = vcat([0.0, L/x2_0], zeros(5))
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(x2_0^2)], zeros(4))
        # end

        x0  = vcat(pend0, s0, t0)
        dx0 = vcat(dpend0, sp0, tp0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false], fill(true, 6), [false])

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_sensitivity_full(Φ::Float64, u::Function, w::Function, θ::Vector{Float64})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - x[4] + 2dx[6]*x[1]
            res[2] = dx[2] - x[5] + 2dx[6]*x[2]
            res[3] = m*dx[4] - dx[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
            res[4] = m*dx[5] - dx[3]*x[2] + k*abs(x[5])*x[5] + m*g
            res[5] = x[1]^2 + x[2]^2 - L^2
            res[6] = x[4]*x[1] + x[5]*x[2]
            # Angle of pendulum
            res[7] = x[7] - atan(x[1] / -x[2])
            # Sensitivity with respect to m
            res[8]  = -x[11] + dx[8] + 2x[1]*dx[13] + 2x[8]*dx[6]
            res[9]  = -x[12] + dx[9] + 2x[2]*dx[13] + 2x[9]*dx[6]
            res[10] = 2k*x[11]*abs(x[4]) - x[1]*dx[10] + m*dx[11] - x[8]*dx[3] + dx[4]
            res[11] = 2k*x[12]*abs(x[5]) - x[2]*dx[10] + m*dx[12] - x[9]*dx[3] + g + dx[5]
            res[12] = -2x[8]*x[1] - 2x[9]*x[2]
            res[13] = x[11]*x[1] + x[12]*x[2] + x[8]*x[4] + x[9]*x[5]
            res[14] = x[14] - (x[1]*x[9] - x[2]*x[8])/(L^2)

            # Sensitivity with respect to L
            res[15]  = -x[18] + dx[15] + 2x[1]*dx[20] + 2x[15]*dx[6]
            res[16]  = -x[19] + dx[16] + 2x[2]*dx[20] + 2x[16]*dx[6]
            res[17] = 2k*x[18]*abs(x[4]) - x[1]*dx[17] + m*dx[18] - x[15]*dx[3]
            res[18] = 2k*x[19]*abs(x[5]) - x[2]*dx[17] + m*dx[19] - x[16]*dx[3]
            res[19] = -2x[15]*x[1] - 2x[16]*x[2] + 2L
            res[20] = x[18]*x[1] + x[19]*x[2] + x[15]*x[4] + x[16]*x[5]
            res[21] = x[21] - (x[1]*x[16] - x[2]*x[15])/(L^2)

            # Sensitivity with respect to g
            res[22]  = -x[25] + dx[22] + 2x[1]*dx[27] + 2x[22]*dx[6]
            res[23]  = -x[26] + dx[23] + 2x[2]*dx[27] + 2x[23]*dx[6]
            res[24] = 2k*x[25]*abs(x[4]) - x[1]*dx[24] + m*dx[25] - x[22]*dx[3]
            res[25] = 2k*x[26]*abs(x[5]) - x[2]*dx[24] + m*dx[26] - x[23]*dx[3] + m
            res[26] = -2x[22]*x[1] - 2x[23]*x[2]
            res[27] = x[25]*x[1] + x[26]*x[2] + x[22]*x[4] + x[23]*x[5]
            res[28] = x[28] - (x[1]*x[23] - x[2]*x[22])/(L^2)

            # Sensitivity with respect to k
            res[29]  = -x[32] + dx[29] + 2x[1]*dx[34] + 2x[29]*dx[6]
            res[30]  = -x[33] + dx[30] + 2x[2]*dx[34] + 2x[30]*dx[6]
            res[31] = 2k*x[32]*abs(x[4]) - x[1]*dx[31] + m*dx[32] - x[29]*dx[3] + abs(x[4])x[4]
            res[32] = 2k*x[33]*abs(x[5]) - x[2]*dx[31] + m*dx[33] - x[30]*dx[3] + abs(x[5])x[5]
            res[33] = -2x[29]*x[1] - 2x[30]*x[2]
            res[34] = x[32]*x[1] + x[33]*x[2] + x[29]*x[4] + x[30]*x[5]
            res[35] = x[35] - (x[1]*x[30] - x[2]*x[29])/(L^2)

            # The default template used to generate these equations was:
            # res[8]  = -s[4] + sp[1] + 2x[1]*sp[6] + 2s[1]*dx[6]
            # res[9]  = -s[5] + sp[2] + 2x[2]*sp[6] + 2s[2]*dx[6]
            # res[10]  = 2k*s[4]*abs(x[4]) - x[1]*sp[3] + m*sp[4] - s[1]*dx[3]
            # res[11] = 2k*s[5]*abs(x[5]) - x[2]*sp[3] + m*sp[5] - s[2]*dx[3]
            # res[12] = -2s[1]*x[1] - 2s[2]*x[2]
            # res[13] = s[4]*x[1] + s[5]*x[2] + s[1]*x[4] + s[2]*x[5]
            # res[14] = x[14] - (x[1]*s[2] - x[2]*s[1])/(L^2)
            # Note that these edxressions don't include parameter-specific terms

            # To obtain the equations written on the form suitable for this
            # function, use the following substitutions:
            # sm[1] -> x[8],  sm[2] -> x[9],  sm[3] -> x[10]
            # sm[4] -> x[11], sm[5] -> x[12], sm[6] -> x[13]
            # sL[1] -> x[15],  sL[2] -> x[16],  sL[3] -> x[17]
            # sL[4] -> x[18], sL[5] -> x[19], sL[6] -> x[20]
            # sg[1] -> x[22],  sg[2] -> x[23],  sg[3] -> x[24]
            # sg[4] -> x[25], sg[5] -> x[26], sg[6] -> x[27]
            # sk[1] -> x[29],  sk[2] -> x[30],  sk[3] -> x[31]
            # sk[4] -> x[32], sk[5] -> x[33], sk[6] -> x[34]

            nothing
        end

        # Finding consistent initial conditions
        # Initial values, the pendulum starts at rest
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        x1_0 = L * sin(Φ)
        x2_0 = -L * cos(Φ)
        dx3_0 = m*g/x2_0
        dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m

        pend0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        sm0  = zeros(7)
        dsm0 = vcat(zeros(3), [-dx4_0/m, -g/m, 0., 0.])
        sL0  = vcat([x1_0/L, x2_0/L], zeros(5))
        dsL0 = vcat([0.,0., -dx3_0/L], zeros(4))
        sg0  = zeros(7)
        sgp0 = vcat(zeros(4), [-1., 0., 0.])
        sk0  = zeros(7)
        dsk0 = zeros(7)
        # t0 = vcat([sin(Φ), -cos(Φ)], zeros(5))
        # tp0 = vcat([0.,0.], [-dx3_0/L], zeros(4))
        # if x1_0 != 0.0
        #     t0  = vcat([L/(2x1_0), L/(2x2_0)], zeros(4), [(x1_0^2-x2_0^2)/(2L*x1_0*x2_0)])
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(2x1_0*x2_0)], zeros(4))
        # else
        #     t0  = vcat([0.0, L/x2_0], zeros(5))
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(x2_0^2)], zeros(4))
        # end

        x0  = vcat(pend0, sm0, sL0, sg0, sk0)
        dx0 = vcat(dpend0, dsm0, dsL0, sgp0, dsk0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = repeat(vcat(fill(true, 6), [false]), 5)

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_sensitivity_sans_g(Φ::Float64, u::Function, w::Function, θ::Vector{Float64})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - x[4] + 2dx[6]*x[1]
            res[2] = dx[2] - x[5] + 2dx[6]*x[2]
            res[3] = m*dx[4] - dx[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
            res[4] = m*dx[5] - dx[3]*x[2] + k*abs(x[5])*x[5] + m*g
            res[5] = x[1]^2 + x[2]^2 - L^2
            res[6] = x[4]*x[1] + x[5]*x[2]
            # Angle of pendulum
            res[7] = x[7] - atan(x[1] / -x[2])
            # Sensitivity with respect to m
            res[8]  = -x[11] + dx[8] + 2x[1]*dx[13] + 2x[8]*dx[6]
            res[9]  = -x[12] + dx[9] + 2x[2]*dx[13] + 2x[9]*dx[6]
            res[10] = 2k*x[11]*abs(x[4]) - x[1]*dx[10] + m*dx[11] - x[8]*dx[3] + dx[4]
            res[11] = 2k*x[12]*abs(x[5]) - x[2]*dx[10] + m*dx[12] - x[9]*dx[3] + g + dx[5]
            res[12] = -2x[8]*x[1] - 2x[9]*x[2]
            res[13] = x[11]*x[1] + x[12]*x[2] + x[8]*x[4] + x[9]*x[5]
            res[14] = x[14] - (x[1]*x[9] - x[2]*x[8])/(L^2)

            # Sensitivity with respect to L
            res[15]  = -x[18] + dx[15] + 2x[1]*dx[20] + 2x[15]*dx[6]
            res[16]  = -x[19] + dx[16] + 2x[2]*dx[20] + 2x[16]*dx[6]
            res[17] = 2k*x[18]*abs(x[4]) - x[1]*dx[17] + m*dx[18] - x[15]*dx[3]
            res[18] = 2k*x[19]*abs(x[5]) - x[2]*dx[17] + m*dx[19] - x[16]*dx[3]
            res[19] = -2x[15]*x[1] - 2x[16]*x[2] + 2L
            res[20] = x[18]*x[1] + x[19]*x[2] + x[15]*x[4] + x[16]*x[5]
            res[21] = x[21] - (x[1]*x[16] - x[2]*x[15])/(L^2)

            # Sensitivity with respect to k
            res[22]  = -x[25] + dx[22] + 2x[1]*dx[27] + 2x[22]*dx[6]
            res[23]  = -x[26] + dx[23] + 2x[2]*dx[27] + 2x[23]*dx[6]
            res[24] = 2k*x[25]*abs(x[4]) - x[1]*dx[24] + m*dx[25] - x[22]*dx[3] + abs(x[4])x[4]
            res[25] = 2k*x[26]*abs(x[5]) - x[2]*dx[24] + m*dx[26] - x[23]*dx[3] + abs(x[5])x[5]
            res[26] = -2x[22]*x[1] - 2x[23]*x[2]
            res[27] = x[25]*x[1] + x[26]*x[2] + x[22]*x[4] + x[23]*x[5]
            res[28] = x[28] - (x[1]*x[23] - x[2]*x[22])/(L^2)

            # The default template used to generate these equations was:
            # res[8]  = -s[4] + sp[1] + 2x[1]*sp[6] + 2s[1]*dx[6]
            # res[9]  = -s[5] + sp[2] + 2x[2]*sp[6] + 2s[2]*dx[6]
            # res[10]  = 2k*s[4]*abs(x[4]) - x[1]*sp[3] + m*sp[4] - s[1]*dx[3]
            # res[11] = 2k*s[5]*abs(x[5]) - x[2]*sp[3] + m*sp[5] - s[2]*dx[3]
            # res[12] = -2s[1]*x[1] - 2s[2]*x[2]
            # res[13] = s[4]*x[1] + s[5]*x[2] + s[1]*x[4] + s[2]*x[5]
            # res[14] = x[14] - (x[1]*s[2] - x[2]*s[1])/(L^2)
            # Note that these edxressions don't include parameter-specific terms

            nothing
        end

        # Finding consistent initial conditions
        # Initial values, the pendulum starts at rest
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        x1_0 = L * sin(Φ)
        x2_0 = -L * cos(Φ)
        dx3_0 = m*g/x2_0
        dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m

        pend0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        sm0  = zeros(7)
        dsm0 = vcat(zeros(3), [-dx4_0/m, -g/m, 0., 0.])
        sL0  = vcat([x1_0/L, x2_0/L], zeros(5))
        dsL0 = vcat([0.,0., -dx3_0/L], zeros(4))
        sk0  = zeros(7)
        dsk0 = zeros(7)
        # t0 = vcat([sin(Φ), -cos(Φ)], zeros(5))
        # tp0 = vcat([0.,0.], [-dx3_0/L], zeros(4))
        # if x1_0 != 0.0
        #     t0  = vcat([L/(2x1_0), L/(2x2_0)], zeros(4), [(x1_0^2-x2_0^2)/(2L*x1_0*x2_0)])
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(2x1_0*x2_0)], zeros(4))
        # else
        #     t0  = vcat([0.0, L/x2_0], zeros(5))
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(x2_0^2)], zeros(4))
        # end

        x0  = vcat(pend0, sm0, sL0, sk0)
        dx0 = vcat(dpend0, dsm0, dsL0, dsk0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = repeat(vcat(fill(true, 6), [false]), 4)

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_sensitivity_sans_g_with_dist_sens_2(Φ::Float64, u::Function, w::Function, θ::Vector{Float64})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - x[4] + 2dx[6]*x[1]
            res[2] = dx[2] - x[5] + 2dx[6]*x[2]
            res[3] = m*dx[4] - dx[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
            res[4] = m*dx[5] - dx[3]*x[2] + k*abs(x[5])*x[5] + m*g
            res[5] = x[1]^2 + x[2]^2 - L^2
            res[6] = x[4]*x[1] + x[5]*x[2]
            # Angle of pendulum
            res[7] = x[7] - atan(x[1] / -x[2])
            # Sensitivity with respect to m
            res[8]  = -x[11] + dx[8] + 2x[1]*dx[13] + 2x[8]*dx[6]
            res[9]  = -x[12] + dx[9] + 2x[2]*dx[13] + 2x[9]*dx[6]
            res[10] = 2k*x[11]*abs(x[4]) - x[1]*dx[10] + m*dx[11] - x[8]*dx[3] + dx[4]
            res[11] = 2k*x[12]*abs(x[5]) - x[2]*dx[10] + m*dx[12] - x[9]*dx[3] + g + dx[5]
            res[12] = -2x[8]*x[1] - 2x[9]*x[2]
            res[13] = x[11]*x[1] + x[12]*x[2] + x[8]*x[4] + x[9]*x[5]
            res[14] = x[14] - (x[1]*x[9] - x[2]*x[8])/(L^2)

            # Sensitivity with respect to L
            res[15]  = -x[18] + dx[15] + 2x[1]*dx[20] + 2x[15]*dx[6]
            res[16]  = -x[19] + dx[16] + 2x[2]*dx[20] + 2x[16]*dx[6]
            res[17] = 2k*x[18]*abs(x[4]) - x[1]*dx[17] + m*dx[18] - x[15]*dx[3]
            res[18] = 2k*x[19]*abs(x[5]) - x[2]*dx[17] + m*dx[19] - x[16]*dx[3]
            res[19] = -2x[15]*x[1] - 2x[16]*x[2] + 2L
            res[20] = x[18]*x[1] + x[19]*x[2] + x[15]*x[4] + x[16]*x[5]
            res[21] = x[21] - (x[1]*x[16] - x[2]*x[15])/(L^2)

            # Sensitivity with respect to k
            res[22]  = -x[25] + dx[22] + 2x[1]*dx[27] + 2x[22]*dx[6]
            res[23]  = -x[26] + dx[23] + 2x[2]*dx[27] + 2x[23]*dx[6]
            res[24] = 2k*x[25]*abs(x[4]) - x[1]*dx[24] + m*dx[25] - x[22]*dx[3] + abs(x[4])x[4]
            res[25] = 2k*x[26]*abs(x[5]) - x[2]*dx[24] + m*dx[26] - x[23]*dx[3] + abs(x[5])x[5]
            res[26] = -2x[22]*x[1] - 2x[23]*x[2]
            res[27] = x[25]*x[1] + x[26]*x[2] + x[22]*x[4] + x[23]*x[5]
            res[28] = x[28] - (x[1]*x[23] - x[2]*x[22])/(L^2)

            # Sensitivty wrt first disturbance parameter
            # (i.e. parameter corresponding to wt[2])
            res[29] = 2dx[6]x[29] - x[32] + dx[29] + 2x[1]dx[34]
            res[30] = 2dx[6]x[30] - x[33] + dx[30] + 2x[2]dx[34]
            res[31] = -dx[3]x[29] + 2k*abs(x[4])*x[32] - x[1]*dx[31] + m*dx[32] - 2*wt[1]*wt[2]
            res[32] = -dx[3]x[30] + 2k*abs(x[5])*x[33] - x[2]*dx[31] + m*dx[33]
            res[33] = 2x[1]x[29] + 2x[2]x[30]
            res[34] = x[4]x[29] + x[5]x[30] + x[1]x[32] + x[2]x[33]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[35] = x[35] - (x[1]*x[30] - x[2]*x[29])/(L^2)

            # Sensitivty wrt second disturbance parameter
            # (i.e. parameter corresponding to wt[3])
            res[36] = 2dx[6]x[36] - x[39] + dx[36] + 2x[1]dx[41]
            res[37] = 2dx[6]x[37] - x[40] + dx[37] + 2x[2]dx[41]
            res[38] = -dx[3]x[36] + 2k*abs(x[4])*x[39] - x[1]*dx[38] + m*dx[39] - 2*wt[1]*wt[3]
            res[39] = -dx[3]x[37] + 2k*abs(x[5])*x[40] - x[2]*dx[38] + m*dx[40]
            res[40] = 2x[1]x[36] + 2x[2]x[37]
            res[41] = x[4]x[36] + x[5]x[37] + x[1]x[39] + x[2]x[40]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[42] = x[42] - (x[1]*x[37] - x[2]*x[36])/(L^2)

            nothing
        end

        # Finding consistent initial conditions
        # Initial values, the pendulum starts at rest
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        x1_0 = L * sin(Φ)
        x2_0 = -L * cos(Φ)
        dx3_0 = m*g/x2_0
        dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m

        pend0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        sm0  = zeros(7)
        dsm0 = vcat(zeros(3), [-dx4_0/m, -g/m, 0., 0.])
        sL0  = vcat([x1_0/L, x2_0/L], zeros(5))
        dsL0 = vcat([0.,0., -dx3_0/L], zeros(4))
        sk0  = zeros(7)
        dsk0 = zeros(7)
        r0 = zeros(7)
        rp0 = zeros(7)
        # t0 = vcat([sin(Φ), -cos(Φ)], zeros(5))
        # tp0 = vcat([0.,0.], [-dx3_0/L], zeros(4))
        # if x1_0 != 0.0
        #     t0  = vcat([L/(2x1_0), L/(2x2_0)], zeros(4), [(x1_0^2-x2_0^2)/(2L*x1_0*x2_0)])
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(2x1_0*x2_0)], zeros(4))
        # else
        #     t0  = vcat([0.0, L/x2_0], zeros(5))
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(x2_0^2)], zeros(4))
        # end

        x0  = vcat(pend0, sm0, sL0, sk0, r0, r0)
        dx0 = vcat(dpend0, dsm0, dsL0, dsk0, rp0, rp0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = repeat(vcat(fill(true, 6), [false]), 6)

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_sensitivity_sans_g_with_dist_sens_3(Φ::Float64, u::Function, w::Function, θ::Vector{Float64})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - x[4] + 2dx[6]*x[1]
            res[2] = dx[2] - x[5] + 2dx[6]*x[2]
            res[3] = m*dx[4] - dx[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
            res[4] = m*dx[5] - dx[3]*x[2] + k*abs(x[5])*x[5] + m*g
            res[5] = x[1]^2 + x[2]^2 - L^2
            res[6] = x[4]*x[1] + x[5]*x[2]
            # Angle of pendulum
            res[7] = x[7] - atan(x[1] / -x[2])
            # Sensitivity with respect to m
            res[8]  = -x[11] + dx[8] + 2x[1]*dx[13] + 2x[8]*dx[6]
            res[9]  = -x[12] + dx[9] + 2x[2]*dx[13] + 2x[9]*dx[6]
            res[10] = 2k*x[11]*abs(x[4]) - x[1]*dx[10] + m*dx[11] - x[8]*dx[3] + dx[4]
            res[11] = 2k*x[12]*abs(x[5]) - x[2]*dx[10] + m*dx[12] - x[9]*dx[3] + g + dx[5]
            res[12] = -2x[8]*x[1] - 2x[9]*x[2]
            res[13] = x[11]*x[1] + x[12]*x[2] + x[8]*x[4] + x[9]*x[5]
            res[14] = x[14] - (x[1]*x[9] - x[2]*x[8])/(L^2)

            # Sensitivity with respect to L
            res[15]  = -x[18] + dx[15] + 2x[1]*dx[20] + 2x[15]*dx[6]
            res[16]  = -x[19] + dx[16] + 2x[2]*dx[20] + 2x[16]*dx[6]
            res[17] = 2k*x[18]*abs(x[4]) - x[1]*dx[17] + m*dx[18] - x[15]*dx[3]
            res[18] = 2k*x[19]*abs(x[5]) - x[2]*dx[17] + m*dx[19] - x[16]*dx[3]
            res[19] = -2x[15]*x[1] - 2x[16]*x[2] + 2L
            res[20] = x[18]*x[1] + x[19]*x[2] + x[15]*x[4] + x[16]*x[5]
            res[21] = x[21] - (x[1]*x[16] - x[2]*x[15])/(L^2)

            # Sensitivity with respect to k
            res[22]  = -x[25] + dx[22] + 2x[1]*dx[27] + 2x[22]*dx[6]
            res[23]  = -x[26] + dx[23] + 2x[2]*dx[27] + 2x[23]*dx[6]
            res[24] = 2k*x[25]*abs(x[4]) - x[1]*dx[24] + m*dx[25] - x[22]*dx[3] + abs(x[4])x[4]
            res[25] = 2k*x[26]*abs(x[5]) - x[2]*dx[24] + m*dx[26] - x[23]*dx[3] + abs(x[5])x[5]
            res[26] = -2x[22]*x[1] - 2x[23]*x[2]
            res[27] = x[25]*x[1] + x[26]*x[2] + x[22]*x[4] + x[23]*x[5]
            res[28] = x[28] - (x[1]*x[23] - x[2]*x[22])/(L^2)

            # Sensitivty wrt first disturbance parameter
            # (i.e. parameter corresponding to wt[2])
            res[29] = 2dx[6]x[29] - x[32] + dx[29] + 2x[1]dx[34]
            res[30] = 2dx[6]x[30] - x[33] + dx[30] + 2x[2]dx[34]
            res[31] = -dx[3]x[29] + 2k*abs(x[4])*x[32] - x[1]*dx[31] + m*dx[32] - 2*wt[1]*wt[2]
            res[32] = -dx[3]x[30] + 2k*abs(x[5])*x[33] - x[2]*dx[31] + m*dx[33]
            res[33] = 2x[1]x[29] + 2x[2]x[30]
            res[34] = x[4]x[29] + x[5]x[30] + x[1]x[32] + x[2]x[33]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[35] = x[35] - (x[1]*x[30] - x[2]*x[29])/(L^2)

            # Sensitivty wrt second disturbance parameter
            # (i.e. parameter corresponding to wt[3])
            res[36] = 2dx[6]x[36] - x[39] + dx[36] + 2x[1]dx[41]
            res[37] = 2dx[6]x[37] - x[40] + dx[37] + 2x[2]dx[41]
            res[38] = -dx[3]x[36] + 2k*abs(x[4])*x[39] - x[1]*dx[38] + m*dx[39] - 2*wt[1]*wt[3]
            res[39] = -dx[3]x[37] + 2k*abs(x[5])*x[40] - x[2]*dx[38] + m*dx[40]
            res[40] = 2x[1]x[36] + 2x[2]x[37]
            res[41] = x[4]x[36] + x[5]x[37] + x[1]x[39] + x[2]x[40]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[42] = x[42] - (x[1]*x[37] - x[2]*x[36])/(L^2)

            # Sensitivty wrt third disturbance parameter
            # (i.e. parameter corresponding to wt[4])
            res[43] = 2dx[6]x[43] - x[46] + dx[43] + 2x[1]dx[48]
            res[44] = 2dx[6]x[44] - x[47] + dx[44] + 2x[2]dx[48]
            res[45] = -dx[3]x[43] + 2k*abs(x[4])*x[46] - x[1]*dx[45] + m*dx[46] - 2*wt[1]*wt[4]
            res[46] = -dx[3]x[44] + 2k*abs(x[5])*x[47] - x[2]*dx[45] + m*dx[47]
            res[47] = 2x[1]x[43] + 2x[2]x[44]
            res[48] = x[4]x[43] + x[5]x[44] + x[1]x[46] + x[2]x[47]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[49] = x[49] - (x[1]*x[44] - x[2]*x[43])/(L^2)

            nothing
        end

        # Finding consistent initial conditions
        # Initial values, the pendulum starts at rest
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        x1_0 = L * sin(Φ)
        x2_0 = -L * cos(Φ)
        dx3_0 = m*g/x2_0
        dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m

        pend0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        sm0  = zeros(7)
        dsm0 = vcat(zeros(3), [-dx4_0/m, -g/m, 0., 0.])
        sL0  = vcat([x1_0/L, x2_0/L], zeros(5))
        dsL0 = vcat([0.,0., -dx3_0/L], zeros(4))
        sk0  = zeros(7)
        dsk0 = zeros(7)
        r0 = zeros(7)
        rp0 = zeros(7)
        # t0 = vcat([sin(Φ), -cos(Φ)], zeros(5))
        # tp0 = vcat([0.,0.], [-dx3_0/L], zeros(4))
        # if x1_0 != 0.0
        #     t0  = vcat([L/(2x1_0), L/(2x2_0)], zeros(4), [(x1_0^2-x2_0^2)/(2L*x1_0*x2_0)])
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(2x1_0*x2_0)], zeros(4))
        # else
        #     t0  = vcat([0.0, L/x2_0], zeros(5))
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(x2_0^2)], zeros(4))
        # end

        x0  = vcat(pend0, sm0, sL0, sk0, r0, r0, r0)
        dx0 = vcat(dpend0, dsm0, dsL0, dsk0, rp0, rp0, rp0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = repeat(vcat(fill(true, 6), [false]), 7)

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_sensitivity_sans_g_with_dist_sens_1(Φ::Float64, u::Function, w::Function, θ::Vector{Float64})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - x[4] + 2dx[6]*x[1]
            res[2] = dx[2] - x[5] + 2dx[6]*x[2]
            res[3] = m*dx[4] - dx[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
            res[4] = m*dx[5] - dx[3]*x[2] + k*abs(x[5])*x[5] + m*g
            res[5] = x[1]^2 + x[2]^2 - L^2
            res[6] = x[4]*x[1] + x[5]*x[2]
            # Angle of pendulum
            res[7] = x[7] - atan(x[1] / -x[2])
            # Sensitivity with respect to m
            res[8]  = -x[11] + dx[8] + 2x[1]*dx[13] + 2x[8]*dx[6]
            res[9]  = -x[12] + dx[9] + 2x[2]*dx[13] + 2x[9]*dx[6]
            res[10] = 2k*x[11]*abs(x[4]) - x[1]*dx[10] + m*dx[11] - x[8]*dx[3] + dx[4]
            res[11] = 2k*x[12]*abs(x[5]) - x[2]*dx[10] + m*dx[12] - x[9]*dx[3] + g + dx[5]
            res[12] = -2x[8]*x[1] - 2x[9]*x[2]
            res[13] = x[11]*x[1] + x[12]*x[2] + x[8]*x[4] + x[9]*x[5]
            res[14] = x[14] - (x[1]*x[9] - x[2]*x[8])/(L^2)

            # Sensitivity with respect to L
            res[15]  = -x[18] + dx[15] + 2x[1]*dx[20] + 2x[15]*dx[6]
            res[16]  = -x[19] + dx[16] + 2x[2]*dx[20] + 2x[16]*dx[6]
            res[17] = 2k*x[18]*abs(x[4]) - x[1]*dx[17] + m*dx[18] - x[15]*dx[3]
            res[18] = 2k*x[19]*abs(x[5]) - x[2]*dx[17] + m*dx[19] - x[16]*dx[3]
            res[19] = -2x[15]*x[1] - 2x[16]*x[2] + 2L
            res[20] = x[18]*x[1] + x[19]*x[2] + x[15]*x[4] + x[16]*x[5]
            res[21] = x[21] - (x[1]*x[16] - x[2]*x[15])/(L^2)

            # Sensitivity with respect to k
            res[22]  = -x[25] + dx[22] + 2x[1]*dx[27] + 2x[22]*dx[6]
            res[23]  = -x[26] + dx[23] + 2x[2]*dx[27] + 2x[23]*dx[6]
            res[24] = 2k*x[25]*abs(x[4]) - x[1]*dx[24] + m*dx[25] - x[22]*dx[3] + abs(x[4])x[4]
            res[25] = 2k*x[26]*abs(x[5]) - x[2]*dx[24] + m*dx[26] - x[23]*dx[3] + abs(x[5])x[5]
            res[26] = -2x[22]*x[1] - 2x[23]*x[2]
            res[27] = x[25]*x[1] + x[26]*x[2] + x[22]*x[4] + x[23]*x[5]
            res[28] = x[28] - (x[1]*x[23] - x[2]*x[22])/(L^2)

            # Sensitivty wrt first disturbance parameter
            # (i.e. parameter corresponding to wt[2])
            res[29] = 2dx[6]x[29] - x[32] + dx[29] + 2x[1]dx[34]
            res[30] = 2dx[6]x[30] - x[33] + dx[30] + 2x[2]dx[34]
            res[31] = -dx[3]x[29] + 2k*abs(x[4])*x[32] - x[1]*dx[31] + m*dx[32] - 2*wt[1]*wt[2]
            res[32] = -dx[3]x[30] + 2k*abs(x[5])*x[33] - x[2]*dx[31] + m*dx[33]
            res[33] = 2x[1]x[29] + 2x[2]x[30]
            res[34] = x[4]x[29] + x[5]x[30] + x[1]x[32] + x[2]x[33]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[35] = x[35] - (x[1]*x[30] - x[2]*x[29])/(L^2)

            nothing
        end

        # Finding consistent initial conditions
        # Initial values, the pendulum starts at rest
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        x1_0 = L * sin(Φ)
        x2_0 = -L * cos(Φ)
        dx3_0 = m*g/x2_0
        dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m

        pend0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        sm0  = zeros(7)
        dsm0 = vcat(zeros(3), [-dx4_0/m, -g/m, 0., 0.])
        sL0  = vcat([x1_0/L, x2_0/L], zeros(5))
        dsL0 = vcat([0.,0., -dx3_0/L], zeros(4))
        sk0  = zeros(7)
        dsk0 = zeros(7)
        r0 = zeros(7)
        rp0 = zeros(7)
        # t0 = vcat([sin(Φ), -cos(Φ)], zeros(5))
        # tp0 = vcat([0.,0.], [-dx3_0/L], zeros(4))
        # if x1_0 != 0.0
        #     t0  = vcat([L/(2x1_0), L/(2x2_0)], zeros(4), [(x1_0^2-x2_0^2)/(2L*x1_0*x2_0)])
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(2x1_0*x2_0)], zeros(4))
        # else
        #     t0  = vcat([0.0, L/x2_0], zeros(5))
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(x2_0^2)], zeros(4))
        # end

        x0  = vcat(pend0, sm0, sL0, sk0, r0)
        dx0 = vcat(dpend0, dsm0, dsL0, dsk0, rp0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = repeat(vcat(fill(true, 6), [false]), 5)

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_sensitivity_full_with_dist_sens_2(Φ::Float64, u::Function, w::Function, θ::Vector{Float64})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - x[4] + 2dx[6]*x[1]
            res[2] = dx[2] - x[5] + 2dx[6]*x[2]
            res[3] = m*dx[4] - dx[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
            res[4] = m*dx[5] - dx[3]*x[2] + k*abs(x[5])*x[5] + m*g
            res[5] = x[1]^2 + x[2]^2 - L^2
            res[6] = x[4]*x[1] + x[5]*x[2]
            # Angle of pendulum
            res[7] = x[7] - atan(x[1] / -x[2])
            # Sensitivity with respect to m
            res[8]  = -x[11] + dx[8] + 2x[1]*dx[13] + 2x[8]*dx[6]
            res[9]  = -x[12] + dx[9] + 2x[2]*dx[13] + 2x[9]*dx[6]
            res[10] = 2k*x[11]*abs(x[4]) - x[1]*dx[10] + m*dx[11] - x[8]*dx[3] + dx[4]
            res[11] = 2k*x[12]*abs(x[5]) - x[2]*dx[10] + m*dx[12] - x[9]*dx[3] + g + dx[5]
            res[12] = -2x[8]*x[1] - 2x[9]*x[2]
            res[13] = x[11]*x[1] + x[12]*x[2] + x[8]*x[4] + x[9]*x[5]
            res[14] = x[14] - (x[1]*x[9] - x[2]*x[8])/(L^2)

            # Sensitivity with respect to L
            res[15]  = -x[18] + dx[15] + 2x[1]*dx[20] + 2x[15]*dx[6]
            res[16]  = -x[19] + dx[16] + 2x[2]*dx[20] + 2x[16]*dx[6]
            res[17] = 2k*x[18]*abs(x[4]) - x[1]*dx[17] + m*dx[18] - x[15]*dx[3]
            res[18] = 2k*x[19]*abs(x[5]) - x[2]*dx[17] + m*dx[19] - x[16]*dx[3]
            res[19] = -2x[15]*x[1] - 2x[16]*x[2] + 2L
            res[20] = x[18]*x[1] + x[19]*x[2] + x[15]*x[4] + x[16]*x[5]
            res[21] = x[21] - (x[1]*x[16] - x[2]*x[15])/(L^2)

            # Sensitivity with respect to g
            res[22]  = -x[25] + dx[22] + 2x[1]*dx[27] + 2x[22]*dx[6]
            res[23]  = -x[26] + dx[23] + 2x[2]*dx[27] + 2x[23]*dx[6]
            res[24] = 2k*x[25]*abs(x[4]) - x[1]*dx[24] + m*dx[25] - x[22]*dx[3]
            res[25] = 2k*x[26]*abs(x[5]) - x[2]*dx[24] + m*dx[26] - x[23]*dx[3] + m
            res[26] = -2x[22]*x[1] - 2x[23]*x[2]
            res[27] = x[25]*x[1] + x[26]*x[2] + x[22]*x[4] + x[23]*x[5]
            res[28] = x[28] - (x[1]*x[23] - x[2]*x[22])/(L^2)

            # Sensitivity with respect to k
            res[29]  = -x[32] + dx[29] + 2x[1]*dx[34] + 2x[29]*dx[6]
            res[30]  = -x[33] + dx[30] + 2x[2]*dx[34] + 2x[30]*dx[6]
            res[31] = 2k*x[32]*abs(x[4]) - x[1]*dx[31] + m*dx[32] - x[29]*dx[3] + abs(x[4])x[4]
            res[32] = 2k*x[33]*abs(x[5]) - x[2]*dx[31] + m*dx[33] - x[30]*dx[3] + abs(x[5])x[5]
            res[33] = -2x[29]*x[1] - 2x[30]*x[2]
            res[34] = x[32]*x[1] + x[33]*x[2] + x[29]*x[4] + x[30]*x[5]
            res[35] = x[35] - (x[1]*x[30] - x[2]*x[29])/(L^2)

            # 22->36, 23->37, 24->38, 25->39, 26->40, 27->41, 28->42

            # DISTURBANCE SENSITIVITIES
            res[36] = 2dx[6]x[36] - x[39] + dx[36] + 2x[1]dx[41]
            res[37] = 2dx[6]x[37] - x[40] + dx[37] + 2x[2]dx[41]
            res[38] = -dx[3]x[36] + 2k*abs(x[4])*x[39] - x[1]*dx[38] + m*dx[39] - 2*wt[1]*wt[2]
            res[39] = -dx[3]x[37] + 2k*abs(x[5])*x[40] - x[2]*dx[38] + m*dx[40]
            res[40] = -2x[1]x[36] - 2x[2]x[37]
            res[41] = x[4]x[36] + x[5]x[37] + x[1]x[39] + x[2]x[40]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[42] = x[42] - (x[1]*x[37] - x[2]*x[36])/(L^2)

            # 29->43, 30->44, 31->45, 32->46, 33->47, 34->48, 35->49

            res[43] = 2dx[6]x[43] - x[46] + dx[43] + 2x[1]dx[48]
            res[44] = 2dx[6]x[44] - x[47] + dx[44] + 2x[2]dx[47]
            res[45] = -dx[3]x[43] + 2k*abs(x[4])*x[46] - x[1]*dx[45] + m*dx[46] - 2*wt[1]*wt[3]
            res[46] = -dx[3]x[44] + 2k*abs(x[5])*x[47] - x[2]*dx[45] + m*dx[47]
            res[47] = -2x[1]x[43] - 2x[2]x[44]
            # res[26] = 2L -2x[15]x[1] - 2x[16]x[2]
            res[48] = x[4]x[43] + x[5]x[44] + x[1]x[46] + x[2]x[47]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[49] = x[49] - (x[1]*x[44] - x[2]*x[43])/(L^2)

            # The default template used to generate these equations was:
            # res[8]  = -s[4] + sp[1] + 2x[1]*sp[6] + 2s[1]*dx[6]
            # res[9]  = -s[5] + sp[2] + 2x[2]*sp[6] + 2s[2]*dx[6]
            # res[10]  = 2k*s[4]*abs(x[4]) - x[1]*sp[3] + m*sp[4] - s[1]*dx[3]
            # res[11] = 2k*s[5]*abs(x[5]) - x[2]*sp[3] + m*sp[5] - s[2]*dx[3]
            # res[12] = -2s[1]*x[1] - 2s[2]*x[2]
            # res[13] = s[4]*x[1] + s[5]*x[2] + s[1]*x[4] + s[2]*x[5]
            # res[14] = x[14] - (x[1]*s[2] - x[2]*s[1])/(L^2)
            # Note that these edxressions don't include parameter-specific terms

            # To obtain the equations written on the form suitable for this
            # function, use the following substitutions:
            # sm[1] -> x[8],  sm[2] -> x[9],  sm[3] -> x[10]
            # sm[4] -> x[11], sm[5] -> x[12], sm[6] -> x[13]
            # sL[1] -> x[15],  sL[2] -> x[16],  sL[3] -> x[17]
            # sL[4] -> x[18], sL[5] -> x[19], sL[6] -> x[20]
            # sg[1] -> x[22],  sg[2] -> x[23],  sg[3] -> x[24]
            # sg[4] -> x[25], sg[5] -> x[26], sg[6] -> x[27]
            # sk[1] -> x[29],  sk[2] -> x[30],  sk[3] -> x[31]
            # sk[4] -> x[32], sk[5] -> x[33], sk[6] -> x[34]

            nothing
        end

        # Finding consistent initial conditions
        # Initial values, the pendulum starts at rest
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        x1_0 = L * sin(Φ)
        x2_0 = -L * cos(Φ)
        dx3_0 = m*g/x2_0
        dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m

        pend0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        sm0  = zeros(7)
        dsm0 = vcat(zeros(3), [-dx4_0/m, -g/m, 0., 0.])
        sL0  = vcat([x1_0/L, x2_0/L], zeros(5))
        dsL0 = vcat([0.,0., -dx3_0/L], zeros(4))
        sg0  = zeros(7)
        sgp0 = vcat(zeros(4), [-1., 0., 0.])
        sk0  = zeros(7)
        dsk0 = zeros(7)
        # t0 = vcat([sin(Φ), -cos(Φ)], zeros(5))
        # tp0 = vcat([0.,0.], [-dx3_0/L], zeros(4))
        # if x1_0 != 0.0
        #     t0  = vcat([L/(2x1_0), L/(2x2_0)], zeros(4), [(x1_0^2-x2_0^2)/(2L*x1_0*x2_0)])
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(2x1_0*x2_0)], zeros(4))
        # else
        #     t0  = vcat([0.0, L/x2_0], zeros(5))
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(x2_0^2)], zeros(4))
        # end

        x0  = vcat(pend0, sm0, sL0, sg0, sk0, zeros(7))
        dx0 = vcat(dpend0, dsm0, dsL0, sgp0, dsk0, zeros(7))
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = repeat(vcat(fill(true, 6), [false]), 6)

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_sensitivity_Lk_with_dist_sens_1(Φ::Float64, u::Function, w_comp::Function, θ::Vector{Float64})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        # NOTE: In this function w_comp is edxected to have two elements: the
        # first should just be the disturbance w, and the second the sensitivity
        # of the disturbance w to the disturbance model parameter

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w_comp(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - x[4] + 2dx[6]*x[1]
            res[2] = dx[2] - x[5] + 2dx[6]*x[2]
            res[3] = m*dx[4] - dx[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
            res[4] = m*dx[5] - dx[3]*x[2] + k*abs(x[5])*x[5] + m*g
            res[5] = x[1]^2 + x[2]^2 - L^2
            res[6] = x[4]*x[1] + x[5]*x[2]
            # Angle of pendulum
            res[7] = x[7] - atan(x[1] / -x[2])
            # Sensitivity equations for L
            res[8]  = dx[8] - x[11] + 2x[8]dx[6] + 2dx[13]x[1]
            res[9]  = dx[9] - x[12] + 2x[9]dx[6] + 2dx[13]x[2]
            res[10] = 2k*x[11]*abs(x[4]) + m*dx[11] - x[8]dx[3] - dx[10]x[1]
            res[11] = 2k*x[12]*abs(x[5]) + m*dx[12] - x[9]dx[3] - dx[10]x[2]
            res[12] = 2x[8]x[1] + 2x[9]x[2] - 2L
            res[13] = x[8]x[4] + x[11]x[1] + x[9]x[5] + x[12]x[2]
            # Sensitivity of angle of pendulum to L
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[8]*x[2])/(L^2)

            # Sensitivity equations for k
            res[15]  = -x[18] + dx[15] + 2x[1]*dx[20] + 2x[15]*dx[6]
            res[16]  = -x[19] + dx[16] + 2x[2]*dx[20] + 2x[16]*dx[6]
            res[17] = abs(x[4])*x[4] + 2k*x[18]*abs(x[4]) - x[1]*dx[17] + m*dx[18] - x[15]*dx[3]
            res[18] = abs(x[5])*x[5] + 2k*x[19]*abs(x[5]) - x[2]*dx[17] + m*dx[19] - x[16]*dx[3]
            res[19] = 2x[15]*x[1] + 2x[16]*x[2]
            res[20] = x[18]*x[1] + x[19]*x[2] + x[15]*x[4] + x[16]*x[5]
            # Sensitivity of angle of pendulum to k
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[21] = x[21] - (x[1]*x[16] - x[15]*x[2])/(L^2)

            # Sensitivty wrt first disturbance parameter
            # (i.e. parameter corresponding to wt[2])
            res[22] = 2dx[6]x[22] - x[25] + dx[22] + 2x[1]dx[27]
            res[23] = 2dx[6]x[23] - x[26] + dx[23] + 2x[2]dx[27]
            res[24] = -dx[3]x[22] + 2k*abs(x[4])*x[25] - x[1]*dx[24] + m*dx[25] - 2*wt[1]*wt[2]
            res[25] = -dx[3]x[23] + 2k*abs(x[5])*x[26] - x[2]*dx[24] + m*dx[26]
            res[26] = 2x[1]x[22] + 2x[2]x[23]
            res[27] = x[4]x[22] + x[5]x[23] + x[1]x[25] + x[2]x[26]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[28] = x[28] - (x[1]*x[23] - x[2]*x[22])/(L^2)

            nothing
        end

        # Finding consistent initial conditions
        # Initial values, the pendulum starts at rest
        u0 = u(0.0)[1]
        w0 = w_comp(0.0)[1]
        x1_0 = L * sin(Φ)
        x2_0 = -L * cos(Φ)
        dx3_0 = m*g/x2_0
        dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m

        pend0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        # pend_dvar = vcat(fill(true, 6))
        s0  = zeros(7)
        sp0 = zeros(7)
        t0 = vcat([sin(Φ), -cos(Φ)], zeros(5))
        tp0 = vcat([0.,0.], [-dx3_0/L], zeros(4))
        r0 = zeros(7)
        rp0 = zeros(7)
        # if x1_0 != 0.0
        #     t0  = vcat([L/(2x1_0), L/(2x2_0)], zeros(4), [(x1_0^2-x2_0^2)/(2L*x1_0*x2_0)])
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(2x1_0*x2_0)], zeros(4))
        # else
        #     t0  = vcat([0.0, L/x2_0], zeros(5))
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(x2_0^2)], zeros(4))
        # end

        x0  = vcat(pend0, t0, s0, r0)
        dx0 = vcat(dpend0, tp0, sp0, rp0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false], fill(true, 6), [false], fill(true, 6), [false])

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_sensitivity_Lk_with_dist_sens_2(Φ::Float64, u::Function, w_comp::Function, θ::Vector{Float64})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        # NOTE: In this function w_comp is edxected to have two elements: the
        # first should just be the disturbance w, and the second the sensitivity
        # of the disturbance w to the disturbance model parameter

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w_comp(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - x[4] + 2dx[6]*x[1]
            res[2] = dx[2] - x[5] + 2dx[6]*x[2]
            res[3] = m*dx[4] - dx[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
            res[4] = m*dx[5] - dx[3]*x[2] + k*abs(x[5])*x[5] + m*g
            res[5] = x[1]^2 + x[2]^2 - L^2
            res[6] = x[4]*x[1] + x[5]*x[2]
            # Angle of pendulum
            res[7] = x[7] - atan(x[1] / -x[2])
            # Sensitivity equations for L
            res[8]  = dx[8] - x[11] + 2x[8]dx[6] + 2dx[13]x[1]
            res[9]  = dx[9] - x[12] + 2x[9]dx[6] + 2dx[13]x[2]
            res[10] = 2k*x[11]*abs(x[4]) + m*dx[11] - x[8]dx[3] - dx[10]x[1]
            res[11] = 2k*x[12]*abs(x[5]) + m*dx[12] - x[9]dx[3] - dx[10]x[2]
            res[12] = 2x[8]x[1] + 2x[9]x[2] - 2L
            res[13] = x[8]x[4] + x[11]x[1] + x[9]x[5] + x[12]x[2]
            # Sensitivity of angle of pendulum to L
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[8]*x[2])/(L^2)

            # Sensitivity equations for k
            res[15]  = -x[18] + dx[15] + 2x[1]*dx[20] + 2x[15]*dx[6]
            res[16]  = -x[19] + dx[16] + 2x[2]*dx[20] + 2x[16]*dx[6]
            res[17] = abs(x[4])*x[4] + 2k*x[18]*abs(x[4]) - x[1]*dx[17] + m*dx[18] - x[15]*dx[3]
            res[18] = abs(x[5])*x[5] + 2k*x[19]*abs(x[5]) - x[2]*dx[17] + m*dx[19] - x[16]*dx[3]
            res[19] = 2x[15]*x[1] + 2x[16]*x[2]
            res[20] = x[18]*x[1] + x[19]*x[2] + x[15]*x[4] + x[16]*x[5]
            # Sensitivity of angle of pendulum to k
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[21] = x[21] - (x[1]*x[16] - x[15]*x[2])/(L^2)

            # Sensitivty wrt first disturbance parameter
            # (i.e. parameter corresponding to wt[2])
            res[22] = 2dx[6]x[22] - x[25] + dx[22] + 2x[1]dx[27]
            res[23] = 2dx[6]x[23] - x[26] + dx[23] + 2x[2]dx[27]
            res[24] = -dx[3]x[22] + 2k*abs(x[4])*x[25] - x[1]*dx[24] + m*dx[25] - 2*wt[1]*wt[2]
            res[25] = -dx[3]x[23] + 2k*abs(x[5])*x[26] - x[2]*dx[24] + m*dx[26]
            res[26] = 2x[1]x[22] + 2x[2]x[23]
            res[27] = x[4]x[22] + x[5]x[23] + x[1]x[25] + x[2]x[26]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[28] = x[28] - (x[1]*x[23] - x[2]*x[22])/(L^2)

            # Sensitivty wrt second disturbance parameter
            # (i.e. parameter corresponding to wt[3])
            res[29] = 2dx[6]x[29] - x[32] + dx[29] + 2x[1]dx[34]
            res[30] = 2dx[6]x[30] - x[33] + dx[30] + 2x[2]dx[34]
            res[31] = -dx[3]x[29] + 2k*abs(x[4])*x[32] - x[1]*dx[31] + m*dx[32] - 2*wt[1]*wt[3]
            res[32] = -dx[3]x[30] + 2k*abs(x[5])*x[33] - x[2]*dx[31] + m*dx[33]
            res[33] = 2x[1]x[29] + 2x[2]x[30]
            res[34] = x[4]x[29] + x[5]x[30] + x[1]x[32] + x[2]x[33]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[35] = x[35] - (x[1]*x[30] - x[2]*x[29])/(L^2)

            nothing
        end

        # Finding consistent initial conditions
        # Initial values, the pendulum starts at rest
        u0 = u(0.0)[1]
        w0 = w_comp(0.0)[1]
        x1_0 = L * sin(Φ)
        x2_0 = -L * cos(Φ)
        dx3_0 = m*g/x2_0
        dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m

        pend0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        # pend_dvar = vcat(fill(true, 6))
        s0  = zeros(7)
        sp0 = zeros(7)
        t0 = vcat([sin(Φ), -cos(Φ)], zeros(5))
        tp0 = vcat([0.,0.], [-dx3_0/L], zeros(4))
        r0 = zeros(7)
        rp0 = zeros(7)
        # if x1_0 != 0.0
        #     t0  = vcat([L/(2x1_0), L/(2x2_0)], zeros(4), [(x1_0^2-x2_0^2)/(2L*x1_0*x2_0)])
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(2x1_0*x2_0)], zeros(4))
        # else
        #     t0  = vcat([0.0, L/x2_0], zeros(5))
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(x2_0^2)], zeros(4))
        # end

        x0  = vcat(pend0, t0, s0, r0, r0)
        dx0 = vcat(dpend0, tp0, sp0, rp0, rp0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false], fill(true, 6), [false], fill(true, 6), [false], fill(true, 6), [false])

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_sensitivity_k_with_dist_sens_1(Φ::Float64, u::Function, w::Function, θ::Vector{Float64})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - x[4] + 2dx[6]*x[1]
            res[2] = dx[2] - x[5] + 2dx[6]*x[2]
            res[3] = m*dx[4] - dx[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
            res[4] = m*dx[5] - dx[3]*x[2] + k*abs(x[5])*x[5] + m*g
            res[5] = x[1]^2 + x[2]^2 - L^2
            res[6] = x[4]*x[1] + x[5]*x[2]
            # Angle of pendulum
            res[7] = x[7] - atan(x[1] / -x[2])

            # Sensitivity with respect to k
            res[8]  = -x[11] + dx[8] + 2x[1]*dx[13] + 2x[8]*dx[6]
            res[9]  = -x[12] + dx[9] + 2x[2]*dx[13] + 2x[9]*dx[6]
            res[10] = 2k*x[11]*abs(x[4]) - x[1]*dx[10] + m*dx[11] - x[8]*dx[3] + abs(x[4])x[4]
            res[11] = 2k*x[12]*abs(x[5]) - x[2]*dx[10] + m*dx[12] - x[9]*dx[3] + abs(x[5])x[5]
            res[12] = -2x[8]*x[1] - 2x[9]*x[2]
            res[13] = x[11]*x[1] + x[12]*x[2] + x[8]*x[4] + x[9]*x[5]
            res[14] = x[14] - (x[1]*x[9] - x[2]*x[8])/(L^2)

            # Sensitivty wrt first disturbance parameter
            # (i.e. parameter corresponding to wt[2])
            res[15] = 2dx[6]x[15] - x[18] + dx[15] + 2x[1]dx[20]
            res[16] = 2dx[6]x[16] - x[19] + dx[16] + 2x[2]dx[20]
            res[17] = -dx[3]x[15] + 2k*abs(x[4])*x[18] - x[1]*dx[17] + m*dx[18] - 2*wt[1]*wt[2]
            res[18] = -dx[3]x[16] + 2k*abs(x[5])*x[19] - x[2]*dx[17] + m*dx[19]
            res[19] = 2x[1]x[15] + 2x[2]x[16]
            res[20] = x[4]x[15] + x[5]x[16] + x[1]x[18] + x[2]x[19]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[21] = x[21] - (x[1]*x[16] - x[2]*x[15])/(L^2)

            nothing
        end

        # Finding consistent initial conditions
        # Initial values, the pendulum starts at rest
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        x1_0 = L * sin(Φ)
        x2_0 = -L * cos(Φ)
        dx3_0 = m*g/x2_0
        dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m

        pend0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        sk0  = zeros(7)
        dsk0 = zeros(7)
        r0 = zeros(7)
        rp0 = zeros(7)

        x0  = vcat(pend0, sk0, r0)
        dx0 = vcat(dpend0, dsk0, rp0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = repeat(vcat(fill(true, 6), [false]), 3)

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_sensitivity_k_with_dist_sens_2(Φ::Float64, u::Function, w::Function, θ::Vector{Float64})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - x[4] + 2dx[6]*x[1]
            res[2] = dx[2] - x[5] + 2dx[6]*x[2]
            res[3] = m*dx[4] - dx[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
            res[4] = m*dx[5] - dx[3]*x[2] + k*abs(x[5])*x[5] + m*g
            res[5] = x[1]^2 + x[2]^2 - L^2
            res[6] = x[4]*x[1] + x[5]*x[2]
            # Angle of pendulum
            res[7] = x[7] - atan(x[1] / -x[2])

            # Sensitivity with respect to k
            res[8]  = -x[11] + dx[8] + 2x[1]*dx[13] + 2x[8]*dx[6]
            res[9]  = -x[12] + dx[9] + 2x[2]*dx[13] + 2x[9]*dx[6]
            res[10] = 2k*x[11]*abs(x[4]) - x[1]*dx[10] + m*dx[11] - x[8]*dx[3] + abs(x[4])x[4]
            res[11] = 2k*x[12]*abs(x[5]) - x[2]*dx[10] + m*dx[12] - x[9]*dx[3] + abs(x[5])x[5]
            res[12] = -2x[8]*x[1] - 2x[9]*x[2]
            res[13] = x[11]*x[1] + x[12]*x[2] + x[8]*x[4] + x[9]*x[5]
            res[14] = x[14] - (x[1]*x[9] - x[2]*x[8])/(L^2)

            # Sensitivty wrt first disturbance parameter
            # (i.e. parameter corresponding to wt[2])
            res[15] = 2dx[6]x[15] - x[18] + dx[15] + 2x[1]dx[20]
            res[16] = 2dx[6]x[16] - x[19] + dx[16] + 2x[2]dx[20]
            res[17] = -dx[3]x[15] + 2k*abs(x[4])*x[18] - x[1]*dx[17] + m*dx[18] - 2*wt[1]*wt[2]
            res[18] = -dx[3]x[16] + 2k*abs(x[5])*x[19] - x[2]*dx[17] + m*dx[19]
            res[19] = 2x[1]x[15] + 2x[2]x[16]
            res[20] = x[4]x[15] + x[5]x[16] + x[1]x[18] + x[2]x[19]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[21] = x[21] - (x[1]*x[16] - x[2]*x[15])/(L^2)

            # Sensitivty wrt second disturbance parameter
            # (i.e. parameter corresponding to wt[3])
            res[22] = 2dx[6]x[22] - x[25] + dx[22] + 2x[1]dx[27]
            res[23] = 2dx[6]x[23] - x[26] + dx[23] + 2x[2]dx[27]
            res[24] = -dx[3]x[22] + 2k*abs(x[4])*x[25] - x[1]*dx[24] + m*dx[25] - 2*wt[1]*wt[3]
            res[25] = -dx[3]x[23] + 2k*abs(x[5])*x[26] - x[2]*dx[24] + m*dx[26]
            res[26] = 2x[1]x[22] + 2x[2]x[23]
            res[27] = x[4]x[22] + x[5]x[23] + x[1]x[25] + x[2]x[26]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[28] = x[28] - (x[1]*x[23] - x[2]*x[22])/(L^2)

            nothing
        end

        # Finding consistent initial conditions
        # Initial values, the pendulum starts at rest
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        x1_0 = L * sin(Φ)
        x2_0 = -L * cos(Φ)
        dx3_0 = m*g/x2_0
        dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m

        pend0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        sk0  = zeros(7)
        dsk0 = zeros(7)
        r0 = zeros(7)
        rp0 = zeros(7)

        x0  = vcat(pend0, sk0, r0, r0)
        dx0 = vcat(dpend0, dsk0, rp0, rp0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = repeat(vcat(fill(true, 6), [false]), 4)

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_sensitivity_k_with_dist_sens_3(Φ::Float64, u::Function, w::Function, θ::Vector{Float64})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - x[4] + 2dx[6]*x[1]
            res[2] = dx[2] - x[5] + 2dx[6]*x[2]
            res[3] = m*dx[4] - dx[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
            res[4] = m*dx[5] - dx[3]*x[2] + k*abs(x[5])*x[5] + m*g
            res[5] = x[1]^2 + x[2]^2 - L^2
            res[6] = x[4]*x[1] + x[5]*x[2]
            # Angle of pendulum
            res[7] = x[7] - atan(x[1] / -x[2])

            # Sensitivity with respect to k
            res[8]  = -x[11] + dx[8] + 2x[1]*dx[13] + 2x[8]*dx[6]
            res[9]  = -x[12] + dx[9] + 2x[2]*dx[13] + 2x[9]*dx[6]
            res[10] = 2k*x[11]*abs(x[4]) - x[1]*dx[10] + m*dx[11] - x[8]*dx[3] + abs(x[4])x[4]
            res[11] = 2k*x[12]*abs(x[5]) - x[2]*dx[10] + m*dx[12] - x[9]*dx[3] + abs(x[5])x[5]
            res[12] = -2x[8]*x[1] - 2x[9]*x[2]
            res[13] = x[11]*x[1] + x[12]*x[2] + x[8]*x[4] + x[9]*x[5]
            res[14] = x[14] - (x[1]*x[9] - x[2]*x[8])/(L^2)

            # Sensitivty wrt first disturbance parameter
            # (i.e. parameter corresponding to wt[2])
            res[15] = 2dx[6]x[15] - x[18] + dx[15] + 2x[1]dx[20]
            res[16] = 2dx[6]x[16] - x[19] + dx[16] + 2x[2]dx[20]
            res[17] = -dx[3]x[15] + 2k*abs(x[4])*x[18] - x[1]*dx[17] + m*dx[18] - 2*wt[1]*wt[2]
            res[18] = -dx[3]x[16] + 2k*abs(x[5])*x[19] - x[2]*dx[17] + m*dx[19]
            res[19] = 2x[1]x[15] + 2x[2]x[16]
            res[20] = x[4]x[15] + x[5]x[16] + x[1]x[18] + x[2]x[19]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[21] = x[21] - (x[1]*x[16] - x[2]*x[15])/(L^2)

            # Sensitivty wrt second disturbance parameter
            # (i.e. parameter corresponding to wt[2])
            res[22] = 2dx[6]x[22] - x[25] + dx[22] + 2x[1]dx[27]
            res[23] = 2dx[6]x[23] - x[26] + dx[23] + 2x[2]dx[27]
            res[24] = -dx[3]x[22] + 2k*abs(x[4])*x[25] - x[1]*dx[24] + m*dx[25] - 2*wt[1]*wt[3]
            res[25] = -dx[3]x[23] + 2k*abs(x[5])*x[26] - x[2]*dx[24] + m*dx[26]
            res[26] = 2x[1]x[22] + 2x[2]x[23]
            res[27] = x[4]x[22] + x[5]x[23] + x[1]x[25] + x[2]x[26]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[28] = x[28] - (x[1]*x[23] - x[2]*x[22])/(L^2)

            # Sensitivty wrt third disturbance parameter
            # (i.e. parameter corresponding to wt[2])
            res[29] = 2dx[6]x[29] - x[32] + dx[29] + 2x[1]dx[34]
            res[30] = 2dx[6]x[30] - x[33] + dx[30] + 2x[2]dx[34]
            res[31] = -dx[3]x[29] + 2k*abs(x[4])*x[32] - x[1]*dx[31] + m*dx[32] - 2*wt[1]*wt[4]
            res[32] = -dx[3]x[30] + 2k*abs(x[5])*x[33] - x[2]*dx[31] + m*dx[33]
            res[33] = 2x[1]x[29] + 2x[2]x[30]
            res[34] = x[4]x[29] + x[5]x[30] + x[1]x[32] + x[2]x[33]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[35] = x[35] - (x[1]*x[30] - x[2]*x[29])/(L^2)

            nothing
        end

        # Finding consistent initial conditions
        # Initial values, the pendulum starts at rest
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        x1_0 = L * sin(Φ)
        x2_0 = -L * cos(Φ)
        dx3_0 = m*g/x2_0
        dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m

        pend0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        sk0  = zeros(7)
        dsk0 = zeros(7)
        r0 = zeros(7)
        rp0 = zeros(7)

        x0  = vcat(pend0, sk0, r0, r0, r0)
        dx0 = vcat(dpend0, dsk0, rp0, rp0, rp0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = repeat(vcat(fill(true, 6), [false]), 5)

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function my_pendulum_adjoint(u::Function, w::Function, θ::Vector{Float64}, T::Float64, sol::DAESolution, sol2::DAESolution, y::Function, dy::Function, xp0::Vector{Float64}, dx::Function, dx2::Function, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function.
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        @assert (np == 3) "my_pendulum_adjoint is hard-coded to only handle exactly three parameters (all dynamical), make sure to pass correct xp0"
        nx = size(xp0,1)
        x  = t -> sol(t)
        x2 = t -> sol2(t)

        Fx = t -> [2dx(t)[6]        0.0               0.0   -1.0                  0.0           0.0   0.0
                    0.0           2*dx(t)[6]          0.0   0.0                   -1.0          0.0   0.0
                    -dx(t)[3]         0.0             0.0   2k*abs(x(t)[4])     0.0           0.0   0.0
                    0.0            -dx(t)[3]          0.0   0.0               2k*abs(x(t)[5]) 0.0   0.0
                    2x(t)[1]      2x(t)[2]          0.0   0.0                   0.0           0.0   0.0
                    x(t)[4]        x(t)[5]          0.0   x(t)[1]            x(t)[2]      0.0   0.0
                    x(t)[2]/(L^2)  -x(t)[1]/(L^2)   0.0   0.0                   0.0           0.0   1.0]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = t -> vcat([1.0   0.0   0.0          0.0   0.0   2x(t)[1]    0.0
                         0.0   1.0   0.0          0.0   0.0   2x(t)[2]    0.0
                         0.0   0.0   -x(t)[1]   m   0.0   0.0           0.0
                         0.0   0.0   -x(t)[2]   0.0   m   0.0           0.0], zeros(3,7))
        Fddx = t -> vcat([  0.0   0.0  0.0            0.0   0.0   2dx(t)[1]    0.0
                            0.0   0.0  0.0            0.0   0.0   2dx(t)[2]    0.0
                            0.0   0.0  -dx(t)[1]    0.0   0.0   0.0            0.0
                            0.0   0.0  -dx(t)[2]    0.0   0.0   0.0            0.0], zeros(3,7))
        Fp = t -> [ .0          .0          .0
                    .0          .0          .0
                    dx(t)[4]    .0  abs(x(t)[4])*x(t)[4]
                    dx(t)[5]+g  .0  abs(x(t)[5])*x(t)[5]
                    .0           L          .0
                    .0          .0          .0
                    .0          .0          .0]
        gₓ  = t -> [0.0    0.0    0.0    0.0    0.0    0.0    2(x2(t)[7]-y(t))/T]
        gdₓ = t -> [0.0    0.0    0.0    0.0    0.0    0.0    2(dx2(t)[7]-dy(t))/T]

        # NOTE: Convention is used that derivatives wrt to θ stack along cols
        # while derivatives wrt to x stack along rows

        ###################### INITIALIZING ADJOINT SYSTEM ####################
        # Indices 1-4 are differential (d), while 5-7 are algebraic (a)
        dinds = 1:4
        ainds = 5:7
        λT  = zeros(7)
        dλT = zeros(7)
        temp = (-gₓ(T))/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        λT[dinds] = zeros(length(dinds))
        λT[ainds] = temp[ainds]
        dλT[dinds] = temp[dinds]
        temp = (-gdₓ(T) + (dλT[dinds]')*(Fx(T)[dinds,:] - Fddx(T)[dinds,:] - Fdx(T)[dinds,:]) + (λT[ainds]')*Fx(T)[ainds,:])/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        dλT[ainds] = temp[ainds]

        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            # TODO: Complete
            # # Completely unreadabe but most efficient version (still not a huge improvement)
            # res[1]  = dx(T-t)[3]*z[3] - dz[1] - 2*dx(T-t)[6]*z[1] + 2*z[5]*x(T-t)[1] - z[6]*x(T-t)[4] - (z[7]*x(T-t)[2])/(L^2)
            # res[2]  = dx(T-t)[3]*z[4] - dz[2] - 2*dx(T-t)[6]*z[2] + 2*z[5]*x(T-t)[2] - z[6]*x(T-t)[5] + (z[7]*x(T-t)[1])/(L^2)
            # res[3]  = dz[3]*x(T-t)[1] - dx(T-t)[2]*z[4] - dx(T-t)[1]*z[3] + dz[4]*x(T-t)[2]
            # res[4]  = z[1] - dz[3]*m - z[6]*x(T-t)[1] - 2*k*z[3]*abs(x(T-t)[4])
            # res[5]  = z[2] - dz[4]*m - z[6]*x(T-t)[2] - 2*k*z[4]*abs(x(T-t)[5])
            # res[6]  = 2*dx(T-t)[1]*z[1] + 2*dx(T-t)[2]*z[2] - 2*dz[1]*x(T-t)[1] - 2*dz[2]*x(T-t)[2]
            # res[7]  = (2*(x2(T-t)[7] - y(T-t)))/T - z[7]
            # res[8]  = dz[8] + dz[3]*abs(x(T-t)[4])*x(T-t)[4] + dz[4]*abs(x(T-t)[5])*x(T-t)[5]

            # TODO: Complete
            # # Medium readable but quite efficient version
            # λ  = z[1:nx]
            # dλ = dz[1:nx]
            # β  = z[nx+1:end]
            # dβ = dz[nx+1:end]
            # res[1]  = dx(T-t)[3]*λ[3] - dλ[1] - 2*dx(T-t)[6]*λ[1] + 2*λ[5]*x(T-t)[1] - λ[6]*x(T-t)[4] - (λ[7]*x(T-t)[2])/(L^2)
            # res[2]  = dx(T-t)[3]*λ[4] - dλ[2] - 2*dx(T-t)[6]*λ[2] + 2*λ[5]*x(T-t)[2] - λ[6]*x(T-t)[5] + (λ[7]*x(T-t)[1])/(L^2)
            # res[3]  = dλ[3]*x(T-t)[1] - dx(T-t)[2]*λ[4] - dx(T-t)[1]*λ[3] + dλ[4]*x(T-t)[2]
            # res[4]  = λ[1] - dλ[3]*m - λ[6]*x(T-t)[1] - 2*k*λ[3]*abs(x(T-t)[4])
            # res[5]  = λ[2] - dλ[4]*m - λ[6]*x(T-t)[2] - 2*k*λ[4]*abs(x(T-t)[5])
            # res[6]  = 2*dx(T-t)[1]*λ[1] + 2*dx(T-t)[2]*λ[2] - 2*dλ[1]*x(T-t)[1] - 2*dλ[2]*x(T-t)[2]
            # res[7]  = (2*(x2(T-t)[7] - y(T-t)))/T - λ[7]
            # res[8]  = dβ[1] + dλ[3]*abs(x(T-t)[4])*x(T-t)[4] + dλ[4]*abs(x(T-t)[5])*x(T-t)[5]
            # # Readable but slightly less efficient version
            # res[1:7] = -(dλ')*Fdx(T-t) + (λ')*(Fddx(T-t)-Fx(T-t)) + gₓ(T-t)
            # res[8]  = dβ[1]+(λ')*Fp(T-t)

            # Super-readable but less efficient version
            res[1:7]  = (-dz[1:7]')*Fdx(T-t) + (z[1:7]')*(Fddx(T-t) - Fx(T-t)) + gₓ(T-t)
            # res[8:10] = -dz[8:10] - (z[1:7]')*Fp(T-t)
            res[8:10] = -dz[8:10] - (Fp(T-t)')*z[1:7]
            nothing
        end

        # Solving backwards in time =>
        # 1. Initial conditions will equal terminal conditions
        # 2. All derivatives have to be negated
        λ0  = λT[:]
        dλ0 = -dλT[:]
        # z0 = λ0
        # dz0 = dλ0
        z0  = vcat(λ0, zeros(np))
        # dz0 = vcat(dλ0, -((λT')*Fp(T))')
        dz0 = vcat(dλ0, -(Fp(T)')*λT)
        # TODO: Can we delete these row below here perhaps?
        # z0  = vcat(λ0, 0.0)
        # dz0 = vcat(dλ0, λ0[1])   # For some reasing (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired

        if N_trans > 0
            @warn "The returned function get_Gp() doesn't fully support N_trans > 0, as sensitivity of internal variables not known at any other time than t=0. A non-rigorous approximation is used instead."
        end
        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            # NOTE: Changes signs to match what I had in my manual calculations, seems correct now
            # Gp = adj_sol.u[end][end-np+1:end] + (((adj_sol.u[end][1:end-np]')*Fdx(0.0))*xp0)[:]
            Gp = adj_sol.u[end-N_trans][nx+1:nx+np] + (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)[:]
            # return 0.0
        end

        dvars = vcat(fill(true, 4), fill(false, 3), fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0 here, λ0: $λ0, dλ0: $dλ0"

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp
    end
end

function NEGATED_my_pendulum_adjoint_monly(u::Function, w::Function, θ::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Vector{Float64}, dx::Function, dx2::Function, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function.
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        @assert (np == 1) "my_pendulum_adjoint_monly is hard-coded to only handle one parameter m, make sure to pass correct xp0"
        nx = size(xp0,1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        Fx = t -> [2dx(t)[6]        0               0   -1                  0           0   0
                    0           2*dx(t)[6]          0   0                   -1          0   0
                    -dx(t)[3]         0             0   2k*abs(x(t)[4])     0           0   0
                    0            -dx(t)[3]          0   0               2k*abs(x(t)[5]) 0   0
                    2x(t)[1]      2x(t)[2]          0   0                   0           0   0
                    x(t)[4]        x(t)[5]          0   x(t)[1]            x(t)[2]      0   0
                    x(t)[2]/(L^2)  -x(t)[1]/(L^2)   0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = t -> vcat([1.0   0   0          0   0   2x(t)[1]    0
                         0   1.0   0          0   0   2x(t)[2]    0
                         0   0   -x(t)[1]   m   0   0           0
                         0   0   -x(t)[2]   0   m   0           0], zeros(3,7))
        Fddx = t -> vcat([  0   0  0            0   0   2dx(t)[1]    0
                            0   0  0            0   0   2dx(t)[2]    0
                            0   0  -dx(t)[1]    0   0   0            0
                            0   0  -dx(t)[2]    0   0   0            0], zeros(3,7))
        Fp = t -> [ .0
                    .0
                    dx(t)[4]
                    dx(t)[5]+g
                    .0
                    .0
                    .0]
        gₓ  = t -> [0    0    0    0    0    0    2(x2(t)[7]-y(t))/T]
        gdₓ = t -> [0    0    0    0    0    0    2(dx2(t)[7]-dy(t))/T]

        # NOTE: Convention is used that derivatives wrt to θ stack along cols
        # while derivatives wrt to x stack along rows

        ###################### INITIALIZING ADJOINT SYSTEM ####################
        # Indices 1-4 are differential (d), while 5-7 are algebraic (a)
        dinds = 1:4
        ainds = 5:7
        λT  = zeros(7)
        dλT = zeros(7)
        temp = (-gₓ(T))/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        λT[dinds] = zeros(length(dinds))
        λT[ainds] = temp[ainds]
        dλT[dinds] = temp[dinds]
        temp = (-gdₓ(T) + (dλT[dinds]')*(Fx(T)[dinds,:] - Fddx(T)[dinds,:] - Fdx(T)[dinds,:]) + (λT[ainds]')*Fx(T)[ainds,:])/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        dλT[ainds] = temp[ainds]

        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            # TODO: Complete
            # Completely unreadabe but most efficient version (still not a huge improvement)
            res[1]  = dx(T-t)[3]*z[3] - dz[1] - 2*dx(T-t)[6]*z[1] - 2*z[5]*x(T-t)[1] - z[6]*x(T-t)[4] - (z[7]*x(T-t)[2])/(L^2)
            res[2]  = dx(T-t)[3]*z[4] - dz[2] - 2*dx(T-t)[6]*z[2] - 2*z[5]*x(T-t)[2] - z[6]*x(T-t)[5] + (z[7]*x(T-t)[1])/(L^2)
            res[3]  = dz[3]*x(T-t)[1] - dx(T-t)[2]*z[4] - dx(T-t)[1]*z[3] + dz[4]*x(T-t)[2]
            res[4]  = z[1] - dz[3]*m - z[6]*x(T-t)[1] - 2*k*z[3]*abs(x(T-t)[4])
            res[5]  = z[2] - dz[4]*m - z[6]*x(T-t)[2] - 2*k*z[4]*abs(x(T-t)[5])
            res[6]  = 2*dx(T-t)[1]*z[1] + 2*dx(T-t)[2]*z[2] - 2*dz[1]*x(T-t)[1] - 2*dz[2]*x(T-t)[2]
            res[7]  = (2*(x2(T-t)[7] - y(T-t)))/T - z[7]
            res[8]  = dz[8] + z[3]*dx(T-t)[4] + z[4]*(dx(T-t)[5]+g)

            # # Super-readable but less efficient version
            # res[1:7]  = (-dz[1:7]')*Fdx(T-t) + (z[1:7]')*(Fddx(T-t) - Fx(T-t)) + gₓ(T-t)
            # res[8] = dz[8] + (Fp(T-t)')*z[1:7]
            nothing
        end

        # Solving backwards in time =>
        # 1. Initial conditions will equal terminal conditions
        # 2. All derivatives have to be negated
        λ0  = λT[:]
        dλ0 = -dλT[:]
        # z0 = λ0
        # dz0 = dλ0
        z0  = vcat(λ0, zeros(np))
        # dz0 = vcat(dλ0, -((λT')*Fp(T))')
        dz0 = vcat(dλ0, -(Fp(T)')*λT)
        # TODO: Can we delete these row below here perhaps?
        # z0  = vcat(λ0, 0.0)
        # dz0 = vcat(dλ0, λ0[1])   # For some reasing (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired

        if N_trans > 0
            @warn "The returned function get_Gp() doesn't fully support N_trans > 0, as sensitivity of internal variables not known at any other time than t=0. A non-rigorous approximation is used instead."
        end
        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            # NOTE: Changes signs to match what I had in my manual calculations, seems correct now
            # Gp = adj_sol.u[end][end-np+1:end] + (((adj_sol.u[end][1:end-np]')*Fdx(0.0))*xp0)[:]
            Gp = adj_sol.u[end-N_trans][nx+1:nx+np] .+ (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)
            # return 0.0
        end

        dvars = vcat(fill(true, 4), fill(false, 3), fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0 here, λ0: $λ0, dλ0: $dλ0"

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp
    end
end

function my_pendulum_adjoint_monly(u::Function, w::Function, θ::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Matrix{Float64}, dx::Function, dx2::Function, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function.
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        @assert (np == 1) "my_pendulum_adjoint_monly is hard-coded to only handle one parameter m, make sure to pass correct xp0"
        nx = size(xp0,1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        Fx = t -> [2dx(t)[6]        0               0   -1                  0           0   0
                    0           2*dx(t)[6]          0   0                   -1          0   0
                    -dx(t)[3]         0             0   2k*abs(x(t)[4])     0           0   0
                    0            -dx(t)[3]          0   0               2k*abs(x(t)[5]) 0   0
                    2x(t)[1]      2x(t)[2]          0   0                   0           0   0
                    x(t)[4]        x(t)[5]          0   x(t)[1]            x(t)[2]      0   0
                    x(t)[2]/(L^2)  -x(t)[1]/(L^2)   0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = t -> vcat([1.0   0   0          0   0   2x(t)[1]    0
                         0   1.0   0          0   0   2x(t)[2]    0
                         0   0   -x(t)[1]   m   0   0           0
                         0   0   -x(t)[2]   0   m   0           0], zeros(3,7))
        Fddx = t -> vcat([  0   0  0            0   0   2dx(t)[1]    0
                            0   0  0            0   0   2dx(t)[2]    0
                            0   0  -dx(t)[1]    0   0   0            0
                            0   0  -dx(t)[2]    0   0   0            0], zeros(3,7))
        Fp = t -> [ .0
                    .0
                    dx(t)[4]
                    dx(t)[5]+g
                    .0
                    .0
                    .0]
        gₓ  = t -> [0    0    0    0    0    0    2(x2(t)[7]-first(y(t)))/T]
        gdₓ = t -> [0    0    0    0    0    0    2(dx2(t)[7]-first(dy(t)))/T]

        # NOTE: Convention is used that derivatives wrt to θ stack along cols
        # while derivatives wrt to x stack along rows

        ###################### INITIALIZING ADJOINT SYSTEM ####################
        # Indices 1-4 are differential (d), while 5-7 are algebraic (a)
        dinds = 1:4
        ainds = 5:7
        λT  = zeros(7)
        dλT = zeros(7)
        temp = (-gₓ(T))/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        λT[dinds] = zeros(length(dinds))
        λT[ainds] = temp[ainds]
        dλT[dinds] = temp[dinds]
        temp = (-gdₓ(T) + (dλT[dinds]')*(Fx(T)[dinds,:] - Fddx(T)[dinds,:] - Fdx(T)[dinds,:]) + (λT[ainds]')*Fx(T)[ainds,:])/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        dλT[ainds] = temp[ainds]

        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            # TODO: Complete
            # Completely unreadabe but most efficient version (still not a huge improvement)
            res[1]  = dz[1] - 2*dx(t)[6]*z[1] + dx(t)[3]*z[3] - 2*x(t)[1]*z[5] - x(t)[4]*z[6] - (x(t)[2]*z[7])/(L^2)
            res[2]  = dz[2] - 2*dx(t)[6]*z[2] + dx(t)[3]*z[4] - 2*x(t)[2]*z[5] - x(t)[5]*z[6] + (x(t)[1]*z[7])/(L^2)
            res[3]  = -x(t)[1]*dz[3] - x(t)[2]*dz[4] - dx(t)[1]*z[3] - dx(t)[2]*z[4]
            res[4]  = m*dz[3] + z[1] - 2*k*abs(x(t)[4])*z[3] - x(t)[1]*z[6]
            res[5]  = m*dz[4] + z[2] - 2*k*abs(x(t)[5])*z[4] - x(t)[2]*z[6]
            res[6]  = 2*x(t)[1]*dz[1] + 2*x(t)[2]*dz[2] + 2*dx(t)[1]*z[1] + 2*dx(t)[2]*z[2]
            res[7]  = (2*(x2(t)[7] - first(y(t))))/T - z[7]
            # res[1]  = dz[1] + 2x(t)[1]*dz[6]    - 2dx(t)[6]*z[1] + z[4] + 2dx(t)[1]*z[6]
            # res[2]  = dz[2] + 2x(t)[2]*dz[6]    - 2dx(t)[6]*z[2] + z[5] + 2dx(t)[2]*z[6]
            # res[3]  = -x(t)[1]*dz[3] + m*dz[4]  + dx(t)[3]*z[1] - dx(t)[1]*z[3] - 2k*abs(x(t)[4])*z[4]
            # res[4]  = -x(t)[2]*dz[3] + m*dz[5]  + dx(t)[3]*z[2] - dx(t)[2]*z[3] - 2k*abs(x(t)[5])*z[5]
            # res[5]  =                           - 2x(t)[1]*z[1] - 2x(t)[2]*z[2]
            # res[6]  =                           - x(t)[4]*z[1] - x(t)[5]*z[2] - x(t)[1]*z[4] - x(t)[2]*z[5]
            # res[7]  =                           - x(t)[2]*z[1]/(L^2) + x(t)[1]*z[2]/(L^2) - z[7]                + (2/T)*(x2(t)[7]-y(t))
            res[8]  = dz[8] - z[3]*dx(t)[4] - z[4]*(dx(t)[5]+g)
            # res[8]  = dz[8] - z[3]*dx(T-t)[4] - z[4]*(dx(T-t)[5]+g)   # NOTE: SIMPLY WRONG; T-t????
            # res[8]  = dz[8] - z[3]*abs(x(t)[4])*x(t)[4] - z[4]*abs(x(t)[5])*x(t)[5]   # from pendelum k-only

            # # Super-readable but less efficient version ALSO NEGATED
            # res[1:7]  = (dz[1:7]')*Fdx(T-t) + (z[1:7]')*(Fddx(T-t) - Fx(T-t)) + gₓ(T-t)
            # res[8] = dz[8] - (Fp(T-t)')*z[1:7]
            nothing
        end

        # z0 = λ0
        # dz0 = dλ0
        z0  = vcat(λT[:], zeros(np))
        # dz0 = vcat(dλ0, -((λT')*Fp(T))')
        dz0 = vcat(dλT[:], -(Fp(T)')*λT)
        # TODO: Can we delete these row below here perhaps?
        # z0  = vcat(λ0, 0.0)
        # dz0 = vcat(dλ0, λ0[1])   # For some reasing (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired

        if N_trans > 0
            @warn "The returned function get_Gp() doesn't fully support N_trans > 0, as sensitivity of internal variables not known at any other time than t=0. A non-rigorous approximation is used instead."
        end
        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            # NOTE: Changes signs to match what I had in my manual calculations, seems correct now
            # Gp = adj_sol.u[end][end-np+1:end] + (((adj_sol.u[end][1:end-np]')*Fdx(0.0))*xp0)[:]
            Gp = adj_sol.u[end-N_trans][nx+1:nx+np] .+ (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)
            # return 0.0
        end

        function get_Gp_debug(adj_sol::DAESolution)
            integral = adj_sol.u[end][nx+1:nx+np]
            term = (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)
            Gp = integral .+ term
            return Gp, integral, term
        end

        function get_term_debug(adj_sol::DAESolution, xps::Matrix{Float64}, times::AbstractVector{Float64})
            term = zeros(length(adj_sol.u))
            for ind=eachindex(adj_sol.u)
                term[ind] = ((adj_sol.u[end+1-ind][1:nx]')*Fdx(times[ind]))*xps[:,ind]
            end
            return term
        end
        debugs = (get_Gp_debug, get_term_debug)

        dvars = vcat(fill(true, 4), fill(false, 3), fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0 here, λ0: $λ0, dλ0: $dλ0"

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp, debugs
    end
end

# Removes factor 1/T from gₓ, should be added back during Gp computation
function DEBUG_my_pendulum_adjoint_monly(u::Function, w::Function, θ::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Vector{Float64}, dx::Function, dx2::Function, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function.
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        @assert (np == 1) "my_pendulum_adjoint_monly is hard-coded to only handle one parameter m, make sure to pass correct xp0"
        nx = size(xp0,1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        Fx = t -> [2dx(t)[6]        0               0   -1                  0           0   0
                    0           2*dx(t)[6]          0   0                   -1          0   0
                    -dx(t)[3]         0             0   2k*abs(x(t)[4])     0           0   0
                    0            -dx(t)[3]          0   0               2k*abs(x(t)[5]) 0   0
                    2x(t)[1]      2x(t)[2]          0   0                   0           0   0
                    x(t)[4]        x(t)[5]          0   x(t)[1]            x(t)[2]      0   0
                    x(t)[2]/(L^2)  -x(t)[1]/(L^2)   0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = t -> vcat([1.0   0   0          0   0   2x(t)[1]    0
                         0   1.0   0          0   0   2x(t)[2]    0
                         0   0   -x(t)[1]   m   0   0           0
                         0   0   -x(t)[2]   0   m   0           0], zeros(3,7))
        Fddx = t -> vcat([  0   0  0            0   0   2dx(t)[1]    0
                            0   0  0            0   0   2dx(t)[2]    0
                            0   0  -dx(t)[1]    0   0   0            0
                            0   0  -dx(t)[2]    0   0   0            0], zeros(3,7))
        Fp = t -> [ .0
                    .0
                    dx(t)[4]
                    dx(t)[5]+g
                    .0
                    .0
                    .0]
        gₓ  = t -> [0    0    0    0    0    0    2(x2(t)[7]-y(t))]
        gdₓ = t -> [0    0    0    0    0    0    2(dx2(t)[7]-dy(t))]

        # NOTE: Convention is used that derivatives wrt to θ stack along cols
        # while derivatives wrt to x stack along rows

        ###################### INITIALIZING ADJOINT SYSTEM ####################
        # Indices 1-4 are differential (d), while 5-7 are algebraic (a)
        dinds = 1:4
        ainds = 5:7
        λT  = zeros(7)
        dλT = zeros(7)
        temp = (-gₓ(T))/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        λT[dinds] = zeros(length(dinds))
        λT[ainds] = temp[ainds]
        dλT[dinds] = temp[dinds]
        temp = (-gdₓ(T) + (dλT[dinds]')*(Fx(T)[dinds,:] - Fddx(T)[dinds,:] - Fdx(T)[dinds,:]) + (λT[ainds]')*Fx(T)[ainds,:])/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        dλT[ainds] = temp[ainds]

        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            # TODO: Complete
            # Completely unreadabe but most efficient version (still not a huge improvement)
            res[1]  = dz[1] - 2*dx(t)[6]*z[1] + dx(t)[3]*z[3] - 2*x(t)[1]*z[5] - x(t)[4]*z[6] - (x(t)[2]*z[7])/(L^2)
            res[2]  = dz[2] - 2*dx(t)[6]*z[2] + dx(t)[3]*z[4] - 2*x(t)[2]*z[5] - x(t)[5]*z[6] + (x(t)[1]*z[7])/(L^2)
            res[3]  = -x(t)[1]*dz[3] - x(t)[2]*dz[4] - dx(t)[1]*z[3] - dx(t)[2]*z[4]
            res[4]  = m*dz[3] + z[1] - 2*k*abs(x(t)[4])*z[3] - x(t)[1]*z[6]
            res[5]  = m*dz[4] + z[2] - 2*k*abs(x(t)[5])*z[4] - x(t)[2]*z[6]
            res[6]  = 2*x(t)[1]*dz[1] + 2*x(t)[2]*dz[2] + 2*dx(t)[1]*z[1] + 2*dx(t)[2]*z[2]
            res[7]  = (2*(x2(t)[7] - y(t)))/T - z[7]
            # res[1]  = dz[1] + 2x(t)[1]*dz[6]    - 2dx(t)[6]*z[1] + z[4] + 2dx(t)[1]*z[6]
            # res[2]  = dz[2] + 2x(t)[2]*dz[6]    - 2dx(t)[6]*z[2] + z[5] + 2dx(t)[2]*z[6]
            # res[3]  = -x(t)[1]*dz[3] + m*dz[4]  + dx(t)[3]*z[1] - dx(t)[1]*z[3] - 2k*abs(x(t)[4])*z[4]
            # res[4]  = -x(t)[2]*dz[3] + m*dz[5]  + dx(t)[3]*z[2] - dx(t)[2]*z[3] - 2k*abs(x(t)[5])*z[5]
            # res[5]  =                           - 2x(t)[1]*z[1] - 2x(t)[2]*z[2]
            # res[6]  =                           - x(t)[4]*z[1] - x(t)[5]*z[2] - x(t)[1]*z[4] - x(t)[2]*z[5]
            # res[7]  =                           - x(t)[2]*z[1]/(L^2) + x(t)[1]*z[2]/(L^2) - z[7]                + (2/T)*(x2(t)[7]-y(t))
            res[8]  = dz[8] - z[3]*dx(t)[4] - z[4]*(dx(t)[5]+g)
            # res[8]  = dz[8] - z[3]*dx(T-t)[4] - z[4]*(dx(T-t)[5]+g)   # NOTE: SIMPLY WRONG; T-t????
            # res[8]  = dz[8] - z[3]*abs(x(t)[4])*x(t)[4] - z[4]*abs(x(t)[5])*x(t)[5]   # from pendelum k-only

            # # Super-readable but less efficient version
            # res[1:7]  = (dz[1:7]')*Fdx(T-t) + (z[1:7]')*(Fddx(T-t) - Fx(T-t)) + gₓ(T-t)
            # res[8] = dz[8] - (Fp(T-t)')*z[1:7]
            nothing
        end

        # z0 = λ0
        # dz0 = dλ0
        z0  = vcat(λT[:], zeros(np))
        # dz0 = vcat(dλ0, -((λT')*Fp(T))')
        dz0 = vcat(dλT[:], -(Fp(T)')*λT)
        # TODO: Can we delete these row below here perhaps?
        # z0  = vcat(λ0, 0.0)
        # dz0 = vcat(dλ0, λ0[1])   # For some reasing (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired

        if N_trans > 0
            @warn "The returned function get_Gp() doesn't fully support N_trans > 0, as sensitivity of internal variables not known at any other time than t=0. A non-rigorous approximation is used instead."
        end
        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            # NOTE: Changes signs to match what I had in my manual calculations, seems correct now
            # Gp = adj_sol.u[end][end-np+1:end] + (((adj_sol.u[end][1:end-np]')*Fdx(0.0))*xp0)[:]
            @warn "adj_sols: $(adj_sol.u[end][nx+1:nx+np]), $(adj_sol.u[1][nx+1:nx+np])"
            Gp = (adj_sol.u[end-N_trans][nx+1:nx+np] .+ (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0))/T
            # return 0.0
        end

        dvars = vcat(fill(true, 4), fill(false, 3), fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0 here, λ0: $λ0, dλ0: $dλ0"

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp
    end
end

function NEGATED_my_pendulum_adjoint_Lonly(u::Function, w::Function, θ::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Vector{Float64}, dx::Function, dx2::Function, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function.
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        @assert (np == 1) "my_pendulum_adjoint_Lonly is hard-coded to only handle one parameter L, make sure to pass correct xp0"
        nx = size(xp0,1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        Fx = t -> [2dx(t)[6]        0               0   -1                  0           0   0
                    0           2*dx(t)[6]          0   0                   -1          0   0
                    -dx(t)[3]         0             0   2k*abs(x(t)[4])     0           0   0
                    0            -dx(t)[3]          0   0               2k*abs(x(t)[5]) 0   0
                    2x(t)[1]      2x(t)[2]          0   0                   0           0   0
                    x(t)[4]        x(t)[5]          0   x(t)[1]            x(t)[2]      0   0
                    x(t)[2]/(L^2)  -x(t)[1]/(L^2)   0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = t -> vcat([1   0   0          0   0   2x(t)[1]    0
                         0   1   0          0   0   2x(t)[2]    0
                         0   0   -x(t)[1]   m   0   0           0
                         0   0   -x(t)[2]   0   m   0           0], zeros(3,7))
        Fddx = t -> vcat([  0   0  0            0   0   2dx(t)[1]    0
                            0   0  0            0   0   2dx(t)[2]    0
                            0   0  -dx(t)[1]    0   0   0            0
                            0   0  -dx(t)[2]    0   0   0            0], zeros(3,7))
        Fp = t -> [ .0
                    .0
                    .0
                    .0
                    -2L   # How does L^2 turn into L? Is there no 2 missing or something?
                    .0
                    .0] # Was last element not dependent on L? Didn't we divide by e.g. L^2?
        gₓ  = t -> [0    0    0    0    0    0    2(x2(t)[7]-y(t))/T]
        gdₓ = t -> [0    0    0    0    0    0    2(dx2(t)[7]-dy(t))/T]

        # NOTE: Convention is used that derivatives wrt to θ stack along cols
        # while derivatives wrt to x stack along rows

        ###################### INITIALIZING ADJOINT SYSTEM ####################
        # Indices 1-4 are differential (d), while 5-7 are algebraic (a)
        dinds = 1:4
        ainds = 5:7
        λT  = zeros(7)
        dλT = zeros(7)
        temp = (-gₓ(T))/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        λT[dinds] = zeros(length(dinds))
        λT[ainds] = temp[ainds]
        dλT[dinds] = temp[dinds]
        temp = (-gdₓ(T) + (dλT[dinds]')*(Fx(T)[dinds,:] - Fddx(T)[dinds,:] - Fdx(T)[dinds,:]) + (λT[ainds]')*Fx(T)[ainds,:])/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        dλT[ainds] = temp[ainds]

        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            # Completely unreadabe but most efficient version (still not a huge improvement)
            res[1]  = dx(T-t)[3]*z[3] - dz[1] - 2*dx(T-t)[6]*z[1] - 2*z[5]*x(T-t)[1] - z[6]*x(T-t)[4] - (z[7]*x(T-t)[2])/(L^2)
            res[2]  = dx(T-t)[3]*z[4] - dz[2] - 2*dx(T-t)[6]*z[2] - 2*z[5]*x(T-t)[2] - z[6]*x(T-t)[5] + (z[7]*x(T-t)[1])/(L^2)
            res[3]  = dz[3]*x(T-t)[1] - dx(T-t)[2]*z[4] - dx(T-t)[1]*z[3] + dz[4]*x(T-t)[2]
            res[4]  = z[1] - dz[3]*m - z[6]*x(T-t)[1] - 2*k*z[3]*abs(x(T-t)[4])
            res[5]  = z[2] - dz[4]*m - z[6]*x(T-t)[2] - 2*k*z[4]*abs(x(T-t)[5])
            res[6]  = 2*dx(T-t)[1]*z[1] + 2*dx(T-t)[2]*z[2] - 2*dz[1]*x(T-t)[1] - 2*dz[2]*x(T-t)[2]
            res[7]  = (2*(x2(T-t)[7] - y(T-t)))/T - z[7]
            res[8]  = dz[8] - 2*L*z[5]

            # # Super-readable but less efficient version
            # res[1:7]  = (-dz[1:7]')*Fdx(T-t) + (z[1:7]')*(Fddx(T-t) - Fx(T-t)) + gₓ(T-t)
            # res[8] = dz[8] + (Fp(T-t)')*z[1:7]
            nothing
        end

        # Solving backwards in time =>
        # 1. Initial conditions will equal terminal conditions
        # 2. All derivatives have to be negated
        λ0  = λT[:]
        dλ0 = -dλT[:]
        # z0 = λ0
        # dz0 = dλ0
        z0  = vcat(λ0, zeros(np))
        # dz0 = vcat(dλ0, -((λT')*Fp(T))')
        dz0 = vcat(dλ0, -(Fp(T)')*λT)
        # TODO: Can we delete these row below here perhaps?
        # z0  = vcat(λ0, 0.0)
        # dz0 = vcat(dλ0, λ0[1])   # For some reasing (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired

        if N_trans > 0
            @warn "The returned function get_Gp() doesn't fully support N_trans > 0, as sensitivity of internal variables not known at any other time than t=0. A non-rigorous approximation is used instead."
        end
        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            # NOTE: Changes signs to match what I had in my manual calculations, seems correct now
            # Gp = adj_sol.u[end][end-np+1:end] + (((adj_sol.u[end][1:end-np]')*Fdx(0.0))*xp0)[:]
            # Gp = adj_sol.u[end-N_trans][nx+1:nx+np] + (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)[:]
            Gp = adj_sol.u[end-N_trans][nx+1:nx+np] .+ (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)
            # return 0.0
        end

        dvars = vcat(fill(true, 4), fill(false, 3), fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0 here, λ0: $λ0, dλ0: $dλ0"

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp
    end
end

function my_pendulum_adjoint_Lonly(u::Function, w::Function, θ::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Matrix{Float64}, dx::Function, dx2::Function, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function.
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        @assert (np == 1) "my_pendulum_adjoint_Lonly is hard-coded to only handle one parameter L, make sure to pass correct xp0"
        nx = size(xp0,1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        Fx = t -> [2dx(t)[6]        0               0   -1                  0           0   0
                    0           2*dx(t)[6]          0   0                   -1          0   0
                    -dx(t)[3]         0             0   2k*abs(x(t)[4])     0           0   0
                    0            -dx(t)[3]          0   0               2k*abs(x(t)[5]) 0   0
                    2x(t)[1]      2x(t)[2]          0   0                   0           0   0
                    x(t)[4]        x(t)[5]          0   x(t)[1]            x(t)[2]      0   0
                    x(t)[2]/(L^2)  -x(t)[1]/(L^2)   0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = t -> vcat([1   0   0          0   0   2x(t)[1]    0
                         0   1   0          0   0   2x(t)[2]    0
                         0   0   -x(t)[1]   m   0   0           0
                         0   0   -x(t)[2]   0   m   0           0], zeros(3,7))
        Fddx = t -> vcat([  0   0  0            0   0   2dx(t)[1]    0
                            0   0  0            0   0   2dx(t)[2]    0
                            0   0  -dx(t)[1]    0   0   0            0
                            0   0  -dx(t)[2]    0   0   0            0], zeros(3,7))
        Fp = t -> [ .0
                    .0
                    .0
                    .0
                    -2L
                    .0
                    .0] # Was last element not dependent on L? Didn't we divide by e.g. L^2?
        gₓ  = t -> [0    0    0    0    0    0    2(x2(t)[7]-first(y(t)))/T]
        gdₓ = t -> [0    0    0    0    0    0    2(dx2(t)[7]-first(dy(t)))/T]

        # NOTE: Convention is used that derivatives wrt to θ stack along cols
        # while derivatives wrt to x stack along rows

        ###################### INITIALIZING ADJOINT SYSTEM ####################
        # Indices 1-4 are differential (d), while 5-7 are algebraic (a)
        dinds = 1:4
        ainds = 5:7
        λT  = zeros(7)
        dλT = zeros(7)
        temp = (-gₓ(T))/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        λT[dinds] = zeros(length(dinds))
        λT[ainds] = temp[ainds]
        dλT[dinds] = temp[dinds]
        temp = (-gdₓ(T) + (dλT[dinds]')*(Fx(T)[dinds,:] - Fddx(T)[dinds,:] - Fdx(T)[dinds,:]) + (λT[ainds]')*Fx(T)[ainds,:])/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        dλT[ainds] = temp[ainds]

        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            # Completely unreadabe but most efficient version (still not a huge improvement)
            res[1]  = dz[1] - 2*dx(t)[6]*z[1] + dx(t)[3]*z[3] - 2*x(t)[1]*z[5] - x(t)[4]*z[6] - (x(t)[2]*z[7])/(L^2)
            res[2]  = dz[2] - 2*dx(t)[6]*z[2] + dx(t)[3]*z[4] - 2*x(t)[2]*z[5] - x(t)[5]*z[6] + (x(t)[1]*z[7])/(L^2)
            res[3]  = -x(t)[1]*dz[3] - x(t)[2]*dz[4] - dx(t)[1]*z[3] - dx(t)[2]*z[4]
            res[4]  = m*dz[3] + z[1] - 2*k*abs(x(t)[4])*z[3] - x(t)[1]*z[6]
            res[5]  = m*dz[4] + z[2] - 2*k*abs(x(t)[5])*z[4] - x(t)[2]*z[6]
            res[6]  = 2*x(t)[1]*dz[1] + 2*x(t)[2]*dz[2] + 2*dx(t)[1]*z[1] + 2*dx(t)[2]*z[2]
            res[7]  = (2*(x2(t)[7] - first(y(t))))/T - z[7]
            # res[1]  = dz[1] + 2x(t)[1]*dz[6]    - 2dx(t)[6]*z[1] + z[4] + 2dx(t)[1]*z[6]
            # res[2]  = dz[2] + 2x(t)[2]*dz[6]    - 2dx(t)[6]*z[2] + z[5] + 2dx(t)[2]*z[6]
            # res[3]  = -x(t)[1]*dz[3] + m*dz[4]  + dx(t)[3]*z[1] - dx(t)[1]*z[3] - 2k*abs(x(t)[4])*z[4]
            # res[4]  = -x(t)[2]*dz[3] + m*dz[5]  + dx(t)[3]*z[2] - dx(t)[2]*z[3] - 2k*abs(x(t)[5])*z[5]
            # res[5]  =                           - 2x(t)[1]*z[1] - 2x(t)[2]*z[2]
            # res[6]  =                           - x(t)[4]*z[1] - x(t)[5]*z[2] - x(t)[1]*z[4] - x(t)[2]*z[5]
            # res[7]  =                           - x(t)[2]*z[1]/(L^2) + x(t)[1]*z[2]/(L^2) - z[7]                + (2/T)*(x2(t)[7]-y(t))
            res[8]  = dz[8] + 2*L*z[5]

            # # Super-readable but less efficient version
            # res[1:7]  = (dz[1:7]')*Fdx(T-t) + (z[1:7]')*(Fddx(T-t) - Fx(T-t)) + gₓ(T-t)
            # res[8] = dz[8] - (Fp(T-t)')*z[1:7]
            nothing
        end

        # z0 = λ0
        # dz0 = dλ0
        z0  = vcat(λT[:], zeros(np))
        # dz0 = vcat(dλ0, -((λT')*Fp(T))')
        dz0 = vcat(dλT[:], -(Fp(T)')*λT)
        # TODO: Can we delete these row below here perhaps?
        # z0  = vcat(λ0, 0.0)
        # dz0 = vcat(dλ0, λ0[1])   # For some reasing (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired

        if N_trans > 0
            @warn "The returned function get_Gp() doesn't fully support N_trans > 0, as sensitivity of internal variables not known at any other time than t=0. A non-rigorous approximation is used instead."
        end
        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            # NOTE: Changes signs to match what I had in my manual calculations, seems correct now
            # Gp = adj_sol.u[end][end-np+1:end] + (((adj_sol.u[end][1:end-np]')*Fdx(0.0))*xp0)[:]
            # Gp = adj_sol.u[end-N_trans][nx+1:nx+np] + (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)[:]
            Gp = adj_sol.u[end-N_trans][nx+1:nx+np] .+ (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)
            # return 0.0
        end

        function get_Gp_debug(adj_sol::DAESolution)
            integral = adj_sol.u[end][nx+1:nx+np]
            term = (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)
            Gp = integral .+ term
            return Gp, integral, term
        end

        function get_term_debug(adj_sol::DAESolution, xps::Matrix{Float64}, times::AbstractVector{Float64})
            term = zeros(length(adj_sol.u))
            for ind=eachindex(adj_sol.u)
                term[ind] = ((adj_sol.u[end+1-ind][1:nx]')*Fdx(times[ind]))*xps[:,ind]
            end
            return term
        end
        debugs = (get_Gp_debug, get_term_debug)

        dvars = vcat(fill(true, 4), fill(false, 3), fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0 here, λ0: $λ0, dλ0: $dλ0"

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp, debugs
    end
end

# NOTE Assumes free dynamical parameters are only k
function NEGATED_my_pendulum_adjoint_konly(u::Function, w::Function, θ::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Vector{Float64}, dx::Function, dx2::Function, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function.
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        @assert (np == 1) "my_pendulum_adjoint_konly is hard-coded to only handle one parameter k, make sure to pass correct xp0"
        nx = size(xp0,1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        Fx = t -> [2dx(t)[6]        0               0   -1                  0           0   0
                    0           2*dx(t)[6]          0   0                   -1          0   0
                    -dx(t)[3]         0             0   2k*abs(x(t)[4])     0           0   0
                    0            -dx(t)[3]          0   0               2k*abs(x(t)[5]) 0   0
                    2x(t)[1]      2x(t)[2]          0   0                   0           0   0
                    x(t)[4]        x(t)[5]          0   x(t)[1]            x(t)[2]      0   0
                    x(t)[2]/(L^2)  -x(t)[1]/(L^2)   0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = t -> vcat([1   0   0          0   0   2x(t)[1]    0
                         0   1   0          0   0   2x(t)[2]    0
                         0   0   -x(t)[1]   m   0   0           0
                         0   0   -x(t)[2]   0   m   0           0], zeros(3,7))
        Fddx = t -> vcat([  0   0  0            0   0   2dx(t)[1]    0
                            0   0  0            0   0   2dx(t)[2]    0
                            0   0  -dx(t)[1]    0   0   0            0
                            0   0  -dx(t)[2]    0   0   0            0], zeros(3,7))
        Fp = t -> [.0; .0; abs(x(t)[4])*x(t)[4]; abs(x(t)[5])*x(t)[5]; .0; .0; .0]
        gₓ  = t -> [0    0    0    0    0    0    2(x2(t)[7]-y(t))/T]
        gdₓ = t -> [0    0    0    0    0    0    2(dx2(t)[7]-dy(t))/T]

        # NOTE: Convention is used that derivatives wrt to θ stack along cols
        # while derivatives wrt to x stack along rows

        ###################### INITIALIZING ADJOINT SYSTEM ####################
        # Indices 1-4 are differential (d), while 5-7 are algebraic (a)
        dinds = 1:4
        ainds = 5:7
        λT  = zeros(7)
        dλT = zeros(7)
        temp = (-gₓ(T))/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        λT[dinds] = zeros(length(dinds))
        λT[ainds] = temp[ainds]
        dλT[dinds] = temp[dinds]
        temp = (-gdₓ(T) + (dλT[dinds]')*(Fx(T)[dinds,:] - Fddx(T)[dinds,:] - Fdx(T)[dinds,:]) + (λT[ainds]')*Fx(T)[ainds,:])/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        dλT[ainds] = temp[ainds]

        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            # Completely unreadabe but most efficient version (still not a huge improvement)
            res[1]  = dx(T-t)[3]*z[3] - dz[1] - 2*dx(T-t)[6]*z[1] - 2*z[5]*x(T-t)[1] - z[6]*x(T-t)[4] - (z[7]*x(T-t)[2])/(L^2)
            res[2]  = dx(T-t)[3]*z[4] - dz[2] - 2*dx(T-t)[6]*z[2] - 2*z[5]*x(T-t)[2] - z[6]*x(T-t)[5] + (z[7]*x(T-t)[1])/(L^2)
            res[3]  = dz[3]*x(T-t)[1] - dx(T-t)[2]*z[4] - dx(T-t)[1]*z[3] + dz[4]*x(T-t)[2]
            res[4]  = z[1] - dz[3]*m - z[6]*x(T-t)[1] - 2*k*z[3]*abs(x(T-t)[4])
            res[5]  = z[2] - dz[4]*m - z[6]*x(T-t)[2] - 2*k*z[4]*abs(x(T-t)[5])
            res[6]  = 2*dx(T-t)[1]*z[1] + 2*dx(T-t)[2]*z[2] - 2*dz[1]*x(T-t)[1] - 2*dz[2]*x(T-t)[2]
            res[7]  = (2*(x2(T-t)[7] - y(T-t)))/T - z[7]
            res[8]  = dz[8] + z[3]*abs(x(T-t)[4])*x(T-t)[4] + z[4]*abs(x(T-t)[5])*x(T-t)[5]

            # # Medium readable but quite efficient version
            # λ  = z[1:nx]
            # dλ = dz[1:nx]
            # β  = z[nx+1:end]
            # dβ = dz[nx+1:end]
            # res[1]  = dx(T-t)[3]*λ[3] - dλ[1] - 2*dx(T-t)[6]*λ[1] - 2*λ[5]*x(T-t)[1] - λ[6]*x(T-t)[4] - (λ[7]*x(T-t)[2])/(L^2)
            # res[2]  = dx(T-t)[3]*λ[4] - dλ[2] - 2*dx(T-t)[6]*λ[2] - 2*λ[5]*x(T-t)[2] - λ[6]*x(T-t)[5] + (λ[7]*x(T-t)[1])/(L^2)
            # res[3]  = dλ[3]*x(T-t)[1] - dx(T-t)[2]*λ[4] - dx(T-t)[1]*λ[3] + dλ[4]*x(T-t)[2]
            # res[4]  = λ[1] - dλ[3]*m - λ[6]*x(T-t)[1] - 2*k*λ[3]*abs(x(T-t)[4])
            # res[5]  = λ[2] - dλ[4]*m - λ[6]*x(T-t)[2] - 2*k*λ[4]*abs(x(T-t)[5])
            # res[6]  = 2*dx(T-t)[1]*λ[1] + 2*dx(T-t)[2]*λ[2] - 2*dλ[1]*x(T-t)[1] - 2*dλ[2]*x(T-t)[2]
            # res[7]  = (2*(x2(T-t)[7] - y(T-t)))/T - λ[7]
            # res[8]  = dβ[1] + λ[3]*abs(x(T-t)[4])*x(T-t)[4] + λ[4]*abs(x(T-t)[5])*x(T-t)[5]

            # # Readable but slightly less efficient version
            # λ  = z[1:nx]
            # dλ = dz[1:nx]
            # β  = z[nx+1:end]
            # dβ = dz[nx+1:end]
            # res[1:7] = -(dλ')*Fdx(T-t) + (λ')*(Fddx(T-t)-Fx(T-t)) + gₓ(T-t)
            # res[8]  = dβ[1]+(λ')*Fp(T-t)

            # # Super-readable but less efficient version
            # res[1:7] = (-dz[1:7]')*Fdx(T-t) + (z[1:7]')*(Fddx(T-t) - Fx(T-t)) + gₓ(T-t)
            # res[8]   = dz[8] + (z[1:7]')*Fp(T-t)
            nothing
        end

        # Solving backwards in time =>
        # 1. Initial conditions will equal terminal conditions
        # 2. All derivatives have to be negated
        λ0  = λT[:]
        dλ0 = -dλT[:]
        # z0 = λ0
        # dz0 = dλ0
        z0  = vcat(λ0, zeros(np))
        dz0 = vcat(dλ0, -(λT')*Fp(T))   # For some reasing (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired
        # TODO: Can we delete these row below here perhaps?
        # z0  = vcat(λ0, 0.0)
        # dz0 = vcat(dλ0, λ0[1])   # For some reasing (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired

        if N_trans > 0
            @warn "The returned function get_Gp() doesn't fully support N_trans > 0, as sensitivity of internal variables not known at any other time than t=0. A non-rigorous approximation is used instead."
        end
        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            # NOTE: Changes signs to match what I had in my manual calculations, seems correct now
            # Gp = adj_sol.u[end][end-np+1:end] + (((adj_sol.u[end][1:end-np]')*Fdx(0.0))*xp0)[:]
            # Gp = adj_sol.u[end-N_trans][nx+1:nx+np] + (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)[:]
            Gp = adj_sol.u[end-N_trans][nx+1:nx+np] .+ (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)
            # return 0.0
        end

        dvars = vcat(fill(true, 4), fill(false, 3), fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0 here, λ0: $λ0, dλ0: $dλ0"

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp
    end
end

# NOTE Assumes free dynamical parameters are only k
function my_pendulum_adjoint_konly(u::Function, w::Function, θ::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Matrix{Float64}, dx::Function, dx2::Function, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function.
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        @assert (np == 1) "my_pendulum_adjoint_konly is hard-coded to only handle one parameter k, make sure to pass correct xp0"
        nx = size(xp0,1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        Fx = t -> [2dx(t)[6]        0               0   -1                  0           0   0
                    0           2*dx(t)[6]          0   0                   -1          0   0
                    -dx(t)[3]         0             0   2k*abs(x(t)[4])     0           0   0
                    0            -dx(t)[3]          0   0               2k*abs(x(t)[5]) 0   0
                    2x(t)[1]      2x(t)[2]          0   0                   0           0   0
                    x(t)[4]        x(t)[5]          0   x(t)[1]            x(t)[2]      0   0
                    x(t)[2]/(L^2)  -x(t)[1]/(L^2)   0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = t -> vcat([1   0   0          0   0   2x(t)[1]    0
                         0   1   0          0   0   2x(t)[2]    0
                         0   0   -x(t)[1]   m   0   0           0
                         0   0   -x(t)[2]   0   m   0           0], zeros(3,7))
        Fddx = t -> vcat([  0   0  0            0   0   2dx(t)[1]    0
                            0   0  0            0   0   2dx(t)[2]    0
                            0   0  -dx(t)[1]    0   0   0            0
                            0   0  -dx(t)[2]    0   0   0            0], zeros(3,7))
        Fp = t -> [.0; .0; abs(x(t)[4])*x(t)[4]; abs(x(t)[5])*x(t)[5]; .0; .0; .0]
        gₓ  = t -> [0.0    0.0    0.0    0.0    0.0    0.0    2(x2(t)[7]-first(y(t)))/T]
        gdₓ = t -> [0.0    0.0    0.0    0.0    0.0    0.0    2(dx2(t)[7]-first(dy(t)))/T]

        # NOTE: Convention is used that derivatives wrt to θ stack along cols
        # while derivatives wrt to x stack along rows

        ###################### INITIALIZING ADJOINT SYSTEM ####################
        # Indices 1-4 are differential (d), while 5-7 are algebraic (a)
        dinds = 1:4
        ainds = 5:7
        λT  = zeros(7)
        dλT = zeros(7)
        temp = (-gₓ(T))/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        λT[dinds] = zeros(length(dinds))
        λT[ainds] = temp[ainds]
        dλT[dinds] = temp[dinds]
        temp = (-gdₓ(T) + (dλT[dinds]')*(Fx(T)[dinds,:] - Fddx(T)[dinds,:] - Fdx(T)[dinds,:]) + (λT[ainds]')*Fx(T)[ainds,:])/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        dλT[ainds] = temp[ainds]

        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            # Completely unreadabe but most efficient version (still not a huge improvement)
            res[1]  = dz[1] - 2*dx(t)[6]*z[1] + dx(t)[3]*z[3] - 2*x(t)[1]*z[5] - x(t)[4]*z[6] - (x(t)[2]*z[7])/(L^2)
            res[2]  = dz[2] - 2*dx(t)[6]*z[2] + dx(t)[3]*z[4] - 2*x(t)[2]*z[5] - x(t)[5]*z[6] + (x(t)[1]*z[7])/(L^2)
            res[3]  = -x(t)[1]*dz[3] - x(t)[2]*dz[4] - dx(t)[1]*z[3] - dx(t)[2]*z[4]
            res[4]  = m*dz[3] + z[1] - 2*k*abs(x(t)[4])*z[3] - x(t)[1]*z[6]
            res[5]  = m*dz[4] + z[2] - 2*k*abs(x(t)[5])*z[4] - x(t)[2]*z[6]
            res[6]  = 2*x(t)[1]*dz[1] + 2*x(t)[2]*dz[2] + 2*dx(t)[1]*z[1] + 2*dx(t)[2]*z[2]
            res[7]  = (2*(x2(t)[7] - first(y(t))))/T - z[7]
            # res[1]  = dz[1] + 2x(t)[1]*dz[6]    - 2dx(t)[6]*z[1] + z[4] + 2dx(t)[1]*z[6]
            # res[2]  = dz[2] + 2x(t)[2]*dz[6]    - 2dx(t)[6]*z[2] + z[5] + 2dx(t)[2]*z[6]
            # res[3]  = -x(t)[1]*dz[3] + m*dz[4]  + dx(t)[3]*z[1] - dx(t)[1]*z[3] - 2k*abs(x(t)[4])*z[4]
            # res[4]  = -x(t)[2]*dz[3] + m*dz[5]  + dx(t)[3]*z[2] - dx(t)[2]*z[3] - 2k*abs(x(t)[5])*z[5]
            # res[5]  =                           - 2x(t)[1]*z[1] - 2x(t)[2]*z[2]
            # res[6]  =                           - x(t)[4]*z[1] - x(t)[5]*z[2] - x(t)[1]*z[4] - x(t)[2]*z[5]
            # res[7]  =                           - x(t)[2]*z[1]/(L^2) + x(t)[1]*z[2]/(L^2) - z[7]                + (2/T)*(x2(t)[7]-y(t))
            res[8]  = dz[8] - z[3]*abs(x(t)[4])*x(t)[4] - z[4]*abs(x(t)[5])*x(t)[5]

            # # Super-readable but less efficient version
            # res[1:7] = (dz[1:7]')*Fdx(t) + (z[1:7]')*(Fddx(t) - Fx(t)) + gₓ(t)
            # res[8]   = dz[8] - (z[1:7]')*Fp(t)
            nothing
        end

        z0  = vcat(λT[:], zeros(np))
        dz0 = vcat(dλT[:], (λT')*Fp(T))   # For some reason (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired

        if N_trans > 0
            @warn "The returned function get_Gp() doesn't fully support N_trans > 0, as sensitivity of internal variables not known at any other time than t=0. A non-rigorous approximation is used instead."
        end
        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            # NOTE: Changes signs to match what I had in my manual calculations, seems correct now
            # Gp = adj_sol.u[end][end-np+1:end] + (((adj_sol.u[end][1:end-np]')*Fdx(0.0))*xp0)[:]
            # Gp = adj_sol.u[end-N_trans][nx+1:nx+np] + (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)[:]
            Gp = adj_sol.u[end-N_trans][nx+1:nx+np] .+ (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)
            # return 0.0
        end

        # The Gp returned by this function is the same as the Gp returned by get_Gp()
        function get_Gp_debug(adj_sol::DAESolution)
            integral = adj_sol.u[end][nx+1:nx+np]
            term = (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)
            Gp = integral .+ term
            return Gp, integral, term
        end

        function get_term_debug(adj_sol::DAESolution, xps::Matrix{Float64}, times::AbstractVector{Float64})
            term = zeros(length(adj_sol.u))
            for ind=eachindex(adj_sol.u)
                term[ind] = ((adj_sol.u[end+1-ind][1:nx]')*Fdx(times[ind]))*xps[:,ind]
            end
            return term
        end

        dvars = vcat(fill(true, 4), fill(false, 3), fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0 here, λ0: $λ0, dλ0: $dλ0"
        debugs = (get_Gp_debug, get_term_debug)

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp, debugs
    end
end

# NOTE Assumes free dynamical parameters are only k. Also identifies a1 and a2 from disturbance model
function my_pendulum_adjoint_konly_with_distsensa(u::Function, w::Function, xw::Function, v::Function, θ::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Matrix{Float64}, dx::Function, dx2::Function, B̃::Matrix{Float64}, B̃θ::Matrix{Float64}, η::Vector{Float64}, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function. 
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        nη = 2
        nx = size(xp0,1)
        nw = length(xw(0.0))
        @assert (np == 3) "my_pendulum_adjoint_konly_with_distsensa is hard-coded to only handle parameters k, a1, and a2. Make sure to pass correct xp0"
        ndist = 3   # length of x_w plus length of w (2+1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        Fx = t -> [2dx(t)[6]        0               0   -1                  0           0   0
                    0           2*dx(t)[6]          0   0                   -1          0   0
                    -dx(t)[3]         0             0   2k*abs(x(t)[4])     0           0   0
                    0            -dx(t)[3]          0   0               2k*abs(x(t)[5]) 0   0
                    2x(t)[1]      2x(t)[2]          0   0                   0           0   0
                    x(t)[4]        x(t)[5]          0   x(t)[1]            x(t)[2]      0   0
                    x(t)[2]/(L^2)  -x(t)[1]/(L^2)   0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = t -> vcat([1   0   0          0   0   2x(t)[1]    0
                         0   1   0          0   0   2x(t)[2]    0
                         0   0   -x(t)[1]   m   0   0           0
                         0   0   -x(t)[2]   0   m   0           0], zeros(3,7))
        Fddx = t -> vcat([  0   0  0            0   0   2dx(t)[1]    0
                            0   0  0            0   0   2dx(t)[2]    0
                            0   0  -dx(t)[1]    0   0   0            0
                            0   0  -dx(t)[2]    0   0   0            0], zeros(3,7))
        Fp = t -> [.0; .0; abs(x(t)[4])*x(t)[4]; abs(x(t)[5])*x(t)[5]; .0; .0; .0]
        gₓ  = t -> [0    0    0    0    0    0    2(x2(t)[7]-y(t))/T]
        gdₓ = t -> [0    0    0    0    0    0    2(dx2(t)[7]-dy(t))/T]

        # NOTE: Convention is used that derivatives wrt to θ stack along cols
        # while derivatives wrt to x stack along rows

        # Creating some of the needed disturbance from η. ASSUMING ONLY DISTURBANCE PARAMETERS ARE a1 AND a2!!!!!!
        A = [-η[1]  -η[2]; 1.0   0.0]
        C = [η[3]   η[4]]
        Aθ = [-1.0  0.0; 0.0    0.0; 0.0    -1.0; 0.0   0.0]
        Cθ = zeros(2,nw) # Only depends on C-parameters, and we only compute sensitivities wrt to a-parameters. We have 2 disturbance parameters

        ###################### INITIALIZING ADJOINT SYSTEM ####################
        # Indices 1-4 are differential (d), while 5-7 are algebraic (a)
        dinds  = 1:4
        ainds  = 5:7
        λinds  = 1:7
        xwinds = 8:9
        winds  = 10
        λT  = zeros(10)
        dλT = zeros(10)
        temp = (-gₓ(T))/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        λT[dinds] = zeros(length(dinds))
        λT[ainds] = temp[ainds]
        dλT[dinds] = temp[dinds]
        temp = (-gdₓ(T) + (dλT[dinds]')*(Fx(T)[dinds,:] - Fddx(T)[dinds,:] - Fdx(T)[dinds,:]) + (λT[ainds]')*Fx(T)[ainds,:])/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        dλT[ainds] = temp[ainds]
        # Terminal conditions due to disturbance id
        λT[xwinds]  = zeros(length(xwinds))
        λT[winds]   = 2w(T)[1]*λT[3]
        dλT[xwinds] = (λT[xwinds]')*A + λT[winds]*C
        dλT[winds]  = 2w(T)[1]*dλT[3] + 2(C*(A*xw(T)+B̃*v(T)))[1]*λT[3]

        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            # Completely unreadabe but most efficient version (still not a huge improvement)
            res[1]  = dz[1] - 2*dx(t)[6]*z[1] + dx(t)[3]*z[3] - 2*x(t)[1]*z[5] - x(t)[4]*z[6] - (x(t)[2]*z[7])/(L^2)
            res[2]  = dz[2] - 2*dx(t)[6]*z[2] + dx(t)[3]*z[4] - 2*x(t)[2]*z[5] - x(t)[5]*z[6] + (x(t)[1]*z[7])/(L^2)
            res[3]  = -x(t)[1]*dz[3] - x(t)[2]*dz[4] - dx(t)[1]*z[3] - dx(t)[2]*z[4]
            res[4]  = m*dz[3] + z[1] - 2*k*abs(x(t)[4])*z[3] - x(t)[1]*z[6]
            res[5]  = m*dz[4] + z[2] - 2*k*abs(x(t)[5])*z[4] - x(t)[2]*z[6]
            res[6]  = 2*x(t)[1]*dz[1] + 2*x(t)[2]*dz[2] + 2*dx(t)[1]*z[1] + 2*dx(t)[2]*z[2]
            res[7]  = (2*(x2(t)[7] - y(t)))/T - z[7]

            # NOTE: η here then has to contain free_parameters and true values for the non-free ones
            res[8]  = dz[8] + z[9] - η[1]*z[8] + η[3]*z[10]
            res[9]  = dz[9] - η[2]*z[8] + η[4]*z[10]
            res[10] = 2w(t)[1]*z[3] - z[10]
            # #          | ------------ from original adjoint system --------------- || ----------------------- new terms ---------------------| 
            # res[11]  = dz[8] - z[3]*abs(x(t)[4])*x(t)[4] - z[4]*abs(x(t)[5])*x(t)[5]  + (z[8:9]')*(Aθ*xw(t) - B̃θ*v(t)) + z[10]*(Cθ*xw(t))
            res[11]  = dz[11] - z[3]*abs(x(t)[4])*x(t)[4] - z[4]*abs(x(t)[5])*x(t)[5]
            res[12]  = dz[12] + (z[8:9]')*(Aθ[1:nw,:]*xw(t) - B̃θ[1:nw,:]*v(t)) # + z[10]*(Cθ*xw(t)) # This last part is equal to zero since we don't have disturbance parameters
            res[13]  = dz[13] + (z[8:9]')*(Aθ[nw+1:2nw,:]*xw(t) - B̃θ[nw+1:2nw,:]*v(t))

            # # Super-readable but less efficient version
            # res[1:7] = (dz[1:7]')*Fdx(t) + (z[1:7]')*(Fddx(t) - Fx(t)) + gₓ(t)
            # res[8]   = dz[8] - (z[1:7]')*Fp(t)
            nothing
        end
        
        z0  = vcat(λT[:], zeros(np))
        second_temp = Matrix{Float64}(undef, nw, nη)
        third_temp  = Matrix{Float64}(undef, 1, nη)
        for ind = 1:nη   # 2 disturbance parameters
            second_temp[:, ind] = Aθ[(ind-1)nw + 1: ind*nw, :]*xw(T) + B̃θ[(ind-1)nw + 1: ind*nw, :]*v(T)
            third_temp[:, ind]  = Cθ[ind:ind, :]*xw(T) # NOTE: SCALAR_OUTPUT is assumed
            # second_temp[:, (ind-1)nw + 1: ind*nw] = Aθ[(ind-1)nw + 1: ind*nw, :]*xw(T) + B̃θ[(ind-1)nw + 1: ind*nw, :]*v(T)
            # third_temp[:, (ind-1)nw + 1: ind]  = Cθ[ind:ind, :]*xw(T) # NOTE: SCALAR_OUTPUT is assumed
        end
        my_temp = hcat([(λT[λinds]')*Fp(T)], - (λT[xwinds]')*second_temp - (λT[winds]')*third_temp)
        dz0 = vcat(dλT[:], my_temp[:])
        # dz0 = vcat(dλT[:], (λT[λinds]')*Fp(T) - (λT[xwinds]')*second_temp - (λT[winds]')*third_temp)   # For some reason (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired

        if N_trans > 0
            @warn "The returned function get_Gp() doesn't fully support N_trans > 0, as sensitivity of internal variables not known at any other time than t=0. A non-rigorous approximation is used instead."
        end
        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            Gp = adj_sol.u[end-N_trans][nx+ndist+1:nx+ndist+np] + (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)[:]     # This is the same as in original adjoint system just because disturbance model has zero initial conditions, independent of parameters
        end

        dvars = vcat(fill(true, 4), fill(false, 3), fill(true, 2), [false], fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0 here, λ0: $λ0, dλ0: $dλ0"

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp
    end
end

# TODO: PARTS OF THIS SEEM TO ASSUME a1 AND a2 ARE PARAMTERS, THAT'S NOT RIGHT IS IT! FIX IT :D
# NOTE Assumes free dynamical parameters are only k. Also identifies a1 and a2 from disturbance model
function my_pendulum_adjoint_konly_with_distsensa1(u::Function, w::Function, xw::Function, v::Function, θ::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Matrix{Float64}, dx::Function, dx2::Function, B̃::Matrix{Float64}, B̃θ::Matrix{Float64}, η::Vector{Float64}, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function. 
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        nη = 1
        nx = size(xp0,1)
        nw = length(xw(0.0))
        @assert (np == 2) "my_pendulum_adjoint_konly_with_distsensa is hard-coded to only handle parameters k, and a1. Make sure to pass correct xp0"
        ndist = 3   # length of x_w plus length of w (2+1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        Fx = t -> [2dx(t)[6]        0               0   -1                  0           0   0
                    0           2*dx(t)[6]          0   0                   -1          0   0
                    -dx(t)[3]         0             0   2k*abs(x(t)[4])     0           0   0
                    0            -dx(t)[3]          0   0               2k*abs(x(t)[5]) 0   0
                    2x(t)[1]      2x(t)[2]          0   0                   0           0   0
                    x(t)[4]        x(t)[5]          0   x(t)[1]            x(t)[2]      0   0
                    x(t)[2]/(L^2)  -x(t)[1]/(L^2)   0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = t -> vcat([1   0   0          0   0   2x(t)[1]    0
                         0   1   0          0   0   2x(t)[2]    0
                         0   0   -x(t)[1]   m   0   0           0
                         0   0   -x(t)[2]   0   m   0           0], zeros(3,7))
        Fddx = t -> vcat([  0   0  0            0   0   2dx(t)[1]    0
                            0   0  0            0   0   2dx(t)[2]    0
                            0   0  -dx(t)[1]    0   0   0            0
                            0   0  -dx(t)[2]    0   0   0            0], zeros(3,7))
        Fp = t -> [.0; .0; abs(x(t)[4])*x(t)[4]; abs(x(t)[5])*x(t)[5]; .0; .0; .0]
        gₓ  = t -> [0    0    0    0    0    0    2(x2(t)[7]-y(t))/T]
        gdₓ = t -> [0    0    0    0    0    0    2(dx2(t)[7]-dy(t))/T]

        # NOTE: Convention is used that derivatives wrt to θ stack along cols
        # while derivatives wrt to x stack along rows

        # Creating some of the needed disturbance from η.
        A = [-η[1]  -η[2]; 1.0   0.0]
        C = [η[3]   η[4]]
        Aθ = [-1.0  0.0; 0.0    0.0; 0.0    -1.0; 0.0   0.0]
        Cθ = zeros(1,nw) # Only depends on C-parameters, and we only compute sensitivities wrt to a-parameters. We have 1 disturbance parameter, thus 1 row (I think, added this later)

        ###################### INITIALIZING ADJOINT SYSTEM ####################
        # Indices 1-4 are differential (d), while 5-7 are algebraic (a)
        dinds  = 1:4
        ainds  = 5:7
        λinds  = 1:7
        xwinds = 8:9
        winds  = 10
        λT  = zeros(10)
        dλT = zeros(10)
        temp = (-gₓ(T))/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        λT[dinds] = zeros(length(dinds))
        λT[ainds] = temp[ainds]
        dλT[dinds] = temp[dinds]
        temp = (-gdₓ(T) + (dλT[dinds]')*(Fx(T)[dinds,:] - Fddx(T)[dinds,:] - Fdx(T)[dinds,:]) + (λT[ainds]')*Fx(T)[ainds,:])/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        dλT[ainds] = temp[ainds]
        # Terminal conditions due to disturbance id
        λT[xwinds]  = zeros(length(xwinds))
        λT[winds]   = 2w(T)[1]*λT[3]
        dλT[xwinds] = (λT[xwinds]')*A + λT[winds]*C
        dλT[winds]  = 2w(T)[1]*dλT[3] + 2(C*(A*xw(T)+B̃*v(T)))[1]*λT[3]

        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            # Completely unreadabe but most efficient version (still not a huge improvement)
            res[1]  = dz[1] - 2*dx(t)[6]*z[1] + dx(t)[3]*z[3] - 2*x(t)[1]*z[5] - x(t)[4]*z[6] - (x(t)[2]*z[7])/(L^2)
            res[2]  = dz[2] - 2*dx(t)[6]*z[2] + dx(t)[3]*z[4] - 2*x(t)[2]*z[5] - x(t)[5]*z[6] + (x(t)[1]*z[7])/(L^2)
            res[3]  = -x(t)[1]*dz[3] - x(t)[2]*dz[4] - dx(t)[1]*z[3] - dx(t)[2]*z[4]
            res[4]  = m*dz[3] + z[1] - 2*k*abs(x(t)[4])*z[3] - x(t)[1]*z[6]
            res[5]  = m*dz[4] + z[2] - 2*k*abs(x(t)[5])*z[4] - x(t)[2]*z[6]
            res[6]  = 2*x(t)[1]*dz[1] + 2*x(t)[2]*dz[2] + 2*dx(t)[1]*z[1] + 2*dx(t)[2]*z[2]
            res[7]  = (2*(x2(t)[7] - y(t)))/T - z[7]

            # NOTE: η here then has to contain free_parameters and true values for the non-free ones
            res[8]  = dz[8] + z[9] - η[1]*z[8] + η[3]*z[10]
            res[9]  = dz[9] - η[2]*z[8] + η[4]*z[10]
            res[10] = 2w(t)[1]*z[3] - z[10]
            # #          | ------------ from original adjoint system --------------- || ----------------------- new terms ---------------------| 
            # res[11]  = dz[8] - z[3]*abs(x(t)[4])*x(t)[4] - z[4]*abs(x(t)[5])*x(t)[5]  + (z[8:9]')*(Aθ*xw(t) - B̃θ*v(t)) + z[10]*(Cθ*xw(t))
            res[11]  = dz[11] - z[3]*abs(x(t)[4])*x(t)[4] - z[4]*abs(x(t)[5])*x(t)[5]   # I don't remember anymore, but those "new terms" above have zero contribution to this line I think
            res[12]  = dz[12] + (z[8:9]')*(Aθ[1:nw,:]*xw(t) - B̃θ[1:nw,:]*v(t)) # + z[10]*(Cθ*xw(t)) # This last part is equal to zero since we don't have disturbance parameters
            # res[13]  = dz[13] + (z[8:9]')*(Aθ[nw+1:2nw,:]*xw(t) - B̃θ[nw+1:2nw,:]*v(t))

            # # Super-readable but less efficient version
            # res[1:7] = (dz[1:7]')*Fdx(t) + (z[1:7]')*(Fddx(t) - Fx(t)) + gₓ(t)
            # res[8]   = dz[8] - (z[1:7]')*Fp(t)
            nothing
        end
        
        z0  = vcat(λT[:], zeros(np))
        second_temp = Matrix{Float64}(undef, nw, nη)
        third_temp  = Matrix{Float64}(undef, 1, nη)
        for ind = 1:nη   # 2 disturbance parameters
            second_temp[:, ind] = Aθ[(ind-1)nw + 1: ind*nw, :]*xw(T) + B̃θ[(ind-1)nw + 1: ind*nw, :]*v(T)
            third_temp[:, ind]  = Cθ[ind:ind, :]*xw(T) # NOTE: SCALAR_OUTPUT is assumed
            # second_temp[:, (ind-1)nw + 1: ind*nw] = Aθ[(ind-1)nw + 1: ind*nw, :]*xw(T) + B̃θ[(ind-1)nw + 1: ind*nw, :]*v(T)
            # third_temp[:, (ind-1)nw + 1: ind]  = Cθ[ind:ind, :]*xw(T) # NOTE: SCALAR_OUTPUT is assumed
        end
        my_temp = hcat([(λT[λinds]')*Fp(T)], - (λT[xwinds]')*second_temp - (λT[winds]')*third_temp)
        dz0 = vcat(dλT[:], my_temp[:])
        # dz0 = vcat(dλT[:], (λT[λinds]')*Fp(T) - (λT[xwinds]')*second_temp - (λT[winds]')*third_temp)   # For some reason (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired

        if N_trans > 0
            @warn "The returned function get_Gp() doesn't fully support N_trans > 0, as sensitivity of internal variables not known at any other time than t=0. A non-rigorous approximation is used instead."
        end
        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            Gp = adj_sol.u[end-N_trans][nx+ndist+1:nx+ndist+np] + (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)[:]     # This is the same as in original adjoint system just because disturbance model has zero initial conditions, independent of parameters
        end

        dvars = vcat(fill(true, 4), fill(false, 3), fill(true, 2), [false], fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0 here, λ0: $λ0, dλ0: $dλ0"

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp
    end
end

function my_pendulum_adjoint_sans_g(u::Function, w::Function, θ::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Matrix{Float64}, dx::Function, dx2::Function, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function.
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        nx = size(xp0,1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        Fx = t -> [2dx(t)[6]        0               0   -1                  0           0   0
                    0           2*dx(t)[6]          0   0                   -1          0   0
                    -dx(t)[3]         0             0   2k*abs(x(t)[4])     0           0   0
                    0            -dx(t)[3]          0   0               2k*abs(x(t)[5]) 0   0
                    2x(t)[1]      2x(t)[2]          0   0                   0           0   0
                    x(t)[4]        x(t)[5]          0   x(t)[1]            x(t)[2]      0   0
                    x(t)[2]/(L^2)  -x(t)[1]/(L^2)   0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = t -> vcat([1   0   0          0   0   2x(t)[1]    0
                         0   1   0          0   0   2x(t)[2]    0
                         0   0   -x(t)[1]   m   0   0           0
                         0   0   -x(t)[2]   0   m   0           0], zeros(3,7))
        Fddx = t -> vcat([  0   0  0            0   0   2dx(t)[1]    0
                            0   0  0            0   0   2dx(t)[2]    0
                            0   0  -dx(t)[1]    0   0   0            0
                            0   0  -dx(t)[2]    0   0   0            0], zeros(3,7))

        Fp = t -> [ .0          .0          .0
                    .0          .0          .0
                    dx(t)[4]    .0          abs(x(t)[4])*x(t)[4]
                    dx(t)[5]+g  .0          abs(x(t)[5])*x(t)[5]
                    .0          -2L         .0
                    .0          .0          .0
                    .0          .0          .0]

        gₓ  = t -> [0    0    0    0    0    0    2(x2(t)[7]-y(t))/T]
        gdₓ = t -> [0    0    0    0    0    0    2(dx2(t)[7]-dy(t))/T]

        # NOTE: Convention is used that derivatives wrt to θ stack along cols
        # while derivatives wrt to x stack along rows

        ###################### INITIALIZING ADJOINT SYSTEM ####################
        # Indices 1-4 are differential (d), while 5-7 are algebraic (a)
        dinds = 1:4
        ainds = 5:7
        λT  = zeros(7)
        dλT = zeros(7)
        temp = (-gₓ(T))/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        λT[dinds] = zeros(length(dinds))
        λT[ainds] = temp[ainds]
        dλT[dinds] = temp[dinds]
        temp = (-gdₓ(T) + (dλT[dinds]')*(Fx(T)[dinds,:] - Fddx(T)[dinds,:] - Fdx(T)[dinds,:]) + (λT[ainds]')*Fx(T)[ainds,:])/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        dλT[ainds] = temp[ainds]

        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            # Completely unreadabe but most efficient version (still not a huge improvement)
            res[1]  = dz[1] - 2*dx(t)[6]*z[1] + dx(t)[3]*z[3] - 2*x(t)[1]*z[5] - x(t)[4]*z[6] - (x(t)[2]*z[7])/(L^2)
            res[2]  = dz[2] - 2*dx(t)[6]*z[2] + dx(t)[3]*z[4] - 2*x(t)[2]*z[5] - x(t)[5]*z[6] + (x(t)[1]*z[7])/(L^2)
            res[3]  = -x(t)[1]*dz[3] - x(t)[2]*dz[4] - dx(t)[1]*z[3] - dx(t)[2]*z[4]
            res[4]  = m*dz[3] + z[1] - 2*k*abs(x(t)[4])*z[3] - x(t)[1]*z[6]
            res[5]  = m*dz[4] + z[2] - 2*k*abs(x(t)[5])*z[4] - x(t)[2]*z[6]
            res[6]  = 2*x(t)[1]*dz[1] + 2*x(t)[2]*dz[2] + 2*dx(t)[1]*z[1] + 2*dx(t)[2]*z[2]
            res[7]  = (2*(x2(t)[7] - y(t)))/T - z[7]
            # res[1]  = dz[1] + 2x(t)[1]*dz[6]    - 2dx(t)[6]*z[1] + z[4] + 2dx(t)[1]*z[6]
            # res[2]  = dz[2] + 2x(t)[2]*dz[6]    - 2dx(t)[6]*z[2] + z[5] + 2dx(t)[2]*z[6]
            # res[3]  = -x(t)[1]*dz[3] + m*dz[4]  + dx(t)[3]*z[1] - dx(t)[1]*z[3] - 2k*abs(x(t)[4])*z[4]
            # res[4]  = -x(t)[2]*dz[3] + m*dz[5]  + dx(t)[3]*z[2] - dx(t)[2]*z[3] - 2k*abs(x(t)[5])*z[5]
            # res[5]  =                           - 2x(t)[1]*z[1] - 2x(t)[2]*z[2]
            # res[6]  =                           - x(t)[4]*z[1] - x(t)[5]*z[2] - x(t)[1]*z[4] - x(t)[2]*z[5]
            # res[7]  =                           - x(t)[2]*z[1]/(L^2) + x(t)[1]*z[2]/(L^2) - z[7]                + (2/T)*(x2(t)[7]-y(t))
            res[8]  = dz[8] - z[3]*dx(t)[4] - z[4]*(dx(t)[5]+g)
            res[9]  = dz[9] + 2*L*z[5]
            res[10]  = dz[10] - z[3]*abs(x(t)[4])*x(t)[4] - z[4]*abs(x(t)[5])*x(t)[5]

            # # Super-readable but less efficient version
            # res[1:7] = (dz[1:7]')*Fdx(t) + (z[1:7]')*(Fddx(t) - Fx(t)) + gₓ(t)
            # res[8]   = dz[8] - (z[1:7]')*Fp(t)
            nothing
        end

        z0  = vcat(λT[:], zeros(np))
        dz0 = vcat(dλT[:], ((λT')*Fp(T))[:])   # For some reasing (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired
        # TODO: Can we delete these row below here perhaps?
        # z0  = vcat(λ0, 0.0)
        # dz0 = vcat(dλ0, λ0[1])   # For some reasing (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired

        if N_trans > 0
            @warn "The returned function get_Gp() doesn't fully support N_trans > 0, as sensitivity of internal variables not known at any other time than t=0. A non-rigorous approximation is used instead."
        end
        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            # NOTE: Changes signs to match what I had in my manual calculations, seems correct now
            # Gp = adj_sol.u[end][end-np+1:end] + (((adj_sol.u[end][1:end-np]')*Fdx(0.0))*xp0)[:]
            # Gp = adj_sol.u[end-N_trans][nx+1:nx+np] + (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)[:]
            Gp = adj_sol.u[end-N_trans][nx+1:nx+np] .+ (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)[:]
            # return 0.0
        end

        dvars = vcat(fill(true, 4), fill(false, 3), fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0 here, λ0: $λ0, dλ0: $dλ0"

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp
    end
end

function my_pendulum_adjoint_sans_g_with_dist_sens_3(u::Function, w::Function, xw::Function, v::Function, θ::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Matrix{Float64}, dx::Function, dx2::Function, B̃::Matrix{Float64}, B̃θ::Matrix{Float64}, η::Vector{Float64}, na::Int, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function.
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        nη = 3
        nx = size(xp0,1)
        nw = length(xw(0.0))
        @assert (np == 6) "my_pendulum_adjoint_konly_with_distsensa is hard-coded to only handle parameters m, L, k, a1, a2, and c1. Make sure to pass correct xp0"
        ndist = 3   # length of x_w plus length of w (2+1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        Fx = t -> [2dx(t)[6]        0               0   -1                  0           0   0
                    0           2*dx(t)[6]          0   0                   -1          0   0
                    -dx(t)[3]         0             0   2k*abs(x(t)[4])     0           0   0
                    0            -dx(t)[3]          0   0               2k*abs(x(t)[5]) 0   0
                    2x(t)[1]      2x(t)[2]          0   0                   0           0   0
                    x(t)[4]        x(t)[5]          0   x(t)[1]            x(t)[2]      0   0
                    x(t)[2]/(L^2)  -x(t)[1]/(L^2)   0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = t -> vcat([1   0   0          0   0   2x(t)[1]    0
                         0   1   0          0   0   2x(t)[2]    0
                         0   0   -x(t)[1]   m   0   0           0
                         0   0   -x(t)[2]   0   m   0           0], zeros(3,7))
        Fddx = t -> vcat([  0   0  0            0   0   2dx(t)[1]    0
                            0   0  0            0   0   2dx(t)[2]    0
                            0   0  -dx(t)[1]    0   0   0            0
                            0   0  -dx(t)[2]    0   0   0            0], zeros(3,7))

        Fp = t -> [ .0          .0          .0
                    .0          .0          .0
                    dx(t)[4]    .0          abs(x(t)[4])*x(t)[4]
                    dx(t)[5]+g  .0          abs(x(t)[5])*x(t)[5]
                    .0          -2L         .0
                    .0          .0          .0
                    .0          .0          .0]

        gₓ  = t -> [0    0    0    0    0    0    2(x2(t)[7]-y(t))/T]
        gdₓ = t -> [0    0    0    0    0    0    2(dx2(t)[7]-dy(t))/T]

        # NOTE: Convention is used that derivatives wrt to θ stack along cols
        # while derivatives wrt to x stack along rows

        # Creating some of the needed disturbance from η.
        A = [-η[1]  -η[2]; 1.0   0.0]
        C = [η[3]   η[4]]
        Aθ = [-1.0  0.0; 0.0    0.0; 0.0    -1.0; 0.0   0.0; 0.0   0.0; 0.0  0.0]
        Cθ = vcat(zeros(2,nw), [0 1])

        ###################### INITIALIZING ADJOINT SYSTEM ####################
        # Indices 1-4 are differential (d), while 5-7 are algebraic (a)
        dinds = 1:4
        ainds = 5:7
        λinds  = 1:7
        xwinds = 8:9
        winds  = 10
        λT  = zeros(10)
        dλT = zeros(10)
        temp = (-gₓ(T))/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        λT[dinds] = zeros(length(dinds))
        λT[ainds] = temp[ainds]
        dλT[dinds] = temp[dinds]
        temp = (-gdₓ(T) + (dλT[dinds]')*(Fx(T)[dinds,:] - Fddx(T)[dinds,:] - Fdx(T)[dinds,:]) + (λT[ainds]')*Fx(T)[ainds,:])/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        dλT[ainds] = temp[ainds]
        # Terminal conditions due to disturbance id
        λT[xwinds]  = zeros(length(xwinds))
        λT[winds]   = 2w(T)[1]*λT[3]
        dλT[xwinds] = (λT[xwinds]')*A + λT[winds]*C
        dλT[winds]  = 2w(T)[1]*dλT[3] + 2(C*(A*xw(T)+B̃*v(T)))[1]*λT[3]

        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            # Completely unreadabe but most efficient version (still not a huge improvement)
            res[1]  = dz[1] - 2*dx(t)[6]*z[1] + dx(t)[3]*z[3] - 2*x(t)[1]*z[5] - x(t)[4]*z[6] - (x(t)[2]*z[7])/(L^2)
            res[2]  = dz[2] - 2*dx(t)[6]*z[2] + dx(t)[3]*z[4] - 2*x(t)[2]*z[5] - x(t)[5]*z[6] + (x(t)[1]*z[7])/(L^2)
            res[3]  = -x(t)[1]*dz[3] - x(t)[2]*dz[4] - dx(t)[1]*z[3] - dx(t)[2]*z[4]
            res[4]  = m*dz[3] + z[1] - 2*k*abs(x(t)[4])*z[3] - x(t)[1]*z[6]
            res[5]  = m*dz[4] + z[2] - 2*k*abs(x(t)[5])*z[4] - x(t)[2]*z[6]
            res[6]  = 2*x(t)[1]*dz[1] + 2*x(t)[2]*dz[2] + 2*dx(t)[1]*z[1] + 2*dx(t)[2]*z[2]
            res[7]  = (2*(x2(t)[7] - y(t)))/T - z[7]
            # res[1]  = dz[1] + 2x(t)[1]*dz[6]    - 2dx(t)[6]*z[1] + z[4] + 2dx(t)[1]*z[6]
            # res[2]  = dz[2] + 2x(t)[2]*dz[6]    - 2dx(t)[6]*z[2] + z[5] + 2dx(t)[2]*z[6]
            # res[3]  = -x(t)[1]*dz[3] + m*dz[4]  + dx(t)[3]*z[1] - dx(t)[1]*z[3] - 2k*abs(x(t)[4])*z[4]
            # res[4]  = -x(t)[2]*dz[3] + m*dz[5]  + dx(t)[3]*z[2] - dx(t)[2]*z[3] - 2k*abs(x(t)[5])*z[5]
            # res[5]  =                           - 2x(t)[1]*z[1] - 2x(t)[2]*z[2]
            # res[6]  =                           - x(t)[4]*z[1] - x(t)[5]*z[2] - x(t)[1]*z[4] - x(t)[2]*z[5]
            # res[7]  =                           - x(t)[2]*z[1]/(L^2) + x(t)[1]*z[2]/(L^2) - z[7]                + (2/T)*(x2(t)[7]-y(t))

            # Adjoint equations for xw and w, not β-equations
            # NOTE: η here then has to contain free_parameters and true values for the non-free ones
            res[8]  = dz[8] + z[9] - η[1]*z[8] + η[3]*z[10]
            res[9]  = dz[9] - η[2]*z[8] + η[4]*z[10]
            res[10] = 2w(t)[1]*z[3] - z[10] # TODO: Why do we need this if it's such a simple algebraic equation???
            # ----------------- β-equations ---------------------
            # For dynamics
            res[11]  = dz[11] - z[3]*dx(t)[4] - z[4]*(dx(t)[5]+g)
            res[12]  = dz[12] + 2*L*z[5]
            res[13]  = dz[13] - z[3]*abs(x(t)[4])*x(t)[4] - z[4]*abs(x(t)[5])*x(t)[5]
            # For disturbance
            res[14]  = dz[14] + (z[8:9]')*(Aθ[1:nw,:]*xw(t) - B̃θ[1:nw,:]*v(t))
            res[15]  = dz[15] + (z[8:9]')*(Aθ[nw+1:2nw,:]*xw(t) - B̃θ[nw+1:2nw,:]*v(t))
            res[16]  = dz[16] + z[10]*([0 1]⋅xw(t))

            # # Super-readable but less efficient version
            # res[1:7] = (dz[1:7]')*Fdx(t) + (z[1:7]')*(Fddx(t) - Fx(t)) + gₓ(t)
            # res[8]   = dz[8] - (z[1:7]')*Fp(t)
            nothing
        end

        z0  = vcat(λT[:], zeros(np))    # Appends terminal value for β-variables
        second_temp = Matrix{Float64}(undef, nw, nη)
        third_temp  = Matrix{Float64}(undef, 1, nη)
        for ind = 1:nη   # 3 disturbance parameters
            if ind <= na
                "Aθ and B̃θ are zero for θ corresponding to parameters in C-matrix"
                second_temp[:, ind] = Aθ[(ind-1)nw + 1: ind*nw, :]*xw(T) + B̃θ[(ind-1)nw + 1: ind*nw, :]*v(T)
            else
                second_temp[:, ind] = zeros(nw, 1)
            end
            third_temp[:, ind]  = Cθ[ind:ind, :]*xw(T) # NOTE: SCALAR_OUTPUT is assumed
        end
        my_temp = hcat((λT[λinds]')*Fp(T), - (λT[xwinds]')*second_temp - (λT[winds]')*third_temp)
        dz0 = vcat(dλT[:], my_temp[:])
        # @info "HERE SIZES: $(size(my_temp)), $(typeof(dz0)), $(size(dz0))"
        # dz0 = vcat(dλT[:], ((λT')*Fp(T))[:])  # OLD, DELETE

        if N_trans > 0
            @warn "The returned function get_Gp() doesn't fully support N_trans > 0, as sensitivity of internal variables not known at any other time than t=0. A non-rigorous approximation is used instead."
        end
        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            Gp = adj_sol.u[end-N_trans][nx+ndist+1:nx+ndist+np] .+ (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)[:]
            # return 0.0
        end

        dvars = vcat(fill(true, 4), fill(false, 3), fill(true, 2), [false], fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0 here, λ0: $λ0, dλ0: $dλ0"

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp
    end
end

function my_pendulum_adjoint_dist_sens_3(u::Function, w::Function, xw::Function, v::Function, θ::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Matrix{Float64}, dx::Function, dx2::Function, B̃::Matrix{Float64}, B̃θ::Matrix{Float64}, η::Vector{Float64}, na::Int, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function.
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        nη = 3
        nx = size(xp0,1)
        nw = length(xw(0.0))
        @assert (np == 3) "my_pendulum_adjoint_konly_with_distsensa is hard-coded to only handle parameters a1, a2, and c1. Make sure to pass correct xp0. Passed np=$np"
        ndist = 3   # length of x_w plus length of w (2+1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        Fx = t -> [2dx(t)[6]        0               0   -1                  0           0   0
                    0           2*dx(t)[6]          0   0                   -1          0   0
                    -dx(t)[3]         0             0   2k*abs(x(t)[4])     0           0   0
                    0            -dx(t)[3]          0   0               2k*abs(x(t)[5]) 0   0
                    2x(t)[1]      2x(t)[2]          0   0                   0           0   0
                    x(t)[4]        x(t)[5]          0   x(t)[1]            x(t)[2]      0   0
                    x(t)[2]/(L^2)  -x(t)[1]/(L^2)   0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = t -> vcat([1   0   0          0   0   2x(t)[1]    0
                         0   1   0          0   0   2x(t)[2]    0
                         0   0   -x(t)[1]   m   0   0           0
                         0   0   -x(t)[2]   0   m   0           0], zeros(3,7))
        Fddx = t -> vcat([  0   0  0            0   0   2dx(t)[1]    0
                            0   0  0            0   0   2dx(t)[2]    0
                            0   0  -dx(t)[1]    0   0   0            0
                            0   0  -dx(t)[2]    0   0   0            0], zeros(3,7))

        # Fp = t -> [ .0          .0          .0
        #             .0          .0          .0
        #             dx(t)[4]    .0          abs(x(t)[4])*x(t)[4]
        #             dx(t)[5]+g  .0          abs(x(t)[5])*x(t)[5]
        #             .0          -2L         .0
        #             .0          .0          .0
        #             .0          .0          .0]

        gₓ  = t -> [0    0    0    0    0    0    2(x2(t)[7]-y(t))/T]
        gdₓ = t -> [0    0    0    0    0    0    2(dx2(t)[7]-dy(t))/T]

        # NOTE: Convention is used that derivatives wrt to θ stack along cols
        # while derivatives wrt to x stack along rows

        # Creating some of the needed disturbance from η. ASSUMING ONLY DISTURBANCE PARAMETERS ARE a1 AND a2!!!!!!
        A = [-η[1]  -η[2]; 1.0   0.0]
        C = [η[3]   η[4]]
        Aθ = [-1.0  0.0; 0.0    0.0; 0.0    -1.0; 0.0   0.0; 0.0   0.0; 0.0  0.0]
        Cθ = vcat(zeros(2,nw), [0 1])

        ###################### INITIALIZING ADJOINT SYSTEM ####################
        # Indices 1-4 are differential (d), while 5-7 are algebraic (a)
        dinds = 1:4
        ainds = 5:7
        λinds  = 1:7
        xwinds = 8:9
        winds  = 10
        λT  = zeros(10)
        dλT = zeros(10)
        temp = (-gₓ(T))/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        λT[dinds] = zeros(length(dinds))
        λT[ainds] = temp[ainds]
        dλT[dinds] = temp[dinds]
        temp = (-gdₓ(T) + (dλT[dinds]')*(Fx(T)[dinds,:] - Fddx(T)[dinds,:] - Fdx(T)[dinds,:]) + (λT[ainds]')*Fx(T)[ainds,:])/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        dλT[ainds] = temp[ainds]
        # Terminal conditions due to disturbance id
        λT[xwinds]  = zeros(length(xwinds))
        λT[winds]   = 2w(T)[1]*λT[3]
        dλT[xwinds] = (λT[xwinds]')*A + λT[winds]*C
        dλT[winds]  = 2w(T)[1]*dλT[3] + 2(C*(A*xw(T)+B̃*v(T)))[1]*λT[3]

        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            # Completely unreadabe but most efficient version (still not a huge improvement)
            res[1]  = dz[1] - 2*dx(t)[6]*z[1] + dx(t)[3]*z[3] - 2*x(t)[1]*z[5] - x(t)[4]*z[6] - (x(t)[2]*z[7])/(L^2)
            res[2]  = dz[2] - 2*dx(t)[6]*z[2] + dx(t)[3]*z[4] - 2*x(t)[2]*z[5] - x(t)[5]*z[6] + (x(t)[1]*z[7])/(L^2)
            res[3]  = -x(t)[1]*dz[3] - x(t)[2]*dz[4] - dx(t)[1]*z[3] - dx(t)[2]*z[4]
            res[4]  = m*dz[3] + z[1] - 2*k*abs(x(t)[4])*z[3] - x(t)[1]*z[6]
            res[5]  = m*dz[4] + z[2] - 2*k*abs(x(t)[5])*z[4] - x(t)[2]*z[6]
            res[6]  = 2*x(t)[1]*dz[1] + 2*x(t)[2]*dz[2] + 2*dx(t)[1]*z[1] + 2*dx(t)[2]*z[2]
            res[7]  = (2*(x2(t)[7] - y(t)))/T - z[7]
            # res[1]  = dz[1] + 2x(t)[1]*dz[6]    - 2dx(t)[6]*z[1] + z[4] + 2dx(t)[1]*z[6]
            # res[2]  = dz[2] + 2x(t)[2]*dz[6]    - 2dx(t)[6]*z[2] + z[5] + 2dx(t)[2]*z[6]
            # res[3]  = -x(t)[1]*dz[3] + m*dz[4]  + dx(t)[3]*z[1] - dx(t)[1]*z[3] - 2k*abs(x(t)[4])*z[4]
            # res[4]  = -x(t)[2]*dz[3] + m*dz[5]  + dx(t)[3]*z[2] - dx(t)[2]*z[3] - 2k*abs(x(t)[5])*z[5]
            # res[5]  =                           - 2x(t)[1]*z[1] - 2x(t)[2]*z[2]
            # res[6]  =                           - x(t)[4]*z[1] - x(t)[5]*z[2] - x(t)[1]*z[4] - x(t)[2]*z[5]
            # res[7]  =                           - x(t)[2]*z[1]/(L^2) + x(t)[1]*z[2]/(L^2) - z[7]                + (2/T)*(x2(t)[7]-y(t))

            # Adjoint equations for xw and w, not β-equations
            # NOTE: η here then has to contain free_parameters and true values for the non-free ones
            res[8]  = dz[8] + z[9] - η[1]*z[8] + η[3]*z[10]
            res[9]  = dz[9] - η[2]*z[8] + η[4]*z[10]
            res[10] = 2w(t)[1]*z[3] - z[10] # TODO: Why do we need this if it's such a simple algebraic equation???
            # ----------------- β-equations ---------------------
            # # For dynamics
            # res[11]  = dz[11] - z[3]*dx(t)[4] - z[4]*(dx(t)[5]+g)
            # res[12]  = dz[12] + 2*L*z[5]
            # res[13]  = dz[13] - z[3]*abs(x(t)[4])*x(t)[4] - z[4]*abs(x(t)[5])*x(t)[5]
            # For disturbance
            res[11]  = dz[11] + (z[8:9]')*(Aθ[1:nw,:]*xw(t) - B̃θ[1:nw,:]*v(t))
            res[12]  = dz[12] + (z[8:9]')*(Aθ[nw+1:2nw,:]*xw(t) - B̃θ[nw+1:2nw,:]*v(t))
            res[13]  = dz[13] + z[10]*([0 1]⋅xw(t))

            # # Super-readable but less efficient version
            # res[1:7] = (dz[1:7]')*Fdx(t) + (z[1:7]')*(Fddx(t) - Fx(t)) + gₓ(t)
            # res[8]   = dz[8] - (z[1:7]')*Fp(t)
            nothing
        end

        z0  = vcat(λT[:], zeros(np))
        second_temp = Matrix{Float64}(undef, nw, nη)
        third_temp  = Matrix{Float64}(undef, 1, nη)
        for ind = 1:nη   # 3 disturbance parameters
            if ind <= na
                "Aθ and B̃θ are zero for θ corresponding to parameters in C-matrix"
                second_temp[:, ind] = Aθ[(ind-1)nw + 1: ind*nw, :]*xw(T) + B̃θ[(ind-1)nw + 1: ind*nw, :]*v(T)
            else
                second_temp[:, ind] = zeros(nw, 1)
            end
            third_temp[:, ind]  = Cθ[ind:ind, :]*xw(T) # NOTE: SCALAR_OUTPUT is assumed
        end
        # my_temp = hcat((λT[λinds]')*Fp(T), - (λT[xwinds]')*second_temp - (λT[winds]')*third_temp)
        my_temp = -(λT[xwinds]')*second_temp - (λT[winds]')*third_temp
        dz0 = vcat(dλT[:], my_temp[:])
        # @info "HERE SIZES: $(size(my_temp)), $(typeof(dz0)), $(size(dz0))"
        # dz0 = vcat(dλT[:], ((λT')*Fp(T))[:])  # OLD, DELETE

        if N_trans > 0
            @warn "The returned function get_Gp() doesn't fully support N_trans > 0, as sensitivity of internal variables not known at any other time than t=0. A non-rigorous approximation is used instead."
        end
        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            Gp = adj_sol.u[end-N_trans][nx+ndist+1:nx+ndist+np] .+ (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)[:]
            # return 0.0
        end

        dvars = vcat(fill(true, 4), fill(false, 3), fill(true, 2), [false], fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0 here, λ0: $λ0, dλ0: $dλ0"

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp
    end
end

function my_pendulum_adjoint_distsensa1(u::Function, w::Function, xw::Function, v::Function, θ::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Matrix{Float64}, dx::Function, dx2::Function, B̃::Matrix{Float64}, B̃θ::Matrix{Float64}, η::Vector{Float64}, na::Int, N_trans::Int=0)
    # na should be the number of disturbance parameters corresponding to the a parameters. In this case, it should really just be 1, so it's actually only an input argument to fit the signature of the other adjoint models
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        nη = 1
        nx = size(xp0,1)
        nw = length(xw(0.0))
        @assert (np == 1) "my_pendulum_adjoint_konly_with_distsensa is hard-coded to only handle parameter a1. Make sure to pass correct xp0"
        ndist = 3   # length of x_w plus length of w (2+1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        Fx = t -> [2dx(t)[6]        0               0   -1                  0           0   0
                    0           2*dx(t)[6]          0   0                   -1          0   0
                    -dx(t)[3]         0             0   2k*abs(x(t)[4])     0           0   0
                    0            -dx(t)[3]          0   0               2k*abs(x(t)[5]) 0   0
                    2x(t)[1]      2x(t)[2]          0   0                   0           0   0
                    x(t)[4]        x(t)[5]          0   x(t)[1]            x(t)[2]      0   0
                    x(t)[2]/(L^2)  -x(t)[1]/(L^2)   0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = t -> vcat([1   0   0          0   0   2x(t)[1]    0
                         0   1   0          0   0   2x(t)[2]    0
                         0   0   -x(t)[1]   m   0   0           0
                         0   0   -x(t)[2]   0   m   0           0], zeros(3,7))
        Fddx = t -> vcat([  0   0  0            0   0   2dx(t)[1]    0
                            0   0  0            0   0   2dx(t)[2]    0
                            0   0  -dx(t)[1]    0   0   0            0
                            0   0  -dx(t)[2]    0   0   0            0], zeros(3,7))
        # Fp = t -> [.0; .0; abs(x(t)[4])*x(t)[4]; abs(x(t)[5])*x(t)[5]; .0; .0; .0]
        gₓ  = t -> [0    0    0    0    0    0    2(x2(t)[7]-first(y(t)))/T]
        gdₓ = t -> [0    0    0    0    0    0    2(dx2(t)[7]-first(dy(t)))/T]

        # NOTE: Convention is used that derivatives wrt to θ stack along cols
        # while derivatives wrt to x stack along rows

        # Creating some of the needed disturbance from η. ASSUMING ONLY FREE DISTURBANCE PARAMETERS ARE a1
        A = [-η[1]  -η[2]; 1.0   0.0]
        C = [η[3]   η[4]]
        Aθ = [-1.0  0.0; 0.0    0.0;]
        Cθ = zeros(1,nw) # Only depends on C-parameters, and we only compute sensitivities wrt to a-parameters. We have 1 disturbance parameter, thus 1 row

        ###################### INITIALIZING ADJOINT SYSTEM ####################
        # Indices 1-4 are differential (d), while 5-7 are algebraic (a)
        dinds  = 1:4
        ainds  = 5:7
        λinds  = 1:7
        xwinds = 8:9
        winds  = 10
        λT  = zeros(10)
        dλT = zeros(10)
        temp = (-gₓ(T))/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        λT[dinds] = zeros(length(dinds))
        λT[ainds] = temp[ainds]
        dλT[dinds] = temp[dinds]
        temp = (-gdₓ(T) + (dλT[dinds]')*(Fx(T)[dinds,:] - Fddx(T)[dinds,:] - Fdx(T)[dinds,:]) + (λT[ainds]')*Fx(T)[ainds,:])/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        dλT[ainds] = temp[ainds]
        # Terminal conditions due to disturbance id
        λT[xwinds]  = zeros(length(xwinds))
        λT[winds]   = 2w(T)[1]*λT[3]
        dλT[xwinds] = (λT[xwinds]')*A + λT[winds]*C
        dλT[winds]  = 2w(T)[1]*dλT[3] + 2(C*(A*xw(T)+B̃*v(T)))[1]*λT[3]

        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            # Completely unreadabe but most efficient version (still not a huge improvement)
            res[1]  = dz[1] - 2*dx(t)[6]*z[1] + dx(t)[3]*z[3] - 2*x(t)[1]*z[5] - x(t)[4]*z[6] - (x(t)[2]*z[7])/(L^2)
            res[2]  = dz[2] - 2*dx(t)[6]*z[2] + dx(t)[3]*z[4] - 2*x(t)[2]*z[5] - x(t)[5]*z[6] + (x(t)[1]*z[7])/(L^2)
            res[3]  = -x(t)[1]*dz[3] - x(t)[2]*dz[4] - dx(t)[1]*z[3] - dx(t)[2]*z[4]
            res[4]  = m*dz[3] + z[1] - 2*k*abs(x(t)[4])*z[3] - x(t)[1]*z[6]
            res[5]  = m*dz[4] + z[2] - 2*k*abs(x(t)[5])*z[4] - x(t)[2]*z[6]
            res[6]  = 2*x(t)[1]*dz[1] + 2*x(t)[2]*dz[2] + 2*dx(t)[1]*z[1] + 2*dx(t)[2]*z[2]
            res[7]  = (2*(x2(t)[7] - first(y(t))))/T - z[7]

            # NOTE: η here then has to contain free_parameters and true values for the non-free ones
            res[8]  = dz[8] + z[9] - η[1]*z[8] + η[3]*z[10]
            res[9]  = dz[9] - η[2]*z[8] + η[4]*z[10]
            res[10] = 2w(t)[1]*z[3] - z[10]
            # β-equations
            res[11]  = dz[11] + (z[8:9]')*(Aθ[1:nw,:]*xw(t) - B̃θ[1:nw,:]*v(t)) # + z[10]*(Cθ*xw(t)) # This last part is equal to zero since we don't have C-parameters

            # # Super-readable but less efficient version
            # res[1:7] = (dz[1:7]')*Fdx(t) + (z[1:7]')*(Fddx(t) - Fx(t)) + gₓ(t)
            # res[8]   = dz[8] - (z[1:7]')*Fp(t)
            nothing
        end
        
        z0  = vcat(λT[:], zeros(np))
        # SETTING UP TERMINAL VALUES FOR β-VARIABLES
        # This part is actually overly general for this model, since here we only identify a single disturbance parameter, corresponding to one of the a-paramters
        second_temp = zeros(nw, nη)
        third_temp  = zeros(1, nη)
        # For variables corresponding to disturbance a-parameters
        for ind = 1:na
            second_temp[:, ind] = Aθ[(ind-1)nw + 1: ind*nw, :]*xw(T) + B̃θ[(ind-1)nw + 1: ind*nw, :]*v(T)
        end
        # For variables corresponding to disturbance c-parameters
        for ind = na+1:nη
            third_temp[:, ind]  = Cθ[ind:ind, :]*xw(T) # NOTE: SCALAR_OUTPUT is assumed
        end
        # my_temp = hcat([(λT[λinds]')*Fp(T)], - (λT[xwinds]')*second_temp - (λT[winds]')*third_temp)
        my_temp = - (λT[xwinds]')*second_temp - (λT[winds]')*third_temp
        dz0 = vcat(dλT[:], my_temp[:])
        # dz0 = vcat(dλT[:], (λT[λinds]')*Fp(T) - (λT[xwinds]')*second_temp - (λT[winds]')*third_temp)   # For some reason (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired

        if N_trans > 0
            @warn "The returned function get_Gp() doesn't fully support N_trans > 0, as sensitivity of internal variables not known at any other time than t=0. A non-rigorous approximation is used instead."
        end
        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            # There are nx+ndist variables before the β-variables, of which there are np
            Gp = adj_sol.u[end-N_trans][nx+ndist+1:nx+ndist+np] + (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)[:]     # This is the same as in original adjoint system just because disturbance model has zero initial conditions, independent of parameters
        end

        function get_Gp_debug(adj_sol::DAESolution)
            integral = adj_sol.u[end][nx+ndist+1:nx+ndist+np]
            term = (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)
            Gp = integral .+ term
            return Gp, integral, term
        end

        function get_term_debug(adj_sol::DAESolution, xps::Matrix{Float64}, times::AbstractVector{Float64})
            term = zeros(length(adj_sol.u))
            for ind=eachindex(adj_sol.u)
                term[ind] = ((adj_sol.u[end+1-ind][1:nx]')*Fdx(times[ind]))*xps[:,ind]
            end
            return term
        end

        dvars = vcat(fill(true, 4), fill(false, 3), fill(true, 2), [false], fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0 here, λ0: $λ0, dλ0: $dλ0"

        debugs = (get_Gp_debug, get_term_debug)

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp, debugs
    end
end

function my_pendulum_adjoint_distsensa2(u::Function, w::Function, xw::Function, v::Function, θ::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Matrix{Float64}, dx::Function, dx2::Function, B̃::Matrix{Float64}, B̃θ::Matrix{Float64}, η::Vector{Float64}, na::Int, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function. 
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        nη = 1
        nx = size(xp0,1)
        nw = length(xw(0.0))
        @assert (np == 1) "my_pendulum_adjoint_konly_with_distsensa is hard-coded to only handle parameter a1. Make sure to pass correct xp0"
        ndist = 3   # length of x_w plus length of w (2+1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        Fx = t -> [2dx(t)[6]        0               0   -1                  0           0   0
                    0           2*dx(t)[6]          0   0                   -1          0   0
                    -dx(t)[3]         0             0   2k*abs(x(t)[4])     0           0   0
                    0            -dx(t)[3]          0   0               2k*abs(x(t)[5]) 0   0
                    2x(t)[1]      2x(t)[2]          0   0                   0           0   0
                    x(t)[4]        x(t)[5]          0   x(t)[1]            x(t)[2]      0   0
                    x(t)[2]/(L^2)  -x(t)[1]/(L^2)   0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = t -> vcat([1   0   0          0   0   2x(t)[1]    0
                         0   1   0          0   0   2x(t)[2]    0
                         0   0   -x(t)[1]   m   0   0           0
                         0   0   -x(t)[2]   0   m   0           0], zeros(3,7))
        Fddx = t -> vcat([  0   0  0            0   0   2dx(t)[1]    0
                            0   0  0            0   0   2dx(t)[2]    0
                            0   0  -dx(t)[1]    0   0   0            0
                            0   0  -dx(t)[2]    0   0   0            0], zeros(3,7))
        # Fp = t -> [.0; .0; abs(x(t)[4])*x(t)[4]; abs(x(t)[5])*x(t)[5]; .0; .0; .0]
        gₓ  = t -> [0    0    0    0    0    0    2(x2(t)[7]-first(y(t)))/T]
        gdₓ = t -> [0    0    0    0    0    0    2(dx2(t)[7]-first(dy(t)))/T]

        # NOTE: Convention is used that derivatives wrt to θ stack along cols
        # while derivatives wrt to x stack along rows

        # Creating some of the needed disturbance from η. ASSUMING ONLY DISTURBANCE PARAMETERS ARE a1 AND a2!!!!!!
        A = [-η[1]  -η[2]; 1.0   0.0]
        C = [η[3]   η[4]]
        Aθ = [0.0  -1.0; 0.0    0.0;]# 0.0    -1.0; 0.0   0.0]
        Cθ = zeros(1,nw) # Only depends on C-parameters, and we only compute sensitivities wrt to a-parameters. We have 1 disturbance parameter, thus 1 row

        ###################### INITIALIZING ADJOINT SYSTEM ####################
        # Indices 1-4 are differential (d), while 5-7 are algebraic (a)
        dinds  = 1:4
        ainds  = 5:7
        λinds  = 1:7
        xwinds = 8:9
        winds  = 10
        λT  = zeros(10)
        dλT = zeros(10)
        temp = (-gₓ(T))/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        λT[dinds] = zeros(length(dinds))
        λT[ainds] = temp[ainds]
        dλT[dinds] = temp[dinds]
        temp = (-gdₓ(T) + (dλT[dinds]')*(Fx(T)[dinds,:] - Fddx(T)[dinds,:] - Fdx(T)[dinds,:]) + (λT[ainds]')*Fx(T)[ainds,:])/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        dλT[ainds] = temp[ainds]
        # Terminal conditions due to disturbance id
        λT[xwinds]  = zeros(length(xwinds))
        λT[winds]   = 2w(T)[1]*λT[3]
        dλT[xwinds] = (λT[xwinds]')*A + λT[winds]*C
        dλT[winds]  = 2w(T)[1]*dλT[3] + 2(C*(A*xw(T)+B̃*v(T)))[1]*λT[3]

        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            # Completely unreadabe but most efficient version (still not a huge improvement)
            res[1]  = dz[1] - 2*dx(t)[6]*z[1] + dx(t)[3]*z[3] - 2*x(t)[1]*z[5] - x(t)[4]*z[6] - (x(t)[2]*z[7])/(L^2)
            res[2]  = dz[2] - 2*dx(t)[6]*z[2] + dx(t)[3]*z[4] - 2*x(t)[2]*z[5] - x(t)[5]*z[6] + (x(t)[1]*z[7])/(L^2)
            res[3]  = -x(t)[1]*dz[3] - x(t)[2]*dz[4] - dx(t)[1]*z[3] - dx(t)[2]*z[4]
            res[4]  = m*dz[3] + z[1] - 2*k*abs(x(t)[4])*z[3] - x(t)[1]*z[6]
            res[5]  = m*dz[4] + z[2] - 2*k*abs(x(t)[5])*z[4] - x(t)[2]*z[6]
            res[6]  = 2*x(t)[1]*dz[1] + 2*x(t)[2]*dz[2] + 2*dx(t)[1]*z[1] + 2*dx(t)[2]*z[2]
            res[7]  = (2*(x2(t)[7] - first(y(t))))/T - z[7]

            # NOTE: η here then has to contain free_parameters and true values for the non-free ones
            res[8]  = dz[8] + z[9] - η[1]*z[8] + η[3]*z[10]
            res[9]  = dz[9] - η[2]*z[8] + η[4]*z[10]
            res[10] = 2w(t)[1]*z[3] - z[10]
            # β-equations
            res[11]  = dz[11] + (z[8:9]')*(Aθ[1:nw,:]*xw(t) - B̃θ[1:nw,:]*v(t)) # + z[10]*(Cθ*xw(t)) # This last part is equal to zero since we don't have C-parameters

            # # Super-readable but less efficient version
            # res[1:7] = (dz[1:7]')*Fdx(t) + (z[1:7]')*(Fddx(t) - Fx(t)) + gₓ(t)
            # res[8]   = dz[8] - (z[1:7]')*Fp(t)
            nothing
        end
        
        z0  = vcat(λT[:], zeros(np))
        # # Old attempt, hard to generalize
        # second_temp = Matrix{Float64}(undef, nw, nη)
        # third_temp  = Matrix{Float64}(undef, 1, nη)
        # for ind = 1:nη   # 1 disturbance parameter
        #     second_temp[:, ind] = Aθ[(ind-1)nw + 1: ind*nw, :]*xw(T) + B̃θ[(ind-1)nw + 1: ind*nw, :]*v(T)
        #     third_temp[:, ind]  = Cθ[ind:ind, :]*xw(T) # NOTE: SCALAR_OUTPUT is assumed
        #     # second_temp[:, (ind-1)nw + 1: ind*nw] = Aθ[(ind-1)nw + 1: ind*nw, :]*xw(T) + B̃θ[(ind-1)nw + 1: ind*nw, :]*v(T)
        #     # third_temp[:, (ind-1)nw + 1: ind]  = Cθ[ind:ind, :]*xw(T) # NOTE: SCALAR_OUTPUT is assumed
        # end
        # New attempt, hopefully more generalizable
        second_temp = zeros(nw, nη)
        third_temp  = zeros(1, nη)
        for ind = 1:na
            second_temp[:, ind] = Aθ[(ind-1)nw + 1: ind*nw, :]*xw(T) + B̃θ[(ind-1)nw + 1: ind*nw, :]*v(T)
        end
        for ind = na+1:nη
            third_temp[:, ind]  = Cθ[ind:ind, :]*xw(T) # NOTE: SCALAR_OUTPUT is assumed
        end
        # my_temp = hcat([(λT[λinds]')*Fp(T)], - (λT[xwinds]')*second_temp - (λT[winds]')*third_temp)
        my_temp = - (λT[xwinds]')*second_temp - (λT[winds]')*third_temp
        dz0 = vcat(dλT[:], my_temp[:])
        # dz0 = vcat(dλT[:], (λT[λinds]')*Fp(T) - (λT[xwinds]')*second_temp - (λT[winds]')*third_temp)   # For some reason (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired

        if N_trans > 0
            @warn "The returned function get_Gp() doesn't fully support N_trans > 0, as sensitivity of internal variables not known at any other time than t=0. A non-rigorous approximation is used instead."
        end
        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            Gp = adj_sol.u[end-N_trans][nx+ndist+1:nx+ndist+np] + (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)[:]     # This is the same as in original adjoint system just because disturbance model has zero initial conditions, independent of parameters
        end

        dvars = vcat(fill(true, 4), fill(false, 3), fill(true, 2), [false], fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0 here, λ0: $λ0, dλ0: $dλ0"

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp
    end
end

function my_pendulum_adjoint_distsensc(u::Function, w::Function, xw::Function, v::Function, θ::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Matrix{Float64}, dx::Function, dx2::Function, B̃::Matrix{Float64}, B̃θ::Matrix{Float64}, η::Vector{Float64}, na::Int, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function. 
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        nη = 1
        nx = size(xp0,1)
        nw = length(xw(0.0))
        @assert (np == 1) "my_pendulum_adjoint_konly_with_distsensa is hard-coded to only handle parameter c1. Make sure to pass correct xp0"
        ndist = 3   # length of x_w plus length of w (2+1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        Fx = t -> [2dx(t)[6]        0               0   -1                  0           0   0
                    0           2*dx(t)[6]          0   0                   -1          0   0
                    -dx(t)[3]         0             0   2k*abs(x(t)[4])     0           0   0
                    0            -dx(t)[3]          0   0               2k*abs(x(t)[5]) 0   0
                    2x(t)[1]      2x(t)[2]          0   0                   0           0   0
                    x(t)[4]        x(t)[5]          0   x(t)[1]            x(t)[2]      0   0
                    x(t)[2]/(L^2)  -x(t)[1]/(L^2)   0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = t -> vcat([1   0   0          0   0   2x(t)[1]    0
                         0   1   0          0   0   2x(t)[2]    0
                         0   0   -x(t)[1]   m   0   0           0
                         0   0   -x(t)[2]   0   m   0           0], zeros(3,7))
        Fddx = t -> vcat([  0   0  0            0   0   2dx(t)[1]    0
                            0   0  0            0   0   2dx(t)[2]    0
                            0   0  -dx(t)[1]    0   0   0            0
                            0   0  -dx(t)[2]    0   0   0            0], zeros(3,7))
        # Fp = t -> [.0; .0; abs(x(t)[4])*x(t)[4]; abs(x(t)[5])*x(t)[5]; .0; .0; .0]
        gₓ  = t -> [0    0    0    0    0    0    2(x2(t)[7]-first(y(t)))/T]
        gdₓ = t -> [0    0    0    0    0    0    2(dx2(t)[7]-first(dy(t)))/T]

        # NOTE: Convention is used that derivatives wrt to θ stack along cols
        # while derivatives wrt to x stack along rows

        # Creating some of the needed disturbance from η. ASSUMING ONLY DISTURBANCE PARAMETERS ARE a1 AND a2!!!!!!
        A = [-η[1]  -η[2]; 1.0   0.0]
        C = [η[3]   η[4]]
        Aθ = zeros(2, 2)
        Cθ = [0.0 1.0] # Only depends on C-parameters, and we only compute sensitivities wrt to a-parameters. We have 1 disturbance parameter, thus 1 row

        ###################### INITIALIZING ADJOINT SYSTEM ####################
        # Indices 1-4 are differential (d), while 5-7 are algebraic (a)
        dinds  = 1:4
        ainds  = 5:7
        λinds  = 1:7
        xwinds = 8:9
        winds  = 10
        λT  = zeros(10)
        dλT = zeros(10)
        temp = (-gₓ(T))/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        λT[dinds] = zeros(length(dinds))
        λT[ainds] = temp[ainds]
        dλT[dinds] = temp[dinds]
        temp = (-gdₓ(T) + (dλT[dinds]')*(Fx(T)[dinds,:] - Fddx(T)[dinds,:] - Fdx(T)[dinds,:]) + (λT[ainds]')*Fx(T)[ainds,:])/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        dλT[ainds] = temp[ainds]
        # Terminal conditions due to disturbance id
        λT[xwinds]  = zeros(length(xwinds))
        λT[winds]   = 2w(T)[1]*λT[3]
        dλT[xwinds] = (λT[xwinds]')*A + λT[winds]*C
        dλT[winds]  = 2w(T)[1]*dλT[3] + 2(C*(A*xw(T)+B̃*v(T)))[1]*λT[3]

        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            # Completely unreadabe but most efficient version (still not a huge improvement)
            res[1]  = dz[1] - 2*dx(t)[6]*z[1] + dx(t)[3]*z[3] - 2*x(t)[1]*z[5] - x(t)[4]*z[6] - (x(t)[2]*z[7])/(L^2)
            res[2]  = dz[2] - 2*dx(t)[6]*z[2] + dx(t)[3]*z[4] - 2*x(t)[2]*z[5] - x(t)[5]*z[6] + (x(t)[1]*z[7])/(L^2)
            res[3]  = -x(t)[1]*dz[3] - x(t)[2]*dz[4] - dx(t)[1]*z[3] - dx(t)[2]*z[4]
            res[4]  = m*dz[3] + z[1] - 2*k*abs(x(t)[4])*z[3] - x(t)[1]*z[6]
            res[5]  = m*dz[4] + z[2] - 2*k*abs(x(t)[5])*z[4] - x(t)[2]*z[6]
            res[6]  = 2*x(t)[1]*dz[1] + 2*x(t)[2]*dz[2] + 2*dx(t)[1]*z[1] + 2*dx(t)[2]*z[2]
            res[7]  = (2*(x2(t)[7] - first(y(t))))/T - z[7]

            # NOTE: η here then has to contain free_parameters and true values for the non-free ones
            res[8]  = dz[8] + z[9] - η[1]*z[8] + η[3]*z[10]
            res[9]  = dz[9] - η[2]*z[8] + η[4]*z[10]
            res[10] = 2w(t)[1]*z[3] - z[10]
            # β-equations
            res[11]  = dz[11] + z[10]*(Cθ⋅xw(t)) # + (z[8:9]')*(Aθ[1:nw,:]*xw(t) - B̃θ[1:nw,:]*v(t))  # This last part is equal to zero since we don't have A-parameters

            # # Super-readable but less efficient version
            # res[1:7] = (dz[1:7]')*Fdx(t) + (z[1:7]')*(Fddx(t) - Fx(t)) + gₓ(t)
            # res[8]   = dz[8] - (z[1:7]')*Fp(t)
            nothing
        end
        
        z0  = vcat(λT[:], zeros(np))
        # # Old attempt, hard to generalize
        # second_temp = Matrix{Float64}(undef, nw, nη)
        # third_temp  = Matrix{Float64}(undef, 1, nη)
        # for ind = 1:nη   # 1 disturbance parameter
        #     second_temp[:, ind] = Aθ[(ind-1)nw + 1: ind*nw, :]*xw(T) + B̃θ[(ind-1)nw + 1: ind*nw, :]*v(T)
        #     third_temp[:, ind]  = Cθ[ind:ind, :]*xw(T) # NOTE: SCALAR_OUTPUT is assumed
        #     # second_temp[:, (ind-1)nw + 1: ind*nw] = Aθ[(ind-1)nw + 1: ind*nw, :]*xw(T) + B̃θ[(ind-1)nw + 1: ind*nw, :]*v(T)
        #     # third_temp[:, (ind-1)nw + 1: ind]  = Cθ[ind:ind, :]*xw(T) # NOTE: SCALAR_OUTPUT is assumed
        # end
        # New attempt, hopefully more generalizable
        second_temp = zeros(nw, nη)
        third_temp  = zeros(1, nη)
        for ind = 1:na
            second_temp[:, ind] = Aθ[(ind-1)nw + 1: ind*nw, :]*xw(T) + B̃θ[(ind-1)nw + 1: ind*nw, :]*v(T)
        end
        for ind = na+1:nη
            third_temp[:, ind]  = Cθ[ind:ind, :]*xw(T) # NOTE: SCALAR_OUTPUT is assumed
        end
        # my_temp = hcat([(λT[λinds]')*Fp(T)], - (λT[xwinds]')*second_temp - (λT[winds]')*third_temp)
        my_temp = - (λT[xwinds]')*second_temp - (λT[winds]')*third_temp
        dz0 = vcat(dλT[:], my_temp[:])
        # dz0 = vcat(dλT[:], (λT[λinds]')*Fp(T) - (λT[xwinds]')*second_temp - (λT[winds]')*third_temp)   # For some reason (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired

        if N_trans > 0
            @warn "The returned function get_Gp() doesn't fully support N_trans > 0, as sensitivity of internal variables not known at any other time than t=0. A non-rigorous approximation is used instead."
        end
        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            Gp = adj_sol.u[end-N_trans][nx+ndist+1:nx+ndist+np] + (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)[:]     # This is the same as in original adjoint system just because disturbance model has zero initial conditions, independent of parameters
        end

        dvars = vcat(fill(true, 4), fill(false, 3), fill(true, 2), [false], fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0 here, λ0: $λ0, dλ0: $dλ0"

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp
    end
end

# NOTE Assumes free dynamical parameters are only k
function crazystab_pendulum_adjoint_konly(u::Function, w::Function, θ::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Vector{Float64}, dx::Function, dx2::Function, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function.
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        @assert (np == 1) "crazystab_pendulum_adjoint_konly is hard-coded to only handle one parameter k, make sure to pass correct xp0"
        nx = size(xp0,1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        # NOTE: Convention is used that derivatives wrt to θ stack along cols
        # while derivatives wrt to x stack along rows

        Fp = t -> [abs(x(t)[4])*x(t)[4]; abs(x(t)[5])*x(t)[5]; .0; .0; .0; .0; .0]  # NOTE: Actually not quite right, since it only should have 6 elements and not 7, but doesn't matter in this particular case

        ###################### INITIALIZING ADJOINT SYSTEM ####################
        # Indices 1-4 are differential (d), while 5-7 are algebraic (a)
        zT  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2*(x2(T)[7]-y(T))/T] # z = [λ1, λ2, λ3, λ4, λ5, d1, λ7]
        dzT = [0.0, 0.0, 0.0, x(T)[2]*zT[7]/(L^2), -x(T)[1]*zT[7]/(L^2), -x(T)[2]*zT[7]/(m*L^2), 2(dx2(T)[7]-dy(T))/T]

        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            # NEW
            res[1]  = dz[4] + x(t)[3]*z[1] - 2*x(t)[1]*z[3] - x(t)[2]*z[7]/(L^2)
            res[2]  = dz[5] + x(t)[3]*z[2] - 2*x(t)[2]*z[3] + x(t)[1]*z[7]/(L^2)
            res[3]  = z[1]*x(t)[1] + z[2]*x(t)[2]
            res[4]  = z[4] + m*z[6] - 2*k*abs(x(t)[4])*z[1] - 2*k*abs(x(t)[5])*z[2]
            res[5]  = m*dz[2] + z[5]
            res[6]  = z[1]*x(t)[4] + z[2]*x(t)[5] + x(t)[2]*dz[2] + x(t)[1]*z[6]
            res[7]  = -z[7] + 2(x(t)[7]-y(t))/T
            res[8]  = dz[8] - z[1]*abs(x(t)[4])*x(t)[4] - z[2]*abs(x(t)[5])*x(t)[5]
            # # OLD
            # # Completely unreadabe but most efficient version (still not a huge improvement)
            # res[1]  = dz[1] - 2*dx(t)[6]*z[1] + dx(t)[3]*z[3] - 2*x(t)[1]*z[5] - x(t)[4]*z[6] - (x(t)[2]*z[7])/(L^2)
            # res[2]  = dz[2] - 2*dx(t)[6]*z[2] + dx(t)[3]*z[4] - 2*x(t)[2]*z[5] - x(t)[5]*z[6] + (x(t)[1]*z[7])/(L^2)
            # res[3]  = -x(t)[1]*dz[3] - x(t)[2]*dz[4] - dx(t)[1]*z[3] - dx(t)[2]*z[4]
            # res[4]  = m*dz[3] + z[1] - 2*k*abs(x(t)[4])*z[3] - x(t)[1]*z[6]
            # res[5]  = m*dz[4] + z[2] - 2*k*abs(x(t)[5])*z[4] - x(t)[2]*z[6]
            # res[6]  = 2*x(t)[1]*dz[1] + 2*x(t)[2]*dz[2] + 2*dx(t)[1]*z[1] + 2*dx(t)[2]*z[2]
            # res[7]  = (2*(x2(t)[7] - y(t)))/T - z[7]
            # # res[1]  = dz[1] + 2x(t)[1]*dz[6]    - 2dx(t)[6]*z[1] + z[4] + 2dx(t)[1]*z[6]
            # # res[2]  = dz[2] + 2x(t)[2]*dz[6]    - 2dx(t)[6]*z[2] + z[5] + 2dx(t)[2]*z[6]
            # # res[3]  = -x(t)[1]*dz[3] + m*dz[4]  + dx(t)[3]*z[1] - dx(t)[1]*z[3] - 2k*abs(x(t)[4])*z[4]
            # # res[4]  = -x(t)[2]*dz[3] + m*dz[5]  + dx(t)[3]*z[2] - dx(t)[2]*z[3] - 2k*abs(x(t)[5])*z[5]
            # # res[5]  =                           - 2x(t)[1]*z[1] - 2x(t)[2]*z[2]
            # # res[6]  =                           - x(t)[4]*z[1] - x(t)[5]*z[2] - x(t)[1]*z[4] - x(t)[2]*z[5]
            # # res[7]  =                           - x(t)[2]*z[1]/(L^2) + x(t)[1]*z[2]/(L^2) - z[7]                + (2/T)*(x2(t)[7]-y(t))
            # res[8]  = dz[8] - z[3]*abs(x(t)[4])*x(t)[4] - z[4]*abs(x(t)[5])*x(t)[5]

            # # # Super-readable but less efficient version
            # # res[1:7] = (dz[1:7]')*Fdx(t) + (z[1:7]')*(Fddx(t) - Fx(t)) + gₓ(t)
            # # res[8]   = dz[8] - (z[1:7]')*Fp(t)
            nothing
        end

        z0  = vcat(zT[:], zeros(np))
        dz0 = vcat(dzT[:], (zT')*Fp(T))   # For some reason (zT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired
        # TODO: Can we delete these rows below here perhaps?
        # z0  = vcat(λ0, 0.0)
        # dz0 = vcat(dλ0, λ0[1])   # For some reason (zT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired

        if N_trans > 0
            @warn "The returned function get_Gp() doesn't fully support N_trans > 0, as sensitivity of internal variables not known at any other time than t=0. A non-rigorous approximation is used instead."
        end
        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            # NOTE: Changes signs to match what I had in my manual calculations, seems correct now
            # Gp = adj_sol.u[end][end-np+1:end] + (((adj_sol.u[end][1:end-np]')*Fdx(0.0))*xp0)[:]
            # Gp = adj_sol.u[end-N_trans][nx+1:nx+np] + (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)[:]
            Gp = adj_sol.u[end-N_trans][nx+1:nx+np] .+ (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)
            # return 0.0
        end

        dvars = vcat([false, true, false, true, true, false, false], fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], T)
        # @info "r0 is: $r0"  # NOTE: Passed t=T since this problem is meant to be solved backwards in time

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp
    end
end

# NOTE Assumes free dynamical parameters are only k
# This one I'm more confident has actually index-1 adjoint system
function crazystab2_pendulum_adjoint_konly(u::Function, w::Function, θ::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Vector{Float64}, dx::Function, dx2::Function, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function.
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        @assert (np == 1) "crazystab_pendulum_adjoint_konly is hard-coded to only handle one parameter k, make sure to pass correct xp0"
        nx = size(xp0,1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        # NOTE: Convention is used that derivatives wrt to θ stack along cols
        # while derivatives wrt to x stack along rows

        Fp = t -> [abs(x(t)[4])*x(t)[4]; abs(x(t)[5])*x(t)[5]; .0; .0; .0; .0; .0]  # NOTE: Actually not quite right, since it only should have 6 elements and not 7, but doesn't matter in this particular case

        ###################### INITIALIZING ADJOINT SYSTEM ####################
        # Indices 1-4 are differential (d), while 5-7 are algebraic (a)
        zT  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2(x2(T)[7]-y(T))/T] # z = [λ1, λ2, λ3, λ4, λ5, d2, λ7]
        dzT = [0.0, 0.0, 0.0, x(T)[2]*zT[7]/(L^2), -x(T)[1]*zT[7]/(L^2), 0.0, 2(dx2(T)[7]-dy(T))/T]

        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            # NEW
            res[1]  = dz[4] + x(t)[3]*z[1] - 2*x(t)[1]*z[3] - x(t)[2]*z[7]/(L^2)
            res[2]  = dz[5] + x(t)[3]*z[2] - 2*x(t)[2]*z[3] + x(t)[1]*z[7]/(L^2)
            res[3]  = z[1]*x(t)[1] + z[2]*x(t)[2] + dz[6]
            res[4]  = z[4] + m*dz[1] - 2*k*abs(x(t)[4])*z[1] - 2*k*abs(x(t)[5])*z[2]
            res[5]  = m*dz[2] + z[5]
            res[6]  = z[1]*x(t)[4] + z[2]*x(t)[5] + x(t)[2]*dz[2] + x(t)[1]*dz[1]
            res[7]  = -z[7] + 2(x(t)[7]-y(t))/T
            res[8]  = dz[8] - z[1]*abs(x(t)[4])*x(t)[4] - z[2]*abs(x(t)[5])*x(t)[5]

            # # # Super-readable but less efficient version
            # # res[1:7] = (dz[1:7]')*Fdx(t) + (z[1:7]')*(Fddx(t) - Fx(t)) + gₓ(t)
            # # res[8]   = dz[8] - (z[1:7]')*Fp(t)
            nothing
        end

        z0  = vcat(zT[:], zeros(np))
        dz0 = vcat(dzT[:], (zT')*Fp(T))   # For some reason (zT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired
        

        if N_trans > 0
            @warn "The returned function get_Gp() doesn't fully support N_trans > 0, as sensitivity of internal variables not known at any other time than t=0. A non-rigorous approximation is used instead."
        end
        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            # NOTE: Changes signs to match what I had in my manual calculations, seems correct now
            # Gp = adj_sol.u[end][end-np+1:end] + (((adj_sol.u[end][1:end-np]')*Fdx(0.0))*xp0)[:]
            # Gp = adj_sol.u[end-N_trans][nx+1:nx+np] + (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)[:]
            Gp = adj_sol.u[end-N_trans][nx+1:nx+np] .+ (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)
            # return 0.0
        end

        dvars = vcat([false, true, false, true, true, false, false], fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], T)
        # @info "r0 is: $r0" # NOTE: Passed t=T since this problem is meant to be solved backwards in time

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp
    end
end

# NOTE Assumes free dynamical parameters are only k
# In contrast to all the other crazystabs, this one is actually correct as well as index 1
function crazystab3_pendulum_adjoint_konly(u::Function, w::Function, θ::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Vector{Float64}, dx::Function, dx2::Function, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function.
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        @assert (np == 1) "crazystab_pendulum_adjoint_konly is hard-coded to only handle one parameter k, make sure to pass correct xp0"
        nx = size(xp0,1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        # NOTE: Convention is used that derivatives wrt to θ stack along cols
        # while derivatives wrt to x stack along rows

        Fp = t -> [abs(x(t)[4])*x(t)[4]; abs(x(t)[5])*x(t)[5]; .0; .0; .0; .0; .0; .0; .0; .0; .0]  # NOTE: Actually not quite right, since it only should have 6 elements and not 7, but doesn't matter in this particular case

        ###################### INITIALIZING ADJOINT SYSTEM ####################
        # Indices 1-4 are differential (d), while 5-7 are algebraic (a) NOTE: I don't think this is correct any longer, old comment from other function

        #       1  2  3  4  5  6  7   8    9  10  11
        # z = [λ1 λ2 λ3 λ4 λ5 λ7 dλ1 dλ2 dλ4 dλ5 ddλ1]
        λ7  = 2(x2(T)[7]-y(T))/T
        dλ7 = 2(dx2(T)[7]-dy(T))/T
        #       λ1   λ2   λ3  λ4   λ5   λ7  dλ1  dλ2           dλ4              dλ5                 ddλ1
        zT  = [0.0, 0.0, 0.0, 0.0, 0.0, λ7, 0.0, 0.0, x(T)[2]*λ7/(L^2), -x(T)[1]*λ7/(L^2), -x(T)[2]*λ7/(m*L^2)]
        #       λ1   λ2   λ3           λ4                λ5         λ7          dλ1                   dλ2        dλ4 dλ5  ddλ1
        dzT = [0.0, 0.0, NaN, x(T)[2]*λ7/(L^2), -x(T)[1]*λ7/(L^2), dλ7, -x(T)[2]*λ7/(m*L^2), x(T)[1]*λ7/(m*L^2), NaN, NaN, NaN]
        a = -3zT[11]*dx(T)[1] - 3dzT[8]*dx(T)[2]
        b = 2*k*abs(x(T)[4])*dzT[7]
        c = 2*k*abs(x(T)[5])*dzT[8]
        d = ( dx(T)[2]*λ7 + x(T)[2]*dλ7)/(L^2)
        e = (-dx(T)[1]*λ7 - x(T)[1]*dλ7)/(L^2)
        dzT[3]  = -(a*m - b*x(T)[1] - c*x(T)[2] + d*x(T)[1] + e*x(T)[2])/(m*L^2)
        dzT[9]  = (b*x(T)[1]^2 + d*x(T)[2]^2 - a*m*x(T)[1] + (c-e)*x(T)[1]*x(T)[2])/(L^2)
        dzT[10]  = (b*x(T)[1]^2 + d*x(T)[2]^2 - a*m*x(T)[1] + (c-e)*x(T)[1]*x(T)[2])/(L^2)
        dzT[11] = ( (b-d)*x(T)[2]^2 + a*m*x(T)[1] + (e-c)*x(T)[1]*x(T)[2] )/(m*L^2)


        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            res[1]  = z[9] + z[1]*x(t)[3] - 2*z[3]*x(t)[1] - (z[6]*x(t)[2])/L^2     # good
            res[2]  = z[10] + z[2]*x(t)[3] - 2*z[3]*x(t)[2] + (z[6]*x(t)[1])/L^2    # good
            res[3]  = z[1]*x(t)[1] + z[2]*x(t)[2]                                   # good
            res[4]  = z[4] + m*z[7] - 2*k*abs(x(t)[4])*z[1]                         # good
            res[5]  = z[5] + m*z[8] - 2*k*abs(x(t)[5])*z[2]                         # good
            res[6]  = - z[6] - 2*(y(t) - x(t)[7])/T                                 # good
            res[7]  = z[1]*dx(t)[1] + z[2]*dx(t)[2] + z[7]*x(t)[1] + z[8]*x(t)[2] # NOTE: Could replace by x4 and x5        # good
            res[8]  = z[11]*x(t)[1] + z[1]*dx(t)[4] + z[2]*dx(t)[5] + 2*z[7]*dx(t)[1] + 2*z[8]*dx(t)[2] + x(t)[2]*dz[8]     # good
            res[9]  = z[9] + m*z[11] - 2*k*abs(x(t)[4])*z[7] - 2*k*sign(x(t)[4])*z[1]*dx(t)[4]                              # good
            res[10] = z[10] + m*dz[8] - 2*k*abs(x(t)[5])*z[8] - 2*k*sign(x(t)[5])*z[2]*dx(t)[5]                             # good
            res[11] = z[8] - dz[2]                                                                                          # good
            res[12]  = dz[12] - z[1]*abs(x(t)[4])*x(t)[4] - z[2]*abs(x(t)[5])*x(t)[5]

            nothing
        end

        z0  = vcat(zT[:], zeros(np))
        dz0 = vcat(dzT[:], (zT')*Fp(T))   # For some reason (zT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired
        

        if N_trans > 0
            @warn "The returned function get_Gp() doesn't fully support N_trans > 0, as sensitivity of internal variables not known at any other time than t=0. A non-rigorous approximation is used instead."
        end
        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            # NOTE: Changes signs to match what I had in my manual calculations, seems correct now
            # Gp = adj_sol.u[end][end-np+1:end] + (((adj_sol.u[end][1:end-np]')*Fdx(0.0))*xp0)[:]
            # Gp = adj_sol.u[end-N_trans][nx+1:nx+np] + (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)[:]
            Gp = adj_sol.u[end-N_trans][nx+1:nx+np] .+ (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)
            # return 0.0
        end

        dvars = vcat([false, true], fill(false, 5), [true], fill(false, 3), fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], T)
        # @info "r0 is: $r0" # NOTE: Passed t=T since this problem is meant to be solved backwards in time

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp
    end
end

# NOTE Assumes free dynamical parameters are only k
# Same as crazystab3, but with reduced number of equations, simplified using the algebraic equations
function crazystab4_pendulum_adjoint_konly(u::Function, w::Function, θ::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Vector{Float64}, dx::Function, dx2::Function, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function.
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        @assert (np == 1) "crazystab_pendulum_adjoint_konly is hard-coded to only handle one parameter k, make sure to pass correct xp0"
        nx = size(xp0,1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        # NOTE: Convention is used that derivatives wrt to θ stack along cols
        # while derivatives wrt to x stack along rows

        # Fp = t -> [abs(x(t)[4])*x(t)[4]; abs(x(t)[5])*x(t)[5]; .0; .0; .0; .0; .0; .0; .0; .0; .0]  # NOTE: Actually not quite right, since it only should have 6 elements and not 7, but doesn't matter in this particular case
        Fp = t -> [abs(x(t)[4])*x(t)[4] - abs(x(t)[5])*x(t)[5]*x(t)[1]/x(t)[2]; .0; .0; .0; .0; .0; .0; .0; .0]  # NOTE: Actually not quite right, since it only should have 6 elements and not 7, but doesn't matter in this particular case

        ###################### INITIALIZING ADJOINT SYSTEM ####################
        # Indices 1-4 are differential (d), while 5-7 are algebraic (a) NOTE: I don't think this is correct any longer, old comment from other function

        #       1  2  3  4   5   6   7   8   9
        # z = [λ1  λ3 λ4 λ5 dλ1 dλ2 dλ4 dλ5 ddλ1]  3,4,5 -> 2,3,4; 7,8,9,10,11 -> 5,6,7,8,9
        λ7  = 2(x2(T)[7]-y(T))/T
        dλ7 = 2(dx2(T)[7]-dy(T))/T
        #       λ1   λ2   λ3  λ4   λ5   λ7  dλ1  dλ2           dλ4              dλ5                 ddλ1
        zTo  = [0.0, 0.0, 0.0, 0.0, 0.0, λ7, 0.0, 0.0, x(T)[2]*λ7/(L^2), -x(T)[1]*λ7/(L^2), -x(T)[2]*λ7/(m*L^2)]
        #       λ1   λ2   λ3           λ4                λ5         λ7          dλ1                   dλ2        dλ4 dλ5  ddλ1
        dzTo = [0.0, 0.0, NaN, x(T)[2]*λ7/(L^2), -x(T)[1]*λ7/(L^2), dλ7, -x(T)[2]*λ7/(m*L^2), x(T)[1]*λ7/(m*L^2), NaN, NaN, NaN]
        a = -3zTo[11]*dx(T)[1] - 3dzTo[8]*dx(T)[2]
        b = 2*k*abs(x(T)[4])*dzTo[7]
        c = 2*k*abs(x(T)[5])*dzTo[8]
        d = ( dx(T)[2]*λ7 + x(T)[2]*dλ7)/(L^2)
        e = (-dx(T)[1]*λ7 - x(T)[1]*dλ7)/(L^2)
        dzTo[3]  = -(a*m - b*x(T)[1] - c*x(T)[2] + d*x(T)[1] + e*x(T)[2])/(m*L^2)
        dzTo[9]  = (b*x(T)[1]^2 + d*x(T)[2]^2 - a*m*x(T)[1] + (c-e)*x(T)[1]*x(T)[2])/(L^2)
        dzTo[10]  = (b*x(T)[1]^2 + d*x(T)[2]^2 - a*m*x(T)[1] + (c-e)*x(T)[1]*x(T)[2])/(L^2)
        dzTo[11] = ( (b-d)*x(T)[2]^2 + a*m*x(T)[1] + (e-c)*x(T)[1]*x(T)[2] )/(m*L^2)
        zT  = [zTo[1], zTo[3], zTo[4], zTo[5], zTo[7], zTo[8], zTo[9], zTo[10], zTo[11]]
        dzT = [dzTo[1], dzTo[3], dzTo[4], dzTo[5], dzTo[7], dzTo[8], dzTo[9], dzTo[10], dzTo[11]]


        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            res[1]  = z[7]  + z[1]*x(t)[3]                  - 2*z[2]*x(t)[1] - 2((x(t)[7]-y(t))*x(t)[2])/(T*L^2)
            res[2]  = z[8] - z[1]*x(t)[3]*x(t)[1]/x(t)[2]  - 2*z[2]*x(t)[2] + 2((x(t)[7]-y(t))*x(t)[1])/(T*L^2)
            res[3]  = z[3] + m*z[5] - 2*k*abs(x(t)[4])*z[1]                
            res[4]  = z[4] + m*z[6] + 2*k*abs(x(t)[5])*z[1]*x(t)[1]/x(t)[2]
            res[5]  = z[1]*dx(t)[1] - z[1]*dx(t)[2]*x(t)[1]/x(t)[2] + z[5]*x(t)[1] + z[6]*x(t)[2] # NOTE: Could replace by x4 and x5
            res[6]  = z[9]*x(t)[1] + z[1]*dx(t)[4] - z[1]*dx(t)[5]*x(t)[1]/x(t)[2] + 2*z[5]*dx(t)[1] + 2*z[6]*dx(t)[2] + x(t)[2]*dz[6]
            res[7]  = z[7]  + m*z[9] - 2*k*abs(x(t)[4])*z[5] - 2*k*sign(x(t)[4])*z[1]*dx(t)[4]                
            res[8] = z[8] + m*dz[6] - 2*k*abs(x(t)[5])*z[6] + 2*k*sign(x(t)[5])*z[1]*dx(t)[5]*x(t)[1]/x(t)[2]
            res[9] = z[6] + dx(t)[1]*z[1]/x(t)[2] - x(t)[1]*dx(t)[2]/(x(t)[2]^2)*z[1] + x(t)[1]*dz[1]/x(t)[2]  # Hopefully correct
            res[10]  = dz[10] - z[1]*abs(x(t)[4])*x(t)[4] + z[1]*abs(x(t)[5])*x(t)[5]*x(t)[1]/x(t)[2]
            nothing
        end

        z0  = vcat(zT[:], zeros(np))
        dz0 = vcat(dzT[:], (zT')*Fp(T))   # For some reason (zT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired
        

        if N_trans > 0
            @warn "The returned function get_Gp() doesn't fully support N_trans > 0, as sensitivity of internal variables not known at any other time than t=0. A non-rigorous approximation is used instead."
        end
        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            # NOTE: Changes signs to match what I had in my manual calculations, seems correct now
            # Gp = adj_sol.u[end][end-np+1:end] + (((adj_sol.u[end][1:end-np]')*Fdx(0.0))*xp0)[:]
            # Gp = adj_sol.u[end-N_trans][nx+1:nx+np] + (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)[:]
            Gp = adj_sol.u[end-N_trans][nx+1:nx+np] .+ (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)
            # return 0.0
        end

        # TODO: FIGURE OUT IF YOUR DVARS ARE RIGHT!!!!

        dvars = vcat([true], fill(false, 4), [true], fill(false, 3), fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], T)
        @info "r0 is: $r0" # NOTE: Passed t=T since this problem is meant to be solved backwards in time

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp
    end
end

function my_pendulum_adjoint_gonly(u::Function, w::Function, θ::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Vector{Float64}, dx::Function, dx2::Function, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function.
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        @assert (np == 1) "my_pendulum_adjoint_Lonly is hard-coded to only handle one parameter L, make sure to pass correct xp0"
        nx = size(xp0,1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        Fx = t -> [2dx(t)[6]        0               0   -1                  0           0   0
                    0           2*dx(t)[6]          0   0                   -1          0   0
                    -dx(t)[3]         0             0   2k*abs(x(t)[4])     0           0   0
                    0            -dx(t)[3]          0   0               2k*abs(x(t)[5]) 0   0
                    2x(t)[1]      2x(t)[2]          0   0                   0           0   0
                    x(t)[4]        x(t)[5]          0   x(t)[1]            x(t)[2]      0   0
                    x(t)[2]/(L^2)  -x(t)[1]/(L^2)   0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = t -> vcat([1   0   0          0   0   2x(t)[1]    0
                         0   1   0          0   0   2x(t)[2]    0
                         0   0   -x(t)[1]   m   0   0           0
                         0   0   -x(t)[2]   0   m   0           0], zeros(3,7))
        Fddx = t -> vcat([  0   0  0            0   0   2dx(t)[1]    0
                            0   0  0            0   0   2dx(t)[2]    0
                            0   0  -dx(t)[1]    0   0   0            0
                            0   0  -dx(t)[2]    0   0   0            0], zeros(3,7))
        Fp = t -> [ .0
                    .0
                    .0
                     m
                    .0
                    .0
                    .0] # Was last element not dependent on L? Didn't we divide by e.g. L^2?
        gₓ  = t -> [0    0    0    0    0    0    2(x2(t)[7]-y(t))/T]
        gdₓ = t -> [0    0    0    0    0    0    2(dx2(t)[7]-dy(t))/T]

        # NOTE: Convention is used that derivatives wrt to θ stack along cols
        # while derivatives wrt to x stack along rows

        ###################### INITIALIZING ADJOINT SYSTEM ####################
        # Indices 1-4 are differential (d), while 5-7 are algebraic (a)
        dinds = 1:4
        ainds = 5:7
        λT  = zeros(7)
        dλT = zeros(7)
        temp = (-gₓ(T))/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        λT[dinds] = zeros(length(dinds))
        λT[ainds] = temp[ainds]
        dλT[dinds] = temp[dinds]
        temp = (-gdₓ(T) + (dλT[dinds]')*(Fx(T)[dinds,:] - Fddx(T)[dinds,:] - Fdx(T)[dinds,:]) + (λT[ainds]')*Fx(T)[ainds,:])/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        dλT[ainds] = temp[ainds]

        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            # Completely unreadabe but most efficient version (still not a huge improvement)
            res[1]  = dz[1] - 2*dx(t)[6]*z[1] + dx(t)[3]*z[3] - 2*x(t)[1]*z[5] - x(t)[4]*z[6] - (x(t)[2]*z[7])/(L^2)
            res[2]  = dz[2] - 2*dx(t)[6]*z[2] + dx(t)[3]*z[4] - 2*x(t)[2]*z[5] - x(t)[5]*z[6] + (x(t)[1]*z[7])/(L^2)
            res[3]  = -x(t)[1]*dz[3] - x(t)[2]*dz[4] - dx(t)[1]*z[3] - dx(t)[2]*z[4]
            res[4]  = m*dz[3] + z[1] - 2*k*abs(x(t)[4])*z[3] - x(t)[1]*z[6]
            res[5]  = m*dz[4] + z[2] - 2*k*abs(x(t)[5])*z[4] - x(t)[2]*z[6]
            res[6]  = 2*x(t)[1]*dz[1] + 2*x(t)[2]*dz[2] + 2*dx(t)[1]*z[1] + 2*dx(t)[2]*z[2]
            res[7]  = (2*(x2(t)[7] - y(t)))/T - z[7]
            # res[1]  = dz[1] + 2x(t)[1]*dz[6]    - 2dx(t)[6]*z[1] + z[4] + 2dx(t)[1]*z[6]
            # res[2]  = dz[2] + 2x(t)[2]*dz[6]    - 2dx(t)[6]*z[2] + z[5] + 2dx(t)[2]*z[6]
            # res[3]  = -x(t)[1]*dz[3] + m*dz[4]  + dx(t)[3]*z[1] - dx(t)[1]*z[3] - 2k*abs(x(t)[4])*z[4]
            # res[4]  = -x(t)[2]*dz[3] + m*dz[5]  + dx(t)[3]*z[2] - dx(t)[2]*z[3] - 2k*abs(x(t)[5])*z[5]
            # res[5]  =                           - 2x(t)[1]*z[1] - 2x(t)[2]*z[2]
            # res[6]  =                           - x(t)[4]*z[1] - x(t)[5]*z[2] - x(t)[1]*z[4] - x(t)[2]*z[5]
            # res[7]  =                           - x(t)[2]*z[1]/(L^2) + x(t)[1]*z[2]/(L^2) - z[7]                + (2/T)*(x2(t)[7]-y(t))
            res[8]  = dz[8] - m*z[4]

            # # Super-readable but less efficient version
            # res[1:7]  = (dz[1:7]')*Fdx(T-t) + (z[1:7]')*(Fddx(T-t) - Fx(T-t)) + gₓ(T-t)
            # res[8] = dz[8] - (Fp(T-t)')*z[1:7]
            nothing
        end

        # z0 = λ0
        # dz0 = dλ0
        z0  = vcat(λT[:], zeros(np))
        # dz0 = vcat(dλ0, -((λT')*Fp(T))')
        dz0 = vcat(dλT[:], -(Fp(T)')*λT)
        # TODO: Can we delete these row below here perhaps?
        # z0  = vcat(λ0, 0.0)
        # dz0 = vcat(dλ0, λ0[1])   # For some reasing (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired

        if N_trans > 0
            @warn "The returned function get_Gp() doesn't fully support N_trans > 0, as sensitivity of internal variables not known at any other time than t=0. A non-rigorous approximation is used instead."
        end
        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            # NOTE: Changes signs to match what I had in my manual calculations, seems correct now
            # Gp = adj_sol.u[end][end-np+1:end] + (((adj_sol.u[end][1:end-np]')*Fdx(0.0))*xp0)[:]
            # Gp = adj_sol.u[end-N_trans][nx+1:nx+np] + (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)[:]
            Gp = adj_sol.u[end-N_trans][nx+1:nx+np] .+ (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)
            # return 0.0
        end

        dvars = vcat(fill(true, 4), fill(false, 3), fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0 here, λ0: $λ0, dλ0: $dλ0"

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp
    end
end

function my_pendulum_adjoint_deb(u::Function, w::Function, θ::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Vector{Float64}, dx::Function, dx2::Function, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function.
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        @assert (np == 1) "my_pendulum_adjoint_Lonly is hard-coded to only handle one parameter L, make sure to pass correct xp0"
        nx = size(xp0,1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        # @warn "my_pendulum_adjoint_deb. m: $m, L: $L, g: $g, k: $k"

        my_ind = 2
        i = my_ind

        Fx = t -> [2dx(t)[6]        0               0   -1                  0           0   0
                    0           2*dx(t)[6]          0   0                   -1          0   0
                    -dx(t)[3]         0             0   2k*abs(x(t)[4])     0           0   0
                    0            -dx(t)[3]          0   0               2k*abs(x(t)[5]) 0   0
                    2x(t)[1]      2x(t)[2]          0   0                   0           0   0
                    x(t)[4]        x(t)[5]          0   x(t)[1]            x(t)[2]      0   0
                    x(t)[2]/(L^2)  -x(t)[1]/(L^2)   0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = t -> vcat([1   0   0          0   0   2x(t)[1]    0
                         0   1   0          0   0   2x(t)[2]    0
                         0   0   -x(t)[1]   m   0   0           0
                         0   0   -x(t)[2]   0   m   0           0], zeros(3,7))
        Fddx = t -> vcat([  0   0  0            0   0   2dx(t)[1]    0
                            0   0  0            0   0   2dx(t)[2]    0
                            0   0  -dx(t)[1]    0   0   0            0
                            0   0  -dx(t)[2]    0   0   0            0], zeros(3,7))
        Fp = t ->       [ Int(i==1)
                          Int(i==2)
                          Int(i==3)
                          Int(i==4)
                          Int(i==5)
                          Int(i==6)
                          Int(i==7)]
        gₓ  = t -> [0    0    0    0    0    0    2(x2(t)[7]-y(t))/T]
        gdₓ = t -> [0    0    0    0    0    0    2(dx2(t)[7]-dy(t))/T]

        # NOTE: Convention is used that derivatives wrt to θ stack along cols
        # while derivatives wrt to x stack along rows

        ###################### INITIALIZING ADJOINT SYSTEM ####################
        # Indices 1-4 are differential (d), while 5-7 are algebraic (a)
        dinds = 1:4
        ainds = 5:7
        λT  = zeros(7)
        dλT = zeros(7)
        temp = (-gₓ(T))/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        λT[dinds] = zeros(length(dinds))
        λT[ainds] = temp[ainds]
        dλT[dinds] = temp[dinds]
        temp = (-gdₓ(T) + (dλT[dinds]')*(Fx(T)[dinds,:] - Fddx(T)[dinds,:] - Fdx(T)[dinds,:]) + (λT[ainds]')*Fx(T)[ainds,:])/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        dλT[ainds] = temp[ainds]

        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            # Completely unreadabe but most efficient version (still not a huge improvement)
            res[1]  = dz[1] - 2*dx(t)[6]*z[1] + dx(t)[3]*z[3] - 2*x(t)[1]*z[5] - x(t)[4]*z[6] - (x(t)[2]*z[7])/(L^2)
            res[2]  = dz[2] - 2*dx(t)[6]*z[2] + dx(t)[3]*z[4] - 2*x(t)[2]*z[5] - x(t)[5]*z[6] + (x(t)[1]*z[7])/(L^2)
            res[3]  = -x(t)[1]*dz[3] - x(t)[2]*dz[4] - dx(t)[1]*z[3] - dx(t)[2]*z[4]
            res[4]  = m*dz[3] + z[1] - 2*k*abs(x(t)[4])*z[3] - x(t)[1]*z[6]
            res[5]  = m*dz[4] + z[2] - 2*k*abs(x(t)[5])*z[4] - x(t)[2]*z[6]
            res[6]  = 2*x(t)[1]*dz[1] + 2*x(t)[2]*dz[2] + 2*dx(t)[1]*z[1] + 2*dx(t)[2]*z[2]
            res[7]  = (2*(x2(t)[7] - y(t)))/T - z[7]
            res[8]  = dz[8] - z[i]

            # β(t) = ∫_t^T - λ'Fθ dt = ∫_T^t λ'Fθ => dβ = λ'Fθ
            # => dβ - λ'Fθ = 0.0

            # Super-readable but less efficient version
            # res[1:7]  = (dz[1:7]')*Fdx(t) + (z[1:7]')*(Fddx(t) - Fx(t)) + gₓ(t)
            # res[8] = dz[8] - (Fp(t)')*z[1:7]
            nothing
        end


        # z0 = λ0
        # dz0 = dλ0
        z0  = vcat(λT[:], zeros(np))
        # dz0 = vcat(dλ0, -((λT')*Fp(T))')
        dz0 = vcat(dλT[:], -(Fp(T)')*λT)
        # TODO: Can we delete these row below here perhaps?
        # z0  = vcat(λ0, 0.0)
        # dz0 = vcat(dλ0, λ0[1])   # For some reasing (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired

        if N_trans > 0
            @warn "The returned function get_Gp() doesn't fully support N_trans > 0, as sensitivity of internal variables not known at any other time than t=0. A non-rigorous approximation is used instead."
        end
        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            # NOTE: Changes signs to match what I had in my manual calculations, seems correct now
            # Gp = adj_sol.u[end][end-np+1:end] + (((adj_sol.u[end][1:end-np]')*Fdx(0.0))*xp0)[:]
            # Gp = adj_sol.u[end-N_trans][nx+1:nx+np] + (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)[:]
            Gp = adj_sol.u[end-N_trans][nx+1:nx+np] .+ (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)
            # return 0.0
        end

        dvars = vcat(fill(true, 4), fill(false, 3), fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0 here, λ0: $λ0, dλ0: $dλ0"

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp
    end
end

function pendulum_adj_stepbystep_NEW(Φ::Float64, u::Function, w::Function, θ::Vector{Float64}, y::Function, x_func::Function, dx_func::Function, λ::Function, dλ::Function, Fp::Function, T::Float64)::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # x  = t -> sol(t)
        Fx = (x, dx) -> [2dx[6]        0               0   -1                  0           0   0
                          0           2*dx[6]          0   0                   -1          0   0
                          -dx[3]         0             0   2k*abs(x[4])        0           0   0
                          0          -dx[3]            0   0               2k*abs(x[5])    0   0
                          2x[1]      2x[2]             0   0                   0           0   0
                          x[4]        x[5]             0   x[1]              x[2]          0   0
                          x[2]/(L^2)  -x[1]/(L^2)      0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = (x, dx) -> vcat([1   0   0          0   0   2x[1]    0
                               0   1   0          0   0   2x[2]    0
                               0   0   -x[1]   m   0   0           0
                               0   0   -x[2]   0   m   0           0], zeros(3,7))
        Fddx = (x, dx) -> vcat([  0   0  0            0   0   2dx[1]    0
                                  0   0  0            0   0   2dx[2]    0
                                  0   0  -dx[1]       0   0   0         0
                                  0   0  -dx[2]       0   0   0         0], zeros(3,7))
       gₓ = (x, dx, t) -> [0.; 0.; 0.; 0.; 0.; 0.; 2(x[7]-first(y(t)))/T]

        # the residual function
        function f!(res, dz, z, θ, t)
            wt = w(t)
            ut = u(t)
            xt = x_func(t)
            dxt = dx_func(t)
            x = xt[1:7]
            dx = dxt[1:7]
            xθ = xt[8:end]
            dxθ = dxt[8:end]

            res[1] = dz[1] - (gₓ(x,dx,t)')*xθ
            res[2] = dz[2] - ( (gₓ(x,dx,t)')*xθ - first((λ(t)')*(Fp(x,dx) + Fx(x,dx)*xθ + Fdx(x,dx)*dxθ)) )
            res[3] = dz[3] - ( -first((λ(t)')*Fp(x,dx)) - ( -gₓ(x,dx,t)' + (λ(t)')*(Fx(x,dx)-Fddx(x,dx)) - (dλ(t)')*Fdx(x,dx) )*xθ )
            res[4] = dz[4] - first(( -(λ(t)')*Fp(x,dx) ))
            # SOME BONUS DEBUG TERMS
            res[5] = dz[5] - ((x[7]-first(y(t)))^2)/T   # Simply computing integral cost
            res[6] = dz[6] + first((λ(t)')*(Fp(x,dx) + Fx(x,dx)*xθ + Fdx(x,dx)*dxθ))    # This should be approximately zero, otherwise forward solution isn't right

            nothing
        end

        # Finding consistent initial conditions
        x_func0 = x_func(0.)
        dx_func0 = dx_func(0.)
        x0 = x_func0[1:7]
        dx0 = dx_func0[1:7]
        xθ0 = x_func0[8:end]
        dxθ0 = dx_func0[8:end]

        dz10 = (gₓ(x0,dx0,0.)')*xθ0
        dz20 = (gₓ(x0,dx0,0.)')*xθ0 - first((λ(0.)')*(Fp(x0,dx0) + Fx(x0,dx0)*xθ0 + Fdx(x0,dx0)*dxθ0))
        dz30 = -first((λ(0.)')*Fp(x0,dx0)) - ( -gₓ(x0,dx0,0.)' + (λ(0.)')*(Fx(x0,dx0)-Fddx(x0,dx0)) - (dλ(0.)')*Fdx(x0,dx0) )*xθ0
        dz40 = -first((λ(0.)')*Fp(x0,dx0))
        dz50 = ((x0[7]-first(y(0.)))^2)/T
        dz60 = -first((λ(0.)')*(Fp(x0,dx0) + Fx(x0,dx0)*xθ0 + Fdx(x0,dx0)*dxθ0))

        z0   = zeros(6)
        dz0  = [dz10, dz20, dz30, dz40, dz50, dz60]

        dvars = fill(true,6)

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, z0, dz0, dvars, r0)
    end
end

function pendulum_adj_stepbystep_dist(u::Function, w::Function, xw::Function, v::Function, θ::Vector{Float64}, y::Function, dy::Function, x_func::Function, dx_func::Function, λ::Function, dλ::Function, Fp::Function, B̃::Matrix{Float64}, B̃θ::Matrix{Float64}, η::Vector{Float64}, T::Float64)::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        np = 1
        nw = length(w(0.0))÷(np+1)
        nxw = length(xw(0.0))÷(np+1)
        nx = 7

        # Pendulum original matrices
        Fx = (x, dx) -> [2dx[6]        0               0   -1                  0           0   0
                          0           2*dx[6]          0   0                   -1          0   0
                          -dx[3]         0             0   2k*abs(x[4])        0           0   0
                          0          -dx[3]            0   0               2k*abs(x[5])    0   0
                          2x[1]      2x[2]             0   0                   0           0   0
                          x[4]        x[5]             0   x[1]              x[2]          0   0
                          x[2]/(L^2)  -x[1]/(L^2)      0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable dx/dt)
        Fdx = (x, dx) -> vcat([1   0   0          0   0   2x[1]    0
                               0   1   0          0   0   2x[2]    0
                               0   0   -x[1]   m   0   0           0
                               0   0   -x[2]   0   m   0           0], zeros(3,7))
        Fddx = (x, dx) -> vcat([  0   0  0            0   0   2dx[1]    0
                                  0   0  0            0   0   2dx[2]    0
                                  0   0  -dx[1]       0   0   0         0
                                  0   0  -dx[2]       0   0   0         0], zeros(3,7))
        gₓ = (x, dx, t) -> [0.; 0.; 0.; 0.; 0.; 0.; 2(x[7]-first(y(t)))/T]

        Fw = (wt) -> [0.0; 0.0; -2*wt; 0.0; 0.0; 0.0; 0.0]

        # NOTE: A LITTLE HARD-CODED FOR NOW, NOT REALLY GENERALIZED FOR OTHER DISTURBANCE MODELS
        # Creating some of the needed disturbance from η. ASSUMING ONLY FREE DISTURBANCE PARAMETERS ARE a1
        A = [-η[1]  -η[2]; 1.0   0.0]
        C = [η[3]   η[4]]
        Aθ = [-1.0  0.0; 0.0    0.0;]
        Cθ = zeros(1,nxw) # Only depends on C-parameters, and we only compute sensitivities wrt to a-parameters. We have 1 disturbance parameter, thus 1 row

        # Matrices extended to include disturbance model
        Fz = (x, dx, wt) -> [Fx(x,dx)      zeros(nx, nxw)   Fw(wt)
                            zeros(nxw,nx)   -A         zeros(nxw, nw)
                            zeros(1,nx)     -C              1.0]

        Fdz = (x, dx) -> [Fdx(x,dx)   zeros(nx,nxw)   zeros(nx,nw)
                         zeros(nxw,nx)  [1.0 0.0; 0.0 1.0]   zeros(nxw,nw)
                         zeros(1,nx+nxw+nw)]

        Fddz = (x, dx) -> [Fddx(x,dx)   zeros(nx,nxw)   zeros(nx,nw)
                          zeros(nxw,nx) zeros(nxw,nxw) zeros(nxw, nw)
                          zeros(1,nx+nxw+nw)]

        gz = (x, dx, t) -> [gₓ(x,dx,t); zeros(nxw+nw,1)]

        F̃p = (x, dx, xwt, vt) -> [Fp(x,dx)
                                -Aθ*xwt - B̃θ*vt
                                -Cθ*xwt]

        # the residual function
        function f!(res, dz, z, θ, t)
            wt = w(t)
            # ut = u(t)
            xt = x_func(t)
            dxt = dx_func(t)
            xwt = xw(t)
            vt = v(t)
            x = xt[1:7]
            dx = dxt[1:7]
            xθ = xt[8:end]
            dxθ = dxt[8:end]
            zθ = vcat(xθ, xwt[3:4], wt[2:2])    # If disturbance model includes forward sensitivities, the sensitivities are the second half of the state variables (3:4) and of the output (2:2). This is hard-coded for pendulum disturbance model
            dzθ = vcat(dxθ, Aθ*xwt[1:2] + A*xwt[3:4] + B̃θ*vt, C*(Aθ*xwt[1:2] + A*xwt[3:4] + B̃θ*vt))

            res[1] = dz[1] - first((gz(x,dx,t)')*zθ)
            res[2] = dz[2] - first( (gz(x,dx,t)')*zθ - (λ(t)')*(F̃p(x,dx,xwt[1:2],vt) + Fz(x,dx,wt[1])*zθ + Fdz(x,dx)*dzθ) )
            res[3] = dz[3] - first( -(λ(t)')*F̃p(x,dx,xwt[1:2],vt) - ( -gz(x,dx,t)' + (λ(t)')*(Fz(x,dx,wt[1])-Fddz(x,dx)) - (dλ(t)')*Fdz(x,dx) )*zθ )
            res[4] = dz[4] - first(( -(λ(t)')*F̃p(x,dx,xwt[1:2],vt) ))
            # SOME BONUS DEBUG TERMS
            res[5] = dz[5] - ((x[7]-first(y(t)))^2)/T   # Simply computing integral cost
            res[6] = dz[6] + first((λ(t)')*(F̃p(x,dx,xwt[1:2],vt) + Fz(x,dx,wt[1])*zθ + Fdz(x,dx)*dzθ))    # This should be approximately zero, otherwise forward solution isn't right

            nothing
        end

        # Finding consistent initial conditions
        x_func0 = x_func(0.)
        dx_func0 = dx_func(0.)
        x0 = x_func0[1:7]
        dx0 = dx_func0[1:7]
        xθ0 = x_func0[8:end]
        dxθ0 = dx_func0[8:end]
        xw0 = xw(0.0)
        w0 = w(0.0)
        zθ0 = vcat(xθ0, xw0[3:4], w0[2:2])
        dzθ0 = vcat(dxθ0, A*xw0[1:2] + B̃*v(0.0), C*(A*xw0[1:2] + B̃*v(0.0)))
        v0 = v(0.)

        dz10 = first((gz(x0,dx0,0.)')*zθ0)
        dz20 = first( (gz(x0,dx0,0.)')*zθ0 - (λ(0.)')*(F̃p(x0,dx0,xw0[1:2],v0) + Fz(x0,dx0,w0[1])*zθ0 + Fdz(x0,dx0)*dzθ0))
        dz30 = -first((λ(0.)')*F̃p(x0,dx0,xw0[1:2],v0) - ( -gz(x0,dx0,0.)' + (λ(0.)')*(Fz(x0,dx0,w0[1])-Fddz(x0,dx0)) - (dλ(0.)')*Fdz(x0,dx0) )*zθ0)
        dz40 = -first((λ(0.)')*F̃p(x0,dx0,xw0[1:2],v0))
        dz50 = ((x0[7]-first(y(0.)))^2)/T
        dz60 = -first((λ(0.)')*(F̃p(x0,dx0,xw0[1:2],v0) + Fz(x0,dx0,w0[1])*zθ0 + Fdz(x0,dx0)*dzθ0))

        z0   = zeros(6)
        dz0  = [dz10, dz20, dz30, dz40, dz50, dz60]

        dvars = fill(true,6)

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, z0, dz0, dvars, r0)
    end
end

function pendulum_adj_stepbystep_k(Φ::Float64, u::Function, w::Function, θ::Vector{Float64}, y::Function, λ::Function, dλ::Function, T::Float64)::Model
    @warn "pendulum_adj_stepbystep_k only adapted for k-parameter currently, no others"
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # x  = t -> sol(t)
        Fx = (x, dx) -> [2dx[6]        0               0   -1                  0           0   0
                          0           2*dx[6]          0   0                   -1          0   0
                          -dx[3]         0             0   2k*abs(x[4])        0           0   0
                          0          -dx[3]            0   0               2k*abs(x[5])    0   0
                          2x[1]      2x[2]             0   0                   0           0   0
                          x[4]        x[5]             0   x[1]              x[2]          0   0
                          x[2]/(L^2)  -x[1]/(L^2)      0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = (x, dx) -> vcat([1   0   0          0   0   2x[1]    0
                               0   1   0          0   0   2x[2]    0
                               0   0   -x[1]   m   0   0           0
                               0   0   -x[2]   0   m   0           0], zeros(3,7))
        Fddx = (x, dx) -> vcat([  0   0  0            0   0   2dx[1]    0
                                  0   0  0            0   0   2dx[2]    0
                                  0   0  -dx[1]       0   0   0         0
                                  0   0  -dx[2]       0   0   0         0], zeros(3,7))
        Fp = (x, dx) -> [0
                        0
                        abs(x[4])*x[4]
                        abs(x[5])*x[5]
                        0
                        0
                        0]
       gₓ = (x, dx, t) -> [0.; 0.; 0.; 0.; 0.; 0.; 2(x[7]-y(t))/T]
       # λ = [1.; 1.; 1.; 1.; 1.; 1.; 1.]

       #TODO: DON'T PASS T. BUT ALSO DON'T CALL THIS gₓ!!!!

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - x[4] + 2dx[6]*x[1]
            res[2] = dx[2] - x[5] + 2dx[6]*x[2]
            res[3] = m*dx[4] - dx[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
            res[4] = m*dx[5] - dx[3]*x[2] + k*abs(x[5])*x[5] + m*g
            res[5] = x[1]^2 + x[2]^2 - L^2
            res[6] = x[4]*x[1] + x[5]*x[2]
            # Angle of pendulum
            res[7] = x[7] - atan(x[1] / -x[2])
            # Sensitivity Equations (wrt k)
            res[8]  = -x[11] + dx[8] + 2x[1]*dx[13] + 2x[8]*dx[6]
            res[9]  = -x[12] + dx[9] + 2x[2]*dx[13] + 2x[9]*dx[6]
            res[10] = abs(x[4])*x[4] + 2k*x[11]*abs(x[4]) - x[1]*dx[10] + m*dx[11] - x[8]*dx[3]
            res[11] = abs(x[5])*x[5] + 2k*x[12]*abs(x[5]) - x[2]*dx[10] + m*dx[12] - x[9]*dx[3]
            res[12] = -2x[8]*x[1] - 2x[9]*x[2]
            res[13] = x[11]*x[1] + x[12]*x[2] + x[8]*x[4] + x[9]*x[5]
            # Sensitivity of angle of pendulum
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[8]*x[2])/(L^2)

            # res[15] = dx[15] - (y(t) - y(t;θ))^2 (point 1)
            res[15] = dx[15] - (y(t)-x[7])^2
            # res[16] = dx[16] - 2(y(t;θ)-y(t))*y_θ(t) (point 2)
            res[16] = dx[16] - 2(x[7]-y(t))*x[14]

            # # Comment out and just use explicilty
            # xθ  = x[8:14]
            # dxθ = dx[8:14]
            # res[17] = dx[17] - 2(y(t;θ)-y(t))*y_θ(t) - (λ')*(Fx(t)*xθ + Fdx(t)*dxθ + Fp(t)) (point 3)
            res[17] = dx[17] - 2(x[7]-y(t))*x[14] + (λ(t)')*(Fx(x, dx)*x[8:14] + Fdx(x, dx)*dx[8:14] + Fp(x, dx))

            # res[18] = dx[18] + (λ')*Fp(x, dx) + ( (λ')*(Fx(x,dx)-Fddx(x,dx)) - (dλ')*Fdx(x,dx) - gx(x,dx,t)' )*xθ (point 4)
            res[18] = dx[18] + (λ(t)')*Fp(x, dx) + ( (λ(t)')*(Fx(x,dx)-Fddx(x,dx)) - (dλ(t)')*Fdx(x,dx) - gₓ(x,dx,t)' )*x[8:14]

            # res[19] = dx[19] + (λ')*Fp(x, dx) (point 5)
            res[19] = dx[19] + (λ(t)')*Fp(x, dx)
            # res[20] = dx[20]  + ( (λ')*(Fx(x,dx)-Fddx(x,dx)) - (dλ')*Fdx(x,dx) - gx(x,dx,t)' )*xθ
            res[20] = dx[20] + ( (λ(t)')*(Fx(x,dx)-Fddx(x,dx)) - (dλ(t)')*Fdx(x,dx) - gₓ(x,dx,t)' )*x[8:14]

            # res[21] = dx[21] - (gx(x,dx,t)')*xθ + (λ(t)')*( Fx(x,dx)*xθ + Fdx(x,dx)*dxθ + Fp(x,dx)
            res[21] = dx[21] - (gₓ(x,dx,t)')*x[8:14] + (λ(t)')*( Fx(x,dx)*x[8:14] + Fdx(x,dx)*dx[8:14] + Fp(x,dx) )

            # res[22] = dx[22] - (λ(t)')*Fdx(t)*dxθ - ( (dλ(t)')*Fdx(t) + (λ(t)')*Fddx(t) )*xθ
            res[22] = dx[22] - (λ(t)')*Fdx(x,dx)*dx[8:14] - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]
            # res[22] = dx[22] - 2*sin(t)*cos(t)*(t^3) - (sin(t)^2)*3*t^2

            # res[23:29] = dx[23:29] - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )[:]

            # Shared between 3.5 and 4
            res[23] = dx[23] - (gₓ(x,dx,t)')*x[8:14] + (λ(t)')*( Fx(x,dx)*x[8:14] + Fp(x,dx) )
            # Extra for 3.5 (for m and k only)
            res[24] = dx[24] + (λ(t)')*Fdx(x,dx)*dx[8:14]
            # Extra for 4 (for m and k only)
            res[25] = dx[25] - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # NOTE: LoL, these are just negated versions of the above, clearly not very equal!
            # For replacement using partial integration
            # LHS
            res[26] = dx[26] - (λ(t)')*Fdx(x,dx)*dx[8:14]
            # RHS (without term, assumed zero, which is the case for m and correct λ)
            res[27] = dx[27] + ((dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # (point 3.9) (rearranged point 4 for easier comparison with point 3.5)
            res[28] = dx[28] - (gₓ(x,dx,t)')*x[8:14] + (λ(t)')*(Fx(x,dx)*x[8:14] + Fp(x, dx)) - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # REALLY SMAL STUFF; JUST DELETE
            res[29] = dx[29] - (dλ(t)')*Fdx(x,dx)*x[8:14]
            res[30] = dx[30] - (λ(t)')*Fddx(x,dx)*x[8:14]

            # res[30]    = dx[30] - 2*sin(t)*cos(t)*cos(t-0.3) + (sin(t)^2)*sin(t-0.3)

            # This is the equations for my_pendulum_adjoint_konly
            # res[1:7] = (-dz[1:7]')*Fdx(T-t) + (z[1:7]')*(Fddx(T-t) - Fx(T-t)) + gₓ(T-t)
            # <=> (z[1:7]')*(Fddx(T-t) - Fx(T-t)) - (dz[1:7]')*Fdx(T-t) + gₓ(T-t)
            # <=> (z[1:7]')*(Fx(T-t) - Fddx(T-t)) + (dz[1:7]')*Fdx(T-t) - gₓ(T-t)
            # THERE IS A SIGN INCONSISTENCY!!!!!!!!!!!!!!!!
            # Yes becuase that one is solved backwards you silly...

            # These equations sensitivity are written on the intuitive form. To
            # obtain form correct for this funciton, do the following replacements:
            # s[1] -> x[8],  s[2] -> x[9],  s[3] -> x[10]
            # s[4] -> x[11], s[5] -> x[12], s[6] -> x[13]
            # and the corresponding replacements for sp and dx

            # res[8]  = -s[4] + sp[1] + 2x[1]*sp[6] + 2s[1]*dx[6]
            # res[9]  = -s[5] + sp[2] + 2x[2]*sp[6] + 2s[2]*dx[6]
            # res[10]  = abs(x[4])*x[4] + 2k*s[4]*abs(x[4]) - x[1]*sp[3] + m*sp[4] - s[1]*dx[3]
            # res[11] = abs(x[5])*x[5] + 2k*s[5]*abs(x[5]) - x[2]*sp[3] + m*sp[5] - s[2]*dx[3]
            # res[12] = -2s[1]*x[1] - 2s[2]*x[2]
            # res[13] = s[4]*x[1] + s[5]*x[2] + s[1]*x[4] + s[2]*x[5]

            nothing
        end

        # Finding consistent initial conditions
        # Initial values, the pendulum starts at rest
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        x1_0 = L * sin(Φ)
        x2_0 = -L * cos(Φ)
        dx3_0 = m*g/x2_0
        dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m

        pend0  = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dpend0 = [0., 0., dx3_0, dx4_0, 0., 0., 0.]
        s0     = zeros(7)
        sp0    = zeros(7)
        G0    = 0.0
        dG0   = (y(0.0)-pend0[7])^2
        Gp0   = 0.0
        dGp0  = 2(pend0[7]-y(0.0))*0.0
        Gp02  = 0.0
        dGp02 = dGp0 - (λ(0.)')*(Fx(pend0, dpend0)*s0 + Fdx(pend0, dpend0)*sp0 + Fp(pend0, dpend0))
        Gp03  = 0.0
        dGp03 = -(λ(0.)')*Fp(pend0, dpend0) - ( (λ(0.)')*(Fx(pend0,dpend0)-Fddx(pend0,dpend0)) -(dλ(0.)')*Fdx(pend0,dpend0) - gₓ(pend0,dpend0,0.0)' )*s0
        Gp04  = 0.0
        dGp04 = -(λ(0.)')*Fp(pend0, dpend0)
        Gp04b  = 0.0
        dGp04b = - ( (λ(0.)')*(Fx(pend0,dpend0)-Fddx(pend0,dpend0)) -(dλ(0.)')*Fdx(pend0,dpend0) - gₓ(pend0,dpend0,0.0)' )*s0
        Gp035  = 0.0
        dGp035 = (gₓ(pend0,dpend0,0.0)')*s0 - (λ(0.0)')*( Fx(pend0,dpend0)*s0 + Fdx(pend0,dpend0)*sp0 + Fp(pend0,dpend0) )
        partial    = 0.0
        dpartial   = (λ(0.0)')*Fdx(pend0,dpend0)*sp0 + ( (dλ(0.0)')*Fdx(pend0,dpend0) + (λ(0.0)')*Fddx(pend0,dpend0) )*s0

        # NOTE: ONLY MAKE SENSE FOR k AND m SINCE ONLY THEN APPROPRIATE QUANTITIES ARE ZERO
        common   = 0.0
        dcommon  = (gₓ(pend0,dpend0,0.0)')*s0 - (λ(0.)')*( Fx(pend0,dpend0)*s0 + Fp(pend0,dpend0) )
        extra35  = 0.0
        dextra35 = -(λ(0.)')*Fdx(pend0,dpend0)*sp0
        extra4   = 0.0
        dextra4  = ( (dλ(0.0)')*Fdx(pend0,dpend0) + (λ(0.0)')*Fddx(pend0,dpend0) )*s0

        partrepL  = 0.0
        dpartrepL = (λ(0.)')*Fdx(pend0,dpend0)*sp0
        partrepR  = 0.0
        dpartrepR = -((dλ(0.)')*Fdx(pend0,dpend0) + (λ(0.)')*Fddx(pend0,dpend0) )*s0

        oscdeb = 0.0
        doscdeb = (gₓ(pend0,dpend0,0.0)')*s0 - (λ(0.0)')*(Fx(pend0,dpend0)*s0 + Fp(pend0, dpend0)) + (λ(0.0)')*Fddx(pend0,dpend0)*s0 + (dλ(0.0)')*Fdx(pend0,dpend0)*s0
        res29  = 0.0
        dres29 = (dλ(0.0)')*Fdx(pend0,dpend0)*s0
        res30  = 0.0
        dres30 = (λ(0.0)')*Fddx(pend0,dpend0)*s0

        # b1  = 0.0
        # db1 = 0.0

        # tue_deb    = (λ(0.0)')*Fdx(pend0, dpend0)
        # dtue_deb   = (dλ(0.0)')*Fdx(pend0, dpend0) + (λ(0.0)')*Fddx(pend0, dpend0)

        x0   = vcat(pend0, s0, G0, Gp0, Gp02, Gp03, Gp04, Gp04b, Gp035, partial, common, extra35, extra4, partrepL, partrepR, oscdeb, res29, res30)#, b1)
        dx0  = vcat(dpend0, sp0, dG0, dGp0, dGp02, dGp03, dGp04, dGp04b, dGp035, dpartial, dcommon, dextra35, dextra4, dpartrepL, dpartrepR, doscdeb, dres29, dres30)#, db1)

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false, true, true, true, true, true, true, true, true], fill(true, 8))

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_adj_stepbystep_L(Φ::Float64, u::Function, w::Function, θ::Vector{Float64}, y::Function, λ::Function, dλ::Function, T::Float64)::Model
    @warn "pendulum_adj_stepbystep_L only adapted for L-parameter currently, no others"
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # x  = t -> sol(t)
        Fx = (x, dx) -> [2dx[6]        0               0   -1                  0           0   0
                          0           2*dx[6]          0   0                   -1          0   0
                         -dx[3]        0               0   2k*abs(x[4])        0           0   0
                          0          -dx[3]            0   0               2k*abs(x[5])    0   0
                          2x[1]       2x[2]            0   0                   0           0   0
                          x[4]        x[5]             0   x[1]               x[2]         0   0
                          x[2]/(L^2)  -x[1]/(L^2)      0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = (x, dx) -> vcat([1   0   0          0   0   2x[1]    0
                               0   1   0          0   0   2x[2]    0
                               0   0   -x[1]   m   0   0           0
                               0   0   -x[2]   0   m   0           0], zeros(3,7))
        Fddx = (x, dx) -> vcat([  0   0  0            0   0   2dx[1]    0
                                  0   0  0            0   0   2dx[2]    0
                                  0   0  -dx[1]    0   0   0            0
                                  0   0  -dx[2]    0   0   0            0], zeros(3,7))
        Fp = (x, dx) -> [0
                         0
                         0
                         0
                        -2L
                         0
                         0]
       gₓ = (x, dx, t) -> [0.; 0.; 0.; 0.; 0.; 0.; 2(x[7]-y(t))/T]
       # λ = [1.; 1.; 1.; 1.; 1.; 1.; 1.]

       #TODO: DON'T PASS T. BUT ALSO DON'T CALL THIS gₓ!!!! EDIT: Why...?

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - x[4] + 2dx[6]*x[1]
            res[2] = dx[2] - x[5] + 2dx[6]*x[2]
            res[3] = m*dx[4] - dx[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
            res[4] = m*dx[5] - dx[3]*x[2] + k*abs(x[5])*x[5] + m*g
            res[5] = x[1]^2 + x[2]^2 - L^2
            res[6] = x[4]*x[1] + x[5]*x[2]
            # Angle of pendulum
            res[7] = x[7] - atan(x[1] / -x[2])

            # Sensitivity equations for L
            res[8]  = dx[8] - x[11] + 2x[8]dx[6] + 2dx[13]x[1]
            res[9]  = dx[9] - x[12] + 2x[9]dx[6] + 2dx[13]x[2]
            res[10] = 2k*x[11]*abs(x[4]) + m*dx[11] - x[8]dx[3] - dx[10]x[1]
            res[11] = 2k*x[12]*abs(x[5]) + m*dx[12] - x[9]dx[3] - dx[10]x[2]
            res[12] = 2x[8]x[1] + 2x[9]x[2] - 2L
            res[13] = x[8]x[4] + x[11]x[1] + x[9]x[5] + x[12]x[2]
            # Sensitivity of angle of pendulum to L
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[8]*x[2])/(L^2)

            # res[15] = dx[15] - (y(t) - y(t;θ))^2 (point 1)
            res[15] = dx[15] - (y(t)-x[7])^2
            # res[16] = dx[16] - 2(y(t;θ)-y(t))*y_θ(t) (point 2)
            res[16] = dx[16] - 2(x[7]-y(t))*x[14]

            # # Comment out and just use explicilty
            # xθ  = x[8:14]
            # dxθ = dx[8:14]
            # res[17] = dx[17] - 2(y(t;θ)-y(t))*y_θ(t) - (λ')*(Fx(t)*xθ + Fdx(t)*dxθ + Fp(t)) (point 3)
            res[17] = dx[17] - 2(x[7]-y(t))*x[14] + (λ(t)')*(Fx(x, dx)*x[8:14] + Fdx(x, dx)*dx[8:14] + Fp(x, dx))

            # res[18] = dx[18] + (λ')*Fp(x, dx) + ( (λ')*(Fx(x,dx)-Fddx(x,dx)) - (dλ')*Fdx(x,dx) - gx(x,dx,t)' )*xθ (point 4)
            res[18] = dx[18] + (λ(t)')*Fp(x, dx) + ( (λ(t)')*(Fx(x,dx)-Fddx(x,dx)) - (dλ(t)')*Fdx(x,dx) - gₓ(x,dx,t)' )*x[8:14]

            # res[19] = dx[19] + (λ')*Fp(x, dx) (point 5)
            res[19] = dx[19] + (λ(t)')*Fp(x, dx)
            # res[20] = dx[20]  + ( (λ')*(Fx(x,dx)-Fddx(x,dx)) - (dλ')*Fdx(x,dx) - gx(x,dx,t)' )*xθ
            res[20] = dx[20] + ( (λ(t)')*(Fx(x,dx)-Fddx(x,dx)) - (dλ(t)')*Fdx(x,dx) - gₓ(x,dx,t)' )*x[8:14]

            # res[21] = dx[21] - (gx(x,dx,t)')*xθ + (λ(t)')*( Fx(x,dx)*xθ + Fdx(x,dx)*dxθ + Fp(x,dx
            res[21] = dx[21] - (gₓ(x,dx,t)')*x[8:14] + (λ(t)')*( Fx(x,dx)*x[8:14] + Fdx(x,dx)*dx[8:14] + Fp(x,dx) )

            # res[22] = dx[22] - (λ(t)')*Fdx(t)*dxθ - ( (dλ(t)')*Fdx(t) + (λ(t)')*Fddx(t) )*xθ
            res[22] = dx[22] - (λ(t)')*Fdx(x,dx)*dx[8:14] - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # Shared between 3.5 and 4
            res[23] = dx[23] - (gₓ(x,dx,t)')*x[8:14] + (λ(t)')*( Fx(x,dx)*x[8:14] + Fp(x,dx) )
            # Extra for 3.5 (for m and k only)
            res[24] = dx[24] + (λ(t)')*Fdx(x,dx)*dx[8:14]
            # Extra for 4 (for m and k only)
            res[25] = dx[25] - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # NOTE: LoL, these are just negated versions of the above, clearly not very equal!
            # For replacement using partial integration
            # LHS
            res[26] = dx[26] - (λ(t)')*Fdx(x,dx)*dx[8:14]
            # RHS (without term, assumed zero, which is the case for m and correct λ)
            res[27] = dx[27] + ((dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # (point 3.9) (rearranged point 4 for easier comparison with point 3.5)
            res[28] = dx[28] - (gₓ(x,dx,t)')*x[8:14] + (λ(t)')*(Fx(x,dx)*x[8:14] + Fp(x, dx)) - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # REALLY SMAL STUFF; JUST DELETE
            res[29] = dx[29] - (dλ(t)')*Fdx(x,dx)*x[8:14]
            res[30] = dx[30] - (λ(t)')*Fddx(x,dx)*x[8:14]

            # This is the equations for my_pendulum_adjoint_konly
            # res[1:7] = (-dz[1:7]')*Fdx(T-t) + (z[1:7]')*(Fddx(T-t) - Fx(T-t)) + gₓ(T-t)
            # <=> (z[1:7]')*(Fddx(T-t) - Fx(T-t)) - (dz[1:7]')*Fdx(T-t) + gₓ(T-t)
            # <=> (z[1:7]')*(Fx(T-t) - Fddx(T-t)) + (dz[1:7]')*Fdx(T-t) - gₓ(T-t)
            # THERE IS A SIGN INCONSISTENCY!!!!!!!!!!!!!!!!
            # Yes becuase that one is solved backwards you silly...

            # These equations sensitivity are written on the intuitive form. To
            # obtain form correct for this funciton, do the following replacements:
            # s[1] -> x[8],  s[2] -> x[9],  s[3] -> x[10]
            # s[4] -> x[11], s[5] -> x[12], s[6] -> x[13]
            # and the corresponding replacements for sp and dx

            # res[8]  = -s[4] + sp[1] + 2x[1]*sp[6] + 2s[1]*dx[6]
            # res[9]  = -s[5] + sp[2] + 2x[2]*sp[6] + 2s[2]*dx[6]
            # res[10]  = abs(x[4])*x[4] + 2k*s[4]*abs(x[4]) - x[1]*sp[3] + m*sp[4] - s[1]*dx[3]
            # res[11] = abs(x[5])*x[5] + 2k*s[5]*abs(x[5]) - x[2]*sp[3] + m*sp[5] - s[2]*dx[3]
            # res[12] = -2s[1]*x[1] - 2s[2]*x[2]
            # res[13] = s[4]*x[1] + s[5]*x[2] + s[1]*x[4] + s[2]*x[5]

            nothing
        end

        # Finding consistent initial conditions
        # Initial values, the pendulum starts at rest
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        x1_0 = L * sin(Φ)
        x2_0 = -L * cos(Φ)
        dx3_0 = m*g/x2_0
        dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m

        pend0  = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dpend0 = [0., 0., dx3_0, dx4_0, 0., 0., 0.]
        s0  = vcat([x1_0/L, x2_0/L], zeros(5))
        ds0 = vcat([0.,0., -dx3_0/L], zeros(4))
        G0    = 0.0
        dG0   = (y(0.0)-pend0[7])^2
        Gp0   = 0.0
        dGp0  = 2(pend0[7]-y(0.0))*0.0
        Gp02  = 0.0
        dGp02 = dGp0 - (λ(0.)')*(Fx(pend0, dpend0)*s0 + Fdx(pend0, dpend0)*ds0 + Fp(pend0, dpend0))
        Gp03  = 0.0
        dGp03 = -(λ(0.)')*Fp(pend0, dpend0) - ( (λ(0.)')*(Fx(pend0,dpend0)-Fddx(pend0,dpend0)) -(dλ(0.)')*Fdx(pend0,dpend0) - gₓ(pend0,dpend0,0.0)' )*s0
        Gp04  = 0.0
        dGp04 = -(λ(0.)')*Fp(pend0, dpend0)
        Gp04b  = 0.0
        dGp04b = - ( (λ(0.)')*(Fx(pend0,dpend0)-Fddx(pend0,dpend0)) -(dλ(0.)')*Fdx(pend0,dpend0) - gₓ(pend0,dpend0,0.0)' )*s0
        Gp035  = 0.0
        dGp035 = (gₓ(pend0,dpend0,0.0)')*s0 - (λ(0.0)')*( Fx(pend0,dpend0)*s0 + Fdx(pend0,dpend0)*ds0 + Fp(pend0,dpend0) )
        partial    = 0.0
        dpartial   = (λ(0.0)')*Fdx(pend0,dpend0)*ds0 + ( (dλ(0.0)')*Fdx(pend0,dpend0) + (λ(0.0)')*Fddx(pend0,dpend0) )*s0

        # tue_deb    = (λ(0.0)')*Fdx(pend0, dpend0)
        # dtue_deb   = (dλ(0.0)')*Fdx(pend0, dpend0) + (λ(0.0)')*Fddx(pend0, dpend0)
        # NOTE: ONLY MAKE SENSE FOR k AND m SINCE ONLY THEN APPROPRIATE QUANTITIES ARE ZERO
        common   = 0.0
        dcommon  = (gₓ(pend0,dpend0,0.0)')*s0 - (λ(0.)')*( Fx(pend0,dpend0)*s0 + Fp(pend0,dpend0) )
        extra35  = 0.0
        dextra35 = -(λ(0.)')*Fdx(pend0,dpend0)*ds0
        extra4   = 0.0
        dextra4  = ( (dλ(0.0)')*Fdx(pend0,dpend0) + (λ(0.0)')*Fddx(pend0,dpend0) )*s0

        partrepL  = 0.0
        dpartrepL = (λ(0.)')*Fdx(pend0,dpend0)*ds0
        partrepR  = 0.0
        dpartrepR = -((dλ(0.)')*Fdx(pend0,dpend0) + (λ(0.)')*Fddx(pend0,dpend0) )*s0
        oscdeb = 0.0
        doscdeb = (gₓ(pend0,dpend0,0.0)')*s0 - (λ(0.0)')*(Fx(pend0,dpend0)*s0 + Fp(pend0, dpend0)) + (λ(0.0)')*Fddx(pend0,dpend0)*s0 + (dλ(0.0)')*Fdx(pend0,dpend0)*s0
        res29  = 0.0
        dres29 = (dλ(0.0)')*Fdx(pend0,dpend0)*s0
        res30  = 0.0
        dres30 = (λ(0.0)')*Fddx(pend0,dpend0)*s0

        x0   = vcat(pend0, s0, G0, Gp0, Gp02, Gp03, Gp04, Gp04b, Gp035, partial, common, extra35, extra4, partrepL, partrepR, oscdeb, res29, res30)
        dx0  = vcat(dpend0, ds0, dG0, dGp0, dGp02, dGp03, dGp04, dGp04b, dGp035, dpartial, dcommon, dextra35, dextra4, dpartrepL, dpartrepR, doscdeb, dres29, dres30)

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false, true, true, true, true, true, true, true, true], fill(true, 8))
        # dvars = vcat(fill(true, 6), [false], fill(true, 6), [false, true, true, true, true, true, true, true, true], fill(true, 7))

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)
        # @info "r0 is $r0"

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_adj_stepbystep_m(Φ::Float64, u::Function, w::Function, θ::Vector{Float64}, y::Function, λ::Function, dλ::Function, T::Float64)::Model
    @warn "pendulum_adj_stepbystep_m only adapted for m-parameter currently, no others"
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # x  = t -> sol(t)
        Fx = (x, dx) -> [2dx[6]        0               0   -1                  0           0   0
                          0           2*dx[6]          0   0                   -1          0   0
                         -dx[3]        0               0   2k*abs(x[4])        0           0   0
                          0          -dx[3]            0   0               2k*abs(x[5])    0   0
                          2x[1]       2x[2]            0   0                   0           0   0
                          x[4]        x[5]             0   x[1]               x[2]         0   0
                          x[2]/(L^2)  -x[1]/(L^2)      0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = (x, dx) -> vcat([1   0   0          0   0   2x[1]    0
                               0   1   0          0   0   2x[2]    0
                               0   0   -x[1]   m   0   0           0
                               0   0   -x[2]   0   m   0           0], zeros(3,7))
        Fddx = (x, dx) -> vcat([  0   0  0            0   0   2dx[1]    0
                                  0   0  0            0   0   2dx[2]    0
                                  0   0  -dx[1]    0   0   0            0
                                  0   0  -dx[2]    0   0   0            0], zeros(3,7))
        Fp = (x, dx) -> [ .0
                  .0
                  dx[4]
                  dx[5]+g
                  .0
                  .0
                  .0]
       gₓ = (x, dx, t) -> [0.; 0.; 0.; 0.; 0.; 0.; 2(x[7]-y(t))/T]
       # λ = [1.; 1.; 1.; 1.; 1.; 1.; 1.]

       #TODO: DON'T PASS T. BUT ALSO DON'T CALL THIS gₓ!!!! EDIT: Why...?

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - x[4] + 2dx[6]*x[1]
            res[2] = dx[2] - x[5] + 2dx[6]*x[2]
            res[3] = m*dx[4] - dx[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
            res[4] = m*dx[5] - dx[3]*x[2] + k*abs(x[5])*x[5] + m*g
            res[5] = x[1]^2 + x[2]^2 - L^2
            res[6] = x[4]*x[1] + x[5]*x[2]
            # Angle of pendulum
            res[7] = x[7] - atan(x[1] / -x[2])

            # Sensitivity with respect to m
            res[8]  = -x[11] + dx[8] + 2x[1]*dx[13] + 2x[8]*dx[6]
            res[9]  = -x[12] + dx[9] + 2x[2]*dx[13] + 2x[9]*dx[6]
            res[10] = 2k*x[11]*abs(x[4]) - x[1]*dx[10] + m*dx[11] - x[8]*dx[3] + dx[4]
            res[11] = 2k*x[12]*abs(x[5]) - x[2]*dx[10] + m*dx[12] - x[9]*dx[3] + g + dx[5]
            res[12] = -2x[8]*x[1] - 2x[9]*x[2]
            res[13] = x[11]*x[1] + x[12]*x[2] + x[8]*x[4] + x[9]*x[5]
            res[14] = x[14] - (x[1]*x[9] - x[2]*x[8])/(L^2)

            # res[15] = dx[15] - (y(t) - y(t;θ))^2 (point 1)
            res[15] = dx[15] - (y(t)-x[7])^2
            # res[16] = dx[16] - 2(y(t;θ)-y(t))*y_θ(t) (point 2)
            res[16] = dx[16] - 2(x[7]-y(t))*x[14]

            # # Comment out and just use explicilty
            # xθ  = x[8:14]
            # dxθ = dx[8:14]
            # res[17] = dx[17] - 2(y(t;θ)-y(t))*y_θ(t) - (λ')*(Fx(t)*xθ + Fdx(t)*dxθ + Fp(t)) (point 3)
            res[17] = dx[17] - 2(x[7]-y(t))*x[14] + (λ(t)')*(Fx(x, dx)*x[8:14] + Fdx(x, dx)*dx[8:14] + Fp(x, dx))

            # res[18] = dx[18] + (λ')*Fp(x, dx) + ( (λ')*(Fx(x,dx)-Fddx(x,dx)) - (dλ')*Fdx(x,dx) - gx(x,dx,t)' )*xθ (point 4)
            res[18] = dx[18] + (λ(t)')*Fp(x, dx) + ( (λ(t)')*(Fx(x,dx)-Fddx(x,dx)) - (dλ(t)')*Fdx(x,dx) - gₓ(x,dx,t)' )*x[8:14]

            # res[19] = dx[19] + (λ')*Fp(x, dx) (point 5)
            res[19] = dx[19] + (λ(t)')*Fp(x, dx)# - 12.1e-7
            # res[20] = dx[20]  + ( (λ')*(Fx(x,dx)-Fddx(x,dx)) - (dλ')*Fdx(x,dx) - gx(x,dx,t)' )*xθ
            res[20] = dx[20] + ( (λ(t)')*(Fx(x,dx)-Fddx(x,dx)) - (dλ(t)')*Fdx(x,dx) - gₓ(x,dx,t)' )*x[8:14] # (bonus point (remainder?))

            # # res[21] = dx[21] - (gx(x,dx,t)')*xθ + (λ(t)')*( Fx(x,dx)*xθ + Fdx(x,dx)*dxθ + Fp(x,dx
            # res[21] = dx[21] - (gₓ(x,dx,t)')*x[8:14] + (λ(t)')*( Fx(x,dx)*x[8:14] + Fdx(x,dx)*dx[8:14] + Fp(x,dx) ) # (point 3.5)

            # res[22] = dx[22] - (λ(t)')*Fdx(t)*dxθ - ( (dλ(t)')*Fdx(t) + (λ(t)')*Fddx(t) )*xθ
            res[22] = dx[22] - (λ(t)')*Fdx(x,dx)*dx[8:14] - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]     # (point 6)

            # res[23:29] = dx[23:29] - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )[:]

            # Shared between 3.5 and 4
            res[23] = dx[23] - (gₓ(x,dx,t)')*x[8:14] + (λ(t)')*( Fx(x,dx)*x[8:14] + Fp(x,dx) )
            # Extra for 3.5 (for m and k only)
            res[24] = dx[24] + (λ(t)')*Fdx(x,dx)*dx[8:14]
            # Extra for 4 (for m and k only)
            res[25] = dx[25] - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # For replacement using partial integration
            # LHS
            res[26] = dx[26] - (λ(t)')*Fdx(x,dx)*dx[8:14]
            # RHS (without term, assumed zero, which is the case for m and correct λ)
            res[27] = dx[27] + ((dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # res[21] = dx[21] - (gx(x,dx,t)')*xθ + (λ(t)')*(Fx(x,dx)*xθ + Fdx(x,dx)*dxθ + Fp(x,dx
            res[21] = dx[21] - (gₓ(x,dx,t)')*x[8:14] + (λ(t)')*(Fx(x,dx)*x[8:14] + Fp(x,dx) + Fdx(x,dx)*dx[8:14] ) # (point 3.5)

            # (point 3.9) (rearranged point 4 for easier comparison with point 3.5)
            res[28] = dx[28] - (gₓ(x,dx,t)')*x[8:14] + (λ(t)')*(Fx(x,dx)*x[8:14] + Fp(x, dx)) - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # REALLY SMAL STUFF; JUST DELETE
            res[29] = dx[29] - (dλ(t)')*Fdx(x,dx)*x[8:14]
            res[30] = dx[30] - (λ(t)')*Fddx(x,dx)*x[8:14]

            # This is the equations for my_pendulum_adjoint_konly
            # res[1:7] = (-dz[1:7]')*Fdx(T-t) + (z[1:7]')*(Fddx(T-t) - Fx(T-t)) + gₓ(T-t)
            # <=> (z[1:7]')*(Fddx(T-t) - Fx(T-t)) - (dz[1:7]')*Fdx(T-t) + gₓ(T-t)
            # <=> (z[1:7]')*(Fx(T-t) - Fddx(T-t)) + (dz[1:7]')*Fdx(T-t) - gₓ(T-t)
            # THERE IS A SIGN INCONSISTENCY!!!!!!!!!!!!!!!!
            # Yes becuase that one is solved backwards you silly...

            # These equations sensitivity are written on the intuitive form. To
            # obtain form correct for this funciton, do the following replacements:
            # s[1] -> x[8],  s[2] -> x[9],  s[3] -> x[10]
            # s[4] -> x[11], s[5] -> x[12], s[6] -> x[13]
            # and the corresponding replacements for sp and dx

            # res[8]  = -s[4] + sp[1] + 2x[1]*sp[6] + 2s[1]*dx[6]
            # res[9]  = -s[5] + sp[2] + 2x[2]*sp[6] + 2s[2]*dx[6]
            # res[10]  = abs(x[4])*x[4] + 2k*s[4]*abs(x[4]) - x[1]*sp[3] + m*sp[4] - s[1]*dx[3]
            # res[11] = abs(x[5])*x[5] + 2k*s[5]*abs(x[5]) - x[2]*sp[3] + m*sp[5] - s[2]*dx[3]
            # res[12] = -2s[1]*x[1] - 2s[2]*x[2]
            # res[13] = s[4]*x[1] + s[5]*x[2] + s[1]*x[4] + s[2]*x[5]

            nothing
        end

        # Finding consistent initial conditions
        # Initial values, the pendulum starts at rest
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        x1_0 = L * sin(Φ)
        x2_0 = -L * cos(Φ)
        dx3_0 = m*g/x2_0
        dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m

        pend0  = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dpend0 = [0., 0., dx3_0, dx4_0, 0., 0., 0.]
        sm0  = zeros(7)
        dsm0 = vcat(zeros(3), [-dx4_0/m, -g/m, 0., 0.])
        G0    = 0.0
        dG0   = (y(0.0)-pend0[7])^2
        Gp0   = 0.0
        dGp0  = 2(pend0[7]-y(0.0))*sm0[7]
        Gp02  = 0.0
        dGp02 = dGp0 - (λ(0.)')*(Fx(pend0, dpend0)*sm0 + Fdx(pend0, dpend0)*dsm0 + Fp(pend0, dpend0))
        Gp03  = 0.0
        dGp03 = -(λ(0.)')*Fp(pend0, dpend0) - ( (λ(0.)')*(Fx(pend0,dpend0)-Fddx(pend0,dpend0)) -(dλ(0.)')*Fdx(pend0,dpend0) - gₓ(pend0,dpend0,0.0)' )*sm0
        Gp04  = 0.0
        dGp04 = -(λ(0.)')*Fp(pend0, dpend0)
        Gp04b  = 0.0
        dGp04b = - ( (λ(0.)')*(Fx(pend0,dpend0)-Fddx(pend0,dpend0)) -(dλ(0.)')*Fdx(pend0,dpend0) - gₓ(pend0,dpend0,0.0)' )*sm0
        Gp035  = 0.0
        dGp035 = (gₓ(pend0,dpend0,0.0)')*sm0 - (λ(0.0)')*( Fx(pend0,dpend0)*sm0 + Fdx(pend0,dpend0)*dsm0 + Fp(pend0,dpend0) )
        partial    = 0.0
        dpartial   = (λ(0.0)')*Fdx(pend0,dpend0)*dsm0 + ( (dλ(0.0)')*Fdx(pend0,dpend0) + (λ(0.0)')*Fddx(pend0,dpend0) )*sm0

        # tue_deb    = (λ(0.0)')*Fdx(pend0, dpend0)
        # dtue_deb   = (dλ(0.0)')*Fdx(pend0, dpend0) + (λ(0.0)')*Fddx(pend0, dpend0)
        # NOTE: ONLY MAKE SENSE FOR k AND m SINCE ONLY THEN APPROPRIATE QUANTITIES ARE ZERO
        common   = 0.0
        dcommon  = (gₓ(pend0,dpend0,0.0)')*sm0 - (λ(0.)')*( Fx(pend0,dpend0)*sm0 + Fp(pend0,dpend0) )
        extra35  = 0.0
        dextra35 = -(λ(0.)')*Fdx(pend0,dpend0)*dsm0
        extra4   = 0.0
        dextra4  = ( (dλ(0.0)')*Fdx(pend0,dpend0) + (λ(0.0)')*Fddx(pend0,dpend0) )*sm0

        partrepL  = 0.0
        dpartrepL = (λ(0.)')*Fdx(pend0,dpend0)*dsm0
        partrepR  = 0.0
        dpartrepR = -((dλ(0.)')*Fdx(pend0,dpend0) + (λ(0.)')*Fddx(pend0,dpend0) )*sm0
        oscdeb = 0.0
        doscdeb = (gₓ(pend0,dpend0,0.0)')*sm0 - (λ(0.0)')*(Fx(pend0,dpend0)*sm0 + Fp(pend0, dpend0)) + (λ(0.0)')*Fddx(pend0,dpend0)*sm0 + (dλ(0.0)')*Fdx(pend0,dpend0)*sm0
        res29  = 0.0
        dres29 = (dλ(0.0)')*Fdx(pend0,dpend0)*sm0
        res30  = 0.0
        dres30 = (λ(0.0)')*Fddx(pend0,dpend0)*sm0

        x0   = vcat(pend0, sm0, G0, Gp0, Gp02, Gp03, Gp04, Gp04b, Gp035, partial, common, extra35, extra4, partrepL, partrepR, oscdeb, res29, res30)
        dx0  = vcat(dpend0, dsm0, dG0, dGp0, dGp02, dGp03, dGp04, dGp04b, dGp035, dpartial, dcommon, dextra35, dextra4, dpartrepL, dpartrepR, doscdeb, dres29, dres30)

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false, true, true, true, true, true, true, true, true], fill(true, 8))

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)
        # @info "r0 is $r0"

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_adj_stepbystep_g(Φ::Float64, u::Function, w::Function, θ::Vector{Float64}, y::Function, λ::Function, dλ::Function, T::Float64)::Model
    @warn "pendulum_adj_stepbystep_L only adapted for L-parameter currently, no others"
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # x  = t -> sol(t)
        Fx = (x, dx) -> [2dx[6]        0               0   -1                  0           0   0
                          0           2*dx[6]          0   0                   -1          0   0
                         -dx[3]        0               0   2k*abs(x[4])        0           0   0
                          0          -dx[3]            0   0               2k*abs(x[5])    0   0
                          2x[1]       2x[2]            0   0                   0           0   0
                          x[4]        x[5]             0   x[1]               x[2]         0   0
                          x[2]/(L^2)  -x[1]/(L^2)      0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = (x, dx) -> vcat([1   0   0          0   0   2x[1]    0
                               0   1   0          0   0   2x[2]    0
                               0   0   -x[1]   m   0   0           0
                               0   0   -x[2]   0   m   0           0], zeros(3,7))
        Fddx = (x, dx) -> vcat([  0   0  0            0   0   2dx[1]    0
                                  0   0  0            0   0   2dx[2]    0
                                  0   0  -dx[1]    0   0   0            0
                                  0   0  -dx[2]    0   0   0            0], zeros(3,7))
        Fp = (x, dx) -> [0
                         0
                         0
                         m
                         0
                         0
                         0]
       gₓ = (x, dx, t) -> [0.; 0.; 0.; 0.; 0.; 0.; 2(x[7]-y(t))/T]
       # λ = [1.; 1.; 1.; 1.; 1.; 1.; 1.]

       #TODO: DON'T PASS T. BUT ALSO DON'T CALL THIS gₓ!!!! EDIT: Why...?

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - x[4] + 2dx[6]*x[1]
            res[2] = dx[2] - x[5] + 2dx[6]*x[2]
            res[3] = m*dx[4] - dx[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
            res[4] = m*dx[5] - dx[3]*x[2] + k*abs(x[5])*x[5] + m*g
            res[5] = x[1]^2 + x[2]^2 - L^2
            res[6] = x[4]*x[1] + x[5]*x[2]
            # Angle of pendulum
            res[7] = x[7] - atan(x[1] / -x[2])

            # Sensitivity equations for L
            res[8]  = dx[8] - x[11] + 2x[8]dx[6] + 2dx[13]x[1]
            res[9]  = dx[9] - x[12] + 2x[9]dx[6] + 2dx[13]x[2]
            res[10] = 2k*x[11]*abs(x[4]) + m*dx[11] - x[8]dx[3] - dx[10]x[1]
            res[11] = 2k*x[12]*abs(x[5]) + m*dx[12] - x[9]dx[3] - dx[10]x[2] + m
            res[12] = 2x[8]x[1] + 2x[9]x[2]
            res[13] = x[8]x[4] + x[11]x[1] + x[9]x[5] + x[12]x[2]
            # Sensitivity of angle of pendulum to L
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[8]*x[2])/(L^2)

            # res[15] = dx[15] - (y(t) - y(t;θ))^2 (point 1)
            res[15] = dx[15] - (y(t)-x[7])^2
            # res[16] = dx[16] - 2(y(t;θ)-y(t))*y_θ(t) (point 2)
            res[16] = dx[16] - 2(x[7]-y(t))*x[14]

            # # Comment out and just use explicilty
            # xθ  = x[8:14]
            # dxθ = dx[8:14]
            # res[17] = dx[17] - 2(y(t;θ)-y(t))*y_θ(t) - (λ')*(Fx(t)*xθ + Fdx(t)*dxθ + Fp(t)) (point 3)
            res[17] = dx[17] - 2(x[7]-y(t))*x[14] + (λ(t)')*(Fx(x, dx)*x[8:14] + Fdx(x, dx)*dx[8:14] + Fp(x, dx))

            # res[18] = dx[18] + (λ')*Fp(x, dx) + ( (λ')*(Fx(x,dx)-Fddx(x,dx)) - (dλ')*Fdx(x,dx) - gx(x,dx,t)' )*xθ (point 4)
            res[18] = dx[18] + (λ(t)')*Fp(x, dx) + ( (λ(t)')*(Fx(x,dx)-Fddx(x,dx)) - (dλ(t)')*Fdx(x,dx) - gₓ(x,dx,t)' )*x[8:14]

            # res[19] = dx[19] + (λ')*Fp(x, dx) (point 5)
            res[19] = dx[19] + (λ(t)')*Fp(x, dx)
            # res[20] = dx[20]  + ( (λ')*(Fx(x,dx)-Fddx(x,dx)) - (dλ')*Fdx(x,dx) - gx(x,dx,t)' )*xθ
            res[20] = dx[20] + ( (λ(t)')*(Fx(x,dx)-Fddx(x,dx)) - (dλ(t)')*Fdx(x,dx) - gₓ(x,dx,t)' )*x[8:14]

            # res[21] = dx[21] - (gx(x,dx,t)')*xθ + (λ(t)')*( Fx(x,dx)*xθ + Fdx(x,dx)*dxθ + Fp(x,dx
            res[21] = dx[21] - (gₓ(x,dx,t)')*x[8:14] + (λ(t)')*( Fx(x,dx)*x[8:14] + Fdx(x,dx)*dx[8:14] + Fp(x,dx) )

            # res[22] = dx[22] - (λ(t)')*Fdx(t)*dxθ - ( (dλ(t)')*Fdx(t) + (λ(t)')*Fddx(t) )*xθ
            res[22] = dx[22] - (λ(t)')*Fdx(x,dx)*dx[8:14] - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # Shared between 3.5 and 4
            res[23] = dx[23] - (gₓ(x,dx,t)')*x[8:14] + (λ(t)')*( Fx(x,dx)*x[8:14] + Fp(x,dx) )
            # Extra for 3.5 (for m and k only)
            res[24] = dx[24] + (λ(t)')*Fdx(x,dx)*dx[8:14]
            # Extra for 4 (for m and k only)
            res[25] = dx[25] - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # NOTE: LoL, these are just negated versions of the above, clearly not very equal!
            # For replacement using partial integration
            # LHS
            res[26] = dx[26] - (λ(t)')*Fdx(x,dx)*dx[8:14]
            # RHS (without term, assumed zero, which is the case for m and correct λ)
            res[27] = dx[27] + ((dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # (point 3.9) (rearranged point 4 for easier comparison with point 3.5)
            res[28] = dx[28] - (gₓ(x,dx,t)')*x[8:14] + (λ(t)')*(Fx(x,dx)*x[8:14] + Fp(x, dx)) - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # REALLY SMAL STUFF; JUST DELETE
            res[29] = dx[29] - (dλ(t)')*Fdx(x,dx)*x[8:14]
            res[30] = dx[30] - (λ(t)')*Fddx(x,dx)*x[8:14]

            nothing
        end

        # Finding consistent initial conditions
        # Initial values, the pendulum starts at rest
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        x1_0 = L * sin(Φ)
        x2_0 = -L * cos(Φ)
        dx3_0 = m*g/x2_0
        dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m

        pend0  = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dpend0 = [0., 0., dx3_0, dx4_0, 0., 0., 0.]
        s0  = zeros(7)
        ds0 = vcat(zeros(5), [-1., 0.])
        G0    = 0.0
        dG0   = (y(0.0)-pend0[7])^2
        Gp0   = 0.0
        dGp0  = 2(pend0[7]-y(0.0))*0.0
        Gp02  = 0.0
        dGp02 = dGp0 - (λ(0.)')*(Fx(pend0, dpend0)*s0 + Fdx(pend0, dpend0)*ds0 + Fp(pend0, dpend0))
        Gp03  = 0.0
        dGp03 = -(λ(0.)')*Fp(pend0, dpend0) - ( (λ(0.)')*(Fx(pend0,dpend0)-Fddx(pend0,dpend0)) -(dλ(0.)')*Fdx(pend0,dpend0) - gₓ(pend0,dpend0,0.0)' )*s0
        Gp04  = 0.0
        dGp04 = -(λ(0.)')*Fp(pend0, dpend0)
        Gp04b  = 0.0
        dGp04b = - ( (λ(0.)')*(Fx(pend0,dpend0)-Fddx(pend0,dpend0)) -(dλ(0.)')*Fdx(pend0,dpend0) - gₓ(pend0,dpend0,0.0)' )*s0
        Gp035  = 0.0
        dGp035 = (gₓ(pend0,dpend0,0.0)')*s0 - (λ(0.0)')*( Fx(pend0,dpend0)*s0 + Fdx(pend0,dpend0)*ds0 + Fp(pend0,dpend0) )
        partial    = 0.0
        dpartial   = (λ(0.0)')*Fdx(pend0,dpend0)*ds0 + ( (dλ(0.0)')*Fdx(pend0,dpend0) + (λ(0.0)')*Fddx(pend0,dpend0) )*s0

        # tue_deb    = (λ(0.0)')*Fdx(pend0, dpend0)
        # dtue_deb   = (dλ(0.0)')*Fdx(pend0, dpend0) + (λ(0.0)')*Fddx(pend0, dpend0)
        # NOTE: ONLY MAKE SENSE FOR k AND m SINCE ONLY THEN APPROPRIATE QUANTITIES ARE ZERO
        common   = 0.0
        dcommon  = (gₓ(pend0,dpend0,0.0)')*s0 - (λ(0.)')*( Fx(pend0,dpend0)*s0 + Fp(pend0,dpend0) )
        extra35  = 0.0
        dextra35 = -(λ(0.)')*Fdx(pend0,dpend0)*ds0
        extra4   = 0.0
        dextra4  = ( (dλ(0.0)')*Fdx(pend0,dpend0) + (λ(0.0)')*Fddx(pend0,dpend0) )*s0

        partrepL  = 0.0
        dpartrepL = (λ(0.)')*Fdx(pend0,dpend0)*ds0
        partrepR  = 0.0
        dpartrepR = -((dλ(0.)')*Fdx(pend0,dpend0) + (λ(0.)')*Fddx(pend0,dpend0) )*s0
        oscdeb = 0.0
        doscdeb = (gₓ(pend0,dpend0,0.0)')*s0 - (λ(0.0)')*(Fx(pend0,dpend0)*s0 + Fp(pend0, dpend0)) + (λ(0.0)')*Fddx(pend0,dpend0)*s0 + (dλ(0.0)')*Fdx(pend0,dpend0)*s0
        res29  = 0.0
        dres29 = (dλ(0.0)')*Fdx(pend0,dpend0)*s0
        res30  = 0.0
        dres30 = (λ(0.0)')*Fddx(pend0,dpend0)*s0

        x0   = vcat(pend0, s0, G0, Gp0, Gp02, Gp03, Gp04, Gp04b, Gp035, partial, common, extra35, extra4, partrepL, partrepR, oscdeb, res29, res30)
        dx0  = vcat(dpend0, ds0, dG0, dGp0, dGp02, dGp03, dGp04, dGp04b, dGp035, dpartial, dcommon, dextra35, dextra4, dpartrepL, dpartrepR, doscdeb, dres29, dres30)

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false, true, true, true, true, true, true, true, true], fill(true, 8))
        # dvars = vcat(fill(true, 6), [false], fill(true, 6), [false, true, true, true, true, true, true, true, true], fill(true, 7))

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)
        # @info "r0 is $r0"

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_adj_stepbystep_deb(Φ::Float64, u::Function, w::Function, θ::Vector{Float64}, y::Function, λ::Function, dλ::Function, T::Float64)::Model
    @warn "pendulum_adj_stepbystep_deb only adapted for pi-parameter currently, no others"
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        my_ind = 2
        i = my_ind
        # @warn "pendulum_adj_stepbystep_deb. m: $m, L: $L, g: $g, k: $k"


        # x  = t -> sol(t)
        Fx = (x, dx) -> [2dx[6]        0               0   -1                  0           0   0
                          0           2*dx[6]          0   0                   -1          0   0
                         -dx[3]        0               0   2k*abs(x[4])        0           0   0
                          0          -dx[3]            0   0               2k*abs(x[5])    0   0
                          2x[1]       2x[2]            0   0                   0           0   0
                          x[4]        x[5]             0   x[1]               x[2]         0   0
                          x[2]/(L^2)  -x[1]/(L^2)      0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = (x, dx) -> vcat([1   0   0          0   0   2x[1]    0
                               0   1   0          0   0   2x[2]    0
                               0   0   -x[1]   m   0   0           0
                               0   0   -x[2]   0   m   0           0], zeros(3,7))
        Fddx = (x, dx) -> vcat([  0   0  0            0   0   2dx[1]    0
                                  0   0  0            0   0   2dx[2]    0
                                  0   0  -dx[1]    0   0   0            0
                                  0   0  -dx[2]    0   0   0            0], zeros(3,7))
        Fp = (x, dx) -> [ Int(i==1)
                          Int(i==2)
                          Int(i==3)
                          Int(i==4)
                          Int(i==5)
                          Int(i==6)
                          Int(i==7)]
       gₓ = (x, dx, t) -> [0.; 0.; 0.; 0.; 0.; 0.; 2(x[7]-y(t))/T]
       # λ = [1.; 1.; 1.; 1.; 1.; 1.; 1.]
       pvec = zeros(7); pvec[i] = 0.
       u0 = u(0.0)[1]
       w0 = w(0.0)[1]
       L_new = sqrt(L^2 - pvec[5])
       x1_0 = L_new * sin(Φ)
       x2_0 = -L_new * cos(Φ)
       x5_0 = -pvec[6]/x2_0
       pend0 = vcat([x1_0, x2_0], zeros(2), [x5_0, 0.0, atan(x1_0 / -x2_0)-pvec[7]])
       dx2_0 = x5_0 - pvec[2]
       dx4_0 = (u0 + w0^2 - pvec[3])/m
       dx5_0 = (-k*abs(x5_0)*x5_0 - m*g - pvec[4])/m
       dpend0 = [-pvec[1], dx2_0, 0.0, dx4_0, dx5_0, 0.0, 0.0]

       s1_0 = 0.0#equivalent -Int(i==5)*x1_0/(2*L^2)
       s2_0 = -Int(i==5)/(2x2_0)  # Very different, is it right?
       s5_0 = -Int(i==6)/(x2_0)
       s0  = [s1_0, s2_0, 0.0, 0.0, s5_0, 0.0, -Int(i==7)]
       ds1_0 = -Int(i==1)
       ds2_0 = -Int(i==2) + s5_0
       ds4_0 = -Int(i==3)/m#Almost equivalent : (Int(i==3)+dx3_0*s1_0)/m
       # ds5_0 = (-Int(i==4)+dx3_0*s2_0)/m
       ds5_0 = (-Int(i==4)+0.0*s2_0)/m
       ds7_0 = Int(i==1)*x2_0/(L^2)#Not even close: (x1_0*(Int(i==1)+s4_0) - x2_0*(Int(i==2)+s5_0))/(L^2)
       ds0 = [ds1_0, ds2_0, 0.0, ds4_0, ds5_0, 0.0, ds7_0]

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - x[4] + 2dx[6]*x[1]
            res[2] = dx[2] - x[5] + 2dx[6]*x[2]
            res[3] = m*dx[4] - dx[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
            res[4] = m*dx[5] - dx[3]*x[2] + k*abs(x[5])*x[5] + m*g
            res[5] = x[1]^2 + x[2]^2 - L^2
            res[6] = x[4]*x[1] + x[5]*x[2]
            # Angle of pendulum
            res[7] = x[7] - atan(x[1] / -x[2])
            # Sensitivity equations for p1                      # here
            res[8]  = dx[8] - x[11] + 2x[8]dx[6] + 2dx[13]x[1]  + Int(i==1)
            res[9]  = dx[9] - x[12] + 2x[9]dx[6] + 2dx[13]x[2] + Int(i==2)
            res[10] = 2k*x[11]*abs(x[4]) + m*dx[11] - x[8]dx[3] - dx[10]x[1] + Int(i==3)
            res[11] = 2k*x[12]*abs(x[5]) + m*dx[12] - x[9]dx[3] - dx[10]x[2] + Int(i==4)
            res[12] = 2x[8]x[1] + 2x[9]x[2] + Int(i==5)
            res[13] = x[8]x[4] + x[11]x[1] + x[9]x[5] + x[12]x[2] + Int(i==6)
            # Sensitivity of angle of pendulum to L
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[8]*x[2])/(L^2) + Int(i==7)

            # res[15] = dx[15] - (y(t) - y(t;θ))^2 (point 1)
            res[15] = dx[15] - (y(t)-x[7])^2
            # res[16] = dx[16] - 2(y(t;θ)-y(t))*y_θ(t) (point 2)
            res[16] = dx[16] - 2(x[7]-y(t))*x[14]

            # # Comment out and just use explicilty
            # xθ  = x[8:14]
            # dxθ = dx[8:14]
            # res[17] = dx[17] - 2(y(t;θ)-y(t))*y_θ(t) - (λ')*(Fx(t)*xθ + Fdx(t)*dxθ + Fp(t)) (point 3)
            res[17] = dx[17] - 2(x[7]-y(t))*x[14] + (λ(t)')*(Fx(x, dx)*x[8:14] + Fdx(x, dx)*dx[8:14] + Fp(x, dx))

            # res[18] = dx[18] + (λ')*Fp(x, dx) + ( (λ')*(Fx(x,dx)-Fddx(x,dx)) - (dλ')*Fdx(x,dx) - gx(x,dx,t)' )*xθ (point 4)
            res[18] = dx[18] + (λ(t)')*Fp(x, dx) + ( (λ(t)')*(Fx(x,dx)-Fddx(x,dx)) - (dλ(t)')*Fdx(x,dx) - gₓ(x,dx,t)' )*x[8:14]

            # res[19] = dx[19] + (λ')*Fp(x, dx) (point 5)
            res[19] = dx[19] + (λ(t)')*Fp(x, dx)
            # res[20] = dx[20]  + ( (λ')*(Fx(x,dx)-Fddx(x,dx)) - (dλ')*Fdx(x,dx) - gx(x,dx,t)' )*xθ
            res[20] = dx[20] + ( (λ(t)')*(Fx(x,dx)-Fddx(x,dx)) - (dλ(t)')*Fdx(x,dx) - gₓ(x,dx,t)' )*x[8:14]

            # res[21] = dx[21] - (gx(x,dx,t)')*xθ + (λ(t)')*( Fx(x,dx)*xθ + Fdx(x,dx)*dxθ + Fp(x,dx
            res[21] = dx[21] - (gₓ(x,dx,t)')*x[8:14] + (λ(t)')*( Fx(x,dx)*x[8:14] + Fdx(x,dx)*dx[8:14] + Fp(x,dx) )

            # res[22] = dx[22] - (λ(t)')*Fdx(t)*dxθ - ( (dλ(t)')*Fdx(t) + (λ(t)')*Fddx(t) )*xθ
            res[22] = dx[22] - (λ(t)')*Fdx(x,dx)*dx[8:14] - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # Shared between 3.5 and 4
            res[23] = dx[23] - (gₓ(x,dx,t)')*x[8:14] + (λ(t)')*( Fx(x,dx)*x[8:14] + Fp(x,dx) )
            # Extra for 3.5 (for m and k only)
            res[24] = dx[24] + (λ(t)')*Fdx(x,dx)*dx[8:14]
            # Extra for 4 (for m and k only)
            res[25] = dx[25] - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # NOTE: LoL, these are just negated versions of the above, clearly not very equal!
            # For replacement using partial integration
            # LHS
            res[26] = dx[26] - (λ(t)')*Fdx(x,dx)*dx[8:14]
            # RHS (without term, assumed zero, which is the case for m and correct λ)
            res[27] = dx[27] + ((dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # (point 3.9) (rearranged point 4 for easier comparison with point 3.5)
            res[28] = dx[28] - (gₓ(x,dx,t)')*x[8:14] + (λ(t)')*(Fx(x,dx)*x[8:14] + Fp(x, dx)) - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # REALLY SMAL STUFF; JUST DELETE
            res[29] = dx[29] - (dλ(t)')*Fdx(x,dx)*x[8:14]
            res[30] = dx[30] - (λ(t)')*Fddx(x,dx)*x[8:14]

            nothing
        end

        # # ORIGINAL
        # # Finding consistent initial conditions
        # # Initial values, the pendulum starts at rest
        # u0 = u(0.0)[1]
        # w0 = w(0.0)[1]
        # x1_0 = L * sin(Φ)
        # x2_0 = -L * cos(Φ)
        # dx3_0 = m*g/x2_0
        # dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m
        # FOR DEB
        pvec = zeros(7); pvec[i] = 0.
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        L_new = sqrt(L^2 - pvec[5])
        x1_0 = L_new * sin(Φ)
        x2_0 = -L_new * cos(Φ)
        x5_0 = -pvec[6]/x2_0
        pend0 = vcat([x1_0, x2_0], zeros(2), [x5_0, 0.0, atan(x1_0 / -x2_0)-pvec[7]])
        dx2_0 = x5_0 - pvec[2]
        dx4_0 = (u0 + w0^2 - pvec[3])/m
        dx5_0 = (-k*abs(x5_0)*x5_0 - m*g - pvec[4])/m
        dpend0 = [-pvec[1], dx2_0, 0.0, dx4_0, dx5_0, 0.0, 0.0]

        s1_0 = 0.0#equivalent -Int(i==5)*x1_0/(2*L^2)
        s2_0 = -Int(i==5)/(2x2_0)  # Very different, is it right?
        s5_0 = -Int(i==6)/(x2_0)
        s0  = [s1_0, s2_0, 0.0, 0.0, s5_0, 0.0, -Int(i==7)]
        ds1_0 = -Int(i==1)
        ds2_0 = -Int(i==2) + s5_0
        ds4_0 = -Int(i==3)/m#Almost equivalent : (Int(i==3)+dx3_0*s1_0)/m
        # ds5_0 = (-Int(i==4)+dx3_0*s2_0)/m
        ds5_0 = (-Int(i==4)+0.0*s2_0)/m
        ds7_0 = Int(i==1)*x2_0/(L^2)#Not even close: (x1_0*(Int(i==1)+s4_0) - x2_0*(Int(i==2)+s5_0))/(L^2)
        ds0 = [ds1_0, ds2_0, 0.0, ds4_0, ds5_0, 0.0, ds7_0]

        # # ORIGINAL, for deb above
        # pend0  = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        # dpend0 = [0., 0., dx3_0, dx4_0, 0., 0., 0.]
        # s0  = zeros(7)
        # ds0 = zeros(7)
        G0    = 0.0
        dG0   = (y(0.0)-pend0[7])^2
        Gp0   = 0.0
        dGp0  = 2(pend0[7]-y(0.0))*0.0
        Gp02  = 0.0
        dGp02 = dGp0 - (λ(0.)')*(Fx(pend0, dpend0)*s0 + Fdx(pend0, dpend0)*ds0 + Fp(pend0, dpend0))
        Gp03  = 0.0
        dGp03 = -(λ(0.)')*Fp(pend0, dpend0) - ( (λ(0.)')*(Fx(pend0,dpend0)-Fddx(pend0,dpend0)) -(dλ(0.)')*Fdx(pend0,dpend0) - gₓ(pend0,dpend0,0.0)' )*s0
        Gp04  = 0.0
        dGp04 = -(λ(0.)')*Fp(pend0, dpend0)
        Gp04b  = 0.0
        dGp04b = - ( (λ(0.)')*(Fx(pend0,dpend0)-Fddx(pend0,dpend0)) -(dλ(0.)')*Fdx(pend0,dpend0) - gₓ(pend0,dpend0,0.0)' )*s0
        Gp035  = 0.0
        dGp035 = (gₓ(pend0,dpend0,0.0)')*s0 - (λ(0.0)')*( Fx(pend0,dpend0)*s0 + Fdx(pend0,dpend0)*ds0 + Fp(pend0,dpend0) )
        partial    = 0.0
        dpartial   = (λ(0.0)')*Fdx(pend0,dpend0)*ds0 + ( (dλ(0.0)')*Fdx(pend0,dpend0) + (λ(0.0)')*Fddx(pend0,dpend0) )*s0

        # tue_deb    = (λ(0.0)')*Fdx(pend0, dpend0)
        # dtue_deb   = (dλ(0.0)')*Fdx(pend0, dpend0) + (λ(0.0)')*Fddx(pend0, dpend0)
        # NOTE: ONLY MAKE SENSE FOR k AND m SINCE ONLY THEN APPROPRIATE QUANTITIES ARE ZERO
        common   = 0.0
        dcommon  = (gₓ(pend0,dpend0,0.0)')*s0 - (λ(0.)')*( Fx(pend0,dpend0)*s0 + Fp(pend0,dpend0) )
        extra35  = 0.0
        dextra35 = -(λ(0.)')*Fdx(pend0,dpend0)*ds0
        extra4   = 0.0
        dextra4  = ( (dλ(0.0)')*Fdx(pend0,dpend0) + (λ(0.0)')*Fddx(pend0,dpend0) )*s0

        partrepL  = 0.0
        dpartrepL = (λ(0.)')*Fdx(pend0,dpend0)*ds0
        partrepR  = 0.0
        dpartrepR = -((dλ(0.)')*Fdx(pend0,dpend0) + (λ(0.)')*Fddx(pend0,dpend0) )*s0
        oscdeb = 0.0
        doscdeb = (gₓ(pend0,dpend0,0.0)')*s0 - (λ(0.0)')*(Fx(pend0,dpend0)*s0 + Fp(pend0, dpend0)) + (λ(0.0)')*Fddx(pend0,dpend0)*s0 + (dλ(0.0)')*Fdx(pend0,dpend0)*s0
        res29  = 0.0
        dres29 = (dλ(0.0)')*Fdx(pend0,dpend0)*s0
        res30  = 0.0
        dres30 = (λ(0.0)')*Fddx(pend0,dpend0)*s0

        x0   = vcat(pend0, s0, G0, Gp0, Gp02, Gp03, Gp04, Gp04b, Gp035, partial, common, extra35, extra4, partrepL, partrepR, oscdeb, res29, res30)
        dx0  = vcat(dpend0, ds0, dG0, dGp0, dGp02, dGp03, dGp04, dGp04b, dGp035, dpartial, dcommon, dextra35, dextra4, dpartrepL, dpartrepR, doscdeb, dres29, dres30)

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false, true, true, true, true, true, true, true, true], fill(true, 8))
        # dvars = vcat(fill(true, 6), [false], fill(true, 6), [false, true, true, true, true, true, true, true, true], fill(true, 7))

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)
        # @info "r0 is $r0"

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function deleteme_stepbystep_deb(Φ::Float64, u::Function, w::Function, θ::Vector{Float64}, y::Function, λ::Function, dλ::Function, T::Float64)::Model
    @warn "pendulum_adj_stepbystep_deb only adapted for pi-parameter currently, no others"
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        my_ind = 2
        i = my_ind
        # @warn "pendulum_adj_stepbystep_deb. m: $m, L: $L, g: $g, k: $k"


        # x  = t -> sol(t)
        Fx = (x, dx) -> [2dx[6]        0               0   -1                  0           0   0
                          0           2*dx[6]          0   0                   -1          0   0
                         -dx[3]        0               0   2k*abs(x[4])        0           0   0
                          0          -dx[3]            0   0               2k*abs(x[5])    0   0
                          2x[1]       2x[2]            0   0                   0           0   0
                          x[4]        x[5]             0   x[1]               x[2]         0   0
                          x[2]/(L^2)  -x[1]/(L^2)      0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = (x, dx) -> vcat([1   0   0          0   0   2x[1]    0
                               0   1   0          0   0   2x[2]    0
                               0   0   -x[1]   m   0   0           0
                               0   0   -x[2]   0   m   0           0], zeros(3,7))
        Fddx = (x, dx) -> vcat([  0   0  0            0   0   2dx[1]    0
                                  0   0  0            0   0   2dx[2]    0
                                  0   0  -dx[1]    0   0   0            0
                                  0   0  -dx[2]    0   0   0            0], zeros(3,7))
        Fp = (x, dx) -> [ Int(i==1)
                          Int(i==2)
                          Int(i==3)
                          Int(i==4)
                          Int(i==5)
                          Int(i==6)
                          Int(i==7)]
       gₓ = (x, dx, t) -> [0.; 0.; 0.; 0.; 0.; 0.; 2(x[7]-y(t))/T]
       # λ = [1.; 1.; 1.; 1.; 1.; 1.; 1.]
       pvec = zeros(7); pvec[i] = 0.
       u0 = u(0.0)[1]
       w0 = w(0.0)[1]
       L_new = sqrt(L^2 - pvec[5])
       x1_0 = L_new * sin(Φ)
       x2_0 = -L_new * cos(Φ)
       x5_0 = -pvec[6]/x2_0
       pend0 = vcat([x1_0, x2_0], zeros(2), [x5_0, 0.0, atan(x1_0 / -x2_0)-pvec[7]])
       dx2_0 = x5_0 - pvec[2]
       dx4_0 = (u0 + w0^2 - pvec[3])/m
       dx5_0 = (-k*abs(x5_0)*x5_0 - m*g - pvec[4])/m
       dpend0 = [-pvec[1], dx2_0, 0.0, dx4_0, dx5_0, 0.0, 0.0]

       s1_0 = 0.0#equivalent -Int(i==5)*x1_0/(2*L^2)
       s2_0 = -Int(i==5)/(2x2_0)  # Very different, is it right?
       s5_0 = -Int(i==6)/(x2_0)
       s0  = [s1_0, s2_0, 0.0, 0.0, s5_0, 0.0, -Int(i==7)]
       ds1_0 = -Int(i==1)
       ds2_0 = -Int(i==2) + s5_0
       ds4_0 = -Int(i==3)/m#Almost equivalent : (Int(i==3)+dx3_0*s1_0)/m
       # ds5_0 = (-Int(i==4)+dx3_0*s2_0)/m
       ds5_0 = (-Int(i==4)+0.0*s2_0)/m
       ds7_0 = Int(i==1)*x2_0/(L^2)#Not even close: (x1_0*(Int(i==1)+s4_0) - x2_0*(Int(i==2)+s5_0))/(L^2)
       ds0 = [ds1_0, ds2_0, 0.0, ds4_0, ds5_0, 0.0, ds7_0]

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - x[4] + 2dx[6]*x[1]
            res[2] = dx[2] - x[5] + 2dx[6]*x[2]
            res[3] = m*dx[4] - dx[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
            res[4] = m*dx[5] - dx[3]*x[2] + k*abs(x[5])*x[5] + m*g
            res[5] = x[1]^2 + x[2]^2 - L^2
            res[6] = x[4]*x[1] + x[5]*x[2]
            # Angle of pendulum
            res[7] = x[7] - atan(x[1] / -x[2])
            # Sensitivity equations for p1                      # here
            res[8]  = dx[8] - x[11] + 2x[8]dx[6] + 2dx[13]x[1]  + Int(i==1)
            res[9]  = dx[9] - x[12] + 2x[9]dx[6] + 2dx[13]x[2] + Int(i==2)
            res[10] = 2k*x[11]*abs(x[4]) + m*dx[11] - x[8]dx[3] - dx[10]x[1] + Int(i==3)
            res[11] = 2k*x[12]*abs(x[5]) + m*dx[12] - x[9]dx[3] - dx[10]x[2] + Int(i==4)
            res[12] = 2x[8]x[1] + 2x[9]x[2] + Int(i==5)
            res[13] = x[8]x[4] + x[11]x[1] + x[9]x[5] + x[12]x[2] + Int(i==6)
            # Sensitivity of angle of pendulum to L
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[8]*x[2])/(L^2) + Int(i==7)

            # res[15] = dx[15] - (y(t) - y(t;θ))^2 (point 1)
            res[15] = dx[15] - (y(t)-x[7])^2
            # res[16] = dx[16] - 2(y(t;θ)-y(t))*y_θ(t) (point 2)
            res[16] = dx[16] - 2(x[7]-y(t))*x[14]
            nothing
        end

        # # ORIGINAL
        # # Finding consistent initial conditions
        # # Initial values, the pendulum starts at rest
        # u0 = u(0.0)[1]
        # w0 = w(0.0)[1]
        # x1_0 = L * sin(Φ)
        # x2_0 = -L * cos(Φ)
        # dx3_0 = m*g/x2_0
        # dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m
        # FOR DEB
        pvec = zeros(7); pvec[i] = 0.
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        L_new = sqrt(L^2 - pvec[5])
        x1_0 = L_new * sin(Φ)
        x2_0 = -L_new * cos(Φ)
        x5_0 = -pvec[6]/x2_0
        pend0 = vcat([x1_0, x2_0], zeros(2), [x5_0, 0.0, atan(x1_0 / -x2_0)-pvec[7]])
        dx2_0 = x5_0 - pvec[2]
        dx4_0 = (u0 + w0^2 - pvec[3])/m
        dx5_0 = (-k*abs(x5_0)*x5_0 - m*g - pvec[4])/m
        dpend0 = [-pvec[1], dx2_0, 0.0, dx4_0, dx5_0, 0.0, 0.0]

        s1_0 = 0.0#equivalent -Int(i==5)*x1_0/(2*L^2)
        s2_0 = -Int(i==5)/(2x2_0)  # Very different, is it right?
        s5_0 = -Int(i==6)/(x2_0)
        s0  = [s1_0, s2_0, 0.0, 0.0, s5_0, 0.0, -Int(i==7)]
        ds1_0 = -Int(i==1)
        ds2_0 = -Int(i==2) + s5_0
        ds4_0 = -Int(i==3)/m#Almost equivalent : (Int(i==3)+dx3_0*s1_0)/m
        # ds5_0 = (-Int(i==4)+dx3_0*s2_0)/m
        ds5_0 = (-Int(i==4)+0.0*s2_0)/m
        ds7_0 = Int(i==1)*x2_0/(L^2)#Not even close: (x1_0*(Int(i==1)+s4_0) - x2_0*(Int(i==2)+s5_0))/(L^2)
        ds0 = [ds1_0, ds2_0, 0.0, ds4_0, ds5_0, 0.0, ds7_0]

        # # ORIGINAL, for deb above
        # pend0  = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        # dpend0 = [0., 0., dx3_0, dx4_0, 0., 0., 0.]
        # s0  = zeros(7)
        # ds0 = zeros(7)
        G0    = 0.0
        dG0   = (y(0.0)-pend0[7])^2
        Gp0   = 0.0
        dGp0  = 2(pend0[7]-y(0.0))*0.0

        x0   = vcat(pend0, s0, G0, Gp0)
        dx0  = vcat(dpend0, ds0, dG0, dGp0)

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false, true, true])
        # dvars = vcat(fill(true, 6), [false], fill(true, 6), [false, true, true, true, true, true, true, true, true], fill(true, 7))

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)
        # @info "r0 is $r0"

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_adj_stepbystep_m_alt(Φ::Float64, u::Function, w::Function, θ::Vector{Float64}, y::Function, λ::Function, dλ::Function, x::Function, dx::Function, T::Float64)::Model
    @warn "pendulum_adj_stepbystep_m only adapted for m-parameter currently, no others"
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # x  = t -> sol(t)
        Fx = t -> [2dx(t)[6]        0                0   -1                   0           0   0
                      0           2*dx(t)[6]         0   0                   -1           0   0
                     -dx(t)[3]        0              0   2k*abs(x(t)[4])      0           0   0
                      0          -dx(t)[3]           0   0               2k*abs(x(t)[5])  0   0
                      2x(t)[1]       2x(t)[2]        0   0                   0            0   0
                      x(t)[4]        x(t)[5]         0   x(t)[1]          x(t)[2]         0   0
                      x(t)[2]/(L^2)  -x(t)[1]/(L^2)  0   0                   0            0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = t -> vcat([1   0   0          0   0   2x(t)[1]    0
                         0   1   0          0   0   2x(t)[2]    0
                         0   0   -x(t)[1]   m   0   0           0
                         0   0   -x(t)[2]   0   m   0           0], zeros(3,7))
        Fddx = t -> vcat([  0   0  0            0   0   2dx(t)[1]    0
                            0   0  0            0   0   2dx(t)[2]    0
                            0   0  -dx(t)[1]    0   0   0            0
                            0   0  -dx(t)[2]    0   0   0            0], zeros(3,7))
        Fp = t -> [ .0
                  .0
                  dx(t)[4]
                  dx(t)[5]+g
                  .0
                  .0
                  .0]
       gₓ = t -> [0.; 0.; 0.; 0.; 0.; 0.; 2(x(t)[7]-y(t))/T]
       # λ = [1.; 1.; 1.; 1.; 1.; 1.; 1.]

       #TODO: DON'T PASS T. BUT ALSO DON'T CALL THIS gₓ!!!! EDIT: Why...?

        # the residual function
        function f!(res, dz, z, θ, t)
            wt = w(t)
            ut = u(t)

            # res[15] = dx[15] - (y(t) - y(t;θ))^2 (point 1)
            res[1] = dz[1] - (y(t)-x(t)[7])^2
            # res[2] = dx[2] - 2(y(t;θ)-y(t))*y_θ(t) (point 2)
            res[2] = dz[2] - 2(x(t)[7]-y(t))*x(t)[14]

            # # Comment out and just use explicilty
            # xθ  = x[8:14]
            # dxθ = dx[8:14]
            # res[3] = dx[3] - 2(y(t;θ)-y(t))*y_θ(t) - (λ')*(Fx(t)*xθ + Fdx(t)*dxθ + Fp(t)) (point 3)
            res[3] = dz[3] - 2(x(t)[7]-y(t))*x(t)[14] + (λ(t)')*(Fx(t)*x(t)[8:14] + Fdx(x, dx)*dx(t)[8:14] + Fp(t))

            # res[4] = dx[4] + (λ')*Fp(x, dx) + ( (λ')*(Fx(t)-Fddx(t)) - (dλ')*Fdx(t) - gx(x,dx,t)' )*xθ (point 4)
            res[4] = dz[4] + (λ(t)')*Fp(t) + ( (λ(t)')*(Fx(t)-Fddx(t)) - (dλ(t)')*Fdx(t) - gₓ(t)' )*x(t)[8:14]

            # res[5] = dx[5] + (λ')*Fp(x, dx) (point 5)
            res[5] = dz[5] + (λ(t)')*Fp(t)
            # res[6] = dx[6]  + ( (λ')*(Fx(t)-Fddx(t)) - (dλ')*Fdx(t) - gx(x,dx,t)' )*xθ
            res[6] = dz[6] + ( (λ(t)')*(Fx(t)-Fddx(t)) - (dλ(t)')*Fdx(t) - gₓ(t)' )*x(t)[8:14] # (bonus point (remainder?))

            # res[7] = dx[7] - (gx(x,dx,t)')*xθ + (λ(t)')*( Fx(t)*xθ + Fdx(t)*dxθ + Fp(x,dx
            res[7] = dz[7] - (gₓ(t)')*x(t)[8:14] + (λ(t)')*( Fx(t)*x[8:14] + Fdx(t)*dx(t)[8:14] + Fp(t) ) # (point 3.5)

            # res[8] = dx[8] - (λ(t)')*Fdx(t)*dxθ - ( (dλ(t)')*Fdx(t) + (λ(t)')*Fddx(t) )*xθ
            res[8] = dz[8] - (λ(t)')*Fdx(t)*dx(t)[8:14] - ( (dλ(t)')*Fdx(t) + (λ(t)')*Fddx(t) )*x(t)[8:14]     # (point 6)

            # res[23:29] = dx[23:29] - ( (dλ(t)')*Fdx(t) + (λ(t)')*Fddx(t) )[:]

            # Shared between 3.5 and 4
            res[9] = dz[9] - (gₓ(t)')*x(t)[8:14] + (λ(t)')*( Fx(t)*x(t)[8:14] + Fp(t) )
            # Extra for 3.5 (for m and k only)
            res[10] = dz[10] + (λ(t)')*Fdx(t)*dx(t)[8:14]
            # Extra for 4 (for m and k only)
            res[11] = dz[11] - ( (dλ(t)')*Fdx(t) + (λ(t)')*Fddx(t) )*x(t)[8:14]

            # For replacement using partial integration
            # LHS
            res[12] = dz[12] - (λ(t)')*Fdx(t)*dx(t)[8:14]
            # RHS (without term, assumed zero, which is the case for m and correct λ)
            res[13] = dz[13] + ((dλ(t)')*Fdx(t) + (λ(t)')*Fddx(t) )*x(t)[8:14]

            # This is the equations for my_pendulum_adjoint_konly
            # res[1:7] = (-dz[1:7]')*Fdx(T-t) + (z[1:7]')*(Fddx(T-t) - Fx(T-t)) + gₓ(T-t)
            # <=> (z[1:7]')*(Fddx(T-t) - Fx(T-t)) - (dz[1:7]')*Fdx(T-t) + gₓ(T-t)
            # <=> (z[1:7]')*(Fx(T-t) - Fddx(T-t)) + (dz[1:7]')*Fdx(T-t) - gₓ(T-t)
            # THERE IS A SIGN INCONSISTENCY!!!!!!!!!!!!!!!!
            # Yes becuase that one is solved backwards you silly...

            # These equations sensitivity are written on the intuitive form. To
            # obtain form correct for this funciton, do the following replacements:
            # s[1] -> x[8],  s[2] -> x[9],  s[3] -> x[10]
            # s[4] -> x[11], s[5] -> x[12], s[6] -> x[13]
            # and the corresponding replacements for sp and dx

            # res[8]  = -s[4] + sp[1] + 2x[1]*sp[6] + 2s[1]*dx[6]
            # res[9]  = -s[5] + sp[2] + 2x[2]*sp[6] + 2s[2]*dx[6]
            # res[10]  = abs(x[4])*x[4] + 2k*s[4]*abs(x[4]) - x[1]*sp[3] + m*sp[4] - s[1]*dx[3]
            # res[11] = abs(x[5])*x[5] + 2k*s[5]*abs(x[5]) - x[2]*sp[3] + m*sp[5] - s[2]*dx[3]
            # res[12] = -2s[1]*x[1] - 2s[2]*x[2]
            # res[13] = s[4]*x[1] + s[5]*x[2] + s[1]*x[4] + s[2]*x[5]

            nothing
        end

        # Finding consistent initial conditions
        # Initial values, the pendulum starts at rest
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        x1_0 = L * sin(Φ)
        x2_0 = -L * cos(Φ)
        dx3_0 = m*g/x2_0
        dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m

        pend0  = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dpend0 = [0., 0., dx3_0, dx4_0, 0., 0., 0.]
        sm0  = zeros(7)
        dsm0 = vcat(zeros(3), [-dx4_0/m, -g/m, 0., 0.])
        G0    = 0.0
        dG0   = (y(0.0)-pend0[7])^2
        Gp0   = 0.0
        dGp0  = 2(pend0[7]-y(0.0))*sm0[7]
        Gp02  = 0.0
        dGp02 = dGp0 - (λ(0.)')*(Fx(0.0)*sm0 + Fdx(0.0)*dsm0 + Fp(0.0))
        Gp03  = 0.0
        dGp03 = -(λ(0.)')*Fp(0.0) - ( (λ(0.)')*(Fx(0.0)-Fddx(0.0)) -(dλ(0.)')*Fdx(0.0) - gₓ(0.0)' )*sm0
        Gp04  = 0.0
        dGp04 = -(λ(0.)')*Fp(0.0)
        Gp04b  = 0.0
        dGp04b = - ( (λ(0.)')*(Fx(0.0)-Fddx(0.0)) -(dλ(0.)')*Fdx(0.0) - gₓ(0.0)' )*sm0
        Gp035  = 0.0
        dGp035 = (gₓ(0.0)')*sm0 - (λ(0.0)')*( Fx(0.0)*sm0 + Fdx(0.0)*dsm0 + Fp(0.0) )
        partial    = 0.0
        dpartial   = (λ(0.0)')*Fdx(0.0)*dsm0 + ( (dλ(0.0)')*Fdx(0.0) + (λ(0.0)')*Fddx(0.0) )*sm0

        # tue_deb    = (λ(0.0)')*Fdx(0.0)
        # dtue_deb   = (dλ(0.0)')*Fdx(0.0) + (λ(0.0)')*Fddx(0.0)
        # NOTE: ONLY MAKE SENSE FOR k AND m SINCE ONLY THEN APPROPRIATE QUANTITIES ARE ZERO
        common   = 0.0
        dcommon  = (gₓ(0.0)')*sm0 - (λ(0.)')*( Fx(0.0)*sm0 + Fp(0.0) )
        extra35  = 0.0
        dextra35 = -(λ(0.)')*Fdx(0.0)*dsm0
        extra4   = 0.0
        dextra4  = ( (dλ(0.0)')*Fdx(0.0) + (λ(0.0)')*Fddx(0.0) )*sm0

        partrepL  = 0.0
        dpartrepL = (λ(0.)')*Fdx(0.0)*dsm0
        partrepR  = 0.0
        dpartrepR = -((dλ(0.)')*Fdx(0.0) + (λ(0.)')*Fddx(0.0) )*sm0

        x0   = vcat(G0, Gp0, Gp02, Gp03, Gp04, Gp04b, Gp035, partial, common, extra35, extra4, partrepL, partrepR)
        dx0  = vcat(dG0, dGp0, dGp02, dGp03, dGp04, dGp04b, dGp035, dpartial, dcommon, dextra35, dextra4, dpartrepL, dpartrepR)

        dvars = vcat([true, true, true, true, true, true, true, true], fill(true, 5))

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)
        # @info "r0 is $r0"

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function model_mohamed(Φ::Float64, u::Function, w::Function, θ::Vector{Float64})
    # NOTE: Φ not used, just passed to have consistent interface
    function f!(res, dx, x, p, t)
        wt = w(t)[1]
        ut = u(t)[1]
        # Dynamic Equations
        # @info "sum: $(ut+wt), θ: $θ, x1: $(x[1])"   # TODO: Sum is not correct here, gotta figure out why!
        res[1] = dx[1] + θ[1]*x[1] + ut + wt
        res[2] = x[2] + 2/( (θ[1]*x[1]+ut+wt)^2 + 1 )
        nothing
    end

    # Finding consistent initial conditions
    # Initial values, the pendulum starts at rest
    u0 = u(0.0)[1]
    w0 = w(0.0)[1]
    x10 = -(u0+w0)/θ[1]
    x0 = [x10, -2/( (θ[1]*x10+u0+w0)^2 + 1 )]
    dx0 = [0.0, 0.0]

    dvars = [true, false]

    r0 = zeros(length(x0))
    f!(r0, dx0, x0, [], 0.0)

    # t -> 0.0 is just a dummy function, not to be used
    Model(f!, t -> 0.0, x0, dx0, dvars, r0)
end

function mohamed_sens(Φ::Float64, u::Function, w::Function, p::Vector{Float64})
    θ = p[1]
    # NOTE: Φ not used, just passed to have consistent interface
    zeta(x, dx, t) = (θ*x[1] + u(t)[1] + w(t)[1])^2 + 1
    dzeta_dx1(x, dx, t) = 2θ*(θ*x[1]+u(t)[1]+w(t)[1])
    dzeta_dθ(x, dx, t)  = 2x[1]*(θ*x[1]+u(t)[1]+w(t)[1])
    # zeta(t) = (θ*x(t)[1] + u(t)[1] + w(t)[1])^2 + 1
    # dzeta_dx1(t) = 2θ*(θ*x(t)[1]+u(t)[1]+w(t)[1])
    # dzeta_dθ(t)  = 2x(t)[1]*(θ*x(t)[1]+u(t)[1]+w(t)[1])

    # @warn "This function doesn't generate a correct sensitivty model, only x0 is correct"
    function f!(res, dx, x, p, t)
        wt = w(t)[1]
        ut = u(t)[1]
        # Dynamic Equations
        res[1] = dx[1] + θ*x[1] + ut + wt
        # res[2] = x[2] + 2/( (θ*x[1]+ut+wt)^2 + 1 )
        res[2] = x[2] + 2/zeta(x, dx, t)
        # Sensitivity equations
        res[3] = dx[3] + θ*x[3] + x[1]
        res[4] = x[4] - 2*( dzeta_dx1(x,dx,t)*x[3]+dzeta_dθ(x,dx,t) )/((zeta(x,dx,t))^2)
        nothing
    end

    # Finding consistent initial conditions
    u0 = u(0.0)[1]
    w0 = w(0.0)[1]
    x10 = -(u0+w0)/θ
    xp10 = (u0+w0)/(θ^2)
    xp20 = 2*( 2*θ*(θ*x10+u0+w0)*xp10 + 2*x10*(θ*x10+u0+w0) )/(((θ*x10 + u0 + w0)^2+1)^2)
    x0 = [x10, -2/( (θ*x10+u0+w0)^2 + 1 ), xp10, xp20]
    dx0 = [0.0, 0.0, 0.0, 0.0]

    dvars = [true, false, true, false]

    r0 = zeros(length(x0))
    f!(r0, dx0, x0, [], 0.0)
    # @info "r0 is $r0"

    # t -> 0.0 is just a dummy function, not to be used
    Model(f!, t -> 0.0, x0, dx0, dvars, r0)
end

# Just writing it from scratch since a lot of time has passed and I've learned some stuff
function mohamed_adjoint_new(u::Function, w::Function, p::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Vector{Float64}, dx::Function, dx2::Function)
    θ = p[1]
    np = size(xp0, 2)
    nx = size(xp0,1)
    # x  = t -> sol(t)
    # x2 = t -> sol2(t)
    zeta(t) = (θ*x(t)[1] + u(t)[1] + w(t)[1])^2 + 1
    dzeta_dx1(t) = 2θ*(θ*x(t)[1]+u(t)[1]+w(t)[1])
    dzeta_dθ(t)  = 2x(t)[1]*(θ*x(t)[1]+u(t)[1]+w(t)[1])

    Fx(t)   = [θ 0.0; -(2/(zeta(t)^2))*dzeta_dx1(t) 1.0]
    Fdx(t)  = [1.0 0.0; 0.0 0.0]
    Fddx(t) = zeros(2,2)
    Fθ(t)   = [x(t)[1]; -(2/(zeta(t)^2))*dzeta_dθ(t)]
    # # FOR x2 AS OUTPUT
    # gₓ(t)   = [0  2(x2(t)[2]-y(t))/T]'
    # dgₓ(t)  = [0  2(dx2(t)[2]-dy(t))/T]'
    # FOR x1 AS OUTPUT
    gₓ(t)   = [2(x2(t)[1]-y(t))/T  0]'
    dgₓ(t)  = [2(dx2(t)[1]-dy(t))/T  0]'

    ###################### INITIALIZING ADJOINT SYSTEM ####################
    # Index 1 is differential (d), while index 2 is algebraic (a)
    λT  = zeros(2)
    dλT = zeros(2)
    λT[1] = 0.0
    # λT[2] = gₓ(T)/F_xa^a = gₓ(T)
    λT[2] = gₓ(T)[2]
    dλT[1] = -(λT')*[-θ; (2/(zeta(T)^2))*dzeta_dx1(T)] - gₓ(T)[1]
    dλT[2] = dgₓ(T)[2]
    #######################################################################

    # the residual function
    function f!(res, dz, z, θ, t)
        # Dynamic Equations
        # res[1] = (dλ')*Fdx(t) + (λ')*(Fddx(t)-Fx(t)) + gₓ(t)
        # res[2] = dβ + (λ')*Fθ
        res[1:2] = (dz[1:2]')*Fdx(t) + (z[1:2]')*(Fddx(t)-Fx(t)) + gₓ(t)'
        res[3] = dz[3] - (z[1:2]')*Fθ(t)
        # res[3] = dz[3] + (z[1:2]')*Fθ(t)    # WRONG
        nothing
    end

    function get_Gp(adj_sol::DAESolution)
        # Gp = adj_sol.u[end][end-np+1:end] + (((adj_sol.u[end][1:end-np]')*Fdx(0.0))*xp0)[:]
        Gp = adj_sol.u[end][nx+1:nx+np] .+ (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)
    end

    # λ0  = λT[:]
    # dλ0 = -dλT[:]
    β0  = 0.0
    dβ0 = -(λT[:]')*Fθ(T)
    # x0  = λ0
    # dx0 = dλ0
    z0  = vcat(λT[:],β0)
    dz0 = vcat(dλT[:], dβ0)

    @warn "AND IN HERE: $z0, $dz0"

    # dvars = [true, false]
    dvars = vcat([true, false], [true])

    # r0 = zeros(length(λ0))
    # f!(r0, dλ0, λ0, [], 0.0)
    r0 = zeros(length(z0))
    f!(r0, dz0, z0, [], 0.0)
    # @info "r0 is: $r0"

    # t -> 0.0 is just a dummy function, not to be used
    Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp
end

function mohamed_λ1(u::Function, w::Function, p::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Vector{Float64}, dx::Function, dx2::Function)
    θ = p[1]
    np = size(xp0, 2)
    nx = size(xp0,1)
    # x  = t -> sol(t)
    # x2 = t -> sol2(t)
    zeta(t) = (θ*x(t)[1] + u(t)[1] + w(t)[1])^2 + 1
    dzeta_dx1(t) = 2θ*(θ*x(t)[1]+u(t)[1]+w(t)[1])
    dzeta_dθ(t)  = 2x(t)[1]*(θ*x(t)[1]+u(t)[1]+w(t)[1])

    Fx(t)   = [θ 0.0; -(2/(zeta(t)^2))*dzeta_dx1(t) 1.0]
    Fdx(t)  = [1.0 0.0; 0.0 0.0]
    Fddx(t) = zeros(2,2)
    Fθ(t)   = [x(t)[1]; -(2/(zeta(t)^2))*dzeta_dθ(t)]
    # # FOR x2 AS OUTPUT
    # gₓ(t)   = [0  2(x2(t)[2]-y(t))/T]'
    # dgₓ(t)  = [0  2(dx2(t)[2]-dy(t))/T]'
    # FOR x1 AS OUTPUT
    gₓ(t)   = [2(x2(t)[1]-y(t))/T  0]'
    dgₓ(t)  = [2(dx2(t)[1]-dy(t))/T  0]'

    ###################### INITIALIZING ADJOINT SYSTEM ####################
    # Index 1 is differential (d), while index 2 is algebraic (a)
    λT  = zeros(2)
    dλT = zeros(2)
    λT[1] = 0.0
    # λT[2] = gₓ(T)/F_xa^a = gₓ(T)
    λT[2] = gₓ(T)[2]
    dλT[1] = -(λT')*[-θ; (2/(zeta(T)^2))*dzeta_dx1(T)] - gₓ(T)[1]
    dλT[2] = dgₓ(T)[2]
    #######################################################################

    # the residual function
    function f!(res, dz, z, p, t)
        res[1] = dz[1] - θ*z[1] + (2/T)*(x2(t)[1]-y(t))
        # res[1] = (dz[1:2]')*Fdx(t) + (z[1:2]')*(Fddx(t)-Fx(t)) + gₓ(t)'
        nothing
    end

    z0  = λT[1:1]#vcat(λT[:],β0)
    dz0 = dλT[1:1]#vcat(dλT[:], dβ0)

    @warn "INIT CONDS IN HERE: $z0, $dz0"

    # dvars = [true, false]
    dvars = [true]

    # r0 = zeros(length(λ0))
    # f!(r0, dλ0, λ0, [], 0.0)
    r0 = zeros(length(z0))
    f!(r0, dz0, z0, [], 0.0)
    # @info "r0 is: $r0"

    # t -> 0.0 is just a dummy function, not to be used
    Model(f!, t -> 0.0, z0, dz0, dvars, r0)
end

function mohamed_stepbystep(Φ::Float64, u::Function, w::Function, p::Vector{Float64}, y::Function, λ::Function, dλ::Function, T::Float64)
    θ = p[1]
    # np = size(xp0, 2)
    # nx = size(xp0,1)
    zeta(x,dx,t) = (θ*x[1] + u(t)[1] + w(t)[1])^2 + 1
    dzeta_dx1(x,dx,t) = 2θ*(θ*x[1]+u(t)[1]+w(t)[1])
    dzeta_dθ(x,dx,t)  = 2x[1]*(θ*x[1]+u(t)[1]+w(t)[1])

    Fx(x,dx,t)   = [θ 0.0; -(2/(zeta(x,dx,t)^2))*dzeta_dx1(x,dx,t) 1.0]
    Fdx(x,dx)  = [1.0 0.0; 0.0 0.0]
    Fddx(x,dx) = zeros(2,2)
    Fp(x,dx,t)   = [x[1]; -(2/(zeta(x,dx,t)^2))*dzeta_dθ(x,dx,t)]
    # FOR x2 AS OUTPUT
    # gₓ(x,dx,t)   = [0; 2(x[2]-y(t))/T]
    # dgₓ(x,dx,t)  = [0; 2(dx[2]-dy(t))/T]
    # FOR x1 AS OUTPUT
    gₓ(x,dx,t)   = [2(x[1]-y(t))/T; 0]
    dgₓ(x,dx,t)  = [2(dx[1]-dy(t))/T; 0]

    # the residual function
    function f!(res, dx, x, p, t)
        wt = w(t)[1]
        ut = u(t)[1]
        # NOTE: MAKE SURE TO ALSO CHANGE gₓ WHEN CHANGING THESE!
        z  = x[1]#x[2]
        zθ = x[3]#x[4]
        # Dynamic Equations
        res[1] = dx[1] + θ*x[1] + ut + wt
        res[2] = x[2] + 2/( (θ*x[1]+ut+wt)^2 + 1 )
        # Sensitivity equations
        res[3] = dx[3] + θ*x[3] + x[1]
        res[4] = x[4] - 2*( dzeta_dx1(x,dx,t)*x[3]+dzeta_dθ(x,dx,t) )/((zeta(x,dx,t))^2)

        # res[5] = dx[5] - (y(t) - y(t;θ))^2 (point 1)
        res[5] = dx[5] - (y(t)-z)^2
        # res[6] = dx[6] - 2(y(t;θ)-y(t))*y_θ(t) (point 2)
        res[6] = dx[6] - 2(z-y(t))*zθ

        # res[7] = dx[7] - 2(y(t;θ)-y(t))*y_θ(t) - (λ')*(Fx(t)*xθ + Fdx(t)*dxθ + Fp(t)) (point 3)
        res[7] = dx[7] - 2(z-y(t))*zθ + (λ(t)')*(Fx(x, dx, t)*x[3:4] + Fdx(x, dx)*dx[3:4] + Fp(x, dx, t))

        # res[8] = dx[8] + (λ')*Fp(x, dx) + ( (λ')*(Fx(x,dx)-Fddx(x,dx)) - (dλ')*Fdx(x,dx) - gx(x,dx,t)' )*xθ (point 4)
        res[8] = dx[8] + (λ(t)')*Fp(x, dx, t) + ( (λ(t)')*(Fx(x,dx,t)-Fddx(x,dx)) - (dλ(t)')*Fdx(x,dx) - gₓ(x,dx,t)' )*x[3:4]

        # res[9] = dx[9] + (λ')*Fp(x, dx) (point 5)
        res[9] = dx[9] + (λ(t)')*Fp(x, dx, t)       # NOTE: This one is the incorrect one!
        # (bonus point) i.e. remainder
        # res[10] = dx[10]  + ( (λ')*(Fx(x,dx)-Fddx(x,dx)) - (dλ')*Fdx(x,dx) - gx(x,dx,t)' )*xθ
        res[10] = dx[10] + ( (λ(t)')*(Fx(x,dx,t)-Fddx(x,dx)) - (dλ(t)')*Fdx(x,dx) - gₓ(x,dx,t)' )*x[3:4]

        # (point 3.5)
        # res[11] = dx[11] - (gx(x,dx,t)')*xθ + (λ(t)')*( Fx(x,dx)*xθ + Fdx(x,dx)*dxθ + Fp(x,dx) )
        res[11] = dx[11] - (gₓ(x,dx,t)')*x[3:4] + (λ(t)')*( Fx(x,dx,t)*x[3:4] + Fdx(x,dx)*dx[3:4] + Fp(x,dx,t) )

        # res[12] = dx[12] - (λ(t)')*Fdx(t)*dxθ - ( (dλ(t)')*Fdx(t) + (λ(t)')*Fddx(t) )*xθ
        res[12] = dx[12] - (λ(t)')*Fdx(x,dx)*dx[3:4] - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[3:4]

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # NOTE: PERHAPS NOT APPLICABLE FOR MOHMOD! !!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Shared between 3.5 and 4
        res[13] = dx[13] - (gₓ(x,dx,t)')*x[3:4] + (λ(t)')*( Fx(x,dx,t)*x[3:4] + Fp(x,dx,t) )
        # Extra for 3.5 (for m and k only)  # NOTE: PERHAPS NOT APPLICABLE FOR MOHMOD!
        res[14] = dx[14] + (λ(t)')*Fdx(x,dx)*dx[3:4]
        # Extra for 4 (for m and k only)    # NOTE: PERHAPS NOT APPLICABLE FOR MOHMOD!
        res[15] = dx[15] - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[3:4]

        # NOTE: LoL, these are just negated versions of the above, clearly not very equal!
        # For replacement using partial integration
        # LHS
        res[16] = dx[16] - (λ(t)')*Fdx(x,dx)*dx[3:4]
        # RHS (without term, assumed zero, which is the case for m and correct λ)
        res[17] = dx[17] + ((dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[3:4]

        # FRIDAY BONUS!
        # Remainder from λ-equation (2D in this case)
        res[18:19] = dx[18:19] + ((λ(t)')*(Fx(x,dx,t)-Fddx(x,dx)) - (dλ(t)')*Fdx(x,dx) - gₓ(x,dx,t)')[:]
        res[20]    = dx[20] + dλ(t)[1] - θ*λ(t)[1] + gₓ(x,dx,t)[1]
        nothing
    end

    u0 = u(0.0)[1]
    w0 = w(0.0)[1]
    x10 = -(u0+w0)/θ
    x20 = -2/( (θ*x10+u0+w0)^2 + 1 )
    dx10  = 0.0
    dx20  = 0.0
    xp10 = (u0+w0)/(θ^2)
    xp20 = 2*( 2*θ*(θ*x10+u0+w0)*dx10 + 2*x10*(θ*x10+u0+w0) )/(((θ*x10 + u0 + w0)^2+1)^2)
    x0   = [x10, x20]
    xp0  = [xp10, xp20]
    dxp10 = 0.0
    dxp20 = 0.0
    dx0   = [dx10, dx20]
    dxp0  = [dxp10, dxp20]

    G0    = 0.0
    dG0   = (y(0.0)-x20)^2
    Gp0   = 0.0
    dGp0  = 2(x20-y(0.0))*xp0[2]
    Gp02  = 0.0
    dGp02 = dGp0 - (λ(0.)')*(Fx(x0, dx0, 0.)*xp0 + Fdx(x0, dx0)*dxp0 + Fp(x0, dx0, 0.))
    Gp03  = 0.0
    # # dGp03 = -(λ(0.)')*Fp(pend0, dpend0) - ( (λ(0.)')*(Fx(pend0,dpend0)-Fddx(pend0,dpend0)) -(dλ(0.)')*Fdx(pend0,dpend0) - gₓ(pend0,dpend0,0.0)' )*s0
    dGp03 = -(λ(0.)')*Fp(x0, dx0,0.) - first( ( (λ(0.)')*(Fx(x0,dx0,0.)-Fddx(x0,dx0)) -(dλ(0.)')*Fdx(x0,dx0) - gₓ(x0,dx0,0.0)' )*xp0 )
    Gp04  = 0.0
    dGp04 = -(λ(0.)')*Fp(x0, dx0, 0.)
    Gp04b  = 0.0
    dGp04b = - ( (λ(0.)')*(Fx(x0,dx0,0.)-Fddx(x0,dx0)) -(dλ(0.)')*Fdx(x0,dx0) - gₓ(x0,dx0,0.0)' )*xp0
    Gp035  = 0.0
    dGp035 = (gₓ(x0,dx0,0.0)')*xp0 - (λ(0.0)')*( Fx(x0,dx0,0.)*xp0 + Fdx(x0,dx0)*dxp0 + Fp(x0,dx0,0.) )
    partial    = 0.0
    dpartial   = (λ(0.0)')*Fdx(x0,dx0)*dxp0 + ( (dλ(0.0)')*Fdx(x0,dx0) + (λ(0.0)')*Fddx(x0,dx0) )*xp0

    # NOTE: ONLY MAKE SENSE FOR k AND m SINCE ONLY THEN APPROPRIATE QUANTITIES ARE ZERO
    common   = 0.0
    dcommon  = (gₓ(x0,dx0,0.0)')*xp0 - (λ(0.)')*( Fx(x0,dx0,0.)*xp0 + Fp(x0,dx0,0.) )
    extra35  = 0.0
    dextra35 = -(λ(0.)')*Fdx(x0,dx0)*dxp0
    extra4   = 0.0
    dextra4  = ( (dλ(0.0)')*Fdx(x0,dx0) + (λ(0.0)')*Fddx(x0,dx0) )*xp0

    partrepL  = 0.0
    dpartrepL = (λ(0.)')*Fdx(x0,dx0)*dxp0
    partrepR  = 0.0
    dpartrepR = -((dλ(0.)')*Fdx(x0,dx0) + (λ(0.)')*Fddx(x0,dx0) )*xp0

    # FRIDAY DEBUG
    vecres1_0 = 0.0
    vecres2_0 = 0.0
    temp = -(λ(0.)')*(Fx(x0,dx0,0.0)-Fddx(x0,dx0)) + (dλ(0.0)')*Fdx(x0,dx0) + gₓ(x0,dx0,0.)'
    dvecres1_0 = temp[1]
    dvecres2_0 = temp[2]

    res1_0     = 0.0
    dres1_0    = -dλ(0.)[1] + θ*λ(0.)[1] - gₓ(x0,dx0,0.)[1]

    x0 = [x10, x20, xp10, xp20, G0, Gp0, Gp02, Gp03, Gp04, Gp04b, Gp035, partial, common, extra35, extra4, partrepL, partrepR, vecres1_0, vecres2_0, res1_0]
    dx0 = [dx10, dx20, dxp10, dxp20, dG0, dGp0, dGp02, dGp03, dGp04, dGp04b, dGp035, dpartial, dcommon, dextra35, dextra4, dpartrepL, dpartrepR, dvecres1_0, dvecres2_0, dres1_0]
    dvars = vcat([true, false, true, false], fill(true, 16))
    # dvars = vcat([true, false, true, false], fill(true, 2))
    # x0 =  [x10, x20, xp10, xp20]
    # dx0 = [dx10, dx20, dxp10, dxp20]
    # dvars = [true, false, true, false]

    # r0 = zeros(length(λ0))
    # f!(r0, dλ0, λ0, [], 0.0)
    r0 = zeros(length(x0))
    f!(r0, dx0, x0, [], 0.0)
    # @info "r0 is: $r0"

    # t -> 0.0 is just a dummy function, not to be used
    Model(f!, t -> 0.0, x0, dx0, dvars, r0)
end

function mohamed_an_stepbystep(u__::Function, w__::Function, θs::Vector{Float64}, x_::Function, dx_::Function, xθ_::Function, dxθ_::Function, y_::Function, λ_::Function, dλ_::Function, T::Float64)
    θ = θs[1]

    u_(t) = u__(t)[1]
    w_(t) = w__(t)[1]
    # TODO: Evaluate if we need any of these!
    zeta(x,dx,t) = (θ*x[1] + u_(t) + w_(t))^2 + 1
    dzeta_dx1(x,dx,t) = 2θ*(θ*x[1]+u_(t)+w_(t))
    dzeta_dθ(x,dx,t)  = 2x[1]*(θ*x[1]+u_(t)+w_(t))

    Fx(x,dx,t)   = [θ 0.0; -(2/(zeta(x,dx,t)^2))*dzeta_dx1(x,dx,t) 1.0]
    Fdx(x,dx)  = [1.0 0.0; 0.0 0.0]
    Fddx(x,dx) = zeros(2,2)
    Fp(x,dx,t)   = [x[1]; -(2/(zeta(x,dx,t)^2))*dzeta_dθ(x,dx,t);;]
    # FOR x2 AS OUTPUT
    # gₓ(x,dx,t)   = [0; 2(x[2]-y(t))/T]
    # # dgₓ(x,dx,t)  = [0; 2(dx[2]-dy(t))/T]
    # FOR x1 AS OUTPUT
    gₓ(x,dx,t)   = [2(x[1]-y_(t))/T; 0;;]
    # dgₓ(x,dx,t)  = [2(dx[1]-dy_(t))/T; 0]

    # the residual function
    function f!(res, dz, z, p, t)
        w = w_(t)
        u = u_(t)
        x = x_(t)
        dx = dx_(t)
        xθ = xθ_(t)
        dxθ = dxθ_(t)
        y = y_(t)
        λ = λ_(t)
        dλ = dλ_(t)
        # NOTE: USING x1 AS OUTPUT CURRENTLY
        # (point 1), cost
        res[1] = dz[1] - (1/T)*(y-x[1])^2
        # (point 2), cost sensitivity

        res[2] = dz[2] - first((gₓ(x,dx,t)')*xθ)
        # (point 3), cost sensitivity with λ-term that's always zero
        res[3] = dz[3] - first((gₓ(x,dx,t)')*xθ + (λ')*(Fx(x,dx,t)*xθ + Fdx(x,dx)*dxθ + Fp(x,dx,t)))
        # (point 4), cost sensitivity after partial integration
        res[4] = dz[4] + first((λ')*Fp(x,dx,t) + ( (λ')*(Fx(x,dx,t)-Fddx(x,dx)) - (dλ')*Fdx(x,dx)-gₓ(x,dx,t)' )*xθ)   # NOTE: Needs term added to it to be correct
        # (point 5), cost sensitivity, assuming that λ satisfied adjoint equation
        res[5] = dz[5] + first((λ')*Fp(x,dx,t))  # NOTE: Needs term added to it to be correct
        nothing
    end

    w = w_(0.)
    u = u_(0.)
    x = x_(0.)
    dx = dx_(0.)
    xθ = xθ_(0.)
    dxθ = dxθ_(0.)
    y = y_(0.)
    λ = λ_(0.)
    dλ = dλ_(0.)
    z0  = zeros(5)
    dz0_1 = (1/T)*(y-x[1])^2
    dz0_2 = first((gₓ(x,dx,0.)')*xθ)
    dz0_3 = first((gₓ(x,dx,0.)')*xθ - (λ')*(Fx(x,dx,0.)*xθ + Fdx(x,dx)*dxθ + Fp(x,dx,0.)))
    dz0_4 = first(-(λ')*Fp(x,dx,0.) - ( (λ')*(Fx(x,dx,0.)-Fddx(x,dx)) - (dλ')*Fdx(x,dx) - gₓ(x,dx,0.)' )*xθ)
    dz0_5 = first(-(λ')*Fp(x,dx,0.))
    dz0 = [dz0_1, dz0_2, dz0_3, dz0_4, dz0_5]

    dvars = fill(true, 5)
    r0 = zeros(length(z0))
    f!(r0, dz0, z0, [], 0.0)
    # @info "r0 is: $r0"

    # t -> 0.0 is just a dummy function, not to be used
    Model(f!, t -> 0.0, z0, dz0, dvars, r0)
end

# TODO: Make into ODE instead?
function mohamed_analytical(u::Function, w::Function, θ::Vector{Float64})
    # NOTE: Φ not used, just passed to have consistent interface
    function f!(res, dx, x, p, t)
        wt = w(t)[1]
        ut = u(t)[1]
        # Dynamic Equations
        res[1] = dx[1] - exp(θ[1]*t)*(ut+wt)
        res[2] = dx[2] - t*exp(θ[1]*t)*(ut+wt)
        nothing
    end

    # Finding consistent initial conditions
    # Initial values, the pendulum starts at rest
    u0 = u(0.0)[1]
    w0 = w(0.0)[1]
    x0 = zeros(2)
    dx0 = [u0+w0, 0.0]

    dvars = [true, true]

    r0 = zeros(length(x0))
    f!(r0, dx0, x0, [], 0.0)

    # t -> 0.0 is just a dummy function, not to be used
    Model(f!, t -> 0.0, x0, dx0, dvars, r0)
end

function fast_heat_transfer_reactor(V0::Float64, T0::Float64, u::Function, w::Function, θ::Vector{Float64})::Model
    let k0 = θ[1], k1 = θ[2], k2 = θ[3], k3 = θ[4], k4 = θ[5]

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - ut[1] + ut[4]
            res[2] = dx[2] - ut[1]*(ut[2]-x[2])/x[1] + k0*TEMPECSPRESSION(-k1/x[4])x[2]
            res[3] = dx[3] + ut[1]x[3]/x[1] - k0*TEMPECSPRESSION(-k1/x[4])x[2]
            res[4] = x[6] - k3*x[5]/x[1] - ut[1]*(ut[3]-x[4])/x[1] + k0*k2*TEMPECSPRESSION(-k1/x[4])*x[2]
            res[5] = dx[4] + k3*x[5]/k4 - ut[5]*(ut[6]-x[4])/k4 - wt[1]^2
            res[6] = x[6] - dx[4]
            nothing
        end

        # Finding consistent initial conditions
        # Initial values, with starting volume V0 and starting temperature T0
        u0 = u(0.0)
        w0 = w(0.0)
        x5_0 = (V0*u0[5]*(u0[6]-T0) - k4*u0[1]*(u0[3]-T0))/((V0+k4)k3)
        dx4_0 = -k3*x5_0/k4 + u0[5]*(u0[6]-T0)/k4
        dx6_0 = -k3*x5_0/V0^2 - u0[1]*dx4_0/V0 - u0[1]*(u0[3]-T0)/V0^2 + k0*k2*u0[1]*u0[2]TEMPECSPRESSION(-k1/T0)/V0

        x0 = [V0; 0.; 0.; T0; x5_0; dx4_0]
        dx0 = [u0[1]-u0[4]; u0[1]u0[2]/V0; 0.; dx4_0; 0.; dx6_0]

        dvars = vcat(fill(true, 4), [false, false])

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_ode(Φ::Float64, u::Function, w::Function, θ::Vector{Float64})::Model_ode
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        # Similarly to the DAE-implementation, we don't use the ability of passing
        # the parameters to the function f. Instead, we create a new problem for
        # new values of the parameters by calling pendulum_ode() again when
        # f(x,p,t) = [x[2]; (m^2)*(L^2)*g*cos(x[1])-k*(L^2)*x[2]^2 + u(t)[1] + w(t)[1]^2]
        f(x,p,t) = [x[2]; (m^2)*(L^2)*g*cos(x[1])-k*(L^2)*x[2]*abs(x[2]) + u(t)[1] + w(t)[1]^2]
        x0 = [Φ; 0.0]
        return Model_ode(f, x0)
    end
end

function pendulum_sensitivity_ode(Φ::Float64, u::Function, w::Function, θ::Vector{Float64})::Model_ode
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        # Similarly to the DAE-implementation, we don't use the ability of passing
        # the parameters to the function f. Instead, we create a new problem for
        # new values of the parameters by calling pendulum_ode() again when
        # parameters change. Therefore, the argument p is unused.
        f(x,p,t) = [x[2];
                    (m^2)*(L^2)*g*cos(x[1])-k*(L^2)*x[2]*abs(x[2]) + u(t)[1] + w(t)[1]^2
                    x[4]
                    -(m^2)*(L^2)*g*sin(x[1])*x[3]-2*k*(L^2)*abs(x[2])*x[4] - (L^2)*x[2]*abs(x[2])]
        x0 = [Φ; 0.0; 0.0; 0.0]
        return Model_ode(f, x0)
    end
end

function simple_model_sens(u::Function, w::Function, θ::Vector{Float64})::Model

    function f(out, dx, x, p, t)
        wt = w(t)
        ut = u(t)
        out[1] = dx[1] + θ[1]*x[1] - ut[1] - wt[1]      # Equation 1
        out[2] = x[2] - x[1]^2                          # Equation 2     x
        out[3] = θ[1]*x[3] + dx[3] + x[1]               # Sensitivity of x₁
        out[4] = -2x[1]*x[3] + x[4]                     # Sensitivity of x₂
    end

    x₀  = [0.0, 0.0, 0.0, 0.0]
    dx₀ = [0.0, 0.0, 0.0, 0.0]
    dvars = [true, false, true, false]

    Model(f, x -> x, x₀, dx₀, dvars, [0.0])
end

function simple_model(u::Function, w::Function, θ::Vector{Float64})::Model

    function f(out, dx, x, p, t)
        wt = w(t)
        ut = u(t)
        out[1] = dx[1] + θ[1]*x[1] - ut[1] - wt[1]      # Equation 1
        out[2] = x[2] - x[1]^2                          # Equation 2     x                  # Sensitivity of x₂
    end

    x₀  = [0.0, 0.0]
    dx₀ = [0.0, 0.0]
    dvars = [true, false]

    Model(f, x -> x, x₀, dx₀, dvars, [0.0])
end

function simulation_plots(T, sols, vars; kwargs...)
  ps = [plot() for var in vars]
  np = length(ps)

  for sol in sols
    for p = 1:np
      plot!(ps[p], sol, tspan=(0.0, T), vars=[vars[p]]; kwargs...)
    end
  end

  ps
end

function problem(m::Model, N::Int, Ts::Float64)
  T = N * Ts
  # ff = DAEFunction(m.f!, jac = m.jac!)
  # ff = DAEFunction{true,true}(m.f!)
  DAEProblem(m.f!, m.dx0, m.x0, (0, T), [], differential_vars=m.dvars)
end

function problem_reverse(m::Model, N::Int, Ts::Float64)
    T = N * Ts
    # ff = DAEFunction(m.f!, jac = m.jac!)
    # ff = DAEFunction{true,true}(m.f!)
    DAEProblem(m.f!, m.dx0, m.x0, (T, 0), [], differential_vars=m.dvars)
end

function problem_ode(m::Model_ode, N::Int, Ts::Float64)
    T = N * Ts
    ODEProblem(m.f, m.x0, (0,T), [])
end

function solve(prob; kwargs...)
  DifferentialEquations.solve(prob, IDA(); kwargs...)
end

# TODO: Might be worth specifying solver instead of letting it be picked automatically
function solve_ode(prob; kwargs...)
    DifferentialEquations.solve(prob; kwargs...)
end

# If these principles are not achieved yet, then that should be fixed
####################################################################
### DESIGN PRINCIPLES BEHIND apply_outputfun AND solve-FUNCTIONS ###
####################################################################
# sol.u is a vector of states (which are themselves vectors)
# After applying output function, we return a vector of outputs (i.e. a vector of vector or a vector of numbers, depending on dimensionality of output)
# After applying sensitivity function, we return a vector of matrices. The rows of the matrices represent different output components, and the columns represent different parameters

# in solve_in_parallel-functions:
# we apply vcat(*...) on the output vector of vectors/numbers. Regardless of if it's vectors or scalars on the inside, this operation will return a vector. For mutlidimensional outputs, 
#       all components of the output for a given time will be stacked next to each other.
# we apply vcat(*...) on the sensitivity vector of matrices. The operation returns a matrix, where rows correspond to time/output component and columns correspond to different parameters. 
#       For the rows, all components of the output for a given time will be stacked next to each other.
####################################################################
# TODO: We can then unite apply_outputfun and apply_outputfun_mvar !!! :D
# TODO: We might not even need the apply_outputfun-functions then?? Maybe

function apply_outputfun(h, sol)
  if sol.retcode != :Success
    throw(ErrorException("Solution retcode: $(sol.retcode)"))
  end
  # NOTE: There are alternative, more recommended, ways of accessing solution
  # than through sol.u: https://diffeq.sciml.ai/stable/basics/solution/
  map(h, sol.u)
end

function apply_two_outputfun(h1, h2, sol)
    if sol.retcode != :Success
        throw(ErrorException("Solution retcode: $(sol.retcode)"))
    end
    # NOTE: There are alternative, more recommended, ways of accessing solution
    # than through sol.u: https://diffeq.sciml.ai/stable/basics/solution/
    map(h1, sol.u), map(h2, sol.u)
end

function solve_in_parallel(solve, is)
  M = length(is)
  p = Progress(M, 1, "Running $(M) simulations...", 50)
  y1 = solve(is[1])
  ny = length(y1[1])
  Y = zeros(ny*length(y1), M)
  Y[:, 1] += vcat(y1...)
  next!(p)
  Threads.@threads for m = 2:M
      y = solve(is[m])
      Y[:, m] += vcat(y...)
      next!(p)
  end
  Y
end

function solve_in_parallel_block(solve, is, k)
    M = length(is)
    p = Progress(M, 1, "Running $(M) simulations with block size $k...", 50)
    y1 = solve(is[1])
    ny = length(y1[1])
    Y = zeros(ny*size(y1,1), M*k)
    Y[:, 1:k] += y1
    next!(p)
    Threads.@threads for m = 2:M
        Y[:, (m-1)*k+1:m*k] += solve(is[m])
        next!(p)
    end
    return Y
end

function solve_adj_in_parallel(solve, is)
  M = length(is)
  p = Progress(M, 1, "Running $(M) simulations...", 50)
  Gp1 = solve(is[1])
  Gps = zeros(length(Gp1), M)
  Gps[:, 1] .+= Gp1
  next!(p)
  Threads.@threads for m = 2:M
      Gp = solve(is[m])
      Gps[:, m] .+= Gp
      next!(p)
  end
  Gps
end

# Handles multivariate outputs by flattening the output
function solve_in_parallel_sens(solve, is)
    M = length(is)
    p = Progress(M, 1, "Running $(M) simulations...", 50)
    y1, sens1 = solve(is[1])
    ny = length(y1[1])
    Ysens = [Matrix{Float64}(undef, ny*length(sens1), length(sens1[1])÷ny) for m=1:M]
    Y = zeros(ny*length(y1), M)
    Y[:,1] += vcat(y1...)   # Flattens the array
    Ysens[1] = vcat(sens1...)
    next!(p)
    Threads.@threads for m = 2:M
        y, sens = solve(is[m])
        Y[:,m] += vcat(y...)   # Flattens the array
        Ysens[m] = vcat(sens...)
        next!(p)
    end
    Y, Ysens
end

# Could the uses of this be replace by just solve_in_parallel? That would be nice
function solve_in_parallel_sens_debug(solve, is, yind, sensinds, sampling_ratio)
    M = length(is)
    p = Progress(M, 1, "Running $(M) simulations...", 50)
    y1 = solve(is[1])
    ny = length(y1[1])

    # Rows of inner matrix are assumed to be different values of t, columns of
    # matrix are assumed to be different elements of the vector-valued process

    # Inner matrix is size of state vector times number of time samples. Or the other way around....?
    matout = [Matrix{Float64}(undef, length(y1), ny) for m=1:M]
    Y = zeros(length(y1[1:sampling_ratio:end,:]), M)
    sens = [zeros(length(y1[1:sampling_ratio:end,:]), length(sensinds)) for m=1:M]
    matout[1] = transpose(hcat(y1...))     # TODO: Would using matout[1][:,:] = transpose(hcat(y1...)) instead avoid allocating new memory for matout[1]?
    Y[:,1]    = matout[1][1:sampling_ratio:end, yind]
    sens[1]   = matout[1][1:sampling_ratio:end, sensinds]
    next!(p)
    Threads.@threads for m = 2:M
        y1 = solve(is[m])
        matout[m] = transpose(hcat(y1...))
        Y[:,m]  = matout[m][1:sampling_ratio:end, yind]
        sens[m] = matout[m][1:sampling_ratio:end, sensinds]
        next!(p)
    end
    matout, Y, sens
end

# NOTE: Not updated according to new principles, I'm not even sure if it's used anymore
function solve_in_parallel_debug(solve, is, yind, sampling_ratio)
    M = length(is)
    p = Progress(M, 1, "Running $(M) simulations...", 50)
    out = solve(is[1])
    matout = [Matrix{Float64}(undef, size(out)) for m=1:M]
    Y = zeros(size(out[1:sampling_ratio:end,:], 1), M)
    # sens = zeros(size(out,1), M)
    matout[1] = out
    Y[:,1]    = out[1:sampling_ratio:end, yind]
    next!(p)
    Threads.@threads for m = 2:M
        out = solve(is[m])
        matout[m] = out
        Y[:,m]  = out[1:sampling_ratio:end, yind]
        next!(p)
    end
    matout, Y
end

# TODO: Is it even used anymore? Can we delete this and the old API?
function get_sol_in_parallel(solve, is)
    M = length(is)
    p = Progress(M, 1, "Running $(M) simulations...", 50)
    solutions = Array{DAESolution, 1}(undef, M)
    # y1 = solve(is[1])
    # Y = zeros(length(y1), M)
    # Y[:, 1] += y1
    next!(p)
    Threads.@threads for m = 1:M
        solutions[m] = solve(is[m])
        next!(p)
    end
    solutions
end
# Old API

function simulate(m::Model, N::Int, Ts::Float64; kwargs...)
  let
    T = N * Ts
    saveat = 0:Ts:T

    prob = problem(m, N, Ts)

    solve(prob, saveat = saveat; kwargs...)
  end
end

function simulate_h(m::Model, N::Int, Ts::Float64, h::Function)
  sol = simulate(m, N, Ts)
  apply_outputfun(h, sol)
end

function simulate_m(mk_model::Function, N::Int, Ts::Float64)
  function f(m)
    model = mk_model(m)
    simulate(model, N, Ts)
  end
end

function simulate_h_m(
  mk_model::Function, N::Int, Ts::Float64, h::Function, ms::Array{Int, 1}
)::Matrix{Float64}

  M = length(ms)
  # Y = hcat([[Threads.Atomic{Float64}(0.0) for i=1:(N+1)] for j=1:M]...)
  Y = zeros(N+1, M)
  p = Progress(M, 1, "Running $(M) simulations...", 50)
  Threads.@threads for m = 1:M
    model = mk_model(ms[m])
    y = simulate_h(model, N, Ts, h)
    # for n = 1:(N+1)
    # Threads.atomic_add!(Y[k, m], y[k])
    Y[:, m] .+= y
    # end
    next!(p)
  end
  # map(y -> y[], Y)
  Y
end
