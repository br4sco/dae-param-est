using DelimitedFiles

# res_500 = readdlm("data/results/delta_500_smallw_largeu_baseline/opt_pars_baseline.csv", ',')
# res_2k = readdlm("data/results/delta_2k_smallw_largeu_baseline/opt_pars_baseline.csv", ',')
# res_3p5k = readdlm("data/results/delta_3p5k_smallw_largeu_baseline/opt_pars_baseline.csv", ',')
# res_5k = readdlm("data/results/delta_5k_smallw_largeu_baseline/opt_pars_baseline.csv", ',')
res_500 = readdlm("data/results/delta_500_smallw_largeu/opt_pars_proposed.csv", ',')
res_2k = readdlm("data/results/delta_2k_smallw_largeu/opt_pars_proposed.csv", ',')
res_3p5k = readdlm("data/results/delta_3p5k_smallw_largeu/opt_pars_proposed.csv", ',')
res_5k = readdlm("data/results/delta_5k_smallw_largeu/opt_pars_proposed.csv", ',')

L0res = hcat(res_500[1,:], res_2k[1,:], res_3p5k[1,:], res_5k[1,:])
L1res = hcat(res_500[2,:], res_2k[2,:], res_3p5k[2,:], res_5k[2,:])
L2res = hcat(res_500[3,:], res_2k[3,:], res_3p5k[3,:], res_5k[3,:])
L3res = hcat(res_500[4,:], res_2k[4,:], res_3p5k[4,:], res_5k[4,:])
LC1res = hcat(res_500[5,:], res_2k[5,:], res_3p5k[5,:], res_5k[5,:])
LC2res = hcat(res_500[6,:], res_2k[6,:], res_3p5k[6,:], res_5k[6,:])
M1res = hcat(res_500[7,:], res_2k[7,:], res_3p5k[7,:], res_5k[7,:])
M2res = hcat(res_500[8,:], res_2k[8,:], res_3p5k[8,:], res_5k[8,:])
M3res = hcat(res_500[9,:], res_2k[9,:], res_3p5k[9,:], res_5k[9,:])
J1res = hcat(res_500[10,:], res_2k[10,:], res_3p5k[10,:], res_5k[10,:])
J2res = hcat(res_500[11,:], res_2k[11,:], res_3p5k[11,:], res_5k[11,:])
γres = hcat(res_500[12,:], res_2k[12,:], res_3p5k[12,:], res_5k[12,:])

a1res = hcat(res_500[13,:], res_2k[13,:], res_3p5k[13,:], res_5k[13,:])
a2res = hcat(res_500[14,:], res_2k[14,:], res_3p5k[14,:], res_5k[14,:])
a3res = hcat(res_500[15,:], res_2k[15,:], res_3p5k[15,:], res_5k[15,:])
c11res = hcat(res_500[16,:], res_2k[16,:], res_3p5k[16,:], res_5k[16,:])
c21res = hcat(res_500[17,:], res_2k[17,:], res_3p5k[17,:], res_5k[17,:])
c31res = hcat(res_500[18,:], res_2k[18,:], res_3p5k[18,:], res_5k[18,:])
c12res = hcat(res_500[19,:], res_2k[19,:], res_3p5k[19,:], res_5k[19,:])
c22res = hcat(res_500[20,:], res_2k[20,:], res_3p5k[20,:], res_5k[20,:])
c32res = hcat(res_500[21,:], res_2k[21,:], res_3p5k[21,:], res_5k[21,:])
c13res = hcat(res_500[22,:], res_2k[22,:], res_3p5k[22,:], res_5k[22,:])
c23res = hcat(res_500[23,:], res_2k[23,:], res_3p5k[23,:], res_5k[23,:])
c33res = hcat(res_500[24,:], res_2k[24,:], res_3p5k[24,:], res_5k[24,:])

# writedlm("data/results/AUTOMATICA_RESULTS/proposed/L0base.csv", L0res, ',')
# writedlm("data/results/AUTOMATICA_RESULTS/proposed/L1base.csv", L1res, ',')
# writedlm("data/results/AUTOMATICA_RESULTS/proposed/L2base.csv", L2res, ',')
# writedlm("data/results/AUTOMATICA_RESULTS/proposed/L3base.csv", L3res, ',')
# writedlm("data/results/AUTOMATICA_RESULTS/proposed/LC1base.csv", LC1res, ',')
# writedlm("data/results/AUTOMATICA_RESULTS/proposed/LC2base.csv", LC2res, ',')
# writedlm("data/results/AUTOMATICA_RESULTS/proposed/M1base.csv", M1res, ',')
# writedlm("data/results/AUTOMATICA_RESULTS/proposed/M2base.csv", M2res, ',')
# writedlm("data/results/AUTOMATICA_RESULTS/proposed/M3base.csv", M3res, ',')
# writedlm("data/results/AUTOMATICA_RESULTS/proposed/J1base.csv", J1res, ',')
# writedlm("data/results/AUTOMATICA_RESULTS/proposed/J2base.csv", J2res, ',')
# writedlm("data/results/AUTOMATICA_RESULTS/proposed/gammabase.csv", γres, ',')

# writedlm("data/results/AUTOMATICA_RESULTS/proposed/L0prop.csv", L0res, ',')
# writedlm("data/results/AUTOMATICA_RESULTS/proposed/L1prop.csv", L1res, ',')
# writedlm("data/results/AUTOMATICA_RESULTS/proposed/L2prop.csv", L2res, ',')
# writedlm("data/results/AUTOMATICA_RESULTS/proposed/L3prop.csv", L3res, ',')
# writedlm("data/results/AUTOMATICA_RESULTS/proposed/LC1prop.csv", LC1res, ',')
# writedlm("data/results/AUTOMATICA_RESULTS/proposed/LC2prop.csv", LC2res, ',')
# writedlm("data/results/AUTOMATICA_RESULTS/proposed/M1prop.csv", M1res, ',')
# writedlm("data/results/AUTOMATICA_RESULTS/proposed/M2prop.csv", M2res, ',')
# writedlm("data/results/AUTOMATICA_RESULTS/proposed/M3prop.csv", M3res, ',')
# writedlm("data/results/AUTOMATICA_RESULTS/proposed/J1prop.csv", J1res, ',')
# writedlm("data/results/AUTOMATICA_RESULTS/proposed/J2prop.csv", J2res, ',')
# writedlm("data/results/AUTOMATICA_RESULTS/proposed/gammaprop.csv", γres, ',')

# writedlm("data/results/AUTOMATICA_RESULTS/proposed/a1prop.csv", a1res, ',')
# writedlm("data/results/AUTOMATICA_RESULTS/proposed/a2prop.csv", a2res, ',')
# writedlm("data/results/AUTOMATICA_RESULTS/proposed/a3prop.csv", a3res, ',')
# writedlm("data/results/AUTOMATICA_RESULTS/proposed/c11prop.csv", c11res, ',')
# writedlm("data/results/AUTOMATICA_RESULTS/proposed/c21prop.csv", c21res, ',')
# writedlm("data/results/AUTOMATICA_RESULTS/proposed/c32prop.csv", c31res, ',')
# writedlm("data/results/AUTOMATICA_RESULTS/proposed/c12prop.csv", c12res, ',')
# writedlm("data/results/AUTOMATICA_RESULTS/proposed/c22prop.csv", c22res, ',')
# writedlm("data/results/AUTOMATICA_RESULTS/proposed/c32prop.csv", c32res, ',')
# writedlm("data/results/AUTOMATICA_RESULTS/proposed/c13prop.csv", c13res, ',')
# writedlm("data/results/AUTOMATICA_RESULTS/proposed/c23prop.csv", c23res, ',')
# writedlm("data/results/AUTOMATICA_RESULTS/proposed/c33prop.csv", c33res, ',')


# --------------------- FOR GENERATING TABLES INSTEAD! ----------------------------

par_names = ["L_0", "L_1", "L_2", "L_3", "L_{C1}", "L_{C2}", "M_1", "M_2", "M_3", "J_1", "J_2", "γ"]
true_vals = [1.0, 1.5, 2.0, 0.5, 0.75, 1.0, 0.1, 0.1, 0.3, 0.4, 0.4, 1.0]

# "L_0 & L_1 & L_2 & L_3 & L_{C1} & L_{C2} & M_1 & M_2 & M_3 & J_1 & J_2 & γ"

num_per_row = 4
num_rows = length(par_names)÷4

tmp = "\textbf{Parameter:} &"
mystr = ""

# THIS SUCKS A BIT ACTUALLY! Why Do it this annoying way though.....??

# for rowind = 0:num_rows
#     for colind = 1:num_per_row
#         if rowind==0
#             tmp *= " " * par_names[colind] * " "
#             if colind != num_per_row
#                 tmp *= "&"
#             end 
#         else
#             mystr = 
#         end
#     end
# end

# I CAN JUST DO csv_write WITH WELL CHOSEN DELIMITER????? <-------------------------------------------------------------------

mystr = ""
for rowind = 1:num_rows
    for colind = 1:num_per_row
        ind = (rowind-1)*num_per_row + colind
        # MEAN
        mystr = string(mean(res_500[ind,:])) * " & " * string(mean(res_2k[ind,:])) * " & " * string(mean(res_3p5k[ind,:])) * " & " * string(mean(res_5k[ind,:]))
        # BIAS
        mystr2 = string(mean(res_500[ind,:])-true_vals[ind]) * " & " * string(mean(res_2k[ind,:])-true_vals[ind]) * " & " * string(mean(res_3p5k[ind,:])-true_vals[ind]) * " & " * string(mean(res_5k[ind,:])-true_vals[ind])
        # VARIANCE
        mystr3 = string(var(res_500[ind,:])) * " & " * string(var(res_2k[ind,:])) * " & " * string(var(res_3p5k[ind,:])) * " & " * string(var(res_5k[ind,:]))
    end
end