using Plots

function crazy_plots(x_vars::Array{Float64,1}, y_vars::Array{Float64,1}, z_vals, trace::Array{Float64, 2})
        lval = 0.9*minimum(z_vals)
        uval = 1.1*maximum(z_vals)
        p = plot(trace[1, [1,1]], trace[1, [2,2]], [lval, uval])
        # p = plot(x_vars, y_vars, z_vals, st=:surface, xlabel="m values", ylabel="k values", zlabel="cost")
        for row in 2:size(trace, 1)
            plot!(p, trace[row, [1,1]], trace[row, [2,2]], [lval, uval])
        end
        plot!(p, x_vars, y_vars, z_vals, st=:surface, xlabel="m values", ylabel="k values", zlabel="cost")
        display(p)
        return p
end

function crazy_anim(x_vars::Array{Float64,1}, y_vars::Array{Float64,1}, z_vals, trace::Array{Float64, 2})
        lval = 0.9*minimum(z_vals)
        uval = 1.1*maximum(z_vals)
        # p = plot(x_vars, y_vars, z_vals, st=:surface, xlabel="m values", ylabel="k values", zlabel="cost")
        anim = @animate for row in 1:size(trace, 1)
            p = plot(trace[row, [1,1]], trace[row, [2,2]], [lval, uval])
            plot!(p, x_vars, y_vars, z_vals, st=:surface, xlabel="m values", ylabel="k values", zlabel="cost", size=(1200,1000))
        end
        gif(anim, "3d_plot.gif", fps = 15)
end

function crazy_anim2(x_vars::Array{Float64,1}, y_vars::Array{Float64,1}, z_vals, trace::Array{Float64, 2}, optimum::Array{Float64,1})
        lval = 0.9*minimum(z_vals)
        uval = 1.1*maximum(z_vals)
        # p = plot(x_vars, y_vars, z_vals, st=:surface, xlabel="m values", ylabel="k values", zlabel="cost")
        anim = @animate for row in 1:(size(trace, 1)+30)
            if row <= size(trace,1)
                p = plot(trace[row, [1,1]], trace[row, [2,2]], [lval, uval])
                plot!(p, x_vars, y_vars, z_vals, st=:surface, xlabel="m values", ylabel="g values", zlabel="cost", size=(1200,1000))
            else
                p = plot(x_vars, y_vars, z_vals, st=:surface, xlabel="m values", ylabel="g values", zlabel="cost", size=(1200,1000))
            end
            plot!(p, optimum[[1,1]], optimum[[2,2]], [lval, uval], linecolor=:green)
        end
        gif(anim, "3d_plot.gif", fps = 15)
end
