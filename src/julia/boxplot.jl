using PyPlot
using DelimitedFiles

thb = readdlm("data/experiments/candidate_7/theta_hat_baseline.csv", ',')
thm = readdlm("data/experiments/candidate_7/theta_hat_mean.csv", ',')
# theta = readdlm("data/experiments/candidate_6/theta.csv", ',')

th0 = 6.25
labels = [L"$5$", L"$10$", L"$20$", L"$30$", L"$40$", L"$50$"]

fig, ax = subplots(1,2, sharey = true)

ax[1,1][:boxplot](thb, showmeans = true, labels = labels)
ax[1,1][:axhline](th0, color = "gray", linestyle = ":")
ax[1,1][:set_ylabel](L"$\hat{\theta}$")
ax[1,1][:set_xlabel](L"$N/10^3$")
ax[1,1][:set_title]("output error")
ax[1,1][:grid]("on", alpha = 0.5)
ax[2,1][:boxplot](thm, showmeans = true, labels = labels)
ax[2,1][:axhline](th0, color = "gray", linestyle = ":")
ax[2,1][:set_xlabel](L"$N/10^3$")
ax[2,1][:set_title]("proposed")
ax[2,1][:grid]("on", alpha = 0.5)


# subplot(121, sharey=true)
# boxplot(thb, showmeans = true, labels = labels)
# axhline(th0, color = "gray", linestyle = "--")
# grid("on")

# subplot(122, sharey=true)
# boxplot(thm, showmeans = true, labels = labels)
# axhline(th0, color = "gray", linestyle = "--")
# grid("on")
