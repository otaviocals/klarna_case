import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp

data = pd.read_csv("train_data.csv")

data = data.drop(columns=["uuid"])
data["has_paid"] = data["has_paid"].astype(str)

data_0 = data.loc[data["default"] == 0]
data_1 = data.loc[data["default"] == 1]

data_0 = data_0.drop(columns=["default"])
data_1 = data_1.drop(columns=["default"])

data_0 = data_0.reset_index(drop=True)
data_1 = data_1.reset_index(drop=True)

columns = data_0.columns
categoricals = [
    "merchant_category",
    "merchant_group",
    "name_in_email",
    "has_paid",
    "account_status",
    "account_worst_status_0_3m",
    "account_worst_status_3_6m",
    "account_worst_status_6_12m",
    "account_worst_status_12_24m",
    "worst_status_active_inv",
    "status_last_archived_0_24m",
    "status_2nd_last_archived_0_24m",
    "status_3rd_last_archived_0_24m",
    "status_max_archived_0_6_months",
    "status_max_archived_0_12_months",
    "status_max_archived_0_24_months",
]

normal_axis = [
    "age",
    "avg_payment_span_0_3m",
    "avg_payment_span_0_12m",
    #"num_arch_written_off_0_12m",
    "num_arch_written_off_12_24m",
]

stats = pd.DataFrame(columns=["feature", "ks-stat", "p-value"])

for column in columns:
    ks_stats = ks_2samp(data_0[column], data_1[column])
    ks_stats_pd = pd.DataFrame(
        {
            "feature": [column],
            "ks-stat": [ks_stats.statistic],
            "p-value": [ks_stats.pvalue],
        }
    )

    stats = stats.append(ks_stats_pd)


    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    colors = ["b", "r"]

    if column not in categoricals:

        if column in normal_axis:
            sns.kdeplot(data_0[column], label="0", ax=ax1, fill=True, color=colors[0])
            sns.kdeplot(data_1[column], label="1", ax=ax2, fill=True, color=colors[1])
        else:
            sns.kdeplot(
                np.log(data_0[column]), label="0", ax=ax1, fill=True, color=colors[0]
            )
            sns.kdeplot(
                np.log(data_1[column]), label="1", ax=ax2, fill=True, color=colors[1]
            )

    else:

        #sns.histplot(data_0[column], label="0", ax=ax1, color=colors[0])
        #sns.histplot(data_1[column], label="1", ax=ax2, color=colors[1])
        ax1.hist([data_0[column],data_1[column]], color=colors)
        n, bins, patches = ax1.hist([data_0[column],data_1[column]])
        ax1.cla()

        width = (bins[1] - bins[0]) * 0.4
        bins_shifted = bins + width

        ax1.bar(bins[:-1], n[0], width, align='edge', color=colors[0], label="0", alpha=0.5)
        ax2.bar(bins_shifted[:-1], n[1], width, align='edge', color=colors[1], label="1", alpha=0.5)

    ax1.set_ylabel("Count", color=colors[0])
    ax2.set_ylabel("Count", color=colors[1])
        # ax1.set_xscale('log')
        # ax2.set_xscale('log')
    ax1.tick_params("y", colors=colors[0])
    ax2.tick_params("y", colors=colors[1])

    fig.legend(loc="upper right")
    if column in normal_axis:
        fig.suptitle(column)
    else:
        fig.suptitle(column + " (LOG)")

    textstr = '\n'.join((
        'ks-stat= {:0.2f}'.format(ks_stats.statistic),
        'p-value= {:0.5f}'.format(ks_stats.pvalue)
        ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

    fig.tight_layout()
    fig.savefig("eda_plots/distrib_{}.png".format(column))
    plt.close()

stats = stats.sort_values(by="ks-stat", ascending=False)
stats = stats.reset_index(drop=True)
print(stats)
stats.to_csv("ks_values.csv", index=False)

# Plot KS
plt.figure()
#plt.suptitle("Metrics")
#ax1.barh(stats["ks-stat"], stats["feature"])
stats_stats = stats.copy()
stats_stats = stats_stats.sort_values(by="ks-stat", ascending=True)
stats_stats.plot.barh( x="feature", y="ks-stat", figsize=(5, 9), legend=False)
plt.ylabel("KS-Statistic")
#ax1.set_ylim(0.5, 1.0)
#fig.tight_layout()
plt.savefig("ks-values.png", bbox_inches="tight", dpi=600)
plt.close()

plt.figure()
#ax.barh(stats["p-value"], stats["feature"])
stats_pvalue = stats.copy()
#stats_pvalue["p-value"] = -1/np.log(stats_pvalue["p-value"]*100000000)
#stats_pvalue["p-value"] = 1/(1+np.exp(-80*(stats_pvalue["p-value"]-np.mean(stats_pvalue["p-value"]))))
stats_pvalue = stats_pvalue.sort_values(by="p-value", ascending=False)
stats_pvalue.plot.barh( x="feature", y="p-value", figsize=(5, 9), legend=False)
plt.xscale('log')
plt.ylabel("P-Value (LOG)")
#ax2.set_ylim(0.0, 1.0)
#fig.tight_layout()
plt.savefig("p-values.png", bbox_inches="tight", dpi=600)
plt.close()

