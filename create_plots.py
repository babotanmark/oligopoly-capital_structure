import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import locale # to have commas as decimal separator
sns.set_theme()
locale.setlocale(locale.LC_NUMERIC, "de_DE")

# general settings
font = {'family' : "Times New Roman",
        'size'   : 12}
plt.rc('font', **font)
plt.rcParams['axes.formatter.use_locale'] = True
line_colors = {
    1: "red",
    2: "blue",
    3: "green"
}


## create mc simulation plots
# read tables
tables = {}
for i in range(1,4):
    tables[i] = pd.read_excel("excels/" + str(i) + "-firms-mc-agg.xlsx")

# plots
n = len(tables[1].index)
x = range(1, n+1)
default = {}
for i in range(1,4):
    default[i] = tables[i].loc[0, "time_of_default"]


# debt to asset ratio
fig, ax = plt.subplots()
for i in range(1,4):
    plt.plot(x, 100*tables[i]["debt_to_asset"], color=line_colors[i], lw=2, label=str(i) + " vállalat")
    plt.axvline(default[i], color=line_colors[i], lw=1, linestyle='dashed')
plt.ylabel("Könyv szerinti Hitel/Eszköz (%)", labelpad=10)
plt.xlabel("Év")
plt.text(0.67, 1.05, "Csőd éve", transform=ax.transAxes, bbox=dict(facecolor='none', ls='--', edgecolor='black'))
fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
#plt.tight_layout()
#plt.show()
#plt.close('all')
plt.savefig("plots/mc-debt.png", bbox_inches="tight", pad_inches=0.2, dpi=200)


# production
fig, ax = plt.subplots()
for i in range(1,4):
    plt.plot(x, tables[i]["prod"], color=line_colors[i], lw=2, label=str(i) + " vállalat")
plt.ylabel("A vállalatok össztermelése", labelpad=10)
plt.xlabel("Év")
fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
#plt.show()
#plt.close('all')
plt.savefig("plots/mc-production.png", bbox_inches="tight", pad_inches=0.2, dpi=200)


# price
# fig, ax = plt.subplots()
# for i in range(1,4):
#     plt.plot(x, tables[i]["price"], color=line_colors[i], lw=2, label=str(i) + " vállalat")
# plt.ylabel("A termék ára")
# plt.xlabel("Év")
# plt.legend()
# #plt.show()
# plt.close('all')


# threshold a / realized a
fig, ax = plt.subplots()
for i in range(1,4):
    plt.plot(x, 100*tables[i]["threshold_a"]/tables[i]["a"], color=line_colors[i], lw=2, label=str(i) + " vállalat")
plt.ylabel("Küszöb A / A (%)", labelpad=10)
plt.xlabel("Év")
fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
plt.tight_layout()
#plt.show()
#plt.close('all')
plt.savefig("plots/mc-threshold.png", bbox_inches="tight", pad_inches=0.2, dpi=200)


# probability of default
# fig, ax = plt.subplots()
# for i in range(1,4):
#     plt.plot(x, tables[i]["prob_of_default"], color=line_colors[i], lw=2, label=str(i) + " vállalat")
# plt.ylabel("Csődvalószínűség", labelpad=10)
# plt.xlabel("Év")
# fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
# plt.show()
# plt.close('all')
#plt.savefig("plots/mc-pod.png", bbox_inches="tight", pad_inches=0.2, dpi=200)


# fcfe
fig, ax = plt.subplots()
for i in range(1,4):
    plt.plot(x, i*tables[i]["fcfe"], color=line_colors[i], lw=2, label=str(i) + " vállalat")
plt.ylabel("Összes realizált FCFE", labelpad=10)
plt.xlabel("Év")
fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
#plt.show()
#plt.close('all')
plt.savefig("plots/mc-fcfe.png", bbox_inches="tight", pad_inches=0.2, dpi=200)


# ts to fcff
# threshold a / realized a
fig, ax = plt.subplots()
for i in range(1,4):
    plt.plot(x, 100*tables[i]["ts"]/tables[i]["fcff"], color=line_colors[i], lw=2, label=str(i) + " vállalat")
plt.ylabel("Adópajzs / FCFF (%)", labelpad=10)
plt.xlabel("Év")
fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
#plt.show()
#plt.close('all')
plt.savefig("plots/mc-tsfcff.png", bbox_inches="tight", pad_inches=0.2, dpi=200)


## mc with shock - only one plot: the one with debt to asset and time of default
# read tables
tables_shock = {}
for i in range(1,4):
    tables_shock[i] = pd.read_excel("excels/" + str(i) + "-firms-mc-shock-agg.xlsx")

# plots
n = len(tables_shock[1].index)
x = range(1, n+1)
default = {}
for i in range(1,4):
    default[i] = tables_shock[i].loc[0, "time_of_default"]

# debt to asset ratio
fig, ax = plt.subplots()
for i in range(1,4):
    plt.plot(x, 100*tables_shock[i]["debt_to_asset"], color=line_colors[i], lw=2, label=str(i) + " vállalat")
    plt.axvline(default[i], color=line_colors[i], lw=1, linestyle='dashed')
plt.ylabel("Könyv szerinti Hitel/Eszköz (%)", labelpad=10)
plt.xlabel("Év")
plt.text(0.435, 1.05, "Csőd éve", transform=ax.transAxes, bbox=dict(facecolor='none', ls='--', edgecolor='black'))
fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
#plt.show()
#plt.close('all')
plt.savefig("plots/mc-shock-debt.png", bbox_inches="tight", pad_inches=0.2, dpi=200)


## scenario plots
scenarios = ["up-down", "down-up"]
for scenario in scenarios:
    if scenario == "up-down":
        name = "ud"
    else:
        name = "du"
    # read tables
    tables = {}
    for i in range(1, 4):
        tables[i] = pd.read_excel("excels/" + str(i) + "-firms-" + scenario + ".xlsx")

    # plots
    n = len(tables[1].index)
    x = range(1, n + 1)

    # closeness to threshold
    fig, ax = plt.subplots()
    for i in range(1, 4):
        plt.plot(x, 100*tables[i]["threshold_a_0"]/tables[1]["a_0"], color=line_colors[i], lw=2, label=str(i) + " vállalat")
    plt.ylabel("Küszöb A / A (%)", labelpad=10)
    plt.xlabel("Év")
    fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1))
    #plt.show()
    #plt.close('all')
    plt.savefig("plots/sc-" + name + "-threshold.png", bbox_inches="tight", pad_inches=0.2, dpi=200)

    # debt to asset ratio
    fig, ax1 = plt.subplots()
    for i in range(1, 4):
        plt.plot(x, 100 * tables[i]["debt_to_asset_0"], color=line_colors[i], lw=2, label=str(i) + " vállalat")
    ax1.set_ylabel("Könyv szerinti Hitel/Eszköz (%)", labelpad=10)
    plt.xlabel("Év")
    ax2 = ax1.twinx()
    ax2.plot(x, tables[1]['a_0'], lw=1, ls='dashed', color='grey', label="A (jobb tengely)")
    ax2.set_ylabel("Árfüggvény szint", labelpad=10)
    fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1))
    #plt.show()
    #plt.close('all')
    plt.savefig("plots/sc-" + name + "-debt.png", bbox_inches="tight", pad_inches=0.2, dpi=200)

    # production
    fig, ax1 = plt.subplots()
    for i in range(1, 4):
        plt.plot(x, tables[i]["prod_0"], color=line_colors[i], lw=2, label=str(i) + " vállalat")
    ax1.set_ylabel("A vállalatok össztermelése", labelpad=10)
    plt.xlabel("Év")
    ax2 = ax1.twinx()
    ax2.plot(x, tables[1]['a_0'], lw=1, ls='dashed', color='grey', label="A (jobb tengely)")
    ax2.set_ylabel("Árfüggvény szint", labelpad=10)
    fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1))
    #plt.show()
    #plt.close('all')
    plt.savefig("plots/sc-" + name + "-prod.png", bbox_inches="tight", pad_inches=0.2, dpi=200)

    # fcfe
    fig, ax1 = plt.subplots()
    for i in range(1, 4):
        plt.plot(x, i*tables[i]["fcfe_0"], color=line_colors[i], lw=2, label=str(i) + " vállalat")
    ax1.set_ylabel("Összes realizált FCFE", labelpad=10)
    plt.xlabel("Év")
    ax2 = ax1.twinx()
    ax2.plot(x, tables[1]['a_0'], lw=1, ls='dashed', color='grey', label="A (jobb tengely)")
    ax2.set_ylabel("Árfüggvény szint", labelpad=10)
    fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1))
    #plt.show()
    #plt.close('all')
    plt.savefig("plots/sc-" + name + "-fcfe.png", bbox_inches="tight", pad_inches=0.2, dpi=200)

    # ts to fcff
    fig, ax1 = plt.subplots()
    for i in range(1, 4):
        plt.plot(x, 100 * tables[i]["ts_0"] / tables[i]["fcff_0"], color=line_colors[i], lw=2, label=str(i) + " vállalat")
    ax1.set_ylabel("Adópajzs / FCFF (%)")
    plt.xlabel("Év")
    ax2 = ax1.twinx()
    ax2.plot(x, tables[1]['a_0'], lw=1, ls='dashed', color='grey', label="A")
    ax2.set_ylabel("A")
    fig.legend(loc="lower center", ncol=4)
    #plt.show()
    plt.close('all')
    plt.savefig("plots/sc-ud-tsfcff.png", bbox_inches="tight", pad_inches=0.2, dpi=200)