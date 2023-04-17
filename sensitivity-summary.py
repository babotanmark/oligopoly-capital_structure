import pandas as pd

df = pd.DataFrame()
changes={
    'trend': [-0.02, 0.02],
    'deviation': [0.02, 0.08],
    'b': [0.001, 0.1],
    'tax_rate': [0.1, 0.2],
    'mach_life': [1, 5],
    'mach_cap': [50, 200],
    'cap_price': [8, 12],
    'rd': [0.05, 0.15],
    'c': [60, 90],
    'a0': [100, 140]
}

## baseline
# read base tables
tables_base = {}
default_base = {}
debt_ratio_base = {}
for i in range(1,4):
    tables_base[i] = pd.read_excel("excels/" + str(i) + "-firms-mc-agg.xlsx")
    default_base[i] = tables_base[i].loc[0, "time_of_default"]
    debt_ratio_base[i] = tables_base[i]["debt_to_asset"].mean()
df['scenario'] = ['base']
for i in range(1,4):
    df['debt_ratio_' + str(i)] = ['{:.4f}'.format(round(debt_ratio_base[i],4))]
for i in range(1,4):
    df['default_' + str(i)] = ['{:.2f}'.format(round(default_base[i],2))]

for param in list(changes.keys()):
    for val in changes[param]:
        # read tables
        tables = {}
        default = {}
        debt_ratio = {}
        for i in range(1,4):
            tables[i] = pd.read_excel("excels/sensitivity/" + str(i) + "-firms-sens-" + param + "=" + str(val) + "-agg.xlsx")
            default[i] = tables[i].loc[0, "time_of_default"]
            debt_ratio[i] = tables[i]["debt_to_asset"].mean()
        df_append = pd.DataFrame()
        df_append['scenario'] = [param + "=" + str(val)]
        for i in range(1, 4):
            ch = 100*(debt_ratio[i] / debt_ratio_base[i] - 1)
            if ch<0:
                ret = '{:.2f}'.format(round(ch, 2)) + '%'
            else:
                ret = '+' + '{:.2f}'.format(round(ch, 2)) + '%'
            df_append['debt_ratio_' + str(i)] = [ret]
        for i in range(1, 4):
            ch = 100*(default[i] / default_base[i] - 1)
            if ch<0:
                ret = '{:.2f}'.format(round(ch, 2)) + '%'
            else:
                ret = '+' + '{:.2f}'.format(round(ch, 2)) + '%'
            df_append['default_' + str(i)] = [ret]
        df = pd.concat([df, df_append], ignore_index=True)

df.set_index('scenario', inplace=True)
df.to_excel("excels/sensitivity/sens-summary.xlsx")

