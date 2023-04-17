import functions as fn
import realization_lib as re
import pandas as pd
import random


real_years = 15
reals = {
    'up-down': [0.5, 0.5, 0.5, 0.5, 0.5, 0, 0.5, 0, 0.5, 0, 1, 0.5, 1, 0.5, 1],
    'down-up': [0.5, 0.5, 0.5, 0.5, 0.5, 1, 0.5, 1, 0.5, 1, 0, 0.5, 0, 0.5, 0]
}

for num_firms in range(1,4):
    memo = {}
    for scenario in list(reals.keys()):
        real = reals[scenario]
        res_raw = pd.DataFrame()
        inp = fn.create_inp(num_firms)
        res = re.realization(inp, real_years, real, memo, False)
        res_raw = re.append_res(res_raw, res, 0)
        res_raw.to_excel("excels/" + str(num_firms) + "-firms-" + str(scenario) + ".xlsx")
        print(str(num_firms) + " done")
