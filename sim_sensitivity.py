import functions as fn
import numpy as np
import realization_lib as re
import pandas as pd
import random
import time


start_time = time.time()
real_num = 100
real_years = 15

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

count=0
for num_firms in range(1,4):
    for param in list(changes.keys()):
        for val in changes[param]:
            count+=1
            random.seed(2023)
            memo = {}
            res_raw = pd.DataFrame()
            res_all = []
            for k in range(real_num):
                inp = fn.create_inp(num_firms)
                inp[param] = val
                inp['mach_price'] = inp['mach_life'] * inp['mach_cap'] * inp['cap_price']
                inp['maturity'] = inp['mach_life']
                x0_prod = (inp['a0'] - inp['c'] - inp['cap_price']) / ((num_firms + 1) * inp['b'])
                inp['x0'] = np.tile(np.array([x0_prod, max(inp['trend'],0), 0.5]), (num_firms, 1))
                res = re.realization(inp, real_years, [], memo, False)
                res_raw = re.append_res(res_raw, res, k)
                res_all.append(res)
                print(str(num_firms) + " firms, " + param + " = " + str(val) + ", " + str(k+1) + " sim")
            res_raw.to_csv("excels/sensitivity/" + str(num_firms) + "-firms-sens-" + param + "=" + str(val) + "-raw.csv")
            res_agg_data = {}
            for key in list(res_all[0].keys()):
                for k in range(real_num):
                    if k==0:
                        sims = res_all[k][key]
                    else:
                        sims = np.vstack((sims, res_all[k][key]))
                res_agg_data[key] = np.nanmean(sims, axis=0)
            res_agg = pd.DataFrame.from_dict(res_agg_data)
            res_agg.to_excel("excels/sensitivity/" + str(num_firms) + "-firms-sens-" + param + "=" + str(val) + "-agg.xlsx")
            print(str(count) + "/60 done")
print("Finished in " + str(time.time()-start_time) + " seconds")

