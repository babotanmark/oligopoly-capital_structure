import functions as fn
import numpy as np
import realization_lib as re
import pandas as pd
import random
import time


start_time = time.time()
real_num = 100
real_years = 15

for num_firms in range(1,4):
    random.seed(2023)
    memo = {}
    res_raw = pd.DataFrame()
    res_all = []
    for k in range(real_num):
        inp = fn.create_inp(num_firms)
        res = re.realization(inp, real_years, [], memo, False)
        res_raw = re.append_res(res_raw, res, k)
        res_all.append(res)
        print(str(num_firms) + " firms, " + str(k+1) + " sim")
    res_raw.to_csv("excels/" + str(num_firms) + "-firms-mc-raw.csv")
    res_agg_data = {}
    for key in list(res_all[0].keys()):
        for k in range(real_num):
            if k==0:
                sims = res_all[k][key]
            else:
                sims = np.vstack((sims, res_all[k][key]))
        res_agg_data[key] = np.nanmean(sims, axis=0)
    res_agg = pd.DataFrame.from_dict(res_agg_data)
    res_agg.to_excel("excels/" + str(num_firms) + "-firms-mc-agg.xlsx")
    print(str(num_firms) + " aggregation done")
print("Finished in " + str(time.time()-start_time) + " seconds")

