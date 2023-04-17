import numpy as np
import copy
import functions as fn
import pandas as pd
import random


def realization(inp, real_years, given_real, memo, test):
    """
    Creates the realization of the market
    It simulates real_years years
    If the firms go bankrupt, it stops
    """
    num_firms = inp['x0'].shape[0]
    res = {}
    res['a'] = np.empty((1, real_years))
    res['price'] = np.empty((1, real_years))
    res['investment'] = np.empty((num_firms, real_years))
    res['prod'] = np.empty((num_firms, real_years))
    res['capacity_usage'] = np.empty((num_firms, real_years))
    res['debt_pct_dec'] = np.empty((num_firms, real_years))
    res['debt_to_asset'] = np.empty((num_firms, real_years))
    res['prob_of_default'] = np.empty((num_firms, real_years))
    res['threshold_a'] = np.empty((num_firms, real_years))
    res['fcfe'] = np.empty((num_firms, real_years))
    res['fcff'] = np.empty((num_firms, real_years))
    res['ts'] = np.empty((num_firms, real_years))
    for key in res.keys():
        res[key].fill(np.nan)
    res['time_of_default'] = real_years

    for i in range(real_years):
        if test:
            dec = copy.deepcopy(inp['x0'])
        else:
            if num_firms == 1:
                if repr(inp) not in memo:
                    memo[repr(inp)] = np.array([fn.optim([], inp, 0, {})[0]])
                dec = memo[repr(inp)]
            else:
                if repr(inp) not in memo:
                    memo[repr(inp)] = fn.nash_equilibrium(inp)
                dec = memo[repr(inp)]
        res['prod'][:,i] = dec[:,0]
        res['debt_pct_dec'][:,i] = dec[:,2]
        if i == 0:
            hist_ocf_ts = fn.create_init_ocf_ts(res['prod'][:, i], res['debt_pct_dec'][:, i], inp['a0'], inp['b'],
                                                inp['c'], inp['f'], inp['tax_rate'], inp['rd'], inp['mach_price'],
                                                inp['mach_cap'], inp['mach_life'])
            inp['ocf0'] = hist_ocf_ts[0]
            inp['ts0'] = hist_ocf_ts[1]
            hist_dec = fn.create_history(inp['mach_life'], np.transpose([res['prod'][:, i]]),
                                         np.transpose([res['debt_pct_dec'][:, i]]))
            hist_data = fn.calc_from_init_history(hist_dec[0], hist_dec[1], inp['mach_cap'], inp['mach_life'],
                                                  inp['mach_price'], inp['maturity'], inp['rd'], inp['n'])
            inp['mach_at_hand0'] = hist_data[0]
            inp['disposal_inp'] = hist_data[1]
            inp['amort_inp'] = hist_data[2]
            inp['debt_payback_inp'] = hist_data[3]
            inp['interest_pay_inp'] = hist_data[4]
            inp['assets0'] = hist_data[5]
            inp['debt0'] = hist_data[6]

        inv_debt = fn.calc_invest_debt(np.transpose([res['prod'][:, i]]), dec[:, 1],
                                          np.transpose([res['debt_pct_dec'][:, i]]), inp['mach_cap'], inp['mach_life'],
                                          inp['mach_price'], inp['maturity'], inp['rd'], inp['mach_at_hand0'],
                                          inp['disposal_inp'], inp['amort_inp'], inp['debt_payback_inp'],
                                          inp['interest_pay_inp'])
        mach_at_hand = inv_debt[7][:]
        res['capacity_usage'][:,i] = res['prod'][:,i] / (mach_at_hand * inp['mach_cap'])
        inv = inv_debt[0]
        res['investment'][:,i] = inv
        debt_borr = inv_debt[1]
        assets = inp['assets0'] + inv - inp['amort0']
        debt = inp['debt0'] + debt_borr - inp['debt_payback0']
        res['debt_to_asset'][:,i] = debt / assets

        fcff = inp['ocf0'] - inv
        d_debt = debt_borr - inp['debt_payback0']
        if i == 0:
            d_debt = np.zeros(num_firms)  # feltétel, hogy eddig egyensúlyban volt
            inp['interest_pay0'] = debt * inp['rd']
        res['fcfe'][:,i] = fcff + inp['ts0'] + d_debt - inp['interest_pay0']
        res['fcff'][:,i] = fcff
        res['ts'][:,i] = inp['ts0']
        if np.any(res['fcfe'][:,i] < 0):
            res['time_of_default'] = i
            break

        # new year
        interest_pay = inv_debt[6][:,0]
        amort = inv_debt[3][:,0]
        debt_payback = inv_debt[5][:,0]
        for j in range(num_firms):
            prod = np.array([res['prod'][:,i][j]])
            prod_dec_others = res['prod'][:,i][np.arange(len(res['prod'][:,i]))!=j]
            if not np.any(prod_dec_others):
                prod_others = np.zeros(1)
            else:
                prod_others = np.array([np.sum(prod_dec_others)])
            prob_of_default = 1 - fn.calc_fcff_ts(prod, prod_others, inp, inp['a0'], np.array([interest_pay[j]]),
                                                  np.array([inv[j]]), np.array([amort[j]]), np.array([debt_borr[j]]),
                                                  np.array([debt_payback[j]]))[2]
            res['prob_of_default'][j,i] = prob_of_default
            threshold_inc = fn.threshold_income(prod*inp['c'], np.array([inp['f']]), np.array([interest_pay[j]]),
                                                np.array([inv[j]]), np.array([amort[j]]), np.array([debt_borr[j]]),
                                                np.array([debt_payback[j]]), inp['tax_rate'])
            res['threshold_a'][j,i] = fn.calc_threshold_a(prod, prod_others, threshold_inc, inp['b'])
        if not given_real:
            rnumber = random.uniform(0,1)
            rnumber_shock = random.uniform(0,1)
        else:
            rnumber = given_real[i]
            rnumber_shock = 1
        if rnumber_shock < inp['prob_shock']:
            res['a'][0, i] = inp['a0'] * (1 + inp['mult_shock']*(inp['trend']-inp['deviation']))
        else:
            if rnumber < inp['p'][0]:
                res['a'][0,i] = inp['a0'] * (1 + fn.get_change(inp['trend'], inp['deviation'])[0])
            elif rnumber < inp['p'][0] + inp['p'][1]:
                res['a'][0,i] = inp['a0'] * (1 + fn.get_change(inp['trend'], inp['deviation'])[1])
            else:
                res['a'][0,i] = inp['a0'] * (1 + fn.get_change(inp['trend'], inp['deviation'])[2])
        res['price'][0,i] = res['a'][0,i] - inp['b'] * np.sum(res['prod'][:,i])
        # ocf
        income = res['prod'][:, i] * res['price'][0,i]
        var_cost = res['prod'][:, i] * inp['c']
        fix_cost = inp['f'] * np.ones(num_firms)
        interest_pay = inv_debt[6][:,0]
        amort = inv_debt[3][:,0]
        ebit = income - var_cost - fix_cost - amort
        tax_on_ebit = np.maximum(ebit * inp['tax_rate'], np.zeros(num_firms))
        inp['ocf0'] = ebit - tax_on_ebit + amort
        tax = np.maximum((ebit - interest_pay) * inp['tax_rate'], np.zeros(num_firms))
        inp['ts0'] = tax_on_ebit - tax

        # overwrite the rest of the input variables
        inp['a0'] = res['a'][0,i]
        inp['mach_at_hand0'] = mach_at_hand
        inp['disposal_inp'] = np.hstack((inv_debt[8][:,1:], np.transpose([np.zeros(num_firms)])))
        inp['amort_inp'] = np.hstack((inv_debt[3][:,1:], np.transpose([np.zeros(num_firms)])))
        inp['debt_payback_inp'] = np.hstack((inv_debt[5][:,1:], np.transpose([np.zeros(num_firms)])))
        inp['interest_pay_inp'] = np.hstack((inv_debt[6][:,1:], np.transpose([np.zeros(num_firms)])))
        inp['assets0'] = assets[:]
        inp['debt0'] = debt[:]
        inp['amort0'] = amort[:]
        inp['debt_payback0'] = debt_payback[:]
        inp['interest_pay0'] = interest_pay[:]
        inp['first_run'] = False
        inp['x0'] = copy.deepcopy(dec)

    res['a'] = np.squeeze(res['a'])
    res['price'] = np.squeeze(res['price'])
    res['prod'] = np.sum(res['prod'], axis=0)
    res['investment'] = np.mean(res['investment'], axis=0)
    res['capacity_usage'] = np.mean(res['capacity_usage'], axis=0)
    res['debt_pct_dec'] = np.mean(res['debt_pct_dec'], axis=0)
    res['debt_to_asset'] = np.mean(res['debt_to_asset'], axis=0)
    res['prob_of_default'] = np.mean(res['prob_of_default'], axis=0)
    res['threshold_a'] = np.mean(res['threshold_a'], axis=0)
    res['fcfe'] = np.sum(res['fcfe'], axis=0)
    res['fcff'] = np.mean(res['fcff'], axis=0)
    res['ts'] = np.mean(res['ts'], axis=0)
    res['time_of_default'] = res['time_of_default']*np.ones(len(res['a']))
    return res


def append_res(res0, res, k):
    res_append = pd.DataFrame(data={
        'a_' + str(k): res['a'],
        'threshold_a_' + str(k): res['threshold_a'],
        'price_' + str(k): res['price'],
        'prod_' + str(k): res['prod'],
        'investment_' + str(k): res['investment'],
        'capacity_usage_' + str(k): res['capacity_usage'],
        'debt_pct_dec_' + str(k): res['debt_pct_dec'],
        'debt_to_asset_' + str(k): res['debt_to_asset'],
        'prob_of_default_' + str(k): res['prob_of_default'],
        'fcff_' + str(k): res['fcff'],
        'ts_' + str(k): res['ts'],
        'fcfe_' + str(k): res['fcfe'],
        'time_of_default_' + str(k): res['time_of_default']
    })
    if res0.empty:
        res_res = res_append
    else:
        res_res = pd.concat([res0, res_append], axis=1)
    return res_res
