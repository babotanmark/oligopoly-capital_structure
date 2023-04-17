import numpy as np
from scipy import optimize
import copy
import time


# num_firms = 1
def create_inp(num_firms):
    """Creates the default input parameters needed for the simulation"""
    inp = {}
    # the ones that stays constant over time
    inp['prob_shock'] = 0
    inp['mult_shock'] = 3
    inp['n'] = 6
    inp['trend'] = 0.00
    inp['deviation'] = 0.05
    inp['p'] = np.array([1/3, 1/3, 1/3])
    inp['b'] = 0.01
    inp['ra'] = 0.12
    inp['tax_rate'] = 0.15
    inp['mach_life'] = 3
    inp['mach_cap'] = 100
    inp['cap_price'] = 10
    inp['mach_price'] = inp['mach_life'] * inp['mach_cap'] * inp['cap_price']
    inp['maturity'] = inp['mach_life']
    inp['rd'] = 0.1
    inp['c'] = 75
    inp['f'] = 0
    # the ones that changes over time
    inp['a0'] = 120
    inp['mach_at_hand0'] = np.zeros(num_firms)
    inp['disposal_inp'] = np.empty((num_firms, inp['n']))
    inp['amort_inp'] = np.empty((num_firms, inp['n']))
    inp['debt_payback_inp'] = np.empty((num_firms, inp['n']))
    inp['interest_pay_inp'] = np.empty((num_firms, inp['n']))
    inp['assets0'] = np.zeros(num_firms)
    inp['debt0'] = np.zeros(num_firms)
    inp['amort0'] = np.zeros(num_firms)
    inp['debt_payback0'] = np.zeros(num_firms)
    inp['ocf0'] = np.zeros(num_firms)
    inp['ts0'] = np.zeros(num_firms)
    inp['interest_pay0'] = np.zeros(num_firms)
    inp['first_run'] = True
    x0_prod = (inp['a0'] - inp['c'] - inp['cap_price'])/((num_firms+1)*inp['b'])
    #x0_prod = 1000/num_firms
    inp['x0'] = np.tile(np.array([x0_prod, max(inp['trend'],0), 0.5]), (num_firms, 1))
    return inp


def get_prod_vector(prod_level, g, n):
    """
    Calculates the production vector(s) from prod_level and g
    :param prod_level: (k,1) 2-d array
    :param g: (k,1) 2-d array
    :param n: integer
    :return: (k,n) 2-d array
    """
    res = np.empty((prod_level.shape[0], n))
    for i in range(n):
        res[:, i] = prod_level * (1 + g) ** i
    return res


def create_history(mach_life, prod_level, debt_pct):
    """
    At first year, the creates the history according to the first year's decision
    :param mach_life: integer
    :param prod_level: (k,1) 2-d array where
    :param debt_pct: (k,1) 2-d array where
    :return:
        0: prod_hist: (k,mach_life-1) 2-d array
        1: debt_pct_hist: (k,mach_life-1) 2-d array
    """
    per_year = prod_level / mach_life
    prod_hist = per_year * np.arange(1, mach_life)
    debt_pct_hist = debt_pct * np.ones(mach_life - 1)
    return [prod_hist, debt_pct_hist]



def create_init_ocf_ts(prod_level, debt_pct, a, b, c, fix, tax_rate, rd, mach_price, mach_cap, mach_life):
    """
    Calculates the initial operative cash flow and tax shield for the first year, when there is no operation yet before decision
    It assumes that the production level is already stable at the level of input
    :param prod_level: float or 1-d array
    :param debt_pct: float or 1-d array
    :param a: float
    :param b: float
    :param c: float
    :param fix: float
    :param tax_rate: float
    :param rd: float
    :param mach_price: float
    :param mach_cap: float (integer)
    :param mach_life: float (integer)
    :return:
        0: ocf_res: float or 1-d array
        1: ts_res: float or 1-d array
    """
    if np.ndim(prod_level) > 0:
        q = np.sum(prod_level)
    else:
        q = prod_level
    ic = np.ceil(prod_level / mach_cap) * mach_price
    amort = ic / mach_life
    ocf_res = ((a - b * q - c) * prod_level - fix - amort) * (1 - tax_rate) + amort
    debt = ic * debt_pct
    ts_res = debt * rd * tax_rate
    return [ocf_res, ts_res]


def calc_from_init_history(dec_hist, debt_pct_hist, mach_cap, mach_life, mach_price, maturity, interest_rate, n):
    """
    Calculate the fix attributes from history data (at first year)
    :param dec_hist: (k,m) 2-d array
    :param debt_pct_hist: (k,m) 2-d array
    the rest of parameters is float (or integer)
    :return:
        0: num of machines already at hand in 0. year: (k,) 1-d array)
        1: disposal vectors (input for the future from year 0 to n-1): (k,n) 2-d array
        2: amortisation vectors (from year 1 to n): (k,n) 2-d array
        3: debt payback vectors: (k,n) 2-d array
        4: interest payment vectors: (k,n) 2-d array
        5: initial assets (=invested capital) in THE BEGINNING of year 0: (k,) 1-d array
        6: initial debt in THE BEGINNING of year 0: (k,) 1-d array
    """
    necessary_machines = np.ceil(dec_hist / mach_cap)
    k = dec_hist.shape[0]
    if necessary_machines.size==0:
        mach_at_hand0_res = np.zeros(k)
    else:
        mach_at_hand0_res = necessary_machines[:, -1]
    mach_at_hand = np.hstack((np.transpose([np.zeros(k)]), necessary_machines[:, :-1]))
    mach_bought = np.maximum(necessary_machines - mach_at_hand, np.zeros(necessary_machines.shape))
    max_index = min(mach_life, n)
    disposal_res = np.zeros((k, n))
    disposal_res[:, 1:max_index] = mach_bought
    investment = mach_bought * mach_price
    amort_assist = investment / mach_life
    amort_res = np.zeros((k, n))
    for i in range(max_index):
        amort_res[:, i] = np.sum(amort_assist[:, i:], axis=1)
    debt_borr = investment * debt_pct_hist
    max_index = min(maturity, n)
    debt_payback_res = np.zeros((k, n))
    debt_payback_res[:, :max_index-1] = debt_borr
    yearly_interest = debt_borr * interest_rate
    interest_pay_res = np.zeros((k, n))
    for i in range(max_index):
        interest_pay_res[:, i] = np.sum(yearly_interest[:, i:], axis=1)
    if necessary_machines.size==0:
        ic0_res = np.zeros(k)
        debt0_res = np.zeros(k)
    else:
        ic0_res = necessary_machines[:, -1] * mach_price
        debt0_res = ic0_res * debt_pct_hist[:, -1]
    return [mach_at_hand0_res, disposal_res, amort_res, debt_payback_res, interest_pay_res, ic0_res, debt0_res]


# from here they are called each year
def calc_invest_debt(prod, g, debt_pct, mach_cap, mach_life, mach_price, maturity, interest_rate,
                     mach_at_hand0, disposal_inp, amort_inp, debt_payback_inp, interest_pay_inp):
    """
    Calculates investment and debt borrowing values
    :param prod: (k,1) or (k,n) 2-d array
    :param g: float or (k,) 1-d array
    :param debt_pct: float or (k,) 1-d array
    :param mach_cap: float
    :param mach_life: float
    :param mach_price: float
    :param maturity: float
    :param interest_rate: float
    :param mach_at_hand0: (k,) 1-d array
    :param disposal_inp: (k,nn) 2-d array
    :param amort_inp: (k,nn) 2-d array
    :param debt_payback_inp: (k,nn) 2-d array
    :param interest_pay_inp: (k,nn) 2-d array
    :return:
        0: inv0_res: (k,) 1-d array
        1: debt_borr0_res: (k,1) 1-d array
        2: inv_res: (n,) 1-d array or []
        3: amort_res: (k,nn) 2-d array
        4: debt_borr_res: (n,) 1-d array or []
        5: debt_payback_res: (k,nn) 2-d array
        6: interest_pay_res: (k,nn) 2-d array
        7: mach_at_hand0_res: (k,) 1-d array
        8: disposal: (k,nn) 2-d array
    """
    k, n = prod.shape
    nn = disposal_inp.shape[1]
    necessary_mach = np.ceil(prod / mach_cap)  # (k,n) # np.squeeze(np.ceil(prod / mach_cap), axis=1)
    disposal = copy.deepcopy(disposal_inp)
    mach_at_hand = mach_at_hand0[:]
    mach_bought = np.zeros((k, n))
    for i in range(n):
        mach_bought[:, i] = np.maximum(necessary_mach[:, i] - mach_at_hand + disposal[:, i], np.zeros(k))
        mach_at_hand = mach_at_hand - disposal[:, i] + mach_bought[:, i]
        if i == 0:
            mach_at_hand0_res = mach_at_hand[:]
        if i + mach_life < nn:
            disposal[:, i + mach_life] += mach_bought[:, i]
    invest = mach_bought * mach_price
    amort_res = copy.deepcopy(amort_inp)
    for i in range(n):
        yearly_amort = invest[:, i] / mach_life
        max_index = min(i + mach_life, nn)
        for j in range(i, max_index):
            amort_res[:, j] += yearly_amort
    debt_borr = (invest.T * debt_pct).T
    debt_payback_res = copy.deepcopy(debt_payback_inp)
    max_index = min(n, nn-maturity+1)
    for i in range(max_index):
        debt_payback_res[:, maturity - 1 + i] += debt_borr[:,i]
    #debt_payback_res[:, maturity - 1:] += debt_borr[:, :nn - maturity + 1]
    interest_pay_res = copy.deepcopy(interest_pay_inp)
    for i in range(n):
        yearly_interest = debt_borr[:, i] * interest_rate
        max_index = min(i + maturity, nn)
        for j in range(i, max_index):
            interest_pay_res[:, j] += yearly_interest
    inv0_res = invest[:, 0]
    if n > 1:
        inv_res = np.append(invest[:, 1:], invest[:, -1] * (1 + g))
    else:
        inv_res = []
    debt_borr0_res = debt_borr[:, 0]
    if n > 1:
        debt_borr_res = np.append(debt_borr[:, 1:], debt_borr[:, -1] * (1 + g))
    else:
        debt_borr_res = []
    return [inv0_res, debt_borr0_res, inv_res, amort_res, debt_borr_res, debt_payback_res, interest_pay_res,
            mach_at_hand0_res, disposal]


# NON-USED
def balance_sheet(assets0, debt0, inv, amort, debt_borr, debt_payback):
    n = len(inv)
    eszkoz_res = np.zeros(n)
    debt_res = np.zeros(n)
    for i in range(n):
        if i == 0:
            eszkoz_res[i] = assets0 + inv[i] - amort[i]
            debt_res[i] = debt0 + debt_borr[i] - debt_payback[i]
        else:
            eszkoz_res[i] = eszkoz_res[i - 1] + inv[i] - amort[i]
            debt_res[i] = debt_res[i - 1] + debt_borr[i] - debt_payback[i]
    return np.array([eszkoz_res, debt_res])


def threshold_income(var_cost, fix_cost, interest_pay, inv, amort, debt_borr, debt_payback, tax_rate):
    inc = ((1 - tax_rate) * (var_cost + fix_cost + interest_pay) - amort * tax_rate + inv - debt_borr + debt_payback) \
          / (1 - tax_rate)
    need_change = inc - var_cost - fix_cost - amort - interest_pay < 0
    indices = np.where(need_change)[0]
    inc[indices] = var_cost[indices] + fix_cost[indices] + interest_pay[indices] + inv[indices] \
                   - debt_borr[indices] + debt_payback[indices]
    return inc


def get_change(trend, deviation):
    change = np.empty(3)
    change[1] = trend
    change[0] = trend + deviation
    change[2] = trend - deviation
    return change


def trinomtree(a0, change, p, threshold):
    n = len(threshold)
    nn = 3 ** n
    res = [np.zeros((nn, n + 1)) for _ in range(3)]
    # 0: trinom tree values
    # 1: trinom tree probabilities
    # 2: default (0/1)
    res[0][(nn - 1) // 2, 0] = a0
    res[1][(nn - 1) // 2, 0] = 1
    prev_indeces = (nn - 1) // 2 * np.ones(1, dtype=int)
    for j in range(n):
        indeces = np.zeros(3 ** (j + 1), dtype=int)
        for i in range(3 ** j):
            nnn = nn // 3 ** (j + 1)
            indeces[3 * i] = prev_indeces[i] - nnn
            indeces[3 * i + 1] = prev_indeces[i]
            indeces[3 * i + 2] = prev_indeces[i] + nnn
            for k in range(3):
                res[0][indeces[3 * i + k], j + 1] = res[0][prev_indeces[i], j] * (1 + change[k])
                res[1][indeces[3 * i + k], j + 1] = res[1][prev_indeces[i], j] * p[k]
                if res[2][prev_indeces[i], j] == 1:
                    res[2][indeces[3 * i + k], j + 1] = 1
                else:
                    if res[0][indeces[3 * i + k], j + 1] < threshold[j]:
                        res[2][indeces[3 * i + k], j + 1] = 1
        prev_indeces = np.zeros(3 ** (j + 1), dtype=int)
        prev_indeces = indeces[:]
    return res


def evaluate_on_trinom_tree(a0, change, p, threshold):
    n = len(threshold)
    res = np.zeros((2, n))
    # 0: probability of default until the given year
    # 1: conditional expected A in the given period, with condition that the firm is still alive
    trinomtree_res = trinomtree(a0, change, p, threshold)
    trinomtree_value = trinomtree_res[0]
    trinomtree_prob = trinomtree_res[1]
    trinomtree_default = trinomtree_res[2]
    for j in range(n):
        res[0, j] = min(np.matmul(trinomtree_prob[:, j + 1], np.transpose(trinomtree_default[:, j + 1])), 1)  # somehow sometimes it is slightly above 1
        if res[0, j] < 1:
            res[1, j] = np.sum(trinomtree_value[:, j + 1] * trinomtree_prob[:, j + 1] * (1 - trinomtree_default[:, j + 1])) / (
                    1 - res[0, j])
    return res


def calc_threshold_a(term, term_others, threshold_inc, b):
    res = threshold_inc / term + b * (term + term_others)
    return res


def calc_price(prod, prod_others, a, b):
    res = a - b * (prod + prod_others)
    return res


def calc_fcff_ts(prod, prod_others, inp, a_prev, interest_pay, inv, amort, debt_borr, debt_payback):
    """
    :param prod: (n,) 1-d array
    :param prod_others: (n,) 1-d array (sum of prods)
    :param inp: dictionary
    :param a_prev: float
    :param interest_pay: (n,) 1-d array
    :param inv: (n,) 1-d array
    :param amort: (n,) 1-d array
    :param debt_borr: (n,) 1-d array
    :param debt_payback: (n,) 1-d array
    :return:
        0: fcff: (n,) 1-d array
        1: tax shield: (n,1) 1-d array
        2: survival probabilities for each year: (n,) 1-d array
    """
    n = len(prod)
    var_cost = np.multiply(inp['c'], prod)
    fix_cost = inp['f'] * np.ones(n)
    threshold_inc = threshold_income(var_cost, fix_cost, interest_pay, inv, amort, debt_borr, debt_payback,
                                     inp['tax_rate'])
    threshold_a = calc_threshold_a(prod, prod_others, threshold_inc, inp['b'])
    eval_tr = evaluate_on_trinom_tree(a_prev, get_change(inp['trend'], inp['deviation']), inp['p'], threshold_a)
    default_prob = eval_tr[0, :]
    survival_prob = 1 - default_prob
    expected_a = eval_tr[1, :]
    expected_price = calc_price(prod, prod_others, expected_a, inp['b'])
    # income statement
    inc = expected_price * prod
    ebitda = inc - var_cost - fix_cost
    ebit = ebitda - amort
    ebt = ebit - interest_pay
    tax = np.maximum(ebt * inp['tax_rate'], np.zeros(n))
    # clean_result = ebt - tax
    # cf
    tax_on_ebit = np.maximum(ebit * inp['tax_rate'], np.zeros(n))
    noplat = ebit - tax_on_ebit
    ocf = noplat + amort  # brutto cf = operativ cf
    fcff = ocf - inv
    tax_shield = tax_on_ebit - tax
    return [fcff, tax_shield, survival_prob]


def calc_internal_value(ocf0, inv0, ts0, fcff, ts, surv_val, g_perp, ra):
    fcff0 = ocf0 - inv0
    n = len(fcff)
    res = fcff0 + ts0
    for i in range(n):
        res += (fcff[i] + ts[i]) * surv_val[i] / (1 + ra) ** (i + 1)
    # terminal value
    tv = (1+g_perp)*((fcff[-1] + ts[-1]) * surv_val[-1] / (1 + ra) ** n) / (ra-g_perp)
    res += tv
    return res


def calc_added_value(dec, prod_others, inp, which_one):
    """
    Calculates the added value if the firms decision is dec, and others' decision is dec_others
    :param dec: 1-d array
    :param prod_others: (n,) 1-d array (sum of prods)
    :param inp: dictionary
    :param which_one: integer
    :return: float
    """
    prod_dec = dec[0]
    g_dec = dec[1]
    debt_pct_dec = dec[2]
    prod = np.squeeze(get_prod_vector(np.array([prod_dec]), np.array([g_dec]), inp['n']))

    if inp['first_run']:
        hist = create_history(inp['mach_life'], np.array([prod_dec]), np.array([debt_pct_dec]))
        prod_hist = np.array([hist[0]])
        debt_pct_hist = np.array([hist[1]])
        ocf0_ts0 = create_init_ocf_ts(prod_dec, debt_pct_dec, inp['a0'], inp['b'], inp['c'], inp['f'], inp['tax_rate'],
                                      inp['rd'], inp['mach_price'], inp['mach_cap'], inp['mach_life'])
        inp['ocf0'][which_one] = ocf0_ts0[0]
        inp['ts0'][which_one] = ocf0_ts0[1]
        init_inp = calc_from_init_history(prod_hist, debt_pct_hist, inp['mach_cap'], inp['mach_life'],
                                          inp['mach_price'], inp['maturity'], inp['rd'], inp['n'])
        inp['mach_at_hand0'][which_one] = init_inp[0]
        inp['disposal_inp'][which_one, :] = init_inp[1]
        inp['amort_inp'][which_one, :] = init_inp[2]
        inp['debt_payback_inp'][which_one, :] = init_inp[3]
        inp['interest_pay_inp'][which_one, :] = init_inp[4]
        inp['assets0'][which_one] = init_inp[5]
        inp['debt0'][which_one] = init_inp[6]
        inp['amort0'][which_one] = 0
        inp['debt_payback0'][which_one] = 0

    res1 = calc_invest_debt(np.array([prod]), g_dec, debt_pct_dec, inp['mach_cap'], inp['mach_life'], inp['mach_price'],
                            inp['maturity'], inp['rd'], np.array([inp['mach_at_hand0'][which_one]]),
                            np.array([inp['disposal_inp'][which_one, :]]), np.array([inp['amort_inp'][which_one, :]]),
                            np.array([inp['debt_payback_inp'][which_one, :]]),
                            np.array([inp['interest_pay_inp'][which_one, :]]))
    inv0 = res1[0].item()
    # debt_borr0 = res1[1]
    ic = inp['assets0'][which_one] + inv0 - inp['amort0'][which_one]
    # ic = balance_sheet(inp['assets0'][which_one], inp['debt0'][which_one], inv0, [inp['amort0'][which_one]], debt_borr0,
    #                    [inp['debt_payback0'][which_one]])[0]
    inv = res1[2]
    amort = np.squeeze(res1[3])
    debt_borr = res1[4]
    debt_payback = np.squeeze(res1[5])
    interest_pay = np.squeeze(res1[6])
    res2 = calc_fcff_ts(prod, prod_others, inp, inp['a0'], interest_pay, inv, amort, debt_borr, debt_payback)
    fcff = res2[0]
    tax_shield = res2[1]
    surv_prob = res2[2]
    internal_value = calc_internal_value(inp['ocf0'][which_one], inv0, inp['ts0'][which_one], fcff, tax_shield,
                                         surv_prob, inp['trend'], inp['ra'])
    added_value = internal_value - ic
    return -added_value  # minus because of optimization


def optim(dec_others, inp, which_one, memo_opt):
    """
    Calculates the optimal decision vector taking the others' decisions as constant
    :param dec_others: (k-1,3) 2-d array or empty array, where k-1 is the number of other firms
    :param inp: dictionary
    :param which_one: integer
    :return: 1-d array of optimal decision vector
    """
    if not np.any(dec_others):  # if empty
        prod_others = np.zeros(inp['n'])
    else:
        prod_dec_others = dec_others[:, 0]
        g_dec_others = dec_others[:, 1]
        prod_others = np.sum(get_prod_vector(prod_dec_others, g_dec_others, inp['n']), axis=0)
    case = repr(prod_others)
    if case not in memo_opt:
        memo_opt[case] = optimize.differential_evolution(calc_added_value, ((1, inp['a0'] / inp['b']), (0, 1), (0, 1)),
                                                         args=(prod_others, inp, which_one),
                                                         seed=2023,
                                                         init='sobol',
                                                         x0=inp['x0'][which_one, :],
                                                         popsize=1,
                                                         strategy='best1bin',
                                                         tol=0.01).x
    return [memo_opt[case], memo_opt]


def reaction_vector(dec_all, inp, memo_opt):
    """
    Calculates the optimal decision for all the firms, taking the other firms decision as constant
    Returns the difference between the vectors of optimal decisions and the input decision vectors
    We are in the Nash-equilibrium if the difference is 0
    (No firms want to change their decision)
    """
    # fsolve calls this function - it is working with 1-d arrays so first we have to convert back to 2-d array
    k = len(dec_all) // 3
    dec_all = np.reshape(dec_all, (k, 3))
    sh = dec_all.shape
    res = np.empty(sh)
    for i in range(sh[0]):
        dec_others = np.delete(dec_all, i, axis=0)
        opt = optim(dec_others, inp, i, memo_opt)
        res[i, :] = opt[0]
        memo_opt = opt[1]
    # we have to give back a 1-d array to fsolve
    res_shaped = dec_all - res
    res_shaped = np.squeeze(np.reshape(res_shaped, (1, k * 3)))
    return res_shaped


def nash_equilibrium(inp):
    """
    Calculates where the reaction_vector will be 0
    So where the Nash-equilibrium is
    """
    memo_opt = {}
    orig_shape = inp['x0'].shape  # fsolve creates 1d-array from input and returns 1d-array
    x0 = inp['x0'].flatten()
    res = optimize.root(reaction_vector, x0, args=(inp, memo_opt), method='df-sane', options={'maxfev': 10}, tol=1).x
    res = np.reshape(res, orig_shape)
    res[:,0] = np.clip(res[:,0], 1, inp['a0']/inp['b'])
    res[:,1:3] = np.clip(res[:,1:3], 0, 1)
    return res
