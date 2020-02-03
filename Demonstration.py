#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 18:56:52 2019

@author: nana
"""

import numpy as np
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import bt
import matplotlib.pyplot as plt
import tkinter as tk
import plotly.graph_obkects as go


root = tk.Tk()
   
categories = pd.read_csv('ModelingData.csv', index_col = "Categories")
tickersCount=[]
tickersCount.append(categories.iloc[0])
print(len(tickersCount))
tickersInd=[]
tickersCoun=[]
tickersCap=[]
startDate=[]
endDate=[]

Tech = tk.IntVar()
ConCy = tk.IntVar()
Ind = tk.IntVar()
Energy = tk.IntVar() 
Comm = tk.IntVar()
Health = tk.IntVar()

usa = tk.IntVar()
ger = tk.IntVar()
chn = tk.IntVar()
ire = tk.IntVar()
nl = tk.IntVar()
jpn = tk.IntVar()
india = tk.IntVar()
isr = tk.IntVar()
ca = tk.IntVar()

lc = tk.IntVar()
mc = tk.IntVar()
sc = tk.IntVar()

before = tk.IntVar()
during = tk.IntVar()
after = tk.IntVar()

tk.Label(root, text="Industry: ").grid(row=0, column=0)
tk.Checkbutton(root, text="Tech", variable=Tech).grid(row=1, column=1)
tk.Checkbutton(root, text="Consumer Cyclical", variable=ConCy).grid(row=1, column=2)
tk.Checkbutton(root, text="Industrials", variable=Ind).grid(row=1, column=3)
tk.Checkbutton(root, text="Energy", variable=Energy).grid(row=1, column=4)
tk.Checkbutton(root, text="Communication Services", variable=Comm).grid(row=1, column=5)
tk.Checkbutton(root, text="Healthcare", variable=Health).grid(row=1, column=6)

tk.Label(root, text="Country: ").grid(row=2, column=0)
tk.Checkbutton(root, text="USA", variable=usa).grid(row=3, column=1)
tk.Checkbutton(root, text="Germany", variable=ger).grid(row=3, column=2)
tk.Checkbutton(root, text="China", variable=chn).grid(row=3, column=3)
tk.Checkbutton(root, text="Ireland", variable=ire).grid(row=3, column=4)
tk.Checkbutton(root, text="Netherlands", variable=nl).grid(row=3, column=5)
tk.Checkbutton(root, text="Japan", variable=jpn).grid(row=3, column=6)
tk.Checkbutton(root, text="India", variable=india).grid(row=3, column=7)
tk.Checkbutton(root, text="Israel", variable=isr).grid(row=3, column=8)
tk.Checkbutton(root, text="Canada", variable=ca).grid(row=3, column=9)

tk.Label(root, text="Market Cap: ").grid(row=4, column=0)
tk.Checkbutton(root, text="Large Cap", variable=lc).grid(row=5, column=1)
tk.Checkbutton(root, text="Medium Cap", variable=mc).grid(row=5, column=2)
tk.Checkbutton(root, text="Small Cap", variable=sc).grid(row=5, column=3)

tk.Label(root, text="Time Horizon (Relative to Financial Crisis): ").grid(row=6, column=0)
tk.Checkbutton(root, text="Before", variable=before).grid(row=7, column=1)
tk.Checkbutton(root, text="During", variable=during).grid(row=7, column=2)
tk.Checkbutton(root, text="After", variable=after).grid(row=7, column=3)


def csvedit():
        if Tech.get() == 1 :
            for i in range(1,135):
                if categories.iat[2, i] == "Tech":
                    if categories.iat[0,i] not in tickersInd :
                        tickersInd.append(categories.iat[0,i])
                        print(tickersInd)

        if ConCy.get() == 1 :
            for i in range(1,135):
                if categories.iat[2, i] == "Consumer Cyclical":
                    if categories.iat[0,i] not in tickersInd :
                        tickersInd.append(categories.iat[0,i])
                        print(tickersInd)

        if Ind.get() == 1 :
            for i in range(1,135):
                if categories.iat[2, i] == "Industrials":
                    if categories.iat[0,i] not in tickersInd :
                        tickersInd.append(categories.iat[0,i])
                        print(tickersInd)

        if Energy.get() == 1 :
            for i in range(1,135):
                if categories.iat[2, i] == "Energy":
                    if categories.iat[0,i] not in tickersInd :
                        tickersInd.append(categories.iat[0,i])
                        print(tickersInd)

        if Comm.get() == 1 :
            for i in range(1,135):
                if categories.iat[2, i] == "Communication Services":
                    if categories.iat[0,i] not in tickersInd :
                        tickersInd.append(categories.iat[0,i])
                        print(tickersInd)

        if Health.get() == 1 :
            for i in range(1,135):
                if categories.iat[2, i] == "Healthcare":
                    if categories.iat[0,i] not in tickersInd :
                        tickersInd.append(categories.iat[0,i])
                        print(tickersInd)




        if usa.get() == 1 :
            for i in range(1,135):
                if categories.iat[1, i] == "USA":
                    if categories.iat[0,i] in tickersInd :
                        tickersCoun.append(categories.iat[0,i])
                        print(tickersCoun)

        if ger.get() == 1 :
            for i in range(1,135):
                if categories.iat[1, i] == "GER":
                    if categories.iat[0,i] in tickersInd :
                        tickersCoun.append(categories.iat[0,i])
                        print(tickersCoun)

        if chn.get() == 1 :
            for i in range(1,135):
                if categories.iat[1, i] == "CHN":
                    if categories.iat[0,i] in tickersInd :
                        tickersCoun.append(categories.iat[0,i])
                        print(tickersCoun)

        if ire.get() == 1 :
            for i in range(1,135):
                if categories.iat[1, i] == "IRE":
                    if categories.iat[0,i] in tickersInd :
                        tickersCoun.append(categories.iat[0,i])
                        print(tickersCoun)

        if nl.get() == 1 :
            for i in range(1,135):
                if categories.iat[1, i] == "NL":
                    if categories.iat[0,i] in tickersInd :
                        tickersCoun.append(categories.iat[0,i])
                        print(tickersCoun)

        if jpn.get() == 1 :
            for i in range(1,135):
                if categories.iat[1, i] == "JPN":
                    if categories.iat[0,i] in tickersInd :
                        tickersCoun.append(categories.iat[0,i])
                        print(tickersCoun)

        if india.get() == 1 :
            for i in range(1,135):
                if categories.iat[1, i] == "IN":
                    if categories.iat[0,i] in tickersInd :
                        tickersCoun.append(categories.iat[0,i])
                        print(tickersCoun)

        if isr.get() == 1 :
            for i in range(1,135):
                if categories.iat[1, i] == "ISR":
                    if categories.iat[0,i] in tickersInd :
                        tickersCoun.append(categories.iat[0,i])
                        print(tickersCoun)

        if ca.get() == 1 :
            for i in range(1,135):
                if categories.iat[1, i] == "CA":
                    if categories.iat[0,i] in tickersInd :
                        tickersCoun.append(categories.iat[0,i])
                        print(tickersCoun)




        if lc.get() == 1 :
            for i in range(1,135):
                if categories.iat[3, i] == "Large Cap":
                    if categories.iat[0,i] in tickersCoun :
                        tickersCap.append(categories.iat[0,i])
                        print(tickersCap)

        if mc.get() == 1 :
            for i in range(1,135):
                if categories.iat[3, i] == "Medium Cap":
                    if categories.iat[0,i] in tickersCoun :
                        tickersCap.append(categories.iat[0,i])
                        print(tickersCap)

        if sc.get() == 1 :
            for i in range(1,135):
                if categories.iat[3, i] == "Small Cap":
                    if categories.iat[0,i] in tickersCoun :
                        tickersCap.append(categories.iat[0,i])
                        print(tickersCap)



        if before.get() == 1 :
            startDate.append('2000-01-01')
            endDate.append('2006-12-31')
            print(startDate)
            print(endDate)

        if during.get() == 1 :
            startDate.append('2007-01-01')
            endDate.append('2008-12-31')
            print(startDate)
            print(endDate)
            
        if after.get() == 1 :
            startDate.append('2009-01-01')
            endDate.append('2019-12-08')
            print(startDate)
            print(endDate)



tk.Button(root, text='Quit', command=root.quit).grid(row=9, column=0)
tk.Button(root, text='Edit', command=csvedit).grid(row=8, column=0)

print(tickersInd)
print(tickersCoun)
print(startDate)
print(endDate)
root.mainloop()


#Optimised portfolio
###################################################################

data = bt.get(tickersCap, start='2015-01-01')

returns = expected_returns.returns_from_prices(data)
returns.to_excel("returns.xlsx")


# Calculate expected returns and sample covariance
mu = expected_returns.mean_historical_return(data)
S = risk_models.sample_cov(data)

# Optimise for maximal Sharpe ratio
ef = EfficientFrontier(mu, S)
raw_weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()

print(cleaned_weights)
ef.portfolio_performance(verbose=True)


latest_prices = get_latest_prices(data)

da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=10000)
allocation, leftover = da.lp_portfolio()
print("Discrete allocation:", allocation)
print("Funds remaining: ${:.2f}".format(leftover))

print(data)



#SMA strategy
##################################################


class SelectWhere(bt.Algo):

    """
    Selects securities based on an indicator DataFrame.

    Selects securities where the value is True on the current date (target.now).

    Args:
        * signal (DataFrame): DataFrame containing the signal (boolean DataFrame)

    Sets:
        * selected

    """
    def __init__(self, signal):
        self.signal = signal

    def __call__(self, target):
        # get signal on target.now
        if target.now in self.signal.index:
            sig = self.signal.ix[target.now]

            # get indices where true as list
            selected = list(sig.index[sig])

            # save in temp - this will be used by the weighing algo
            target.temp['selected'] = selected

        # return True because we want to keep on moving down the stack
        return True



# simple backtest to test long-only allocation
def long_only_ew(tickers, start='2015-01-01', name='long_only_ew'):
    s = bt.Strategy(name, [bt.algos.RunOnce(),
                           bt.algos.SelectAll(),
                           bt.algos.WeighEqually(),
                           bt.algos.Rebalance()])
    data1 = bt.get(tickers, start=start)
    return bt.Backtest(s, data1)


# create the backtests
benchmark = long_only_ew('^GSPC', name='S&P500')


#setting rfr
riskfree =  bt.get('^IRX', start='2015-01-01')
riskfree_rate = riskfree.mean() / 100
print(riskfree_rate)

type(riskfree_rate)

riskfree_rate = float(riskfree_rate)
type(riskfree_rate)



#Strategy 2 - MA Cross
################################################ 


class WeighTarget(bt.Algo):
    """
    Sets target weights based on a target weight DataFrame.

    Args:
        * target_weights (DataFrame): DataFrame containing the target weights

    Sets:
        * weights

    """

    def __init__(self, target_weights):
        self.tw = target_weights

    def __call__(self, target):
        # get target weights on date target.now
        if target.now in self.tw.index:
            w = self.tw.ix[target.now]

            # save in temp - this will be used by the weighing algo
            # also dropping any na's just in case they pop up
            target.temp['weights'] = w.dropna()

        # return True because we want to keep on moving down the stack
        return True
    
    
## download some data & calc SMAs
sma50 = data.rolling(50).mean()
sma200 = data.rolling(200).mean()



## now we need to calculate our target weight DataFrame
# first we will copy the sma200 DataFrame since our weights will have the same strucutre
tw = sma200.copy()

# set appropriate target weights
tw[sma50 > sma200] = (1/len(data.columns))
tw[sma50 <= sma200] = -(1/len(data.columns))

# here we will set the weight to 0 - this is because the sma200 needs 200 data points before
# calculating its first point. Therefore, it will start with a bunch of nulls (NaNs).
tw[sma200.isnull()] = 0.0

# Now set up the MA_cross strategy for our moving average cross strategy
MA_cross = bt.Strategy('MA_cross', [bt.algos.WeighTarget(tw),
                                    bt.algos.Rebalance()])

test_MA = bt.Backtest(MA_cross, data)
res_MA = bt.run(test_MA)


# Plot security weights to test logic
# Note we expect a picture with immediate jumps between 0.2 and -0.2 
#res_MA.plot_security_weights()

# Plot the Equity curve
#res_MA.plot()

# Show the computed results
res_MA.set_riskfree_rate(riskfree_rate)


#strategy - Inverse portfolio
#####################################################
s_inv = bt.Strategy('Inverse of Volatility', 
                       [bt.algos.RunMonthly(),
                       bt.algos.SelectAll(),
                       bt.algos.WeighInvVol(),
                       bt.algos.Rebalance()])

b_inv = bt.Backtest(s_inv, data)


res_inv = bt.run(b_inv)
res_inv.plot_security_weights()

# Plot security weights to test logic
# Note we expect a picture with immediate jumps between 0.2 and -0.2 
#res_MA.plot_security_weights()

#strategy - Random 10
#####################################################
s_random = bt.Strategy('Random 10', 
                       [bt.algos.RunMonthly(),
                       bt.algos.SelectRandomly(n=10),
                       bt.algos.WeighRandomly(),
                       bt.algos.Rebalance()])

b_random = bt.Backtest(s_random, data)

# run all the backtests!
#res2.set_riskfree_rate(riskfree_rate)

#res2.plot(freq='m')
#res2.display()

result = bt.run(test_MA, b_inv, b_random, benchmark)
result.set_riskfree_rate(riskfree_rate)
result.plot()
result.display()
results_key = result.stats.assign()
results_key.to_excel("Demonstration.xlsx")