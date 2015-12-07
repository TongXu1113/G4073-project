
# Technical trading strategies & Boosting signal generator for single stock
# Alec Myres, Yeyun Chen and Tong Xu
# MATH G4073 - Columbia University

import os
os.chdir('C:/Users/Tong/Google Drive/Schoolwork/Columbia/3rd Semester/Quant Methods in Investment Management/py/dec-6')

# Load useful python libraries
import numpy as np
import pandas as pd
#import datetime, dateutil, os, sys
import boostingnew

# min/max cumulative sum for an array
def min_max_cum_sum(array):
  cs = np.cumsum(array)
  return np.min(cs), np.max(cs)
  
def action2position(df_act, df_return, df_idreturn, rw):
    result = df_return.copy(deep=True)
    act = df_act.shift(1).copy(deep=True)
    count = 0
    for k in range(2,len(df_act)):
        if act[k] == 0:
            count += 1
            if count == 1:
                #kkkk = df_idreturn.loc[k]
                #result.loc[k] = kkkk
                #result.loc[k] = df_idreturn.loc[k]
                qqqq = act.loc[k-1]
                act.loc[k] = qqqq
                act.loc[k-1] = 0
                #df_act.loc[k] = df_act.loc[k-1]
                #df_act.loc[k-1] = 0
            elif count <= rw - 1:
                qqqq = act.loc[k-1]
                act.loc[k] = qqqq
    
    #for k in range(2,len(df_act)):
    #    if act[k] != act[k-1]:
    #        kkkk = df_idreturn.loc[k]
    #        result.loc[k] = kkkk
    result = result * act + 1
    return result

def action2position1(df_act, rw):
    result = [1]*len(df_act)
    act = df_act.shift(1).copy(deep=True)
    count = 0
    for k in range(2,len(df_act)):
        if act[k] == 0:
            count += 1
            if count == 1:
                #result.loc[k] = df_idreturn.loc[k]
                qqqq = act.loc[k-1]
                act.loc[k] = qqqq
                act.loc[k-1] = 0
                #df_act.loc[k] = df_act.loc[k-1]
                #df_act.loc[k-1] = 0
            elif count <= rw - 1:
                qqqq = act.loc[k-1]
                act.loc[k] = qqqq
    result = result * act
    return result


# --------------------
# INDICATORS
# --------------------

# Simple moving average
def get_sma(df_col, w = 20):
  return pd.rolling_mean(df_col, window = w)

# Exponentially weighted moving average
def get_ema(df_col, alpha = 0.95):
  s = (1.0 + alpha)/(1.0 - alpha)
  return pd.ewma(df_col, span = s)

# Bollinger bands
def get_bollinger(df_col, w = 20, s = 2):
  bandwidth = pd.rolling_std(df_col, window = w)
  upper = get_sma(df_col, w) + bandwidth*s
  lower = get_sma(df_col, w) - bandwidth*s
  return upper, lower

# Momentum (differences)
def get_mom(df_col, per = 12):
  roll = (np.roll(df_col, 0) - np.roll(df_col, per))[per:]
  fill = [np.nan]*per
  return pd.Series(fill + list(roll))

# Momentum EMA
def get_mom_ema(df_col, per = 12, alpha = 0.75):
  mom = get_mom(df_col, per)
  ema_mom = get_ema(mom, alpha)
  return mom/ema_mom

# Acceleration (difference of price change)
def get_accel(df_col, per = 10):
    mom_t = get_mom(df_col, per)[per:]
    roll = (np.roll(mom_t, 0) - np.roll(mom_t, 1))[1:]
    fill = [np.nan]*(per + 1)
    return pd.Series(fill + list(roll))

# rate of change
def get_roc(df_col, per = 10):
  roll = (get_mom(df_col, per)/np.roll(df_col, per))[per:]
  fill = [np.nan]*per
  return pd.Series(fill + list(roll))

# moving average convergence divergence
# difference between two moving averages of slow(s) and fast(f) periods
def get_macd(df_col, s = 26, f = 12):
    return(pd.ewma(df_col, span = s) - pd.ewma(df_col, span = f))

# MACD signal line: moving average of MACD of past n periods
def get_macds(df_col, s = 26, f = 12, n = 9):
    return(pd.ewma(get_macd(df_col, s, f), span = n))

# Relative strength index
def get_rsi(df_col, per = 14):
    p_up = [np.nan]*df_col.size
    p_dn = [np.nan]*df_col.size
    for i in range(1, df_col.size):
        if (df_col[i] > df_col[i-1]):
            p_up[i] = df_col[i]
        elif (df_col[i] < df_col[i-1]):
            p_dn[i] = df_col[i]

    ma_up = [np.nan]*df_col.size
    ma_dn = [np.nan]*df_col.size
    rsi   = [np.nan]*df_col.size
    for j in range(per, df_col.size):
        ma_up[j] = np.nanmean(p_up[j-per:j])
        ma_dn[j] = np.nanmean(p_dn[j-per:j])
        rsi[j] = 100 - 100/(1 + ma_up[j]/ma_dn[j])
    return pd.Series(rsi)

# Stochastic Oscillator
def get_stoch(high, low, close, n = 12, per = 3):
    Fast_K = [np.nan]*high.size
    Fast_D = [np.nan]*high.size
    Slow_K = [np.nan]*high.size
    Slow_D = [np.nan]*high.size
    for i in range(n, high.size):
        P_high = high[i-n: i]
        P_low = low[i-n: i]
        Fast_K[i] = (close[i]-min(P_low))/(max(P_high)-min(P_low))
    Fast_K = pd.Series(Fast_K)
    Fast_D = pd.Series(pd.rolling_mean(Fast_K, window = per))
    Slow_K = pd.Series(pd.rolling_mean(Fast_K, window = per))
    Slow_D = pd.Series(pd.rolling_mean(Slow_K, window = per))
    return Fast_K, Fast_D, Slow_K, Slow_D

def get_adl(close, high, low, volume):
    CLV = np.divide((2*close - high - low), (high-low))
    result = np.cumsum(CLV * volume)
    return result

# --------------------
# SIGNALS
# --------------------

# Derive buy/sell signals from technical indicators
#  1: buy
# -1: sell
#  0: no action

# Strategy 1: Bollinger Bands
def GetStrategyBollinger(prices, n = 20, s = 2):
    action = [0]*prices.size
    bollUpper, bollLower = get_bollinger(prices, n, s)
    for i in range(1, prices.size):
        if (prices[i] > bollUpper[i]):
            action[i] = -1
        elif (prices[i] < bollLower[i]):
            action[i] = 1     
    return action 

# Strategy 2: Momentum
def GetStrategyMomentum(prices, n = 12):
    action = [0]*prices.size
    mom  = get_mom(prices, n)
    momEMA = get_mom_ema(prices, n)
    for i in range(1, prices.size):
        if ~np.isnan(mom[i-1]) and ~np.isnan(momEMA[i-1]):
            if (mom[i] > momEMA[i] and mom[i-1] <= momEMA[i-1]):
                action[i] = 1
            elif (mom[i] < momEMA[i] and mom[i-1] >= momEMA[i-1]):
                action[i] = -1
    return action         

# Strategy 3: Rate of Change
def GetStrategyROC(prices, n = 10):
    action = [0]*prices.size
    ROC  = get_roc(prices, n)
    for i in range(1, prices.size):
        if ~np.isnan(ROC[i-1]):
            if (ROC[i-1] <= 0) and (ROC[i] > 0):
                action[i] = 1
            elif (ROC[i-1] >= 0) and (ROC[i] < 0):
                action[i] = -1
    return action         

# Strategy 4: Acceleration
def GetStrategyAcceleration(prices, n = 12):
    action = [0]*prices.size
    accel = get_accel(prices, n)
    for i in range(1, prices.size):
        if ~np.isnan(accel[i-1]):
            if (accel[i-1] + 1 <= 0) and (accel[i] > 0):
                action[i] = 1
            elif (accel[i-1] + 1 >= 0) and (accel[i] < 0):
                action[i] = -1
    return action

# Strategy 5: Moving Average Convergence Difference
def GetStrategyMACD(prices, s = 26, f = 12, n = 9):
    macd = get_macd(prices, s, f)
    macds = get_macds(prices, s, f, n)
    action = [0]*prices.size
    for i in range(1, prices.size):
      if (macd[i-1] <= macds[i]) and (macd[i] > macds[i]):
        action[i] = 1
      elif (macd[i-1] >= macds[i]) and (macd[i] < macds[i]):
        action[i] = -1
    return action

# Strategy 6: Relative Strength Index
def GetStrategyRSI(prices, per = 14, lower = 30, upper = 70):
    rsi = get_rsi(prices, per)
    action = [0]*prices.size
    for i in range(per, prices.size):
      if (rsi[i-1] >= lower) and (rsi[i] < lower):
        action[i] = 1
      elif (rsi[i-1] <= upper) and (rsi[i] > upper):
        action[i] = -1
    return action

# Strategy 7: Fast Stochastic Trading Rule
def GetStrategyFast(high, low, close):
    Fast_K, Fast_D, Slow_K, Slow_D = get_stoch(high, low, close)
    action = [0]*close.size
    for i in range(0, close.size):
      if (np.isnan(Fast_K[i]) == False) and (np.isnan(Fast_D[i]) == False):
          if Fast_K[i] > Fast_D[i] and Fast_K[i-1] <= Fast_D[i-1]:
              action[i] = 1
          elif Fast_K[i] < Fast_D[i] and Fast_K[i-1] >= Fast_D[i-1]:
              action[i] = -1
    return action

# Strategy 8: Slow Stochastic Trding Rule
def GetStrategySlow(high, low, close):
    Fast_K, Fast_D, Slow_K, Slow_D = get_stoch(high, low, close)
    action = [0]*close.size
    for i in range(0, close.size):
      if (np.isnan(Slow_K[i]) == False) and (np.isnan(Slow_D[i]) == False):
          if Slow_K[i] > Slow_D[i] and Slow_K[i-1] <= Slow_D[i-1]:
              action[i] = 1
          elif Slow_K[i] < Slow_D[i] and Slow_K[i-1] >= Slow_D[i-1]:
              action[i] = -1
    return action

# Strategy 9: Money Flow Index
def GetStrategyMFI(open, high, low, close, volume, n, lower, upper):
    MF = (high + low + close)/3 * volume * np.sign(close - open)
    PMF = MF[MF>0]
    NMF = MF[MF<0]
    PMF = get_sma(PMF,n)
    NMF = get_sma(NMF,n)
    MF[MF>0] = PMF
    MF[MF<0] = NMF
    MFR = MF.copy(deep=True)
    MFR[~np.isnan(MF)] = 1.00
    MFI = MF.copy(deep=True)
    MFI[~np.isnan(MF)] = 50.0
    neg = 0
    pos = 0
    for i in range(0,MF.size):
        if MF[i]<0:
            neg=i
        elif MF[i]>0:
            pos=i
        
        if neg>0 and pos>0 and ~np.isnan(MF[i]):
            MFR[i]=abs(MF[pos]/MF[neg])
            MFI[i]=100-100/(1+MFR[i])
    action = [0]*close.size
    action = np.where(MFI<=lower,1,action)
    action = np.where(MFI>=upper,-1,action)
    return action

# Strategy 6 new: Chaikin Oscillator
def GetStrategyCHO(close, high, low, volume, n1, n2):
    ADL = get_adl(close, high, low, volume)
    CHO = pd.ewma(ADL, n1) - pd.ewma(ADL, n2)
    CHOstd = pd.rolling_std(CHO,20)
    CHO1 = np.divide(CHO, CHOstd)
    result = close.copy(deep=True)
    result = np.where(result>=0, 0, 0)
    result = np.where(CHO1>=1,1,result)
    result = np.where(CHO1<=-1,-1,result)
    return result


def GenerateSignals(df_o, file_name, rw = 5, threshhold = 0.03, boosting_period = 120, trading_period = 40, RUS_multiplier = 1.5, performance_output = False):
    df = df_o.copy(deep=True)
    df = df[['date','split adjusted px','open','high','low','volume']]
    df.rename(columns = {'split adjusted px':'Price'}, inplace = True)
    df.rename(columns = {'date':'Date'}, inplace = True)
    df.rename(columns = {'open':'Open'}, inplace = True)
    #df['Date'] = map(lambda x: dateutil.parser.parse(x).strftime('%Y-%m-%d'), df['date'])
    #df = df[['Date','Price']].sort('Date').reset_index(drop = True)
    
    # Add y columns
    #rw = 5 # rolling window
    #threshhold = 0.03
    df['logreturn'] = np.log(df['Price']) - np.log(df['Price'].shift(1))
    df['idlogreturn'] = np.log(df['Price']) - np.log(df['Open'])
    df['idreturn'] = np.divide(df['Price'], df['Open']) - 1
    df['return'] = np.divide(df['Price'], df['Price'].shift(1)) - 1
    df['min_rollsum'] = map(lambda x: min_max_cum_sum(df['logreturn'][x:x+rw])[0], df.index)
    df['max_rollsum'] = map(lambda x: min_max_cum_sum(df['logreturn'][x:x+rw])[1], df.index)
    df['y_b'] = np.where(df['max_rollsum'] > threshhold, 1, 0)
    df['y_s'] = np.where(df['min_rollsum'] < -threshhold, -1, 0)
    
    # Add indicator columns to data frame
    df['action1'] = GetStrategyBollinger(df['Price'])
    #df['action2'] = GetStrategyMomentum(df['Price'])
    df['action3'] = GetStrategyROC(df['Price'])
    df['action4'] = GetStrategyAcceleration(df['Price'])
    df['action5'] = GetStrategyMACD(df['Price'])
    df['action6'] = GetStrategyCHO(df.Price,df.high,df.low,df.volume,3,10)
    df['action7'] = GetStrategyFast(df.high, df.low, df.Price)
    df['action8'] = GetStrategySlow(df.high, df.low, df.Price)
    #df['action9'] = GetStrategyMFI(df.Open, df.high, df.low, df.Price, df.volume, 5, 40, 60)
    
    # Split into buy/sell components
    df['action1buy'] = np.where(df['action1'] == -1, 0, df['action1'])
    df['action1sell'] = np.where(df['action1'] == 1, 0, df['action1'])
    #df['action2buy'] = np.where(df['action2'] == -1, 0, df['action2'])
    #df['action2sell'] = np.where(df['action2'] == 1, 0, df['action2'])
    df['action3buy'] = np.where(df['action3'] == -1, 0, df['action3'])
    df['action3sell'] = np.where(df['action3'] == 1, 0, df['action3'])
    df['action4buy'] = np.where(df['action4'] == -1, 0, df['action4'])
    df['action4sell'] = np.where(df['action4'] == 1, 0, df['action4'])
    df['action5buy'] = np.where(df['action5'] == -1, 0, df['action5'])
    df['action5sell'] = np.where(df['action5'] == 1, 0, df['action5'])
    df['action6buy'] = np.where(df['action6'] == -1, 0, df['action6'])
    df['action6sell'] = np.where(df['action6'] == 1, 0, df['action6'])
    df['action7buy'] = np.where(df['action7'] == -1, 0, df['action7'])
    df['action7sell'] = np.where(df['action7'] == 1, 0, df['action7'])
    df['action8buy'] = np.where(df['action8'] == -1, 0, df['action8'])
    df['action8sell'] = np.where(df['action8'] == 1, 0, df['action8'])
    #df['action9buy'] = np.where(df['action9'] == -1, 0, df['action9'])
    #df['action9sell'] = np.where(df['action9'] == 1, 0, df['action9'])
    
    df_copy = df.copy(deep=True)
    #df_copy = df[['y_b','y_s','action1buy','action1sell','action2buy','action2sell','action3buy','action3sell','action4buy','action4sell','action5buy','action5sell','action7buy','action7sell','action8buy','action8sell','action9buy','action9sell',]].copy(deep=True)
    dfo = df.copy(deep=True)
    df_pos = df[['Date','Price','logreturn','return']].copy(deep=True)
    alphas = 0
    strategies = 0
    for idx in range(0,((len(df_pos.Price)+1)/trading_period)-3):
        start_time = idx * trading_period
        del alphas
        del strategies
        df_copy = dfo.copy(deep=True)
        alphas, strategies = boostingnew.update_weights(df_copy[(df_copy.index > start_time) & (df_copy.index <= start_time + boosting_period)], "buy", RUS_multiplier)
    
        for i in range(len(alphas)):
            if alphas[i]<0 :
                df_copy[strategies[i] + 'wgt'] = np.where(df_copy[strategies[i]] == 0, -1, 1)*alphas[i]
            else:
                df_copy[strategies[i] + 'wgt'] = np.where(df_copy[strategies[i]] == 0, -1, 1)*alphas[i]
    
        columns = map(lambda x: x+"wgt", strategies)
        df_copy['final'] = df_copy[columns].sum(axis = 1)
        df_copy['final_ind'] = np.where(df_copy['final'] >= 0, 1, 0)
            
        alphas_buy = alphas
        strategies_buy = strategies
        
        if idx==0:
            df_pos['buy'] = df_copy['final_ind'].copy(deep=True)
        
        df_pos['buy'][(df_copy.index > start_time + boosting_period) & (df_copy.index <= start_time + boosting_period + trading_period)] = df_copy['final_ind'][(df_copy.index > start_time + boosting_period) & (df_copy.index <= start_time + boosting_period + trading_period)].copy(deep=True)
        
        del alphas
        del strategies
        df_copy = dfo.copy(deep=True)
        alphas, strategies = boostingnew.update_weights(df_copy[(df_copy.index > start_time) & (df_copy.index <= start_time + boosting_period)], "sell", RUS_multiplier)
        
        alphas_sell = alphas
        strategies_sell = strategies
        
        for i in range(len(alphas)):
            if alphas[i]<0 :
                df_copy[strategies[i] + 'wgt'] = np.where(df_copy[strategies[i]] == 0, -1, 1)*alphas[i]
            else:
                df_copy[strategies[i] + 'wgt'] = np.where(df_copy[strategies[i]] == 0, -1, 1)*alphas[i]
        
        columns = map(lambda x: x+"wgt", strategies)
        df_copy['final'] = df_copy[columns].sum(axis = 1)
        df_copy['final_ind'] = np.where(df_copy['final'] >= 0, 1, 0)
        
        if idx==0:
            df_pos['sell'] = df_copy['final_ind'].copy(deep=True)
        
        df_pos['sell'][(df_copy.index > start_time + boosting_period) & (df_copy.index <= start_time + boosting_period + trading_period)] = df_copy['final_ind'][(df_copy.index > start_time + boosting_period) & (df_copy.index <= start_time + boosting_period + trading_period)].copy(deep=True)
    
    
    
    
    df_pos['final'] = df_pos['buy'] - df_pos['sell']
    
    if performance_output == True:
        df_pos['logreturn1'] = df_pos['logreturn'].copy(deep=True)
        df_pos['return1'] = df_pos['return'].copy(deep=True)
        df_pos['y_b'] = df.y_b.copy(deep=True)
        df_pos['y_s'] = df.y_s.copy(deep=True)
        df_pos['Strat1'] = df.action1.copy(deep=True)
        #df_pos['Strat2'] = df.action2.copy(deep=True)
        df_pos['Strat3'] = df.action3.copy(deep=True)
        df_pos['Strat4'] = df.action4.copy(deep=True)
        df_pos['Strat5'] = df.action5.copy(deep=True)
        df_pos['Strat6'] = df.action6.copy(deep=True)
        df_pos['Strat7'] = df.action7.copy(deep=True)
        df_pos['Strat8'] = df.action8.copy(deep=True)
        #df_pos['Strat9'] = df.action9.copy(deep=True)
        
        df_pos['positions'] = action2position1(df_pos.final, rw)
        
        df_pos['ret1'] = action2position(df_pos.Strat1, df_pos.return1, df.idreturn, rw)
        #df_pos['ret2'] = action2position(df_pos.Strat2, df_pos.return1, df.idreturn, rw)
        df_pos['ret3'] = action2position(df_pos.Strat3, df_pos.return1, df.idreturn, rw)
        df_pos['ret4'] = action2position(df_pos.Strat4, df_pos.return1, df.idreturn, rw)
        df_pos['ret5'] = action2position(df_pos.Strat5, df_pos.return1, df.idreturn, rw)
        df_pos['ret6'] = action2position(df_pos.Strat6, df_pos.return1, df.idreturn, rw)
        df_pos['ret7'] = action2position(df_pos.Strat7, df_pos.return1, df.idreturn, rw)
        df_pos['ret8'] = action2position(df_pos.Strat8, df_pos.return1, df.idreturn, rw)
        #df_pos['ret9'] = action2position(df_pos.Strat9, df_pos.return1, df.idreturn, rw)
        
        #df_pos['cumprod1'][boosting_period:len(df_pos.ret1)] = np.cumprod(df_pos['ret1'][boosting_period:len(df_pos.ret1)])
        ##df_pos['cumprod2'][boosting_period:len(df_pos.ret1)] = np.cumprod(df_pos['ret2'][boosting_period:len(df_pos.ret1)])
        #df_pos['cumprod3'][boosting_period:len(df_pos.ret1)] = np.cumprod(df_pos['ret3'][boosting_period:len(df_pos.ret1)])
        #df_pos['cumprod4'][boosting_period:len(df_pos.ret1)] = np.cumprod(df_pos['ret4'][boosting_period:len(df_pos.ret1)])
        #df_pos['cumprod5'][boosting_period:len(df_pos.ret1)] = np.cumprod(df_pos['ret5'][boosting_period:len(df_pos.ret1)])
        #df_pos['cumprod6'][boosting_period:len(df_pos.ret1)] = np.cumprod(df_pos['ret6'][boosting_period:len(df_pos.ret1)])
        #df_pos['cumprod7'][boosting_period:len(df_pos.ret1)] = np.cumprod(df_pos['ret7'][boosting_period:len(df_pos.ret1)])
        #df_pos['cumprod8'][boosting_period:len(df_pos.ret1)] = np.cumprod(df_pos['ret8'][boosting_period:len(df_pos.ret1)])
        ##df_pos['cumprod9'][boosting_period:len(df_pos.ret1)] = np.cumprod(df_pos['ret9'][boosting_period:len(df_pos.ret1)])
        
        df_pos['tret'] = action2position(df_pos.final, df_pos.return1, df.idreturn, rw)
        df_pos['BOOST'] = 1
        df_pos['BOOST'][boosting_period:len(df_pos.tret)] = np.cumprod(df_pos['tret'][boosting_period:len(df_pos.tret)])
        
        df_pos[boosting_period:len(df_pos.Date)].to_csv(file_name + '_perfornamce.csv')
        
    df_output = pd.concat([df_pos[['Date','final']], df_o[['split adjusted px','open','high','low']]], axis=1)
    df_output.rename(columns = {'split adjusted px':'PX_Close'}, inplace = True)
    df_output.rename(columns = {'open':'PX_Open'}, inplace = True)
    df_output.rename(columns = {'high':'PX_High'}, inplace = True)
    df_output.rename(columns = {'low':'PX_Low'}, inplace = True)
    df_output.rename(columns = {'final':'Signal'}, inplace = True)
    df_output[boosting_period:len(df_output.Date)].to_csv(file_name + '.csv')
    
    return 

#Your code to manipulate function above
