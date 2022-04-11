#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import yfinance as yf

from sympy import Symbol, solve, solveset, Interval
from sklearn.linear_model import LinearRegression


# In[2]:


class QUANT:    
    def __init__(self):
        self.info = '자주 활용하는 금융공학함수를 정리함'
    
    ## 보유상품을 기간말까지 적립식으로 투자했을 때의 기하수익률 산출
    def get_ACI_CAGR(self, ls):
        '''
        적립식 투자 기하수익률(%) 산출 함수
        '''
        er_ls = 1+ ((ls[len(ls)-1] - ls)/ls)
        er_ls = er_ls[:-1]
        CAGR = (er_ls.product()**(1/len(er_ls)))
        CAGR = (CAGR - 1)*100
        return CAGR
    
    ## 기간당 기하수익률
    def get_ACI_RPP(self, ls):
        '''
        적립식 투자 기하수익률(%)을 기간으로 분할하는 함수
        '''
        CAGR = self.get_ACI_CAGR(ls)
        CAGR = (1+CAGR*(1/100))
        length = len(ls)-1
        RPP = (CAGR**(length*(2/(length*(length+1)))))
        RPP = (RPP - 1)*100
        return RPP
    
    ## 보유기간(월), 연수익률을 기준으로 정기납입시 최종수익률을 산출하는 함수
    def YR_to_TR(self, duration, YR, last_month = False):
        '''
        보유기간, 연수익률로 정기납입 최종수익률을 산출하는 함수
        '''
        MR = YR**(1/12)
        if last_month == False:
            n = duration - 1
            c = MR*((MR**n - 1)/(MR-1))
        elif last_month == True:
            n = duration
            c = ((MR**n - 1)/(MR-1))       
        TR = c/n
        return TR
    
    ## 보유기간, 최종수익률을 기준으로 정기납입시 단위기간수익률을 산출하는 함수
    def TR_to_PR(self, duration, TR, last_month = False):
        '''
        보유기간을 기준으로 최종수익률을 월율화하는 함수
        '''
        r=Symbol('r')
        c = TR
        if last_month == False:
            n = duration - 1
            equation = r*((r**(n) - 1)/(r-1)) - n*c
        elif last_month == True:
            n = duration
            equation = ((r**(n) - 1)/(r-1)) - n*c
        val = solveset(equation, r, Interval(0, 999))
        PR = float(list(val)[0])
        return PR
    
    ## YTD 산출 함수 -> get_YTDs와 연결
    def cal_YTD(self, df, ticker, year, method ='g'):
        '''
        **YTD(연수익률) 산출 함수**
        - 산술수익률:a
        - 기하수익률:g
        '''
        first = df[ticker][(df.index.year == year-1)][-1]
        last = df[ticker][(df.index.year == year)][-1]
        if method == 'g':
            YTD = np.log(last/first)
        elif method == 'a':
            YTD = (last-first)/first
        return YTD

    ## YTD 산출결과 병합함수 -> cal_YTD에 종속
    def get_YTDs(self, df, ticker, method ='g'):
        '''
        **YTD(연수익률) 도시 함수**
        - 산술수익률:a
        - 기하수익률:g
        '''
        temp_idx= df.index.year.unique()[1:]
        temp_ls = []    
        for i in range(len(temp_idx)-1):
            if method == 'g':
                YTD = self.cal_YTD(df, ticker, temp_idx[i], method ='g')
            elif method == 'a':
                YTD = self.cal_YTD(df, ticker, temp_idx[i], method ='a')
            temp_ls.append(YTD)   

        YTD_df = pd.DataFrame(temp_ls, columns=['YTD'], index=temp_idx[:-1])
        YTD_df = YTD_df.T
        YTD_df['MEAN'] = ((YTD_df+1).product(axis=1)**(1/len(YTD_df.columns))) -1
        YTD_df = YTD_df.T
        return YTD_df
    
    ## MDD 산출 <- HRR의 최소값
    def cal_HRR(self, df, ticker, method ='g'):
        '''
        **HRR(고점대비하락률) 산출 함수**
        - 산술수익률:a
        - 기하수익률:g
        '''
        temp_ls = []
        for i in range(len(df)):
            if method == 'g':
                val = np.log(df[ticker][i]/df[ticker][:i+1].max())
            elif method == 'a':
                val = (df[ticker][i] - df[ticker][:i+1].max())/df[ticker][:i+1].max()
            temp_ls.append(val)
        HRR = pd.DataFrame(temp_ls, columns=['HRR'], index=df.index)
        return HRR
    
    ## 연환산 하방 변동성
    def cal_YDD(self, df, ticker, method ='g', unit = 'daily'):
        '''
        **YDD(연평균하방리스크) 산출 함수**
        - 산술수익률:a
        - 기하수익률:g
        '''
        if method == 'g':
            target_ls = np.log(df[ticker]/df[ticker].shift(1))
        elif method == 'a':
            target_ls = (df[ticker]-df[ticker].shift(1))/df[ticker].shift(1)
        if unit == 'daily':
            YDD = (((((target_ls[target_ls<0])**2).sum()/(len(target_ls)-1))*250)**0.5)
        elif unit == 'monthly':
            YDD = (((((target_ls[target_ls<0])**2).sum()/(len(target_ls)-1))*12)**0.5)
        return YDD
    
    ## 연환산 수익률 산출
    def cal_YRR(self, df, ticker, method ='g', unit = 'daily'):
        '''
        **YDD(연평균수익률) 산출 함수**
        - 산술수익률:a
        - 기하수익률:g
        '''
        if method == 'g':
            total_err = np.log(df[ticker][-1]/df[ticker][0])
        elif method == 'a':
            total_err = (df[ticker][-1]-df[ticker][0])/df[ticker][0]
        if unit == 'daily':
            yrr = (1+total_err)**(250/len(df))-1
        elif unit == 'monthly':
            yrr = (1+total_err)**(12/len(df))-1
        return yrr
    
    ## 연수익 레포트 -> get_YTDs에 종속
    def get_YTD_report(self, df, method='g'):
        '''
        **YTD(연수익률) 도시 함수**
        - 산술수익률:a
        - 기하수익률:g
        '''
        YTD_df = pd.DataFrame(columns=df.columns)
        for i in df.columns:
            if (df[i].dtype == float)|(df[i].dtype == int):
                YTD_df[i] = self.get_YTDs(df, i, method)['YTD']
            else:
                YTD_df[i] = np.nan
            YTD_df = YTD_df.dropna(axis=1)
        return YTD_df

    ## 최대낙폭 레포트 출력 -> cal_HRR에 종속
    def get_MDD_report(self, df, method='g'):
        '''
        **MDD(최대낙폭) 출력 함수**
        - 산술수익률:a
        - 기하수익률:g
        '''
        ## HRR(전고점대비하락율) 산출
        HRR_df = pd.DataFrame(columns=df.columns)
        for i in df.columns:
            if (df[i].dtype == float)|(df[i].dtype == int):
                HRR_df[i] = self.cal_HRR(df, i, method)['HRR']
            else:
                HRR_df[i] = np.nan
            HRR_df = HRR_df.dropna(axis=1)
        ##MDD(전고점대비최고하락율) 도출  
        temp_ls = []
        for i in HRR_df.columns:
            temp = HRR_df[i][HRR_df[i] == HRR_df[i].min()]
            temp_ls.append([i, temp.values[0], temp.index[0]])
            MDD_df = pd.DataFrame(temp_ls, columns=['Ticker', 'MDD', 'Date'])
        return MDD_df, HRR_df

    ## 소랜토 비율 레포트 출력 -> cal_YRR / cal_YDD에 종속
    def get_SRTR_report(self, df, method='g', unit = 'daily'):
        '''
        **STRT(sortino ratio) 도시 함수**
        - 산술수익률:a
        - 기하수익률:g
        '''
        temp_ls = []
        for i in df.columns:
            if (df[i].dtype == float)|(df[i].dtype == int):
                YRR = self.cal_YRR(df, i, method, unit)
                YDD = self.cal_YDD(df, i, method, unit)
                SRTR = YRR/YDD ##무위험 수익률을 무시한다.
                temp_ls.append([i, YRR, YDD, SRTR])
                SRTR_df = pd.DataFrame(temp_ls, columns=['Ticker', 'YRR', 'YDD', 'Sortino Ratio'])
        return SRTR_df
    
    def get_rets_df(self, df):
        '''기하변동률 df 출력 함수'''
        rets = np.log(df/df.shift(1))
        rets = rets.dropna()
        return rets
        
        ## get_beta_df에 활용
    def cal_beta(self, rets, ticker1, ticker2):
        '''두 자산의 변동률 간 Beta(회귀계수) 산출 함수'''
        X = rets[ticker1]
        y = rets[ticker2]

        line_fitter = LinearRegression()
        line_fitter.fit(X.values.reshape(-1,1), y)
        beta = line_fitter.coef_[0]

        return beta
        
        ## cal_beta()에 연결
    def get_beta_df(self, rets):
        '''Beta(인덱스 ticker가 1단위 변했을 때 컬럼 ticker가 얼마나 움직이는지) 도시 함수'''
        betas = [self.cal_beta(rets, i, j) for i in rets.columns for j in rets.columns]
        tickers = [(i, j) for i in rets.columns for j in rets.columns]
        beta_df = pd.DataFrame(np.array(betas).reshape((len(rets.columns), len(rets.columns))), columns=rets.columns, index=rets.columns)

        return beta_df
    
    def get_odds_df(self, rets, rf):
        '''자산의 무위험수익률 대비 승률 df 출력 함수'''
        odds = (rets>rf).sum()/len(rets)
        odds = pd.DataFrame(odds, columns=['odds'])
        return odds
    
class DATA:    
    def __init__(self):
        self.info = '데이터를 불러오고 병합 계량하는 함수를 정리함'
        
    ## 데이터 불러오기
    def get_df(self, ticker):
        '''야후 파이낸스에서 자산가 데이터를 불러옴'''
        ## get_merged_df()에 활용
        profit = yf.Ticker(ticker)
        df = profit.history(period="max")
        return df

    ## 데이터 병합
    def get_merged_df(self, *tickers):
        '''야후 파이낸스에서 불러온 자산가 데이터를 병합하여 하나로 구성함'''
        ## get_df()에 종속
        dfs = [self.get_df(tickers[i]) for i in range(len(tickers))]
        temp_df = [dfs[i]['Close'] for i in range(len(dfs))]
        temp_df = pd.DataFrame(temp_df, index=tickers).T
        return temp_df
    
    ## 데이터 지수화(동일비교)
    def index_values(self, orgin_df):
        '''시계열 df를 기준시점 기준으로 지수화(1로 변환)하는 함수'''
        df = orgin_df.copy()
        for i in df.columns:
            if (df[i].dtype == float)|(df[i].dtype == int):
                df[i] = df[i]/df[i][0]
            else:
                df[i] = df[i]
        return df
    
    ## 데이터 지수화(동일비교)
    def modi_ts(self, df, units='d'):
        '''일단위 시계열 데이터 일(d),월(m),분기(q),반기(h),년(y) 단위 변환 함수'''
        new_df = df.copy()
        new_df['year'] = new_df.index.year
        new_df['month'] = new_df.index.month
        new_df['half'] = ((new_df['month']-1)//6)+1
        new_df['quarter'] = ((new_df['month']-1)//3)+1

        if units == 'd':
            new_df = df.copy()
        if units == 'm':
            ## 월단위 변환
            new_df = new_df.drop_duplicates(['year', 'month'], keep='last')
        if units == 'q':
            ## 분기단위 변환
            new_df = new_df.drop_duplicates(['year', 'quarter'], keep='last')
        if units == 'h':
            ## 반기단위 변환
            new_df = new_df.drop_duplicates(['year', 'half'], keep='last')
        if units == 'y':
            ## 연단위 변환
            new_df = new_df.drop_duplicates('year', keep='last')
        ## 기간인덱스 제거
        new_df = new_df.drop(columns=['year', 'month', 'half', 'quarter'])
    
        return new_df
    
    
    def cal_ema(self, ts, n):
        '''
        ts: Time Series, n = term
        EMA(지수이동평균)리턴
        '''
        ## 승수산출
        k = 2/(1+n)
        ## 초기값(SMA:단순평균)
        ini_val = ts[:n].mean()
        ## 공간구성
        temp_ls = [np.nan for i in range(n-1)]
        temp_ls.append(ini_val)
        ## EMA산출 저장
        for i in range(len(ts[n:])):
            val = ts[n + i]*k + temp_ls[n-1+i]*(1-k)
            temp_ls.append(val)
        ema_ts = pd.Series(temp_ls, index=ts.index)

        return ema_ts


# In[ ]:




