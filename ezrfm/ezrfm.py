import pandas as pd
import numpy as np

class RFM:
    def __init__(self, data):
        self.raw_data = data
        self.data_model = None
        self.init_df = None
        self.prep_data = None
    
    def get_data_model(self, customerID, orderID, orderDate, qty, spend):
        named_df = self.raw_data.rename(columns={
            customerID: 'customerID',
            orderID: 'orderID',
            orderDate: 'orderDate',
            qty: 'qty',
            spend: 'spend'
        }).copy()
        columns = ['customerID','orderID','orderDate','qty','spend']
        
        named_df = named_df[columns]

        self.data_model = named_df.copy()

        return self.data_model
    
    def init_data(self):
        init_df = self.data_model.groupby(['customerID','orderID']).agg(
            orderDate = pd.NamedAgg(column='orderDate', aggfunc=lambda x: x.max()),
            sum_qty = pd.NamedAgg(column='qty', aggfunc='sum'),
            sum_spend = pd.NamedAgg(column='spend', aggfunc='sum')
        ).reset_index()

        init_df['date'] = init_df['orderDate'].dt.date
        init_df['year'] = init_df['orderDate'].dt.year
        init_df['quarter'] = init_df['orderDate'].dt.quarter
        init_df['month'] = init_df['orderDate'].dt.month
        init_df['week'] = init_df['orderDate'].dt.isocalendar().week
        init_df['day'] = init_df['orderDate'].dt.day

        init_df['year_quarter'] = init_df['year'].astype(str)+'-'+init_df['quarter'].astype(str).apply(lambda x: x.zfill(2))
        init_df['year_month'] = init_df['year'].astype(str)+'-'+init_df['month'].astype(str).apply(lambda x: x.zfill(2))
        init_df['year_week'] = init_df['year'].astype(str)+'-'+init_df['week'].astype(str).apply(lambda x: x.zfill(2))

        self.init_df = init_df.copy()
        return self.init_df

    def prep_rfmt(self):
        init_df = self.init_df.copy()
        target = init_df['orderDate'].max() + np.timedelta64(1,'D')
        rfmt = init_df.groupby('customerID').agg(
            recency=pd.NamedAgg(column='orderDate', aggfunc=lambda x: (target-x.max()).days),
            frequency=pd.NamedAgg(column='orderID', aggfunc='nunique'),
            monetary=pd.NamedAgg(column='sum_spend', aggfunc='sum'),
            tenure=pd.NamedAgg(column='orderDate', aggfunc=lambda x: (target-x.min()).days),
            basketSize=pd.NamedAgg(column='sum_spend', aggfunc='mean'),
            stdMonetary=pd.NamedAgg(column='sum_spend', aggfunc='std'),
            minMonetary=pd.NamedAgg(column='sum_spend', aggfunc='min'),
            maxMonetary=pd.NamedAgg(column='sum_spend', aggfunc='max'),
            rangeMonetary=pd.NamedAgg(column='sum_spend', aggfunc=lambda x: x.max()-x.min()),
            shopWeek=pd.NamedAgg('year_week', aggfunc='nunique')
        )
        rfmt['freq_w'] = rfmt['frequency']/rfmt['shopWeek']
        rfmt['weekSize'] = rfmt['monetary']/rfmt['shopWeek']
        
        return rfmt.reset_index().fillna(0).round(2)

    def prep_evolving(self, lags=2):
        evol_basket = self.init_df.copy()

        for lag in range(lags+1):
            column = '_{}'.format(lag)
            evol_basket[column] = evol_basket.groupby('customerID')['sum_spend'].shift(lag).round(2)

        evol_basket = evol_basket.groupby('customerID').agg(
            {"_{}".format(i): lambda x: x.iloc[-1] for i in range(lags+1)}
        )

        evol_basket.columns = ['lag{}'.format(col) for col in evol_basket.columns]

        return evol_basket.reset_index().fillna(0)

    def prep_evolving_pct(self, lags=2):
        evol_basket = self.init_df.copy()

        for lag in range(lags+1):
            column = '_{}'.format(lag)
            evol_basket[column] = evol_basket.groupby('customerID')['sum_spend'].pct_change(lag+1).round(2)

        evol_basket = evol_basket.groupby('customerID').agg(
            {"_{}".format(i): lambda x: x.iloc[-1] for i in range(lags+1)}
        )

        evol_basket.columns = ['pct{}'.format(col) for col in evol_basket.columns]

        return evol_basket.reset_index().fillna(0)

    def prep(self, lags=2):
        self.init_data()
        rfmt_df = self.prep_rfmt()
        evol_df = self.prep_evolving(lags=lags)
        evol_df_pct = self.prep_evolving_pct(lags=lags)

        merged_df = rfmt_df.merge(evol_df,how='left',left_on='customerID',right_on='customerID')
        merged_df = merged_df.merge(evol_df_pct,how='left',left_on='customerID',right_on='customerID')
        self.prep_data = merged_df
        return self.prep_data

    def to_frame(self):
        return self.prep_data

    def shape(self):
        return self.prep_data.shape

    def check_up(self, method="raw"):
        
        if method=="prep":
            data = self.prep_data
        else:
            data = self.raw_data
        row, col = data.shape
        checked_df = pd.DataFrame({
            'dtypes': data.dtypes,
            'na': data.isna().sum().values,
            'na_ratio':((data.isna().sum()/row)*100).values.round(2),
            'nunique': data.nunique().values,
            'uni_ratio': ((data.nunique()/row)*100).values.round(2),
            '<=0': [(data[col].values <= 0).any() if data[col].dtypes in [int, float] else False for col in data.columns],
            '>0': [(data[col].values > 0).any() if data[col].dtypes in [int, float] else False for col in data.columns]

        })
        return checked_df