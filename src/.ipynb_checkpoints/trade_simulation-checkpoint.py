import pandas as pd

def trade_on_pred(actual, pred, spread, threshold):
    df = pd.DataFrame(columns = ['actual','pred'])
    df.actual=actual
    df.pred=pred
    
    df['long']  = df.apply(lambda row: row['actual'] + spread/100  if trade_signal(row['actual'],row['pred'],threshold)=='buy' else None, axis=1)
    df['short'] = df.apply(lambda row: row['actual']               if trade_signal(row['actual'],row['pred'],threshold)=='sell' else None, axis=1)
        
    pos_long  = df.long.notna().sum()
    pos_short = df.short.notna().sum()
    pos_total = pos_long - pos_short
    trades    = pos_long + pos_short
    
    long_mtm  = round( pos_long  * (df.actual.iloc[-1] - df.long.mean()),3)*100
    short_mtm = round(-pos_short * (df.actual.iloc[-1] - df.short.mean()),3)*100
    mtm_total = round(long_mtm + short_mtm,3)
    
    trade_results = {'Long':[pos_long, long_mtm],
                     'Short':[-pos_short,short_mtm],
                     'Total':[pos_total,mtm_total],
                     'Trades':[trades,df.shape[0]],
                     'Spread':[spread,'']
                    }
    
    return pd.DataFrame(trade_results,index=['Pos','MTM'])


def trade_signal(actual_price, pred_price, threshold):
    if abs(pred_price - actual_price) <= threshold/100:
        if actual_price < pred_price:
            return 'buy'
        if actual_price > pred_price:
            return 'sell'
