import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import statsmodels.api as sm
import pylab as py
import scipy.stats as stats
from src.trade_simulation import trade_on_pred

def mlp_learnig_curve_plot(history,title):
    df=pd.DataFrame()
    df['train'] = history.history['mae']
    df['test'] = history.history['val_mae']
    df['epoch']=[i for i in range(1,len(df['train'])+1)]
    fig = px.line(df,x='epoch',y=['train','test'],labels={'value':'mae'}, title=title)
    fig.update_xaxes(nticks=10)
    fig.update_yaxes(nticks=10)
    fig.update_layout(width=600,height=400)
    
    return fig

def test_predicted_plot(actual, pred, index, title,legend_title):
    df=pd.DataFrame(pred,columns = ['predicted'])
    df['actual']=actual
    df.index=index
    fig = px.line(df,x=df.index,y=['actual','predicted'],title=title)
    fig.update_layout(legend_title=legend_title, width=800,height=600)
    # legend=dict(yanchor="top",y=0.98,xanchor="left",x=0.02)
    return fig


def trade_simulation_plot(actual,pred,spread,title):
    threshold=[x for x in range(1,21)]
    mtm=[]

    for tr in threshold:
        mtm.append(trade_on_pred(actual,pred,spread,tr).loc['MTM','Total'])
    
    df_trade = pd.DataFrame({'threshold':threshold,'MTM':mtm})
    fig = px.line(df_trade, x='threshold',y='MTM',
                  markers=True,title=title,
                 labels={'threshold':'threshold (bp)','MTM':'MTM (bp)'})
    fig.update_layout(width=600,height=400)
    fig.update_xaxes(nticks=11)
    fig.update_yaxes(nticks=15)
    return fig


def residuals_dist_plot(actual,pred,hist_type,curve_type,title):
    pred_err=(pred-actual)*100
    group_labels = 'pred err'

    fig = ff.create_distplot([pred_err], [group_labels], bin_size=2, 
                             histnorm=hist_type,curve_type=curve_type,
                             colors=None, show_rug=False)
    fig.update_layout(title_text=title, bargap=0.05, width=600, height=400)
    fig.update_xaxes(nticks=20)
    fig.update_yaxes(nticks=10)
    return fig


def correlation_matrix_plot(df_corrs,title):
    fig = px.imshow(df_corrs.values, x=df_corrs.columns, y=df_corrs.index,
                    color_continuous_scale='portland',aspect="auto")
    fig.update_traces(text=df_corrs.values.round(3), texttemplate="%{text}")
    fig.update_layout(title=title,title_x=0.5)
    fig.update(layout_coloraxis_showscale=False)
    fig.update_xaxes(side="top",tickprefix="<b>")
    fig.update_yaxes(tickprefix="<b>")
    fig.update_layout(width=1000,height=600)
    # fig.update_yaxes(tickangle = -90,tickprefix="<b>",ticksuffix ="</b><br>")
    return fig;


def residuals_QQ_plot(actual,pred,title):
    res=actual-pred
    sm.qqplot(res,stats.norm,fit=True, line ='45')
    py.title(title, fontdict=None, loc='center', pad=None)
    return py

    