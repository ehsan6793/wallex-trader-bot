import sys, json, pandas as pd, tempfile, pathlib
from bot.config import Settings
from bot.backtest import run
def load(syms):
    out={}
    for s in syms:
        g=pd.read_csv(f'data_snapshot/{s}.csv',index_col=0,parse_dates=True)
        out[s]=g
    return out
if __name__=='__main__':
    syms=sys.argv[1].split(',')
    s=Settings(); s.log_dir=pathlib.Path(tempfile.mkdtemp()); s.data_dir=s.log_dir
    for kv in sys.argv[2:]:
        k,v=kv.split('='); setattr(s,k,type(getattr(s,k))(v))
    res=run(load(syms),s,test_days=float(__import__("os").environ.get("TEST_DAYS","7")),verbose=False)
    m=res['metrics']
    print(json.dumps({k:m[k] for k in ['return_pct','max_drawdown_pct','trades','win_rate_pct','avg_trade_ret_pct','profit_factor','buy_hold_benchmark_pct'] if k in m}, ensure_ascii=False))
    print([ (x['time'][5:16], x['threshold'], x['valid_auc'], x['no_edge']) for x in res['models']])
    if len(res['trades']): print(res['trades'][['symbol','entry_time','ret','exit_reason']].to_string())
