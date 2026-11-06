import os
import pandas as pd
import numpy as np
import pandas_ta_classic
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from itertools import combinations
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import warnings

RESULTS = 'results_yahoo'
os.makedirs(RESULTS, exist_ok=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")
tf.config.experimental.enable_op_determinism()
import matplotlib.pyplot as plt
ARCHS = ['BiRNN', 'GRU', 'LSTM']
UNITS = [16, 32, 64, 128, 256]
SEEDS = [1, 2, 3]
COMBOS = {'B,G,L': [0, 1, 2], 'B,G': [0, 1], 'B,L': [0, 2], 'G,L': [1, 2]}
META = ['ANN', 'GRU', 'LSTM']
CHOSEN = {'B': ('BiRNN', 128), 'G': ('GRU', 16), 'L': ('LSTM', 16)}

SYMBOLS = {'ADC': 2140, 'ALRAJHI': 1120, 'EMAAR EC': 4220, 'FITAIHI': 4180, 'GACO': 6020,
           'SARCO': 2030, 'SEC': 5110, 'SPIMACO': 2070, 'STC': 7010, 'TASNEE': 2060}
EPOCHS = 1000
FEATURES = ['open', 'high', 'low', 'close',
    'alma_5_6.0_0.85', 'ldecay_5', 'dema_5', 'ema_5', 'fwma_5', 'hma_5', 'hl2', 'hlc3',
    'hwma_0.2_0.1_0.1', 'jma_5_0', 'kama_5_2_30', 'lr_5', 'median_5', 'midpoint_5',
    'midprice_5', 'pwma_5', 'qtl_5_0.5', 'rma_5', 'sinwma_5', 'sma_5', 'ssf_5_2',
    'swma_5', 't3_5_0.7', 'tema_5', 'trima_5', 'vwap_d', 'vwma_5', 'wcp', 'wma_5', 'zl_ema_5']

# Yahoo Finance split (bounds inclusive). Yahoo's Saudi history is only clean after the
# 2013-06-29 weekend change (Thu-Fri -> Fri-Sat), so data starts there.
START     = '2013-06-29'
TRAIN_END = '2022-12-31'
VAL       = ('2023-01-01', '2023-12-31')
TEST      = ('2024-01-01', str((pd.Timestamp.today() - pd.Timedelta(days=1)).date()))  # today may be unfinished

import yfinance as yf

# Corporate actions Yahoo has no split record for: symbol -> [(effective date, ratio)].
MISSING_SPLITS = {6020: [('2017-02-14', 0.6)]}   # GACO 10:6 capital reduction

def load_yahoo(symbols, folder='data/yfinance', refresh=False):
    """Daily history of each <symbol>.SR in the Kaggle layout: symbol, date, open, high, low, close, volume.
    Downloads once and caches one CSV per symbol; refresh=True downloads again."""
    os.makedirs(folder, exist_ok=True)
    parts = []
    for sym in symbols:
        path = f'{folder}/{sym}.SR.csv'
        if refresh or not os.path.exists(path):
            df = yf.Ticker(f'{sym}.SR').history(period='max', auto_adjust=False)
            df.index = df.index.tz_localize(None).normalize().rename('date')
            df.to_csv(path)
        df = pd.read_csv(path, parse_dates=['date'])
        df.columns = df.columns.str.lower()
        df['volume'] = df['volume'].astype(float)
        for day, ratio in MISSING_SPLITS.get(sym, []):
            before = df['date'] < day
            df.loc[before, ['open', 'high', 'low', 'close']] /= ratio
            df.loc[before, 'volume'] *= ratio
        df = df.drop_duplicates('date', keep='last')
        parts.append(df[['date', 'open', 'high', 'low', 'close', 'volume']].assign(symbol=sym))
    raw = pd.concat(parts, ignore_index=True)
    return raw[raw['date'] >= START].reset_index(drop=True)

raw = load_yahoo(SYMBOLS.values())
raw.groupby('symbol')['date'].agg(['min', 'max', 'count'])

flat = lambda Xw: Xw.reshape(len(Xw), -1)

def load_ticker(raw,symbol,calendar_ffill=False):
    df = raw[raw['symbol'] == symbol].copy()
    df = df.set_index('date').sort_index()
    assert df.index.is_unique
    df = df.rename(columns={'volume_traded': 'volume'})

    if calendar_ffill:   # authors' code: every calendar day, all columns forward-filled
        days = pd.date_range(df.index.min(), df.index.max(), freq='D', name='date')
        df = df.reindex(days).ffill()[['open', 'high', 'low', 'close', 'volume']]
    else:                # trading days only (current behaviour)
        df = df[df['volume'] > 0]
        prev = df['close'].shift(1).fillna(df['close'])
        df['open'] = df['open'].fillna(prev)
        df['high'] = df['high'].fillna(np.maximum(prev, df['close']))
        df['low']  = df['low'].fillna(np.minimum(prev, df['close']))
        df = df[['open', 'high', 'low', 'close', 'volume']]

    assert not df.isna().any().any()
    return df

def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train / val / test on the configured dates (bounds inclusive). Never shuffled."""
    train = df.loc[:TRAIN_END]
    val = df.loc[VAL[0]:VAL[1]]
    test = df.loc[TEST[0]:TEST[1]]
    return train, val, test

def all_indicators(df):
    parts = [df[['open','high','low','close','volume']]]
    failed = {}
    for name in df.ta.indicators(as_list=True):
        try:
            out = getattr(df.ta,name)(length=5, lookahead = False)
        except Exception as e:
            failed[name] = type(e).__name__
            continue
        if out is None:
            continue
        elif isinstance(out, pd.Series):
            # out.name = out.name.split("_", 1)[0]
            out.name = out.name.lower()
        elif isinstance(out, pd.DataFrame):
            # out.columns =  out.columns.str.split('_').str[0]
            out.columns =  out.columns.str.lower()
        parts.append(out)
    X = pd.concat(parts,axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how='all')
    assert X.columns.is_unique
    return X, failed

def prepare(raw,symbol,calendar_ffill=False):
    X = all_indicators(load_ticker(raw,symbol,calendar_ffill))[0][FEATURES].dropna()
    train, val, test = split(X)
    x_scaler = StandardScaler().fit(train[FEATURES])
    y_scaler = StandardScaler().fit(train[["close"]])
    def scale(part):
        xs = pd.DataFrame(x_scaler.transform(part[FEATURES]), index=part.index, columns=FEATURES)
        ys = pd.Series(y_scaler.transform(part[['close']]).ravel(), index=part.index, name='y')
        return xs, ys
    return scale(train), scale(val), scale(test), x_scaler, y_scaler
def make_windows(X,y,w=5,h=5):
    n = len(X) - w - h + 1
    Xw = np.stack([X[i:i + w] for i in range(n)])
    Yw = np.stack([y[i + w:i + w + h] for i in range(n)])
    return Xw, Yw

def build_model(arch, units, lr=1e-3, w=5, n_features=34, h=5):
    layer = {
        'GRU':   lambda: keras.layers.GRU(units, activation='relu'),
        'LSTM':  lambda: keras.layers.LSTM(units, activation='relu'),
        'BiRNN': lambda: keras.layers.Bidirectional(keras.layers.SimpleRNN(units, activation='relu')),
    }[arch]()
    model = keras.Sequential([keras.Input(shape=(w, n_features)), layer, keras.layers.Dense(h)])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse')
    return model
def fit(model, Xtr, ytr, Xva, yva, epochs, label=''):
    stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20,
                                         restore_best_weights=True)
    history = model.fit(Xtr, ytr, validation_data=(Xva, yva),
                        epochs=epochs, batch_size=int(0.1 * len(Xtr)), shuffle=True,
                        callbacks=[stop], verbose=0)
    print(f'{label:16s} epochs {len(history.history["loss"]):4d}  val {min(history.history["val_loss"]):.6f}')
    return history, model

def train(arch, units, epochs, Xtr, ytr, Xva, yva, seed=42):
    keras.utils.set_random_seed(seed)
    model = build_model(arch, units, 1e-3)
    return fit(model, Xtr, ytr, Xva, yva, epochs, label=f'{arch} {units}')
def metrics(pred, y, y_scaler):
    p = y_scaler.inverse_transform(pred.reshape(-1, 1)).reshape(pred.shape)
    t = y_scaler.inverse_transform(y.reshape(-1, 1)).reshape(y.shape)
    e = p - t
    return {
        'MAE':  np.mean(np.abs(e)),
        'MSE':  np.mean(e ** 2),
        'RMSE': np.sqrt(np.mean(e ** 2)),
        'MRE':  np.mean(np.abs(e) / np.abs(t)),
    }
def build_meta(kind, n_base, lr=1e-3, h=5):
    body = {
        'ANN':  lambda: [keras.layers.Flatten(),
                         keras.layers.Dense(128, activation='relu'),
                         keras.layers.Dense(64, activation='relu'),
                         keras.layers.Dense(32, activation='relu')],
        'GRU':  lambda: [keras.layers.GRU(128, activation='relu')],
        'LSTM': lambda: [keras.layers.LSTM(128, activation='relu')],
    }[kind]()
    model = keras.Sequential([keras.Input(shape=(h, n_base)), *body, keras.layers.Dense(h)])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse')
    return model
def oof_folds(n, n_blocks=5, first=0.4, n_es=250, gap=5):
    """
    Walk-forward folds over n training windows.

        |------ fit ------| gap |--- es ---| gap |-- block --|
          train the model        early stop       predict (OOF)

    The first 40% is never predicted; the rest is cut into n_blocks blocks.
    """
    cuts = np.linspace(int(n * first), n, n_blocks + 1).astype(int)

    for block_start, block_end in zip(cuts[:-1], cuts[1:]):
        es_end   = block_start - gap        # leave a gap before the block
        es_start = es_end - n_es            # ~1 year for early stopping
        fit_end  = es_start - gap           # leave a gap before early stopping

        fit   = slice(0, fit_end)
        es    = slice(es_start, es_end)
        block = slice(block_start, block_end)
        yield fit, es, block
def naive_pred(Xw, h=5):
    ci = FEATURES.index('close')
    return np.repeat(Xw[:, -1, ci:ci+1], h, axis=1)

def naive_mse(Xw, yw):
    return np.mean((naive_pred(Xw) - yw) ** 2)        
def run_stacking(X_meta, y_meta, X_meta_va, y_va, tag=''):
    out, models = {}, {}
    for combo, cols in COMBOS.items():
        for kind in META:
            name = f'STK_{kind}({combo}){tag}'
            keras.utils.set_random_seed(42)
            meta = build_meta(kind, len(cols))
            _, meta = fit(meta, X_meta[:, :, cols], y_meta,
                          X_meta_va[:, :, cols], y_va, EPOCHS, label=name)
            out[name] = meta.predict(X_meta_va[:, :, cols], verbose=0)
            models[name] = (meta, cols)
    return out, models
def drift_pred(Xw, ytr, h=5):
    d = np.diff(ytr).mean()                       # mean daily change, train only
    return naive_pred(Xw, h) + d * np.arange(1, h + 1)

def arima_pred(y_fit, y_target, order, y_between=None, w=5, h=5):
    res = ARIMA(y_fit.values, order=order).fit()
    before = [y_fit] if y_between is None else [y_fit, y_between]
    full = res.apply(np.concatenate([s.values for s in before + [y_target]]))
    offset = sum(len(s) for s in before)
    out = []
    for i in range(len(y_target) - w - h + 1):
        origin = offset + i + w - 1
        out.append(full.get_prediction(start=origin + 1, end=origin + h, dynamic=True).predicted_mean)
    return np.array(out)
def window_rmse(pred, y, y_scaler, step=5):
    p = y_scaler.inverse_transform(pred.reshape(-1, 1)).reshape(pred.shape)
    t = y_scaler.inverse_transform(y.reshape(-1, 1)).reshape(y.shape)
    return np.sqrt(((p - t) ** 2).mean(axis=1))[::step]

def win_loss(preds, y, y_scaler, alpha=0.05):
    r = {k: window_rmse(p, y, y_scaler) for k, p in preds.items()}
    pairs = list(combinations(r, 2))
    pvals = [wilcoxon(r[a], r[b]).pvalue for a, b in pairs]
    reject = multipletests(pvals, alpha=alpha, method='holm')[0]
    score = {k: {'win': 0, 'loss': 0, 'tie': 0} for k in r}
    for (a, b), sig in zip(pairs, reject):
        if not sig:
            score[a]['tie'] += 1; score[b]['tie'] += 1
        elif np.median(r[a] - r[b]) < 0:
            score[a]['win'] += 1; score[b]['loss'] += 1
        else:
            score[b]['win'] += 1; score[a]['loss'] += 1
    df = pd.DataFrame(score).T
    df['win_rate'] = df['win'] / (df['win'] + df['loss']).replace(0, np.nan)
    return df.sort_values(['win', 'loss'], ascending=[False, True])

def add_averages(p):
    p['AVG(B,G,L)'] = (p['B'] + p['G'] + p['L']) / 3
    p['AVG(B,G)']   = (p['B'] + p['G']) / 2
    p['AVG(B,L)']   = (p['B'] + p['L']) / 2
    p['AVG(G,L)']   = (p['G'] + p['L']) / 2
    return p

def oof_predictions(Xwtr, ywtr, seed=42):
    X_blocks, y_blocks = [], []
    for fit_sl, es_sl, block_sl in oof_folds(len(Xwtr)):
        block_preds = []
        for arch, units in CHOSEN.values():
            _, m = train(arch, units, EPOCHS, Xwtr[fit_sl], ywtr[fit_sl],
                         Xwtr[es_sl], ywtr[es_sl], seed=seed)
            block_preds.append(m.predict(Xwtr[block_sl], verbose=0))
        X_blocks.append(np.stack(block_preds, axis=-1))
        y_blocks.append(ywtr[block_sl])
    return np.concatenate(X_blocks), np.concatenate(y_blocks)

def run_ticker(symbol, seed=42):
    keras.backend.clear_session()                        # free memory from the previous ticker
    (xtr, ytr), (xva, yva), (xte, yte), _, ys = prepare(raw, symbol)
    Xwtr, ywtr = make_windows(xtr, ytr)
    Xwva, ywva = make_windows(xva, yva)
    Xwte, ywte = make_windows(xte, yte)

    # base models + averages
    base = {k: train(a, u, EPOCHS, Xwtr, ywtr, Xwva, ywva, seed)[1] for k, (a, u) in CHOSEN.items()}
    pv = add_averages({k: m.predict(Xwva, verbose=0) for k, m in base.items()})
    pt = add_averages({k: m.predict(Xwte, verbose=0) for k, m in base.items()})

    # stacking on full OOF
    X_meta, y_meta = oof_predictions(Xwtr, ywtr, seed)
    meta_va = np.stack([pv['B'], pv['G'], pv['L']], axis=-1)
    meta_te = np.stack([pt['B'], pt['G'], pt['L']], axis=-1)
    stk_va, stk_models = run_stacking(X_meta, y_meta, meta_va, ywva)
    pv |= stk_va
    for name, (meta, cols) in stk_models.items():
        pt[name] = meta.predict(meta_te[:, :, cols], verbose=0)

    # baselines
    pv['Naive'], pt['Naive'] = naive_pred(Xwva), naive_pred(Xwte)
    pv['RW drift'], pt['RW drift'] = drift_pred(Xwva, ytr), drift_pred(Xwte, ytr)

    aic = {(p, 1, q): ARIMA(ytr.values, order=(p, 1, q)).fit().aic for p in range(3) for q in range(3)}
    order = min(aic, key=aic.get)
    pv['ARIMA'] = arima_pred(ytr, yva, order)
    pt['ARIMA'] = arima_pred(ytr, yte, order, y_between=yva)

    xgb = XGBRegressor(n_estimators=2000, learning_rate=0.05, max_depth=4,
                       tree_method='hist', early_stopping_rounds=50, random_state=seed)
    xgb.fit(flat(Xwtr), ywtr, eval_set=[(flat(Xwva), ywva)], verbose=False)
    pv['XGBoost'], pt['XGBoost'] = xgb.predict(flat(Xwva)), xgb.predict(flat(Xwte))

    _, gru256 = train('GRU', 256, EPOCHS, Xwtr, ywtr, Xwva, ywva, seed)
    pv['GRU-256'], pt['GRU-256'] = gru256.predict(Xwva, verbose=0), gru256.predict(Xwte, verbose=0)

    val  = pd.DataFrame({k: metrics(p, ywva, ys) for k, p in pv.items()}).T
    test = pd.DataFrame({k: metrics(p, ywte, ys) for k, p in pt.items()}).T
    return val, test, win_loss(pt, ywte, ys), order


def main():
    for name, sym in SYMBOLS.items():
        if os.path.exists(f'{RESULTS}/{name}_winloss.csv'):
            continue                                  # already done: allows resuming after a crash
        print('=' * 20, name)
        val, test, wl, order = run_ticker(sym)
        val.to_csv(f'{RESULTS}/{name}_val.csv')
        test.to_csv(f'{RESULTS}/{name}_test.csv')
        wl.to_csv(f'{RESULTS}/{name}_winloss.csv')
        print(name, 'ARIMA', order)

if __name__ == "__main__":
    main()    