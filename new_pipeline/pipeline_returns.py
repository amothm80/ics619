"""Yahoo pipeline, change-target variant. Copy of pipeline_yahoo.py with five changes:

1. Data: prices dividend-adjusted (Yahoo Adj Close), automated checks, any ticker of the TASI list.
2. Target: the next 5 closes as returns relative to the last input close; inputs relative to the close.
3. Global mode: one model per architecture trained on all eligible tickers at once.
4. Shrinkage toward naive: forecast = last + lam * (model - last), lam chosen per ticker on val.
5. iTransformer (T) and DA-RNN (D) as extra base models, in the averages, stacking and shrinkage.

Errors are reported in SAR on Yahoo's Close (split-adjusted, not dividend-adjusted), as in
pipeline_yahoo.py, so the tables compare directly. Paths are relative to this file:

    python pipeline_returns.py --mode data --tickers all     # step 1: checks + naive reference
    python pipeline_returns.py --mode per_ticker             # step 2 (+4) on the ten paper tickers
    python pipeline_returns.py --mode sweep_global           # step 3a: unit sizes for the global models
    python pipeline_returns.py --mode global                 # step 3 (+4)
"""
import os
import argparse
import warnings
import numpy as np
import pandas as pd
import pandas_ta_classic  # noqa: F401  (registers df.ta)
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from itertools import combinations
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")
tf.config.experimental.enable_op_determinism()

HERE = os.path.dirname(os.path.abspath(__file__))      # paths work from any working directory
RESULTS = f'{HERE}/results_returns'
DATA = f'{HERE}/data/yfinance'
SYMBOL_LIST = f'{HERE}/tasi_symbols.csv'

# Same split as pipeline_yahoo.py. The test end is fixed to the earlier run's last day, so new
# downloads do not change the test set.
START = '2013-06-29'
SPLITS = {'train': (START, '2022-12-31'), 'val': ('2023-01-01', '2023-12-31'),
          'test': ('2024-01-01', '2026-10-28')}
W, H = 5, 5
EPOCHS = 1000
MAX_JUMP = 0.25                     # |daily return| above this: unrecorded corporate action or bad data
MIN_ROWS = {'train': 500, 'val': 200, 'test': 250}   # test: >= 50 non-overlapping windows for Wilcoxon
SWEEP = {'BiRNN': [64, 128], 'GRU': [16, 64, 128], 'LSTM': [16, 64, 128],   # global unit sweep
         'iTransformer': [32, 64, 128],                                     # d_model
         'DA-RNN': [32, 64, 128]}                                           # encoder = decoder units
LAMBDAS = np.round(np.linspace(0, 1, 21), 2)

COMBOS = {'B,G,L': [0, 1, 2], 'B,G': [0, 1], 'B,L': [0, 2], 'G,L': [1, 2], 'B,G,L,T,D': [0, 1, 2, 3, 4]}
META = ['ANN', 'GRU', 'LSTM']
CHOSEN = {'B': ('BiRNN', 128), 'G': ('GRU', 16), 'L': ('LSTM', 16), 'T': ('iTransformer', 64),
          'D': ('DA-RNN', 64)}
AVERAGES = {'AVG(B,G,L)': 'BGL', 'AVG(B,G)': 'BG', 'AVG(B,L)': 'BL', 'AVG(G,L)': 'GL',
            'AVG(B,G,L,T,D)': 'BGLTD', 'AVG(G,T)': 'GT', 'AVG(G,D)': 'GD'}
SHRUNK = list(CHOSEN) + list(AVERAGES)

SYMBOLS = {'ADC': 2140, 'ALRAJHI': 1120, 'EMAAR EC': 4220, 'FITAIHI': 4180, 'GACO': 6020,
           'SARCO': 2030, 'SEC': 5110, 'SPIMACO': 2070, 'STC': 7010, 'TASNEE': 2060}
PAPER = {code: name for name, code in SYMBOLS.items()}
MISSING_SPLITS = {6020: [('2017-02-14', 0.6)]}   # GACO 10:6 capital reduction, not in Yahoo

FEATURES = ['open', 'high', 'low', 'close',
    'alma_5_6.0_0.85', 'ldecay_5', 'dema_5', 'ema_5', 'fwma_5', 'hma_5', 'hl2', 'hlc3',
    'hwma_0.2_0.1_0.1', 'jma_5_0', 'kama_5_2_30', 'lr_5', 'median_5', 'midpoint_5',
    'midprice_5', 'pwma_5', 'qtl_5_0.5', 'rma_5', 'sinwma_5', 'sma_5', 'ssf_5_2',
    'swma_5', 't3_5_0.7', 'tema_5', 'trima_5', 'vwap_d', 'vwma_5', 'wcp', 'wma_5', 'zl_ema_5']
N_FEATURES = len(FEATURES)       # 33 ratios to the close + 1 log return


# ---------------------------------------------------------------- 1. data

def universe(which):
    """code -> label. 'paper': the ten paper tickers; 'all': the TASI list without REITs."""
    if which == 'paper':
        return dict(PAPER)
    t = pd.read_csv(SYMBOL_LIST)
    t = t[t['sector'] != 'REITs']
    return {c: PAPER.get(c, str(c)) for c in t['code']}


def load_symbol(code, refresh=False):
    """Clean daily bars of <code>.SR from START to the test end, or None if Yahoo has nothing.
    open/high/low/close are dividend-adjusted; close_raw is Yahoo's Close, used for errors in SAR."""
    path = f'{DATA}/{code}.SR.csv'
    if refresh or not os.path.exists(path):
        import yfinance as yf
        os.makedirs(DATA, exist_ok=True)
        df = yf.Ticker(f'{code}.SR').history(period='max', auto_adjust=False)
        if df.empty:
            return None, {}
        df.index = df.index.tz_localize(None).normalize().rename('date')
        df.to_csv(path)
    df = pd.read_csv(path, parse_dates=['date'])
    if df.empty:
        return None, {}
    df.columns = df.columns.str.lower()
    df = df.drop_duplicates('date', keep='last').set_index('date').sort_index()
    df['volume'] = df['volume'].astype(float)
    prices = ['open', 'high', 'low', 'close', 'adj close']
    for day, ratio in MISSING_SPLITS.get(code, []):
        df.loc[df.index < day, prices] /= ratio
        df.loc[df.index < day, 'volume'] *= ratio
    df = df.loc[START:SPLITS['test'][1]]

    # Zero-volume rows are Yahoo placeholders (flat bar = previous close), not real sessions.
    report = {'placeholders': int((df['volume'] <= 0).sum()),
              'placeholders_val': int((df.loc[slice(*SPLITS['val']), 'volume'] <= 0).sum())}
    df = df[(df['volume'] > 0) & (df['close'] > 0)]

    prev = df['close'].shift(1).fillna(df['close'])
    df['open'] = df['open'].fillna(prev)
    df['high'] = df['high'].fillna(np.maximum(prev, df['close']))
    df['low'] = df['low'].fillna(np.minimum(prev, df['close']))

    factor = df['adj close'].fillna(df['close']) / df['close']
    out = df[['open', 'high', 'low', 'close']].mul(factor, axis=0)
    out['volume'] = df['volume']
    out['close_raw'] = df['close']
    return out, report


def check(df):
    """Cut the history at the last daily move above MAX_JUMP (Tadawul's limit is +-10%;
    the first 5 days after listing have no limit). Returns the kept bars and a report."""
    r = df['close'].pct_change()
    r.iloc[:5] = 0
    jumps = r.index[r.abs() > MAX_JUMP]
    report = {'first': df.index[0].date(), 'last': df.index[-1].date(),
              'moves_over_10pct': int((r.abs() > 0.105).sum()), 'jumps': len(jumps),
              'cut_before': jumps[-1].date() if len(jumps) else None}
    if len(jumps):
        df = df.loc[jumps[-1]:]
    return df, report


def all_indicators(df):
    parts = [df[['open', 'high', 'low', 'close', 'volume']]]
    for name in df.ta.indicators(as_list=True):
        try:
            out = getattr(df.ta, name)(length=5, lookahead=False)
        except Exception:
            continue
        if out is None:
            continue
        elif isinstance(out, pd.Series):
            out.name = out.name.lower()
        elif isinstance(out, pd.DataFrame):
            out.columns = out.columns.str.lower()
        parts.append(out)
    X = pd.concat(parts, axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    return X.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')


def features(df):
    """The paper's 34 indicators, made scale-free: each one / close - 1, and close -> log return."""
    lv = all_indicators(df)[FEATURES]
    F = lv.drop(columns='close').div(lv['close'], axis=0) - 1
    F['ret'] = np.log(lv['close']).diff()                    # last column: target series of T and D
    return F.replace([np.inf, -np.inf], np.nan).dropna()


def ticker_data(code, refresh=False):
    """((bars, {split: feature rows}) or None, report) for one ticker."""
    report = {'code': code}
    df, rep = load_symbol(code, refresh)
    if df is None or len(df) < 100:
        return None, report | {'status': 'no data'}
    df, rep2 = check(df)
    report |= rep | rep2
    try:
        F = features(df)
    except KeyError:
        return None, report | {'status': 'indicator missing'}
    parts = {s: F.loc[a:b] for s, (a, b) in SPLITS.items()}
    report |= {f'n_{s}': len(p) for s, p in parts.items()}
    short = [s for s, p in parts.items() if len(p) < MIN_ROWS[s]]
    if short:
        return None, report | {'status': 'too short: ' + ','.join(short)}
    return (df, parts), report | {'status': 'ok'}


def windows(F, df):
    """Windows of one split. X: inputs (n, W, 34), unscaled. R: next H returns relative to the last
    input close (adjusted prices). In SAR (close_raw): last input close, input closes, true closes."""
    c = df.loc[F.index, 'close'].to_numpy()
    raw = df.loc[F.index, 'close_raw'].to_numpy()
    o = np.arange(len(F) - W - H + 1) + W - 1             # forecast origin = last input row
    inp = o[:, None] + np.arange(-W + 1, 1)
    fut = o[:, None] + np.arange(1, H + 1)
    return {'X': F.to_numpy()[inp], 'R': c[fut] / c[o, None] - 1,
            'last': raw[o], 'inp': raw[inp], 'true': raw[fut], 'close': raw}


def load_universe(codes, refresh=False):
    """{label: (bars, parts)} for every usable ticker; writes the data report."""
    data, rows = {}, []
    for code, label in codes.items():
        d, rep = ticker_data(code, refresh)
        if d is not None:
            data[label] = d
            s = windows(d[1]['test'], d[0])
            naive = np.repeat(s['last'][:, None], H, axis=1)
            rep['naive_test_rmse'] = rmse(naive, s['true'])
        rows.append({'label': label} | rep)
        print(f'{label:10s} {rep["status"]}')
    os.makedirs(RESULTS, exist_ok=True)
    pd.DataFrame(rows).to_csv(f'{RESULTS}/data_report.csv', index=False)
    print(f'{len(data)} of {len(codes)} tickers usable; report in {RESULTS}/data_report.csv')
    return data


# ---------------------------------------------------------------- models (as pipeline_yahoo.py)

def build_itransformer(d_model, n_layers=2, n_heads=4, dropout=0.1, lr=1e-3):
    """iTransformer (Liu et al., ICLR 2024): each of the 34 input series becomes one token (its W
    values embedded to d_model); attention runs across the series, the feed-forward net within each
    token. The forecast is read from the token of the target series (the log return, last column)."""
    inp = keras.Input(shape=(W, N_FEATURES))
    x = keras.layers.Permute((2, 1))(inp)                       # (34 tokens, W)
    x = keras.layers.Dense(d_model)(x)                          # embed each series
    for _ in range(n_layers):                                   # post-norm encoder, as in the paper
        a = keras.layers.MultiHeadAttention(n_heads, d_model // n_heads, dropout=dropout)(x, x)
        x = keras.layers.LayerNormalization()(x + keras.layers.Dropout(dropout)(a))
        f = keras.layers.Dense(2 * d_model, activation='gelu')(x)
        f = keras.layers.Dropout(dropout)(keras.layers.Dense(d_model)(f))
        x = keras.layers.LayerNormalization()(x + f)
    x = keras.layers.Cropping1D((N_FEATURES - 1, 0))(x)         # keep the target token
    out = keras.layers.Dense(H)(keras.layers.Flatten()(x))
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse')
    return model


@keras.utils.register_keras_serializable(package="ics619")
class DARNN(keras.layers.Layer):
    """Dual-stage attention RNN (Qin et al., IJCAI 2017), unrolled over the W input days.
    Encoder: an LSTM whose input at day t is x_t weighted by input attention over the 34 series,
    conditioned on the previous encoder state. Decoder: an LSTM over the target series (log return,
    last column) with temporal attention over all encoder states. Returns [d_W; c_W], from which a
    Dense layer forecasts the next H returns at once (the paper predicts one step)."""

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        _, T, n = input_shape
        m = p = self.units
        self.enc, self.dec = keras.layers.LSTMCell(m), keras.layers.LSTMCell(p)
        self.We, self.Ue, self.ve = keras.layers.Dense(T), keras.layers.Dense(T, use_bias=False), keras.layers.Dense(1, use_bias=False)
        self.Wd, self.Ud, self.vd = keras.layers.Dense(m), keras.layers.Dense(m, use_bias=False), keras.layers.Dense(1, use_bias=False)
        self.w_tilde = keras.layers.Dense(1)
        self.enc.build((None, n))
        self.dec.build((None, 1))
        self.We.build((None, 2 * m)); self.Ue.build((None, n, T)); self.ve.build((None, n, T))
        self.Wd.build((None, 2 * p)); self.Ud.build((None, T, m)); self.vd.build((None, T, m))
        self.w_tilde.build((None, 1 + m))

    def call(self, x):
        ops = keras.ops
        T = x.shape[1]
        zeros = lambda k: ops.zeros((ops.shape(x)[0], k))
        h, s = zeros(self.units), zeros(self.units)
        Ux = self.Ue(ops.transpose(x, (0, 2, 1)))                        # (batch, n, T)
        states = []
        for t in range(T):                                               # encoder with input attention
            e = self.ve(ops.tanh(ops.expand_dims(self.We(ops.concatenate([h, s], -1)), 1) + Ux))
            alpha = ops.softmax(ops.squeeze(e, -1), axis=-1)             # weight of each series
            h, [h, s] = self.enc(alpha * x[:, t, :], [h, s])
            states.append(h)
        Hs = ops.stack(states, axis=1)                                   # (batch, T, m)
        UH = self.Ud(Hs)
        y = x[:, :, -1:]                                                 # target series
        d, sd = zeros(self.units), zeros(self.units)

        def context(d, sd):                                              # temporal attention
            l = self.vd(ops.tanh(ops.expand_dims(self.Wd(ops.concatenate([d, sd], -1)), 1) + UH))
            beta = ops.softmax(l, axis=1)                                # weight of each day
            return ops.sum(beta * Hs, axis=1)

        for t in range(T):                                               # decoder
            c = context(d, sd)
            d, [d, sd] = self.dec(self.w_tilde(ops.concatenate([y[:, t], c], -1)), [d, sd])
        return ops.concatenate([d, context(d, sd)], -1)

    def get_config(self):
        return super().get_config() | {'units': self.units}


def build_darnn(units, lr=1e-3):
    inp = keras.Input(shape=(W, N_FEATURES))
    x = keras.layers.Dense(units)(DARNN(units)(inp))                    # W_y [d; c] + b, as in the paper
    model = keras.Model(inp, keras.layers.Dense(H)(x))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse')
    return model


def build_model(arch, units, lr=1e-3):
    if arch == 'iTransformer':
        return build_itransformer(units, lr=lr)
    if arch == 'DA-RNN':
        return build_darnn(units, lr=lr)
    layer = {
        'GRU': lambda: keras.layers.GRU(units, activation='relu'),
        'LSTM': lambda: keras.layers.LSTM(units, activation='relu'),
        'BiRNN': lambda: keras.layers.Bidirectional(keras.layers.SimpleRNN(units, activation='relu')),
    }[arch]()
    model = keras.Sequential([keras.Input(shape=(W, N_FEATURES)), layer, keras.layers.Dense(H)])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse')
    return model


def build_meta(kind, n_base, lr=1e-3):
    body = {
        'ANN': lambda: [keras.layers.Flatten(),
                        keras.layers.Dense(128, activation='relu'),
                        keras.layers.Dense(64, activation='relu'),
                        keras.layers.Dense(32, activation='relu')],
        'GRU': lambda: [keras.layers.GRU(128, activation='relu')],
        'LSTM': lambda: [keras.layers.LSTM(128, activation='relu')],
    }[kind]()
    model = keras.Sequential([keras.Input(shape=(H, n_base)), *body, keras.layers.Dense(H)])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse')
    return model


def fit(model, Xtr, ytr, Xva, yva, epochs, label='', batch=None):
    """Early stopping on val loss (patience 20, best weights). batch=None: 10% of the training set."""
    stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    history = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs,
                        batch_size=batch or int(0.1 * len(Xtr)), shuffle=True,
                        callbacks=[stop], verbose=0)
    print(f'{label:16s} epochs {len(history.history["loss"]):4d}  val {min(history.history["val_loss"]):.6f}')
    return model


def train(arch, units, Xtr, ytr, Xva, yva, seed=42, batch=None):
    keras.utils.set_random_seed(seed)
    return fit(build_model(arch, units), Xtr, ytr, Xva, yva, EPOCHS, f'{arch} {units}', batch)


def oof_folds(n, n_blocks=5, first=0.4, n_es=250, gap=5):
    """Walk-forward folds: |-- fit --| gap |-- es --| gap |-- block --|; first 40% never predicted."""
    cuts = np.linspace(int(n * first), n, n_blocks + 1).astype(int)
    for start, end in zip(cuts[:-1], cuts[1:]):
        es_end = start - gap
        es_start = es_end - n_es
        yield slice(0, es_start - gap), slice(es_start, es_end), slice(start, end)


def oof_predictions(Xtr, ytr, seed=42):
    X_blocks, y_blocks = [], []
    for fit_sl, es_sl, block in oof_folds(len(Xtr)):
        preds = [train(a, u, Xtr[fit_sl], ytr[fit_sl], Xtr[es_sl], ytr[es_sl], seed).predict(Xtr[block], verbose=0)
                 for a, u in CHOSEN.values()]
        X_blocks.append(np.stack(preds, axis=-1))
        y_blocks.append(ytr[block])
    return np.concatenate(X_blocks), np.concatenate(y_blocks)


# ---------------------------------------------------------------- forecasts in SAR

def to_sar(s, r):
    """Predicted returns -> predicted closes in SAR."""
    return s['last'][:, None] * (1 + r)


def add_averages(p):
    for name, members in AVERAGES.items():
        p[name] = sum(p[k] for k in members) / len(members)
    return p


def arima_pred(y_fit, y_target, order, y_between=None):
    """Rolling-origin ARIMA with parameters fixed on train; 5-step dynamic forecast per window."""
    res = ARIMA(y_fit, order=order).fit()
    before = [y_fit] if y_between is None else [y_fit, y_between]
    full = res.apply(np.concatenate(before + [y_target]))
    offset = sum(len(s) for s in before)
    return np.array([full.get_prediction(start=offset + i + W, end=offset + i + W + H - 1,
                                         dynamic=True).predicted_mean
                     for i in range(len(y_target) - W - H + 1)])


def baselines(s, parts, arima=False):
    """Naive, RW with train drift, the authors' RW and ARIMA; rolling ARIMA if arima=True."""
    k = np.arange(1, H + 1)
    mu = parts['train']['ret'].mean()                        # mean daily log return on train
    ar_paper = ARIMA(s['train']['close'], order=(5, 1, 0)).fit().forecast(H)
    out = {}
    for sp in ('val', 'test'):
        w = s[sp]
        out[sp] = {
            'Naive': np.repeat(w['last'][:, None], H, axis=1),
            'RW drift': w['last'][:, None] * np.exp(mu * k),
            'RW (paper)': w['last'][:, None] + np.diff(w['inp'], axis=1).mean(axis=1, keepdims=True) * k,
            'ARIMA (paper)': np.tile(ar_paper, (len(w['last']), 1)),
        }
    if arima:
        y = {sp: s[sp]['close'] for sp in s}
        aic = {(p, 1, q): ARIMA(y['train'], order=(p, 1, q)).fit().aic for p in range(3) for q in range(3)}
        order = min(aic, key=aic.get)
        out['val']['ARIMA'] = arima_pred(y['train'], y['val'], order)
        out['test']['ARIMA'] = arima_pred(y['train'], y['test'], order, y_between=y['val'])
    return out


def shrink(pv, pt, s):
    """SHR_<model> = last + lam * (model - last), lam on a 0.05 grid chosen by val RMSE."""
    lams = {}
    for k in SHRUNK:
        last = s['val']['last'][:, None]
        err = {lam: rmse(last + lam * (pv[k] - last), s['val']['true']) for lam in LAMBDAS}
        lam = min(err, key=err.get)
        lams[k] = lam
        for p, sp in ((pv, 'val'), (pt, 'test')):
            last = s[sp]['last'][:, None]
            p[f'SHR_{k}'] = last + lam * (p[k] - last)
    return lams


# ---------------------------------------------------------------- evaluation

def rmse(p, t):
    return float(np.sqrt(np.mean((p - t) ** 2)))


def metrics_table(preds, true):
    rows = {}
    for k, p in preds.items():
        e = p - true
        rows[k] = {'MAE': np.mean(np.abs(e)), 'MSE': np.mean(e ** 2),
                   'RMSE': np.sqrt(np.mean(e ** 2)), 'MRE': np.mean(np.abs(e) / np.abs(true))}
    df = pd.DataFrame(rows).T
    df['vs_naive_%'] = 100 * (df['RMSE'] / df.loc['Naive', 'RMSE'] - 1)
    return df.sort_values('RMSE')


def win_loss(preds, true, alpha=0.05):
    """Pairwise Wilcoxon on per-window RMSE (non-overlapping 5-day windows), Holm-corrected."""
    r = {k: np.sqrt(((p - true) ** 2).mean(axis=1))[::H] for k, p in preds.items()}
    pairs = list(combinations(r, 2))
    pvals = [wilcoxon(r[a], r[b]).pvalue if np.any(r[a] != r[b]) else 1.0 for a, b in pairs]
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


def save(out, label, pv, pt, s, lams):
    val, test = metrics_table(pv, s['val']['true']), metrics_table(pt, s['test']['true'])
    val.to_csv(f'{out}/{label}_val.csv')
    test.to_csv(f'{out}/{label}_test.csv')
    win_loss(pt, s['test']['true']).to_csv(f'{out}/{label}_winloss.csv')
    pd.Series(lams, name='lambda').to_csv(f'{out}/{label}_lambda.csv')
    print(test[['RMSE', 'vs_naive_%']].head(8).round(4).to_string())


def summary(out, labels, name):
    """RMSE vs naive (%) per ticker and model, plus pooled win/loss/tie, for the given tickers."""
    labels = [l for l in labels if os.path.exists(f'{out}/{l}_winloss.csv')]
    if not labels:
        return
    vs = pd.DataFrame({l: pd.read_csv(f'{out}/{l}_test.csv', index_col=0)['vs_naive_%'] for l in labels})
    wl = sum(pd.read_csv(f'{out}/{l}_winloss.csv', index_col=0)[['win', 'loss', 'tie']] for l in labels)
    table = pd.DataFrame({'mean_vs_naive_%': vs.mean(axis=1), 'median_vs_naive_%': vs.median(axis=1),
                          'tickers_beating_naive': (vs < 0).sum(axis=1)}).join(wl)
    table['win_rate'] = table['win'] / (table['win'] + table['loss']).replace(0, np.nan)
    table = table.sort_values('mean_vs_naive_%')
    table.to_csv(f'{out}/summary_{name}.csv')
    vs.round(2).to_csv(f'{out}/vs_naive_{name}.csv')
    print(f'\n== {name}: {len(labels)} tickers\n{table.round(3).to_string()}')


# ---------------------------------------------------------------- 2. per-ticker models

def run_ticker(df, parts, seed=42):
    """Full framework on one ticker: bases, averages, OOF stacking, baselines, XGBoost, GRU-256, shrinkage."""
    keras.backend.clear_session()
    s = {sp: windows(parts[sp], df) for sp in SPLITS}
    xs = StandardScaler().fit(parts['train'].to_numpy())
    ys = StandardScaler().fit(s['train']['R'].reshape(-1, 1))
    X = {sp: ((s[sp]['X'] - xs.mean_) / xs.scale_).astype('float32') for sp in s}
    Y = {sp: ((s[sp]['R'] - ys.mean_) / ys.scale_).astype('float32') for sp in s}
    sar = lambda sp, yhat: to_sar(s[sp], yhat.reshape(-1, H) * ys.scale_ + ys.mean_)

    base = {k: train(a, u, X['train'], Y['train'], X['val'], Y['val'], seed) for k, (a, u) in CHOSEN.items()}
    raw = {sp: {k: m.predict(X[sp], verbose=0) for k, m in base.items()} for sp in ('val', 'test')}

    X_meta, y_meta = oof_predictions(X['train'], Y['train'], seed)
    meta_in = {sp: np.stack([raw[sp][k] for k in CHOSEN], axis=-1) for sp in ('val', 'test')}
    for combo, cols in COMBOS.items():
        for kind in META:
            name = f'STK_{kind}({combo})'
            keras.utils.set_random_seed(seed)
            meta = fit(build_meta(kind, len(cols)), X_meta[:, :, cols], y_meta,
                       meta_in['val'][:, :, cols], Y['val'], EPOCHS, name)
            for sp in ('val', 'test'):
                raw[sp][name] = meta.predict(meta_in[sp][:, :, cols], verbose=0)

    flat = lambda a: a.reshape(len(a), -1)
    xgb = XGBRegressor(n_estimators=2000, learning_rate=0.05, max_depth=4, tree_method='hist',
                       early_stopping_rounds=50, random_state=seed)
    xgb.fit(flat(X['train']), Y['train'], eval_set=[(flat(X['val']), Y['val'])], verbose=False)
    gru256 = train('GRU', 256, X['train'], Y['train'], X['val'], Y['val'], seed)
    for sp in ('val', 'test'):
        raw[sp]['XGBoost'] = xgb.predict(flat(X[sp]))
        raw[sp]['GRU-256'] = gru256.predict(X[sp], verbose=0)

    pv = add_averages({k: sar('val', p) for k, p in raw['val'].items()})
    pt = add_averages({k: sar('test', p) for k, p in raw['test'].items()})
    b = baselines(s, parts, arima=True)
    pv |= b['val']
    pt |= b['test']
    lams = shrink(pv, pt, s)
    return pv, pt, s, lams


def run_per_ticker(data, seed=42):
    out = f'{RESULTS}/per_ticker'
    os.makedirs(out, exist_ok=True)
    for label, (df, parts) in data.items():
        if os.path.exists(f'{out}/{label}_winloss.csv'):
            continue                                     # already done: resume after a crash
        print('=' * 20, label)
        save(out, label, *run_ticker(df, parts, seed))
    summary(out, list(data), 'all')


# ---------------------------------------------------------------- 3. one global model

def pooled(data):
    """Windows of every ticker, scalers fitted on the pooled train rows, pooled scaled train/val sets."""
    s = {l: {sp: windows(parts[sp], df) for sp in SPLITS} for l, (df, parts) in data.items()}
    xs = StandardScaler().fit(np.concatenate([parts['train'].to_numpy() for _, parts in data.values()]))
    ys = StandardScaler().fit(np.concatenate([s[l]['train']['R'] for l in s]).reshape(-1, 1))
    sx = lambda a: ((a - xs.mean_) / xs.scale_).astype('float32')
    sy = lambda a: ((a - ys.mean_) / ys.scale_).astype('float32')
    pool = lambda sp: (np.concatenate([sx(s[l][sp]['X']) for l in s]), np.concatenate([sy(s[l][sp]['R']) for l in s]))
    (Xtr, Ytr), (Xva, Yva) = pool('train'), pool('val')
    print(f'{len(s)} tickers: {len(Xtr):,} train and {len(Xva):,} val windows')
    return s, sx, ys, (Xtr, Ytr, Xva, Yva)


def global_model(arch, units, sets, seed, batch):
    """Train one global model, or load it if it was trained before (by the sweep or a global run)."""
    path = f'{RESULTS}/global/models/{arch}_{units}.keras'
    if os.path.exists(path):
        return keras.models.load_model(path)
    model = train(arch, units, *sets, seed, batch)
    model.save(path)
    return model


def run_sweep_global(data, seed=42, batch=2048):
    """Unit sweep for the global models (SWEEP), chosen by pooled val loss. Writes unit_sweep.csv
    and units.json, which run_global uses."""
    os.makedirs(f'{RESULTS}/global/models', exist_ok=True)
    keras.backend.clear_session()
    _, _, _, sets = pooled(data)
    rows = []
    for arch, sizes in SWEEP.items():
        for units in sizes:
            model = global_model(arch, units, sets, seed, batch)
            rows.append({'arch': arch, 'units': units, 'val_loss': model.evaluate(sets[2], sets[3], verbose=0)})
            print(rows[-1])
    sweep = pd.DataFrame(rows)
    sweep.to_csv(f'{RESULTS}/global/unit_sweep.csv', index=False)
    best = sweep.loc[sweep.groupby('arch')['val_loss'].idxmin()].set_index('arch')['units']
    units = {k: int(best[a]) for k, (a, _) in CHOSEN.items()}
    pd.Series(units).to_json(f'{RESULTS}/global/units.json')
    print('chosen units:', units)


def run_global(data, seed=42, batch=2048):
    """All base models trained once on the pooled training windows of all tickers, early-stopped on the
    pooled val windows, then evaluated per ticker. Unit sizes from units.json (run sweep_global
    first) or, without it, the per-ticker sizes in CHOSEN."""
    out = f'{RESULTS}/global'
    os.makedirs(f'{out}/models', exist_ok=True)
    keras.backend.clear_session()
    units = {k: u for k, (_, u) in CHOSEN.items()}
    if os.path.exists(f'{out}/units.json'):
        units = pd.read_json(f'{out}/units.json', typ='series').to_dict()
    print('units:', units)
    s, sx, ys, sets = pooled(data)
    base = {k: global_model(a, units[k], sets, seed, batch) for k, (a, _) in CHOSEN.items()}
    del sets

    for label, (df, parts) in data.items():
        print('=' * 20, label)
        sl = s[label]
        p = {sp: add_averages({k: to_sar(sl[sp], m.predict(sx(sl[sp]['X']), verbose=0) * ys.scale_ + ys.mean_)
                               for k, m in base.items()}) for sp in ('val', 'test')}
        b = baselines(sl, parts, arima=label in SYMBOLS)
        pv, pt = p['val'] | b['val'], p['test'] | b['test']
        lams = shrink(pv, pt, sl)
        save(out, label, pv, pt, sl, lams)

    summary(out, [l for l in data if l in SYMBOLS], 'paper')
    summary(out, list(data), 'all')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['data', 'per_ticker', 'sweep_global', 'global'], default='data')
    ap.add_argument('--tickers', choices=['paper', 'all'], default=None,
                    help='default: paper for data and per_ticker, all for sweep_global and global')
    ap.add_argument('--refresh', action='store_true', help='download every Yahoo history again')
    ap.add_argument('--batch', type=int, default=2048, help='batch size of the global models')
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    which = a.tickers or ('paper' if a.mode in ('data', 'per_ticker') else 'all')
    data = load_universe(universe(which), a.refresh)
    if a.mode == 'per_ticker':
        run_per_ticker(data, a.seed)
    elif a.mode == 'sweep_global':
        run_sweep_global(data, a.seed, a.batch)
    elif a.mode == 'global':
        run_global(data, a.seed, a.batch)


if __name__ == '__main__':
    main()
