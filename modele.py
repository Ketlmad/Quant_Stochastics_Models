"""Schwartz-Smith (2000) - estimation multi-produits Barchart.

Trois modules scientifiques SS et un module SSX: donnees.py / kalman.py / modele.py / model_SSX.py.
Sorties : 3 Excel données + 2 Excel modèles + 2 PDF consolidés.
Les prix du Parquet Barchart ne sont jamais modifiés.
"""
from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2
from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday,
                                     USMartinLutherKingJr, USMemorialDay,
                                     USLaborDay, USPresidentsDay,
                                     USThanksgivingDay, nearest_workday)
from donnees import BARCHART, PRODUCTS, ROOT, DATA_OUTPUT, generate, global_paths
from kalman import kalman, matrices

LOG = logging.getLogger('schwartz-smith')
DT = 1.0 / 252.0
PARAM_NAMES = ('k', 'sigmax', 'lambdax', 'mu', 'sigmae', 'lambdae', 'pxe')
INITIAL = np.array([.70, .25, -.17, -.0115, .19, -.0165, .10])
MODEL_DIRNAME = 'sorties_models'
RESULT_FILE = 'ALL_Model_Results.xlsx'
REPORT_FILE = 'ALL_Technical_Dashboard.pdf'
TENOR_NAMES = ('30.days', '60.days', '90.days', '120.days', '150.days', '180.days')
LOWER7 = np.array([1e-5, 1e-5, -10., -10., 1e-5, -10., -1.])
UPPER7 = np.array([10., 10., 10., 10., 10., 10., 1.])


class NYSECalendar(AbstractHolidayCalendar):
    rules = [Holiday('New Year', month=1, day=1, observance=nearest_workday),
             USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
             Holiday('Juneteenth', month=6, day=19, observance=nearest_workday,
                     start_date='2022-01-01'),
             Holiday('Independence', month=7, day=4, observance=nearest_workday),
             USLaborDay, USThanksgivingDay,
             Holiday('Christmas', month=12, day=25, observance=nearest_workday)]


def future_dates(reference_date: str | pd.Timestamp, steps: int = 40) -> pd.DatetimeIndex:
    """Premier jour strictement après la dernière observation (pas la date d'exécution)."""
    d = pd.Timestamp(reference_date).normalize()
    holidays = NYSECalendar().holidays(start=d, end=d + pd.Timedelta(days=120))
    return pd.bdate_range(d + pd.Timedelta(days=1), periods=steps + 25).difference(holidays)[:steps]


def load_input(product: str, source: Path, stamp: str):
    legacy = PRODUCTS[product][1]
    filename = global_paths(source, stamp)['model']
    prices = pd.read_excel(filename, sheet_name=legacy)
    ttm = pd.read_excel(filename, sheet_name=f'{legacy}_TTM')
    dates = pd.DatetimeIndex(pd.to_datetime(prices.iloc[:, 0]))
    dates_ttm = pd.DatetimeIndex(pd.to_datetime(ttm.iloc[:, 0]))
    y = prices.iloc[:, 1:].to_numpy(dtype=float)
    tau = ttm.iloc[:, 1:].to_numpy(dtype=float)
    if not dates.equals(dates_ttm) or y.shape[1] != 6 or tau.shape != y.shape:
        raise ValueError(f'{product}: matrices 6 maturites ou dates desynchronisees')
    if not np.isfinite(y).all() or np.any(y <= 0):
        raise ValueError(f'{product}: prix non strictement positifs, log impossible')
    if not np.isfinite(tau).all() or not np.allclose(tau, tau[0], atol=1e-12, rtol=0):
        raise ValueError(f'{product}: maturites constantes invalides')
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError(f'{product}: dates du modele non triees ou non uniques')
    return dates, y, tau[0]


def previous_state(path: Path, param_names=PARAM_NAMES) -> dict[str, dict]:
    """Ne réutilise QUE les derniers points convergés, stockés dans l'Excel ALL."""
    if not path.exists():
        return {}
    try:
        tab = pd.read_excel(path, sheet_name='WARM_START')
        state = {}
        for _, row in tab.iterrows():
            vals = np.array([row.get(k, np.nan) for k in param_names] +
                            [row.get(f'err_{j+1}', np.nan) for j in range(6)], float)
            if bool(row.get('converged')) and np.isfinite(vals).all():
                state[str(row['product'])] = {'psi': vals, 'asof': str(row.get('asof', '')),
                                               'first_date': str(row.get('first_date', '')),
                                               'x0_short': row.get('x0_short', np.nan),
                                               'x0_long': row.get('x0_long', np.nan)}
        return state
    except (ValueError, KeyError, OSError) as exc:
        LOG.warning('WARM_START non lu : %s', exc)
        return {}



def legacy_log_seeds(log_dir: Path) -> dict[str, dict]:
    """Import initial des 7 parametres depuis les logs V1.2 CONVERGES.

    Les 6 erreurs de mesure n'etaient pas exportees par V1.2 : on les
    initialise separement, sans pretendre recuperer les 13 parametres.
    """
    seeds = {}
    for logfile in sorted(log_dir.glob('stochastic_*.log'), reverse=True):
        try:
            content = logfile.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        accepted = set(re.findall(r'DONE\s*\|\s*([A-Za-z_]+)\s*\|[^\n]*converged=True', content))
        for product, body in re.findall(r'\|\s*([A-Za-z_]+)\s*\|\s*theta\s*\[([^]\n]+)\]', content):
            if product not in PRODUCTS or product not in accepted or product in seeds:
                continue
            found = dict((name, float(value)) for name, value in
                         re.findall(r'([A-Za-z_]+)=(-?[0-9.eE+\-]+)', body))
            if all(x in found for x in PARAM_NAMES):
                seeds[product] = {'psi': np.array([found[k] for k in PARAM_NAMES]),
                                  'origin': 'legacy_log_7p'}
    return seeds


def _data_initialization(prices: np.ndarray, maturities: np.ndarray,
                         warm: dict | None = None):
    """Warm-start structurel historique ou heuristique fondée sur les données.

    x0 est TOUJOURS reconstruit sur la 1ere courbe. Le dernier état filtré de
    la veille n'est pas un x0 valide pour refiltrer l'intégralité de l'historique.
    """
    n = prices.shape[1]
    y = np.log(prices)
    ret = np.diff(y, axis=0)
    if len(ret) > 20:
        vol_long = np.std(ret[:, -1], ddof=1) * np.sqrt(252)
        vol_spread = np.std(ret[:, 0] - ret[:, -1], ddof=1) * np.sqrt(252)
    else:
        vol_long, vol_spread = .19, .15
    if not np.isfinite(vol_long):
        vol_long = .19
    if not np.isfinite(vol_spread):
        vol_spread = .15
    params = np.r_[INITIAL, np.full(n, .0046)]
    params[4] = np.clip(vol_long, .05, 1.25)
    params[1] = np.clip(vol_spread / max(.15, 1-np.exp(-.70*(maturities[-1]-maturities[0]))), .06, 1.25)
    if len(y) >= 30:
        span = min(len(y) - 1, 252)
        params[3] = float(np.clip((y[-1, -1]-y[-span-1, -1])*252/span, -.3, .3))
    if warm is not None:
        prev = np.asarray(warm['psi'], float)
        if len(prev) == 7+n and np.isfinite(prev).all():
            params = prev.copy()
            origin = 'last_converged'
        elif len(prev) == 7 and np.isfinite(prev).all():
            params[:7] = prev
            origin = 'legacy_log_7p'
        else:
            origin = 'data_driven'
    else:
        origin = 'data_driven'
    lower = np.r_[LOWER7, np.full(n, 1e-5)]
    upper = np.r_[UPPER7, np.full(n, .5)]
    params = np.maximum(lower + 1e-8, np.minimum(upper-1e-8, params))
    k = params[0]
    _, _, Q, d, H = matrices(params, maturities, DT)
    # Identification ridge sur la courbe initiale, limite les états explosifs
    x_guess = float(np.clip((y[0, 0] - y[0, -1] - (d[0] - d[-1])) /
                            max(H[0, 0] - H[-1, 0], 1e-5), -0.7, 0.7))
    eq_guess = float(np.mean(y[0] - d - H[:, 0] * x_guess))
    x0 = np.array([x_guess, eq_guess])
    # P0 reste fixé pendant l'optimisation, comme dans MainSS.m
    P0 = Q.copy()
    return params, lower, upper, x0, P0, origin


def estimate(prices, tau, product, warm=None, solver='SLSQP', maxiter=1000):
    y = np.log(prices)
    psi0, lb, ub, x0, P0, origin = _data_initialization(prices, tau, warm)
    scales = np.r_[1., .25, .25, .05, .25, .05, .25, np.repeat(.005, len(tau))]
    n = len(y)

    def objective(z):
        nll = kalman(y, z*scales, tau, x0, P0, DT)
        return nll/n if np.isfinite(nll) else 1e30

    initial_score = kalman(y, psi0, tau, x0, P0, DT)
    options = {'maxiter': maxiter, 'disp': False}
    if solver == 'SLSQP':
        options['ftol'] = 1e-9
    elif solver == 'L-BFGS-B':
        options.update({'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-7, 'maxls': 40})
    elif solver == 'trust-constr':
        options.update({'gtol': 1e-7, 'xtol': 1e-9})
    else:
        raise ValueError(f'Unknown solver {solver}')
    result = minimize(objective, psi0/scales, method=solver,
                      bounds=list(zip(lb/scales, ub/scales)), options=options)
    candidate = result.x * scales
    candidate_nll = kalman(y, candidate, tau, x0, P0, DT)
    # Never accept a statistically worse result than the tested start vector.
    if np.isfinite(candidate_nll) and candidate_nll <= initial_score + 1e-7:
        psi, nll = candidate, candidate_nll
        converged = bool(result.success)
    else:
        psi, nll, converged = psi0, initial_score, False
        LOG.warning('%s: solution rejetee, retour au point de depart', product)
    nll, trace = kalman(y, psi, tau, x0, P0, DT, details=True)
    if trace is None:
        raise RuntimeError(f'{product}: trace de Kalman indisponible')
    LOG.info('%s | NLL %.4f -> %.4f | %d iterations | convergence=%s | warm=%s',
             product, initial_score, nll, int(result.nit), converged, origin)
    return psi, trace, {'nll': nll, 'nll_initial': initial_score,
                        'converged': converged, 'iterations': int(result.nit),
                        'optimizer': solver, 'origin': origin, 'message': str(result.message),
                        'x0': x0, 'P0': P0}


def _lb_pvalue(series: np.ndarray, lags: int = 10) -> float:
    series = np.asarray(series, float)
    series = series[np.isfinite(series)]
    count = len(series)
    lagmax = min(lags, count//5)
    if lagmax < 1 or np.std(series) <= 1e-15:
        return float('nan')
    e = series-series.mean()
    denom = e @ e
    stat = count*(count+2)*sum((float(e[l:] @ e[:-l])/denom)**2/(count-l)
                               for l in range(1, lagmax+1))
    return float(chi2.sf(stat, lagmax))


def _forecasts(psi: np.ndarray, tau: np.ndarray, last_state: np.ndarray,
               end_date: str, days: int = 40, seed: int = 2000, product='Wheat',
               paths: int = 300):
    A, c, Q, d, H = matrices(psi, tau, DT)
    dates = future_dates(end_date, days)
    code = sum((i+1)*ord(ch) for i, ch in enumerate(product))
    rng = np.random.default_rng(np.random.SeedSequence([seed, code]))
    sims = np.broadcast_to(np.asarray(last_state, float), (paths, 2)).copy()
    quant = np.empty((len(dates), len(tau), 5))
    for t in range(len(dates)):
        sims = sims @ A.T + c + rng.multivariate_normal(np.zeros(2), Q, size=paths)
        fwd_prices = np.exp(sims @ H.T + d)
        quant[t] = np.quantile(fwd_prices, [.05, .25, .50, .75, .95], axis=0).T
    return dates, quant


def product_results(product, dates, prices, tau, psi, trace, info, seed=2000,
                    forecast_days=40):
    code = PRODUCTS[product][1]
    n, N = prices.shape
    y = np.log(prices)
    _, _, _, d, H = matrices(psi, tau, DT)
    fitted = np.exp(trace['states'] @ H.T + d)
    residuals = y - np.log(fitted)
    innovations = trace['innovations']
    errors_price = (fitted - prices)/prices
    rmse_by = np.sqrt(np.mean(residuals**2, axis=0))
    rmse_pct = float(np.sqrt(np.mean(errors_price**2))*100)
    corrs = np.corrcoef(innovations.T)
    mean_ljung = [_lb_pvalue(innovations[:, i]) for i in range(N)]
    k, sx, lx, mu, se, le, rho = psi[:7]
    premium = le + np.expm1(k*tau)/(k*tau) * (-lx)  # Convention historique MainSS
    iv = np.sqrt(np.maximum(0., np.exp(-2*k*tau)*sx**2 + se**2 +
                             2*np.exp(-k*tau)*rho*sx*se))
    close_date = str(dates[-1].date())
    future_dates_, q = _forecasts(psi, tau, trace['states'][-1], close_date,
                                  forecast_days, seed, product)
    par_row = {'product': product, 'code': code, 'asof': close_date,
               'first_date': str(dates[0].date()), 'nobs': n,
               'nll': info['nll'], 'nll_initial': info['nll_initial'],
               'nll_per_obs': info['nll']/n,
               'converged': info['converged'], 'iterations': info['iterations'],
               'optimizer': info['optimizer'], 'warm_source': info['origin'],
               'message': info['message'], 'x0_short': info['x0'][0],
               'x0_long': info['x0'][1], 'last_short': trace['states'][-1, 0],
               'last_long': trace['states'][-1, 1],
               'rmse_price_pct': rmse_pct, 'lb_pvalue_min': float(np.nanmin(mean_ljung)),
               'residual_max_corr': float(np.nanmax(np.abs(corrs-np.eye(N))))}
    for name, val in zip(PARAM_NAMES, psi[:7]):
        par_row[name] = val
    for j, val in enumerate(psi[7:]):
        par_row[f'err_{j+1}'] = val

    dt_index = pd.DatetimeIndex(dates)
    def wide(values, columns):
        x = pd.DataFrame(values, columns=columns)
        x.insert(0, 'date', dt_index)
        x.insert(0, 'product', product)
        return x

    latent = wide(trace['states'], ['short_term', 'long_term'])
    latent['estimated_spot'] = np.exp(trace['states'].sum(axis=1))
    fit = wide(np.c_[prices, fitted],
               [f'observed_{int(round(v*252))}d' for v in tau] +
               [f'fitted_{int(round(v*252))}d' for v in tau])
    resid = wide(residuals, [f'resid_log_{int(round(v*252))}d' for v in tau])
    inn = wide(innovations, [f'innovation_log_{int(round(v*252))}d' for v in tau])
    curves = []
    for j, mat in enumerate(tau):
        days = int(round(mat*252))
        curves.append({'product': product, 'date': close_date, 'maturity_days': days,
                       'observed': prices[-1, j], 'fitted': fitted[-1, j],
                       'residual_pct': errors_price[-1, j]*100,
                       'rmse_log': rmse_by[j], 'risk_premium_per_year': premium[j],
                       'instant_vol_annual': iv[j]})
    forecast = []
    for day_i, fdate in enumerate(future_dates_):
        for j, mat in enumerate(tau):
            days = int(round(mat*252))
            p5, p25, p50, p75, p95 = q[day_i, j]
            forecast.append({'product': product, 'forecast_date': fdate,
                             'maturity_days': days, 'horizon_sessions': day_i+1,
                             'p05': p5, 'p25': p25, 'p50': p50,
                             'p75': p75, 'p95': p95,
                             'signal_pct': (p50/prices[-1, j]-1)*100})
    return {'row': par_row, 'curves': pd.DataFrame(curves), 'forecast': pd.DataFrame(forecast),
            'latent': latent, 'fit': fit, 'residual': resid, 'innovation': inn,
            'pdf': {'product': product, 'date': close_date, 'dates': dt_index,
                    'prices': prices, 'fitted': fitted, 'states': trace['states'],
                    'residuals': residuals, 'innovations': innovations,
                    'tau': tau, 'curves': curves, 'forecasts': pd.DataFrame(forecast),
                    'params': par_row, 'corrs': corrs, 'pvals': mean_ljung}}


def _save_book(path: Path, results: list, previous: dict, status: list,
               result_file=RESULT_FILE, param_names=PARAM_NAMES, extra_tables=None):
    if not results:
        return None
    result = path / result_file
    tmp = path / ('_' + result_file)
    def together(name):
        return pd.concat([row[name] for row in results], ignore_index=True)
    params = pd.DataFrame([r['row'] for r in results])
    prev = {}
    for name, entry in previous.items():
        prev[name] = entry
    for r in results:
        if r['row']['converged']:
            row = r['row']
            prev[row['product']] = {'psi': np.array([row[k] for k in param_names] +
                                                     [row[f'err_{j+1}'] for j in range(6)]),
                                    'asof': row['asof'], 'first_date': row['first_date'],
                                    'x0_short': row['x0_short'], 'x0_long': row['x0_long']}
    warm_rows = []
    for product, v in prev.items():
        row = {'product': product, 'asof': v['asof'], 'first_date': v.get('first_date', ''),
               'converged': True, 'x0_short': v.get('x0_short', np.nan),
               'x0_long': v.get('x0_long', np.nan)}
        row.update(dict(zip(param_names, v['psi'][:len(param_names)])))
        row.update({f'err_{j+1}': val for j, val in enumerate(v['psi'][len(param_names):])})
        warm_rows.append(row)
    data = {'STATUS': pd.DataFrame(status), 'PARAMETERS': params,
            'WARM_START': pd.DataFrame(warm_rows, columns=['product', 'asof', 'first_date', 'converged', 'x0_short', 'x0_long'] + list(param_names) + [f'err_{j+1}' for j in range(6)]),
            'CURVE_LATEST': together('curves'), 'FORECAST_MC': together('forecast'),
            'LATENTS': together('latent'), 'FITTED_PRICES': together('fit'),
            'RESIDUALS_LOG': together('residual'), 'INNOVATIONS_LOG': together('innovation')}
    if extra_tables:
        data.update(extra_tables)
    with pd.ExcelWriter(tmp, engine='xlsxwriter', datetime_format='yyyy-mm-dd') as w:
        fmt_head = w.book.add_format({'bold': True, 'font_color': '#FFFFFF',
                                      'bg_color': '#12243B', 'font_size': 10,
                                      'bottom': 2, 'bottom_color': '#14639C'})
        fmt_dates = w.book.add_format({'num_format': 'yyyy-mm-dd'})
        for name, tab in data.items():
            tab.to_excel(w, sheet_name=name, index=False)
            ws = w.sheets[name]
            ws.set_row(0, 24)
            for j, colname in enumerate(tab.columns):
                ws.write_string(0, j, str(colname), fmt_head)
            ws.freeze_panes(1, min(2, tab.shape[1]))
            ws.autofilter(0, 0, len(tab), tab.shape[1]-1)
            ws.set_column(0, 1, 17)
            ws.set_column(2, min(tab.shape[1]-1, 35), 16)
            for j, column in enumerate(tab.columns):
                if str(column) in ('date', 'forecast_date') and pd.api.types.is_datetime64_any_dtype(tab[column]):
                    ws.set_column(j, j, 17, fmt_dates)
    tmp.replace(result)
    LOG.info('Excel global modele: %s | %s', result, ', '.join(data))
    return result


def _make_pdf(path: Path, results: list, status: list, stamp: str, source_file: str):
    if not results:
        return None
    # Import tardif pour laisser --data-only indépendant du PDF.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import MaxNLocator
    from scipy.stats import probplot

    NAVY, SKY, YELLOW, RED = '#12243B', '#14639C', '#DA9E3C', '#A74846'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': NAVY, 'text.color': NAVY,
                         'xtick.color': '#5F6C78', 'ytick.color': '#5F6C78',
                         'axes.edgecolor': '#C9D3DC', 'axes.grid': True,
                         'grid.alpha': .18, 'axes.titleweight': 'bold',
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    pdf = path / REPORT_FILE
    temp = path / ('_' + REPORT_FILE)
    with PdfPages(temp, metadata={'Title': 'Schwartz-Smith | Multi-Commodity Quant Diagnostics',
                                 'Author': 'Stochastic Models | Quant Research',
                                 'Subject': 'Kalman / Maximum Likelihood diagnostics'}) as writer:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(.06, .90, 'STOCHASTIC MODELS  /  QUANT RESEARCH', fontsize=12, color=SKY, weight='bold')
        fig.text(.06, .83, 'Schwartz-Smith | Multi-Commodity', fontsize=24, weight='bold')
        fig.text(.06, .775, 'Calibration and risk diagnostics', fontsize=15, color='#53677B')
        npass = sum(s['status']=='CONVERGED' for s in status)
        fig.text(.06, .69, f'{len(status)} produits selectionnes    |    {len(results)} estimations terminees    |    {npass} convergences', fontsize=11)
        ax = fig.add_axes([.06, .17, .89, .44]); ax.axis('off')
        cols = ['Produit', 'Etat', 'Seances', 'NLL / obs.', 'RMSE %', 'Init.']
        rows = []
        scores = {r['row']['product']: r['row'] for r in results}
        for s in status:
            r = scores.get(s['product'])
            rows.append([s['product'], s['status'], str(r['nobs']) if r else '—',
                         f"{r['nll_per_obs']:.2f}" if r else '—',
                         f"{r['rmse_price_pct']:.2f}" if r else '—',
                         r['warm_source'] if r else '—'])
        table = ax.table(cellText=rows, colLabels=cols, cellLoc='left', colLoc='left',
                         loc='upper left', colWidths=[.20,.19,.10,.15,.12,.17])
        table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.35)
        for (i,j), cell in table.get_celld().items():
            cell.set_edgecolor('#E3E7EB')
            if i==0:
                cell.set_facecolor(NAVY); cell.get_text().set_color('white'); cell.get_text().set_weight('bold')
            elif i%2==0:
                cell.set_facecolor('#F0F4F8')
        fig.text(.06,.12,f'Extraction: {source_file}  |  Edition: {stamp}',size=8,color='#607285')
        fig.text(.06,.08,'NLL = negative log-likelihood; RMSE = erreur relative ajuste / observe. Non-convergence distincte d une erreur.',size=8)
        writer.savefig(fig, bbox_inches='tight'); plt.close(fig)

        # Vue transversale : vitesse de retour a la moyenne / volatilites
        # et correlation des chocs latents long terme aux dates communes.
        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
        fig.subplots_adjust(left=.09, right=.95, top=.82, bottom=.19, wspace=.31)
        fig.suptitle('Cross-Commodity | Factor Risk Landscape', fontsize=20,
                     fontweight='bold', y=.93, color=NAVY)
        axes[0].set_title('Short-term mean-reversion half-life vs volatility')
        for row in results:
            pp = row['row']
            half_life = np.log(2) / pp['k']
            axes[0].scatter(half_life, 100*pp['sigmax'], s=40,
                            color=SKY if pp['converged'] else YELLOW, alpha=.85)
            axes[0].annotate(PRODUCTS[pp['product']][1],
                             (half_life, 100*pp['sigmax']), xytext=(3,3),
                             textcoords='offset points', fontsize=7)
        axes[0].set_xlabel('Half-life in years (ln 2 / k)')
        axes[0].set_ylabel('Short-term sigma, annualised (%)')
        axes[0].set_xlim(left=0)
        series = []
        for row in results:
            d = row['pdf']
            latent_ret = pd.Series(np.diff(d['states'][:,1]), index=d['dates'][1:],
                                   name=PRODUCTS[d['product']][1])
            series.append(latent_ret)
        coview = pd.concat(series, axis=1).corr(min_periods=25)
        corr = coview.to_numpy(dtype=float)
        cmap = plt.get_cmap('RdBu_r').copy(); cmap.set_bad('#E0E6EC')
        axes[1].imshow(np.ma.masked_invalid(corr), cmap=cmap, vmin=-1, vmax=1)
        axes[1].grid(False)
        axes[1].set_title('Correlation of daily long-term factor innovations')
        labels = coview.columns.tolist()
        axes[1].set_xticks(np.arange(len(labels)), labels, rotation=65, ha='right', fontsize=7)
        axes[1].set_yticks(np.arange(len(labels)), labels, fontsize=7)
        fig.text(.09, .10, 'Factors are estimated separately per asset; correlations are descriptive, not a common-factor identification.',
                 fontsize=9, color='#54697D')
        fig.text(.09, .075, 'Half-life = ln(2)/k, not the historical MATLAB expression that multiplies by k.',
                 fontsize=8, color='#54697D')
        writer.savefig(fig); plt.close(fig)

        for row in results:
            d = row['pdf']; p = d['params']
            fig, axes = plt.subplots(3, 2, figsize=(11.69, 8.27))
            fig.subplots_adjust(left=.07, right=.96, top=.82, bottom=.10, wspace=.25, hspace=.46)
            fig.suptitle(f"{d['product']}   |   {PRODUCTS[d['product']][1]}   |   {d['date']}",
                         fontsize=18, fontweight='bold', y=.96, color=NAVY)
            fig.text(.07,.89,
                     f"{'CONVERGED' if p['converged'] else 'NOT CONVERGED'}  ·  {p['nobs']:,} seances  ·  "
                     f"NLL/obs. {p['nll_per_obs']:.2f}  ·  RMSE prix {p['rmse_price_pct']:.2f}%  ·  "
                     f"Init. {p['warm_source']}", size=9, color=SKY)
            # price fit: first/last maturity - show last 250 days for legibility
            ax = axes[0,0]; sl = slice(max(0,len(d['dates'])-252),None)
            for j, color in [(0, SKY),(-1,YELLOW)]:
                days = int(round(d['tau'][j]*252))
                ax.plot(d['dates'][sl], d['prices'][sl,j], color=color, lw=1.15, label=f'{days}d observed')
                ax.plot(d['dates'][sl], d['fitted'][sl,j], color=color, ls='--', lw=1, alpha=.72, label=f'{days}d fit')
            ax.set_title('Market vs filtered fit - last 252 sessions')
            ax.legend(ncol=2, fontsize=7, frameon=False)
            ax.tick_params(axis='x', rotation=25)

            ax = axes[0,1]
            ax.plot(d['dates'][sl], d['states'][sl,0], color=SKY, label='Short-term (log)')
            ax.plot(d['dates'][sl], d['states'][sl,1], color=YELLOW, label='Long-term (log)')
            ax.set_title('Latent state paths - last 252 sessions')
            ax.legend(frameon=False, fontsize=7)
            ax.tick_params(axis='x', rotation=25)

            ax = axes[1,0]
            rmse = [100*np.sqrt(np.mean(d['residuals'][:,i]**2)) for i in range(6)]
            ax.bar([int(round(x*252)) for x in d['tau']], rmse, width=20, color=SKY, alpha=.83)
            ax.set_title('In-sample log-error RMSE by tenor')
            ax.set_xlabel('Maturity - days'); ax.set_ylabel('RMSE (log points x 100)')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))

            ax = axes[1,1]
            im = ax.imshow(d['corrs'], cmap='RdBu_r', vmin=-1, vmax=1)
            ax.grid(False)
            ax.set_xticks(range(6), [str(int(round(x*252)))+'d' for x in d['tau']])
            ax.set_yticks(range(6), [str(int(round(x*252)))+'d' for x in d['tau']])
            ax.set_title('Innovation correlation matrix')
            for (i,j), value in np.ndenumerate(d['corrs']):
                ax.text(j,i,f'{value:.2f}',ha='center',va='center',fontsize=6,
                        color='white' if abs(value)>.6 else NAVY)

            ax = axes[2,0]
            tenor = [x['maturity_days'] for x in d['curves']]
            premium = [100*x['risk_premium_per_year'] for x in d['curves']]
            vol = [100*x['instant_vol_annual'] for x in d['curves']]
            ax.plot(tenor, premium, marker='o', color=SKY, label='Risk premium (MATLAB)')
            ax2 = ax.twinx(); ax2.plot(tenor, vol, marker='s', color=YELLOW, label='Inst. vol')
            ax.set_title('Model-implied risk premium and annualised volatility')
            ax.set_xlabel('Days'); ax.set_ylabel('Premium % p.a.', color=SKY)
            ax2.set_ylabel('Volatility % p.a.', color=YELLOW)
            ax.grid(alpha=.18)

            ax = axes[2,1]
            f = d['forecasts']
            for tenor_days, color in [(int(round(d['tau'][0]*252)), SKY), (int(round(d['tau'][-1]*252)),YELLOW)]:
                group = f.loc[f.maturity_days.eq(tenor_days)].iloc[:20]
                xx = np.arange(1, len(group)+1)
                ax.plot(xx,group.p50.to_numpy(),color=color,label=f'{tenor_days}d median')
                ax.fill_between(xx,group.p05.to_numpy(),group.p95.to_numpy(),color=color,alpha=.12)
                ax.axhline(d['prices'][-1,0 if color==SKY else -1], color=color,linestyle=':',alpha=.6)
            ax.set_title('20-session Monte Carlo fan (5-95% bands)')
            ax.set_xlabel('Sessions ahead'); ax.legend(frameon=False,fontsize=7)
            fig.text(.07,.047,
                     f"k={p['k']:.4f}  sx={p['sigmax']:.4f}  mu={p['mu']:.4f}  "
                     f"se={p['sigmae']:.4f}  rho={p['pxe']:.4f}  |  "
                     f"Ljung-Box min p={p['lb_pvalue_min']:.3g}  |  "
                     'Scenario fan: conditional on latest filtered state; excludes measurement noise.',
                     fontsize=7, color='#52677A')
            writer.savefig(fig); plt.close(fig)
    temp.replace(pdf)
    LOG.info('Dashboard PDF: %s (%d pages)', pdf, len(results)+2)
    return pdf


def main():
    parser = argparse.ArgumentParser(description='Barchart -> Schwartz-Smith SS & SSX: multi-assets')
    parser.add_argument('--all', action='store_true', help='Tous les produits (defaut)')
    parser.add_argument('--model', choices=['SS','SSX','both'], default='both',
                        help='SS historique, SSX deux OU, ou les deux (defaut)')
    parser.add_argument('--ssx-starts', type=int, default=2,
                        help='Points de depart pour le SSX (defaut 2)')
    parser.add_argument('--product', choices=sorted(PRODUCTS), help='Un seul produit (sans toucher aux autres warm starts)')
    parser.add_argument('--data-only', action='store_true', help='Trois bases ALL sans estimation')
    parser.add_argument('--no-prepare', action='store_true', help='Reutiliser les trois bases du jour')
    parser.add_argument('--source', type=Path, default=BARCHART)
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--date', default=date.today().isoformat())
    parser.add_argument('--optimizer', choices=['SLSQP', 'L-BFGS-B', 'trust-constr'], default='SLSQP')
    parser.add_argument('--maxiter', type=int, default=1000)
    parser.add_argument('--last', type=int, default=0, help='Nombre de seances pour estimation; 0 = toutes')
    parser.add_argument('--seed', type=int, default=2000)
    parser.add_argument('--forecast-days', type=int, default=40)
    args = parser.parse_args()
    if args.forecast_days < 1 or args.last < 0:
        parser.error('--forecast-days doit etre >0, --last >=0')
    output = Path(args.output) / 'output'
    data_dir = output / 'data_processed'
    model_dir = output / MODEL_DIRNAME
    logs = Path(args.output) / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs / f'stochastic_{datetime.now():%Y%m%d_%H%M%S}.log'
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(levelname)-7s | %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(log_file, encoding='utf-8')], force=True)
    products = [args.product] if args.product else list(PRODUCTS)
    LOG.info('START | %s produits | source=%s | data=%s | models=%s',
             len(products), args.source, data_dir, model_dir)
    if not args.no_prepare:
        try:
            generate(source=args.source, output=data_dir, run_date=args.date)
        except Exception:
            LOG.exception('FAIL | preparation des donnees')
            raise SystemExit(1)
    if args.data_only:
        LOG.info('SUCCESS | Trois bases ALL dans %s', data_dir)
        return
    if missing := [x for x in global_paths(data_dir, args.date).values() if not x.exists()]:
        LOG.error('Fichiers absents: %s', missing)
        raise SystemExit(1)
    model_dir.mkdir(parents=True, exist_ok=True)
    exits = []
    if args.model in ('SS', 'both'):
        previous = previous_state(model_dir / RESULT_FILE)
        legacy = legacy_log_seeds(logs)
        results, statuses = [], []
        for product in products:
            LOG.info('SS | START | %s', product)
            try:
                dates, prices, tau = load_input(product, data_dir, args.date)
                if args.last:
                    dates, prices = dates[-args.last:], prices[-args.last:]
                if len(dates) < 30:
                    raise ValueError('Au moins 30 seances requises pour estimation')
                psi, trace, info = estimate(prices, tau, product,
                                            previous.get(product, legacy.get(product)),
                                            args.optimizer, args.maxiter)
                result = product_results(product, dates, prices, tau, psi, trace, info,
                                         args.seed, args.forecast_days)
                results.append(result)
                state = 'CONVERGED' if info['converged'] else 'NOT_CONVERGED'
                statuses.append({'product': product, 'status': state, 'comment': info['message']})
                LOG.info('SS | DONE | %s | %s | NLL %.6f', product, state, info['nll'])
            except Exception as exc:
                statuses.append({'product': product, 'status': 'FAILED', 'comment': str(exc)})
                LOG.exception('SS | FAIL | %s', product)
        if results:
            _save_book(model_dir, results, previous, statuses)
            try:
                _make_pdf(model_dir, results, statuses, args.date, str(args.source))
            except Exception:
                LOG.exception('SS | FAIL | Dashboard PDF')
                exits.append(1)
        if any(s['status']=='FAILED' for s in statuses): exits.append(1)
        elif any(s['status']=='NOT_CONVERGED' for s in statuses): exits.append(2)
    if args.model in ('SSX','both'):
        from model_SSX import run_ssx
        exits.append(run_ssx(products, data_dir, model_dir, args))
    LOG.info('FIN | models=%s | exit_codes=%s', args.model, exits)
    if any(x == 1 for x in exits): raise SystemExit(1)
    if any(x == 2 for x in exits): raise SystemExit(2)


if __name__ == '__main__':
    main()
