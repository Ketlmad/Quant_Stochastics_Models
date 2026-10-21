"""SSX two-mean-reverting-factor extension of Schwartz--Smith.

Direct scientific port of MainSSX.m / Kalman_Estimation_SSX.m with explicit
corrections documented in README.md. Barchart and maturities are unchanged.
Importantly this module shares the same Barchart input, one Kalman module,
and the global Excel writer with modele.py; it has independent model states.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from donnees import PRODUCTS
from kalman import kalman_ssx, matrices_ssx, _ou_integral
from modele import (DT, _data_initialization, _lb_pvalue, _save_book,
                    future_dates, load_input, previous_state)

LOG = logging.getLogger('schwartz-smith-ssx')
PARAM_NAMES_SSX = ('k','sigmax','lambdax','mu','sigmae','lambdae','pxe','gamma')
RESULT_SSX = 'ALL_Model_Results_SSX.xlsx'
REPORT_SSX = 'ALL_Technical_Dashboard_SSX.pdf'
LOWER = np.array([1e-5,1e-5,-10.,-10.,1e-5,-10.,-.999,1e-5])
UPPER = np.array([10.,10.,10.,10.,10.,10.,.999,10.])
SCALES = np.array([.5,.25,.25,.05,.25,.05,.25,.20])


def initialize_ssx(prices, tau, warm=None, ss_prior=None):
    """Warm SSX, optional SS seed, or data-driven. Do not reuse final latent
    state from yesterday as initial state of a full historical refilter.
    """
    n = prices.shape[1]
    base, _, _, _, _, _ = _data_initialization(prices,tau)
    origin = 'data_driven'
    psi = np.r_[base[:7],min(.25,max(.015,.4*base[0])),base[7:]]
    if ss_prior is not None and len(ss_prior['psi']) >= 7:
        vec=np.asarray(ss_prior['psi'],float)
        if np.isfinite(vec[:7]).all():
            psi[:7] = vec[:7]
            psi[7] = min(.25,max(.005,.4*psi[0]))
            origin = 'SS_seed'
    if warm is not None:
        vec=np.asarray(warm['psi'],float)
        if len(vec)==8+n and np.isfinite(vec).all():
            psi=vec.copy()
            origin='last_converged_SSX'
    low=np.r_[LOWER,np.full(n,1e-5)]
    high=np.r_[UPPER,np.full(n,.5)]
    psi=np.clip(psi, low+1e-7, high-1e-7)
    # Long-run equilibrium is by definition slower than short-run deviation.
    psi[7]=min(psi[7],psi[0]*.8)
    _,_,Q,d,H=matrices_ssx(psi,tau,DT)
    y=np.log(prices[0])
    # Decompose first observed curve into the two latent components.
    # Small regularizer prevents catastrophic x0 if factors nearly collinear.
    center = y.mean()
    init = np.linalg.solve(H.T@H + 1e-5*np.eye(2), H.T@(y-d))
    if not np.isfinite(init).all() or max(abs(init))>25:
        init=np.array([0.,center])
    x0=np.asarray(init,dtype=float)
    # Covariance initialized at the discretized state innovation covariance.
    P0=Q.copy()
    return psi,low,high,x0,P0,origin


def estimate_ssx(prices,tau,product,warm=None,ss_prior=None,solver='SLSQP',
                 maxiter=1000,starts=2):
    y=np.log(prices)
    p0,lb,ub,x0,P0,origin=initialize_ssx(prices,tau,warm,ss_prior)
    scales=np.r_[SCALES,np.full(len(tau),.005)]
    bounds=list(zip(lb/scales,ub/scales))
    n=len(y)
    best_psi=p0.copy()
    best_score=kalman_ssx(y,p0,tau,x0,P0,DT)
    initial_score=best_score
    best_success=False
    best_run=None
    attempts=[]
    if not np.isfinite(best_score):
        raise RuntimeError(f'{product}: SSX initial NLL non-finie')
    for j in range(max(1,starts)):
        start=p0.copy()
        if j==1:
            start[7]=max(lb[7]+1e-5,min(start[0]*.85,max(.015,start[7]*1.7)))
        elif j>1:
            start[7]=max(lb[7]+1e-5,min(start[0]*.3,max(.005,start[7]*.5)))
        # One single constraint establishes the short-term vs long-term labeling.
        # L-BFGS-B cannot enforce nonrectangular bounds => add explicit penalty.
        def objective(z):
            ps=np.asarray(z)*scales
            if ps[7]>=ps[0]-1e-4:
                return 1e5 + 1e6*max(0.,ps[7]-ps[0]+1e-4)**2
            out=kalman_ssx(y,ps,tau,x0,P0,DT)
            return out/n if np.isfinite(out) else 1e30
        options={'maxiter':maxiter,'disp':False}
        constraints=[]
        if solver=='SLSQP':
            options['ftol']=1e-9
            constraints=[{'type':'ineq','fun':lambda z: z[0]*scales[0]-z[7]*scales[7]-1e-4}]
        elif solver=='L-BFGS-B':
            options.update({'maxfun':10000,'ftol':1e-12,'gtol':1e-7,'maxls':40})
        elif solver=='trust-constr':
            from scipy.optimize import LinearConstraint
            M=np.zeros((1,len(start)));M[0,0]=scales[0];M[0,7]=-scales[7]
            constraints=[LinearConstraint(M,1e-4,np.inf)]
            options.update({'gtol':1e-7,'xtol':1e-9})
        else:
            raise ValueError(solver)
        try:
            res=minimize(objective,start/scales,method=solver,bounds=bounds,
                         constraints=constraints,options=options)
            candidate=np.asarray(res.x)*scales
            score=kalman_ssx(y,candidate,tau,x0,P0,DT) if candidate[7]<candidate[0]-1e-4 else np.inf
            attempts.append({'start':j+1,'success':bool(res.success),'iterations':int(res.nit),
                             'nll':float(score),'message':str(res.message)})
            if np.isfinite(score) and score <= best_score + 1e-7 and (score < best_score-1e-7 or res.success):
                best_psi,best_score,best_success,best_run=candidate,score,bool(res.success),res
        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
            attempts.append({'start':j+1,'success':False,'iterations':0,'nll':np.inf,
                             'message':f'Exception {exc}'})
    final,trace=kalman_ssx(y,best_psi,tau,x0,P0,DT,details=True)
    if trace is None:
        raise RuntimeError(f'{product}: SSX final Kalman unavailable')
    info={'nll':float(final),'nll_initial':float(initial_score),'converged':best_success,
          'iterations':int(best_run.nit) if best_run is not None else 0,
          'optimizer':solver,'origin':origin,
          'message':str(best_run.message) if best_run is not None else 'No successful optimization',
          'x0':x0,'P0':P0,'attempts':attempts}
    LOG.info('SSX | %s | NLL %.3f -> %.3f | converged=%s | starts=%d | k=%.4f gamma=%.4f',
             product,initial_score,final,best_success,len(attempts),best_psi[0],best_psi[7])
    return best_psi,trace,info


def _forecasts_ssx(psi,tau,last_state,end_date,days=40,seed=2000,product='Wheat',paths=1200):
    A,c,Q,d,H=matrices_ssx(psi,tau,DT)
    days_idx=future_dates(end_date,days)
    code=sum((i+1)*ord(ch) for i,ch in enumerate(product))
    rng=np.random.default_rng(np.random.SeedSequence([seed,code,8675309]))
    sims=np.broadcast_to(np.asarray(last_state,float),(paths,2)).copy()
    quant=np.empty((len(days_idx),len(tau),5))
    for t in range(len(days_idx)):
        sims=sims@A.T+c+rng.multivariate_normal(np.zeros(2),Q,size=paths)
        future=np.exp(sims@H.T+d)
        quant[t]=np.quantile(future,[.05,.25,.5,.75,.95],axis=0).T
    return days_idx,quant


def product_results_ssx(product, dates, prices, tau, psi, trace, info,
                        seed=2000,forecast_days=40):
    _,legacy,_=PRODUCTS[product]
    y=np.log(prices);n,N=prices.shape
    _,_,_,d,H=matrices_ssx(psi,tau,DT)
    fitted=np.exp(trace['states']@H.T+d)
    residuals=y-np.log(fitted)
    innovations=trace['innovations']
    error_pct=(fitted-prices)/prices
    corrs=np.corrcoef(innovations.T)
    pvals=[_lb_pvalue(innovations[:,j]) for j in range(N)]
    k,sx,lx,mu,se,le,rho,gamma=psi[:8]
    Bk=_ou_integral(k,tau);Bg=_ou_integral(gamma,tau)
    variance=sx*sx*_ou_integral(2*k,tau)+se*se*_ou_integral(2*gamma,tau)+2*rho*sx*se*_ou_integral(k+gamma,tau)
    dP=mu*Bg+.5*variance
    exp_spot=np.exp(trace['states']@H.T+dP)
    spot=np.exp(trace['states'].sum(axis=1))
    # Exact log annualized P-vs-Q premium; not a realized return forecast.
    log_premium_pa=(lx*Bk+le*Bg)/tau
    instantaneous_vol=np.sqrt(np.maximum(0.,np.exp(-2*k*tau)*sx*sx+
                    np.exp(-2*gamma*tau)*se*se+
                    2*rho*sx*se*np.exp(-(k+gamma)*tau)))
    close_date=str(dates[-1].date())
    fdates,quant=_forecasts_ssx(psi,tau,trace['states'][-1],close_date,
                                forecast_days,seed,product)
    pars={'product':product,'code':legacy,'asof':close_date,
          'first_date':str(dates[0].date()),'nobs':n,'nll':info['nll'],
          'nll_initial':info['nll_initial'],'nll_per_obs':info['nll']/n,
          'converged':info['converged'],'iterations':info['iterations'],
          'optimizer':info['optimizer'],'warm_source':info['origin'],'message':info['message'],
          'x0_short':info['x0'][0],'x0_long':info['x0'][1],
          'last_short':trace['states'][-1,0],'last_long':trace['states'][-1,1],
          'rmse_price_pct':float(np.sqrt(np.mean(error_pct**2))*100),
          'lb_pvalue_min':float(np.nanmin(pvals)),
          'residual_max_corr':float(np.nanmax(np.abs(corrs-np.eye(N)))),
          'half_life_short_years':float(np.log(2)/k),
          'half_life_long_years':float(np.log(2)/gamma),
          'aic':float(2*len(psi)+2*info['nll']),
          'bic':float(len(psi)*np.log(n*N)+2*info['nll']),
          'near_identification_boundary':bool(abs(rho)>.98 or abs(k-gamma)<.015)}
    pars.update(dict(zip(PARAM_NAMES_SSX,psi[:8])))
    pars.update({f'err_{j+1}':v for j,v in enumerate(psi[8:])})
    def wide(values,names):
        d=pd.DataFrame(values,columns=names)
        d.insert(0,'date',pd.DatetimeIndex(dates));d.insert(0,'product',product)
        return d
    tenor=[int(round(x*252)) for x in tau]
    latent=wide(np.c_[trace['states'],spot],['short_term','long_term','estimated_spot'])
    fit=wide(np.c_[prices,fitted], [f'observed_{v}d' for v in tenor]+[f'fitted_{v}d' for v in tenor])
    residual=wide(residuals,[f'resid_log_{v}d' for v in tenor])
    innovation=wide(innovations,[f'innovation_log_{v}d' for v in tenor])
    expected=wide(exp_spot,[f'expected_spot_{v}d' for v in tenor])
    curve=[]
    for j,day in enumerate(tenor):
        curve.append({'product':product,'date':close_date,'maturity_days':day,
                      'observed':prices[-1,j],'fitted':fitted[-1,j],
                      'expected_spot_P':exp_spot[-1,j],
                      'estimated_spot':spot[-1],
                      'rmse_log':float(np.sqrt(np.mean(residuals[:,j]**2))),
                      'risk_premium_log_ann':float(log_premium_pa[j]),
                      'instant_vol_annual':float(instantaneous_vol[j]),
                      'implied_carry_log_ann':float(np.log(fitted[-1,j]/spot[-1])/tau[j])})
    forecast=[]
    for i,day in enumerate(fdates):
        for j,mat in enumerate(tenor):
            a,b,c0,d0,e=quant[i,j]
            forecast.append({'product':product,'forecast_date':day,
                             'maturity_days':mat,'horizon_sessions':i+1,
                             'p05':a,'p25':b,'p50':c0,'p75':d0,'p95':e,
                             'signal_pct':float((c0/prices[-1,j]-1)*100)})
    curve=pd.DataFrame(curve);forecast=pd.DataFrame(forecast)
    return {'row':pars,'curves':curve,'forecast':forecast,'latent':latent,
            'fit':fit,'residual':residual,'innovation':innovation,'expected':expected,
            'pdf':{'product':product,'date':close_date,'dates':pd.DatetimeIndex(dates),
                   'prices':prices,'fitted':fitted,'states':trace['states'],
                   'spot':spot,'expected':exp_spot,'residuals':residuals,
                   'innovations':innovations,'tau':tau,'curves':curve,
                   'forecasts':forecast,'params':pars,'corrs':corrs,'pvals':pvals}}


def _compare_ss(path,results):
    old=path/'ALL_Model_Results.xlsx'
    if not old.exists():return pd.DataFrame()
    try:
        ss=pd.read_excel(old,sheet_name='PARAMETERS')
        records=[]
        for r in results:
            p=r['row']; matches=ss.loc[ss['product'].eq(p['product'])]
            if matches.empty:continue
            a=matches.iloc[0]
            comparable=(bool(a['converged']) and p['converged'] and
                        int(a['nobs'])==p['nobs'] and str(a['asof'])==p['asof'] and
                        str(a['first_date'])==p['first_date'])
            aic_ss=2*(13+float(a['nll']))
            records.append({'product':p['product'],'same_sample':bool(comparable),
                            'nobs':p['nobs'],'SS_RMSE_pct':float(a['rmse_price_pct']),
                            'SSX_RMSE_pct':p['rmse_price_pct'],
                            'SS_NLL':float(a['nll']),'SSX_NLL':p['nll'],
                            'SS_AIC':aic_ss if comparable else np.nan,
                            'SSX_AIC':p['aic'] if comparable else np.nan,
                            'delta_AIC_SSX_minus_SS':(p['aic']-aic_ss) if comparable else np.nan,
                            'preferred_by_AIC':('SSX' if p['aic']<aic_ss else 'SS') if comparable else 'NOT_COMPARABLE'})
        return pd.DataFrame(records)
    except (ValueError,OSError,KeyError) as exc:
        LOG.warning('SS comparison unavailable: %s',exc)
        return pd.DataFrame()


def _make_ssx_pdf(path,results,status,run_date,comparison):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import MaxNLocator

    NAVY='#15263E';TEAL='#147C8C';ORANGE='#C18A3A';GRAY='#586B7B'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.edgecolor':'#CBD4DD','axes.grid':True,'grid.alpha':.18,
                         'axes.titleweight':'bold','figure.facecolor':'white'})
    target=path/REPORT_SSX;temp=path/('_'+REPORT_SSX)
    with PdfPages(temp,metadata={'Title':'SSX | Two Mean-Reverting Factors | Quant Research',
                                 'Author':'Stochastic Models'}) as book:
        fig=plt.figure(figsize=(11.69,8.27))
        fig.text(.065,.90,'STOCHASTIC MODELS  /  QUANT RESEARCH',color=TEAL,size=12,weight='bold')
        fig.text(.065,.83,'SSX  |  TWO MEAN-REVERTING FACTORS',color=NAVY,size=20,weight='bold')
        fig.text(.065,.76,'Extended Schwartz-Smith: transient and equilibrium Ornstein-Uhlenbeck dynamics',color=GRAY,size=11)
        fig.text(.065,.68,f'{len(results)} modeled products   •   {sum(r["row"]["converged"] for r in results)} converged   •   As of {run_date}',size=10)
        ax=fig.add_axes([.06,.19,.90,.40]);ax.axis('off')
        rows=[]
        for r in results:
            p=r['row']
            rows.append([p['product'],f"{p['rmse_price_pct']:.2f}%",f"{p['k']:.3f}",f"{p['gamma']:.3f}",
                         f"{p['half_life_long_years']:.2f}",
                         'YES' if p['converged'] else 'NO',
                         '!' if p['near_identification_boundary'] else '—'])
        table=ax.table(cellText=rows,colLabels=['Product','RMSE','k','gamma','Long HL (y)','Converged','ID risk'],
                       loc='upper left',cellLoc='left',colLoc='left',colWidths=[.22,.10,.10,.10,.14,.16,.09])
        table.auto_set_font_size(False);table.set_fontsize(8);table.scale(1,1.40)
        for (i,j),cell in table.get_celld().items():
            cell.set_edgecolor('#E2E7EC')
            if i==0:
                cell.set_facecolor(NAVY);cell.get_text().set_color('white');cell.get_text().set_weight('bold')
            elif i%2==0:cell.set_facecolor('#F2F5F7')
        fig.text(.065,.12,'SSX is a research extension, not the original 2000 Schwartz-Smith model.',color=GRAY)
        fig.text(.065,.085,'!  parameter-correlation near ±1 or nearly equal mean-reversion speeds; forecasts require out-of-sample validation.',color=GRAY)
        book.savefig(fig);plt.close(fig)

        fig,axes=plt.subplots(1,2,figsize=(11.69,8.27));fig.subplots_adjust(left=.10,right=.94,wspace=.32,top=.80,bottom=.18)
        fig.suptitle('Cross-Commodity  |  Model Selection & Reversion',color=NAVY,fontsize=17,weight='bold')
        for r in results:
            p=r['row'];axes[0].scatter(p['half_life_short_years'],p['half_life_long_years'],s=45,color=TEAL)
            axes[0].annotate(p['code'],(p['half_life_short_years'],p['half_life_long_years']),xytext=(4,3),textcoords='offset points',fontsize=7)
        axes[0].set_title('Short versus equilibrium half-life')
        axes[0].set_xlabel('Short-term half-life (years)');axes[0].set_ylabel('Equilibrium half-life (years)')
        if not comparison.empty and comparison['same_sample'].any():
            c=comparison.loc[comparison.same_sample].sort_values('delta_AIC_SSX_minus_SS')
            axes[1].barh(c['product'],c['delta_AIC_SSX_minus_SS'],color=[TEAL if z<0 else ORANGE for z in c['delta_AIC_SSX_minus_SS']])
            axes[1].axvline(0,color=NAVY,lw=.8)
            axes[1].set_title('SSX - SS  |  delta AIC (negative favors SSX)')
            axes[1].tick_params(axis='y',labelsize=7)
        else:
            axes[1].axis('off');axes[1].text(.1,.5,'SS comparable historical results unavailable',size=12,color=GRAY)
        book.savefig(fig);plt.close(fig)

        for r in results:
            p=r['row'];d=r['pdf'];t=d['dates'];sl=slice(max(0,len(t)-252),None)
            fig,axs=plt.subplots(3,2,figsize=(11.69,8.27))
            fig.subplots_adjust(left=.08,right=.96,top=.82,bottom=.11,hspace=.50,wspace=.28)
            fig.suptitle(f"{p['product']}   |   SSX TWO OU   |   {p['asof']}",fontsize=16,color=NAVY,weight='bold',y=.96)
            fig.text(.08,.89,f"RMSE {p['rmse_price_pct']:.2f}%   •   k={p['k']:.4f} gamma={p['gamma']:.4f}   •   rho={p['pxe']:.3f}   •   {'CONVERGED' if p['converged'] else 'NOT CONVERGED'}",color=TEAL,size=9)
            a=axs[0,0]
            for j,col in [(0,TEAL),(-1,ORANGE)]:
                label=int(round(d['tau'][j]*252))
                a.plot(t[sl],d['prices'][sl,j],color=col,lw=1,label=f'{label}d market')
                a.plot(t[sl],d['fitted'][sl,j],color=col,ls='--',lw=1,alpha=.7,label=f'{label}d SSX')
            a.set_title('Futures curve fit  |  last 252 sessions');a.legend(fontsize=7,frameon=False,ncol=2)
            a.tick_params(axis='x',rotation=25)

            a=axs[0,1]
            a.plot(t[sl],d['states'][sl,0],color=TEAL,label='Transient OU')
            a.plot(t[sl],d['states'][sl,1],color=ORANGE,label='Equilibrium OU')
            a.set_title('Both latent factors mean revert');a.legend(fontsize=7,frameon=False)
            a.tick_params(axis='x',rotation=25)

            a=axs[1,0];c=d['curves']
            a.plot(c['maturity_days'],c['fitted'],color=TEAL,marker='o',label='Futures Q')
            a.plot(c['maturity_days'],c['expected_spot_P'],color=ORANGE,marker='s',label='E_P future spot')
            a.axhline(c['estimated_spot'].iloc[0],color=GRAY,ls=':',lw=1,label='Spot estimated')
            a.set_title('P physical expectation versus Q futures')
            a.set_xlabel('Tenor (business days)');a.legend(fontsize=7,frameon=False)

            a=axs[1,1]
            a.plot(c['maturity_days'],100*c['risk_premium_log_ann'],color=TEAL,marker='o')
            a.set_title('Annualized P-Q log premium (model implied)')
            a.set_xlabel('Tenor (business days)');a.set_ylabel('log premium % p.a.')
            a.axhline(0,color=GRAY,lw=.8)

            a=axs[2,0];series=d['forecasts']
            for tenor,col in [(int(round(d['tau'][0]*252)),TEAL),
                              (int(round(d['tau'][-1]*252)),ORANGE)]:
                g=series.loc[series.maturity_days.eq(tenor)].iloc[:20]
                x=np.arange(1,len(g)+1)
                a.plot(x,g.p50.to_numpy(),color=col,label=f'{tenor}d median')
                a.fill_between(x,g.p05.to_numpy(),g.p95.to_numpy(),color=col,alpha=.13)
            a.set_title('20-session conditional P Monte-Carlo  |  5%-95%')
            a.set_xlabel('Forecast sessions');a.legend(fontsize=7,frameon=False)

            a=axs[2,1]
            rms=100*np.sqrt(np.mean(d['residuals']**2,axis=0))
            a.bar([int(round(z*252)) for z in d['tau']],rms,width=19,color=TEAL)
            a.set_title('In-sample fit error by tenor')
            a.set_xlabel('Tenor (days)');a.set_ylabel('log-RMSE × 100')
            a.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=7))
            fig.text(.08,.048,f"Equilibrium half-life {p['half_life_long_years']:.2f} yr    |    Ljung-Box min p {p['lb_pvalue_min']:.2g}    |    Monte-Carlo excludes measurement noise and is NOT out-of-sample validated.",color=GRAY,size=7)
            book.savefig(fig);plt.close(fig)
    temp.replace(target)
    LOG.info('SSX | PDF %s | %d pages',target,len(results)+2)
    return target


def run_ssx(products, data_dir, model_dir, args):
    previous=previous_state(model_dir/RESULT_SSX,PARAM_NAMES_SSX)
    ss_previous=previous_state(model_dir/'ALL_Model_Results.xlsx')
    results,statuses=[],[]
    for product in products:
        LOG.info('SSX | START | %s',product)
        try:
            dates,prices,tau=load_input(product,data_dir,args.date)
            if args.last:dates,prices=dates[-args.last:],prices[-args.last:]
            if len(dates)<30:raise ValueError('Minimum 30 observations')
            psi,trace,info=estimate_ssx(prices,tau,product,previous.get(product),
                                         ss_previous.get(product),args.optimizer,
                                         args.maxiter,args.ssx_starts)
            result=product_results_ssx(product,dates,prices,tau,psi,trace,info,
                                        args.seed,args.forecast_days)
            results.append(result)
            statuses.append({'product':product,'status':'CONVERGED' if info['converged'] else 'NOT_CONVERGED','comment':info['message']})
        except Exception as exc:
            statuses.append({'product':product,'status':'FAILED','comment':str(exc)})
            LOG.exception('SSX | FAIL | %s',product)
    if results:
        comparison=_compare_ss(model_dir,results)
        _save_book(model_dir,results,previous,statuses,result_file=RESULT_SSX,
                   param_names=PARAM_NAMES_SSX,
                   extra_tables={'EXPECTED_SPOT_P':pd.concat([r['expected'] for r in results],ignore_index=True),
                                 'COMPARISON_SS':comparison if not comparison.empty else pd.DataFrame({'info':['No comparable SS model results']})})
        try:_make_ssx_pdf(model_dir,results,statuses,args.date,comparison)
        except Exception:
            LOG.exception('SSX | FAIL | PDF');return 1
    LOG.info('SSX | completed %d/%d | converged %d',len(results),len(products),sum(r['row']['converged'] for r in results))
    if any(s['status']=='FAILED' for s in statuses):return 1
    if any(s['status']=='NOT_CONVERGED' for s in statuses):return 2
    return 0
