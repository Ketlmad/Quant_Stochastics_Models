# Stochastic Models | Barchart | Schwartz–Smith & SSX

**V1.4 — Deux modèles, un pipeline Barchart, deux dashboards PDF, deux classeurs globaux.**

Projet de recherche : modèle original **Schwartz–Smith (SS, 2000)** et extension **SSX à deux facteurs de retour à la moyenne**. Le code MATLAB transmis est conservé comme référence conceptuelle ; le port SSX corrige explicitement plusieurs incohérences numériques et économiques (ci-dessous). Il ne constitue donc pas une reproduction *bit à bit* du MATLAB SSX d'origine.

## Installation / exécution

Décompresser directement dans `G:\Mickael\Projets\Stochastic_Models`. `Main.bat` installe les `requirements.txt` au moyen du bootstrap Windows existant (venv partagé dans le dossier parent), puis lance le programme.

```bat
Main.bat
Main.bat --model SS
Main.bat --model SSX
Main.bat --model both
Main.bat --product Wheat --model both
Main.bat --data-only
```

**`Main.bat` lance par défaut les 14 produits et les deux modèles.** Le fichier Barchart le plus récent (par nom `YYYY-MM-DD_all_futures.parquet`) est lu à chaque exécution ; pas d'API Barchart ni de modification de la base d'origine. Le fichier de préparation est partagé par SS et SSX : pas de double lecture ou double interpolation.

Options utiles : `--no-prepare`, `--last 1000` (échantillon réduit), `--maxiter 1000`, `--ssx-starts 2`, `--optimizer SLSQP` ou `L-BFGS-B`, `--forecast-days 40`, `--source CHEMIN`, `--output CHEMIN`.

## Architecture minimaliste

```text
Stochastic_Models/
├── Main.bat                 Lanceur unique (SS + SSX)
├── Bootstrap.bat            Initialisation environnement Windows
├── requirements.txt
├── donnees.py               Barchart -> 3 classeurs normalisés
├── kalman.py                Filtre SS + filtre SSX (équations séparées)
├── modele.py                Modèle SS, préparation commune, exports SS
├── model_SSX.py             Modèle SSX et exports SSX
├── README.md
├── logs/
└── output/
    ├── data_processed/
    │   ├── Base_ALL_Modelisation.xlsx
    │   ├── Base_ALL_Market_Price.xlsx
    │   └── Base_ALL_Market_Option_TTM.xlsx
    └── sorties_models/
        ├── ALL_Model_Results.xlsx
        ├── ALL_Technical_Dashboard.pdf
        ├── ALL_Model_Results_SSX.xlsx
        └── ALL_Technical_Dashboard_SSX.pdf
```

Le classeur SS historique conserve **ses noms de fichiers et neuf onglets**. Le classeur SSX présente les mêmes vues de base, un onglet `EXPECTED_SPOT_P` et un onglet `COMPARISON_SS`. `WARM_START` est **indépendant pour chaque modèle** : les 7 paramètres SS ne sont pas confondus avec les 8 paramètres SSX. Les estimations non convergées ne remplacent jamais une calibration précédemment convergée.

La préparation des prix interpole six maturités fixes de **30, 60, 90, 120, 150 et 180 jours de bourse**. Elle reprend les conventions existantes de `donnees.py` sans modifier les données source Barchart.

## Vérification des forecasts existants SS

Sur le classeur fourni `ALL_Model_Results.xlsx`, l'onglet `FORECAST_MC` a **3 360 lignes** : 14 produits × 6 maturités × 40 séances futures. Les champs `p05`, `p25`, `p50`, `p75`, `p95` sont renseignés et croissants. Les simulations sont **sous la mesure historique P** (évolution des deux facteurs) et les contrats futurs à la date projetée sont valorisés avec l'interception risque-neutre **Q**. Elles **ne sont pas** des prévisions validées hors échantillon ; elles n'incluent pas de bruit d'observation. Le PDF SS ne montre que les 20 premières séances pour la lisibilité.

### Diagnostic rapide sur ces résultats

Les 14 calibrations portent le statut convergé ; RMSE de reconstruction en échantillon de 0,09 % (ZW) à 0,73 % (ZS). Plusieurs produits ont des corrélations latentes presque égales à −1 (`ZM` ≈ −0,9985 ; `EBM` ≈ −0,9964 ; `ZS` ≈ −0,9893 ; `RS` ≈ −0,9851). Le modèle est donc susceptible de rencontrer une **faible identification des composantes**, même si l'ajustement apparent est bon. De nombreuses innovations échouent au test de Ljung–Box. Le SSX ne doit pas être jugé sur RMSE seul : examiner AIC à échantillon identique, p-values, stabilité des paramètres et backtests.

## Théorie : SSX deux Ornstein–Uhlenbeck

Au contraire du SS original (équilibre non stationnaire), SSX suit :

\[
\log S_t=X_t+E_t,\qquad
 dX_t=-kX_tdt+\sigma_xdW_{x,t},\qquad
 dE_t=(\mu-\gamma E_t)dt+\sigma_e dW_{e,t},
\]

avec `corr(dWx, dWe) = ρ`. Sous mesure risque-neutre Q, les primes constantes `λx` et `λe` déplacent les dérives de `−λx` et `−λe` :

\[
\log F(t,T)=e^{-k\tau}X_t+e^{-\gamma\tau}E_t+D_Q(\tau),
\]

\[
D_Q(\tau)=-\lambda_x B_k(\tau)+(\mu-\lambda_e)B_\gamma(\tau)+\tfrac12 V(\tau),
\quad B_a(\tau)=\frac{1-e^{-a\tau}}{a},
\]

\[
V(\tau)=\sigma_x^2B_{2k}(\tau)+\sigma_e^2B_{2\gamma}(\tau)+2\rho\sigma_x\sigma_e B_{k+\gamma}(\tau).
\]

Le spot futur attendu sous P remplace `D_Q` par `D_P=μ Bγ + V/2`. La prime du dashboard est `log(E_P[S_T]/F(t,T))/τ = [λx Bk + λe Bγ]/τ` : une **prime implicite P–Q log annualisée**, pas un rendement réalisé. On impose `0<gamma<k` pour identifier le facteur d'équilibre comme le facteur lent ; c'est une convention d'identification et une restriction du domaine de calibration.

### Corrections par rapport au MATLAB `MainSSX.m` et `Kalman_Estimation_SSX.m`

1. **Dynamique du long terme.** Le MATLAB utilise une transition `exp(-gamma*dt)` mais un terme constant `mu*dt`. La discrétisation exacte est `mu*(1-exp(-gamma*dt))/gamma`. Le nouveau filtre applique la formule exacte, y compris près de `gamma = 0`.
2. **Vraisemblance.** Le filtre MATLAB écarte le terme du déterminant et le terme quadratique à `t=1`, mais conserve la constante gaussienne sur `N*nobs`. La nouvelle version garde les trois termes et chaque observation, comme le filtre SS, avec prédiction puis mise à jour.
3. **Covariance initiale.** Le code MATLAB mélange des expressions avec ambiguïté de parenthèses (ex. `sigmax^2/2*k` au lieu de `sigmax^2/(2*k)`). La nouvelle version initialise `P0=Q` à partir des intégrales OU exactes.
4. **Mise à jour covariance.** `Joseph form` et factorisation Cholesky sont utilisés pour la stabilité numérique (notamment lorsque ρ tend vers −1). L'objectif n'est pas de faire correspondre chaque dernier chiffre à `pinv` MATLAB.
5. **Demi-vie.** Les demi-vies correctes sont `ln(2)/k` et `ln(2)/gamma`, et non `ln(2)*k` / `ln(2)*gamma`.
6. **Statistiques de significativité.** Les t-tests MATLAB avec `dof=1`, Hessiennes potentiellement indéfinies puis extraction de la seule partie réelle ont été **exclus** des exports : ils ne fournissent pas d'inférence fiable. L'ajout d'une Hessienne robuste / profils de vraisemblance viendra après validation de l'identifiabilité.
7. **Primes et expected spot.** On distingue explicitement `F_Q(t,T)` de `E_P[S_T]`. Les anciennes expressions de `Cost_Carry` et `risk_premium_bd` ne sont pas reprises telles quelles, car elles confondent parfois niveaux, horizons et signes.
8. **Format d'entrée.** L'ancien MATLAB note « All prices in log », tout en appelant `log(data)`. Le Python utilise les prix positifs Barchart **une seule fois** via `log(prix)`.

L'extension à deux OU s'inscrit dans la lignée du modèle général à deux retours à la moyenne de **Korn (2005)**, et des modèles long/short à facteurs moyennants évoqués par **Peters, Briers, Shevchenko & Doucet (2013)** ; elle n'est **pas** une affirmation que la version originale Schwartz–Smith (2000) serait stationnaire.

### Documentation scientifique

- Schwartz, E. & Smith, J. E. (2000). *Short-Term Variations and Long-Term Dynamics in Commodity Prices*. Management Science 46(7):893–911. https://doi.org/10.1287/mnsc.46.7.893.12034
- Korn, O. (2005). *Drift matters: An analysis of commodity derivatives*. Journal of Futures Markets 25(3):211–241. https://doi.org/10.1002/fut.20139
- Peters, G. W., Briers, M., Shevchenko, P. V. & Doucet, A. (2013). *Calibration and Filtering for Multi Factor Commodity Models with Seasonality*. Methodology and Computing in Applied Probability 15(4):841–874. https://doi.org/10.1007/s11009-012-9286-7

## Validation & limites

- Test limite `gamma → 0` : écart maximal des matrices SS vs SSX < `1e−12`, et écart sur NLL < `5e−9` pour un jeu de 65 observations synthétiques.
- Test synthétique d'optimisation SSX et de simulation à six maturités : exécuté.
- Écriture du classeur SSX et génération/inspection de son PDF : exécutées sur un jeu synthétique.
- **Les 14 calibrations complètes SSX sur le Parquet quotidien n'ont pas encore été exécutées sur le poste Windows**. Les paramètres SSX et l'amélioration prédictive ne sont donc pas encore validés empiriquement.
- Les prévisions 5%-95% sont des **intervalles de scénario issus du modèle**, pas des intervalles de confiance calibrés ou des garanties de couverture.
- Une comparaison AIC n'est activée que lorsque SS et SSX ont le même début, la même fin et le même nombre d'observations. Une comparaison sur un seul jeu ne remplace pas l'analyse hors échantillon.

## Conseils de production

`Main.bat --model both` reconstruit les bases et lance SS puis SSX sur les 14 produits. Cela peut prendre sensiblement plus de temps qu'un lancement SS seul, en particulier avec deux points de départ SSX. Les erreurs d'un produit sont journalisées et n'empêchent pas les autres de tourner. Un statut `NOT_CONVERGED` est conservé dans `STATUS` et la calibration non convergée ne remplace pas `WARM_START`. Pour exécuter seulement la nouvelle calibration sans recalculer SS : `Main.bat --model SSX --no-prepare`.