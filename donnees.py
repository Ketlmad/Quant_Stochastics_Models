"""Barchart -> trois classeurs globaux Excel, un onglet par produit.

Les données Barchart ne sont jamais modifiées. La construction des courbes
reprend les conventions du projet MATLAB/R précédent.
"""
from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parent
BARCHART = Path(r"G:\Mickael\Projets\Barchart\futur\futur_extract")
DATA_OUTPUT = ROOT / "output" / "data_processed"

# Code Barchart, code historique des trois classeurs, prefixe MATLAB.
PRODUCTS = {
    "Wheat": ("ZW", "ZW", "W"),
    "HRW_Wheat": ("KE", "KE", "KE"),
    "Spring_Wheat": ("KW", "KW", "KW"),
    "Corn": ("ZC", "ZC", "ZC"),
    "Colza": ("XR", "ECO", "ECO"),
    "Milling_Wheat": ("ML", "EBM", "EBM"),
    "Soymeal": ("ZM", "ZM", "ZM"),
    "Soybeans": ("ZS", "ZS", "ZS"),
    "Soybean_Oil": ("ZL", "ZL", "ZL"),
    "Sugar": ("SW", "WENT", "WENT"),
    "Sugar11": ("SB", "SB", "SB"),
    "Canola": ("RS", "RS", "RS"),
    "Palm_Oil": ("KO", "KO", "KO"),
    "Euronext_Corn": ("XB", "EP", "EP"),
}
MATURITIES_DAYS = (30, 60, 90, 120, 150, 180)
START_DATE = "2015-06-26"  # None pour charger tout l'historique.
DATE_COL = "date_concat_final.1..nrow.date_concat_final....burning_last..."
LOG = logging.getLogger("barchart")


def latest_parquet(source: Path) -> Path:
    if source.is_file():
        return source
    candidates = sorted(source.glob("*_all_futures.parquet"))
    if not candidates:
        raise FileNotFoundError(f"Aucun Parquet Barchart dans {source}")
    return candidates[-1]


def global_paths(output: Path, run_date: str) -> dict[str, Path]:
    """Exactement trois fichiers a noms stables, quel que soit le jour."""
    prefix = "Base_ALL_"  # noms stables : les trois fichiers sont actualises, pas multiplies
    return {
        "model": output / f"{prefix}Modelisation.xlsx",
        "market": output / f"{prefix}Market_Price.xlsx",
        "ttm": output / f"{prefix}Market_Option_TTM.xlsx",
    }


def _curve(ttm: np.ndarray, prices: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Spline cubique naturelle, extrapolation linéaire aux extrémités."""
    order = np.argsort(ttm, kind="stable")
    x, y = ttm[order], prices[order]
    x, first = np.unique(x, return_index=True)
    y = y[first]
    if x.size < 3:
        return np.full(target.size, np.nan)
    spline = CubicSpline(x, y, bc_type="natural", extrapolate=True)
    result = np.asarray(spline(target))
    left, right = target < x[0], target > x[-1]
    result[left] = y[0] + spline(x[0], 1) * (target[left] - x[0])
    result[right] = y[-1] + spline(x[-1], 1) * (target[right] - x[-1])
    return result


def prepare_one(df: pd.DataFrame, product: str,
                maturity_days=MATURITIES_DAYS,
                start_date: str | None = START_DATE) -> dict:
    """Retourne les quatre tables destinées aux trois classeurs globaux.

    Deux tables (prix et maturités constantes) vont dans Modelisation.
    Les mêmes dates sont conservées pour les trois bases.
    """
    code, legacy, _ = PRODUCTS[product]
    fields = ["date", "product", "symbol", "echeance", "close"]
    if "source_file" in df.columns:
        fields.append("source_file")
    g = df.loc[df["product"].eq(code), fields].copy()
    if g.empty:
        raise ValueError(f"{product} ({code}) absent du Parquet")
    g["date"] = pd.to_datetime(g["date"]).dt.normalize()
    g["echeance"] = pd.to_datetime(g["echeance"]).dt.normalize()
    g["close"] = pd.to_numeric(g["close"])
    if start_date:
        g = g.loc[g["date"] >= pd.Timestamp(start_date)]
    if g.empty:
        raise ValueError(f"{product} ({code}) : aucune ligne après {start_date}")
    if "source_file" in g.columns:
        g.sort_values(["date", "symbol", "source_file"], kind="stable", inplace=True)
    # Une seule quote par contrat/jour pour le pivot, sans correction des cours.
    g.drop_duplicates(["date", "symbol"], keep="first", inplace=True)

    expiry = (g["echeance"] - pd.DateOffset(months=1)).dt.normalize()
    expiry = expiry.map(lambda d: d.replace(day=15) if pd.notna(d) else pd.NaT)
    g["ttm"] = (expiry - g["date"]).dt.days / 252.0
    g.loc[g["ttm"] < 0, "ttm"] = np.nan
    g["label"] = "close." + legacy + "_" + g["symbol"].astype(str).str.slice(len(code))
    g.sort_values(["echeance", "symbol"], kind="stable", inplace=True)
    columns = g["label"].drop_duplicates().tolist()
    p = g.pivot(index="date", columns="label", values="close").reindex(columns=columns).sort_index()
    t = g.pivot(index="date", columns="label", values="ttm").reindex(index=p.index, columns=columns)
    eligible_dates = p.notna().sum(axis=1) >= 3
    p, t = p.loc[eligible_dates], t.loc[eligible_dates]

    targets = np.asarray(maturity_days, dtype=float) / 252.0
    prices, ttms = p.to_numpy(dtype=float), t.to_numpy(dtype=float)
    curves = np.full((len(p), len(targets)), np.nan)
    for i in range(len(p)):
        valid = np.isfinite(prices[i]) & np.isfinite(ttms[i])
        if valid.sum() >= 3:
            curves[i] = _curve(ttms[i, valid], prices[i, valid], targets)
    # Les cours source restent intacts ; seules les dates non modélisables sont ignorées.
    valid_model_dates = np.isfinite(curves).all(axis=1) & (curves > 0).all(axis=1)
    p, t, curves = p.iloc[valid_model_dates], t.iloc[valid_model_dates], curves[valid_model_dates]
    if p.empty:
        raise ValueError(f"{product} : aucune date admissible pour log(Prix)")

    dates = p.index.strftime("%Y-%m-%d")
    model_prices = pd.DataFrame(curves, columns=[f"X{v}.days" for v in maturity_days])
    model_prices.insert(0, DATE_COL, dates)
    model_ttm = pd.DataFrame(np.tile(targets, (len(p), 1)),
                             columns=[f"TTM_{v}days" for v in maturity_days])
    model_ttm.insert(0, DATE_COL, dates)
    market = p.copy()
    market.insert(0, "date", dates)
    market_ttm = t.copy()
    market_ttm.columns = ["TTM_" + field for field in market_ttm.columns]
    market_ttm.insert(0, "date", dates)
    return {
        "product": product, "code": code, "legacy": legacy,
        "rows": len(p), "contracts": len(columns),
        "prices": model_prices, "fixed_ttm": model_ttm,
        "market": market, "market_ttm": market_ttm,
    }


def generate(products: list[str] | None = None,
             source: Path = BARCHART, output: Path = DATA_OUTPUT,
             run_date: str | None = None, start_date: str | None = START_DATE,
             maturity_days=MATURITIES_DAYS) -> list[dict]:
    """Une lecture Parquet -> 3 classeurs ALL, chacun avec tous les produits.

    Fonction idempotente : rejouer l'exécution remplace les trois classeurs
    datés, sans créer de fichiers d'entrée individuels par matière.
    """
    names = list(PRODUCTS) if products is None else list(dict.fromkeys(products))
    if unknown := set(names) - PRODUCTS.keys():
        raise ValueError(f"Instruments inconnus : {sorted(unknown)}")
    if not names:
        raise ValueError("Liste des matières vide")

    source_file = latest_parquet(Path(source))
    LOG.info("Barchart source : %s", source_file)
    df = pd.read_parquet(source_file)
    required = {"date", "product", "symbol", "echeance", "close"}
    if missing := required - set(df.columns):
        raise ValueError(f"Colonnes Barchart manquantes : {sorted(missing)}")

    stamp = run_date or date.today().isoformat()
    dest = Path(output)
    dest.mkdir(parents=True, exist_ok=True)
    files = global_paths(dest, stamp)
    summaries = []
    # Un seul writer par famille de données, quelle que soit la matière traitée.
    with (pd.ExcelWriter(files["model"], engine="xlsxwriter") as wm,
          pd.ExcelWriter(files["market"], engine="xlsxwriter") as wp,
          pd.ExcelWriter(files["ttm"], engine="xlsxwriter") as wt):
        for product in names:
            data = prepare_one(df, product, maturity_days, start_date)
            legacy = data["legacy"]
            data["prices"].to_excel(wm, index=False, sheet_name=legacy)
            data["fixed_ttm"].to_excel(wm, index=False, sheet_name=f"{legacy}_TTM")
            data["market"].to_excel(wp, index=False, sheet_name=legacy)
            data["market_ttm"].to_excel(wt, index=False, sheet_name=legacy)
            for writer, sheet in ((wm, legacy), (wm, f"{legacy}_TTM"),
                                  (wp, legacy), (wt, legacy)):
                writer.sheets[sheet].freeze_panes(1, 1)
                writer.sheets[sheet].set_column(0, 0, 19)
            record = {key: data[key] for key in ("product", "code", "legacy", "rows", "contracts")}
            record["snapshot"] = source_file.name
            record["run_date"] = stamp
            summaries.append(record)
            LOG.info("%s [%s] : %d séances, %d contrats", product, legacy,
                     data["rows"], data["contracts"])
        index = pd.DataFrame(summaries)
        for writer in (wm, wp, wt):
            index.to_excel(writer, index=False, sheet_name="INDEX")
            writer.sheets["INDEX"].freeze_panes(1, 0)
            writer.sheets["INDEX"].set_column(0, 2, 21)
            writer.sheets["INDEX"].set_column(3, 4, 14)
    for f in files.values():
        LOG.info("Excel global : %s", f)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Barchart -> 3 Excel globaux (toutes matières)")
    parser.add_argument("products", nargs="*", help="Produits ciblés ; vide / ALL = tous")
    parser.add_argument("--source", type=Path, default=BARCHART)
    parser.add_argument("--output", type=Path, default=DATA_OUTPUT)
    parser.add_argument("--start", default=START_DATE, help="Début d'échantillon ou ALL")
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args()
    names = list(PRODUCTS) if not args.products or "ALL" in args.products else args.products
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    generate(names, args.source, args.output, args.date,
             None if args.start == "ALL" else args.start)


if __name__ == "__main__":
    main()
