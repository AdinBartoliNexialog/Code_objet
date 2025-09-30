# main.py  (version avec plot)
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Dict
import pandas as pd
import logging
import sys

try:
    from CompanyDataLoader import CompanyDataLoader
except Exception:
    CompanyDataLoader = None
    print("NO CORRECT IMPORT")

try:
    from costs.carbon_cost import CarbonCost
    from costs.energy_cost import EnergyCost
except Exception:
    CarbonCost = None
    EnergyCost = None

# plotting
import matplotlib.pyplot as plt


def load_bilan_excel(path: str | Path, instrument: Optional[str] = None) -> pd.DataFrame:
    """Lit l'Excel et renvoie un DataFrame indexé par Date.

    Ce loader est identique à celui attendu par CompanyDataLoader.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_excel(p, sheet_name=0)
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    if "Date" not in df.columns:
        raise ValueError("La feuille Excel doit contenir une colonne 'Date'.")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    if "Instrument" in df.columns:
        if instrument is None:
            unique_instr = df["Instrument"].dropna().unique()
            if len(unique_instr) > 1:
                raise ValueError(f"Plusieurs instruments trouvés: {list(unique_instr)}. Passe --instrument.")
            if len(unique_instr) == 1:
                instrument = unique_instr[0]
        if instrument is not None:
            df = df[df["Instrument"] == instrument]
        df = df.drop(columns=["Instrument"], errors="ignore")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def parse_args(argv=None):
    p = argparse.ArgumentParser("Main pipeline minimal")
    p.add_argument("--bilan", required=True, help="Chemin vers le fichier bilan.xlsx")
    p.add_argument("--instrument", required=False, help="Ticker / instrument à filtrer (col Instrument dans l'Excel)")
    p.add_argument("--out", required=False, help="Dossier de sortie pour CSV et figure")
    p.add_argument("--plot-only-top", required=False, type=int, help="(optionnel) ne tracer que les N premières colonnes pour lisibilité")
    return p.parse_args(argv)


def plot_bilan_with_projection(
    bilan_hist: pd.DataFrame,
    bilan_proj: pd.DataFrame,
    out_dir: Optional[Path] = None,
    top_n: Optional[int] = None,
) -> None:
    """
    Trace les séries historiques et projetées.

    - bilan_hist : DataFrame indexé par date (historique)
    - bilan_proj : DataFrame indexé par date (projection)
    - out_dir : si fourni, dossier où sauvegarder la figure (PNG)
    - top_n : si fourni, ne tracer que les premières top_n colonnes pour lisibilité
    """
    # choisir colonnes à tracer
    cols = list(bilan_hist.columns)
    if top_n is not None and top_n > 0:
        cols = cols[:top_n]

    if not cols:
        print("Aucune colonne à tracer.")
        return

    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    # tracer historique
    for col in cols:
        # tracer la série historique (trait plein)
        if col in bilan_hist.columns:
            series_hist = pd.to_numeric(bilan_hist[col], errors="coerce")
            ax.plot(series_hist.index, series_hist.values, linestyle="-", label=f"{col} (hist)")

    # tracer projection (trait pointillé)
    for col in cols:
        if col in bilan_proj.columns:
            series_proj = pd.to_numeric(bilan_proj[col], errors="coerce")
            ax.plot(series_proj.index, series_proj.values, linestyle="--", label=f"{col} (proj)")

    # belle mise en forme
    ax.set_xlabel("Date")
    ax.set_ylabel("Valeur")
    ax.set_title("Bilan : historique (plein) vs projection (pointillé)")
    ax.legend(fontsize="small", ncol=2)
    plt.tight_layout()

    # sauvegarde ou affichage
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig_path = out_dir / "bilan_projection.png"
        plt.savefig(fig_path, dpi=200)
        print(f"Figure saved to {fig_path}")
        plt.close()
    else:
        plt.show()


def main(argv=None):
    args = parse_args(argv)

    # logger
    logger = logging.getLogger("main")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    if not logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info("Chargement du bilan")
    bilan_df = load_bilan_excel(args.bilan, instrument=args.instrument)
    logger.info(f"Bilan shape={bilan_df.shape}")

    # préparer macro_df test (3 périodes) — tu peux remplacer par ton API
    dates = pd.date_range("2025-12-31", periods=20, freq="YE")
    macro_df = pd.DataFrame({"gdp_growth": [0.02]*20, "inflation": [0.01]*20}, index=dates)

    if CompanyDataLoader is None:
        raise RuntimeError("CompanyDataLoader non importable. Vérifie le module data_loader_company.py")

    loader = CompanyDataLoader(id=args.instrument or "company", nace_code=None, bilan=bilan_df)

    logger.info("Projection du bilan (mode test)")
    loader.project_bilan(macro_df, model_paths={})
    bilan_proj = loader.get_bilan_proj()

    logger.info("Application des coûts simples si disponibles")
    if CarbonCost is None or EnergyCost is None:
        logger.warning("Modules de coût non trouvés — les DataFrames seront retournés sans colonnes de coût")
        bilan_hist_costed = bilan_df.copy()
        bilan_proj_costed = bilan_proj.copy()
    else:
        # pour test: appliquer mêmes facteurs à toutes les colonnes
        carbon = CarbonCost({col: 0.001 for col in bilan_df.columns})
        energy = EnergyCost({col: 0.05 for col in bilan_df.columns})
        bilan_hist_costed = energy.apply(carbon.apply(bilan_df))
        bilan_proj_costed = energy.apply(carbon.apply(bilan_proj))

    outputs = {
        "bilan_historique": bilan_df,
        "bilan_historique_costed": bilan_hist_costed,
        "bilan_proj": bilan_proj,
        "bilan_proj_costed": bilan_proj_costed,
    }

    if args.out:
        outp = Path(args.out)
        outp.mkdir(parents=True, exist_ok=True)
        for name, df in outputs.items():
            fn = outp / f"{name}.csv"
            df.to_csv(fn)
            logger.info(f"Saved {name} -> {fn}")

    # ------------------ PLOTTING ------------------
    try:
        top_n = args.plot_only_top if hasattr(args, "plot_only_top") else None
    except Exception:
        top_n = None

    # si l'utilisateur a demandé un dossier de sortie, on sauvegarde la figure dedans,
    # sinon on affiche la fenêtre interactive.
    out_dir = Path(args.out) if args.out else None
    # tracer les colonnes du bilan historique et leurs projections
    plot_bilan_with_projection(bilan_hist=bilan_df, bilan_proj=bilan_proj, out_dir=out_dir, top_n=top_n)

    logger.info("Done — pipeline minimal terminé")
    return outputs


if __name__ == "__main__":
    main()
