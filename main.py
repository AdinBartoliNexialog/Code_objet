"""Pipeline simple et extensible (version adaptée aux coûts simples).

Objectif
--------
Adapter le pipeline pour utiliser des modules de coûts **simples** qui
implémentent la méthode `apply(df: pd.DataFrame) -> pd.DataFrame` (comme
`CarbonCost` et `EnergyCost` fournis précédemment).

Principes
- Le pipeline reste minimaliste et lisible.
- Les modules de coûts sont des objets simples : `module.apply(df)`.
- Le pipeline applique les coûts **sur le bilan historique** ET **sur le bilan projeté**
  et retourne les deux DataFrames (non-stressé / projeté) avec colonnes de coût ajoutées.

Contenu
-------
- `SimplePipeline` : charge les CSV, initialise `CompanyDataLoader`, appelle
  `project_bilan`, applique les modules de coût sur le bilan historique et
  le bilan projeté, et retourne un dict contenant les DataFrames.

Usage
-----
- fournir des instances de modules de coûts (ex: `CarbonCost`, `EnergyCost`) qui
  implémentent `apply(df)`.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List
from pathlib import Path
import pandas as pd
import logging
import sys

# Importer le CompanyDataLoader
try:
    from data_loader_company import CompanyDataLoader
except Exception:  # pragma: no cover
    CompanyDataLoader = None  # type: ignore


class SimplePipeline:
    """Pipeline minimal adapté aux modules de coût simples.

    Parameters
    ----------
    bilan_path, macro_path : Path | str
        Chemins vers les CSV.
    model_paths : dict
        mapping variable->model_path
    cost_modules : list
        liste d'objets simples implémentant `apply(df: pd.DataFrame) -> pd.DataFrame`
    config : dict
        paramètres supplémentaires pour la pipeline / modules
    """

    def __init__(self, bilan_path: str | Path, macro_path: str | Path, model_paths: Dict[str, str] | None = None, cost_modules: List[object] | None = None, config: Dict[str, Any] | None = None):
        self.bilan_path = Path(bilan_path)
        self.macro_path = Path(macro_path)
        self.model_paths = {} if model_paths is None else dict(model_paths)
        self.cost_modules = [] if cost_modules is None else list(cost_modules)
        self.config = {} if config is None else dict(config)

        # logger simple
        self.logger = logging.getLogger(self.__class__.__name__)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _load_csv(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df

    def _apply_costs_chain(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applique séquentiellement les modules de coût sur `df`.

        Chaque module doit implémenter `apply(df: pd.DataFrame) -> pd.DataFrame`.
        La fonction retourne la copie finale résultante.
        """
        df_curr = df.copy()
        for mod in self.cost_modules:
            # api simple : apply(df) -> df
            try:
                df_curr = mod.apply(df_curr)
            except Exception as e:
                # log léger et continuer (on ne veut pas casser le pipeline pour un module)
                self.logger.warning("Module %s failed: %s. Skipping this module.", getattr(mod, "__class__", type(mod)), e)
        return df_curr

    def run(self) -> Dict[str, pd.DataFrame]:
        """Exécute le pipeline et retourne un dict avec les DataFrames:

        {
            'bilan_historique': df_historique,
            'bilan_historique_costed': df_historique_costed,
            'bilan_proj': df_proj,
            'bilan_proj_costed': df_proj_costed,
        }
        """
        self.logger.info("Start SimplePipeline (costs-simple)")

        bilan_df = self._load_csv(self.bilan_path)
        macro_df = self._load_csv(self.macro_path)

        if CompanyDataLoader is None:
            raise RuntimeError("CompanyDataLoader not importable. Vérifie les chemins.")

        loader = CompanyDataLoader(id="company_main", nace_code=None, bilan=bilan_df)
        loader.set_variable_model_paths(self.model_paths)

        # projection
        self.logger.info("Projection du bilan...")
        loader.project_bilan(macro_df, model_paths=self.model_paths)
        df_proj = loader.get_bilan_proj()

        # appliquer les modules de coût sur le bilan historique
        if self.cost_modules:
            self.logger.info("Application des modules de coût sur le bilan historique...")
            df_hist_costed = self._apply_costs_chain(bilan_df)

            self.logger.info("Application des modules de coût sur le bilan projeté...")
            df_proj_costed = self._apply_costs_chain(df_proj)
        else:
            df_hist_costed = bilan_df.copy()
            df_proj_costed = df_proj.copy()

        self.logger.info("Pipeline terminé")
        return {
            "bilan_historique": bilan_df,
            "bilan_historique_costed": df_hist_costed,
            "bilan_proj": df_proj,
            "bilan_proj_costed": df_proj_costed,
        }


# ---------------- Exemple d'utilisation ----------------
if __name__ == "__main__":
    # chemins fictifs
    bilan_path = "data/bilan.csv"
    macro_path = "data/macro.csv"

    # exemples : on suppose que tu as les classes CarbonCost et EnergyCost définies
    try:
        from costs.carbon_cost import CarbonCost  # type: ignore
        from costs.energy_cost import EnergyCost  # type: ignore

        carbon = CarbonCost({"chiffre_affaires": 0.001, "resultat_net": 0.002})
        energy = EnergyCost({"consommation_energie": 0.05})
        cost_modules = [carbon, energy]
    except Exception:
        cost_modules = []

    p = SimplePipeline(bilan_path=bilan_path, macro_path=macro_path, model_paths={}, cost_modules=cost_modules, config={})

    try:
        outputs = p.run()
        print("Bilan projeté costed (extrait):")
        print(outputs["bilan_proj_costed"].head())
    except Exception as e:
        print("Erreur lors du run:", e)
