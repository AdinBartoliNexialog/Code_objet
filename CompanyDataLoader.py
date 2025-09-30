"""Data loader simplifié pour une entreprise.

Objectif
--------
Fournir une version claire et facile à lire du `CompanyDataLoader` adaptée au
format d'entrée que tu m'as donné (DataFrame indexé par Date, colonnes = postes
comptables). Cette version :
- garde uniquement les méthodes utiles (getters/setters minimalistes)
- `get_last(column)` renvoie la dernière valeur non-NaN
- `project_bilan(macro_df, model_paths=None)` : pour chaque colonne du bilan
  crée un `BilanVariable` et appelle sa méthode `projection(macro_df, last_val)`
  (implementation de projection simple : modèle constant si aucun pickle)
- sélection macro par variable via un dictionnaire `MACRO_SELECTIONS` simple

Remarques
---------
- La logique de chargement de vrais modèles pickle est laissée en dehors et
  commentée : pour les tests on utilise un modèle constant basé sur la dernière
  valeur historique.
- La classe est volontairement concise pour faciliter la lecture et la suite du
  travail (tu me demanderas d'ajouter des features ensuite si besoin).
"""
from __future__ import annotations

from typing import Any, Optional, Dict
import pandas as pd
import logging

# Importer la classe BilanVariable simplifiée (projection constante)
try:
    from projection import BilanVariable
except Exception:  # pragma: no cover - fallback minimal
    # fallback très simple : utiliser une classe locale légère
    class BilanVariable:  # type: ignore
        def __init__(self, name: str, model_path: Optional[str] = None) -> None:
            self.name = name
            self.model_path = model_path

        def projection(self, macro_df: pd.DataFrame, last_val: float = 0.0) -> pd.Series:
            # modèle constant fallback
            return pd.Series([last_val] * len(macro_df), index=macro_df.index, name=self.name)


# Mapping simple variable -> colonnes macro pertinentes
# Adapte ces règles selon ton jeu de variables macro.
MACRO_SELECTIONS: Dict[str, tuple] = {
    "Debt - Total": ("taux_interet", "inflation"),
    "Total Assets": ("gdp_growth",),
    "Net Income after Tax": ("gdp_growth", "inflation"),
    "Earnings before Interest & Taxes (EBIT)": ("gdp_growth",),
    # par défaut, si variable absente ici -> on utilisera toutes les colonnes macro
}

# logger simple
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class CompanyDataLoader:
    """Container simple pour les données comptables d'une entreprise.

    Parameters
    ----------
    id : Any
        Identifiant de l'entreprise (ex: ticker).
    nace_code : Optional[str]
        Code NACE/APE (optionnel).
    bilan : pd.DataFrame
        DataFrame historique indexé par Date, colonnes = postes comptables.
    """

    def __init__(self, id: Any, nace_code: Optional[str], bilan: pd.DataFrame) -> None:
        if not isinstance(bilan, pd.DataFrame):
            raise TypeError("bilan must be a pandas DataFrame")

        self._id = id
        self._nace_code = nace_code
        self._bilan = bilan.copy()
        self._bilan_proj: Optional[pd.DataFrame] = None
        # mapping variable->model path (optionnel)
        self.variable_model_paths: Dict[str, str] = {}

    # --- getters / setters minimalistes ---
    def get_id(self) -> Any:
        return self._id

    def get_bilan(self) -> pd.DataFrame:
        return self._bilan.copy()

    def get_bilan_proj(self) -> Optional[pd.DataFrame]:
        return None if self._bilan_proj is None else self._bilan_proj.copy()

    def set_variable_model_paths(self, paths: Dict[str, str]) -> None:
        self.variable_model_paths.update(paths)

    def get_last(self, column_name: str) -> Optional[float]:
        """Retourne la dernière valeur non-NaN de la colonne donnée ou None."""
        if column_name not in self._bilan.columns:
            raise KeyError(f"Colonne '{column_name}' introuvable dans le bilan")
        s = pd.to_numeric(self._bilan[column_name], errors="coerce")
        non_null = s.dropna()
        if non_null.empty:
            return None
        return float(non_null.iloc[-1])

    def _select_macro_columns(self, var_name: str, macro_df: pd.DataFrame) -> pd.DataFrame:
        """Retourne un DataFrame macro_df filtré selon `MACRO_SELECTIONS`.

        Si aucune règle définie pour `var_name`, on renvoie l'ensemble des
        colonnes de macro_df.
        """
        if not isinstance(macro_df, pd.DataFrame):
            raise ValueError("macro_df must be a pandas DataFrame")

        cols = MACRO_SELECTIONS.get(var_name)
        if cols is None:
            return macro_df.copy()
        # ne garder que les colonnes présentes
        keep = [c for c in cols if c in macro_df.columns]
        if not keep:
            # si aucune colonne pertinente trouvée, renvoyer DataFrame vide (mais conservant l'index)
            return macro_df.iloc[:, :0].copy()
        return macro_df[keep].copy()

    def project_bilan(self, macro_df: pd.DataFrame, model_paths: Optional[Dict[str, str]] = None) -> None:
        """Projette chaque colonne du bilan sur l'horizon `macro_df.index`.

        Comportement simplifié pour tests :
        - si model_paths fournit un chemin pour la variable et que le modèle
          peut être chargé (non testé ici), on l'utiliserait (code de chargement
          à ajouter plus tard)
        - sinon on utilise `BilanVariable.projection(macro_df, last_val)` avec
          `last_val` la dernière valeur historique (ou NaN si absente)

        Le résultat est stocké dans `self._bilan_proj`.
        """
        if not isinstance(macro_df, pd.DataFrame):
            raise ValueError("macro_df must be a pandas DataFrame")

        local_model_paths = dict(self.variable_model_paths)
        if model_paths:
            local_model_paths.update(model_paths)

        series_list = []

        for var in self._bilan.columns:
            last = self.get_last(var)
            model_path = local_model_paths.get(var)

            # préparer BilanVariable (projection.py fournit une version simple)
            bilan_var = BilanVariable(var, model_path=model_path)

            # sélectionner colonnes macro pertinentes
            macro_sel = self._select_macro_columns(var, macro_df)

            # Si model_path explicitement 'constant' ou absent -> fallback constant
            if not model_path or model_path == "constant":
                if last is None:
                    # pas d'historique -> série de NaN
                    s = pd.Series([pd.NA] * len(macro_df), index=macro_df.index, name=var)
                else:
                    s = bilan_var.projection(macro_sel, last_val=last)
                series_list.append(s)
                continue

            # Sinon, ici on pourrait charger un modèle pickle/joblib (code laissé commenté)
            try:
                # EXEMPLE (commenté) :
                # import joblib
                # model = joblib.load(model_path)
                # preds = model.predict(macro_sel)
                # s = pd.Series(preds, index=macro_sel.index, name=var)

                # pour l'instant : utiliser le fallback constant pour garder le test simple
                if last is None:
                    s = pd.Series([pd.NA] * len(macro_df), index=macro_df.index, name=var)
                else:
                    s = bilan_var.projection(macro_sel, last_val=last)

            except Exception as e:
                logger.warning("Erreur projection pour %s (model %s) : %s — fallback constant utilisé", var, model_path, e)
                if last is None:
                    s = pd.Series([pd.NA] * len(macro_df), index=macro_df.index, name=var)
                else:
                    s = pd.Series([last] * len(macro_df), index=macro_df.index, name=var)

            series_list.append(s)

        # concaténation
        if series_list:
            df_proj = pd.concat(series_list, axis=1)
        else:
            df_proj = pd.DataFrame(index=macro_df.index)

        # conserver l'ordre des colonnes historique
        df_proj = df_proj.reindex(columns=list(self._bilan.columns))
        self._bilan_proj = df_proj.copy()


