"""Data loader pour une classe "Entreprise" — version étendue.

Cette version implémente `project_bilan` :
- itère sur les colonnes du bilan historique
- récupère la dernière valeur connue (avec `get_last`)
- crée un objet `BilanVariable` pour chaque variable de bilan en utilisant un
  chemin de modèle (si fourni)
- sélectionne les variables macro pertinentes pour chaque variable de bilan
  via un `match` / `case` (switch)
- appelle `BilanVariable.projection(macro_df_selected)` et rassemble les
  `pd.Series` retournées dans `self._bilan_proj` (DataFrame)

Notes d'utilisation :
- `project_bilan` prend en argument `macro_df` (DataFrame des variables
  macroéconomiques pour la période de projection) et un dictionnaire optionnel
  `model_paths` mapping variable_bilan -> chemin du modèle (ex: {'chiffre_affaires': 'models/ca.joblib'}).
- Si aucun modèle n'est fourni pour une variable, la projection par défaut est
  une simple extrapolation : on prolonge la dernière valeur connue constante
  sur l'index de `macro_df`.

"""
from __future__ import annotations

from typing import Any, Optional, Dict
import pandas as pd

# tentative d'import de la classe BilanVariable définie dans l'autre fichier
try:
    from bilan_variable import BilanVariable
except Exception:  # pragma: no cover - import may échouer en tests isolés
    # fallback minimal: définir une classe stand-in qui lèvera une erreur si on
    # tente d'utiliser la projection via un modèle externe
    class BilanVariable:  # type: ignore
        def __init__(self, name: str, model_path: Optional[str] = None) -> None:
            self._name = name
            self._model_path = model_path

        def projection(self, macro_df: pd.DataFrame) -> pd.Series:
            raise RuntimeError("BilanVariable non disponible — import failed")


class CompanyDataLoader:
    """Container léger pour les données d'une entreprise.

    Attributs
    ---------
    id : Any
        Identifiant unique de l'entreprise.
    nace_code : Optional[str]
        Code NACE/APE.
    bilan : pd.DataFrame
        DataFrame contenant les variables comptables (colonnes = variables,
        index = dates / périodes si disponible).
    bilan_proj : Optional[pd.DataFrame]
        DataFrame de projection du bilan (initialement None ou vide).
    variable_model_paths : Dict[str, str]
        Map des noms de variables bilan vers des chemins de modèles (optionnel).
    """

    def __init__(
        self,
        id: Any,
        nace_code: Optional[str],
        bilan: pd.DataFrame,
        bilan_proj: Optional[pd.DataFrame] = None,
        variable_model_paths: Optional[Dict[str, str]] = None,
    ) -> None:
        # validations simples
        if not isinstance(bilan, pd.DataFrame):
            raise TypeError("bilan must be a pandas DataFrame")
        if bilan_proj is not None and not isinstance(bilan_proj, pd.DataFrame):
            raise TypeError("bilan_proj must be a pandas DataFrame or None")

        self._id = id
        self._nace_code = nace_code
        # make copies to avoid side-effects from outside
        self._bilan = bilan.copy()
        self._bilan_proj = None if bilan_proj is None else bilan_proj.copy()
        self.variable_model_paths: Dict[str, str] = {} if variable_model_paths is None else dict(variable_model_paths)

    # --- getters ---
    def get_id(self) -> Any:
        return self._id

    def get_nace(self) -> Optional[str]:
        return self._nace_code

    def get_bilan(self) -> pd.DataFrame:
        return self._bilan.copy()

    def get_bilan_proj(self) -> Optional[pd.DataFrame]:
        return None if self._bilan_proj is None else self._bilan_proj.copy()

    # --- setter / update utiles ---
    def set_bilan(self, new_bilan: pd.DataFrame) -> None:
        if not isinstance(new_bilan, pd.DataFrame):
            raise TypeError("new_bilan must be a pandas DataFrame")
        self._bilan = new_bilan.copy()

    def set_bilan_proj(self, new_bilan_proj: pd.DataFrame) -> None:
        if not isinstance(new_bilan_proj, pd.DataFrame):
            raise TypeError("new_bilan_proj must be a pandas DataFrame")
        self._bilan_proj = new_bilan_proj.copy()

    def set_variable_model_paths(self, paths: Dict[str, str]) -> None:
        """Fournit ou met à jour la map variable->model path."""
        self.variable_model_paths.update(paths)

    # --- utilitaires ---
    def get_last(self, column_name: str) -> Any:
        if column_name not in self._bilan.columns:
            raise KeyError(f"Colonne '{column_name}' introuvable dans le bilan")

        series = self._bilan[column_name]
        non_null = series.dropna()
        if non_null.empty:
            return None
        return non_null.iloc[-1]

    def _select_macro_for_variable(self, var_name: str, macro_df: pd.DataFrame) -> pd.DataFrame:
        """Sélectionne (filtre) les colonnes de macro_df pertinentes pour var_name.

        Implémentation via un switch (match/case). Renvoie un DataFrame contenant
        uniquement les colonnes explicatives nécessaires.

        Règles par défaut : retourne macro_df non filtré.
        """
        # Exemple de règles ; adapte selon tes besoins.
        # On suppose que macro_df contient des colonnes comme 'gdp_growth', 'inflation', 'taux_interet', etc.
        try:
            match var_name:
                case "chiffre_affaires":
                    return macro_df[[c for c in macro_df.columns if c in ("gdp_growth", "inflation")]]
                case "resultat_net":
                    return macro_df[[c for c in macro_df.columns if c in ("gdp_growth", "taux_interet")]]
                case "actif_total":
                    return macro_df[[c for c in macro_df.columns if c in ("gdp_growth",)]]
                case "dettes":
                    return macro_df[[c for c in macro_df.columns if c in ("taux_interet", "inflation")]]
                case _:
                    # par défaut : on renvoie toutes les variables macro
                    return macro_df.copy()
        except Exception:
            # si la sélection échoue (p.ex colonnes manquantes), retourner au moins un DataFrame vide
            return macro_df.iloc[:, :0].copy()

    def project_bilan(self, macro_df: pd.DataFrame, model_paths: Optional[Dict[str, str]] = None) -> None:
        """Remplit `self._bilan_proj` en projetant chaque colonne du bilan.

        Parameters
        ----------
        macro_df : pd.DataFrame
            DataFrame des variables macroéconomiques pour la période de projection
            (index = périodes de projection).
        model_paths : Optional[Dict[str,str]]
            Dictionnaire optionnel mapping variable_bilan -> chemin du modèle.
            Si fourni, il surcharge `self.variable_model_paths` pour l'appel courant.

        Comportement
        -----------
        - Pour chaque colonne `var` de `self._bilan` :
            - récupère la dernière valeur connue via `get_last(var)`
            - instancie `BilanVariable(var, model_path)` où model_path est cherché dans
              `model_paths` puis `self.variable_model_paths`
            - sélectionne les colonnes macro pertinentes via `_select_macro_for_variable`
            - si un modèle est disponible, appelle `BilanVariable.projection(selected_macro_df)`
              et récupère une `pd.Series` qu'on ajoute au DataFrame de projection
            - sinon, on effectue une projection par défaut : on prolonge la dernière
              valeur connue constante sur l'index de `macro_df`.
        - Met à jour `self._bilan_proj` avec le DataFrame construit.
        """
        if not isinstance(macro_df, pd.DataFrame):
            raise ValueError("macro_df must be a pandas DataFrame")

        # Surcharge locale des model paths si fournie
        local_model_paths = dict(self.variable_model_paths)
        if model_paths is not None:
            local_model_paths.update(model_paths)

        projected_series = []

        for var in self._bilan.columns:
            last_val = self.get_last(var)

            # déterminer le model path
            model_path = local_model_paths.get(var)

            # préparer l'objet BilanVariable
            bilan_var = BilanVariable(var, model_path=model_path)

            # sélectionner les variables macro pertinentes
            selected_macro = self._select_macro_for_variable(var, macro_df)

            # si le modèle est absent ou model_path is None -> fallback = constante
            if model_path is None:
                # extrapolation constante à partir de last_val
                if last_val is None:
                    # pas de donnée historique -> série de NaN
                    s = pd.Series([pd.NA] * len(macro_df), index=macro_df.index, name=var)
                else:
                    s = pd.Series([last_val] * len(macro_df), index=macro_df.index, name=var)
                projected_series.append(s)
                continue

            # sinon, essayer d'appeler la projection du BilanVariable
            try:
                s = bilan_var.projection(selected_macro)
                if not isinstance(s, pd.Series):
                    # tenter de convertir
                    s = pd.Series(s, index=selected_macro.index, name=var)
            except Exception as e:
                # En cas d'erreur lors de la prédiction, log léger (print) et fallback
                print(f"Projection échouée pour {var} (model: {model_path}) — fallback constant. Erreur: {e}")
                if last_val is None:
                    s = pd.Series([pd.NA] * len(macro_df), index=macro_df.index, name=var)
                else:
                    s = pd.Series([last_val] * len(macro_df), index=macro_df.index, name=var)

            projected_series.append(s)

        # concaténation en DataFrame final
        if projected_series:
            df_proj = pd.concat(projected_series, axis=1)
        else:
            # aucun var à projeter -> DataFrame vide indexé par macro_df.index
            df_proj = pd.DataFrame(index=macro_df.index)

        # s'assurer que l'ordre des colonnes correspond à l'historique
        df_proj = df_proj.reindex(columns=list(self._bilan.columns))

        self._bilan_proj = df_proj.copy()


# ------------------------- Exemple minimal -------------------------
if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "chiffre_affaires": [100.0, 120.0, None, 150.0],
            "resultat_net": [10.0, 12.0, 11.0, None],
        },
        index=pd.to_datetime(["2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31"]),
    )

    macro = pd.DataFrame(
        {"gdp_growth": [0.02, 0.025, 0.03], "inflation": [0.01, 0.015, 0.02], "taux_interet": [0.01, 0.012, 0.013]},
        index=pd.to_datetime(["2024-12-31", "2025-12-31", "2026-12-31"]),
    )

    loader = CompanyDataLoader(id=1, nace_code="A01", bilan=df)
    # on définit un model_path fictif pour 'chiffre_affaires' (ici None pour demo)
    loader.set_variable_model_paths({"chiffre_affaires": None})

    loader.project_bilan(macro)
    print("Bilan projeté:", loader.get_bilan_proj())
