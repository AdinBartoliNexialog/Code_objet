"""Classe pour représenter une variable de bilan et sa méthode de projection.

La classe `BilanVariable` contient :
- un nom (string) de la variable de bilan (ex: 'chiffre_affaires')
- un chemin vers le modèle de projection (model_path) qui sera utilisé pour
  prédire la série projetée à partir d'un DataFrame de variables macro.

La méthode `projection(macro_df)` prend en entrée un `pandas.DataFrame` contenant
les variables macroéconomiques pertinentes (chaque ligne = période) et doit
retourner une `pandas.Series` correspondant aux valeurs projetées de la
variable de bilan, indexée comme `macro_df`.

Comportement implémenté:
- Si `model_path` est défini, la méthode tente de charger le modèle via
  `joblib.load` puis `pickle.load` si nécessaire. Le modèle chargé doit
  implémenter `predict(X)` ou être callable.
- Si la prédiction fonctionne, la méthode renvoie une `pd.Series` nommée par
  `self.name` et indexée par `macro_df.index`.
- En cas d'erreur (modèle introuvable, interface inattendue, etc.), une
  exception informative est levée.

Remarque: on laisse la logique de construction/entraînement du modèle en dehors
de cette classe — elle doit juste servir d'adaptateur pour appliquer un modèle
existants aux variables macro fournies.
"""
from __future__ import annotations

from typing import Optional, Any
import pandas as pd
import os
import pickle

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover - joblib n'est pas toujours présent
    joblib = None  # type: ignore


class BilanVariable:
    """Représente une variable de bilan et son modèle de projection.

    Parameters
    ----------
    name : str
        Nom de la variable de bilan (ex: 'chiffre_affaires').
    model_path : Optional[str]
        Chemin vers un fichier contenant le modèle de projection. Le modèle doit
        soit être un objet scikit-learn-like (avec .predict), soit un callable
        acceptant un DataFrame/array et renvoyant des prédictions.
    """

    def __init__(self, name: str, model_path: Optional[str] = None) -> None:
        self._name = str(name)
        self._model_path = None if model_path is None else str(model_path)

    # --- getters/setters ---
    def get_name(self) -> str:
        return self._name

    def get_model_path(self) -> Optional[str]:
        return self._model_path

    def set_model_path(self, path: str) -> None:
        self._model_path = str(path)

    # --- méthode principale ---
    def projection(self, macro_df: pd.DataFrame) -> pd.Series:
        """Projette la variable de bilan à partir d'un DataFrame macro.

        Parameters
        ----------
        macro_df : pd.DataFrame
            DataFrame des variables macroéconomiques (index = périodes, colonnes
            = variables explicatives). Chaque ligne correspond à une période de
            projection.

        Returns
        -------
        pd.Series
            Série des valeurs projetées de la variable de bilan, indexée par
            `macro_df.index` et nommée `self._name`.

        Raises
        ------
        ValueError, FileNotFoundError, RuntimeError
            En cas de problème (entrée invalide, modèle absent, modèle sans
            méthode predict/callable, etc.).
        """
        if not isinstance(macro_df, pd.DataFrame):
            raise ValueError("macro_df must be a pandas DataFrame")

        if self._model_path is None:
            raise RuntimeError(
                "Aucun model_path défini pour cette variable. Appelez set_model_path(path) "
                "ou fournissez un model_path à l'initialisation."
            )

        if not os.path.exists(self._model_path):
            raise FileNotFoundError(f"Model file not found: {self._model_path}")

        model = self._load_model(self._model_path)

        # Supporter scikit-like .predict ou callable
        if hasattr(model, "predict"):
            try:
                preds = model.predict(macro_df)
            except Exception as e:
                raise RuntimeError(f"Erreur lors de l'appel model.predict: {e}")
        elif callable(model):
            try:
                preds = model(macro_df)
            except Exception as e:
                raise RuntimeError(f"Erreur lors de l'appel du callable modèle: {e}")
        else:
            raise RuntimeError("Le modèle chargé n'est ni callable ni n'implémente .predict()")

        # Convertir en pd.Series et s'assurer que la longueur correspond
        preds_series = pd.Series(preds, index=macro_df.index, name=self._name)
        return preds_series

    def _load_model(self, path: str) -> Any:
        """Tente de charger un modèle depuis le disque.

        Essaie joblib.load puis pickle.load. Si aucune méthode ne fonctionne,
        lève une exception informative.
        """
        # 1) joblib
        if joblib is not None:
            try:
                return joblib.load(path)
            except Exception:
                # on continue vers pickle
                pass

        # 2) pickle
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Impossible de charger le modèle depuis {path}: {e}")


# ------------------ Exemple d'utilisation ------------------
if __name__ == "__main__":
    # Exemple minimal (sans modèle réel) :
    df_macro = pd.DataFrame(
        {"gdp_growth": [0.02, 0.025, 0.03], "inflation": [0.01, 0.015, 0.02]},
        index=pd.to_datetime(["2024-12-31", "2025-12-31", "2026-12-31"]),
    )

    var = BilanVariable("chiffre_affaires", model_path=None)
    try:
        s = var.projection(df_macro)  # va lever RuntimeError car model_path est None
    except Exception as e:
        print("Exemple:", e)

    # Si tu as un modèle enregistré (ex: 'models/ca_model.joblib') :
    # var.set_model_path('models/ca_model.joblib')
    # preds = var.projection(df_macro)
    # print(preds)
