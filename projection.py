"""Classe pour représenter une variable de bilan et sa méthode de projection.

Version simplifiée pour tests : utilise un modèle constant basé sur la dernière valeur connue.
Les lignes pour charger un vrai modèle pickle sont laissées en commentaire.
"""
from __future__ import annotations

from typing import Optional, Any
import pandas as pd

class ConstantModel:
    """Modèle test : renvoie toujours la dernière valeur connue."""
    def __init__(self, last_val):
        self.last_val = last_val

    def predict(self, X):
        print(len(X))
        print(X)
        n = len(X)
        return [self.last_val] * n

class BilanVariable:
    """Représente une variable de bilan et son modèle de projection."""

    def __init__(self, name: str, model_path: Optional[str] = None) -> None:
        self._name = str(name)
        self._model_path = None if model_path is None else str(model_path)

    def get_name(self) -> str:
        return self._name

    def get_model_path(self) -> Optional[str]:
        return self._model_path

    def set_model_path(self, path: str) -> None:
        self._model_path = str(path)

    def projection(self, macro_df: pd.DataFrame, last_val: float = 100.0) -> pd.Series:
        """
        Projection simplifiée : modèle constant basé sur last_val.

        Parameters
        ----------
        macro_df : pd.DataFrame
            Variables macroéconomiques (index = périodes).
        last_val : float
            Valeur constante à projeter (simule la dernière valeur connue).

        Returns
        -------
        pd.Series
            Série projetée indexée par macro_df.index et nommée self._name.
        """
        if not isinstance(macro_df, pd.DataFrame):
            raise ValueError("macro_df must be a pandas DataFrame")

        # Version simple pour test : modèle constant
        const_model = ConstantModel(last_val)
        preds = const_model.predict(macro_df)
        return pd.Series(preds, index=macro_df.index, name=self._name)

        # Version avec pickle/joblib (la laisser en commentaire pour l'instant)
        # if self._model_path is None:
        #     raise RuntimeError("Aucun model_path défini")
        # model = self._load_model(self._model_path)
        # if hasattr(model, "predict"):
        #     preds = model.predict(macro_df)
        # elif callable(model):
        #     preds = model(macro_df)
        # else:
        #     raise RuntimeError("Le modèle chargé n'est ni callable ni n'implémente .predict()")
        # return pd.Series(preds, index=macro_df.index, name=self._name)

