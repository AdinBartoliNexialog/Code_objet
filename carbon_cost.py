import pandas as pd

class CarbonCost:
    def __init__(self, emission_factors: dict):
        """
        emission_factors : dict {colonne_bilan: facteur_emission}
        """
        self.emission_factors = emission_factors

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applique le coût carbone sur un bilan projeté ou non.
        Retourne une copie du DataFrame avec des colonnes de coût ajoutées.
        """
        df_out = df.copy()
        for col, factor in self.emission_factors.items():
            if col in df_out.columns:
                df_out[f"{col}_carbon_cost"] = df_out[col] * factor
        return df_out
