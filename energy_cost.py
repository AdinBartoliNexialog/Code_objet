import pandas as pd

class EnergyCost:
    def __init__(self, energy_factors: dict):
        """
        energy_factors : dict {colonne_bilan: facteur_energie}
        """
        self.energy_factors = energy_factors

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        for col, factor in self.energy_factors.items():
            if col in df_out.columns:
                df_out[f"{col}_energy_cost"] = df_out[col] * factor
        return df_out
