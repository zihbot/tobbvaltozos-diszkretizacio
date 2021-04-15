import pandas as pd

class MultivariateDiscretizer:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data