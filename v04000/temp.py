import gc
import lightgbm


class LGBM_Model():

    def __init__(
        self, params, train_data, valid_data, feature_cols, categorical_cols=None,
        train_weight=None, valid_weight=None, custom_eval=None, custom_obj=None
    ):
        super().__init__()

    def _convert_dataset(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def get_importance(self):
        pass

    def save_importance(self):
        pass
