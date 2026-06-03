import os
import yaml
import joblib
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

class Trainer:
    def __init__(self):
        self.config = self.load_config()
        self.model_name = self.config['model']['name']
        self.pipeline = self.create_pipeline()

    def load_config(self):
        with open('config.yml', 'r') as config_file:
            return yaml.safe_load(config_file)

    def create_pipeline(self):
        preprocessor = ColumnTransformer(transformers=[
            ('minmax', MinMaxScaler(), ['AnnualPremium']),
            ('standard', StandardScaler(), ['Age', 'RegionID']),
            ('onehot', OneHotEncoder(sparse_output=False, drop='first'), ['Gender', 'PastAccident'])
        ])

        params = {k: v for k, v in self.config['model']['params'].items() if v is not None}

        models = {
            'GradientBoostingClassifier': GradientBoostingClassifier,
            'RandomForestClassifier': RandomForestClassifier,
            'DecisionTreeClassifier': DecisionTreeClassifier,
        }
        model = models[self.model_name](**params)

        return Pipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(sampling_strategy=1.0)),
            ('model', model)
        ])

    def feature_target_separator(self, data):
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X, y

    def train_model(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)

    def save_model(self):
        store_path = self.config['model']['store_path']
        os.makedirs(store_path, exist_ok=True)
        joblib.dump(self.pipeline, f'{store_path}/model.pkl')
