import optuna
import pickle
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from prince import MCA

class MCATransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=10000):
        self.n_components = n_components
        self.mca = MCA(n_components=self.n_components)
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
        
    def fit(self, X, y=None):
        X_encoded = self.ohe.fit_transform(X)
        self.mca.fit(X_encoded)
        self.n_components = X_encoded.shape[1]
        return self
    
    def transform(self, X, y=None):
        X_encoded = self.ohe.transform(X)
        return self.mca.transform(X_encoded)

def build_preprocessor(X):
    # We have a list of categorical and numerical column names
    sorted_cols = ['engine-location', 'aspiration', 'num-of-cylinders', 'drive-wheels', 'fuel-type',
                   'body-style', 'engine-type', 'make', 'fuel-system', 'num-of-doors']
    all_columns = set(X.columns)
    categorical_features = sorted_cols
    numerical_features = list(all_columns - set(categorical_features))

    # Define imputers for categorical and numerical features
    cat_imputer = SimpleImputer(strategy='constant', fill_value='empty')
    num_imputer = SimpleImputer(strategy='median')

    # Define the column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', Pipeline([
                ('imputer', cat_imputer),
                ('mca', MCATransformer())
            ]), categorical_features),
            ('num', Pipeline([
                ('imputer', num_imputer),
                ('passthrough', 'passthrough')
            ]), numerical_features)
        ]
    )
    return preprocessor

def objective(X, y, trial):
    preprocessor = build_preprocessor(X)

    # Hyperparameters for SVR
    C = trial.suggest_float("C", 1e-10, 9e6, log=True)
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])
    if kernel == "poly":
        degree = trial.suggest_int("degree", 1, 5)
    else:
        degree = 3
    gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
    coef0 = trial.suggest_float("coef0", -1.0, 1.0)
    
    # Hyperparameters for SelectKBest
    k = trial.suggest_int("k", 1, X.shape[1])  
    
    print(f"Trial {trial.number} - Parameters: {trial.params}")
    
    # Build the pipeline with the suggested hyperparameters
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),
        ('f_selector', SelectKBest(k=k)),
        ('classifier', SVR(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0))
    ])
    
    # Evaluate the pipeline using k-fold cross-validation and return the average accuracy
    return cross_val_score(pipeline, X, y, n_jobs=-1, cv=5).mean()


