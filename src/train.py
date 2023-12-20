import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from prince import MCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import optuna
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from model import objective, build_preprocessor, MCATransformer

pd.set_option('display.max_columns', None)

file = 'data/Automobile_data.csv'
df = pd.read_csv(file)

columns_with_nans = ['normalized-losses','price','bore','stroke','horsepower','peak-rpm']
for col in columns_with_nans:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove rows with missing 'price' values
filtered_df = df.dropna(subset=['price'])

# Select specific columns from the dataframe
filtered_df = filtered_df[['make', 'fuel-type', 'aspiration',
       'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',
       'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
       'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke',
       'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg',
       'highway-mpg', 'price']]

# Separate the independent variables and the target variable
X = filtered_df.drop(columns=['price'])
y = filtered_df['price']

# Perform a train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)  # Using 10% for testing




# ##########################################################################################################


pruner = optuna.pruners.MedianPruner(
    n_startup_trials=20, 
    n_warmup_steps=0, 
    interval_steps=1
)

study = optuna.create_study(direction="maximize", pruner=pruner)  


TIME_LIMIT = 60  # 1 minute for demonstration purposes

for _ in range(5):  # NUM_ITERATIONS is the number of times you wish to restart the optimization
    study.optimize(lambda trial: objective(X_train, y_train, trial), n_trials=None, timeout=TIME_LIMIT)

#     study.optimize(objective(X_train,y_train), n_trials=None, timeout=TIME_LIMIT)






print("Best hyperparameters found by the study:", study.best_params)
print("Best R^2 value achieved with the above hyperparameters:", study.best_value)

best_params = study.best_params
# Build the pipeline with the best hyperparameters

preprocessor = build_preprocessor(X_train)

pipeline_best = Pipeline([
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),
    ('f_selector', SelectKBest(k=best_params["k"])),
    ('regressor', SVR(
        C=best_params["C"],
        kernel=best_params["kernel"],
        degree=best_params.get("degree", 3),  # Use .get() to avoid errors if "degree" is not in the parameters
        gamma=best_params["gamma"],
        coef0=best_params["coef0"]
    ))
])

pipeline_best.fit(X_train, y_train)

# Predictions on train and test sets
y_train_predicted = pipeline_best.predict(X_train)
y_test_predicted = pipeline_best.predict(X_test)

# Calculating metrics
r2_train_score = r2_score(y_train, y_train_predicted)
r2_test_score = r2_score(y_test, y_test_predicted)

mae_train_score = mean_absolute_error(y_train, y_train_predicted)
mae_test_score = mean_absolute_error(y_test, y_test_predicted)

mse_train_score = mean_squared_error(y_train, y_train_predicted)
mse_test_score = mean_squared_error(y_test, y_test_predicted)

rmse_train_score = np.sqrt(mse_train_score)
rmse_test_score = np.sqrt(mse_test_score)

# Display metrics
print("\n=== Metrics for Training Data ===")
print(f"R2 Score: {r2_train_score}")
print(f"Mean Absolute Error (MAE): {mae_train_score}")
print(f"Mean Squared Error (MSE): {mse_train_score}")
print(f"Root Mean Squared Error (RMSE): {rmse_train_score}")

print("\n=== Metrics for Testing Data ===")
print(f"R2 Score: {r2_test_score}")
print(f"Mean Absolute Error (MAE): {mae_test_score}")
print(f"Mean Squared Error (MSE): {mse_test_score}")
print(f"Root Mean Squared Error (RMSE): {rmse_test_score}\n")



# Save the model to a file
with open('model/trained_model.pkl', 'wb') as file:
    pickle.dump(pipeline_best, file)


