import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from model import objective, build_preprocessor, MCATransformer

# Load the trained model from the .pkl file
def load_model(model_path):
    """Load the model from a .pkl file."""
    with open(model_path, 'rb') as file:
        return pickle.load(file)

# Load the model
model_path = "model/trained_model.pkl"
model = load_model(model_path)

# Set display options for pandas
pd.set_option('display.max_columns', None)

# Load the dataset
file = 'data/Automobile_data.csv'
df = pd.read_csv(file)

# Convert columns with potential non-numeric values to numeric, setting errors='coerce' to handle exceptions
columns_with_nans = ['normalized-losses','price','bore','stroke','horsepower','peak-rpm']
for col in columns_with_nans:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove rows with missing 'price' values
filtered_df = df.dropna(subset=['price'])

# Select specific columns from the dataframe
selected_cols = ['make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 
                 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
                 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 
                 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

filtered_df = filtered_df[selected_cols]

# Separate the independent variables and the target variable
X = filtered_df.drop(columns=['price'])
y = filtered_df['price']

# Perform a train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)  # Using 10% for testing

# Make predictions using the loaded model
predictions = model.predict(X_test)
print("Predictions:", predictions)
