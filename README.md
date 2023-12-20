# Automobile Price Prediction

This project focuses on predicting the price of automobiles based on various features using Multiple Correspondence Analysis (MCA) and other techniques. The analysis indicates that the first three components of MCA play a crucial role in the prediction, followed by several numeric columns.

## Table of Contents

- [Installation](#installation)
- [File Structure](#file-structure)
- [Usage](#usage)
- [EDA and Reports](#eda-and-reports)
- [Results](#results)


## Installation

1. Clone the repository:

`git clone https://github.com/GabrielBaglietto/MCAcars`

2. Navigate to the project directory:

`cd MCAcars`

3. Install the required packages:

`pip install -r requirements.txt`


## File Structure

- `data/`: Directory containing the dataset for the analysis.
- `model/`: Directory where the trained model is saved.
- `src/`: Contains the source code for training and predicting.
- `train.py`: Script for training the model.
- `model.py`: Contains the model definition.
- `predict.py`: Script for making predictions using the trained model.
- `automobile_price_analysis.ipynb`: Jupyter notebook with EDA and predictive modeling results.

## Usage

To train the model:

`python src/train.py`

To make predictions:

`python src/predict.py`


## EDA

- Detailed EDA can be viewed in the Jupyter notebook `automobile_price_analysis.ipynb`.


## Results

The analysis has shown that three MCA components are essential in predicting automobile prices. Following these components, the significant numeric columns influencing the predictions are:
- curb-weight
- city-mpg
- highway-mpg
- engine-size
- peak-rpm
- horsepower
- width

The predictive model achieved an R^2 score of over 0.9 in the test set.

