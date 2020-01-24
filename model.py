# Python Modules
from datetime import datetime
import typing
# 3rd Party Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics
from scipy.optimize import minimize
from scipy import stats
import seaborn as sn


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """ Filter malformed data. 

    Args: 
        df: Art market data to filter.

    Returns: 
        Filtered art market data.
    """
    # Drop records that are missing estimate_low and estimate_high.
    low_estimate_filter: pd.Series = (df.estimate_low > 0).rename('low_estimate_filter')
    high_estimate_filter: pd.Series = (df.estimate_high > 0).rename('high_estimate_filter')
    
    return df.loc[low_estimate_filter & high_estimate_filter].copy()


def preprocess_inputs(df: pd.DataFrame, zscore: bool = False) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
    """ Preprocess a dataframe into values useful for modeling purposes.

    Args:
        df: Art market data to preprocess.
        zscore: Whether to standardize to z-score.

    Returns:
        inputs: Preprocessed inputs for modeling.
        outputs: True values.
    """

    # Create dummy variables for auction_house variable.
    houses: pd.DataFrame = pd.get_dummies(df.auction_house)

    # Use auction_date to create categorical year columns to capture potential time-dependent trends.
    years: pd.DataFrame = pd.get_dummies(pd.to_datetime(df.auction_date).dt.year)

    # Convert estimates and hammer price to log10.
    log10_estimate_high: pd.Series = pd.Series(np.log10(df.estimate_high), index=df.index, name='log10_estimate_high')
    log10_estimate_low: pd.Series = pd.Series(np.log10(df.estimate_low), index=df.index, name='log10_estimate_low')

    # Unsold lots have a hammer price of -1. Take the log of sold lots, set others to nan.
    log10_hammer_price: pd.Series = pd.Series(index=df.index, name='log10_hammer_price')
    log10_hammer_price.loc[df.hammer_price > 0] = np.log10(df.hammer_price.loc[df.hammer_price > 0])

    # Column of ones for offset.
    ones: pd.Series = pd.Series(1, index=df.index, name='ones', dtype=int)
  
    # Flag for normalizing inputs on the same scale making for easier model interpretability.
    inputs_to_normalize: pd.DataFrame = pd.concat([log10_estimate_low, log10_estimate_high], axis=1)
    if zscore:
        inputs_to_normalize = (inputs_to_normalize - inputs_to_normalize.mean()) / inputs_to_normalize.std()

    inputs: pd.DataFrame = pd.concat([houses, years, ones, inputs_to_normalize], axis=1)
    outputs: pd.DataFrame = pd.concat([log10_hammer_price, log10_estimate_low], axis=1)

    return inputs, outputs


def train_model(inputs: pd.DataFrame, outputs: pd.DataFrame) -> pd.DataFrame:
    """ Train model coefficients.

    Args:
        inputs: Art market data to train over.
        outputs: True values.

    Results:
        Trained model.
    """
    # Compute the model. Here we use a simple linear model with a custom loss function.
    def custom_loss_func(beta: np.ndarray, X: np.ndarray, outputs: pd.DataFrame) -> np.ndarray:
        """ Custom loss: If hammer price == -1, loss = (yhat - estimate_low)**2 if yhat > estimte_low, else 0. """
        yhat: np.ndarray = np.dot(X, beta).transpose()

        sold_residual: np.ndarray = yhat - outputs.log10_hammer_price
        sold_loss: np.ndarray = outputs.log10_hammer_price.notna() * np.power(sold_residual, 2) 

        # Don't penalize underprediction when unsold.
        unsold_residual: np.ndarray = np.maximum(yhat - outputs.log10_estimate_low, 0)
        unsold_loss: np.ndarray = outputs.log10_hammer_price.isna() * np.power(unsold_residual, 2)

        return sold_loss.sum() + unsold_loss.sum()

    beta_init: np.ndarray = np.array([0] * inputs.shape[1])
    model = minimize(custom_loss_func, beta_init, args=(inputs.to_numpy(), outputs), options={'maxiter': 500}, method='BFGS')

    return model


def evaluate_model(inputs: pd.DataFrame, outputs: pd.DataFrame, model: pd.DataFrame) -> pd.DataFrame:
    """ Evaluate metrics about the model.

    Args:
        inputs: Art market data to predict.
        outputs: True values.
    
    Side effects:
        Generates `./report.png`
    """

    predicted_price: pd.Series = pd.Series(np.dot(inputs.values, model.x), index=inputs.index, name='predicted_price')

    # Create the estimated prices for each.
    sold_lots: pd.Series = outputs.log10_hammer_price.notna()

    sold_residuals: pd.Series = (predicted_price.loc[sold_lots] - outputs.loc[sold_lots, 'log10_hammer_price']).rename('sold_residuals')

    # Calculate the probability that a prediction will be above the low estimate.
    prob_lot_sells = stats.norm.cdf(predicted_price - outputs.log10_estimate_low, scale=sold_residuals.std())

    # -- Plot some useful metrics --

    # Log Model errors.
    fig, axes = plt.subplots(nrows=2, ncols=3)
    fig.set_size_inches(20, 12)

    axes[0,0].hist(sold_residuals, bins=50)
    axes[0,0].set_title('Model Residuals')

    # Show the coefficients of the model.
    axes[0,1].bar(x=inputs.columns.astype(str), height=model.x)
    axes[0,1].set_xticklabels(inputs.columns, rotation=45)
    axes[0,1].set_title('Model Coefficients')
    
    # Normal prob plot of errors.
    stats.probplot(sold_residuals, plot=axes[1,0])

    # Show a confusion matrix for categorization of whether a lot will sell above the estimate.
    conf_matrix = metrics.confusion_matrix(outputs.log10_hammer_price.notna(), predicted_price > outputs.log10_estimate_low)
    sn.heatmap(pd.DataFrame(conf_matrix, index=['!Sale', 'Sale'], columns=['E[!Sale]', 'E[Sale]']), ax=axes[1,1], 
                            annot=True, annot_kws={"size": 6}, fmt='.6g')

    # Show a histogram of the probability that a lot sells.
    axes[0,2].hist(prob_lot_sells, bins=50)
    axes[0,2].set_title('Probability Distribution for Does Lot Sell Estimate')

    # Scatter plot to show prediction vs reality.
    axes[1,2].scatter(predicted_price.loc[sold_lots], outputs.log10_hammer_price.loc[sold_lots])

    rsquared = np.power(np.corrcoef(predicted_price.loc[sold_lots], outputs.log10_hammer_price.loc[sold_lots])[0,1], 2)
    axes[1,2].set_title(f'Model Estimate vs Realized : {rsquared:.4f} R squared')

    # plt.figure(figsize=(20, 12), dpi=320, facecolor='w', edgecolor='k')
    plt.savefig('report.png', bbox_inches='tight', dpi=320)

if __name__ == "__main__":
    # Load data.
    df: pd.DataFrame = pd.read_csv('artists/picasso.csv') 

    # Filter malformed data.
    filtered: pd.DataFrame = filter_data(df)

    # Preprocess.
    inputs, outputs = preprocess_inputs(filtered, zscore=True)

    # Train model.
    model = train_model(inputs, outputs)

    # Evaluate model.
    evaluate_model(inputs, outputs, model)
