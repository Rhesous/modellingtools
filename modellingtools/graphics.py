import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_factor_imp(regobj, nb_class, var_names = None):
    """
    Plot factor importance for regression and GBM.
    For the moment this function is only implemented for GBM and linear model from SKlearn, and output the n_class most
    important variables. For regression it is the value of the coefficient that is taken into account, and for GBM the
    relative influence.
    Args:
        regobj: The regression object, either a GBM or linear model already fitted.
        nb_class: Number of variables to consider.
        var_names: Vector containing the names of the variables.

    Returns: plot the graphic.

    """
    # if reglin
    if "linear_model" in regobj.__module__:
        df_coefs_ = pd.DataFrame({'coeff':regobj.coef_},index= var_names)
        df_coefs_ = df_coefs_.reindex(df_coefs_.coeff.abs().sort_values(inplace=False,ascending=True).index)
        xlab_ = 'Coefficient value'
        title_ = '{} biggest coefficients for linear regression'.format(nb_class)
    # if gbm
    elif "gradient_boosting" in regobj.__module__:
        df_coefs_ = pd.DataFrame({'coeff':regobj.feature_importances_ *100},index= var_names)
        df_coefs_ = df_coefs_.reindex(df_coefs_.coeff.abs().sort_values(inplace=False,ascending=True).index)
        xlab_ = 'Relative Importance'
        title_ = 'Variable Importance - {} most important variables'.format(nb_class)

    nb_class = min(nb_class, df_coefs_.shape[0])
    pos = np.arange(nb_class) + .5
    plt.subplot(1, 1, 1)
    col =  df_coefs_.coeff.apply(lambda x: "steelblue" if x >0 else "firebrick")
    plt.barh(pos,  df_coefs_[-nb_class:]['coeff'], align='center',color=col[-nb_class:])
    plt.yticks(pos, df_coefs_.index.tolist()[-nb_class:])
    plt.xlabel(xlab_)
    plt.title(title_);


def plot_means_match(data, var_targ, ytarg, pred):
    """
    Plot the means of actual and fitted value, for each class defined as value of the var target. If the var target
    as more than 20 variables, it will be cutted into bins.
    Args:
        data: database
        var_targ: variable to use for splitting the database
        ytarg: target variable, actual values
        pred: fitted values

    Returns: plot the graphic.

    """
    df = data.copy()
    df['pred'] = pred
    df['target'] = ytarg
    if len(set(df[var_targ])) > 20:
        df[var_targ] = pd.cut(df[var_targ], 20)

    gb = df.groupby(var_targ)
    counts = gb.size().to_frame(name='counts')
    gb = (counts
          .join(gb.agg({'target': 'mean'}).rename(columns={'target': 'actual'}))
          .join(gb.agg({'pred': 'mean'}).rename(columns={'pred': 'predict'}))
          .reset_index()
          )
    var_x = np.arange(len(gb[var_targ]))
    fig, ax = plt.subplots()
    plt.bar(var_x, gb.counts)
    plt.xticks(var_x, gb[var_targ])
    ax.grid(False)
    ax2 = ax.twinx()
    plt.title(var_targ)
    plt.plot(var_x, gb.actual, color="red", marker='o')
    plt.plot(var_x, gb.predict, color="green", marker='o')
    ax2.grid(False)
    plt.tight_layout()
    fig.autofmt_xdate()
    plt.xlabel(var_targ)


def lift_curve(ytarg, pred):
    """
    Plot a lift curve, that divide the values between 20 equal bands of predicted values, and compare in each band
    the mean of both actual and fitted values.
    Args:
        ytarg: target values
        pred: fitted values

    Returns: plot the graphic

    """
    data = (pd.DataFrame({"actual": ytarg, "model": pred})
            .groupby(pd.cut(pred, np.percentile(a=pred, q=np.arange(0, 100, 5))))
            .agg(['mean', 'count']))

    fig, ax = plt.subplots()
    ax.grid(False)
    plt.title("WIP")
    plt.plot(data.index.categories.astype(str), data['actual']['mean'], color="red", marker='o')
    plt.plot(data.index.categories.astype(str), data['model']['mean'], color="green", marker='o')
    plt.tight_layout()
    fig.autofmt_xdate()
    plt.legend(('Actual', 'Model'))