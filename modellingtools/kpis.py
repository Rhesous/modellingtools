import numpy as np

def Gini(y_true, y_pred):
    """
    Calculate the normalized gini index without weights. It is a float between -1 and 1.
    The higher the value is, the closer the model, only taking into account the ordering of the values.
    A negative Gini means most of the time that the model inverts high values and low values.
        ordering of the target value.
    From kaggle : https://www.kaggle.com/jpopham91/gini-scoring-simple-and-efficient
    Args:
        y_true: true value of target variable
        y_pred: predicted values to compare

    Returns:
        The gini index as a float value.

    """
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]

    # sort rows on prediction column
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:, 0].argsort()][::-1, 0]
    pred_order = arr[arr[:, 1].argsort()][::-1, 0]

    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(1 / n_samples, 1, n_samples)

    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    # normalize to true Gini coefficient
    return G_pred / G_true