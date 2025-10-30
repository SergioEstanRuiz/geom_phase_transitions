import numpy as np

def empirical_correlation(x, y, tau: float) -> float:
    """Compute the empirical correlation between two lists of numbers.

    Args:
        x (list[float]): First list of numbers.
        y (list[float]): Second list of numbers.
        tau (float): Time lag to apply to the second list.

    Returns:
        float: Empirical correlation between x and y.
    """
    x = np.array(x)
    y = np.array(y)

    if tau > 0:
        x_tau = x[:-tau]
        y_tau = y[tau:]
        x_stand = (x_tau - np.mean(x_tau)) / np.std(x_tau)
        y_stand = (y_tau - np.mean(y_tau)) / np.std(y_tau)
        correlation = np.mean(x_stand * y_stand)
    elif tau < 0:
        tau = -tau
        x_tau = x[tau:]
        y_tau = y[:-tau]
        x_stand = (x_tau - np.mean(x_tau)) / np.std(x_tau)
        y_stand = (y_tau - np.mean(y_tau)) / np.std(y_tau)
        correlation = np.mean(x_stand * y_stand)
    else:
        x_stand = (x - np.mean(x)) / np.std(x)
        y_stand = (y - np.mean(y)) / np.std(y)
        correlation = np.mean(x_stand * y_stand)
    # Notice that the above, E_xy, is equivalent to the correlation because we have unitary variance

    return correlation

def list_lagged_correlation(x, y, max_lag: int) -> list:
    """Compute the lagged correlation between two lists of numbers.

    Args:
        x (list[float]): First list of numbers.
        y (list[float]): Second list of numbers.
        max_lag (int): Maximum lag to compute.

    Returns:
        list[float]: List of lagged correlations.
    """
    correlations = []
    for tau in range(-max_lag,max_lag + 1):
        corr = empirical_correlation(x, y, tau)
        correlations.append(corr)
    return correlations

def grokking_test1(test_acc):
    if test_acc[-1] > 0.95:
        return True
    else:
        return False

def grokking_test2(train_acc, test_acc):
    dummy = False
    for i in range(len(train_acc)):
        if train_acc[i] > 0.9 and test_acc[i] < 0.5:
            dummy = True
    if dummy:
        if test_acc[-1] > 0.9:
            return True
    return False

def grokking_test3(train_acc, test_acc):
    level = (train_acc - test_acc).mean() # Riemann sum approximation of area between curves
    mask = (test_acc[-1] > 0.9) and (train_acc[-1] > 0.9)
    return level if mask else 0