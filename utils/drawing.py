from utils.metrics import list_lagged_correlation
import matplotlib.pyplot as plt

def draw(results, experiment_name: str):
    max_lag = 5
    val_loss = results["val_loss"].values
    llc = results["llc"].values
    lagged_corr = list_lagged_correlation(llc, val_loss, max_lag=max_lag)
    lags = range(-max_lag, max_lag + 1)
    # clear previous plot
    plt.clf()
    # plot lagged correlation
    plt.plot(lags, lagged_corr)
    plt.xlabel("Lag")
    plt.ylabel("Lagged Correlation")
    plt.title(f"Lagged Correlation between Val Loss and LLC for modular addition task")
    # save the figure
    plt.savefig(f"results/{experiment_name}/lagged_correlation.png")