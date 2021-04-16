import pandas as pd
import matplotlib.pyplot as plt


def read_data():
    data = pd.read_csv("grid_search_results.csv")
    data = data.drop(columns="Unnamed: 0")
    return data


def plot_data(df, x_col, y_cols):
    for y_col in y_cols:
        plt.plot(df.loc[:, x_col], df.loc[:, y_col])
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.show()
    

if __name__ == "__main__":
    gsr_df = read_data()
    gsr_df = gsr_df.sort_values(by="rank_test_score")

    plot_data(gsr_df, x_col="mean_test_score", y_cols=["param_dropout", "param_hidden_layers", "param_units", "mean_fit_time"])

    print(gsr_df.columns)