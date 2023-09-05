import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

sns.set(style="ticks", context="talk")
plt.style.use("dark_background")

import matplotlib as mpl

mpl.rcParams["axes.formatter.useoffset"] = False
mpl.rcParams["axes.formatter.limits"] = (-1_000_000, 1_000_000)


def plot_variable_pairs(df):
    """
    Plots all pairwise relationships in a dataframe along with the regression line for each pair.

    Parameters:
    df (pandas.DataFrame): The input dataframe.
    """
    # Set the style of the plot
    sns.set(style="ticks", context="talk")
    plt.style.use("dark_background")

    # Plot the pairwise relationships with regression line
    sns.pairplot(
        df, kind="reg", plot_kws={"scatter_kws": {"s": 1, "color": "lightcoral"}}
    )


def plot_categorical_and_continuous_vars(df, cat_vars, cont_vars):
    """
    Plots 3 different plots for visualizing a categorical variable and a continuous variable.

    Parameters:
    df (pandas.DataFrame): The input dataframe.
    cat_vars (list): The list of column names that hold the categorical features.
    cont_vars (list): The list of column names that hold the continuous features.
    """

    for cat_var in cat_vars:
        for cont_var in cont_vars:
            # Plot a boxplot of the continuous variable for each category
            plt.figure(figsize=(12, 8))
            sns.boxplot(x=cat_var, y=cont_var, data=df, palette="pastel")
            plt.title(f"{cont_var} by {cat_var}")
            sns.set(style="ticks", context="talk")
            plt.style.use("dark_background")
            plt.show()

            # Plot a histogram of the continuous variable for each category

            plt.figure(figsize=(12, 8))
            for cat in df[cat_var].unique():
                sns.histplot(
                    df[df[cat_var] == cat][cont_var],
                    label=cat,
                    alpha=0.5,
                    kde=True,
                    palette="pastel",
                )
            plt.title(f"{cont_var} by {cat_var}")
            plt.legend()
            sns.set(style="ticks", context="talk")
            plt.style.use("dark_background")
            plt.show()

            # Plot a violinplot of the continuous variable for each category
            plt.figure(figsize=(12, 8))
            sns.violinplot(x=cat_var, y=cont_var, data=df, palette="pastel")
            plt.title(f"{cont_var} by {cat_var}")
            sns.set(style="ticks", context="talk")
            plt.style.use("dark_background")
            plt.show()
