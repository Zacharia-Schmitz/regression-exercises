import seaborn as sns
import matplotlib.pyplot as plt


def plot_variable_pairs(df):
    """
    Plots all pairwise relationships in a dataframe along with the regression line for each pair.

    Parameters:
    df (pandas.DataFrame): The input dataframe.
    """
    # Set the style of the plot
    sns.set(style="ticks", color_codes=True)

    # Plot the pairwise relationships with regression line
    sns.pairplot(df, kind="reg")


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
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=cat_var, y=cont_var, data=df)
            plt.title(f"{cont_var} by {cat_var}")
            plt.show()

            # Plot a histogram of the continuous variable for each category
            plt.figure(figsize=(8, 6))
            for cat in df[cat_var].unique():
                sns.histplot(
                    df[df[cat_var] == cat][cont_var], label=cat, alpha=0.5, kde=True
                )
            plt.title(f"{cont_var} by {cat_var}")
            plt.legend()
            plt.show()

            # Plot a violinplot of the continuous variable for each category
            plt.figure(figsize=(8, 6))
            sns.violinplot(x=cat_var, y=cont_var, data=df)
            plt.title(f"{cont_var} by {cat_var}")
            plt.show()
