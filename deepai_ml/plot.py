import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


# Time series plot
def plot_range(start_date, end_date, col, data, title, figsize):
    """ Plot time series for a given date range

    :param start_date:
    :param end_date:
    :param col: the series column
    :param data: time series dataframe
    :param title: plot title
    :param figsize: plot size
    :return:
    """
    data.loc[start_date:end_date][col].plot(figsize=figsize, title=title)
    plt.show()


def plot_acf(data, col, lags, figsize):
    """ Plot autocorrelation for a time series data

    :param data: time series dataframe
    :param col (optional): the series column
    :param lags: number of lags
    :param figsize: plot size
    :return:
    """
    if col is not None:
        series = data[col]
    else:
        series = data
    fig,ax = plt.subplots(2,1,figsize=figsize)
    sm.graphics.tsa.plot_acf(series, lags=lags, ax=ax[0])
    sm.graphics.tsa.plot_pacf(series, lags=lags, ax=ax[1])
    plt.show()


# Value counts plot
def catplot(data, targ, height=2.5, aspect=3, orient='h'):
    """Plot target variable distribution using seaborn

    :param data: pandas dataframe
    :param targ: target variable
    :param height:
    :param aspect:
    :param orient:
    :return:
    """
    ax = sns.catplot(y=targ, kind='count', data=data, height=height, aspect=aspect, orient=orient)
    return ax


# Categorical plot
def barplot_categorical(df, targ, col, y_label, figsize=(10,6)):
    """ Bar Plot distribution of target variable by a categorical feature

    :param df: pandas dataframe
    :param targ: target variable
    :param col:
    :param y_label:
    :param figsize:
    :return:
    """
    fig, ax = plt.subplots(figsize=figsize)
    data = df.groupby(col)[targ].value_counts(normalize=True).to_frame()
    data.columns = [y_label]
    data.reset_index(inplace=True)
    sns.barplot(x=col, y= y_label, hue=targ, data=data, ax=ax)
    ax.set_yticklabels(['{:,.0%}'.format(y) for y in ax.get_yticks()])


def pieplot_categorical(pos, neg, col, title_pos, title_neg, suptitle_size=16, figsize=(12, 6)):
    """ Pie Plot distribution of a categorical feature by target variable

    :param pos: data where targ var = 1
    :param neg: data where targ var = 0
    :param col: a categorical feature
    :param title_pos:
    :param title_neg:
    :param suptitle_size:
    :param figsize:
    :return:
    """

    fig, ax = plt.subplots(1,2, figsize=figsize)
    fig.suptitle("{} distribution".format(col), fontsize=suptitle_size)

    #plot pos data
    data_pos = pos[col].value_counts()
    labels_pos = pos[col].value_counts().keys()
    ax[0].pie(data_pos, labels=labels_pos, autopct='%1.1f%%', shadow=True)
    ax[0].set_title(title_pos)

    #plot neg data
    data_neg = neg[col].value_counts()
    labels_neg = neg[col].value_counts().keys()
    ax[1].pie(data_neg, labels=labels_neg, autopct='%1.1f%%', shadow=True)
    ax[1].set_title(title_neg)


# Numerical Plot
def pairplot(data, vars, targ, height=4, aspect=1):
    """ Pairplot of numerical features using seaborn

    :param data: pandas dataframe
    :param vars: numerical variables
    :param targ: target variable
    :param height:
    :param aspect:
    :return:
    """
    ax = sns.pairplot(data=data, vars=vars, hue=targ, height=height, aspect=aspect)
    return ax


def distplot(data, figsize, bins=None, hist=True, kde=True, rug=False, fit=None,
             hist_kws=None, kde_kws=None, rug_kws=None, fit_kws=None, color=None,
             vertical=False, norm_hist=False, axlabel=None, label=None):
    """ Histogram of categorical feature

    Parameters:
    a : Series, 1d-array, or list.

    Observed data. If this is a Series object with a name attribute, the name will be used to label the data axis.

    bins : argument for matplotlib hist(), or None, optional

    Specification of hist bins, or None to use Freedman-Diaconis rule.

    hist : bool, optional

    Whether to plot a (normed) histogram.

    kde : bool, optional

    Whether to plot a gaussian kernel density estimate.

    rug : bool, optional

    Whether to draw a rugplot on the support axis.

    fit : random variable object, optional

    An object with fit method, returning a tuple that can be passed to a pdf method a positional arguments following an grid of values to evaluate the pdf on.

    {hist, kde, rug, fit}_kws : dictionaries, optional

    Keyword arguments for underlying plotting functions.

    color : matplotlib color, optional

    Color to plot everything but the fitted curve in.

    vertical : bool, optional

    If True, observed values are on y-axis.

    norm_hist : bool, optional

    If True, the histogram height shows a density rather than a count. This is implied if a KDE or fitted density is plotted.

    axlabel : string, False, or None, optional

    Name for the support axis label. If None, will try to get it from a.namel if False, do not set a label.

    label : string, optional

    Legend label for the relevent component of the plot

    ax : matplotlib axis, optional

    if provided, plot on this axis

    Returns:
    ax : matplotlib Axes

    Returns the Axes object with the plot for further tweaking.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.distplot(data, bins=bins, hist=hist, kde=kde, rug=rug, fit=fit,
                 hist_kws=hist_kws, kde_kws=kde_kws, rug_kws=rug_kws, fit_kws=fit_kws, color=color,
                 vertical=vertical, norm_hist=norm_hist, axlabel=axlabel, label=label, ax=ax)


# Correlation plot
def plot_correlation_heatmap(data, title, title_size=20, figsize=(14,12)):
    """ Plot correlation heatmap of variables

    :param data: pandas dataframe
    :param title: plot title
    :param title_size:
    :param figsize:
    :return:
    """
    colormap = plt.cm.RdBu
    plt.figure(figsize=figsize)
    plt.title(title, size=title_size)
    sns.heatmap(data.corr(),linewidths=0.5, cmap=colormap, annot=True)
    plt.show()


# Feature importance plot
def plot_feat_importance(data, clf, title="Feature importance", figsize=(10,8)):
    """ Plot feature importance

    :param data: train set
    :param clf: classifier that support feature importance (randomforest, xgboost, gradient boost, etc.)
    :param title: plot title
    :param figsize:
    :return:
    """
    imp = pd.Series(data=clf.feature_importances_, index=data.columns).sort_values(ascending=False)
    plt.figure(figsize=figsize)
    plt.title(title)
    ax = sns.barplot(y=imp.index, x=imp.values, palette="Blues_d", orient='h')
    return ax

