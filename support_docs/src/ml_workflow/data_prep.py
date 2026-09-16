import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm


def null_summary(df):
    """Takes dataset as parameter and returns the number of NULL entries per variable in the dataset."""
    nulls = pd.DataFrame(df.isnull().sum(), columns=['# NULLS'])
    percNulls = pd.DataFrame(df.isnull().mean(), columns=['% NULLS'])
    return pd.concat([nulls, percNulls], axis=1)

def classify_columns(df, max_categorical=10):
    categorical, continuous = [], []
    for col in df.columns:
        if df[col].dtype.name in ('object', 'category', 'bool'):
            categorical.append(col)
        elif df[col].nunique(dropna=True) <= max_categorical:
            categorical.append(col)          # numeric but low-cardinality → categorical
        else:
            continuous.append(col)
    return continuous, categorical

def plot_numeric(df, x, bins=50):
    """Plots the distribution of a numeric variable
    plot_numeric(df, variable, bins (default 250))"""
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.histplot(df[x], bins=bins, kde=True, color='steelblue', edgecolor='white', ax=ax)
    ax.set_title(f'Distribution of {x}')
    ax.set_xlabel(x)
    ax.set_ylabel('Count')
    sns.despine(ax=ax)
    return ax

def plot_categorical(df, x):
    fig, ax = plt.subplots(figsize=(15, 8))
    order = df[x].value_counts().index
    sns.countplot(data=df, x=x, color='steelblue', edgecolor='white', order=order, ax=ax)
    ax.set_title(f'Counts of {x}')
    ax.set_xlabel(x)
    ax.set_ylabel('Count')
    sns.despine(ax=ax)
    return ax


def plot_relation_ordinal(df, x, y, alpha = 0.4, size = 3):
    """Plots the distribution of a numeric variable
    plot_numeric(df, x_variable, y_variable, alpha (default 0.4), size (default 3)"""
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.boxplot(data=df, x=x, y=y, color='lightgray', fliersize=0, ax=ax)
    sns.stripplot(data=df, x=x, y=y, color='steelblue', alpha=alpha, jitter=0.25, size=size, ax=ax)
    ax.set_title(f'{y} by {x}')
    sns.despine(ax=ax)
    return ax

def plot_relation_continuous(df, x, y, alpha = 0.4):
    """Plots the distribution of a numeric variable
    plot_numeric(df, x_variable, y_variable, alpha (default 0.4)"""
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.scatterplot(data=df, x=x, y=y, color='steelblue', alpha=alpha, edgecolor=None, ax=ax)
    ax.set_title(f'{y} vs {x}')
    sns.despine(ax=ax)
    return ax

def plot_correlation(df, method='pearson'):
    """Plots the correlation matrix of the numeric variables in the dataset, standard
    method is Pearson"""
    corr = df.select_dtypes(include=np.number).corr(method=method)
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, vmin=-1, vmax=1,
                annot=True, fmt='.2f', square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_title('Correlation matrix')
    return ax

def compute_vif(df):
    X = df.select_dtypes(include=np.number).dropna()
    X = sm.add_constant(X)
    vif = pd.DataFrame({
        'variable': X.columns,
        'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    return vif[vif['variable'] != 'const'].sort_values('VIF', ascending=False).reset_index(drop=True)

def make_dummies(df, cols=None, drop_first=True):
    """Creates dummy variables for categorical columns in the dataset. If no columns are specified,
      it will create dummies for all object and category type columns. 
      The drop_first parameter allows you to drop the first level of each categorical variable to
      avoid multicollinearity."""
    if cols is None:
        cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return pd.get_dummies(df, columns=cols, drop_first=drop_first)

def standardise(df, cols=None, stats=None):
    """Standardizes the specified columns in the dataset using mean and standard deviation. 
    If no columns are specified, it will standardize all numeric columns."""
    out = df.copy()
    if cols is None:
        cols = out.select_dtypes(include=np.number).columns.tolist()
    if stats is None:
        stats = {'mean': out[cols].mean(), 'std': out[cols].std()}
    out[cols] = (out[cols] - stats['mean']) / stats['std']
    return out, stats