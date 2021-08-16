#!/usr/bin/env python
# coding: utf-8

# # Capstone Week 5
# ---

# # Index
# - [Capstone Objectives](#Capstone-Objectives)
# - [Read in Data](#Read-in-Data)
#     - [Merge 2018 and 2019](#Merge-2018-and-2019)
#     - [Make advisor and firm dictionary mapper](#Make-advisor-and-firm-dictionary-mapper)
# - [EDA](#EDA)
# - [Data Cleaning](#Data-Cleaning)
#     - [Train-Test-Split](#Train-Test-Split)
# - [Missing Data](#Missing-Data)
#     - [How big of a problem is missing data?](#How-big-of-a-problem-is-missing-data?)
#     - [Three types of missing data](#Three-types-of-missing-data)
#     - [Strategies for handling missing data](#Strategies-for-handling-missing-data)
#         - [Weight Class Adjustment Example](#Weight-Class-Adjustment-Example)
#     - [Imputation Strategies](#Imputation-Strategies)
#     - [Missingness Tests](#Missingness-Tests)
#     - [MCAR Data](#MCAR-Data)
#     - [MAR Data](#MAR-Data)
#     - [NMAR Data](#NMAR-Data)
#     - [Missing data workflow](#Missing-data-workflow)
#     - [Custom Cleaning Functions](#Custom-Cleaning-Functions)
#     - [Create Cleaning Pipeline](#Create-Cleaning-Pipeline)
# - [Model building](#Model-building)
# - [Make predictions](#Make-predictions)
# - [Feature Engineering](#Feature-Engineering)
#     - [Variable Inflation Factor (VIF)](#Variable-Inflation-Factor-(VIF))
# - [Residuals](#Residuals)
# - [Classification](#Classification)
# - [Model Interpretation](#Model-Interpretation)

# # Capstone Objectives
# - Assist sales and marketing by improving their targeting
# - Predict sales for 2019 using the data for 2018
# - Estimate the probability of adding a new fund in 2019

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

pd.set_option('display.max_columns', 50)


# [Back to Top](#Index)
# # Read in Data

# In[2]:


df18 = pd.read_excel("../../capstone-run2/Transaction Data.xlsx", sheet_name='Transactions18')
df19 = pd.read_excel("../../capstone-run2/Transaction Data.xlsx", sheet_name='Transactions19')
firm = pd.read_excel("../../capstone-run2/Firm Information.xlsx", sheet_name="Rep summary")


# ## Merge 2018 and 2019

# In[3]:


df = pd.merge(
    df18,
    df19,
    on='CONTACT_ID',
    suffixes=['_2018', '_2019']
)
df.head()


# ## Make advisor and firm dictionary mapper

# In[4]:


adviser_lookup = {
    idx: contact_id 
        for idx, contact_id in enumerate(df['CONTACT_ID'])
}


# In[5]:


adviser_lookup[10]


# In[6]:


firm_lookup = {idx: contact_id for idx, contact_id in enumerate(firm['Contact ID'])}


# In[7]:


firm_lookup[10]


# [Back to Top](#Index)
# # EDA

# In[8]:


# !conda install -yc conda-forge pandas-profiling


# In[9]:


# from pandas_profiling import ProfileReport

# missing_diagrams = {
#     'heatmap': True, 'dendrogram': True, 'matrix':True, 'bar': True,
# }

# profile = ProfileReport(df, title='Nuveen Profile Report', missing_diagrams=missing_diagrams)

# profile.to_file(output_file="nuveen_profiling.html")


# [Back to Top](#Index)
# # Data Cleaning

# In[10]:


# make a variable to keep all of the columns we want to drop
COLS_TO_DROP = [
    'refresh_date_2019', 'refresh_date_2018', 'CONTACT_ID',
    'Contact ID', 'CustomerID', 'Firm ID', 'Office ID',
    'Channel','Sub channel', 'Firm name'
]

COLS_TO_KEEP = [
    'no_of_sales_12M_1', 'no_of_Redemption_12M_1', 'no_of_sales_12M_10K',
    'no_of_Redemption_12M_10K', 'no_of_funds_sold_12M_1',
    'no_of_funds_redeemed_12M_1', 'no_of_fund_sales_12M_10K',
    'no_of_funds_Redemption_12M_10K', 'no_of_assetclass_sold_12M_1',
    'no_of_assetclass_redeemed_12M_1', 'no_of_assetclass_sales_12M_10K',
    'no_of_assetclass_Redemption_12M_10K', 'No_of_fund_curr',
    'No_of_asset_curr', 'AUM', 'sales_curr', 'sales_12M_2018',
    'redemption_curr', 'redemption_12M', 'new_Fund_added_12M_2018',
    'aum_AC_EQUITY', 'aum_AC_FIXED_INCOME_MUNI',
    'aum_AC_FIXED_INCOME_TAXABLE', 'aum_AC_MONEY', 'aum_AC_MULTIPLE',
    'aum_AC_PHYSICAL_COMMODITY', 'aum_AC_REAL_ESTATE', 'aum_AC_TARGET',
    'aum_P_529', 'aum_P_ALT', 'aum_P_CEF', 'aum_P_ETF', 'aum_P_MF',
    'aum_P_SMA', 'aum_P_UCITS', 'aum_P_UIT', 
]

FIRM_COLS = ['Contact ID', 'Channel','Sub channel',]


# # Make `Firm` data pipeline

# In[11]:


df = pd.merge(df, firm, left_on="CONTACT_ID", right_on='Contact ID')
df.head(1)


# In[12]:


X = df.drop(['sales_12M_2019', 'new_Fund_added_12M_2019'], axis=1)
y_reg = df['sales_12M_2019']
y_cl = df['new_Fund_added_12M_2019']


# ## Train-Test-Split

# In[13]:


X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.3, random_state=24
)
y_train_cl, y_test_cl = y_cl[y_train_reg.index], y_cl[y_test_reg.index]


# [Back to Top](#Index)
# # Missing Data

# Up to this point we haven't put much thought into dealing with missing data. Missing data is EVERYWHERE and it's important to know how to do data science with missing data. It can significantly undermine our results!

# ### How big of a problem is missing data?

# This is difficult question because we only see what we observe. We can use simulated data to help answer this question, but we cannot quantify the impact of missing data in our real data projects.
# 
# See this resource: https://github.com/matthewbrems/jupytercon-missing-data-2018

# ### Three types of missing data

# 1. **MCAR**: Missing Completely at Random
#     - Some intern spills their coffee on your surveys in random order
#     - Flip coin of missingness
#     
#     
# 2. **MAR**: Missing at Random
#     - I adminster a survey about income. Those who are female are less likely to respond to the question about income.
#     - Missing data is conditional on data have observed.
# 
# 
# 3. **NMAR**: Not Missing at Random (Worst type!)
#     - I adminster a survey that includes a question about income. Those who have lower incomes are less likely to respond to the question about income.
#     - Data of interest are systematically different for respondents and nonrespondents
#     - Whether the data are missing or not depends on the value of the unobserved value itself!

# [Back to Top](#Index)
# ### Strategies for handling missing data
# 1. **Avoid it** (best option, if possible)
#     - Use sound design when collecting data
#     - Improve survey questioning and design
#     - Drop all rows with _any_ missing value
#     
#     
# 2. **Ignore it** (second best option, if possible): 
#     - Assume your respondents are close enough to the sample of non-respondents
#     - Drop any observation with _any_ missing value
#     
#     
# 3. **Account for it** (most common):
#     - Weight class adjustments
#     - Determine why data are missing
#     - Employ a strategy for accounting for missing data

# [Back to Top](#Index)
# #### Weight Class Adjustment Example
# 
# I'm estimating job satisfaction among two departments: finance and accounting. Both departments are the same size (A: 50%, F: 50%).
# 
# $$W_{finance} = \frac{true\;proportion}{proportion\;of\;responses} = \frac{0.50}{0.25} = 2$$
# <br>
# $$W_{accounting} = \frac{true\;proportion}{proportion\;of\;responses} = \frac{0.50}{0.75} = \frac{2}{3}$$

# [Back to Top](#Index)
# ### Imputation Strategies
# 
# 1. Deductive Imputation: use logical relationships to fill in value **VALID**
# 
#     - Respondent says the were not victim of crime, but left "victim of a violent crime" question blank.
#     - If someone has 2 children in year 1, `NaN` children in year 2, and 2 children in year 3, we can _probably_ impute that in year 2 they still had 2 children.
#     - PRO: Valid method, requires minimal "inference"
#     - CON: Time consuming and requires specific coding
# 
# 
# 2. Mean/Median/Mode: use measure central tendency to fill value **INVALID**
# 
#      - PRO: Easy to implement
#      - CON: Significantly distorts histogram (underestimates variance) and results will look more precise than they really are
#      
# 
# 3. Regression Imputation: replace missing based on predicted value from regression line **INVALID**
# 
#     - PRO: Easy to understand
#     - CON: Distorts distribution and underestimates variance still because there is no randomness in the prediction
#     
#     
# 4. Stochastic Regression Imputation:
# 
#     - Replace missing with predicted value from regression line plus random draw from normal distribution `N(0, s)`, where `s` is estimataed from model residuals **INVALID**
#     
#     - PRO: Easy to understand and better than just regression technique
#     - CON: Still under estimate variance because selecting single point from normal distribution of error
#     
#     
# 5. Multiply Stochastic Regression Imputation: pull multiple values from distribution. Replace missing with predicted value from line with random error.
# 
#     - PRO: Better than number 4
#     - CON: All `Beta` coefficients are constant, so still not credible
#     
#     
# 6. Proper Multiply Stochastic Regression Imputation: Called Multiple Imputation by Chained Equations [(MICE)](https://stats.stackexchange.com/questions/421545/multiple-imputation-by-chained-equations-mice-explained)
# 
#     - Create `n` copies of your data set (let's say, 10)
#     - For each dataset:
#         - Generate coefficients for your regression model
#             - For each missing value:
#                 - Replace `NaN` with a value predicted from a regression
#             - Do your "final analysis" or generate your final prediction
#     - Aggregate your analysis/predictions across all data sets so you have one complete analysis
#     - These predictions were created by properly estimating the variance in your data
#     - PRO: Very good method, **VALID**
#     - CON: Takes more effort to implement (`fancyimpute` or `mice` in R)

# [Back to Top](#Index)
# ### Missingness Tests
# 
# 1. Little's Test for MCAR
#     - $H_0 : MCAR$
#     - $H_A : MAR$
#     - There is no test for NMAR!
# 2. Split your data into "observed" and "unobserved" and compare them
#     - Split missing `income` and observed `income` sets. Do the other variables have the same distributions?
# 3. Think about missing data process. Can you come up with a reasonable answer based on how missing data came about?

# [Back to Top](#Index)
# ### MCAR Data
# 
# Use any of the methods we previously discussed:
# - Deductive imputation
# - Proper imputation
# - Stochastic Regression Imputation
# - Complete-Case Removal (unbiased, but variance will be higher because our sample size is smaller!)

# ### MAR Data
# 
# Use one of the following methods:
# - Deductive imputation
# - Proper imputation
# - Stochastic Regression Imputation

# ### NMAR Data
# 
# Use one of the following methods:
# - Deductive imputation
# - Advanced methods: selection models and pattern mixture models

# [Back to Top](#Index)
# ### Missing data workflow
# 1. How much missing data do I have?
# 2. For each variable, estimate the type of missing data
# 3. What is the best method for handling missing values?

# ## Custom Cleaning Functions

# Let's create functions that do some basic housekeeping

# In[14]:


def extract_columns(df):
    '''extract out columns not listed in COLS_TO_DROP variable'''
    cols_to_keep = [col for col in df.columns if col not in COLS_TO_DROP]
    return df.loc[:, cols_to_keep].copy()


def fillna_values(df):
    '''fill nan values with zero'''
    if isinstance(df, type(pd.Series(dtype='float64'))):
        return df.fillna(0)
    num_df = df.select_dtypes(include=['number']).fillna(0)
    non_num_df = df.select_dtypes(exclude=['number'])
    return pd.concat([num_df, non_num_df], axis=1)


def negative_to_zero(df):
    if isinstance(df, type(pd.Series(dtype='float64'))):
        return df.apply(lambda x: max(0, x))
    else:
        return df.select_dtypes(include='number').clip(lower=0)


# [Back to Top](#Index)
# ## Create Cleaning Pipeline
# 
# - Pipeline for target variable
# - Pipeline for features

# In[15]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PowerTransformer
from sklearn.preprocessing import StandardScaler


# In[16]:


extract_columns_trans = FunctionTransformer(extract_columns)
fillna_values_trans = FunctionTransformer(fillna_values)
negative_to_zero_trans = FunctionTransformer(negative_to_zero)


# Make pipeline for regression target variable

# In[17]:


targ_pipe_reg = Pipeline([
    ('fillna_values_trans', fillna_values_trans),
    ('negative_to_zero_trans', negative_to_zero_trans),
    ('PowerTransformer', PowerTransformer(standardize=False))
])

y_train_reg = pd.Series(
    targ_pipe_reg.fit_transform(y_train_reg.to_frame()).squeeze(),
    index=y_train_reg.index
)
y_test_reg = pd.Series(
    targ_pipe_reg.transform(y_test_reg.to_frame()).squeeze(),
    index=y_test_reg.index
)


# In[18]:


y_train_reg.hist(bins=50)


# Transform the classification target

# In[19]:


from sklearn.preprocessing import Binarizer

targ_pipe_cl = Pipeline([
    ('fillna_values_trans', fillna_values_trans),
    ('Binarizer', Binarizer(threshold=0))
])

y_train_cl = pd.Series(
    targ_pipe_cl
        .fit_transform(y_train_cl.to_frame())
        .reshape(-1), index=y_train_cl.index)

y_test_cl = pd.Series(
    targ_pipe_cl
        .transform(y_test_cl.to_frame())
        .reshape(-1), index=y_test_cl.index)
y_test_cl


# Create the pipeline for the features

# In[20]:


X_train.head()


# In[21]:


feat_pipe = Pipeline([
    ('extract_columns_trans', extract_columns_trans),
    ('fillna_values_trans', fillna_values_trans),
    ('StandardScaler', StandardScaler()),
])

X_train_prepared = feat_pipe.fit(X_train).transform(X_train)
X_test_prepared = feat_pipe.transform(X_test)


# **TRANSFORM** Test set

# In[22]:


X_train_prepared = pd.DataFrame(
    X_train_prepared,
    index=X_train.index,
    columns=COLS_TO_KEEP
)

X_test_prepared = pd.DataFrame(
    feat_pipe.transform(X_test),
    index=X_test.index,
    columns=COLS_TO_KEEP
)


# In[23]:


X_test_prepared


# [Back to Top](#Index)
# # Model building
# - Evaluate baseline model
# - Create new models
# - Create evaluation function and cross validate

# ### Decision Tree Regressor

# In[53]:


from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


# In[50]:


dtr = DecisionTreeRegressor()


# In[51]:


dtr.fit(X_train_prepared, y_train_reg)


# In[52]:


dtr.predict(X_test_prepared)


# In[54]:


dtc = DecisionTreeClassifier()
dtc.fit(X_train_prepared, y_train_cl)


# In[55]:


dtc.predict(X_test_prepared)


# In[76]:


import seaborn as sns


# In[77]:


sns.heatmap(X_train.corr())


# In[ ]:





# In[74]:


# Create Decision Tree classifer object
clf = DecisionTreeClassifier(max_leaf_nodes=4, random_state=0)

# Train Decision Tree Classifer
model = clf.fit(X_train_prepared,y_train_cl)

#Predict the response for test dataset
y_pred = model.predict(X_test_prepared)


from sklearn import tree
text_representation = tree.export_text(clf)
print(text_representation)
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf,filled=True)


# In[57]:


X_train_prepared2 = X_train_prepared.copy()


# In[58]:


X_train_prepared2['dt_result'] = dtc.predict(X_train_prepared)


# In[62]:


for_next_model = X_train_prepared2[X_train_prepared2['dt_result']==1].index


# In[63]:


y_train_reg2 = y_train_reg.copy()


# In[64]:


y_train_reg2.loc[for_next_model]


# In[65]:


X_train_prepared.shape


# In[71]:


X_train['refresh_date_2018']


# In[ ]:





# In[ ]:





# In[ ]:





# In[24]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA, TruncatedSVD
import xgboost as xgb


# In[25]:


lr = LinearRegression()
lr.fit(X_train_prepared, y_train_reg)


# In[26]:


from sklearn.model_selection import cross_validate


# In[27]:


-cross_validate(
    lr, 
    X_train_prepared, 
    y_train_reg, 
    cv=3, 
    scoring='neg_root_mean_squared_error', 
    return_train_score=True
)['test_score']


# Make a plot of predictions vs actual

# In[28]:


y_test_reg_preds = lr.predict(X_test_prepared)


# In[29]:


fig, axes = plt.subplots(figsize=(8, 6))

axes.scatter(x=y_test_reg, y=y_test_reg_preds)

axes.plot([0, 20000000], [0,20000000])
axes.set_title("Actual vs Predicted - Regression")
axes.set_xlabel("Actual")
axes.set_ylabel("Predicted");


# In[30]:


def evaluate_model(model, X, y):
    print("Cross Validation Scores:")
    print(-cross_validate(model, X, y, scoring='neg_root_mean_squared_error')['test_score'])
    print('-'*55)
    preds = model.predict(X)
    lim = max(preds.max(), y.max())
    fig, ax = plt.subplots(1,1,figsize=(7,5))
    ax.scatter(x=y, y=preds, alpha=0.4)
    ax.plot([0, lim], [0, lim])
    ax.set_xlim([0, lim])
    ax.set_ylim([0, lim])
    ax.set_title("Actual vs Predicted - Regression")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted");


# In[31]:


evaluate_model(lr, X_test_prepared, y_test_reg)


# ## Make function to output deciles

# In[32]:


def output_deciles(model, X, y):
    results = pd.DataFrame(model.predict(X), index=X.index, columns=['predictions'])
    results['actual'] = y.values
    results['deciles'] = pd.qcut(results['predictions'], 10, labels=False)
    results['contact_id'] = results.index.map(adviser_lookup)
    return results


# In[33]:


regression_deciles = output_deciles(lr, X_test_prepared, y_test_reg)


# In[34]:


regression_deciles.groupby('deciles')[['actual']].mean()


# In[79]:


from sklearn.feature_selection import RFE


# In[80]:


rfe = RFE(lr, n_features_to_select=10)


# In[81]:


rfe.fit(X_train_prepared, y_train_reg)


# In[83]:


X_train_prepared.columns[rfe.support_]


# In[84]:


rfe.predict(X_test_prepared)


# In[92]:


for feat, imp in zip(X_train_prepared.columns,dtr.feature_importances_):
    print(sorted(feat, imp, lambda x: x[0]))


# In[ ]:


pd.qcut()


# In[78]:


X_train_prepared


# In[ ]:





# In[ ]:





# [Back to Top](#Index)
# ## Residual Analysis

# In[35]:


y_test_reg_preds = lr.predict(X_test_prepared)


# In[36]:


y_test_reg_preds


# In[37]:


# get the residuals
residuals = y_test_reg_preds - y_test_reg


# In[38]:


# plot predictions vs residuals
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14,10))

# plot scatter plot on upper left plot
axes[0,0].scatter(x=y_test_reg_preds, y=residuals, alpha=0.5)
axes[0,0].set(xlabel='Predictions', ylabel='Residuals')

# plot a hist on upper right plot
axes[0,1].hist(residuals, bins=50)
axes[0,1].set(xlabel='Residuals', ylabel='Frequency');


# In[39]:


from statsmodels.api import qqplot


# In[41]:


img = qqplot(residuals, fit=True, line='r', ax=axes[1,0])


# In[42]:


plt.savefig('qq_plot.png')


# In[43]:


ls -la


# In[45]:


pd.Series(y_test_reg_preds).to_csv('regression_results.csv')


# In[46]:


ls -la


# In[48]:


X_train


# In[ ]:


targ_pipe_reg.named_steps['PowerTransformer'].inverse_transform(y_test_reg_preds.reshape(-1,1)).squeeze()


# In[ ]:





# In[ ]:


# !pip install scikit-plot


# In[ ]:


import scikitplot as skplt


# In[ ]:


def evaluate_classifier(X, y, model):
    pass
    # print classification report
    # create lift charts
    # create gains charts


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf = RandomForestClassifier()


# In[ ]:


rf.fit(X_train_prepared, y_train_cl)


# In[ ]:


y_test_cl_preds = rf.predict_proba(X_test_prepared)


# In[ ]:


skplt.metrics.plot_lift_curve(y_test_cl, y_test_cl_preds);


# In[ ]:


np.sort(y_test_cl_preds)[:10]


# ## Make Classifcation Deciles

# In[ ]:


def output_deciles_class(model, X, y):
    results = pd.DataFrame(model.predict_proba(X)[:, 1], index=X.index, columns=['predictions'])
    results['actual'] = y.values
    results['deciles'] = pd.qcut(results['predictions'], 10, labels=False)
    results['contact_id'] = results.index.map(adviser_lookup)
    return results


# In[ ]:


class_results = output_deciles_class(rf, X_test_prepared, y_test_cl)


# In[ ]:


class_results.groupby('deciles')[['actual']].mean()


# In[ ]:


df.isnull().sum()


# In[ ]:




