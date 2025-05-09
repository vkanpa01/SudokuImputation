import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import time
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

from scipy.stats import skew 
from scipy.stats import kurtosis

import warnings
warnings.filterwarnings("ignore")

##################################################################

# null_counter : how many null values per column?
def nc(df):
    null_counter = dict()
    for feat in df.columns:
        null_counter[feat] = df[feat].isna().sum()
            
    return null_counter


# calculates means for all columns in df (excluding nulls)
def mean_calc(df, nc):
    means = dict()
    for col in list(df.columns):
        if col in nc:
            nulls = nc[col]
        else:
            nulls = 0

        col_sum = df[col].sum()
        col_n = df.shape[0]-nulls
        means[col] = col_sum/col_n
        
    return means


# performs linear regression between the completely not-null subset of two columns in the dataframe
def slr(sub_df, model, impute=False):
    
    not_null_values = sub_df[~sub_df.isnull().any(axis=1)]
    c1 = not_null_values[not_null_values.columns[0]].values.reshape(-1, 1)
    c2 = not_null_values[not_null_values.columns[1]].values.reshape(-1, 1)

    model.fit(c1, c2)
    r_squared = model.score(c1, c2)
    
    if impute:
        reg = model.fit(c1, c2)
        null_values = sub_df[sub_df.isnull().any(axis=1)]
        
        return r_squared, null_values, reg
        
    else:
        return r_squared, not_null_values.shape[0]



def matrix_builder(df):
    r2_coefs = dict()
    missing_vals = dict()
    idx = 0
    
    for col_1 in df.columns:
        
        if idx % 10 == 0:
            print("{} columns complete.".format(idx))
            
        r2_vals = []
        data_size = []

        for col_2 in df.columns:

            sub_df = df[[col_1, col_2]]
            model = LinearRegression()

            r_squared, present_data = slr(sub_df, model)

            r2_vals.append(r_squared)
            data_size.append(present_data)

        r2_coefs[col_1] = r2_vals
        missing_vals[col_1] = data_size
        
        idx+=1

    print('\tBuilding adjacency matrix...')
    adj_matrix = pd.DataFrame(columns=df.columns, data = r2_coefs)
    print('\tBuilding data missingness matrix...')
    data_matrix = pd.DataFrame(columns=df.columns, data = missing_vals)
    return r2_coefs, adj_matrix, data_matrix



def centrality_calc(r2_coefs):
    centralities = dict()

    for feat_ in r2_coefs.keys():
        node_centrality = (sum(r2_coefs[feat_])-1)/47
        centralities[feat_] = node_centrality

    c_node = max(centralities, key=centralities.get)
        
    return c_node, centralities



def colinear_search(r2_coefs, c_node):
    c_node_r2 = r2_coefs[c_node]

    feat_names = list(r2_coefs.keys())

    zipped = zip(feat_names, c_node_r2)
    hit = sorted(list(zipped),key=lambda x: x[1])[::-1]#[1:2][0]
    
    return hit



def sudoku(np_centralities, r2_coefs, df, N_COMPS=10):

    '''
    imputed_feats = []
    '''
    
    for c_node in np_centralities[:,0]:

        '''
        # ----------------
        print("This iteration is on {}".format(c_node) + " which has an original node centrality of {}.".format(centralities[c_node]))
        r2_coefs_iter, _, __ = matrix_builder(df)
        most_central, centralities_iter = centrality_calc_mod(r2_coefs_iter)
        c_val = centralities_iter[most_central[
        if most_central != c_node and most_central not in imputed_feats:
            print("ALERT! The true most central feature is {}".format(most_central) + " which has an updated node centrality of {}.".format(c_val))
        # -------------------
        '''
        
        colinears = colinear_search(r2_coefs, c_node)
        colinears_dict = dict(colinears)
    
        COMP_THRESH = len({k:v for (k,v) in colinears_dict.items() if v > 0.9})
        
        if COMP_THRESH > N_COMPS:
            N_COMPS = COMP_THRESH
    
        colinears = np.array(colinears)
        t10 =colinears[0:N_COMPS,0]
        
        df_sub = df[t10]
        c_node_partners = np.array(colinears)[1:N_COMPS,0]
    
        all_null_count = sum(df_sub.T.isna().sum() == N_COMPS)
        total = df_sub.shape[0]
        print("{}% of rows are completely null across the top ".format(round(100*all_null_count/total, 2)) 
              + str(N_COMPS) + " colinear hits in {}.".format(c_node))
        
        for partner in c_node_partners:
            temp_pair = df[[c_node, partner]]
        
            t1 = temp_pair[temp_pair[c_node].isna()]
            
            impute_set = t1.dropna(subset=[partner])
            regress_set = temp_pair.dropna()
        
            # x = partner
            x = regress_set.values[:,1].reshape(-1, 1)
            
            # y = c_node
            y = regress_set.values[:,0].reshape(-1, 1)
            
            model = LinearRegression()
            model.fit(x, y)
            
            # model.coef_ * x[20] + model.intercept_
            
            impute_set['ind'] = impute_set.index
            impute_partners = impute_set[[partner, 'ind']].values
            
            for pair in impute_partners:
                impute_val = model.coef_[0] * pair[0] + model.intercept_[0]
                df.loc[pair[1], c_node] = impute_val
        '''
        impute_feats.append(c_node)
        '''
    return df


def sudoku_imputation(df, N_COMPS=40):
    print("Beginning sudoku imputation with a max partner selection of {}".format(N_COMPS)) 
    total_nc = nc(df)
    print('Total nullity calculations complete.')
    means = mean_calc(df, total_nc)
    print('Point statistic calculations complete.')
    print('Adjacency matrix under construction...')
    r2_coefs, adj_matrix, data_matrix = matrix_builder(df)
    print('Adjacency matrix construction complete.')
    print('Network centrality computation underway...')
    c_node, centralities = centrality_calc(r2_coefs)
    print('Network centrality computation complete.')
    np_centralities = np.array(list(centralities.items()))
    np_centralities = np.array(pd.DataFrame(np_centralities).astype({1: 'float64'}).sort_values(by=1, ascending=False))

    print('Performing sudoku imputation...')
    df = sudoku(np_centralities, r2_coefs, df)

    # FALL BACK STRATEGY IF NULLS ARE LEFT UNHANDLED
    remaining_nulls = df.isna().sum().sum()
    if remaining_nulls > 0:
        print("There are {} values NOT handled by Sudoku imputation. Mean values were used instead.".format(remaining_nulls))
        df = df.fillna(df.mean())

    return [r2_coefs, adj_matrix, data_matrix], total_nc, df