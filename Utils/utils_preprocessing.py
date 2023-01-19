#### Librairies ####

# Classic.
import pandas as pd
import numpy as np

# Data viz.
import plotly.express as px

# ML and metrics.
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import f1_score, recall_score, roc_auc_score

# Stats
from scipy.stats import spearmanr, f_oneway, pearsonr

# Others.
from typing import List

####### Custom functions - PREPROCESSING #######

def missing_values_table(df: pd.DataFrame) -> pd.DataFrame :

        """
        Add doc string here
        """

        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


def show_miss_values(df):

    df_miss_values = missing_values_table(df)

    fig = px.histogram(df_miss_values, x="% of Total Values", nbins=20)
    fig.update_layout(width=800, height=450, title="Missing values check")
    fig.show()



def check_correlation_classif_task(input_df: pd.DataFrame, 
                                   target: str, 
                                   threshold_categ: float = 0.001,
                                   threshold_num: float = 0.8,
                                   verbosity: int = 0) -> None :

   """This function print P Values after ANOVA test between categorical features and target. 
      It does the same job for Pearson coefficient between numerical features dans target."""

   # Delete NaN and fixe target to str type.
   df_wo_nan = input_df[input_df[target] != np.nan]
   df_wo_nan[target] = df_wo_nan[target].apply(lambda x : str(x))

   # Split in categorical and numerical features.
   cat_features = df_wo_nan.select_dtypes(exclude=["float64","int64"]).columns.tolist()
   numerical_features = df_wo_nan.select_dtypes(exclude=["category","object"]).columns.tolist()
   cat_features.remove(target)

   dict_anova = {}

   for i in numerical_features:
      df_courant = df_wo_nan[[i, target]]
      df_courant.dropna(inplace=True)

      CategoryGroupLists = df_courant.groupby(target)[i].apply(list)
      AnovaResults = f_oneway(*CategoryGroupLists)
      dict_anova[i] = AnovaResults[1]

   return pd.DataFrame(pd.Series(dict_anova), columns=["P value"]).sort_values(by="P value")