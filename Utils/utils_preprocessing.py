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
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from category_encoders.target_encoder import TargetEncoder

# Stats
from scipy.stats import spearmanr, f_oneway, pearsonr, chi2_contingency

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
        mis_val_percent = np.round(100 * df.isnull().sum() / len(df), 2)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table = pd.DataFrame({"Missing_values": mis_val, 
                                      "% of Total Values": mis_val_percent,
                                      "Feature_type": df.dtypes})

        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
               "There are " + str(mis_val_table[mis_val_table.Missing_values != 0].shape[0]) +
               " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table.sort_values(by="Missing_values", ascending=False)


def show_miss_values(df):

    df_miss_values = missing_values_table(df)

    fig = px.histogram(df_miss_values, x="% of Total Values", nbins=20)
    fig.update_layout(width=800, height=450, title="Missing values check", template="plotly_dark")
    fig.show()



def check_correlation_classif_task_continuous(input_df: pd.DataFrame, 
                                              target: str) -> pd.DataFrame:

   """This function print P Values after ANOVA test between a continuous features and target. 
      It does the same job for Pearson coefficient between numerical features dans target."""

   # Delete NaN and fixe target to str type.
   df_wo_nan = input_df[input_df[target] != np.nan]
   df_wo_nan[target] = df_wo_nan[target].apply(lambda x : str(x))

   # Split in categorical and numerical features.
   numerical_features = df_wo_nan.select_dtypes(exclude=["category","object"]).columns.tolist()

   # Initialize a dict to store results of ANOVA test.
   dict_anova = {}

   for i in numerical_features:
      df_courant = df_wo_nan[[i, target]]
      df_courant.dropna(inplace=True)

      CategoryGroupLists = df_courant.groupby(target)[i].apply(list)
      AnovaResults = f_oneway(*CategoryGroupLists)
      dict_anova[i] = AnovaResults[1]

   return pd.DataFrame(pd.Series(dict_anova), columns=["P-value"]).sort_values(by="P-value")



def check_correlation_classif_task_categorical(input_df: pd.DataFrame, 
                                               target: str) -> pd.DataFrame:

   """This function print P Values after ANOVA test between categorical features and target. 
      It does the same job for Pearson coefficient between numerical features dans target."""

   # Delete NaN and fixe target to str type.
   df_wo_nan = input_df[input_df[target] != np.nan]
   df_wo_nan[target] = df_wo_nan[target].apply(lambda x : str(x))

   # Split in categorical and numerical features.
   cat_features = df_wo_nan.select_dtypes(exclude=["float64","int64"]).columns.tolist()
   cat_features.remove(target)

   dict_test = {i: [] for i in cat_features}

   for i in cat_features:
      df_courant = df_wo_nan[[i, target]]
      df_courant.dropna(inplace=True)

      contingency_table = pd.crosstab(df_courant.TARGET, df_courant[i])
      n = contingency_table.sum().sum()
      chi2, p, dof, expected = chi2_contingency(contingency_table)
      v = chi2/n
      dict_test[i].extend([v, p])

   return pd.DataFrame(dict_test, index=["Cramer coeff", "Chi-deux P-value"]).T.sort_values(by="Cramer coeff", ascending=False)



def impute_cat_feature(X_train, X_test, y, threshold_deletion=0.5, strategy="most_frequent", encoder="OneHotEncoder"):

    """This function impute and encode the categorical features for the X given in argument.
       You can choose 3 different types of encoding: OneHotEncoder, OrdinalEncoder, TargerEncoder
       
       It returns, X_train and X_test encoded and imputed for categorical features"""

    # Split numerical/categorical features.
    cat_features = X_train.select_dtypes(exclude=["float64","int64"]).columns.tolist()
    numerical_features = X_train.select_dtypes(exclude=["category","object"]).columns.tolist()

    # Delete too sparse features for training set.
    serie_ratio_nan = (X_train[cat_features].isna().sum() / X_train.shape[0])
    new_cat_features = [i for i in cat_features if i not in serie_ratio_nan[serie_ratio_nan > threshold_deletion].index]

    # Imputation for training and testing set.
    imp_cat_features = SimpleImputer(missing_values=np.nan, strategy=strategy)
    df_imputed_cat_features_train = imp_cat_features.fit_transform(X_train[new_cat_features])
    df_imputed_cat_features_test = imp_cat_features.transform(X_test[new_cat_features])

    # Define encoder.
    if encoder == "OneHotEncoder":
        encoder_technique = OneHotEncoder(handle_unknown = 'ignore')
        df_cat_encoded_train = pd.DataFrame(encoder_technique.fit_transform(df_imputed_cat_features_train).toarray())
        df_cat_encoded_test = pd.DataFrame(encoder_technique.transform(df_imputed_cat_features_test).toarray())

    if encoder == "OrdinalEncoder":
        encoder_technique = OrdinalEncoder()
        df_cat_encoded_train = pd.DataFrame(encoder_technique.fit_transform(df_imputed_cat_features_train))
        df_cat_encoded_test = pd.DataFrame(encoder_technique.transform(df_imputed_cat_features_test))

    if encoder == "TargetEncoder":
        encoder_technique = TargetEncoder()
        df_cat_encoded_train = pd.DataFrame(encoder_technique.fit_transform(df_imputed_cat_features_train, y))
        df_cat_encoded_test = pd.DataFrame(encoder_technique.transform(df_imputed_cat_features_test))

    # Return Imputed and Encoded Df.
    return df_cat_encoded_train, df_cat_encoded_test