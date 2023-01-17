#### Librairies ####

# Classic.
import pandas as pd

# Data viz.
import plotly.express as px

# ML and metrics.
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import f1_score, recall_score, roc_auc_score








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



####### Custom functions - MODELIZATION #######

def quality(y_test, y_pred, y_prob):
    
    """
    Cette fonction retourne différentes mesures de qualité pour la prédiction effectuée
    """
    
    # Justesse.
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_prob)
    
    # Print des métriques.
    print('Accuracy : {}'.format(accuracy))
    print('Precision : {}'.format(precision))
    print('Recall : {}'.format(recall))
    print('Score F1 : {}'.format(F1))
    print('AUC score : {}'.format(AUC))