#### Librairies ####

# Classic.
import pandas as pd
import numpy as np

# Data viz.
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# ML and metrics.
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import f1_score, recall_score, roc_auc_score


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



def confusion_matrix(y_test, y_prediction):
    
    """
    Cette fonction retourne une matrice de confusion.
    """
    
    # Création de la matrice de confusion.
    df_matrice_confusion = pd.DataFrame(columns=['Predicted Negative','Predicted Positive'], 
                                        index=['Real Negative','Real Positive'])

    # DataFrame de comparaison.
    df_pred_compare = pd.concat([pd.Series(y_test.reset_index(drop=True)), pd.Series(y_prediction)], axis=1)
    df_pred_compare.columns = ['Real category', 'Prediction']
    
    # Masque suivant les tp,tn, fp...
    mask_real_pos = (df_pred_compare['Real category']==1)
    mask_pred_pos = (df_pred_compare['Prediction']==1)

    mask_real_neg = (df_pred_compare['Real category']==0)
    mask_pred_neg = (df_pred_compare['Prediction']==0)
    
    # Négatif.
    true_negative = df_pred_compare[mask_real_neg & mask_pred_neg].shape[0]
    false_negative = df_pred_compare[mask_real_pos & mask_pred_neg].shape[0]

    # Positif.
    false_positive = df_pred_compare[mask_real_neg & mask_pred_pos].shape[0]
    true_positive = df_pred_compare[mask_real_pos & mask_pred_pos].shape[0]

    # Remplissage de la matrice.
    df_matrice_confusion['Predicted Negative'] = ["{} (TN)".format(true_negative), "{} (FN)".format(false_negative)]
    df_matrice_confusion['Predicted Positive'] = ["{} (FP)".format(false_positive), "{} (TP)".format(true_positive)]
    
    return df_matrice_confusion


def proba_seuil(liste_proba, seuil):

    """
    Cette fonction classe un individu dans la classe positive suivant un seuil donné en entrée.
    """
    
    classe = []
    
    for i in liste_proba:
        if i>=seuil:
            classe.append(1)
        else:
            classe.append(0)
            
    return classe


def custom_metric(y_test, y_prediction):
    
    # Score avec un classifieur aléatoire.
    baseline = 11136
    
    # Score avec le meilleur classifieur.
    best = 97512
    
    # Calcul du nombre de TN & FN.
    df_pred_compare = pd.concat([pd.Series(y_test.reset_index(drop=True)), pd.Series(y_prediction)], axis=1)
    df_pred_compare.columns = ['Real category', 'Prediction']
    
    mask1 = (df_pred_compare['Real category']==1)
    mask2 = (df_pred_compare['Prediction']==0)
    mask3 = (df_pred_compare['Real category']==0)
    
    TN = df_pred_compare[mask2 & mask3].shape[0]
    FN = df_pred_compare[mask1 & mask2].shape[0]
    
    # Calcul du score du classifieur.
    score = 2*TN + (-10)*FN
    
    return (score-baseline)/(best-baseline)



def score_seuil(prob_pos):

    my_score = []
    
    # Parcours des seuils.
    for i in np.arange(0.0001,1.1,0.025):

        predd = proba_seuil(prob_pos, i)

        my_score.append(custom_metric(Y_validation, predd))
    
    return my_score



def viz_p7(y_test, y_pred_proba, model_name):
    
    """
    Cette fonction retourne une visualisation ROC CURVE aisni qu'une viz du score construit en fonction du 
    seuil de classification
    """
    
    # Calcul des taux et des seuils.
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_pred_proba)
    
    plt.figure(figsize = (24, 18))
    
    # Premier subplot.
    plt.subplot(2, 2, 1)
    
    plt.title('ROC Curve', weight='bold', size=20)
    
    plt.plot(list_sampled(false_positive_rate),
                 list_sampled(true_positive_rate), label="{}".format(model_name))
    
    # Noms des axes
    plt.xlabel('Taux de faux positifs', size=16, weight='bold')
    plt.ylabel('Taux de vrais postifis', size=16, weight='bold')
    
    # Affichage de la légende.
    plt.legend(loc="lower right")
    
    # Deuxième subplot.
    plt.subplot(2,2,2)
    
    plt.title("Score 'métier' en fonction du seuil", weight='bold', size=20)
    plt.xlabel('Treshold', size=16, weight='bold')
    plt.ylabel('Score', size=16, weight='bold')
    sns.lineplot(x=np.arange(0.0001,1.1,0.025), y=score_seuil(y_pred_proba), color = 'blue', marker='o')