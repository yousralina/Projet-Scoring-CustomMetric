# Projet : Implémentation d'un modèle de scoring 

**Autor :** Louis BIRENHOLZ  
**Date :** 04/05/2020  
**Durée totale du projet :** 110 heures  
**Vidéo :** https://www.youtube.com/watch?v=RLGKGFu-9aE&feature=emb_title  

## Background du projet :

Pour ce projet, nous nous positionnons en tant que Data Scientist au sein d'une société financière qui propose des **crédits à la consommation** pour des personnes ayant peu ou pas du tout d'historique de prêt.

Notre entreprise souhaite développer **un modèle de scoring** de la probabilité de défaut de paiement du client. Aussi, nous voulons créer un **dashboard interactif** pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit.

On accorde une importance particulière à la phase de cleaning des données du fait du **déséquilibre des classes** sur les targets.

## Key points du projet :

- Cleaning & Undersampling de la classe majoritaire 
- Feature Engineering & OHE
- Imputation & Standardisation
- Classification avec Régression Logistique (baseline), RandomForestClassifier, LightGBM 
- Création d'une custom metric adaptée à la problématique métier (pénalisaition forte des faux négatifs)
- Oversampling avec **SMOTE**
- Interprétabilité des modèles avec **LIME** & **SHAP**
- Création d'un dashboard intéractif pour le client avec **Streamlit**, **Seaborn**, **Plotly** & **Pickle**  
- Déploiement du Dashboard via Docker et Google Cloud Platform

## Data :

Les données du projet sont disponibles ici : https://www.kaggle.com/competitions/home-credit-default-risk

## Dashboard :

Ce dashboard est disponible dans le Repo associé sur mon Github (https://github.com/Louis-bir/dashboard-project-credits.app)

## Overview performances :

Le modèle final que j'ai sélectionné donne les performances suivantes :

| Metrics        | Perf         |
| ------------- |:-------------:|
| Accuracy      | 0.945         | 
| Precision     | 0.882         |
| Recall        | 0.67          |  
| AUC score     |  0.91         |
