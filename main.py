import json
from datetime import time

import requests
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_validate, TimeSeriesSplit, GridSearchCV

pages = ["Le projet PaPy_Sportif", "Présentation du 1er dataset", "Enrichissement du dataset",
         "Methodologie & Modélisation", "Prédiction & Stratégie de paris", "Conclusion & Perspectives"]
page = st.sidebar.radio("choisissez_la_page", pages)

if page == pages[0]:
    st.title("""# Papy_sportif # """)
    st.header("une petite phrase")
    st.header("ce projet est présenté par: Ilan, Valentin, Nawal, Oussama")

    st.image("img.jpg")
    st.image("img2.jpg")

    # url linkdin
    st.sidebar.image("linkdin.png")
    st.sidebar.header("linkdin SEGHIR OUSSAMA: " + "https://www.linkedin.com/in/oussama-seghir-75039612b/")
    st.sidebar.header("DELAHAYE VALENTIN: ")
    st.sidebar.header("linkdin HAMCHAOUI NAWAL: ")
    st.sidebar.header("linkdin CUKROWICZ ILAN: ")
if page == pages[1]:
    st.sidebar.image("linkdin.png")
    st.sidebar.header("linkdin SEGHIR OUSSAMA: " + "https://www.linkedin.com/in/oussama-seghir-75039612b/")
    st.sidebar.header("DELAHAYE VALENTIN: ")
    st.sidebar.header("linkdin HAMCHAOUI NAWAL: ")
    st.sidebar.header("linkdin CUKROWICZ ILAN: ")

if page == pages[2]:
    st.sidebar.image("linkdin.png")
    st.sidebar.header("linkdin SEGHIR OUSSAMA: " + "https://www.linkedin.com/in/oussama-seghir-75039612b/")
    st.sidebar.header("DELAHAYE VALENTIN: ")
    st.sidebar.header("linkdin HAMCHAOUI NAWAL: ")
    st.sidebar.header("linkdin CUKROWICZ ILAN: ")
# Modelisation
if page == pages[3]:

    st.title("Methodologie")

    st.sidebar.header("linkdin SEGHIR OUSSAMA: " + "https://www.linkedin.com/in/oussama-seghir-75039612b/")
    st.sidebar.header("DELAHAYE VALENTIN: ")
    st.sidebar.header("linkdin HAMCHAOUI NAWAL: ")
    st.sidebar.header("linkdin CUKROWICZ ILAN: ")
    # texte
    st.text("*Pay_sportif est un projet qui traite une problematique de classification\n"
            "*Pay_sportif à pour but de battre les algorithmes de BOOKMAKERS\n"
            "*le travail a été fait sur 3data_set.\n")
    st.write("les deux BOOKMAKERS sont")
    # centrer la photo
    col1, col2, col3 = st.columns([1, 10, 1])

    with col1:
        st.write("")

    with col2:
        st.image("bookmakers.png")

    with col3:
        st.write("")
        # tracer le graphe accurarcy bookmakers
        df_book = pd.read_csv('df_final_rolling_ready_100_with_bookodds.csv')
        df_book = df_book[['player_1_win', 'player_1_B365', 'player_2_B365', 'player_1_PS', 'player_2_PS']]
    fig1 = plt.figure()
    train_sizes = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    B365_acc = []
    PS_acc = []
    for train_size in train_sizes:
        nb_rows_train = int(round(len(df_book) * train_size, 0))
        df = df_book[nb_rows_train:]
        B365_acc.append(len(df[((df['player_1_win'] == 1) & (df['player_1_B365'] < df['player_2_B365'])) |
                               ((df['player_1_win'] == 0) & (df['player_2_B365'] < df['player_1_B365']))]) / len(df))
        PS_acc.append(len(df[((df['player_1_win'] == 1) & (df['player_1_PS'] < df['player_2_PS'])) |
                             ((df['player_1_win'] == 0) & (df['player_2_PS'] < df['player_1_PS']))]) / len(df))
    plt.plot(train_sizes, B365_acc, 'b--', label='B635')
    plt.plot(train_sizes, PS_acc, 'g--', label='Pinnacle')
    plt.legend()
    plt.title('Accuracy des bookmakers sur la période de test en fonction du train size')
    plt.xlabel('train_size')
    plt.ylabel('accuracy')
    plt.show()
    st.pyplot(fig1)
    st.write('accuracy de nos deux bookmakers calculées à partir du jeu de test '
             '(30% de l’ensemble desdonnées), soit 8600 matchs, à savoir :\n'
             '- 66.35% pour B365 \n'
             '- 67.47% pour Pinnacle.')
    st.title("Modelisation")

    # st.image("bagarre.jpg",width = 150)
    st.text("-Data_set1: c'est le data_set du Kaggle à l'état brut\n"
            "-Data_set2: integration de la methode Rolling sans les cotes des BOKKMAKERS\n"
            "-Data_set3: est le data_set2 + cotes des BOOKMAKERS")
    st.image("organigramme_1.PNG")
    st.image("accurarcy_models_en_fct_rolling.PNG")
    st.image("organigramme2.PNG")
    # uploaded_file = st.file_uploader("Uploadyour file here please")
    # if uploaded_file:
    # st.header('detail des scores pour chaque model')
    # df = pd.read_csv(uploaded_file, sep=';', encoding='latin-1')
    # st.write(df)

    df_score_naif = pd.read_csv("score_modele_naif.csv", sep=';', encoding='latin-1')

    df_score_cross_val_tempo = pd.read_csv("score_cross_val_tempo.csv", sep=';', encoding='latin-1')

    df_book_odd = pd.read_csv("score_voting_classfier.csv", sep=';', encoding='latin-1')

    option = st.selectbox('choose data_set please',
                          ('df_score_naif', 'df_score_cross_val_tempo', 'df_book_odd'))
    st.write('You selected:', option)

    if option == 'df_score_naif':
        df_score_naif
    if option == 'df_score_cross_val_tempo':
        df_score_cross_val_tempo
    if option == 'df_book_odd':
        df_book_odd
    # Demo
    df_final = pd.read_csv("df_final_rolling_ready_50_with_bookodds.csv", sep=';', encoding='latin-1')
    match = df_final.iloc[20067:28666, :]

    option = st.selectbox('choose the match please', match)
    st.write('You selected:', option)
    game = match[match['matchs'] == option]
    game

    model_Best_GBC = load('Best_GBC.joblib')
    model_Best_KNN = load('Best_KNN.joblib')
    model_Best_LR = load('Best_LR.joblib')
    # model_Best_RF = load('Best_RF.joblib')
    model_Best_SVM = load('Best_SVM.joblib')
    model_Best_VC_LR_GBC = load('Best_VC_LR_GBC.joblib')
    model_Best_XGBOOST = load('Best_XGBOOST.joblib')
    original_list = ["model_Best_GBC", "model_Best_KNN", "model_Best_LR", "model_Best_SVM", "model_Best_VC_LR_GBC",
                     "model_Best_XGBOOST"]

    options = st.selectbox('choose the model please', original_list)
    st.write('You selected:', options)
    if options == model_Best_GBC:
        pred = model_Best_GBC.predict(game)

    if options == model_Best_KNN:
        pred = model_Best_KNN.predict(game)

    if options == model_Best_LR:
        pred = model_Best_LR.predict(game)

    if options == model_Best_SVM:
        pred = model_Best_SVM.predict(game)

    if options == model_Best_VC_LR_GBC:
        pred = model_Best_VC_LR_GBC.predict(game)

    if options == model_Best_XGBOOST:
        pred = model_Best_XGBOOST.predict(game)

if page == pages[4]:
    # strategie de pari
    st.title("Prédictions")
    st.header("Détails des prédictions de nos modèles")

    model_choice = st.selectbox("Choisissez un modèle de prédiction", (
        "Voting Classifier (recommandé)", "K plus proches voisins", "Random Forest", "Régression Logistique"))
    st.write('Vous avez choisi : ', model_choice)
    # Choix de l'index du match : retourne les infos d'une ligne du DF : noms des joueurs, cotes, prediction proba

    match_choice = st.text_input("Entrez l'index d'un match (0 à 8600)", 0)
    st.write('Match séléctionné : #', match_choice)
    st.write('Détails du match')

    value_bets = load('value_bets.joblib')
    st.title("Stratégie de Paris")
    st.header("Choix du meilleur modèle")
    st.write(
        'Tableau récapitulatif des résultats en misant 1€ sur chaque côte à espérence de gain positive selon le modèle')
    st.image('recap_strat_papy.png')
    st.write('Le Voting Classifier nous donne le profit et maximal sur un volume très conséquent')

    st.header("Estimation des gains")
    bet_choice = st.text_input("Entrez la somme en € que vous êtes prêt à parier sur chaque match")
    st.write(
        f"En pariant {bet_choice}€ sur chaque match considéré comme rentable par le modèle, votre profit total sur 1 "
        f"an est estimé à {round(value_bets['money_won'].sum() / 4, 2)}")
    st.write(
        f"Probabilité qu'un pari soit gagnant : {round(1 - (value_bets[value_bets['money_won'] < 0].shape[0] / value_bets.shape[0] * 100, 2))}%")
    st.write(
        f"Probabilité qu'un pari soit perdant : {round(value_bets[value_bets['money_won'] < 0].shape[0] / value_bets.shape[0] * 100, 2)}%")
    st.write(
        f"NB : Pour un risque de ruine inférieur à 1%, nous vous recommandons d'avoir un capital de {bet_choice * 180}€")

    st.sidebar.image("linkdin.png")
    st.sidebar.header("linkdin SEGHIR OUSSAMA: " + "https://www.linkedin.com/in/oussama-seghir-75039612b/")
    st.sidebar.header("DELAHAYE VALENTIN: ")
    st.sidebar.header("linkdin HAMCHAOUI NAWAL: ")
    st.sidebar.header("linkdin CUKROWICZ ILAN: ")

if page == pages[5]:
    st.sidebar.image("linkdin.png")
    st.sidebar.header("linkdin SEGHIR OUSSAMA: " + "https://www.linkedin.com/in/oussama-seghir-75039612b/")
    st.sidebar.header("DELAHAYE VALENTIN: ")
    st.sidebar.header("linkdin HAMCHAOUI NAWAL: ")
    st.sidebar.header("linkdin CUKROWICZ ILAN: ")

    df = pd.read_csv("df_final_rolling_ready_50_with_bookodds.csv", sep=";")
    # st.dataframe(df.head())
    st.sidebar.header("les parameteres d'entrée")
    option = st.sidebar.selectbox("choisissez un match s il vous plait", df["matchs"])

    st.sidebar.write("votre selection", option)
