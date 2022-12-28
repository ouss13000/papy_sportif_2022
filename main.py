
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from joblib import dump, load
import xgboost as xgb

pages = ["Présentation", "Enjeux", "présentation des données","enrichissement des données", "Data Visualisation", "Modélisation",
         "analyse de meilleur modèle ", "strategie_de_pari", "Conclusion & préspectives"]
page = st.sidebar.radio("choisissez_la_page", pages)

if page == pages[0]:
    st.title("""# Papy_sportif # """)
    st.header("ce projet est présenté par: Ilan, Valentin, Nawal, Oussama")

    st.image("img.jpg")
    st.image("img2.jpg")


if page == pages[4]:
    df_atp_data = pd.read_csv("atp_data.csv")

    fig1 = plt.figure()

    top_winner = df_atp_data['Winner'].value_counts().head(10)
    sns.barplot(y=top_winner.index, x=top_winner.values);
    fig2 = plt.figure()
    top_loser = df_atp_data['Loser'].value_counts().head(10)
    sns.barplot(y=top_loser.index, x=top_winner.values);
    st.pyplot(fig1)
    st.pyplot(fig2)

if page == pages[4]:
    st.sidebar.header("les parameteres d'entrée")

    model = load('Best_XGBOOST.joblib')

if page == pages[5]:

    def user_input():
        name_of_first_player = st.sidebar.slider('nom_de_joueur_1 ')
        name_of_second_player = st.sidebar.slider('nom_de_joueur_2 ')
        data = {'name_of_first_player': name_of_first_player,
                'name_of_second_player': name_of_second_player}
        param = pd.DataFrame(data, index=[0])
        return param


    df = user_input()
    st.subheader('le match sera joué entre')
    st.write(df)