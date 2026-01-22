import streamlit as st
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

st.set_page_config(page_title="Iris Dashboard", layout="wide")

st.title('Iris - Dashboard Complet')

# Charger les données
df = pd.read_csv('Iris.csv', sep=';')
quant_vars = ['PetalLength', 'PetalWidth', 'SepalLength', 'SepalWidth']
# Palette rouge/blanc
palette = ['#DC143C', '#FF6B6B', '#FFB3B3']  # Rouge foncé, rouge moyen, rouge clair

# Charger le modèle et scaler
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# Sidebar pour la prédiction
st.sidebar.header('Prédiction')
sepal_length = st.sidebar.slider('SepalLength', float(df['SepalLength'].min()), float(df['SepalLength'].max()), float(df['SepalLength'].mean()))
sepal_width = st.sidebar.slider('SepalWidth', float(df['SepalWidth'].min()), float(df['SepalWidth'].max()), float(df['SepalWidth'].mean()))
petal_length = st.sidebar.slider('PetalLength', float(df['PetalLength'].min()), float(df['PetalLength'].max()), float(df['PetalLength'].mean()))
petal_width = st.sidebar.slider('PetalWidth', float(df['PetalWidth'].min()), float(df['PetalWidth'].max()), float(df['PetalWidth'].mean()))

if st.sidebar.button('Prédire'):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    st.sidebar.success(f'Prédiction: {prediction}')

# Onglets principaux
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Aperçu des données",
    "Distribution des espèces",
    "Variables quantitatives",
    "Nuages de points & Corrélations",
    "Boxplots & Analyse par espèce"
])

# ==================== TAB 1: Aperçu des données ====================
with tab1:
    st.header('Aperçu des données')

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Premières lignes')
        st.dataframe(df.head(10))
    with col2:
        st.subheader('Statistiques descriptives')
        st.dataframe(df.describe())

    st.subheader('Filtrer par espèce')
    species_filter = st.multiselect('Species', options=df['Species'].unique(), default=list(df['Species'].unique()))
    if species_filter:
        filtered = df[df['Species'].isin(species_filter)]
        st.dataframe(filtered.describe())

# ==================== TAB 2: Distribution des espèces ====================
with tab2:
    st.header('Distribution des espèces (Exercice 1)')

    counts = df['Species'].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Histogramme des effectifs')
        fig, ax = plt.subplots(figsize=(8, 5))
        counts.plot(kind='bar', color=palette, ax=ax)
        ax.set_title('Effectif des différentes espèces d\'iris')
        ax.set_xlabel('Espèces')
        ax.set_ylabel('Effectif')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:
        st.subheader('Diagramme en secteurs')
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        counts.plot(kind='pie', autopct='%1.1f%%', colors=palette, ax=ax2)
        ax2.set_title('Répartition des espèces d\'iris')
        ax2.set_ylabel('')
        st.pyplot(fig2)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader('Barres horizontales')
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        counts.plot(kind='barh', color=palette, ax=ax3)
        ax3.set_title('Effectif en barres horizontales')
        ax3.set_xlabel('Effectif')
        ax3.set_ylabel('Espèces')
        st.pyplot(fig3)

    with col4:
        st.subheader('Effectifs')
        st.write(counts)

# ==================== TAB 3: Variables quantitatives ====================
with tab3:
    st.header('Analyse des variables quantitatives (Exercice 2)')

    st.subheader('Résumés numériques')
    for var in quant_vars:
        with st.expander(f"Résumé pour {var}"):
            st.write(df[var].describe())

    st.subheader('Histogrammes des 4 variables')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    palette_q = ['#DC143C', '#FF6B6B', '#FFB3B3', '#FF4444']  # Palette rouge pour histogrammes
    for ax, var, col in zip(axes, quant_vars, palette_q):
        sns.histplot(df[var], kde=True, ax=ax, color=col, bins=15)
        ax.set_title(f'Histogramme de {var}')
        ax.set_xlabel(var)
        ax.set_ylabel('Effectif')
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader('FacetGrid: Petal par espèce')
    fig_facet = plt.figure(figsize=(12, 4))
    g = sns.FacetGrid(df, col='Species', height=4)
    g.map_dataframe(sns.scatterplot, x='PetalLength', y='PetalWidth')
    g.set_axis_labels('PetalLength', 'PetalWidth')
    g.add_legend()
    st.pyplot(g.fig)

# ==================== TAB 4: Nuages de points & Corrélations ====================
with tab4:
    st.header('Nuages de points & Corrélations (Exercices 3 & 5)')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('PetalLength vs PetalWidth')
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df, x='PetalLength', y='PetalWidth', hue='Species', palette=palette, ax=ax1)
        ax1.set_title('Nuage de points : PetalLength vs PetalWidth')
        ax1.legend(title='Species')
        st.pyplot(fig1)

        corr_petal = df['PetalLength'].corr(df['PetalWidth'])
        st.metric("Corrélation", f"{corr_petal:.3f}")
        if abs(corr_petal) >= 0.7:
            st.info("Forte corrélation linéaire")
        elif abs(corr_petal) >= 0.4:
            st.info("Corrélation modérée")
        else:
            st.info("Faible corrélation")

    with col2:
        st.subheader('SepalLength vs SepalWidth')
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df, x='SepalLength', y='SepalWidth', hue='Species', palette=palette, ax=ax2)
        ax2.set_title('Nuage de points : SepalLength vs SepalWidth')
        ax2.legend(title='Species')
        st.pyplot(fig2)

        corr_sepal = df['SepalLength'].corr(df['SepalWidth'])
        st.metric("Corrélation", f"{corr_sepal:.3f}")
        if abs(corr_sepal) >= 0.7:
            st.info("Forte corrélation linéaire")
        elif abs(corr_sepal) >= 0.4:
            st.info("Corrélation modérée")
        else:
            st.info("Faible corrélation")

    st.subheader('Matrice de corrélation globale')
    col1, col2 = st.columns([1, 2])

    with col1:
        corr_global = df[quant_vars].corr()
        fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr_global, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax_corr)
        ax_corr.set_title('Corrélation globale')
        st.pyplot(fig_corr)

    with col2:
        st.subheader('Corrélations par espèce')
        species_choice = st.selectbox('Sélectionner une espèce', df['Species'].unique())
        corr_s = df[df['Species'] == species_choice][quant_vars].corr()
        fig_corr_s, ax_corr_s = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr_s, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax_corr_s)
        ax_corr_s.set_title(f'Corrélations - {species_choice}')
        st.pyplot(fig_corr_s)

    st.subheader('Pairplot complet')
    fig_pair = sns.pairplot(df, vars=quant_vars, hue='Species', palette=palette)
    fig_pair.fig.suptitle('Pairplot des variables quantitatives par Species', y=1.02)
    st.pyplot(fig_pair.fig)

# ==================== TAB 5: Boxplots & Analyse par espèce ====================
with tab5:
    st.header('Boxplots & Analyse par espèce (Exercice 4)')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Boxplot PetalLength par espèce')
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='Species', y='PetalLength', data=df, palette=palette, ax=ax1)
        ax1.set_title('Boxplot de PetalLength par Species')
        st.pyplot(fig1)

        st.write("**Statistiques PetalLength par espèce:**")
        st.dataframe(df.groupby('Species')['PetalLength'].describe())

    with col2:
        st.subheader('Boxplot SepalLength par espèce')
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='Species', y='SepalLength', data=df, palette=palette, ax=ax2)
        ax2.set_title('Boxplot de SepalLength par Species')
        st.pyplot(fig2)

        st.write("**Statistiques SepalLength par espèce:**")
        st.dataframe(df.groupby('Species')['SepalLength'].describe())

    st.subheader('Boxplots pour toutes les variables')
    var_choice = st.selectbox('Variable à analyser', quant_vars)
    fig_box, ax_box = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Species', y=var_choice, data=df, palette=palette, ax=ax_box)
    ax_box.set_title(f'Boxplot de {var_choice} par Species')
    st.pyplot(fig_box)

    st.write(f"**Médianes et IQR pour {var_choice}:**")
    medians = df.groupby('Species')[var_choice].median()
    iqr = df.groupby('Species')[var_choice].quantile(0.75) - df.groupby('Species')[var_choice].quantile(0.25)
    stats_df = pd.DataFrame({'Médiane': medians, 'IQR': iqr})
    st.dataframe(stats_df)

st.sidebar.markdown("---")
st.sidebar.info("Dashboard Iris - Tous les exercices du TP")
