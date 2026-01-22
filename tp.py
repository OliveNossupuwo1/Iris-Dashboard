 
# Importation des bibliothèques de base nécessaires 
import os
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
# Chargement du jeu de données Iris     


df = pd.read_csv('Iris.csv', sep=';') 

# Dossier pour sauvegarder les figures
os.makedirs('figures', exist_ok=True)

# Afficher les premières lignes du jeu de données 
print(df.head()) 
# Statistiques descriptives pour comprendre la distribution des caractéristiques
print(df.describe()) 


###Exercice 1 : ###
# Visualisation de la répartition des classes 
"""sns.countplot(x='Species', data=df) 
#plt.title('Distribution des espèces d\'iris') 
#plt.show()"""

#afficher l'effectif de chacune des modalités de la variable Species
print(df['Species'].value_counts())
# Calculer les effectifs et définir une palette (une couleur par espèce)
counts = df['Species'].value_counts()
species = counts.index
palette = sns.color_palette('Set2', n_colors=len(species))

# Tracer les diagrammes avec une couleur par bande/espèce
counts.plot(kind='bar', color=palette)
plt.title('Effectif en histogramme des différentes espèces d\'iris')
plt.xlabel('Espèces')
plt.ylabel('Effectif')
plt.savefig('figures/species_count_bar.png', bbox_inches='tight')
plt.show()

counts.plot(kind='pie', autopct='%1.1f%%', colors=palette)
plt.title('Effectif en secteurs des différentes espèces d\'iris')
plt.ylabel('')
plt.savefig('figures/species_count_pie.png', bbox_inches='tight')
plt.show()

counts.plot(kind='bar', color=palette)
plt.title('Effectif en barres groupées des différentes espèces d\'iris')
plt.xlabel('Espèces')
plt.ylabel('Effectif')
plt.savefig('figures/species_count_bar2.png', bbox_inches='tight')
plt.show()

counts.plot(kind='barh', color=palette)
plt.title('Effectif en cascades des différentes espèces d\'iris')
plt.xlabel('Effectif')
plt.ylabel('Espèces')
plt.savefig('figures/species_count_barh.png', bbox_inches='tight')
plt.show()


###Exercice 2 : ####
# Définir les variables quantitatives (colonnes) à analyser
quant_vars = ['PetalLength', 'PetalWidth', 'SepalLength', 'SepalWidth']

# Palette pour les histogrammes (une couleur par variable)
palette_q = sns.color_palette('Set2', n_colors=len(quant_vars))

# 1) Résumés numériques pour chaque variable quantitative
for var in quant_vars:
	print(f"Résumé pour la variable {var} :")
	print(df[var].describe())
	print()

# 2) Histogramme pour PetalLength (avec KDE)
plt.figure(figsize=(8, 6))
g = sns.FacetGrid(df, col='Species', height=4)
g.map_dataframe(sns.scatterplot, x='PetalLength', y='PetalWidth')
g.set_axis_labels('PetalLength', 'PetalWidth')
g.add_legend()
g.fig.savefig('figures/facet_petal_by_species.png', bbox_inches='tight')
plt.show()
plt.show()

# 3) Réaliser le même type d'analyse pour les autres variables quantitatives
# Affichage regroupé : 2x2 sous-graphes
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for ax, var, col in zip(axes, quant_vars, palette_q):
	sns.histplot(df[var], kde=True, ax=ax, color=col, bins=15)
	ax.set_title(f'Histogramme de {var}')
	ax.set_xlabel(var)
	ax.set_ylabel('Effectif')

plt.tight_layout()
fig.savefig('figures/histograms_2x2.png', bbox_inches='tight')
plt.show()

### Exercice 3 : ###
# 1) Nuage de points PetalLength vs PetalWidth pour les 150 individus
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PetalLength', y='PetalWidth', hue='Species', palette=palette)
plt.title('Nuage de points : PetalLength vs PetalWidth (coloré par espèce)')
plt.xlabel('PetalLength')
plt.ylabel('PetalWidth')
plt.legend(title='Species')
plt.savefig('figures/scatter_petal.png', bbox_inches='tight')
plt.show()

# Calculer et afficher la corrélation entre PetalLength et PetalWidth
corr_petal = df['PetalLength'].corr(df['PetalWidth'])
print(f"Corrélation PetalLength vs PetalWidth : {corr_petal:.2f}")
if abs(corr_petal) >= 0.7:
	print("Interprétation : forte corrélation linéaire entre longueur et largeur du pétale.")
elif abs(corr_petal) >= 0.4:
	print("Interprétation : corrélation modérée.")
else:
	print("Interprétation : faible corrélation.")

# 2) Autre croisement choisi : SepalLength vs SepalWidth
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='SepalLength', y='SepalWidth', hue='Species', palette=palette)
plt.title('Nuage de points : SepalLength vs SepalWidth (coloré par espèce)')
plt.xlabel('SepalLength')
plt.ylabel('SepalWidth')
plt.legend(title='Species')
plt.savefig('figures/scatter_sepal.png', bbox_inches='tight')
plt.show()

corr_sepal = df['SepalLength'].corr(df['SepalWidth'])
print(f"Corrélation SepalLength vs SepalWidth : {corr_sepal:.2f}")
if abs(corr_sepal) >= 0.7:
	print("Interprétation : forte corrélation linéaire entre longueur et largeur du sépale.")
elif abs(corr_sepal) >= 0.4:
	print("Interprétation : corrélation modérée.")
else:
	print("Interprétation : faible corrélation.")

### Exercice 4 : BOXPLOT (variable qualitative vs quantitative)
# 1) Longueur des pétales par espèce
plt.figure(figsize=(8, 6))
sns.boxplot(x='Species', y='PetalLength', data=df, palette=palette)
plt.title('Boxplot de PetalLength par Species')
plt.xlabel('Species')
plt.ylabel('PetalLength')
plt.savefig('figures/boxplot_petal_by_species.png', bbox_inches='tight')
plt.show()

print('\nStatistiques de PetalLength par espèce:')
print(df.groupby('Species')['PetalLength'].describe())

# Interprétation courte automatique (médiane et dispersion)
medians = df.groupby('Species')['PetalLength'].median()
iqr = df.groupby('Species')['PetalLength'].quantile(0.75) - df.groupby('Species')['PetalLength'].quantile(0.25)
for s in medians.index:
	print(f"{s}: médiane={medians[s]:.2f}, IQR={iqr[s]:.2f}")

# 2) Autre variable quantitative: SepalLength par espèce
plt.figure(figsize=(8, 6))
sns.boxplot(x='Species', y='SepalLength', data=df, palette=palette)
plt.title('Boxplot de SepalLength par Species')
plt.xlabel('Species')
plt.ylabel('SepalLength')
plt.show()

print('\nStatistiques de SepalLength par espèce:')
print(df.groupby('Species')['SepalLength'].describe())

medians_s = df.groupby('Species')['SepalLength'].median()
iqr_s = df.groupby('Species')['SepalLength'].quantile(0.75) - df.groupby('Species')['SepalLength'].quantile(0.25)
for s in medians_s.index:
	print(f"{s}: médiane={medians_s[s]:.2f}, IQR={iqr_s[s]:.2f}")

### Exercice 5 : Analyse par espèce, corrélations et modèle de classification (KNN)

# 1) Représentations séparées par espèce et superposition
print('\nTracé des nuages de points séparés par espèce et pairplot global...')
# nuages séparés (facets)
g = sns.FacetGrid(df, col='Species', height=4)
g.map_dataframe(sns.scatterplot, x='PetalLength', y='PetalWidth')
g.set_axis_labels('PetalLength', 'PetalWidth')
g.add_legend()
plt.show()

# pairplot coloré par espèce (superpose l'information espèce)
p = sns.pairplot(df, vars=quant_vars, hue='Species', palette=palette)
p.fig.suptitle('Pairplot des variables quantitatives par Species', y=1.02)
p.fig.savefig('figures/pairplot_quant_vars.png', bbox_inches='tight')
plt.show()

# 2) Calculer les corrélations globales et par espèce
print('\nMatrice de corrélation globale (quantitatives):')
corr_global = df[quant_vars].corr()
print(corr_global)
plt.figure(figsize=(6, 5))
sns.heatmap(corr_global, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Corrélation globale entre variables quantitatives')
plt.gcf().savefig('figures/corr_global.png', bbox_inches='tight')
plt.show()

print('\nCorrélations par espèce:')
for s in df['Species'].unique():
	print(f"\nEspèce: {s}")
	corr_s = df[df['Species'] == s][quant_vars].corr()
	print(corr_s)
	plt.figure(figsize=(5, 4))
	sns.heatmap(corr_s, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
	plt.title(f'Corrélations - {s}')
	plt.gcf().savefig(f'figures/corr_{s}.png', bbox_inches='tight')
	plt.show()

# 3) Préparer les données pour le modèle (X, y), train/test, normalisation
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import numpy as np

print('\nPréparation des données pour le modèle...')
# Veiller à utiliser le nom correct de la colonne target
X = df.drop('Species', axis=1)
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4) Créer et entraîner un modèle KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 5) Évaluer le modèle
y_pred = knn.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred, labels=np.unique(y))
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Matrice de confusion - KNN')
plt.xlabel('Prédictions')
plt.ylabel('Vraies classes')
plt.gcf().savefig('figures/confusion_matrix_knn.png', bbox_inches='tight')
plt.show()

accuracy = accuracy_score(y_test, y_pred)
print(f"Exactitude du modèle KNN : {accuracy * 100:.2f}%")
print('Rapport de classification :')
print(classification_report(y_test, y_pred))

# 6) Optimisation simple des hyper-paramètres KNN (GridSearch)
param_grid = {'n_neighbors': list(range(1, 11)), 'p': [1, 2]}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
print(f"Meilleurs paramètres KNN (GridSearch) : {grid.best_params_}, score CV={grid.best_score_:.3f}")

# 7) Comparaison rapide avec d'autres modèles (entraînés sur le même split)
models = {
	'KNN': KNeighborsClassifier(n_neighbors=grid.best_params_['n_neighbors'], p=grid.best_params_['p']),
	'LogisticRegression': LogisticRegression(solver='lbfgs', max_iter=200),
	'DecisionTree': DecisionTreeClassifier(random_state=42),
	'GaussianNB': GaussianNB(),
	'SVM': SVC(kernel='rbf', probability=False),
	'MLP': MLPClassifier(max_iter=500, random_state=42)
}

results = {}
for name, model in models.items():
	model.fit(X_train, y_train)
	ypred = model.predict(X_test)
	acc = accuracy_score(y_test, ypred)
	results[name] = acc
	print(f"{name} accuracy: {acc:.3f}")

print('\nComparaison des modèles (accuracy sur test):')
for k, v in results.items():
	print(f"- {k}: {v:.3f}")

# Sauvegarder le meilleur modèle trouvé (GridSearch) et le scaler pour le déploiement
import pickle
final_model = grid.best_estimator_
with open('model.pkl', 'wb') as f:
	pickle.dump(final_model, f)
with open('scaler.pkl', 'wb') as f:
	pickle.dump(scaler, f)
print('\nModèle et scaler sauvegardés dans model.pkl et scaler.pkl')
