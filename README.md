# 📊 Analyse des Taux d'Incidence de Xeljanz

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![GitHub](https://img.shields.io/github/stars/[votre-nom-utilisateur]/[nom-du-repo]?style=for-the-badge)

## 📋 Présentation du Projet

Ce projet est une application web interactive développée avec **Streamlit** et **Plotly**, conçue pour une analyse approfondie des données de sécurité du médicament **Xeljanz**. L'objectif principal est de visualiser de manière claire et interactive les taux d'incidence (TI) et les intervalles de confiance à 95 % (IC 95%) de divers effets indésirables liés aux différents dosages du traitement.

L'application intègre un **chatbot propulsé par OpenAI** pour permettre aux utilisateurs de poser des questions en langage naturel sur les données et d'obtenir des explications instantanées.

## ✨ Fonctionnalités Clés

* **📈 Graphique Forest Plot Interactif** : Une visualisation claire et dynamique qui permet de comparer les taux d'incidence entre les groupes de dosage.
* **🎚️ Filtres Dynamiques** : Les utilisateurs peuvent sélectionner les groupes de dosage et les effets indésirables pour personnaliser l'affichage du graphique.
* **📊 Résumé des Données en Temps Réel** : Un tableau de bord affiche des métriques clés (nombre d'effets, de groupes, etc.) en fonction des filtres appliqués.
* **💬 Chatbot Intégré (OpenAI)** : Un assistant conversationnel IA répond aux questions sur les données affichées dans l'application, fournissant des informations précises et basées sur le contexte.
* **🎨 Thèmes Personnalisables** : La possibilité de choisir entre des thèmes de couleurs variés et de basculer entre les modes clair et sombre.

## 💻 Technologies Utilisées

* **Python** : Le langage principal du projet.
* **Streamlit** : Pour le développement de l'application web interactive.
* **Plotly** : Pour la création du graphique Forest Plot.
* **Pandas** : Pour la manipulation et l'analyse des données.
* **NumPy** : Pour les calculs numériques.
* **OpenAI** : Pour l'intégration du chatbot et le traitement du langage naturel.

## 🚀 Lancement de l'Application en Local

Pour lancer l'application sur votre machine, suivez ces étapes simples.

**Prérequis :**
* Python 3.7+
* pip

1.  **Clonez le dépôt :**
    ```sh
    git clone [https://github.com/](https://github.com/)[votre-nom-utilisateur]/[nom-du-repo].git
    cd [nom-du-repo]
    ```

2.  **Créez le fichier de secrets :**
    Créez un dossier `.streamlit` et un fichier `secrets.toml` à l'intérieur, puis ajoutez votre clé API OpenAI.
    ```toml
    # .streamlit/secrets.toml
    openai_api_key = "votre_clé_api_openai"
    ```

3.  **Installez les dépendances :**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Lancez l'application :**
    ```sh
    streamlit run app.py
    ```

## 📂 Structure du Dépôt
