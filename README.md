# Data Mining Projects 📊

This repository contains multiple projects covering several important topics in Data Mining.

## About ℹ️

> Under The Supervision of [Prof. Ehsan Nazerfard](https://scholar.google.com/citations?user=Cl5tre8AAAAJ&hl=en) 👨‍🏫
>
> Spring 2023 🌸

## 📁 Repository Structure

```
Data-Mining-Projects/
├── Preproccesing Data/              # Project 1: Data Preprocessing
├── Classification And Regression/   # Project 2: Regression & Classification
├── Kmeans Algorithm/                # Project 3: K-means clustering
├── Final Project - Persian Spotify/ # Project 4: Final project
└── README.md
```
Each folder contains its own Jupyter notebook(s) and, where applicable, its dataset.

---

## 1. Data Preprocessing 🧹

The objective of this project is to employ a variety of preprocessing techniques to showcase the significance of comprehending, cleansing, and refining the raw dataset. The considered aspects encompass:

1.  Managing NaN values ❓
2.  Processing non-numeric data through Label Encoding and One Hot Encoding 🔤➡️🔢
3.  Implementing Data Augmentation 📈
4.  Utilizing Upsampling and Downsampling methods ⚖️
5.  Applying Smotetomek and Smoteenn approaches 🔄
6.  Normalizing the data 📊
7.  Conducting Principal Component Analysis (PCA) 📉
8.  Creating plots and visualizations 📈📊

**Libraries:** `Scikit-learn`, `Pandas`, `Imbalanced-learn`, `Matplotlib` 🐍

**About the Dataset:**
The dataset considered for this project is **Palmer Penguin** 🐧. This collection was collected to identify three different breeds of penguins (Adelie, Gentoo and Chinstrap). There are 7 features for each penguin.

---

## 2. Regression and Classification 📈📊

The objective of this project is to demonstrate the deployment of various machine learning techniques on housing price data, illustrating the application and impact of both classification and regression methods.

1.  Q-box analysis 📦
2.  Comparison between Linear Regression and Polynomial Regression ➕➗📈
3.  Calculation of Mean Squared Error 📐
4.  Classification methods such as Decision Trees 🌳, Random Forests 🌲🌲🌲, K-Nearest Neighbors (KNN) 👨‍👩‍👧‍👦, Linear and Non-Linear Support Vector Machines (SVM) ⚔️
5.  Multi-class classification employing Deep Learning techniques 🧠🤖
6.  Utilization of a Confusion Matrix 🤔✅❌

**Libraries:** `Scikit-learn`, `Tensorflow`, `Pandas`, `Numpy`, `Matplotlib` 🐍

**About the Dataset:** The dataset considered for this project is the **House Price Prediction** (houseprice.csv) 🏠. This collection includes the characteristics of the area, the number of rooms, having parking, storage, elevator, address and the price of the house corresponding to them.

---

## 3. Kmeans Algorithm 🎯

The target of this project is to gain insight into clusters through practical exploration and to create clusters using the Python language.

1.  Generate a Similarity matrix using Cosine Similarity and Euclidean distance. 📐
2.  Implementation of the K-means algorithm

**Result:**

![C1](https://github.com/Amirbehnam1009/Linear-Algebra-Projects/assets/117163007/c5c0dce9-ccf7-4763-a40c-22c26907d624)
![C2](https://github.com/Amirbehnam1009/Linear-Algebra-Projects/assets/117163007/14ddba42-ff2d-487e-9ca5-c0865d097e9c)
![C3](https://github.com/Amirbehnam1009/Linear-Algebra-Projects/assets/117163007/8ef58f7a-2183-4e21-b3bb-634332eee9d4)
![C4](https://github.com/Amirbehnam1009/Linear-Algebra-Projects/assets/117163007/acb65c31-5f4e-4f2e-9868-437e10b9cbe1)

**Libraries:** `Scikit-learn`, `matplotlib`, `numpy` 🐍

---

## 4. Final Project: Persian Spotify 🎵🇮🇷

A project aimed at making various predictions using a dataset of Persian music from Spotify (`Spotfiy_Persian_Artists.csv`).

1.  **Data analysis & EDA** — song counts and average duration per artist, trends in duration/loudness/audio attributes (acousticness, danceability, energy, speechiness, liveness, valence) over release year, top-10 tracks/artists by popularity, correlation heatmap, and feature boxplots. 🔍📊
2.  **Preprocessing** — categorical NaNs filled with `"None"`, numerical NaNs (`popularity`, `album_total_tracks`) filled with the **median** (chosen for robustness to outliers), categorical fields label-encoded, features standardized, and dimensionality visualized via 2D/3D PCA. 🧹
3.  **Regression** — predicts `popularity` using 4 selected features (`energy`, `danceability`, `speechiness`, `loudness`) with `LinearRegression`, evaluated via MAE, MSE, and RMSE (`random_state=42`, 70/30 split). 🎶📈
4.  **Classification** — builds an `is_sonnati` (traditional vs. non-traditional) label from a curated list of 17 traditional Persian artists (e.g. Shajarian, Kalhor, Bastami), then compares AdaBoost and Random Forest classifiers via a shared `fit_and_eval` evaluation function — **Random Forest** was found to perform best, attributed to its robustness on high-dimensional data and resistance to overfitting. 🪕🎸

**Libraries:** `Scikit-learn`, `XGBoost`, `matplotlib`, `numpy`, `Seaborn`, `Pandas` 🐍

**About the Dataset:** 10,632 songs from 69 Iranian artists, with 32 features per track (metadata plus Spotify audio features like danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, and tempo).

**Evaluation metrics used:** Accuracy, Precision, Recall, F1-score, and a Confusion Matrix.

---

## 🛠️ Getting Started

Each project is a self-contained Jupyter notebook. To run one:

```bash
git clone https://github.com/Amirbehnam1009/Data-Mining-Projects.git
cd "Data-Mining-Projects/Final Project - Persian Spotify"   # or any other project folder
pip install scikit-learn pandas numpy matplotlib seaborn imbalanced-learn tensorflow
jupyter notebook
```


## 📄 License
Developed for academic purposes as part of Amirkabir university course.
