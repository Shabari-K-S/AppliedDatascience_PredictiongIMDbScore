# IMDb Score Prediction README

This Jupyter Notebook provides a machine learning model to predict IMDb scores based on various movie features. The code includes data preprocessing, exploratory data analysis, data cleaning, and model building. Below are the steps to run this code:

## Prerequisites
1. You need to have Python installed on your system.
2. Install Jupyter Notebook and the required libraries using pip:
   ```bash
   pip install jupyter numpy pandas seaborn matplotlib scikit-learn scipy
   ```

## Steps to Run the Code
1. Clone or download the Jupyter Notebook file and the dataset ('movie_metadata.csv') to your local machine.
2. Open a terminal and navigate to the directory containing the Jupyter Notebook and the dataset.
3. Start a Jupyter Notebook session:
   ```bash
   jupyter notebook
   ```
4. In the Jupyter Notebook dashboard, open the 'IMDb_Score_Prediction.ipynb' file.
5. Run the code cells in the notebook sequentially by clicking on each cell and pressing Shift + Enter.
6. You can interact with the code, view visualizations, and see model performance metrics as the code executes.
7. The final model, a Random Forest Classifier, is saved as 'model.pkl' and can be used for IMDb score predictions.

## Dataset Used
In this project, we used a movie metadata dataset obtained from Kaggle. The dataset contains a wealth of information about movies and their IMDb scores. It is essential to understand how various factors, such as directors, actors, critic reviews, and viewer reactions, influence a movie's IMDb rating. You can access the dataset on Kaggle at the following URL: [IMDb Score Prediction Dataset](https://www.kaggle.com/code/saurav9786/imdb-score-prediction-for-movies/input).

## About the Dataset
The dataset consists of 28 columns, providing comprehensive information about movies and their IMDb scores. IMDb score is a crucial metric for assessing a movie's success. A higher IMDb score often indicates a more successful movie, while a lower score may imply less success. The dataset allows us to analyze various factors that can influence IMDb ratings, helping us make better predictions and insights.

The key features in the dataset include:
- Director information
- Actor details
- Critic reviews
- Viewer reactions
- Genre
- Budget
- Movie duration
- Release year
- And more

We utilized this dataset to build machine learning models for IMDb score prediction and movie analysis. The code and Jupyter Notebook provide a step-by-step guide on how to preprocess the data, perform exploratory data analysis, clean the data, build regression models, and even predict IMDb score categories.

Feel free to explore and modify the code to gain a deeper understanding of IMDb score prediction and movie analysis using this rich dataset.


## Understanding the Code
The code consists of the following sections:
- Importing necessary libraries and reading the dataset.
- Data preprocessing, cleaning, and feature engineering.
- Exploratory data analysis with visualizations.
- Building and evaluating regression models:
  - Linear Regression
  - Polynomial Regression
  - Decision Tree Regression
  - Random Forest Regression
- Building and evaluating a classification model for IMDb score categories.
- Saving the trained Random Forest Classifier model for IMDb score predictions.

## Data Sources
The dataset used in this code ('movie_metadata.csv') is assumed to be available in the same directory as the Jupyter Notebook. This dataset contains various features of movies, including IMDb scores, which are used to train and evaluate the models.
