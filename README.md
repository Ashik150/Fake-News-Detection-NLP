# Fake News Detection using NLP

This project implements a system to detect fake news articles using Natural Language Processing (NLP) techniques and machine learning models. The system preprocesses text data, extracts features using TF-IDF, and then trains and evaluates classification models to distinguish between real and fake news.

## Dataset

The project utilizes the `WELFake_Dataset.csv`. This dataset contains news articles labeled as either real (1) or fake (0).

## Project Workflow

1.  **Data Loading & Initial Exploration:** The dataset is loaded, and initial checks for null values and data structure are performed.
2.  **Data Cleaning:**
    *   The 'Unnamed: 0' column is dropped.
    *   Null values in text fields are filled with empty strings.
3.  **Feature Engineering & Preprocessing:**
    *   Text features (`title` and `text`) are combined (though the code primarily uses `text`).
    *   Text preprocessing steps include:
        *   Punctuation removal.
        *   Stopword removal (using NLTK's English stopwords).
        *   Lemmatization (using NLTK's WordNetLemmatizer).
4.  **Visualization:**
    *   A pie chart shows the distribution of real vs. fake news labels.
    *   Word clouds are generated for both fake and real news text to visualize frequent terms.
5.  **TF-IDF Vectorization:** Text data is converted into numerical feature vectors using Term Frequency-Inverse Document Frequency (TF-IDF).
6.  **Model Training & Evaluation:**
    *   The dataset is split into training and testing sets.
    *   Two models are trained and evaluated:
        *   Multinomial Naive Bayes
        *   Logistic Regression
    *   Evaluation metrics include Accuracy and Confusion Matrix. (The `evaluate_model` function also calculates ROC AUC and Precision-Recall AUC, though these are not explicitly displayed in the main script flow for the chosen models).
7.  **Results Visualization:** A bar chart compares the accuracy of the trained models.
8.  **Prediction on New Text:** The system allows for inputting random text, preprocessing it, and predicting its class using the trained models.
9.  **Model Persistence:** The trained Logistic Regression model is saved using `pickle` for later use.

## Technologies Used

*   Python
*   Pandas (Data manipulation)
*   NumPy (Numerical operations)
*   Seaborn & Matplotlib (Data visualization)
*   Plotly Express (Interactive visualizations)
*   Scikit-learn (Machine learning: `train_test_split`, `TfidfVectorizer`, `MultinomialNB`, `LogisticRegression`, metrics)
*   NLTK (Natural Language Toolkit: `word_tokenize`, `stopwords`, `WordNetLemmatizer`)
*   WordCloud (Generating word clouds)
*   Pickle (Model saving/loading)

## How to Run

1.  Ensure you have Python installed.
2.  Install the required libraries:
    ```bash
    pip install pandas numpy seaborn matplotlib scikit-learn wordcloud nltk plotly pickle
    ```
3.  Download NLTK resources (run these in a Python interpreter):
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt') # For word_tokenize, if not already present
    ```
4.  Make sure the `WELFake_Dataset.csv` file is in the correct path specified in the script (`C:\\Users\\HP\\Downloads\\Datasets\\WELFake_Dataset.csv`) or update the path accordingly.
5.  Run the Python script.
