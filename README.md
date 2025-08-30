# AWS Review Software Analysis

This repository contains Python code for analyzing and modeling the AWS review software dataset. The dataset is loaded using pandas, preprocessed, and then used to train Decision Tree Classifier models with both TF-IDF and Count Vectorization.

## Dataset

The dataset used in this analysis is named `aws_review_sofware_dataset.csv`. It contains the following columns:

- `Unnamed: 0`: Index column
- `overall`: Overall rating
- `verified`: Verification status of the review
- `reviewTime`: Time of the review
- `reviewerID`: ID of the reviewer
- `asin`: Amazon Standard Identification Number
- `style`: Style of the review
- `reviewerName`: Name of the reviewer
- `reviewText`: Text of the review
- `summary`: Summary of the review
- `unixReviewTime`: Unix timestamp of the review time
- `vote`: Number of votes the review received
- `image`: Image associated with the review (if any)

## Preprocessing

The following preprocessing steps are performed on the dataset:

1. Sampling: Randomly select 10,000 samples from the dataset.
2. Handling Missing Values: Check for and handle any missing values in the dataset.
3. Tokenization: Tokenize the review text into sentences and words using NLTK.
4. Lemmatization: Lemmatize the words in the reviews using the PyWSD library.
5. Vectorization: Vectorize the words using TF-IDF Vectorizer and Count Vectorizer from scikit-learn.

## Model Training

Two models are trained using different vectorization techniques:

1. **TF-IDF Vectorization:**
   - A Decision Tree Classifier model is trained using the TF-IDF matrix and the target variable `verified`. The accuracy of the trained model on the training data is approximately 99.69%.

2. **Count Vectorization:**
   - A Decision Tree Classifier model is trained using the Count Vectorizer matrix and the target variable `verified`. The accuracy of the trained model on the training data is approximately 99.46%.
# Sentiment Analysis Projects

## Overview
This repository contains two Jupyter Notebooks demonstrating **Sentiment Analysis** using different text vectorization techniques:

1. **Count Vectorizer**: Converts text into a matrix of token counts.
2. **TF-IDF Vectorizer**: Converts text into term frequency-inverse document frequency (TF-IDF) values to measure word importance.

## Included Notebooks

1. **Sentiment Analysis with Count Vectorizer (`Sentimental_analysis_count_vectorizer.ipynb`)**
   - Uses CountVectorizer to transform textual data into numerical form for sentiment classification.
   
2. **Sentiment Analysis with TF-IDF Vectorizer (`Sentimental_analysis_TF-IDF_Vectorizer.ipynb`)**
   - Uses TF-IDF to transform text while reducing the impact of frequently occurring words.

## Dependencies
Ensure you have the following libraries installed:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk
```

## Preprocessing Steps
- **Data Loading**: Load the dataset containing text and sentiment labels.
- **Text Cleaning**: Remove special characters, punctuation, stopwords, and perform tokenization.
- **Vectorization**:
  - **Count Vectorizer**: Represents text based on word frequency.
  - **TF-IDF Vectorizer**: Assigns weights to words based on their importance.
- **Splitting Data**: Divide the dataset into training and testing sets.

## Model Training & Validation
- Train **machine learning models** (e.g., Logistic Regression, Naïve Bayes, SVM) on vectorized text.
- Evaluate performance using metrics such as **accuracy, precision, recall, and F1-score**.
- Perform **hyperparameter tuning** for improved performance.

## Expected Results
- Compare the effectiveness of **Count Vectorizer vs. TF-IDF** in sentiment classification.
- Identify which approach works better for specific datasets.

## References
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
- [NLTK Documentation](https://www.nltk.org/)

