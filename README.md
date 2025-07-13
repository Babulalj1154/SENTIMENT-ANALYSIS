## SENTIMENT-ANALYSIS

*COMPANY* : CODTECH IT SOLUTIONS

*NAME* : BABULAL JANGID

*INTERN ID* : CT06DF2424

*DOMAIN8 : DATA ANALYTICS

8DURATION* : 6 WEEKS

MENTOR : NEELA SANTOSH


#: Sentiment Analysis using NLP on TripAdvisor Hotel Reviews

## Project Overview

This project focuses on performing sentiment analysis on hotel reviews sourced from the TripAdvisor dataset, aiming to classify reviews as **positive** or **negative** based on the review text. The primary goal is to apply Natural Language Processing (NLP) techniques using Python and machine learning libraries to preprocess the data, build a sentiment classification model, and evaluate its performance.

The entire project was implemented in **Google Colab**, and the dataset was downloaded from **Kaggle**.



## Dataset

The dataset used is titled **TripAdvisor Hotel Reviews**, which contains thousands of real user reviews along with ratings on a scale of 1 to 5. For simplicity in sentiment analysis:
- Reviews with ratings **4 and 5** were classified as **positive (1)**
- Reviews with ratings **1, 2, and 3** were classified as **negative (0)**

The dataset was loaded using Pandas and read with proper encoding (`latin-1`) to handle special characters.



## Tools and Libraries Used

The following Python libraries were utilized:
- **Pandas** & **NumPy**: Data handling and manipulation
- **NLTK**: Natural language preprocessing (stopword removal, stemming)
- **Scikit-learn**: Feature extraction, model training, and evaluation
- **Matplotlib** & **Seaborn**: Visualization
- **WordCloud**: Visual representation of frequent terms in reviews



## Data Preprocessing

Data cleaning and preparation steps included:
1. **Text Cleaning**: Removal of special characters, digits, and extra spaces.
2. **Lowercasing**: To standardize the text for better analysis.
3. **Stopword Removal**: Using NLTK’s English stopwords list.
4. **Stemming**: Using PorterStemmer to reduce words to their root forms.

A new column called `clean_review` was created to store the processed version of each review.



## Feature Extraction

We converted the cleaned text into numerical features using the **TF-IDF Vectorizer** from Scikit-learn. This technique considers both the frequency of a word and its rarity across documents, providing a better representation for sentiment classification.



## Model Building

A **Multinomial Naive Bayes** classifier was used for model training due to its efficiency and effectiveness in text classification tasks. The dataset was split into **80% training** and **20% testing** sets using `train_test_split()`.



## Evaluation Metrics

After training, the model was tested on the test set. Key evaluation metrics included:
- **Accuracy**: ~76.8%
- **Classification Report**: Provided precision, recall, and F1-score for both classes.
- **Confusion Matrix**: Visualized the number of correct and incorrect predictions for both positive and negative sentiments.

The model performed very well on positive sentiment classification but had room for improvement in detecting negative reviews.



## Visualization

- A **Word Cloud** was generated to highlight the most frequent words in hotel reviews.
- A **confusion matrix** heatmap was plotted using Seaborn to interpret the model’s performance.



## Conclusion

This project demonstrated the power of NLP and machine learning for text classification. By cleaning and transforming raw review text into structured data, we successfully built a working sentiment analysis model. Future improvements could include experimenting with different models (e.g., Logistic Regression, SVM, or LSTM), using advanced NLP techniques like lemmatization, or applying deep learning methods for better accuracy.


## Contact
Author: Babulal Jangid
Email: jangidbabulal760@gmail.com


## License
This project is licensed under the MIT License - feel free to use, share, or modify with attribution.
