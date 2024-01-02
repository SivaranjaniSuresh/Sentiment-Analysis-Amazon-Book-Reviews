
# Sentiment Analysis on Amazon Books Reviews 📚

## Abstract
In this project, sentiment analysis was conducted on a dataset sourced from Amazon Books Reviews - Goodreads-books reviews and descriptions of each book. The dataset, a subset of the Amazon review Dataset, includes feedback from 3 million users on 212,404 unique books, spanning from May 1996 to July 2014.

## Table of Contents
- [Datasets](#datasets)
- [Data Characteristics](#data-characteristics)
- [Interesting Datasets](#why-are-these-interesting-datasets)
- [Data Cleaning](#data-cleaning)
- [Balancing Classes](#balancing-classes)
- [Categorizing Sentiments](#categorizing-sentiments)
- [Text Cleaning](#text-cleaning)
- [Data Augmentation](#data-augmentation)
- [Data Preprocessing](#data-preprocessing)

## Datasets
### Books_rating.csv
- Id: Unique identifier of the book
- Title: Title of the book
- Price: Price of the book
- User_id: Unique identifier of the user who rated the book
- profileName: Name of the user who rated the book
- review/helpfulness: Helpfulness rating of the review
- review/score: Rating ranging from 0 to 5 for the book
- review/time: Time when the review was given
- review/summary: Summary of a text review
- review/text: Full text of the review

### books_data.csv
- Title: Title of the book
- description: Brief summary or overview of the book's content
- authors: Names of the authors who wrote the book
- Image: URL linking to the book cover image
- previewLink: Link to access a preview or more detailed information about the book on Google Books
- publisher: Name of the publishing company
- publishedDate: Date when the book was officially published
- infoLink: Link providing additional information about the book on Google Books
- categories: Genres or categories to which the book belongs
- ratingsCount: Total count of ratings received for the book

## Data Characteristics
The dataset includes feedback from 3 million users on 212,404 unique books. The 'review/score' column, ranging from 0 to 5, serves as the basis for sentiment categorization. The reviews cover a wide span of time, offering insights into user sentiments over two decades of book reviewing on Amazon.

## Why Are These Interesting Datasets?
The dataset presents a compelling opportunity for sentiment analysis due to its scale, diversity, and historical context.

## Data Cleaning
- Column Selection and Renaming
- Handling Missing Values
- Removing Duplicate Entries
- Sentiment Labeling
- Visualizing Label Distribution

## Balancing Classes
To achieve a balanced representation of sentiments, a subset of the dataset was created by selecting a fixed number of samples from each sentiment class.

## Categorizing Sentiments
Numerical sentiment labels (2 for positive, 0 for negative, and 1 for neutral) were categorized into corresponding textual categories.

## Text Cleaning
- Lowercasing
- Removing Special Characters, Punctuation, and Extra Whitespace

## Data Augmentation
Several text augmentation techniques were employed, including Synonym Replacement, Word Shuffling, and Adding Noise.

## Data Preprocessing
Data preprocessing steps involved removing stop words, lemmatization, and tokenization.

## Data Splitting for Machine Learning Training and Testing

The raw text data undergoes crucial preprocessing steps to prepare it for machine learning models. Initially, the tokenized words are converted into space-separated strings, creating coherent text representations. To transform these textual inputs into numerical features, the Term Frequency-Inverse Document Frequency (TF-IDF) vectorization technique is employed. TF-IDF captures the importance of words within documents by considering their frequency and rarity across the entire dataset.

## Model Building

## Multinomial Naive Bayes
- Trained using a TF-IDF vectorizer within a pipeline.
- Achieved an accuracy of approximately 81% on the test data.
- Detailed classification report offering insights into precision, recall, and F1-score for each sentiment class.

## K-Nearest Neighbors (KNN)
- Utilized both TF-IDF vectorizer and CountVectorizer (optional) within a pipeline with KNN classifier.
- Attained an accuracy of about 73% on the test data.
- Extracted top features for each class, revealing significant words influencing predictions.

## Gradient Boosting
- Employed a Gradient Boosting Classifier with TF-IDF vectorization in a pipeline.
- Obtained an accuracy of approximately 75% on the test data.
- Analyzed feature importances to identify words crucial for sentiment analysis.

## Logistic Regression
- Trained a Logistic Regression model using TF-IDF vectors.
- Achieved an accuracy of 81% on the test data.
- Comprehensive classification report detailing model performance for each sentiment category.

## Support Vector Machine (SVM)
- Implemented an SVM classifier with a linear kernel using TF-IDF vectors.
- Demonstrated high accuracy, achieving 82% on the test data.
- Generated a detailed classification report highlighting model effectiveness.

## Random Forest
- Employed a Random Forest classifier with TF-IDF vectorization.
- Showcased strong performance with an accuracy of 85% on the test data.
- Provided a comprehensive classification report for each sentiment class.

## XLNet
- Utilized XLNet, a transformer-based model for sentiment analysis.
- Evaluated the model on a validation set, achieving a validation accuracy of 77.06%.
- Conducted a comprehensive analysis of model performance with a focus on accuracy, precision, recall, and F1-score for each sentiment category.

## BERT
- Employed the BERT (Bidirectional Encoder Representations from Transformers) model.
- Achieved a validation accuracy of 75.78%, showcasing the model's effectiveness in capturing sentiment from book reviews.
- Evaluated the BERT model using accuracy as a primary metric, with potential inclusion of precision, recall, and F1-score in a detailed classification report.

## Pearson Correlation
### Key Steps:
1. **User Input:** User provides a set of book titles and corresponding review scores.
2. **Data Processing:** Ratings data is filtered to include users who have reviewed books from the input.
3. **Pearson Correlation Calculation:** Users are grouped based on their book reviews. Pearson Correlation is calculated between the input user and others, indicating similarity in rating patterns.
4. **Top Similar Users Selection:** Users with the highest Pearson Correlation are identified as the top similar users.
5. **Weighted Rating Calculation:** A weighted rating is computed by multiplying the similarity index with each user's review scores.
6. **Recommendation Score Aggregation:** Weighted average recommendation scores are calculated, considering the collective influence of similar users.
7. **Top Recommendations:** Book recommendations are generated by sorting candidates based on their weighted average recommendation scores.

## Enhancing Model Performance Through Advanced Techniques
Explored advanced techniques, employing methods such as Randomized Search Cross-Validation to optimize our models. These techniques are vital for ensuring that our models are not just accurate, but also robust and versatile.

## Random Forest Classifier Optimization
The Random Forest classifier was carefully fine-tuned. Leveraging Randomized Search Cross-Validation, explored a variety of hyperparameters, including the number of estimators, maximum depth, and minimum samples split. Through this process, identified the optimal configuration: 300 estimators, unlimited depth, and a minimum of 2 samples required to split. This meticulous tuning significantly bolstered the model's accuracy and generalizability.

## Naive Bayes Classifier Enhancement
The Naive Bayes classifier, a fundamental algorithm in text classification, was also optimized. By employing Grid Search Cross-Validation, explored different configurations, including term frequency-inverse document frequency (TF-IDF) vectorization parameters and additive smoothing (alpha). The best hyperparameters were determined through this process, enhancing the Naive Bayes model's accuracy and precision.

## K-Nearest Neighbors (KNN) Classifier Refinement
In the case of the KNN classifier, employed a combination of techniques. Utilizing Randomized Search Cross-Validation, explored n-gram ranges and weighting functions, refining the KNN model. By selecting the most appropriate hyperparameters, our KNN classifier exhibited superior performance, accurately capturing complex relationships in the data.

## Gradient Boosting Classifier Optimization
For the Gradient Boosting classifier, applied a systematic approach. Employing Randomized Search Cross-Validation, explored n-gram ranges, the number of boosting stages, learning rates, and maximum tree depths. This detailed exploration led us to the optimal configuration, ensuring that our Gradient Boosting model was finely tuned for accuracy and efficiency.

## Support Vector Machine (SVM) Classifier Enhancement
The SVM classifier, known for its ability to handle complex data, was meticulously fine-tuned. Using Randomized Search Cross-Validation, explored different kernel functions, regularization parameters, and gamma values. The best hyperparameters were selected, empowering our SVM model to accurately classify sentiment in textual data.

## Comprehensive Evaluation and Visualization
In our evaluation process, not only focused on accuracy but also delved into the nuances of precision, recall, and F1-score for each class. Also, visualized the classification report through a heatmap, providing a comprehensive overview of our models' performance across different sentiment classes. This visualization offered valuable insights into the strengths and areas for improvement for each classifier.


