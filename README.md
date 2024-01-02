
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

