Smart_Comment_Safety_Analyzer

An AI-powered web application that detects and classifies social media comments into the following categories:

Good

Offensive

Threatening

This project uses Machine Learning and Natural Language Processing (NLP) techniques to analyze text in real time and provide appropriate intervention support for harmful comments.

Project Overview

With the rapid growth of social media platforms, detecting harmful or threatening content has become increasingly important.

This application:

Analyzes user-submitted comments

Classifies them using ML and NLP techniques

Displays prediction confidence using visualization

Provides help resources if harmful content is detected

Offers motivational support if the user declines help

Technology Stack

Python

Flask (Web Framework)

Scikit-learn (Machine Learning)

NLTK (Natural Language Processing)

TF-IDF Vectorization

Logistic Regression

VADER Sentiment Analysis

HTML and CSS

Chart.js (Data Visualization)

Machine Learning Approach
1. Text Preprocessing

Convert text to lowercase

Remove URLs

Remove special characters

Remove stopwords

2. Feature Extraction

TF-IDF Vectorization is used to convert text into numerical features.

3. Model

Logistic Regression classifier is used for multi-class classification.

4. Hybrid Logic

Combines machine learning prediction with VADER sentiment compound score to improve classification robustness.

Features

Real-time comment classification

Bar chart showing prediction confidence

Help prompt for Offensive and Threatening comments

Cybercrime helpline integration

Motivational support option

Clean and modular Flask project structure

Styled web interface with background design

Application Flow

The user enters a comment.

The system predicts the sentiment category.

The system displays:

The entered comment

The predicted sentiment

A confidence bar chart

If the comment is:

Good: Only the result and chart are displayed.

Offensive or Threatening: The user is asked whether they need help.

If Yes: Helpline resources are displayed.

If No: Motivational support messages are displayed.
