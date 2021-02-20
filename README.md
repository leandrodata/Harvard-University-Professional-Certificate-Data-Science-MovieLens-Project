## HarvardX9---Capstone-Project

This is the Capstone Project presented at Harvard University online "Professional Certificate" course on Data Science, lead by Professor Rafael A. Irizarry on EdX platform.

The goal of this project was to create a *movie recommendation system* to predict movie ratings using the MovieLens data set. 

The key steps that were performed included Exploratory Data Analysis (EDA), data visualization, data wrangling and training the machine learning algorithm,
using the inputs in one subset to predict movie ratings in the validation set.

### Historic Context

The recommendation system presented at the Harvard course went through some of the data analysis strategies used by the winning team of the 2006 "Netflix Challenge",  when the company challenged the data science community to improve its recommendation algorithm by 10%. The winner would get a million dollars prize. 
In September 2009, the winners were announced.

### Dataset description

The data set version of MovieLens used in this report, the 10M version of the original data set, is a small subset of a much larger set with millions of ratings, 
in order to make the computation faster.

It's important to mention that the Netflix data is not publicly available, but the GroupLens research lab generated their own database with over 20 million ratings 
for over 27,000 movies by more than 138,000 users.

Harvard and Professor Rafael A. Irizarry made a small subset of this data via the dslabs package containing 9,000,055 observations of 6 variables, 
where each row represents a rating given by one user to one movie.

### Methods and Analysis

Initially, a test set was created using the **createDataPartition** function of the caret package to make possible to assess the accuracy of the model. In this project, 10% of the data was assigned to the test set and 90% to the train set.

### Residual Mean Squared Error (RMSE)

An important part of this project was to provide the RMSE using only the training set, and experimenting with multiple parameters. 
That strategy followed the Netflix challenge winning project that based their work on the residual mean squared error (RMSE) on the test set.

### Results

The base model used here assumed the same rating for all movies and didn't achieve a good enough result: 
its RMSE was more than 1, or in other words, an error of an entire star! 

To improve this result we added an 'effect' (or bias)representing the average ranking for *movie i* on Step Two and, following the same logic, an effect for *user u* on Step Three.

Step Three achieved a good improvement on the performance of the model with an RMSE of 0.8653488.

The problem, though, was that when we apply it to the test set, it rates unknown movies on the top, showing that something wasn't quite right.

To sort this out, we've moved ahead to Step Four, applying the concept of **regularization**, penalizing large estimates that are formed using small sample sizes, 
and adjusting for noisy estimates that were withholding the model to perform better, reaching a final RMSE of 0.864817.
