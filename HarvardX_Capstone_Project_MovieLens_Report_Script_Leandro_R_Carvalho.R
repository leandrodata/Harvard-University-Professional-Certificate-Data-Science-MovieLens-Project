#title: "HarvardX Capstone Project - MovieLens Report"
#author: "Leandro Rodrigues Carvalho"

#Description: Script in R format containing the code and comments that generated 
#the model to predict movie ratings and RMSE score.

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

#Note: this process could take a couple of minutes

#Code provided

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

#MovieLens 10M dataset:
#https://grouplens.org/datasets/movielens/10m/
#http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)

colnames(movies) <- c("movieId", "title", "genres")

#if using R 4.0 or later:

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

#Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

#Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

#Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#Saving data as R objects

save(edx, file = 'edx.RData')
save(validation, file = 'validation.RData')

#Dataset description and explorations

str(edx)

unique_col <- edx %>%
  summarize(unique_users = n_distinct(userId),
            unique_movies = n_distinct(movieId),
            unique_genres = n_distinct(genres))

knitr::kable(unique_col)

tot_observation <- length(edx$rating) + length(validation$rating) 
tot_observation

#Some of the most popular genres in the dataset are "Drama", "Comedy", "Thriller" and "Romance".

genres = c("Drama", "Comedy", "Thriller", "Romance")
sapply(genres, function(g) {
  sum(str_detect(edx$genres, g))
})

#The movie with the greatest number of ratings is "Pulp Fiction"

edx %>% group_by(movieId, title) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

#It's important to understand that not every user rated every movie. 
#There are users on the rows and movies on the columns with many empty cells. 

keep <- edx %>%
     dplyr::count(movieId) %>%
     top_n(5) %>%
     pull(movieId)

tab <- edx %>%
     filter(userId %in% c(13:20)) %>% 
     filter(movieId %in% keep) %>% 
     select(userId, title, rating) %>% 
     spread(title, rating)

tab %>% knitr::kable()

#Ratings profile

table_rating <- as.data.frame(table(edx$rating))
colnames(table_rating) <- c('Rating', 'Frequencies')

knitr::kable(table_rating)

#Visualizing the ratings

table_rating %>% ggplot(aes(Rating, Frequencies)) +
geom_bar(stat = 'identity') +
labs(x='Ratings', y='Count') +
ggtitle('Distribution of ratings')

#Methods and Analysis

#Residual Mean Squared Error (RMSE)

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#Building up the model step-by-step

#Step One

mu_hat <- mean(edx$rating)
mu_hat

step_one_rmse <- RMSE(validation$rating, mu_hat)
step_one_rmse

rmse_project_results <- data_frame(Method = "Step One/Base Model", RMSE = step_one_rmse)

rmse_project_results %>% knitr::kable()

#Step Two

#lm(rating ~ as.factor(movieID), data = movielens)

mu <- mean(edx$rating)

movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

movie_avgs %>% qplot(b_i, geom ="histogram", bins = 20, data = ., color = I("black"))

predicted_ratings_movie_bias <- mu + validation %>%
  left_join(movie_avgs, by = 'movieId') %>%
  pull(b_i)

step_two_rmse <- RMSE(predicted_ratings_movie_bias, validation$rating)
step_two_rmse

rmse_project_results <- bind_rows(rmse_project_results, data_frame(Method = "Step Two/Movie Bias",
                                                                   RMSE = step_two_rmse))

rmse_project_results %>% knitr::kable()

#Step Three

user_avgs <- edx %>%
  left_join(movie_avgs, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

user_avgs %>% qplot(b_u, geom ="histogram", bins = 30, data = ., color = I("black"))

predicted_ratings_user_bias <- validation %>%
  left_join(movie_avgs, by = 'movieId') %>%
  left_join(user_avgs, by = 'userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

step_three_rmse <- RMSE(predicted_ratings_user_bias, validation$rating)
step_three_rmse

rmse_project_results <- bind_rows(rmse_project_results, data_frame(Method = "Step Three/User Bias",
RMSE = step_three_rmse))

rmse_project_results %>% knitr::kable()

#Step Four

#Regularization

lambda <- 3

mu <- mean(edx$rating)

movie_reg_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n())

data_frame(original = movie_avgs$b_i, 
           regularlized = movie_reg_avgs$b_i, 
           n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5)

#To further explore lambda as a tuning parameter we can use cross-validation.

lambdas <- seq(0, 10, 0.25)

mu <- mean(edx$rating)

just_the_sum <- edx %>% 
  group_by(movieId) %>% 
  summarize(s = sum(rating - mu), n_i = n())

rmses <- sapply(lambdas, function(l){
  predicted_ratings <- validation %>% 
    left_join(just_the_sum, by = 'movieId') %>% 
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, validation$rating))
})

#Plotting the result

qplot(lambdas, rmses)

#And figuring out the which.min rmse

lambdas[which.min(rmses)]

#Regularization for the estimate user effects as well. 
#Here we use cross-validation to pick a lambda.

lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){

  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))

  predicted_ratings <- validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
    return(RMSE(predicted_ratings, validation$rating))
})

#To visualize

ggplot(data.frame(lambdas = lambdas, rmses = rmses ), aes(lambdas, rmses)) +
geom_point()

#For the full model, the optimal lambda is:

lambda <- lambdas[which.min(rmses)]
lambda

#Results and Conclusion

rmse_project_results <- bind_rows(rmse_project_results,
                          data_frame(Method = "Step Four/Regularized Movie + User Effect Model",  
                                     RMSE = min(rmses)))

rmse_project_results %>% knitr::kable()
