# BikeShareAnalysis.R

# Libraries
library(tidyverse)
library(vroom)
library(tidymodels)
library(poissonreg)
library(glmnet)

# Read in the data -----------------------------------
bike <- vroom("C:/Users/BYU Rental/STAT348/KaggleBikeShare/train.csv")
bike_test <- vroom("C:/Users/BYU Rental/STAT348/KaggleBikeShare/test.csv")

# Data Cleaning -------------------------------------

# Remove the "casual" and "registered" columns from the training data
# (they don't exist in the testing data)
bike <- bike |> select(-casual, -registered)

# Feature Engineering --------------------------------------
glimpse(bike)
bike_recipe <- recipe(count ~ ., data = bike) |> 
  step_mutate(weather = if_else(weather == 4, 3, weather)) |> 
  # Reassign the few data points that are weather category 4
  step_time(datetime, features="hour") |> 
  step_mutate(season = factor(season)) |> 
  step_mutate(holiday = factor(holiday)) |> 
  step_mutate(workingday = factor(workingday)) |> 
  step_mutate(weather = factor(weather)) |> 
  step_mutate(windspeed_squared = windspeed**2) |> 
  step_mutate(atemp_squared = atemp**2)
prepped_recipe <- prep(bike_recipe)
baked_bike_training <- bake(prepped_recipe, new_data=bike)
baked_bike_training

# Recipe for penalized regression
penalized_recipe <- recipe(count ~ ., data = bike) |> 
  step_mutate(weather = if_else(weather == 4, 3, weather)) |> 
  # Reassign the few data points that are weather category 4
  step_time(datetime, features="hour") |> 
  step_mutate(season = factor(season)) |> 
  step_mutate(holiday = factor(holiday)) |> 
  step_mutate(workingday = factor(workingday)) |> 
  step_mutate(weather = factor(weather)) |> 
  step_mutate(windspeed_squared = windspeed**2) |> 
  step_mutate(atemp_squared = atemp**2) |> 
  step_dummy(all_nominal_predictors()) |> 
  step_rm(datetime) |> 
  step_normalize(all_numeric_predictors()) # Standardizes the variables

# Recipe for fitting log-count to the predictors
log_bike <- bike
log_bike$count <- log(log_bike$count)
log_recipe <- recipe(count ~ ., data = log_bike) |> 
  step_mutate(weather = if_else(weather == 4, 3, weather)) |> 
  # Reassign the few data points that are weather category 4
  step_time(datetime, features="hour") |> 
  step_mutate(season = factor(season)) |> 
  step_mutate(holiday = factor(holiday)) |> 
  step_mutate(workingday = factor(workingday)) |> 
  step_mutate(weather = factor(weather)) |> 
  step_mutate(windspeed_squared = windspeed**2) |> 
  step_mutate(atemp_squared = atemp**2)

# Recipe for penalized log-count
preg_log_recipe <- recipe(count ~ ., data = log_bike) |> 
  step_mutate(weather = if_else(weather == 4, 3, weather)) |> 
  # Reassign the few data points that are weather category 4
  step_time(datetime, features="hour") |> 
  step_mutate(season = factor(season)) |> 
  step_mutate(holiday = factor(holiday)) |> 
  step_mutate(workingday = factor(workingday)) |> 
  step_mutate(weather = factor(weather)) |> 
  step_mutate(windspeed_squared = windspeed**2) |> 
  step_mutate(atemp_squared = atemp**2) |> 
  step_dummy(all_nominal_predictors()) |> 
  step_rm(datetime) |> 
  step_normalize(all_numeric_predictors()) # Standardizes the variables  

# Linear Regression ----------------------------------------
# Set up the linear model
bike_lm <- linear_reg() |> 
  set_engine("lm")

# Set up the workflow
bike_lm_workflow <- workflow() |> 
  add_recipe(bike_recipe) |> 
  add_model(bike_lm) |> 
  fit(data = bike) # Use the original recipe and the original data.
# The workflow step automatically preps and bakes.

# First look at predictions
bike_lm_predictions <- predict(bike_lm_workflow,
                               new_data = bike_test)
bike_lm_predictions

# Clean up predictions (count >= 0)
bike_lm_predictions$.pred <- if_else(bike_lm_predictions$.pred < 0, 0, bike_lm_predictions$.pred)
bike_lm_predictions

# Create dataset to write/export
bike_lm_predictions_export <- data.frame("datetime" = bike_test$datetime, "count" = bike_lm_predictions$.pred)
bike_lm_predictions_export$datetime <- as.character(format(bike_lm_predictions_export$datetime))


# Poisson Regression --------------------------------

# Set the model
poisson_model <- poisson_reg() |> set_engine("glm")
# Set the workflow
bike_pois_workflow <- workflow() |> add_recipe(bike_recipe) |> 
  add_model(poisson_model) |> fit(data = bike)

# Predictions
bike_poisson_pred <- predict(bike_pois_workflow, new_data = bike_test)
bike_poisson_pred

# Create dataset to write/export
bike_poisson_export <- data.frame("datetime" = as.character(format(bike_test$datetime)), "count" = bike_poisson_pred$.pred)


# Log-count -----------------------------------------
# Set up the linear model
log_model <- linear_reg() |> 
  set_engine("lm")

# Set up the workflow
bike_log_workflow <- workflow() |> 
  add_recipe(log_recipe) |> 
  add_model(log_model) |> 
  fit(data = log_bike) # Use the original recipe and the original data.
# The workflow step automatically preps and bakes.

# Log Predictions
bike_log_predictions <- predict(bike_log_workflow,
                               new_data = bike_test)
bike_log_predictions
log_model_export <- data.frame("datetime" = as.character(format(bike_test$datetime)), 
                               "count" = exp(bike_log_predictions$.pred))


# Penalized Regression ---------------------
preg_model <- linear_reg(penalty = 15, mixture = 0.5) |> 
  set_engine("glmnet")
preg_workflow <- workflow() |> 
  add_recipe(penalized_recipe) |> 
  add_model(preg_model) |> 
  fit(data = bike)

preg_predictions <- predict(preg_workflow, new_data = bike_test)
preg_predictions$.pred <- if_else(preg_predictions$.pred < 0, 0, preg_predictions$.pred)
preg_predictions

preg_export <- data.frame("datetime" = as.character(format(bike_test$datetime)), 
                          "count" = preg_predictions$.pred)


# Penalized Log ---------------------
# I noticed that this isn't very useful. The lower the penalty is, the higher 
# the score performs
preg_log <- linear_reg(penalty = 0.05, mixture = 0.5) |> 
  set_engine("glmnet")
preg_log_workflow <- workflow() |> 
  add_recipe(preg_log_recipe) |> 
  add_model(preg_log) |> 
  fit(data = log_bike)

# Predictions
preg_log_predictions <- predict(preg_log_workflow, new_data = bike_test)
preg_log_predictions

# Transform back from log
preg_log_export <- data.frame("datetime" = as.character(format(bike_test$datetime)),
                              "count" = exp(preg_log_predictions$.pred))


# Write the data -----------------------
vroom_write(bike_lm_predictions_export, "C:/Users/BYU Rental/STAT348/KaggleBikeShare/lm_submission.csv", delim = ",")
vroom_write(bike_poisson_export, "C:/Users/BYU Rental/STAT348/KaggleBikeShare/poisson_submission.csv", delim = ",")
vroom_write(log_model_export, "C:/Users/BYU Rental/STAT348/KaggleBikeShare/log_count_submission.csv", delim = ",")
vroom_write(preg_export, "C:/Users/BYU Rental/STAT348/KaggleBikeShare/penalized_submission.csv", delim = ",")
vroom_write(preg_log_export, "C:/Users/BYU Rental/STAT348/KaggleBikeShare/penalized_log_submission.csv", delim = ",")
