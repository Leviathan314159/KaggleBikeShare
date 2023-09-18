# BikeShareAnalysis.R

# Libraries
library(tidyverse)
library(vroom)
library(tidymodels)
library(poissonreg)

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

# Write the data -----------------------
vroom_write(bike_lm_predictions_export, "C:/Users/BYU Rental/STAT348/KaggleBikeShare/lm_submission.csv", delim = ",")
vroom_write(bike_poisson_export, "C:/Users/BYU Rental/STAT348/KaggleBikeShare/poisson_submission.csv", delim = ",")
