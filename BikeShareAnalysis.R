# BikeShareAnalysis.R

# Libraries
library(tidyverse)
library(vroom)
library(tidymodels)
library(poissonreg)
library(glmnet)
library(rpart)
library(ranger)
library(DataExplorer)
library(parsnip)

# Read in the data -----------------------------------
base_folder <- "KaggleBikeShare/"
bike <- vroom(paste0(base_folder, "train.csv"))
bike_test <- vroom(paste0(base_folder, "test.csv"))

# Data Cleaning -------------------------------------

# Remove the "casual" and "registered" columns from the training data
# (they don't exist in the testing data)
# Create separate dataframes for the casual counts and the registered counts
casual_log_bike <- bike |> select(-count, -registered) |> 
  mutate(casual = if_else(casual == 0, 1e-100, 0))
casual_log_bike$casual <- log(casual_log_bike$casual)
registered_log_bike <- bike |> select(-count, -casual) |> 
  mutate(registered = if_else(registered == 0, 1e-100, 0))
registered_log_bike$registered <- log(registered_log_bike$registered)
casual_bike <- bike |> select(-count, -registered)
registered_bike <- bike |> select(-count, -casual)
bike <- bike |> select(-casual, -registered)
log_bike <- bike
log_bike$count <- log(log_bike$count)

# Feature Engineering --------------------------------------
glimpse(bike)
bike_recipe <- recipe(count ~ ., data = bike) |> 
  step_mutate(weather = if_else(weather == 4, 3, weather)) |> 
  # Reassign the few data points that are weather category 4
  step_time(datetime, features="hour") |> 
  step_mutate(datetime_year = year(datetime)) |> 
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
  step_mutate(datetime_year = year(datetime)) |> 
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
log_recipe <- recipe(count ~ ., data = log_bike) |> 
  step_mutate(weather = if_else(weather == 4, 3, weather)) |> 
  # Reassign the few data points that are weather category 4
  step_time(datetime, features="hour") |> 
  step_mutate(datetime_year = year(datetime)) |> 
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
  step_mutate(datetime_year = year(datetime)) |> 
  step_mutate(season = factor(season)) |> 
  step_mutate(holiday = factor(holiday)) |> 
  step_mutate(workingday = factor(workingday)) |> 
  step_mutate(weather = factor(weather)) |> 
  step_mutate(windspeed_squared = windspeed**2) |> 
  step_mutate(atemp_squared = atemp**2) |> 
  step_dummy(all_nominal_predictors()) |> 
  step_rm(datetime) |> 
  step_normalize(all_numeric_predictors()) # Standardizes the variables  

tree_log_recipe <- recipe(count ~ ., data = log_bike) |> 
  # Basic data cleaning and formatting 
  # (minimal feature engineering for decision tree)
  step_mutate(weather = if_else(weather == 4, 3, weather)) |> 
  step_time(datetime, features="hour") |> 
  step_mutate(datetime_year = year(datetime)) |> 
  step_mutate(season = factor(season)) |> 
  step_mutate(holiday = factor(holiday)) |> 
  step_mutate(workingday = factor(workingday)) |> 
  step_mutate(weather = factor(weather))

stack_recipe <- recipe(count ~ ., data = log_bike) |> 
  step_mutate(weather = if_else(weather == 4, 3, weather)) |> 
  # Reassign the few data points that are weather category 4
  step_time(datetime, features="hour") |> 
  step_mutate(datetime_year = year(datetime)) |> 
  step_mutate(season = factor(season)) |> 
  step_mutate(holiday = factor(holiday)) |> 
  step_mutate(workingday = factor(workingday)) |> 
  step_mutate(weather = factor(weather)) |> 
  step_mutate(windspeed_squared = windspeed**2) |> 
  step_mutate(atemp_squared = atemp**2) |> 
  step_rm(datetime) |> 
  step_dummy(all_nominal_predictors()) |> 
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


# Automatic parameter tuning ------------------------------------
auto_preg_model <- linear_reg(penalty = tune(),
                              mixture = tune()) |> 
  set_engine("glmnet")

# Set the workflow
auto_preg_wf <- workflow() |> 
  add_recipe(preg_log_recipe) |> 
  add_model(auto_preg_model)

# Create a grid of values to iteratively tune
tuning_grid <- grid_regular(penalty(), mixture(), levels = 10)

# Split the data for cross validation
folds <- vfold_cv(data = bike, v = 10, repeats = 1)

# Run the cross validation
cv_results <- auto_preg_wf |> 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(rmse, mae, rsq))

# Plot the results to visualize
collect_metrics(cv_results) |> 
  filter(.metric == "rmse") %>% 
  ggplot(data = ., aes(x = penalty, y = mean, color = factor(mixture))) + 
  geom_line()

# It looks like the best tuning parameters are around lambda = 0.10, nu = 0.25
# Apply the tuning parameters
preg_auto_log <- linear_reg(penalty = 0.10, mixture = 0.25) |> 
  set_engine("glmnet")
auto_preg_log_workflow <- workflow() |> 
  add_recipe(preg_log_recipe) |> 
  add_model(preg_auto_log) |> 
  fit(data = log_bike)

# Predictions
auto_preg_log_predictions <- predict(auto_preg_log_workflow, new_data = bike_test)
auto_preg_log_predictions

# Transform back from log
auto_preg_log_export <- data.frame("datetime" = as.character(format(bike_test$datetime)),
                              "count" = exp(auto_preg_log_predictions$.pred))


# Regression tree (log-count) ------------------------------

tree_log_model <- decision_tree(tree_depth = tune(),
                                cost_complexity = tune(),
                                min_n = tune()) |> 
  set_engine("rpart") |> 
  set_mode("regression")

# Workflow
tree_log_wf <- workflow() |> 
  add_recipe(tree_log_recipe) |> 
  add_model(tree_log_model)

# Set up the grid of tuning values
?grid_regular
tree_tuning_grid <- grid_regular(tree_depth(), cost_complexity(), 
                                 min_n(), levels = 5)

# Set up K-fold CV
tree_folds <- vfold_cv(data = log_bike, v = 10, repeats = 1)

# Find best tuning parameters
tree_cv_results <- tree_log_wf |> 
  tune_grid(resamples = tree_folds,
            grid = tree_tuning_grid,
            metrics = metric_set(rmse, mae, rsq))

# Plot the results to visualize
# Keep in mind that we need to observe all three metrics:
# tree_depth, cost_complexity, and min_n
collect_metrics(tree_cv_results) |> 
  filter(.metric == "rmse") %>% 
  ggplot(data = ., aes(x = tree_depth, y = mean, color = factor(cost_complexity), shape = factor(min_n))) + 
  geom_point()

# Finalize the workflow using the best tuning parameters and predict
# The best tuning parametrics I got were:
# tree_depth = 15
# cost_complexity = 3.16227766016838e-06
# min_n = 21
tree_log_model <- decision_tree(tree_depth = 15,
                                cost_complexity = 3.16227766016838e-06,
                                min_n = 21) |> 
  set_engine("rpart") |> 
  set_mode("regression")

tree_log_wf <- workflow() |> 
  add_recipe(tree_log_recipe) |> 
  add_model(tree_log_model) |> 
  fit(data = log_bike)

tree_log_predictions <- predict(tree_log_wf, new_data = bike_test)
tree_log_predictions

# Prepare the data for export
tree_log_export <- data.frame("datetime" = as.character(format(bike_test$datetime)),
                              "count" = exp(tree_log_predictions$.pred))

# Random Forest ---------------------------

# Set the model
forest_model <- rand_forest(mtry = tune(),
                            min_n = tune(),
                            trees = 500) |> 
  set_engine("ranger") |> 
  set_mode("regression")

# Create a workflow
forest_wf <- workflow() |> 
  add_recipe(tree_log_recipe) |> 
  add_model(forest_model) # |> 
  # fit(data = log_bike)

# Set up the grid with the tuning values
forest_grid <- grid_regular(mtry(range = c(1, (length(log_bike)-1))), min_n())

# Set up the K-fold CV
forest_folds <- vfold_cv(data = log_bike, v = 10, repeats = 1)

# Find best tuning parameters
forest_cv_results <- forest_wf |> 
  tune_grid(resamples = forest_folds,
            grid = forest_grid,
            metrics = metric_set(rmse, mae, rsq))

collect_metrics(forest_cv_results) |> 
  filter(.metric == "rmse") %>% 
  ggplot(data = ., aes(x = mtry, y = mean, color = factor(min_n))) + 
  geom_point()

# Finalize the workflow using the best tuning parameters and predict
# The best parameters were mtry = 9 and min_n = 2
forest_final_model <- rand_forest(mtry = 9,
                            min_n = 2,
                            trees = 500) |> 
  set_engine("ranger") |> 
  set_mode("regression")

forest_final_wf <- workflow() |> 
  add_recipe(tree_log_recipe) |> 
  add_model(forest_final_model) |> 
  fit(data = log_bike)

forest_predictions <- predict(forest_final_wf, new_data = bike_test)
forest_predictions

forest_export <- data.frame("datetime" = as.character(format(bike_test$datetime)),
                            "count" = exp(forest_predictions$.pred))


# Model Stacking -------------------------------
library(stacks)
# Split the data for CV
stack_folds <- vfold_cv(log_bike, v = 10, repeats = 1)

# Create a control grid
untuned_model <- control_stack_grid()
tuned_model <- control_stack_resamples()

# Adding penalized regression models
stack_preg_model <- linear_reg(penalty = tune(),
                               mixture = tune()) |> 
  set_engine("glmnet")

# Adding decision tree models
stack_tree_model <- decision_tree(tree_depth = tune(),
                                  cost_complexity = tune(),
                                  min_n = tune()) |> 
  set_engine("rpart") |> 
  set_mode("regression")

# Set up workflows
stack_preg_wf <- workflow() |> 
  add_recipe(stack_recipe) |> 
  add_model(stack_preg_model)

stack_tree_wf <- workflow() |> 
  add_recipe(stack_recipe) |> 
  add_model(stack_tree_model)

# Grid of values to tune over
stack_preg_tuning_grid <- grid_regular(penalty(),
                                       mixture(),
                                       levels = 5)
stack_tree_tuning_grid <- grid_regular(tree_depth(),
                                       cost_complexity(),
                                       min_n(),
                                       levels = 5)

# Run the CV
stack_preg_models <- stack_preg_wf |> 
  tune_grid(resamples = stack_folds,
            grid = stack_preg_tuning_grid,
            metrics = metric_set(rmse, mae, rsq),
            control = untuned_model)
stack_tree_models <- stack_tree_wf |> 
  tune_grid(resamples = stack_folds,
            grid = stack_tree_tuning_grid,
            metrics = metric_set(rmse, mae, rsq),
            control = untuned_model)

# Create other resampling objects using other ML algorithms
stack_lin_reg <- linear_reg() |> 
  set_engine("lm")
stack_lin_reg_wf <- workflow() |> 
  add_recipe(stack_recipe) |> 
  add_model(stack_lin_reg)
stack_lin_reg_model <- fit_resamples(stack_lin_reg_wf,
                                     resamples = stack_folds,
                                     metrics = metric_set(rmse),
                                     control = tuned_model)

# Specify which models to include
stack_prep <- stacks() |> 
  add_candidates(stack_preg_models) |> 
  add_candidates(stack_lin_reg_model) |> 
  add_candidates(stack_tree_models)

# Fit the stacked model
stack_model <- stack_prep |> 
  blend_predictions() |> 
  fit_members()

# Make predictions using the stacked data
stack_predictions <- stack_model |> predict(new_data = bike_test)
stack_predictions

# Prepare the predictions for export
stack_export <- data.frame("datetime" = as.character(bike_test$datetime),
                           "count" = exp(stack_predictions$.pred))


# Designing a model to get under 0.44 on Kaggle ---------------------------
# The closest I got was with a random forest model.



# Trying the bart model -----------------------
# Set the model
bart_model <- bart(trees = 500) |> 
  set_engine("dbarts") |> 
  set_mode("regression")

# Create a workflow
bart_wf <- workflow() |> 
  add_recipe(tree_log_recipe) |> 
  add_model(bart_model) |> 
  fit(data = log_bike)

# Make predictions
bart_predictions <- predict(bart_wf, new_data = bike_test)
bart_predictions

bart_export <- data.frame("datetime" = as.character(format(bike_test$datetime)),
                          "count" = exp(bart_predictions$.pred))

# Write the data -----------------------
# See above for the base folder. The rest of the name is the file name and extension.
vroom_write(bike_lm_predictions_export, paste0(base_folder, "lm_submission.csv"), delim = ",")
vroom_write(bike_poisson_export, paste0(base_folder, "poisson_submission.csv"), delim = ",")
vroom_write(log_model_export, paste0(base_folder, "log_count_submission.csv"), delim = ",")
vroom_write(preg_export, paste0(base_folder, "penalized_submission.csv"), delim = ",")
vroom_write(preg_log_export, paste0(base_folder, "penalized_log_submission.csv"), delim = ",")
vroom_write(auto_preg_log_export, paste0(base_folder, "autotuned_penalized_log_count.csv"), delim = ",")
vroom_write(tree_log_export, paste0(base_folder, "tree_log_count.csv"), delim = ",")
vroom_write(forest_export, paste0(base_folder, "forest_log_count.csv"), delim = ",")
vroom_write(stack_export, paste0(base_folder, "stacked_log_count.csv"), delim = ",")
vroom_write(larger_forest_export, paste0(base_folder, "larger_forest_log.csv"), delim = ",")
vroom_write(better_forest_export, paste0(base_folder, "better_forest.csv"), delim=",")
vroom_write(bart_export, paste0(base_folder, "bart.csv"), delim=",")
