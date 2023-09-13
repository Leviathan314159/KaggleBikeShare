# BikeShareAnalysis.R

# Libraries
library(tidyverse)
library(vroom)
library(tidymodels)

# Read in the data
bike <- vroom("C:/Users/BYU Rental/STAT348/KaggleBikeShare/train.csv")

# Data Cleaning
# Change the 1 day that has weather category 4
bike$weather
updated_weather <- if_else(bike$weather == 4, 3, bike$weather)
unique(updated_weather) # This line is to test that the previous step worked
bike$weather <- updated_weather
# Update variables that should be factors
bike$season <- as.factor(bike$season)
bike$holiday <- as.factor(bike$holiday)
bike$workingday <- as.factor(bike$workingday)
bike$weather <- as.factor(bike$weather)

# Feature Engineering
glimpse(bike)
bike_recipe <- recipe(count ~ ., data = bike) |> 
  step_time(datetime, features="hour") |> 
  step_dummy(c(season, holiday, workingday, weather))
prepped_recipe <- prep(bike_recipe)
bake(prepped_recipe, new_data=bike)
