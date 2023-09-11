##
## Bike Share EDA Code
##

## Import Libraries
library(tidyverse)
library(vroom)

## Read in the Data
bike <- vroom("C:/Users/BYU Rental/STAT348/KaggleBikeShare/train.csv")

# R tools for performing an EDA
# dplyr::glimpse(dataset) - lists the variable type of each column
# skimr::skim(dataset) - gives a nice overview of the dataset
# DataExplorer::plot_intro(dataset) - visualization of glimpse()
# DataExplorer has many other helpful functions

# The variable datetime contains information that may be useful to divide
# into multiple variables. More on this later???

# EDA
library(DataExplorer)
dplyr::glimpse(bike)

# The following variables need to be treated like factors:
# season, holiday, workingday, weather

plot_intro(bike)
corr_plot <- plot_correlation(bike)
corr_plot
plot_bar(bike)
hist_plot <- plot_histogram(bike)
hist_plot
plot_missing(bike)

# More exploratory plots
ggplot(data = bike, mapping = aes(x = as.factor(weather), y = count)) + geom_boxplot()
ggplot(data = bike, mapping = aes(x = temp, y = count)) + geom_point()
ggplot(data = bike, mapping = aes(x = datetime, y = count)) + geom_point()
ggplot(data = bike, mapping = aes(x = as.factor(season), y = count)) + geom_boxplot()
ggplot(data = bike, mapping = aes(x = as.factor(season), 
                                  y = count, color = as.factor(weather))) + 
  geom_boxplot() + labs(x = "season", color = "weather")

# Include this plot: shows the relationship between temp and season, and how count responds
temp_season_plot <- ggplot(data = bike, mapping = aes(x = temp, 
                                                      y = count, color = as.factor(season))) + 
  geom_point() + labs(color = "season")
temp_season_plot
# This seems to indicate to me that they are assigning the seasons in this way:
# Spring = Jan-Mar
# Summer = Apr-Jun
# Fall = Jul-Sep
# Winter = Oct-Dec

# Relationship between count and datetime
time_plot <- ggplot(data = bike, mapping = aes(x = datetime, y = count)) + 
  geom_point() + geom_smooth()
time_plot
# It looks like the number of bike rentals is higher in the middle of the year,
# and is higher in 2012 than it is in 2011. Would it be useful to add year as 
# a variable???

# Relationship between workingday and count
work_plot <- ggplot(data = bike, 
                    mapping = aes(x = as.factor(workingday), 
                                  fill = as.factor(season))) + 
  geom_bar(position = "dodge") + labs(x="Working Day (y/n)", fill="season")
work_plot

# Make the panel plot
library(patchwork)
# time_plot, temp_season_plot, corr_plot, hist_plot
(time_plot + temp_season_plot) / (corr_plot + work_plot)
ggsave("C:/Users/BYU Rental/STAT348/KaggleBikeShare/BikeShare 4 Panel Plot.png")

# REMEMBER to git add, git commit -m, git push