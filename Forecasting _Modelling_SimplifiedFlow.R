# Redevelopment Forecasting Project ----

#This README serves as a user guide for running the Redevelopment Forecasting Models in R (v0), held in this repository.

# Project Showcase Notice ----
#This repository contains code and materials for the purpose of showcasing the project. Please note that sensitive and official parts of the project have been either partially or totally removed.
#Purpose
#The content provided here is intended to demonstrate the capabilities, design patterns, and methodologies employed in the project. It is not meant for production use and may lack certain functionalities that are part of the full, official version.
#Limitations
#- **Sensitive Information**: Any sensitive information, including but not limited to private keys, credentials, and personal data, has been removed or anonymized.
#- **Partial Functionality**: Some sections of the code may have been modified or omitted to ensure the security and privacy of the underlying system or data. As such, the repository may not represent the full functionality of the original project.
#- **Showcase Only**: The provided code and documents are intended for viewing and demonstration purposes only.

#   1) Load Library
#   2) Data Setup
#   3) Converting to a time series
# 
#   4) Time Series Insights: EDA and Beyond
#       - Descriptive statistics
#       - Time series plot
#       - Outliers
#       - Testing Normality 
#       - Decomposing Time series
#       - Autocorrelation and Ljung-Box test
#   5) Forecasts: 
#       5a) Seasonal Naive Method
#       5b) Holt-Winters Method
#       5c) Forecast combination
#       5d) ETS
#       5e) ARIMA
#       5f) Dynamic Regression
#       5g) TBATS 
#       
# 
#   6) Forecast evaluation:
#       6a) Residuals
#       6b) Accuracy
# 
#   7) Sensitivity analysis
#
#   8) Write the output to Excel and S3:
#      
#
# 
# Each step can be navigated from the document outline icon - top right (or Ctrl+Shift+O)


# 1) Load Library----
library(botor) 
library(readxl)
library(openxlsx)
library(reshape2)
library(tsibble)
library(dplyr)
library(lubridate)
library(tidyverse)
library(tidyr)
library(purrr)
library(psych)
library(ggplot2)
library(forecast)
library(seastests)
library(feasts)
library(fpp2)
library(TTR)
library(timetk)
library(Rdbtools)
library(seasonal)


# 2) Data setup ----
# 1.1  Set inputs - these will be used when extracting the raw data from the S3
case_type <- "InternalUseOnly" # how to select from data the case type name
period_start <- dmy("InternalUseOnly")

#1.2) Set the folder in which the final output will be uploade to in S3
output_path <- paste0("s3:InternalUseOnly/", case_type,"/")

#2) Load the data from S3 Athena
#Load the latest snapshot available in the data (latest date in InternalUseOnly_snapshot_date variable)

# InternalUseOnly SQL CODE


InternalUseOnly_extract <- read_sql(InternalUseOnly_sql)

# 3) Case type data----
InternalUseOnly<- InternalUseOnly_extract %>% 
  group_by(issue_month, case_type_name_group) %>% 
  summarise(cases = n())


#3.1) MANUAL STEP - check whether the last row of the data is incomplete or not.
#If the raw gets updated at a point where the latest data is incomplete, then it should be removed to not skew the forecast.
#The code below will be commented out, but should be run in case the row needs to be removed.

# case_numbers <- case_numbers %>% 
#   head(-1)


# 3.2) Converting to a time series
InternalUseOnly_ts <- ts(InternalUseOnly$cases, start = c(InternalUseOnly,1), frequency = 12)

### Time series plot ----
png(file="InternalUseOnly_plot.jpeg")


autoplot(InternalUseOnly_ts) + labs(x = "Years", y = "Number of Cases",
                               title = "Time series plot: InternalUseOnly Cases")


dev.off()


# 4) Time Series Insights: EDA and Beyond----
### Descriptive statistics ----
#describe(sInternalUseOnly$cases)
desc_stat_InternalUseOnly <- describe(InternalUseOnly$cases)

# Seasonal plots
png(file="InternalUseOnly_Seasonal_plot.jpeg")

ggseasonplot(InternalUseOnly_ts) + labs(x = "Years", y = "Number of Cases",
                                          title = "Seasonal plot: InternalUseOnly Cases")


dev.off()

### Outliers ----
# Outliers based on Time series data
# It comes down to judgement to determine whether you want to clean the data of outliers or not. 

# Flag outliers
find_ts_outlier_InternalUseOnly <- tsoutliers(InternalUseOnly_ts)$index
#Clean outliers and replace missing values. 
#Read into the tsclean() function by running ?tsclean to understand how the replacement is done. 
#The calculation is based on STL decomposition and linear interpolation. 
#Code has been commented, but uncomment and run if required.
#time_series <- ts_clean(time_series, replace.missing = TRUE)

# Creating table time series outliers
Cases_ts_InternalUseOnly <- InternalUseOnly[find_ts_outlier_InternalUseOnly,3]
Data_ts_InternalUseOnly <- InternalUseOnly[find_ts_outlier_InternalUseOnly,1]
ts_outliers_InternalUseOnly <-  as.data.frame(tsoutliers(InternalUseOnly_ts))
ts_outliers_InternalUseOnly$Outliers <- "Time Series InternalUseOnly"
ts_outliers_InternalUseOnly <- cbind(ts_outliers_InternalUseOnly,Data_ts_InternalUseOnly,Cases_ts_InternalUseOnly) %>% relocate(Outliers)


# Outliers based on z-score
# Outliers based on z-score function
### Functions
# create function for outlier based on z-score
# Note: 3.29 is a value recommended that makes sense in a wide range of solutions (Warner, R. M. (2020). Applied statistics II: Multivariable and multivariate techniques. Sage InternalUseOnlyications). 
# It's been expected that no cases above 3,29 - these cases are significant outliers (to be greater than about 3.29 9 (Field, A., Miles, J., & Field, Z. (2012), Discovering statistics using R.)
# outlier_z_score_above <-  function(x) {
#outlier_z_score_above <-  function(x) {
 # 3.29*sd(x, na.rm = TRUE) + mean(x, na.rm = TRUE)
#}
#outlier_z_score_below <-  function(x) {
  #3.29*sd(x, na.rm = TRUE) - mean(x, na.rm = TRUE)
#}

outlier_z_score_above(InternalUseOnly_ts)
outlier_z_score_below(InternalUseOnly_ts)

z_score_above_InternalUseOnly <- round(outlier_z_score_above(InternalUseOnly_ts), 0)
z_score_below_InternalUseOnly <- round(outlier_z_score_below(InternalUseOnly_ts), 0)
z_score_InternalUseOnly <- as.data.frame(cbind(z_score_above_InternalUseOnly, z_score_below_InternalUseOnly))
z_score_InternalUseOnly$Outliers <- "z-score InternalUseOnly"
z_score_InternalUseOnly<- z_score_InternalUseOnly %>% relocate(Outliers)

### Decomposing Time series ---- 
InternalUseOnly_ts_componets_add <- decompose(InternalUseOnly_ts)
InternalUseOnly_ts_componets_mlt <- decompose(InternalUseOnly_ts, type = "mult")
# ** Plots
png(file="dec_add_InternalUseOnly.jpeg")


plot(InternalUseOnly_ts_componets_add)


dev.off()


png(file="dec_mtl_InternalUseOnly.jpeg")


plot(InternalUseOnly_ts_componets_mlt)


dev.off()



### Autocorrelation and Ljung-Box test ----
# The null Hypothesis H0 is that the residuals are independently distributed. 
# The alternative hypothesis is that the residuals are not independently distributed and exhibit a serial correlation.)
# p-value < .05 means the residuals contain an autocorrellation 
#  Ljung-Box test: a more formal test for autocorrelation, in addition to looking at the ACF plot
ggAcf(InternalUseOnly_ts) + labs(y = "Number of Cases",
                                  title = "Autocorrelation plot: InternalUseOnly")
# Ljung-Box test to test white noise
Box.test((InternalUseOnly_ts), lag = 24, fitdf = 0, type = "Lj")


# 5) Forecast ----
### 5a) Seasonal Naive Method ----

#The data that we deal with has a seasonality aspect to it. 
#When using forecasting techniques, it's important we keep this in mind and apply the appropriate forecasting method. 

#For naive forecasts in general, we set all future forecast values to be the same as the last observed value. 
#Seasonal naive forecasting is a variation of the standard naive forecast. 
#You set future forecasted values to be the same as the last observed value from the SAME SEASON. 
#For example, with monthly data, the forecast for all future February values is equal to the last observed February value. 
#With quarterly data, the forecast of all future Q2 values is equal to the last observed Q2 value
seasonal_naive_InternalUseOnly <- snaive(InternalUseOnly_ts, h = 60)

# Plot the forecast
png(file = "seasonal_naive__InternalUseOnly_plot.jpeg")

autoplot(seasonal_naive_InternalUseOnly)


dev.off()

# Create the "summary" of the forecast - this shows the forecast values we are interested in as well as the error measures. 
summary(seasonal_naive_InternalUseOnly)

# Convert the time series object into a tibble
seasonal_naive_table_InternalUseOnly <- tk_tbl(seasonal_naive_InternalUseOnly, rename_index = "Date", timetk_idx = "TRUE") %>% 
  mutate(Date = as.Date(paste0("01", Date), "%d%b%Y"))



### 5b) Holt-Winters method ----
#This technique is an extension of exponential smoothing that accounts for trend and seasonality.
holt_winters_InternalUseOnly <- hw(InternalUseOnly_ts, seasonal = "multiplicative", h = 60, damped = TRUE, phi = 0.9)

# Autoplot the forecast
png(file = "holt_winters_InternalUseOnly_plot.jpeg")


autoplot(holt_winters_InternalUseOnly)

dev.off()

# Create the summary
summary(holt_winters_InternalUseOnly)

# Convert time series object into tibble
holt_winters_table_InternalUseOnly <- tk_tbl(holt_winters_InternalUseOnly, rename_index = "Date", timetk_idx = "TRUE") %>% 
  mutate(Date = as.Date(paste0("01", Date), "%d%b%Y"))


### 5c) Forecast Combinations and final table Models----
#Taking an average of the point forecast of both seasonal naive and Holt-Winters
fc_models_InternalUseOnly <- seasonal_naive_table_InternalUseOnly %>% 
  rename("seasonal_naive" = "Point Forecast") %>% 
  select(Date, seasonal_naive) %>% 
  left_join(., holt_winters_table_InternalUseOnly %>% rename("holt_winters" = "Point Forecast") %>% select(Date, holt_winters), by = "Date") %>% 
  mutate(Snl_HW_Combination = (seasonal_naive + holt_winters)/2)

### 5d)  ETS: Error, Trend, Seasonality ----
# ETS provides a completely automatic way of producing forecasts 
## for a wide range of time series.
ets_InternalUseOnly <- ets(InternalUseOnly_ts)
checkresiduals(ets_InternalUseOnly)
# ETS(A,A,A) - Additive Holt-Winters Method; residuals not white noise
ets_InternalUseOnly<- ets(InternalUseOnly_InternalUseOnly)
checkresiduals(ets_InternalUseOnly)
# ETS (M,Ad,M)	Holt-Winters’ damped method; residuals not white noise, 
ets_InternalUseOnly <-ets(InternalUseOnly_InternalUseOnly)
checkresiduals(ets_InternalUseOnly)
# ETS (M,Ad,N) Additive damped trend method; residuals not white noise

# PLot Model based on ETS
InternalUseOnly_ts %>% ets() %>%  forecast(h = 24) %>% autoplot() + labs(y = "Number of Cases",
                                                              subtitle = "InternalUseOnlyic Cases: 2015-2021")
InternalUseOnly_InternalUseOnly%>% ets() %>% forecast(h = 32) %>% autoplot() + 
  labs(y = "Number of Cases",
       subtitle = "InternalUseOnlyic Cases: 2015-2020")
InternalUseOnly_InternalUseOnly %>% ets() %>% forecast(h = 48) %>% autoplot() + 
  labs(y = "Number of Cases",
       subtitle = "InternalUseOnlyic Cases: 2015-2019")
## note: plot actual date over
## ETS doesn't work for InternalUseOnly series (it might be cyclic data)

# Transformations ----
# Box-Cox transformations for time series
autoplot(InternalUseOnly_ts)
BoxCox.lambda(InternalUseOnly_ts)
InternalUseOnly_ts %>% BoxCox(lambda = 1.406798) %>% autoplot()

# Making a time series stationary for non-seasonal data
## Diff(), log-1 differences:  
autoplot(InternalUseOnly_ts)
autoplot(diff(InternalUseOnly_ts))
# Plot the ACF of the differenced InternalUseOnlyic law
ggAcf(diff(InternalUseOnly_ts)) + labs(y = "Number of Cases",
                            title = "Differecing Autocorrelation plot: InternalUseOnlyic Law Cases")
# Ljung-Box test to test white noise
Box.test((diff(InternalUseOnly_ts)), lag = 24, fitdf = 0, type = "Lj")

# Making Seasonal difference for stationary
## Diff() for Lag value and log transformations: 
autoplot(InternalUseOnly_ts)
# Take logs and seasonal differences of h02
difflog_InternalUseOnly <- diff(log(InternalUseOnly_ts), lag = 12)
autoplot(difflog_InternalUseOnly)
# Take another difference and plot
diff_difflog_InternalUseOnly <- diff(difflog_InternalUseOnly)
autoplot(diff_difflog_InternalUseOnly)
# Plot ACF of diff_difflog_InternalUseOnly
ggAcf(diff_difflog_InternalUseOnly)



### 5e) ARIMA for non-seasonal time series ----
# Fit an automatic ARIMA model  
## ARIMA(p,d,q) + c=drift or constant (trend); p=outregressio;d=differencing; q= lagged error
### With two lots of differencing (d) of any kind, the forecast has a trend without including the constant 

# Forecasting with ARIMA models: The model specification makes a big impact on the forecast!
# Plot forecasts from an ARIMA(1,0,0) = first-order autoregressive mode
# Plot forecasts from ARIMA(0,1,0) = random walk
# Plot forecasts from ARIMA(1,1,0) = differenced first-order autoregressive model
# Plot forecasts from an ARIMA(0,1,1) without constant = simple exponential smoothing
# Plot forecasts from an ARIMA(0,1,1) with constant = simple exponential smoothing with growth
# Plot forecasts from an ARIMA(0,2,1) or (0,2,2) without constant = Linear exponential smothing  
# Plot forecasts from an ARIMA(1,1,2) without constant = damped-trend linear exponential smoothing

# Use the parameter genereted by auto.arima() or  
## ARIMA(0,1,1)(1,0,0)[12]; nota this data is for seasonal Arima, it's just e.g.
### ARIMA(0,1,1) = linear exponential smoothing
InternalUseOnly_ts %>% Arima(order = c(0,1,1), include.constant = FALSE) %>% forecast() %>% autoplot()
# A model with d=1 and no constant, forecasts converge to a non-zero value close to the last observation.
InternalUseOnly_ts %>% Arima(order = c(1,0,0), include.constant = FALSE) %>% forecast() %>% autoplot()
#  A model with d=0 and no constant, forecasts converge to 0.
InternalUseOnly_ts %>% Arima(order = c(1,0,0), include.constant = TRUE) %>% forecast() %>% autoplot()
#  A model with d=0 and a constant, forecasts converge to the mean of the data.
InternalUseOnly_ts %>% Arima(order = c(0,1,0), include.constant = FALSE) %>% forecast() %>% autoplot()
#  A model with d=1 and no constant, forecasts converge to a non-zero value close to the last observation.
InternalUseOnly_ts %>% Arima(order = c(1,1,0), include.constant = FALSE) %>% forecast() %>% autoplot()
# Simple exponential smoothing: Another strategy for correcting autocorrelated errors in a random walk model 
InternalUseOnly_ts %>% Arima(order = c(0,1,1), include.constant = FALSE) %>% forecast() %>% autoplot()
InternalUseOnly_ts %>% Arima(order = c(0,1,1), include.constant = TRUE) %>% forecast() %>% autoplot()
# #  A model with d=1 and a constant, forecasts converge to  a linear function with slope based on the whole series
InternalUseOnly_ts %>% Arima(order = c(0,2,1), include.constant = FALSE) %>% forecast() %>% autoplot()
#  A model with d=2 and no constant, forecasts converge to  a linear function with slope based on the last few observations.
InternalUseOnly_ts %>% Arima(order = c(1,1,1), include.constant = FALSE) %>% forecast() %>% autoplot()

# Comparing arima and ets method on non-seasonal data using errors
# Set up forecast functions for ETS and ARIMA models
#fets <- function(x, h) {
 # forecast(ets(x), h = h)
#}
#farima <- function(x, h) {
 # forecast(auto.arima(x), h = h)
#}
# Compute CV errors for ETS on austa as e1
e1 <- tsCV(InternalUseOnly_ts, fets, h = 12)
# Compute CV errors for ARIMA on austa as e2
e2 <- tsCV(InternalUseOnly_ts, farima, h = 12)
# Find MSE of each model class
mean(e1^2, na.rm = TRUE)
mean(e2^2, na.rm = TRUE)
# Plot 10-year forecasts using the best model class
austa %>% farima(h = 10) %>% autoplot()

# Seasonal ARIMA models ----
## ARIMA(p,d,q) (P,D,Q)m; m= seasoanl period (number observations each year of data)
## P, D and Q, all uppercase, referring to seasonal versions
### lambda = 0: applying a log transformation

# Check that the logged InternalUseOnlyic law data have stable variance
InternalUseOnly_ts %>% log() %>% autoplot()
autoarima_InternalUseOnly <- auto.arima(InternalUseOnly_ts, lambda = 0)
# ARIMA(0,1,1)(1,0,0)[12] 
checkresiduals(autoarima_InternalUseOnly)
# Residuals ~ white noise
summary(autoarima_InternalUseOnly)
# Find the AICc value and the number of differences used
AICc <- InternalUseOnly
d <- InternalUseOnly
D <- InternalUseOnly
# Plot forecasts of fit
autoarima_InternalUseOnly %>% forecast(h = 12) %>% autoplot()

# To make auto.arima() work harder to find a good model, 
## add the optional argument stepwise = FALSE to look at a much larger collection of models.
# Find an ARIMA model for euretail
autoarima_InternalUseOnly_0 <- auto.arima(InternalUseOnly_ts)
# Don't use a stepwise search
autoarima_InternalUseOnly_1 <- auto.arima(InternalUseOnly_ts, stepwise = FALSE)
# AICc of better model
summary(autoarima_InternalUseOnly_0)
summary(autoarima_InternalUseOnly_1)
AICc <- 1029.21 # same AICc

# ARIMA on seasonal data 
# # Fit an ARIMA
fit_arima_InternalUseOnly_InternalUseOnly<- auto.arima(InternalUseOnly_InternalUseOnly)
# # ARIMA(1,1,1)(2,0,0)[12] 
# # Check that both models have white noise residuals
checkresiduals(fit_arima_InternalUseOnly_InternalUseOnly)
# # Looks white noise
# 
# # Produce forecasts for each model: to InternalUseOnly
# ## Note: Set h to the number of total period time in your test set
fc_arima_InternalUseOnly_InternalUseOnly<- forecast(fit_arima_InternalUseOnly_InternalUseOnly, h = 36)

# Comparing auto.arima() and ets() on seasonal data 
# training data
InternalUseOnly_InternalUseOnly<- subset(InternalUseOnly_ts, end = 72)

# Fit an ARIMA and an ETS model to the training data
fit_arima_InternalUseOnly_InternalUseOnly<- auto.arima(InternalUseOnly_InternalUseOnly)
fit_ets_InternalUseOnly_InternalUseOnly<- ets(InternalUseOnly_InternalUseOnly)

# Check that both models have white noise residuals
checkresiduals(fit_arima_InternalUseOnly_InternalUseOnly)
# Looks white noise
checkresiduals(fit_ets_InternalUseOnly_InternalUseOnly)
# Not looks white noise

# Produce forecasts for each model: to InternalUseOnly
## Note: Set h to the number of total quarters in your test set
fc_arima_InternalUseOnly_InternalUseOnly<- forecast(fit_arima_InternalUseOnly_InternalUseOnly, h = 12)
fc_ets_InternalUseOnly_InternalUseOnly<- forecast(fit_ets_InternalUseOnly_InternalUseOnly, h = 12)

# Use accuracy() to find best model based on RMSE
## You do not need to set up a test set. Just pass the whole of dataset 
#### as the test set to accuracy() and it will find the relevant part to use in comparing with the forecasts.
accuracy(fc_arima_InternalUseOnly_InternalUseOnly, InternalUseOnly_ts)
accuracy(fc_ets_InternalUseOnly_InternalUseOnly, InternalUseOnly_ts)
bettermodel <- fc_ets_InternalUseOnly_InternalUseOnly

### 5f) #  Dynamic regression ----
# Time plot of both variables
autoplot(InternalUseOnly, facets = TRUE)
# Fit ARIMA model
fit <- auto.arima(advert[, "InternalUseOnly"], xreg = InternalUseOnly[, "InternalUseOnly"], stationary = TRUE)
# Check model. Increase in sales for each unit increase in advertising
InternalUseOnly <- coefficients(fit)[3]
# Forecast fit as fc
## Forecast from the fitted model specifying the next InternalUseOnly
fc <- forecast(fit, xreg = rep(10, 6))
# Plot fc with x and y labels
autoplot(fc) + xlab("InternalUseOnly") + ylab("InternalUseOnly")

# Matrix of regressors
xreg <- cbind(MaxTemp = elec[, "InternalUseOnly"], 
          MaxTempSq = elec[,   "InternalUseOnly"]^2, 
            Workday = elec[, "InternalUseOnly"])

# Dynamic Harmonic regression ----
# Set up harmonic regressors of order 13
armonics <- fourier(InternalUseOnly, K = 13)
# Fit regression model with ARIMA errors
#it <- auto.arima(InternalUseOnly, xreg = harmonics, seasonal = FALSE)
# Forecasts next 3 years
InternalUseOnly <- fourier(InternalUseOnly, K = 13, h = 156)
fc <- forecast(fit, xreg = InternalUseOnly)
# Plot forecasts fc
autoplot(fc)

# Harmonic regression for multiple seasonality----
fit <- tslm(InternalUseOnly ~ fourier(InternalUseOnly, K = c(10, 10)))
# Forecast 
fc <- forecast(fit, newdata = data.frame(fourier(InternalUseOnly, K = c(10, 10), h = 20 * 48)))
# Plot the forecasts
autoplot(fc)
# Check the residuals of fit
checkresiduals(fit)

### 5g)  TBATS model ----
# Plot the gas data
autoplot(InternalUseOnly_ts)
# Fit a TBATS model to the gas data
fit <- tbats(InternalUseOnly_ts)
# Forecast the series for the next 1 years
fc <- forecast(fit, h = 12)
# Plot the forecasts
autoplot(fc)
# Record the Box-Cox parameter and the order of the Fourier terms
lambda <- InternalUseOnly
K <- InternalUseOnly



# 6) Forecast evaluation ----
# To improve forecast and provide a evidence-based analysis, there are four forecast evaluations put in place (please, note this is an ongoing work)
## a) Residuals' properties, which is found below
## b) Accuracy 
## c) Sensitivity analysis, which might be performed in this R code or in a separate excel file
## d) Models vs Actual, which is reported in a separate excel file


### 6a) Check Residuals ----
# Residuals are useful in checking whether a model has adequately captured the information in the data. 
#It is useful to evaluate various models and implement the models that perform better to deliver official forecasts.
 
# A good forecasting method will yield residuals with the following properties: 
# 1) Residuals are uncorrelated; 2) Residuals have zero mean; 
# Any forecasting method that does not satisfy these properties can be improved.
# Useful properties for prediction intervals: 
# 3) Residuals have constant variance; 4) Residuals are normally distributed
# source: https://otexts.com/fpp2/residuals.html

#  Check Residuals: Seasonal naive method
png(file="r_Sn_InternalUseOnly.jpeg")

checkresiduals(seasonal_naive_InternalUseOnly)

dev.off()

#  Ljung-Box test: a more formal test for autocorrelation, in addition to looking at the ACF plot
# Null Hypothesis is that the residuals are autocorrelated 
# p-value < .05 means the residuals contain an autocorrellation
Ljung_box_snl_InternalUseOnly <- checkresiduals(seasonal_naive_InternalUseOnly)
Ljung_box_snl_InternalUseOnly <- as.matrix(Ljung_box_snl_InternalUseOnly$p.value)
colnames(Ljung_box_snl_InternalUseOnly) <- 'Ljung_box_Snl, p-value (InternalUseOnly)'

#  Check Residuals: Holt-Winters method
png(file="r_HW_InternalUseOnly.jpeg")

checkresiduals(holt_winters_InternalUseOnly)

dev.off()

#  Ljung-Box test: a more formal test for autocorrelation, in addition to looking at the ACF plot
# Null Hypothesis is that the residuals are autocorrelated
# p-value < .05 means the residuals contain an autocorellation 
Ljung_box_HW_InternalUseOnly <- checkresiduals(holt_winters_InternalUseOnly)
Ljung_box_HW_InternalUseOnly <- as.matrix(Ljung_box_HW_InternalUseOnly$p.value)
colnames(Ljung_box_HW_InternalUseOnly) <- 'Ljung_box_HW, p-value (InternalUseOnly)'
####***  



### 6b) Accuracy ----
# It is important to evaluate forecast accuracy using genuine forecasts. 
# Consequently, the size of the residuals is not a reliable indication of how large true forecast errors are likely to be.
# The accuracy of forecasts can only be determined by considering how well a model performs on new data 
# that were not used when fitting the model.
# When choosing models, it is common practice to separate the available data into two portions, 
# training data (it is used to estimate any parameters of a forecasting method) 
# test data (it is used to evaluate its accuracy)

## Training and test data
accuracy1_trainingdata_InternalUseOnly <-  "Training data = InternalUseOnly"
accuracy1_testdata_InternalUseOnly <- "Test data = InternalUseOnly"
trainingdata__accuracy1_InternalUseOnly <- subset(InternalUseOnly_ts, end = 87)
testdata_accuracy1_InternalUseOnly <- window(InternalUseOnly_ts, start = c(InternalUseOnly, 4))

# Forecast using training data
#  Seasonal Naive Method
sln_trainingdata_accuracy1_InternalUseOnly <- snaive(trainingdata__accuracy1_InternalUseOnly, h = 60)
# Holt-Winters method
hw_trainingdata_accuracy1_InternalUseOnly <- hw(trainingdata__accuracy1_InternalUseOnly, seasonal = "multiplicative", h = 60, damped = TRUE, phi = 0.9)

# Calculate accuracy
#  Seasonal Naive Method
seasonal_naive_accuracy1_InternalUseOnly <- accuracy(sln_trainingdata_accuracy1_InternalUseOnly, testdata_accuracy1_InternalUseOnly) 
# Holt-Winters method
holt_winters_accuracy1_InternalUseOnly <- accuracy(hw_trainingdata_accuracy1_InternalUseOnly, testdata_accuracy1_InternalUseOnly)

# Put them together in one table
# We can measure forecast accuracy by summarising the forecast errors in different ways.
# Two common ways are: RMSE (Root mean squared error) and MAPE (Mean absolute percentage error)
sln_accuracy_rmse_mape_InternalUseOnly <- seasonal_naive_accuracy1_InternalUseOnly[,c("RMSE", "MAPE")] 
hw_accuracy_rmse_mape_InternalUseOnly <- holt_winters_accuracy1_InternalUseOnly[,c("RMSE", "MAPE")]
models_accuracy_rmse_mape_InternalUseOnly <- as.table(cbind(sln_accuracy_rmse_mape_InternalUseOnly,hw_accuracy_rmse_mape_InternalUseOnly))
colnames(models_accuracy_rmse_mape_InternalUseOnly) <- c("Seasonal_RMSE", "Seasonal_MAPE", "HW_RMSE", "HW_MAPE")
models_accuracy_rmse_mape_InternalUseOnly # for worksheet
# Look at the lower value of RMSE or MASE for test data

# 7) Sensitivity analysis ---- 
### s1: changing Alpha parameter = 0.81 for Holt-Winters method based on the past forecast dated InternalUseOnly
s1_name <- "s1_Holt-Winters"
s1_adjustment <- "From InternalUseOnly Holt-Winters Forecast Alpha = 0.81"
s1_holt_winters_InternalUseOnly <- hw(InternalUseOnly_ts, seasonal = "multiplicative", h = 60, damped = TRUE, phi = 0.9, alpha = 0.81)
summary(s1_holt_winters_InternalUseOnly)

# Convert time series object into tibble
s1_holt_winters_InternalUseOnly <- tk_tbl(s1_holt_winters_InternalUseOnly, rename_index = "Date", timetk_idx = "TRUE") %>% 
  mutate(Date = as.Date(paste0("01", Date), "%d%b%Y"))


# 8) Write output Excel and S3 ----

# Set file name
#Extract date of current extract
snapshot_date <- InternalUseOnly_extract %>% 
  distinct(InternalUseOnly_snapshot_date) %>% 
  pull() %>% 
  format("%Y%m")

filename <- paste0(snapshot_date, "_", case_type, "_R_forecast.xlsx")


# # Create a blank workbook
wb <- createWorkbook()
# Add some sheets to the workbook
addWorksheet(wb, "Raw_InternalUseOnly")
addWorksheet(wb, "Forecast_Models_InternalUseOnly")
addWorksheet(wb, "SnNaive_InternalUseOnly")
addWorksheet(wb, "HW_InternalUseOnly")
addWorksheet(wb, "Residuals_InternalUseOnly")
addWorksheet(wb, "Accurancy_InternalUseOnly")
addWorksheet(wb, "Sensitivity")


# Write the data to the sheets
writeData(wb, sheet = "Raw_InternalUseOnly", InternalUseOnly, startRow = 3, startCol = 1)
writeData(wb, sheet = "Raw_InternalUseOnly", desc_stat_InternalUseOnly, startRow = 4, startCol = 5)
writeData(wb, sheet = "Raw_InternalUseOnly", ts_outliers_InternalUseOnly, startRow = 7, startCol = 5)
writeData(wb, sheet = "Raw_InternalUseOnly", z_score_InternalUseOnly, startRow = 13, startCol = 5)
#writeData(wb, sheet = "Raw_InternalUseOnly", iqr_outliers_InternalUseOnly, startRow = 7, startCol = 11)
#writeData(wb, sheet = "Raw_InternalUseOnly", shap_test_InternalUseOnly, startRow = 16, startCol = 5)
insertImage(wb, sheet = "Raw_InternalUseOnly", file = "InternalUseOnly_plot.jpeg", startRow = 21, startCol = 5, width = 7, height = 4.5, units = "in")
insertImage(wb, sheet = "Raw_InternalUseOnly", file = "InternalUseOnly_Seasonal_plot.jpeg", startRow = 45, startCol = 5, width = 8, height = 5.5, units = "in")

writeData(wb, sheet = "Forecast_Models_InternalUseOnly", fc_models_InternalUseOnly)
insertImage(wb, sheet = "Forecast_Models_InternalUseOnly", file = "dec_add_InternalUseOnly.jpeg", startRow = 1, startCol = 7, width = 5, height = 3.5, units = "in")
insertImage(wb, sheet = "Forecast_Models_InternalUseOnly", file = "dec_mtl_InternalUseOnly.jpeg", startRow = 20, startCol = 7, width = 5, height = 3.5, units = "in")

writeData(wb, sheet = "SnNaive_InternalUseOnly", seasonal_naive_table_InternalUseOnly)
insertImage(wb, sheet = "SnNaive_InternalUseOnly", file = "seasonal_naive__InternalUseOnly_plot.jpeg", startRow = 3, startCol = 8, width = 7, height = 4.5, units = "in")

writeData(wb, sheet = "HW_InternalUseOnly", holt_winters_table_InternalUseOnly)
insertImage(wb, sheet = "HW_InternalUseOnly", file = "holt_winters_InternalUseOnly_plot.jpeg", startRow = 3, startCol = 8, width = 7, height = 4.5, units = "in")

insertImage(wb, sheet = "Residuals_InternalUseOnly", file = "r_Sn_InternalUseOnly.jpeg", width = 7, height = 4.5, units = "in")
writeData(wb, sheet = "Residuals_InternalUseOnly", Ljung_box_snl_InternalUseOnly, startRow = 25, startCol = 2)
insertImage(wb, sheet = "Residuals_InternalUseOnly", file = "r_HW_InternalUseOnly.jpeg", startRow = 1, startCol = 11, width = 7, height = 4.5, units = "in")
writeData(wb, sheet = "Residuals_InternalUseOnly", Ljung_box_HW_InternalUseOnly, startRow = 25, startCol = 11)

writeData(wb, sheet = "Accurancy_InternalUseOnly", case_type, startRow = 1, startCol = 1)
writeData(wb, sheet = "Accurancy_InternalUseOnly", accuracy1_trainingdata_InternalUseOnly, startRow = 2, startCol = 1)
writeData(wb, sheet = "Accurancy_InternalUseOnly", accuracy1_testdata_InternalUseOnly, startRow = 3, startCol = 1)
writeData(wb, sheet = "Accurancy_InternalUseOnly", models_accuracy_rmse_mape_InternalUseOnly, startRow = 5)

writeData(wb, sheet = "Sensitivity", s1_name, startRow = 2, startCol = 1)
writeData(wb, sheet = "Sensitivity", s1_adjustment, startRow = 3, startCol = 1)
writeData(wb, sheet = "Sensitivity", s1_holt_winters_InternalUseOnly, startRow = 4, startCol = 1)

# Save Workbook
saveWorkbook(wb, filename, overwrite = T)


#Write the file to S3
s3_upload_file(filename, paste0(output_path, filename))


#Remove the file from the working directory ----
file.remove(filename)
# remove jpeg
jpeg.files <- list.files() %>% grepl(".jpeg", .)
list.files()[jpeg.files] %>% file.remove()


# add after QA performed on InternalUseOnly
### s1: changing Alpha parameter = 0.81 for Holt-Winters method based on the past forecast dated InternalUseOnly
ls1_name <- "ls5_Holt-Winters"
ls1_adjustment <- "Holt-Winters Forecast Alpha = 1"
ls1_holt_winters_InternalUseOnly <- hw(InternalUseOnly_ts, seasonal = "multiplicative", h = 60, damped = TRUE, phi = 0.9, alpha = 0.99)


