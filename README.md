# Redevelopment Forecasting Project

This README serves as a user guide for running the Redevelopment Forecasting Models in R (v0), held in InternalUseOnly repository.

# Project Showcase Notice
This repository contains code and materials for the purpose of showcasing the project. Please note that sensitive and official parts of the project have been either partially or totally removed to protect proprietary information and adhere to privacy and security guidelines.

Purpose
The content provided here is intended to demonstrate the capabilities, design patterns, and methodologies employed in the project. It is not meant for production use and may lack certain functionalities that are part of the full, official version.

Limitations
- **Sensitive Information**: Any sensitive information, including but not limited to private keys, credentials, and personal data, has been removed or anonymised.
- **Partial Functionality**: Some sections of the code may have been modified or omitted to ensure the security and privacy of the underlying system or data. As such, the repository may not represent the full functionality of the original project.
- **Showcase Only**: The provided code and documents are intended for viewing and demonstration purposes only.


## Contents

 
#### **Getting started** 

##### [a. Analytical Platform (AP) and AWS s3 access](#aps3) 

##### [b. Accessing the model from Github](#github) 

##### [c. Models Info](#modelsinfo) 

 

#### **User Instructions**

##### [d. A time series of actual InternalUseOnly volumes broken down by InternalUseOnly](#actual) 

##### [e. Actual Visualisation and Explonatory Analysis](#stats)

##### [f. Running Models](#run-models) 

##### [g. Forecast evaluation](#evaluation)  

##### [h. Output Excel](#output)  

 

#### **Technical Guidance**

##### [i. Models](#models) 

##### [j. R packages and versions](#rpack)  

##### [k. Parameters/Variables](#parameters) 

##### [l. Process flow Diagrams](#processflow)

##### [m. Assumptions and QA](#assumptionsQA) 
  

## Getting started

<a name="aps3"></a>

### a. Analytical Platform and AWS s3 access

You will initially need to have access to the Analytical Platform (AP). If you do not, guidance on how to set this up is found here:

<https://InternalUseOnly>

You will also need access to the 'InternalUseOnly-forecast' s3 bucket, and this can be setup by the admin for that bucket.

*How to add another AP user to a bucket* + Go to Analytical Platform Control Panel. + click the "Warehouse Data" tab. + click on the name of the bucket you want to manage access to. + type the platform username or email of the person to be added in the grant access input box. + click grant access.
  

<a name="github"></a>  

### b. Accessing the model from Github

Firstly (if you haven't done so already) you should link your AP account to GitHub following these instructions: <https://InternalUseOnly>

You should then follow the instructions in step 1 at the link below to clone the `InternalUseOnly_forecast_modelling` GitHub repository to the local area of your platform account. You can then run the model from R on your platform account. <https://InternalUseOnlyo>

Step 2 and beyond will be used should you want to amend code in the model scripts and push them back onto GitHub.

   

<a name="modelsinfo"></a>

### c. Models Info

**Aim of the model**: this model aims to forecast InternalUseOnly. The outputs are used by InternalUseOnly to plan  resourcing and to project income received by InternalUseOnly to assist with financial budgeting.

**Background**: InternalUseOnly

**Model Approach**: InternalUseOnly

**Data**: InternalUseOnly

## User Instructions
**Note: The number list of the present ReadMe file reflect the number list of R script modelling named 'v0_InternalUseOnly' (e.g., 1 is Load library for both files).**

<a name="actual"></a> 

### d. A time series of actual caseload volumes broken down by InternalUseOnly

**1.  Load the library**

Load the library and check all packages are loading correctly for you.

&nbsp;

**2.  Data setup**

**2.1  Set inputs**

These will be used when extracting the InternalUseOnly raw data from HMCTS Dataset.
To extract InternalUseOnly data, you need to replace the case type name with the InternalUseOnly you want to forecast.

*InternalUseOnly objects*

+ InternalUseOnly <- "InternalUseOnly". 
+ InternalUseOnly <- "InternalUseOnly".


Then, you need to insert the period of time when you want to extract data inside the *function dmy()*. 
```r
# For example, if you want to extract data from January 2015, the code would be as follows:
period_start <- dmy("01-01-2015")
```
&nbsp;

**2.2. InternalUseOnly**

&nbsp;

**2.3. Set the folder in which the final output will be upload to in s3**

We currently use *InternalUseOnly* folder in s3. 
```r 
# The code is as follows: 
output_path <- paste0("s3:InternalUseOnly/", InternalUseOnly,"/")
```
&nbsp;

**2.4. Load the data from s3 Athena**

Load the latest snapshot available in the data (latest date in InternalUseOnly_snapshot_date variable).
You need just to run the *InternalUseOnly_sql* function and create *InternalUseOnly_extract* object using *read_sql()*.

&nbsp;

**2.5. Extract your InternalUseOnly raw data**

2.5a You need just to rune the code - Make sure that the object name is the case type you want to model (this should have been changed when you have had replaced it as described by 2.2 section).

2.5b  The codes help you to perform a 'MANUAL STEP': check whether the last row of the data is incomplete or not.
If the raw gets updated at a point where the latest data is incomplete, then it should be removed to not skew the forecast.
The code below will be commented out, but should be run in case the row needs to be removed.

&nbsp;

**2.6. Converting raw data to a time series format**

The code converts raw data in a suitable format (time series format - ts) to run forecast models. As for 2.2 section, the object should be named based on your InternalUseOnly followed by _ts. **This object assignment avoids overwriting the raw data in your memory environment so that you are always able to come back to your raw data and check if the data you are working is right.**

```r
# For example, if you want to convert the raw data from January 2015 to August 2022 into time serie format, the code would be as follows: 
InternalUseOnly_ts <- ts(InternalUseOnly$cases, start = c(2015,1), end = c(2022,8), frequency = 12)
## The frequency parameter is the number of observations per unit of time. As the caseload dataset has monthly observations, the frequency parameter is 12. 
```
Note: It is important to correctly define your start and end date.

**Checking if the ts object has the same values as the raw data object is recommended**.  

&nbsp;
<a name="stats"></a>
&nbsp;

### e. Actual Visualisation and Explonatory Analysis

Before running models, it is crucial to understand your data. To accomplish this, the following section is about **actual visualisation** and some primary **explanatory analysis**. It also recommended looking at **previous forecasts** to understand what worked and what did not, their assumptions, and models, along with **contextualising** your data and case type.

&nbsp;

**3.1. Actual Visualisation**

The first analysis you might want to do is to visualise your raw data. The code chunk uses two functions that takes an object of type ts and creates a ggplot object suitable. As raw data might contain some seasonal aspects, the code displays and saves two plots: 1) The common *Time plot*; 2) *Seasonal plot* which is like a time plot except that the data are plotted against the seasons in separate years.

You need to do the following: 

+ Check if the arguments of png(), autoplot() and ggseasonpplot () are updated automatically based on your case type replaced in section 2.2.
+ Replace the case type name *InternalUseOnly* with your case type name in the title string. 

&nbsp;

**3.2. Descriptive statistics**

Descriptive statistics are based on the *describe() function* and calculate the most relevant **descriptive statistics** such as mean, ranges, skew, and so forth. A complete list of arguments can be found using R help. 

Along with providing statistical results, an additional benefit is that these results can be used to check the data. For example, the range statistics (min, max, range) are most useful for **data checking** to detect coding errors, and should be found in early analyses of the data. 

Check if the code is updated automatically based on your case type replaced in section 2.2.

```r
# For InternalUseOnly, the code would be as follows:
desc_stat_InternalUseOnly<- describe(InternalUseOnly$cases)
```
&nbsp;

**3.3. Outliers**

Identifying outiliers help you to spot an observation that appears to differ significantly from other observations in the data. An outlier tell you something unique about a situation that then you might want to discuss with other analysts and gather more intelligence. 

Once an outlier is found, it comes down to judgment to determine whether you want to reduce the impact of these observations by removing or replacing the values. A rule of thumb could be that you replace an outlier only if the outlier is related to measurement error or data error, or it is not part of the system/process/drivers.

We use two methods of identifying outliers: Outliers based on time series data and Outliers based on z-score. The results of both techniques are written into the Excel file.


**Outliers based on time series data**

The calculation is based on on STL decomposition and linear interpolation. The *tsclean()* function is to clean and replace outlier. Read into the *tsclean()* function by running ?tsclean to understand how the replacement is done. If you want to replace the outlier inside the data, you have to uncomment the code and run it. You can also replace manually the outliers by using other function or methods. If you replace an outlier, it is recommended to assign to the object a different name than the object that contains the outlier. 

Regardless if you want or not to replace the outlier, the information about whether your data contains outliers is written and exported into the Excel file. To accomplish this, the code creates a table which displays this information. 

You need to do the following: 

+ Check if the codes are updated automatically based on your case type replaced in section 2.2.
+ Replace the case type name *InternalUseOnly* with your case type name in the dataframe column of Outliers string. 


**Outliers based on z-score**

As identifying outliers is an important step, we use a second method: Z-score. Taking into account the results of both techniques, we can better understand what observation appears to differ significantly from other observations in the data.

The calculation is based on z-score. So, the function  uses mean and standard deviation - the value of z-score 3.29, approximately 3, meaning that the outlier is three times the standard deviation to the mean. Therefore, it's been expected that no cases above 3,29 - these cases are significant outliers (Field, A., Miles, J., & Field, Z. (2012), Discovering statistics using R). The 3.29 is a value recommended that makes sense in a wide range of solutions (Warner, R. M. (2020). Applied statistics II: Multivariable and multivariate techniques). 

The outliers based on z-score generate two values: z-score above the mean and z-score below the mean. Any value beyond this threshold is considered an outlier. 

You need to do the following: 

+ Check if the codes are updated automatically based on your case type replaced in section 2.2.
+ Replace the case type name *InternalUseOnly* with your case type name in the dataframe column of Outliers string. 



&nbsp;

**3.4. Autocorrelation and Ljung-Box test**

Just as correlation measures the extent of a linear relationship between two variables, autocorrelation measures the linear relationship between lagged values of a time series - they indicate that past values influence the current value. Note that time series that show no autocorrelation are called white noise. If a time series is NOT white noise, it might indicate that there are trends and seasonal patterns inside the time series, and you should further optimize the model.

Two methods are used to assess autocorrelation: 1) ACF plot using ggAcf() function - ACF stands for Autocorellation Function; 2) Ljung-Box test using Box.test() function.


Before running the codes, you need to do the following: 

+ Check if the codes are updated automatically based on your case type replaced in section 2.2.
+ Replace the case type name *InternalUseOnly* with your case type name in the title string. 


*1. Plot using ggAct() function*

When you run the code, you can see blue dashed lines. The dashed blue lines indicate whether the correlations are significantly different from zero. In details, these blue lines refer to 95% of the spikes in the ACF to lie within  
±2/√T where T is the length of the time series. If one or more large spikes are outside these bounds, or if substantially more than 5% of spikes are outside these bounds, then the series is probably not white noise. 

An example is shown below: 
InternalUseOnly


*2. Ljung-Box test using Box.test() function*

The Ljung-Box test is a more formal test for autocorrelation, in addition to looking at the ACF plot. It is hypothesis testing. The null Hypothesis is that the residuals are independently distributed. The alternative hypothesis is that the residuals are not independently distributed and exhibit a serial correlation. A p-value < .05 means the residuals contain an autocorrelation.

+ Check if the code is updated automatically based on your case type replaced in section 2.2.

Note: Generally, time series that uses a pure raw data is not white noise. For this reason, the autocorellation assessments can be evaluated in R console during the modelling, and there are not exported in the Excel file.

You can find more info about autorrecalltion in R on https://otexts.com/fpp3/acf.html and https://otexts.com/fpp2/wn.html, and by running in R console ?ggAcf() and ?Box.test(). 

An example of Ljung-Box test is shown below: 
InternalUseOnly



&nbsp;
<a name="run-models"></a>
&nbsp;

### f. Running Models

As discussed in 'Models Info' section, we currently use seven models:  

+ 5a) Holt-Winters
+ 5b) Seasonal Naïve
+ 5c) Forecast Combinations
+ 5d) ETS
+ 5e) ARIMA
+ 5f) Dynamic Regression
+ 5g) TBATS 

Running models in R-studio is quite straightforward, you only need to do the following: 

+ Check if the models are included in the R script.
+ Check if the function and parameters are written according to the 'i. Model' and 'k. Parameters/Variables' Guidance section of the present ReadMe. 




&nbsp;
<a name="evaluation"></a>
&nbsp;

### g. Forecast evaluation

To improve forecast and provide a evidence-based analysis, four forecast evaluations are put in place (please, note this is an ongoing work): 1) *Residuals*; 2) *Accuracy based on training and test data*; 3) *Sensitivity Analysis*; 4) *Accuracy based on Actual vs Models*.


**6.1. Residuals**

Residuals are useful in checking whether a model has adequately captured the information in the data. 
A good forecasting method will yield residuals with the following properties: 

+ a) Residuals are uncorrelated; 
+ b) Residuals have zero mean; 
+ c) Residuals have constant variance; 
+ d) Residuals are normally distributed.

Any forecasting method that does not satisfy *a* and *b* properties can be improved. Whereas any forecasting method that does not satisfy *c* and *d* properties will probably be quite good, but prediction intervals that are computed assuming a normal distribution may be inaccurate. 
You can find more info about residuals in R on https://otexts.com/fpp3/diagnostics.html and https://otexts.com/fpp2/residuals.html, and by running in R console ?checkresiduals(). 


We assess residuals using *checkresiduals* function. Similar to autocorrelation, we extract two types of information: 1) Visual using plot; 2) Formal test using Ljung_box test.

Before running the codes, you need to do the following for each model: 

+ Check if the codes are updated automatically based on your case type replaced in section 2.2.
+ Replace the case type name InternalUseOnly* with your case type name in the colnames Ljung-Box test string. 

*1. Visual using plot*

The visualisation consists of three diagnostic graphs.

An example is shown below: 
InternalUseOnly


The first graph on the top refers to the mean and variance of residuals. The second graph on the bottom left is residuals autocorrelaton. The third graph on the bottom right is the residual histogram to check if the residuals are normally distributed. 


*2. Ljung-Box test*

It is the same autocorrelation test on 3.4 section but this time we assess each model. We exported the value on the Excel file.

&nbsp;

**6.2. Accuracy based on training and test data**

It is important to evaluate forecast accuracy using genuine forecasts. Consequently, the size of the residuals is not a reliable indication of how large true forecast errors are likely to be. The accuracy of forecasts can only be determined by considering how well a model performs on new data that were not used when fitting the model. When choosing models, it is common practice to separate the available data into two portions: 

+ **Training data**: It is used to estimate any parameters of a forecasting method.
+ **Test data**: It is used to evaluate its accuracy.

*More info: https://otexts.com/fpp2/accuracy.html*

Before running the codes, you need to do the following for both training data and test data: 

+ Check if the object names are updated automatically based on your case type replaced in section 2.2.
+ Change the strings and the subset() and widow() functions based on how you want to separate the data. 

Note: When separating the data, consider how InternalUseOnly might affect your training data and test data.

*Forecast using training data*

Once you have the training data, it is used to forecast for each model as in section 5, but instead of using all the data, you will use the training data. Therefore, the process is similar to section 5; you only need to do the following:

+ Check if the object names are updated automatically based on your case type replaced in section 2.2.
+ Check if the models you want to evaluate accuracy are there, if not add them. 
+ Check if the parameters are fine with you, otherwise replace them. 

*Calculate accuracy*

At this point, you have forecasts based on your training data. It is time to calculate their accuracy on test data using the accuracy()function -  it takes training data for the first argument and test data for the second argument. 

As above, you need to do the following:

+ Check if the object names are updated automatically based on your case type replaced in section 2.2.
+ Check if the models you want to evaluate accuracy are there, if not add them. 

*Accuracy table*

The accuracy measures are exported by creating a table. We can measure forecast accuracy by summarising the forecast errors in different ways. Two common ways are: RMSE (Root mean squared error) and MAPE (Mean absolute percentage error), which are exported. If you are interested in other measures and/or prefer to create a table differently, feel free to change the code based on your needs. 

As usual, you need to do the following:

+ Check if the object names are updated automatically based on your case type replaced in section 2.2.
+ Check if the models you want to evaluate the accuracy are there, if not add them. 

*Accuracy evaluation*

The better model is the model that generates the lower value of RMSE or MASE for test data.

&nbsp;

**6.3 Sensitivity Analysis**

As a forecast evaluation, you might want to perform a sensitivity analysis. The reason for sensitivity analysis might be to replace outliers, adopt a new method/model, change parameters, and so on. A recommendation is to name your object correctly in R, and avoid overwriting objects. If you need to export the results, remember to add them to the output excel section.  

&nbsp;

**6.4 Accuracy based on Actual vs Models**

The 'accuracy based on Actual vs Models' is performed and reported in a separate Excel file. When the monthly InternalUseOnly is automatised in R, we might want to move this evaluation in R and export them in s3. An illustration can be found on `S:InternalUseOnly Forecast Modelling Guide(v0).docx`. 





&nbsp;
<a name="output"></a>
&nbsp;

### h. Output Excel

The output excel refers to the *d. section - 2.3* and the object name is **output_path**. Everything is set up such as date forecast and case type name, apart from the file name. So you need to do the following:

+ Check if the file name is appropriate (e.g. R_forecast).
+ Save the file. The file will be saved in your work environment.
+ Open and check if the file looks right, for instance, if it contains forecasts, plots, and so forth. 
+ Upload the file to s3.
+ Remove the excel file that you just saved it, and plots/pictures files from your work environment.
+ Push the code into GitHub. 
+ Download the file from s3 in DOM1. 

Note: We plan to make consistent name files and folders generated on InternalUseOnly forecast round across all the InternalUseOnly on s3 and DOM1. For more info, please refers to the related guidance and documentation or speak to the InternalUseOnly Team. 



&nbsp;
&nbsp;


## Technical Guidance  


<a name="models"></a>
&nbsp;

### i. Model

All of the three models are programmed in R, using the following script *v0_InternalUseOnly_Modelling.R*.


**Holt-Winters** 

This model is a time series technique that accounts for trend and seasonality by employing triple exponential smoothing. Three smoothing equations are related to: the **level** *ℓt*, **the trend** *bt*, the **seasonal component** *st* with the corresponding smoothing **parameters** *alpha: α, beta: β, gamma: γ*. We use Holt-Winters multiplicative method, which has the following technical details:

+ *alpha* - the alpha parameter of the Holt-Winters filter. Specifies how to smooth the level component. If numeric, it must be within the half-open unit interval (0, 1]. A small value means that older values in x are weighted more heavily. Values near 1.0 mean that the latest value has more weight.
+ *beta* - 	the beta parameter of the Holt-Winters filter. Specifies how to smooth the trend component. If numeric, it must be within the unit interval [0, 1]. A small value means that older values in x are weighted more heavily. Values near 1.0 mean that the latest value has more weight.
+ *gamma* - the gamma parameter of the Holt-Winters filter. Specifies how to smooth the seasonal component. If numeric, it must be within the unit interval [0, 1]. A small value means that older values in x are weighted more heavily. Values near 1.0 mean that the latest value has more weight.
https://docs.tibco.com/pub/enterprise-runtime-for-R/3.1.1/doc/html/Language_Reference/stats/HoltWinters.html

The function  is *hw()* and parameters are estimated by minimizing the sum of squared errors. A list and description of the parameters used in the modelling can be found in the 'Parameters' section. 



**Seasonal Naïve**

This model is a time series technique that uses **seasonal observation from the last year of data**. 
The forecast for time T + h is written as


where m = the seasonal period, and k is the integer part of (h-1)/m (i.e., the number of complete years in the forecast period prior to time T = h). As for Holt-Winters, a list and description of the parameters can be found in the 'Parameters' section. 


**Combinations**

The combination model is a **simple average** among models. 

**ETS**

The ETS (Error, Trend, Seasonality) model is a versatile time series forecasting approach that extends exponential smoothing to effectively model data exhibiting trends and seasonal patterns. It encompasses three principal components: Error (E), Trend (T), and Seasonality (S), each with specific characteristics and parameters. This model is adept at forecasting in scenarios where data shows evident trends and seasonal behaviors.

## Model Components and Parameters

ETS model is characterized by:

- **Error Component (E):** Manages the treatment of residuals. Options include Additive (A) for constant error variance across all levels, and Multiplicative (M) for proportionally changing variance with the level.

- **Trend Component (T):** Describes the data's trend handling. Available types are No trend (N), Additive (A) for linear trends, and Multiplicative (M) for exponential growth trends.

- **Seasonal Component (S):** Outlines how seasonality is addressed. Choices are No seasonality (N), Additive (A) for constant seasonal effects, and Multiplicative (M) for seasonally varying effects.

### Using the `ets()` Function

The `ets()` function in R is employed for fitting an ETS model to your data. It offers automatic selection of the best fitting model based on criteria like the Akaike Information Criterion (AIC).

#### Key Function Parameters:

- **model:** Defines the model with a three-character code (e.g., "AAA" for Additive errors, trends, and seasonality). "ZZZ" allows for automatic model selection.

- **damped:** Indicates if the trend component should be damped, useful for preventing unrealistic long-term forecasts.

- **alpha, beta, gamma:** These are smoothing parameters for the level, trend, and seasonal components, controlling the impact of the most recent observations. They are optimized automatically if not specified.

### Forecasting

Once an ETS model is fitted using `ets()`, the `forecast()` function can generate forecasts. This includes point forecasts, confidence intervals, and prediction intervals, providing a detailed future outlook.

## Reference
For a comprehensive understanding of the ETS model's theoretical underpinnings and its application in R, refer to *Forecasting: Principles and Practice (2nd ed.)* by Rob J Hyndman and George Athanasopoulos, available at [https://otexts.com/fpp2/ets.html](https://otexts.com/fpp2/ets.html).


**ARIMA**

The ARIMA (AutoRegressive Integrated Moving Average) model is a popular and widely used statistical approach for time series forecasting. It combines autoregressive features, differencing (to achieve stationarity), and moving average components to model data with trends and non-seasonal patterns. ARIMA models are characterized by their flexibility in handling a wide range of time series data.

## Model Components

ARIMA models are specified by three primary parameters (p, d, q):

- **p: Autoregressive order** - The number of lag observations included in the model, or the lag order. This represents the relationship between an observation and a specified number of lagged observations.

- **d: Differencing order** - The number of times that the raw observations are differenced to make the time series stationary. Stationarity is a crucial aspect of ARIMA modeling, as non-stationary data can lead to unreliable forecasts.

- **q: Moving average order** - The size of the moving average window, or the order of the moving average. This parameter models the relationship between an observation and a residual error from a moving average model applied to lagged observations.

### Using the `auto.arima()` Function in R

The `auto.arima()` function from the `forecast` package in R simplifies the process of fitting an ARIMA model by automatically selecting the best parameters (p, d, q) based on AIC, BIC, or AICc.

#### Key Function Parameters:

- **x:** The time series data to be modeled.

- **seasonal:** Logical flag indicating whether to fit a seasonal ARIMA model. Defaults to `TRUE`.

- **stepwise:** Logical flag indicating whether to use stepwise search to find the best model. Defaults to `TRUE`.

- **approximation:** Logical flag indicating whether to use an approximation to speed up the search for the best model. Useful for large datasets. Defaults to `FALSE` for small datasets.

- **trace:** Logical flag indicating whether to trace the search for the best model. Useful for debugging or understanding the model selection process.

### Forecasting

After fitting an ARIMA model using `auto.arima()`, the `forecast()` function can generate forecasts. This function provides point forecasts, confidence intervals, and prediction intervals, offering a comprehensive view of the model's future predictions.

## Reference
For detailed insights into ARIMA modeling and its application in R, consult *Forecasting: Principles and Practice (3rd ed.)* by Rob J Hyndman and George Athanasopoulos, available at [https://otexts.com/fpp3/arima.html](https://otexts.com/fpp3/arima.html).

**Dynamic Regression**

Dynamic Regression models extend traditional regression analysis by incorporating time series elements into the regression framework. These models are particularly useful for modeling and forecasting time series data where the response variable is influenced by its own past values and/or the past values of other predictor variables. This approach is often applied in econometrics, finance, and other fields where time series data are analyzed.

## Model Overview

Dynamic Regression models can handle both stationary and non-stationary time series. They often include autoregressive (AR) terms, moving average (MA) components, lagged variables, and other time-varying covariates. This flexibility allows for the modeling of complex temporal dynamics and the capture of relationships that evolve over time.

### Key Components:

- **Lagged Response Variables:** Incorporating past values of the response variable to account for autocorrelation.

- **Lagged Predictors:** Using past values of predictor variables to capture their delayed effects on the response variable.

- **ARIMA Errors:** Modeling the error term as an ARIMA process to account for autocorrelation and non-stationarity in the residuals.

### Using the `dynlm()` Function in R

The `dynlm()` function from the `dynlm` package in R offers a convenient way to fit Dynamic Regression models. It provides a formula interface similar to `lm()` but is specifically designed to handle time series objects and incorporate lagged terms easily.

#### Key Function Parameters:

- **formula:** A symbolic description of the model to be fitted. The formula can include lagged terms using `L()`, e.g., `L(variable, 1)` for a 1-period lag.

- **data:** The dataset containing the variables in the model. Should ideally be a time series object or data frame.


**TBATS**

The TBATS model (an acronym for Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend, and Seasonal components) is an advanced time series forecasting method that is particularly suited for handling data with multiple seasonal patterns, non-integer seasonality, and high frequency. Unlike simpler exponential smoothing models, TBATS can accommodate complex seasonal structures through a decomposition approach that includes a Fourier series to model seasonality, a Box-Cox transformation to stabilize variance, and ARMA models for capturing error autocorrelation.

### Technical Details:

+ **Box-Cox Transformation**: Adjusts the series to stabilize variance across time. The transformation parameter *λ* can take any real value, with *λ=0* equivalent to a log transformation.

+ **Fourier Series for Seasonality**: Uses a trigonometric representation of seasonal components to model complex and multiple seasonal patterns. The number of harmonics *K* is selected based on model fit.

+ **ARMA Errors**: Incorporates autoregressive (AR) and moving average (MA) components to model residuals, improving forecast accuracy by capturing autocorrelation in the error terms.

+ **Trend Component**: TBATS includes an option for a damped or undamped trend to model the time series trend behavior over time.

### Using the `tbats()` Function:

The `forecast` package in R provides the `tbats()` function for fitting a TBATS model to your time series data. The function automatically selects the best fitting model based on the data's characteristics.


&nbsp;

<a name="rpack"></a>
&nbsp;

### j. R packages and versions

R version 4.1.2 (2021-11-01)

+ **botor** v0.3.0
+ **readxl** v1.3.1
+ **openxlsx** v4.2.5
+ **reshape2** v1.4.4
+ **tsibble** v1.1.1
+ **dplyr** v1.0.8
+ **lubridate** v1.8.0
+ **tidyverse** v.1.3.1
+ **tidyr** v1.2.0
+ **purrr** v0.3.4
+ **psych** v2.2.5
+ **ggplot2** v3.3.5
+ **forecast** v8.16
+ **seastests** v0.15.4
+ **feasts** v.2.2
+ **fpp2** v2.4
+ **TTR** v0.24.3
+ **timetk** vtimetk
+ **Rdbtools** v0.2.0
+ **seasonal** v1.9.0



&nbsp;
&nbsp;

<a name="parameters"></a>
&nbsp;

### k. Parameters/Variables

Parameters/Variables marked **Constant** would not be expected to change from run to run of the model. Parameters/Variables marked **UPDATE** will require updating each time you run a new forecast round, but the others should be reviewed as to whether the input has changed and needs to be updated.

However, please update the documentation if you want to add/edit analysis/models or improve the coding. You might need to update the version. 

To facilitate the navigation, the parameters/variables are grouped based on the present guidance sections:

**d. A time series of actual caseload volumes broken down by InternalUseOnly case types**

+ `InternalUseOnly` _UPDATE_- name of case type (e.g. "InternalUseOnly").
+ `period_start` _UPDATE_ - period start from extracting data (e.g, "InternalUseOnly").
+ `output_path` _Constant_ - location of output excel file on s3. However, we plan to make consistent name files and folders generated on InternalUseOnly forecast round across all the case types on s3.
+ `InternalUseOnly` _UPDATE_ - object that contains data on the case type you want to forecast. If needed, for instance, you want to forecast InternalUseOnly case type, you can change it by following the instructions in 2.2.


**e. Actual Visualisation and Explonatory Analysis**

+ `file` _Constant_ - the file name plots. It will be changed automatically following the instructions in 2.2.
+ `title`, `Outliers` _UPDATE_ - title plots and outliers strings.
+ `outlier_z_score_above` and `outlier_z_score_below` _Constant_ - the parameters are set at 3.29, meaning that the outlier is 3.29 times the standard deviation to the mean.\
**Box.test()**
+ `lag` _Constant_ - the statistic will be based on lag autocorrelation coefficients. It is set at 24.
+ `fitdf` _Constant_ - number of degrees of freedom to be subtracted if x is a series of residuals. It is set at 0.
+ `type` _Constant_ - test to be performed. It is set at Lj, which is Ljung-Box test.


**f. Running Models**

+ `file` _Constant_ - the file name plots. It will be changed automatically following the instructions in 2.2.
+ `seasonal` _Constant_ - type of seasonality in hw model. It is set at "multiplicative".
+ `h` _Constant_ - number of periods for forecasting. It is set at 60, meaning 5 years.
+ `damped` _Constant_ - if it TRUE, use a damped trend. 
+ `phi` _Constant_ - it captures the trend that is more in line with historical data along with slight damping to be on the conservative side while forecasting. It is set at 0.9 because research has shown that the assumption of a constant trend in the forecast tends to produce over forecasting. Typically, the damped value is set between 0.8 and 0.98. Note: if the value is equal to 1, the forecast would be the same as a constant trend without a damped parameter. 
+ `alpha`, `beta`, `gamma` _Constant_ - parameters for level, trend, and seasonal components. There are NULL, meaning they will be estimated. To omit them, they needs to be FALSE.

**g. Forecast evaluation**

+ `file` and `colnames` _Constant_ - the file name plots. It will be changed automatically following the instructions in 2.2.
+ `accuracy1_trainingdata_InternalUseOnly` and `accuracy1_testdata_InternalUseOnly` _UPDATE_ - should be years and months you want to split the data - they are strings.
+ `end` _UPDATE_ - logical expression indicating elements or rows to keep. As caseload has a monthly observations, this parameter reflects the number of months you what to use as a traning data. 
+ `start` _UPDATE_ - the start time of the period of interest that you want to test the model. The end period is automatically included based on data setup on 2.6 section. 


&nbsp;
&nbsp;


<a name="processflow"></a>
&nbsp;
   

### l. Model process flow diagrams

The process flow below shows how all the sections feed into the model to produce the forecast outputs.

InternalUseOnly



&nbsp;
&nbsp;

<a name="assumptionsQA"></a>
&nbsp;

### m. Assumptions and QA

Assumptions and QA logs can be found on DOM1. 

**InternalUseOnly**: `S:InternalUseOnly`


&nbsp;
&nbsp;
