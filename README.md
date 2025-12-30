# samba-rice-price-forecasting
Time series analysis and forecasting of monthly average retail prices of Samba rice in Sri Lanka using ARIMA and Auto-ARIMA models. Includes data preprocessing, stationarity testing, model evaluation, and future price forecasting.



# Load libraries
library(fpp2)
library(tseries)
library(forecast)
library(ggplot2)
library(FinTS)
library(gridExtra)

# 1. Load and Inspect the Data
rice_data <- read.csv("Cleaned_Rice_Data.csv", header = TRUE) # Replace with your file path
rice_data$date <- as.Date(rice_data$date, format = "%m/%d/%Y")
rice_data$price <- as.numeric(rice_data$price)

# Check for missing or invalid data
rice_data <- rice_data[!is.na(rice_data$date) & !is.na(rice_data$price), ]
rice_data$price <- as.numeric(rice_data$price)
missing_data <- sum(is.na(rice_data$price) & is.na(rice_data$date))
if (missing_data == 0) print("No Missing Values") else print("There is Missing Values")


# Check data summary and structure
print(summary(rice_data))
print(str(rice_data))

# Ensure there are valid observations
if (nrow(rice_data) == 0) {
  stop("No valid observations in the dataset.")
}

# Create a time series object (assuming monthly data starting July 2006)
ts_data <- ts(rice_data$price, start = c(2006, 7), frequency = 12)

# Plot the original time series
autoplot(ts_data, main = "Original Time Series plot '2006-07-15  - 2022-06-15' ", ylab = "Price (LKR)")

# 2. Split Data into Training and Test Sets
n <- length(ts_data)
train_size <- floor(0.8 * n)
train_ts <- window(ts_data, end = c(2006 + (train_size - 1) %/% 12, (train_size - 1) %% 12 + 1))
test_ts <- window(ts_data, start = c(2006 + train_size %/% 12, train_size %% 12 + 1))

# 3. Decompose and Analyze Seasonality
acf(train_ts,main="ACF plot ")
pacf(train_ts,main="PACF plot ")
decomposed_ts <- decompose(train_ts)
plot(decomposed_ts)

# Check for seasonality using a statistical test
seasonality_test <- isSeasonal(train_ts)
print(paste("Is the series seasonal? ", seasonality_test))

# 4. Stationarity Check
adf_test <- adf.test(train_ts)
print(adf_test)
if (adf_test$p.value <= 0.05) {
  print("The series is stationary.")
} else {
  print("The series is not stationary. Differencing is required.")
}

# 5. Differencing to Ensure Stationarity
ts_diff1 <- diff(train_ts)
autoplot(ts_diff1, main = "First Differenced Series", ylab = "Differenced Price")

adf_test_diff1 <- adf.test(ts_diff1)
print(adf_test_diff1)
if (adf_test_diff1$p.value <= 0.05) {
  print("The differenced series is stationary.")
} else {
  print("Further differencing might be required.")
}

# 6. ACF and PACF Analysis
acf_plot <- acf(ts_diff1,main='ACF plot of  1st differencing')
pacf_plot <- pacf(ts_diff1,main='PACF plot of  1st differencing')


# 7. Fit ARIMA Models
# Automatic ARIMA
auto_arima_model <- auto.arima(train_ts, seasonal = F, trace = TRUE)

# Manual ARIMA
manual_arima_model <- Arima(train_ts, order = c(2, 1, 1))

# Print model summaries
print(summary(auto_arima_model))
print(summary(manual_arima_model))

F# 8. Assumption Checks
# Residual analysis for Auto ARIMA
checkresiduals(auto_arima_model)

shapiro_test_auto <- shapiro.test(residuals(auto_arima_model))
print(shapiro_test_auto)
plot(residuals(auto_arima_model), main = "Residuals of Auto ARIMA Model")
qqnorm(residuals(auto_arima_model))
qqline(residuals(auto_arima_model), col = 'red')


# Ljung-Box test for autocorrelation of residuals
ljung_box_auto <- Box.test(residuals(auto_arima_model), lag = 20, type = 'Ljung-Box')
print(ljung_box_auto)

acf(residuals(auto_arima_model), main = "ACF of Residuals")
pacf(residuals(auto_arima_model), main = "PACF of Residuals")



# Residual analysis for Manual ARIMA
checkresiduals(manual_arima_model)
plot(residuals(manual_arima_model), main = "Residuals of Manual ARIMA Model")
qqnorm(residuals(manual_arima_model))
qqline(residuals(manual_arima_model), col = 'red')
acf(residuals(manual_arima_model), main = "ACF of Residuals")
pacf(residuals(manual_arima_model), main = "PACF of Residuals")

# Plot residuals

acf(residuals(auto_arima_model), main = "ACF of Residuals (Auto ARIMA)")


# 9. Model Forecasting
forecast_auto <- forecast(auto_arima_model, h = 12)
forecast_manual <- forecast(manual_arima_model, h = 12)

autoplot(forecast_auto) +
  ggtitle("Forecast from Auto ARIMA") +
  xlab("Time") +
  ylab("Price (LKR)")

autoplot(forecast_manual) +
  ggtitle("Forecast from Manual ARIMA") +
  xlab("Time") +
  ylab("Price (LKR)")

# 10. Model Evaluation
rmse_auto <- sqrt(mean((forecast_auto$mean - test_ts)^2))
mape_auto <- mean(abs((forecast_auto$mean - test_ts) / test_ts)) * 100
print(paste("Auto ARIMA - RMSE:", round(rmse_auto, 2), "MAPE:", round(mape_auto, 2)))

rmse_manual <- sqrt(mean((forecast_manual$mean - test_ts)^2))
mape_manual <- mean(abs((forecast_manual$mean - test_ts) / test_ts)) * 100
print(paste("Manual ARIMA - RMSE:", round(rmse_manual, 2), "MAPE:", round(mape_manual, 2)))


# 9. Combine forecast with original data for comparison
autoplot(ts_data, series = "Original Data", color = "blue") +
  autolayer(fitted(manual_arima_model), series = "Fitted (Manual ARIMA)", color = "red") +
  autolayer(forecast(manual_arima_model, h = 10)$mean, series = "Forecast (Manual ARIMA)", color = "red") +
  ggtitle("Comparison of Original Data and Manual ARIMA's Forecast") +
  xlab("Time") +
  ylab("Price (LKR)") +
  scale_color_manual(values = c("blue", "red", "red")) +
  theme_minimal() +
  theme(legend.title = element_blank())


autoplot(ts_data, series = "Original Data", color = "blue") +
  autolayer(fitted(auto_arima_model), series = "Fitted (Manual ARIMA)", color = "red") +
  autolayer(forecast(auto_arima_model, h = 10)$mean, series = "Forecast (Manual ARIMA)", color = "red") +
  ggtitle("Comparison of Original Data and Auto ARIMA's Forecast") +
  xlab("Time") +
  ylab("Price (LKR)") +
  scale_color_manual(values = c("blue", "red", "red")) +
  theme_minimal() +
  theme(legend.title = element_blank())

