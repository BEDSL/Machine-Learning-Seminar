# The Diabetes Dataset is a widely used dataset in machine learning for predicting the
# likelihood of diabetes onset based on specific diagnostic features. The dataset was 
# originally collected by the National Institute of Diabetes and Digestive and Kidney Diseases. 
# It consists of health measurements from female patients aged 21 and older, 
# all of Pima Indian heritage.

# Problem Statement:
# The goal of this project is to develop a machine learning model that can predict 
# whether a patient will develop diabetes based on certain health metrics. 
# A successful model can aid healthcare providers in identifying high-risk 
# patients earlier, improving patient outcomes through preventive care.

# Description of the Dataset:
# The dataset contains 768 observations with 8 health-related features, 
# plus a binary target variable (Outcome), indicating the presence (1) 
# or absence (0) of diabetes. The features are as follows:

# 1. Pregnancies: Number of times the patient has been pregnant.
# 2. Glucose: Plasma glucose concentration after a 2-hour oral glucose tolerance test.
# 3. BloodPressure: Diastolic blood pressure (mm Hg).
# 4. SkinThickness: Triceps skinfold thickness (mm).
# 5. Insulin: 2-hour serum insulin (mu U/ml).
# 6. BMI: Body mass index (weight in kg/(height in m)^2).
# 7. DiabetesPedigreeFunction: A score reflecting the likelihood of diabetes based on family history.
# 8. Age: Age of the patient in years.
# 9. Outcome: Target variable, 1 if the patient has diabetes, 0 otherwise.



# Load necessary libraries
library(data.table)
library(ggplot2)
library(caret)
library(xgboost)
library(pROC)
library(dplyr)
library(ROCR)
library(randomForest)
library(e1071)
library(corrplot)

# Load the dataset
data <- fread("diabetes.csv")

# Data overview
head(data)
summary(data)
str(data)

# Check for missing values (NA) after replacing zeros
cat("Number of missing values in each column:\n")
colSums(is.na(data))

# Visualize missing values using a bar plot
missing_values <- colSums(is.na(data))
missing_df <- data.frame(Feature = names(missing_values), MissingValues = missing_values)
ggplot(missing_df, aes(x = Feature, y = MissingValues)) +
  geom_bar(stat = "identity", fill = "orange") +
  labs(title = "Missing Values per Feature", x = "Feature", y = "Number of Missing Values") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# Preprocessing: Replace zero values with NA for columns where 0 is not a valid value
cols_to_replace <- c("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")
data[, (cols_to_replace) := lapply(.SD, function(x) replace(x, x == 0, NA)), .SDcols = cols_to_replace]

# Replace zeros with NA in columns where zero is not valid
replace_zero_cols <- c("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")
data[, (replace_zero_cols) := lapply(.SD, function(x) replace(x, x == 0, NA)), .SDcols = replace_zero_cols]

# Check missing values after replacement
colSums(is.na(data))

# Impute missing values using median imputation
preprocess_params <- preProcess(data, method = 'medianImpute')
data <- predict(preprocess_params, data)

# Distribution of Outcome
ggplot(data, aes(x = Outcome)) + 
  geom_bar(fill = 'skyblue') +
  labs(title = "Distribution of Outcome", x = "Outcome", y = "Count")

# Boxplot for Age vs Outcome
ggplot(data, aes(x = as.factor(Outcome), y = Age)) +
  geom_boxplot(fill = 'coral') +
  labs(title = "Boxplot of Age vs Outcome", x = "Outcome", y = "Age")

# Density plot of Glucose
ggplot(data, aes(x = Glucose, fill = as.factor(Outcome))) + 
  geom_density(alpha = 0.5) + 
  labs(title = "Density Plot of Glucose", x = "Glucose", fill = "Outcome")

# Correlation matrix
corr_matrix <- cor(data[ , -9])  # Removing outcome for correlation calculation
corrplot::corrplot(corr_matrix, method = "circle", type = "lower", tl.col = "black", tl.srt = 45)

# Outlier Detection
boxplot(data$Insulin, main = "Boxplot for Insulin", col = "red")
boxplot(data$Glucose, main = "Boxplot for Glucose", col = "blue")

#Scaling Features
scaler <- preProcess(data[ , -9], method = 'scale')  # Exclude the outcome column
scaled_data <- predict(scaler, data[ , -9])

# Combine scaled data with the outcome
data_scaled <- cbind(scaled_data, Outcome = data$Outcome)

# Train-Test Split
set.seed(123)
trainIndex <- createDataPartition(data_scaled$Outcome, p = 0.8, list = FALSE)
trainData <- data_scaled[trainIndex, ]
testData <- data_scaled[-trainIndex, ]

# Model Training
# Random Forest
rf_model <- randomForest(as.factor(Outcome) ~ ., data = trainData)
rf_pred <- predict(rf_model, testData)

# SVM
svm_model <- svm(as.factor(Outcome) ~ ., data = trainData, probability = TRUE)
svm_pred <- predict(svm_model, testData, probability = TRUE)

# XGBoost Default
xgb_train <- xgb.DMatrix(data = as.matrix(trainData[ , -9]), label = trainData$Outcome)
xgb_test <- xgb.DMatrix(data = as.matrix(testData[ , -9]), label = testData$Outcome)
xgb_default <- xgboost(data = xgb_train, nrounds = 100, objective = "binary:logistic", eval_metric = "auc")

xgb_pred <- predict(xgb_default, xgb_test)
xgb_pred_binary <- ifelse(xgb_pred > 0.5, 1, 0)

# 9. Evaluation Metrics
# Random Forest
rf_conf_matrix <- confusionMatrix(as.factor(rf_pred), as.factor(testData$Outcome))
rf_acc <- rf_conf_matrix$overall['Accuracy']
rf_auc <- roc(testData$Outcome, as.numeric(rf_pred))$auc


# SVM
svm_conf_matrix <- confusionMatrix(as.factor(svm_pred), as.factor(testData$Outcome))
svm_acc <- svm_conf_matrix$overall['Accuracy']
svm_auc <- roc(testData$Outcome, as.numeric(attr(svm_pred, "probabilities")[, 2]))$auc

# XGBoost
xgb_conf_matrix <- confusionMatrix(as.factor(xgb_pred_binary), as.factor(testData$Outcome))
xgb_acc <- xgb_conf_matrix$overall['Accuracy']
xgb_auc <- roc(testData$Outcome, xgb_pred)$auc

# ROC Curve Plotting - must be run together
rf_roc <- prediction(as.numeric(rf_pred), testData$Outcome)
rf_perf <- performance(rf_roc, "tpr", "fpr")
plot(rf_perf, col = "blue", main = "ROC Curve - Random Forest, SVM, XGBoost", lwd = 2)

svm_roc <- prediction(attr(svm_pred, "probabilities")[, 2], testData$Outcome)
svm_perf <- performance(svm_roc, "tpr", "fpr")
plot(svm_perf, col = "green", add = TRUE, lwd = 2)


xgb_roc <- prediction(xgb_pred, testData$Outcome)
xgb_perf <- performance(xgb_roc, "tpr", "fpr")
plot(xgb_perf, col = "red", add = TRUE)

legend("bottomright", legend = c("Random Forest", "SVM", "XGBoost"), col = c("blue", "green", "red"), lty = 1)


# Hyperparameter Tuning for XGBoost - reduced for time efficiency 
xgb_grid <- expand.grid(
  nrounds = c(100, 300),             
  eta = c(0.01, 0.1),               
  max_depth = c(3, 5),               
  min_child_weight = c(0.1, 0.3),    
  subsample = c(0.7, 1.0),           
  colsample_bytree = 1,              
  gamma = 0                         
)

xgb_train_control <- trainControl(method = "cv", number = 5, verboseIter = TRUE)

xgb_tuned_model <- train(
  x = as.matrix(trainData[ , -9]),
  y = as.factor(trainData$Outcome),
  method = "xgbTree",
  trControl = xgb_train_control,
  tuneGrid = xgb_grid,
  metric = "Accuracy"
)

# Best hyperparameters
xgb_tuned_model$bestTune

# Train using the best model
best_xgb <- xgb_tuned_model$finalModel

testData_scaled <- predict(preprocess_params, testData)
# Predictions with the tuned XGBoost
xgb_tuned_pred <- predict(best_xgb, as.matrix(testData[ , -9]))
xgb_tuned_pred_binary <- ifelse(xgb_tuned_pred > 0.5, 1, 0)

# Confusion Matrix and ROC for tuned XGBoost
xgb_tuned_conf_matrix <- confusionMatrix(as.factor(xgb_tuned_pred_binary), as.factor(testData$Outcome))
xgb_tuned_acc <- xgb_tuned_conf_matrix$overall['Accuracy']
xgb_tuned_auc <- roc(testData$Outcome, xgb_tuned_pred)$auc

# Cross-Validation for XGBoost
set.seed(123)
xgb_cv <- xgb.cv(
  params = list(objective = "binary:logistic", eval_metric = "auc", eta = 0.1, max_depth = 10, gamma = 0.1),
  data = xgb_train,
  nrounds = 200,
  nfold = 5,
  verbose = TRUE,
  early_stopping_rounds = 10
)

# Best iteration from cross-validation
best_iter <- xgb_cv$best_iteration


# Train final model using the best iteration from CV
xgb_final <- xgboost(
  data = xgb_train,
  nrounds = best_iter,
  params = list(objective = "binary:logistic", eta = 0.1, max_depth = 10, gamma = 0.1)
)


# Feature Importance for XGBoost
xgb.importance(model = best_xgb) %>%
  xgb.plot.importance()

# Interpretation
# The most important features for predicting diabetes from the XGBoost model are "Glucose" and "BMI".
# These features are crucial in determining the likelihood of a patient having diabetes.
# XGBoost with tuned hyperparameters performed slightly better than the default model and other algorithms.

# Model Comparison Summary
model_comparison <- data.frame(
  Model = c("Random Forest", "SVM", "XGBoost Default", "XGBoost Tuned"),
  AUC = c(rf_auc, svm_auc, xgb_auc, xgb_tuned_auc)
)

print(model_comparison)


# In this dataset, SVM achieved a better AUC compared to XGBoost likely due to the data having simpler 
# or more linear relationships, which SVM handles efficiently using its margin-based classification approach.
# Can you improve the model through hyperparameter tuning? 