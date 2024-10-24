
# Logistic Regression -----------------------------------------------------

# Imagine you're a bank deciding which customers are likely to subscribe to a new product.'
# You have tons of customer data—age, job, marital status, and more. 
# But how do you turn this sea of information into a simple 'yes' or 'no' decision?
# That's where logistic regression comes in! Unlike linear regression,
# which predicts continuous numbers, logistic regression helps us predict the probability
# of an outcome falling into a specific category, like 'yes' or 'no.' 
# It takes all those attributes and learns how each one nudges a customer towards a 'yes' or 'no.'  
# Think of it as the model creating a scoring system, where higher scores mean a higher likelihood of a 'yes.'
# Simple, yet powerful—let's dive into how it works!


# 1. Read the data -----------------------------------------------------------
# We'll be using a dataset from a direct marketing campaign conducted by a Portuguese bank, 
# which utilized phone calls to reach potential customers. 
# The campaign's goal was to sell subscriptions to a bank term deposit, represented by 
# the target variable 'y' (indicating whether a customer subscribed or not). 
# Our objective with the logistic regression model is to predict the likelihood of a customer 
# subscribing based on various predictor variables, such as demographic information 
# and other customer attributes.

##First, set the working directory:
setwd("C:/Users/Admin/OneDrive - Academia de Studii Economice din Bucuresti/Documents/Materiale predare/Master Business Analytics/regresie logistica/SEMINAR")

# Then install and load the libraries:
#install.packages("readr")
library(readr)
#install.packages("tidymodels")
library(tidymodels)

# Read the dataset and convert the target variable to a factor:
bank_df <- read_csv2("bank-full.csv")
bank_df$y = as.factor(bank_df$y)


# 2. Variable Descriptions ------------------------------------------------


#   age: Age of the customer (numeric).
# 
#   job: Type of job (categorical) – Options: "admin.", "blue-collar", 
# "entrepreneur", "housemaid", "management", "retired", "self-employed",
# "services", "student", "technician", "unemployed", "unknown".
# 
#   marital: Marital status (categorical): "divorced", "married", 
# "single", "unknown". Note: "divorced" includes widowed clients.
# 
#   education: Highest degree of the customer (categorical): "basic.4y",
# "basic.6y", "basic.9y", "high.school", "illiterate", "professional.course",
# "university.degree", "unknown".
# 
#   default: Has credit in default? (categorical): "no", "yes", "unknown".
# 
#   housing: Has a housing loan? (categorical): "no", "yes", "unknown".
# 
#   loan: Has a personal loan? (categorical): "no", "yes", "unknown".
# 
#   contact: Contact communication type (categorical): "cellular", "telephone".
# 
#   month: Last contact month of the year (categorical): "jan", "feb", "mar", ...,
# "nov", "dec".
# 
#   day_of_week: Last contact day of the week (categorical): "mon", "tue", "wed",
# "thu", "fri".
# 
#   campaign: Number of times the client was contacted during this campaign 
# (numeric, includes the last contact).
# 
#   pdays: Number of days since the client was last contacted from a previous 
# campaign (numeric). A value of 999 indicates the client
# was not previously contacted.
# 
#   previous: Number of contacts performed before this campaign
# for this client (numeric).
# 
#   poutcome: Outcome of the previous marketing campaign (categorical):
# "failure", "nonexistent", "success".
# 
#   emp.var.rate: Employment variation rate – quarterly indicator (numeric).
# 
#   cons.price.idx: Consumer price index – monthly indicator (numeric).
# 
#   cons.conf.idx: Consumer confidence index – monthly indicator (numeric).
# 
#   euribor3m: Euribor 3-month rate – daily indicator (numeric).
# 
#   nr.employed: Number of employees – quarterly indicator (numeric).
# 
#   y: Has the client subscribed to a term deposit? (binary): "yes", "no".

# 3. Visualize the Data ------------------------------------------------------

# View the dataset:
View(bank_df)

#install.packages("ggplot2")
library(ggplot2)

# Visualize the Target Variable: Subscription Status:

# Before diving into the modeling process, it's crucial to understand 
# the distribution of our target variable, y, which indicates whether a customer 
# subscribed to the term deposit. Here, we use a pie chart to illustrate 
# the proportion of customers who subscribed versus those who did not. 
# This simple visualization provides a quick overview of the dataset's balance 
# and can help guide our model-building process. 
# For example, if the chart shows a significant imbalance (more 'no' than 'yes'), 
# it might indicate that we'll need to account for this in our model evaluation 
# to avoid biased predictions.

# In our example, are the two classes balanced?

# Let's see:

# Firstly, create a data frame for the pie chart
pie_data <- bank_df %>%
  group_by(y) %>%
  summarise(count = n()) %>%
  mutate(prop = count / sum(count) * 100, 
         ypos = cumsum(prop) - 0.5 * prop)

# Then generate the pie chart
ggplot(pie_data, aes(x = "", y = prop, fill = y)) +
  geom_bar(stat = "identity", width = 1, color = "white") +
  coord_polar("y", start = 0) +
  theme_void() +
  geom_text(aes(y = ypos, label = paste0(round(prop, 1), "%")), color = "white") +
  labs(title = "Proportion of Subscription Status")


# Exporatory analysis about the other features

# Summary of the dataset
glimpse(bank_df)
summary(bank_df)


# Plot job occupation against the target variable:
ggplot(bank_df, aes(job, fill = y)) +
  geom_bar() +
  coord_flip()

# Visualize the distribution of age across the target variable
ggplot(bank_df, aes(x = age, fill = y)) +
  geom_histogram(bins = 30, alpha = 0.7) +
  labs(title = "Distribution of Age by Subscription Status")

# Visualize a correlation heatmap for numeric variables
#install.packages("GGally")
library(GGally)
ggcorr(bank_df %>% select_if(is.numeric), label = TRUE)


# 4. Splitting the Data ---------------------------------------------------

# Split data into train and test
set.seed(421)
split <- initial_split(bank_df, prop = 0.8, strata = y)
train <- split %>% 
  training()
test <- split %>% 
  testing()

# 5. Fitting the Model ----------------------------------------

# In R, there are two commonly used workflows for modeling logistic regression: 
#   the base R approach and 
#   the tidymodels framework.

# The base R workflow is straightforward, using functions like glm() to 
# fit the model and summary() to provide a model overview. 
# It is simple and effective for quick analyses. Please, check:
?glm()

# The tidymodels workflow, however, offers a more flexible and consistent 
# interface for modeling. '
# It allows for easier management of multiple models and includes powerful tools 
# for data preprocessing, tuning, and evaluation.
?logistic_reg()

# In this lab, we'll be using the tidymodels workflow to build our logistic 
# regression model, as it provides a more structured approach that is particularly
# useful for handling complex datasets and model comparisons.



## Train a logistic regression model:

# To create the model, declare a logistic_reg() model. 
# This needs mixture and penalty arguments which control 
# the amount of regularization. 
# A mixture value of 1 denotes a lasso model and 0 denotes ridge regression. 
# Values in between are also allowed. The penalty argument denotes the strength 
# of the regularization.

# Call the fit() method to train the model on the training data created in 
# the previous step. 
# This takes a formula for its first argument.
# On the left-hand side of the formula, you use the target variable 
# (in this case y). 
# On the right-hand side, you can include any features you like. 
# A period means "use all the variables that weren't written on the left-hand 
# side of the formula. 


model <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification") %>%
  fit(y ~ ., data = train)

# Model summary:
tidy(model)

# Coefficients:
coeff <- tidy(model) %>% 
  select(c(term, estimate, p.value)) %>% 
  arrange(desc(abs(estimate))) %>% 
  filter(abs(estimate) > 0.5) %>% 
  mutate(odds_ratio = exp(estimate))

coeff

# The coefficient estimate for poutcomesuccess is 2.34, which, when
# exponentiated, gives an odds ratio of 10.3. 
# This means that if the outcome of the previous marketing campaign was a 
# "success," the odds of the customer subscribing to the current term deposit are 
# 10.3 times higher compared to when the previous outcome was not a success.
#
# The p-value is extremely small (< 0.001), indicating that this effect is 
# statistically significant. In other words, there is strong evidence that a 
# previous success has a substantial positive impact on the likelihood 
# of the customer subscribing.


# The coefficient estimate for contactunknown is -1.62, which gives an odds 
# ratio of 0.197. This means that if the contact type is "unknown," the odds of
# the customer subscribing to the term deposit are reduced to approximately 19.7%
# of the odds when the contact type is not "unknown." In other words, "unknown" 
# contact type significantly lowers the likelihood of a customer subscribing.
#
#The p-value (0.0039) is less than 0.05, indicating that this effect is 
# statistically significant. It suggests that "unknown" contact types have a 
# negative and meaningful impact on the probability of subscription.

# Plot the feature importance:
ggplot(coeff, aes(x = term, y = estimate, fill = term)) + geom_col() + coord_flip()

# 6. Making predictions ---------------------------------------------------

# Use the predict() function to make predictions on the testing data. 
# You have two options for the type of predictions:
#   
#   type = "class": This option returns the most likely target value for 
# each observation. In this case, it will output either "yes" or "no," indicating 
# whether the model predicts the client will subscribe to a term deposit.
# 
#   type = "prob": This option provides the probability of each possible target 
# value for each observation. It returns both the probability of "yes" and 
# the probability of "no," which sum to one for each observation. 
# This allows you to see not only the predicted outcome but also the model’s 
# confidence in that prediction.

# Class Predictions
pred_class <- predict(model,
                      new_data = test,
                      type = "class")

# Class Probabilities
pred_proba <- predict(model,
                      new_data = test,
                      type = "prob")

results <- test %>%
  select(y) %>%
  bind_cols(pred_class, pred_proba)
print(results)


# 7. Evaluating the Model -------------------------------------------------

##Confusion matrix:

# A confusion matrix is a table used to evaluate the performance of a 
# classification model by comparing its predictions with the actual values. 
# It provides a detailed breakdown of the model's performance by displaying 
# the counts of true positives, true negatives, false positives
# and false negatives.
 
# The confusion matrix is the foundation for many other evaluation metrics. 
# From this matrix, we can derive key metrics such as accuracy, precision, recall,
# F1-score, and more. Each of these metrics offers different insights into 
# the model's performance, helping us understand not just how often the model 
# is correct, but also how it handles positive and negative cases. 
# By examining the confusion matrix, we gain a comprehensive view of the model's 
# strengths and weaknesses.

conf_mat(results, truth = y,
         estimate = .pred_class)

##Accuracy:

# Accuracy is a metric used to evaluate the performance of a classification model. 
# It measures the proportion of correct predictions made by the model 
# out of all predictions. 
# In other words, accuracy indicates how often the model correctly classifies 
# an observation as either "yes" or "no." 
# While accuracy is a useful starting point, it's important to remember that it
# can be misleading in cases of imbalanced datasets, where one class is much more
# common than the other. 
# In such cases, additional metrics like precision and recall provide a more 
# complete picture of model performance.

accuracy(results, truth = y, estimate = .pred_class)

##Precision:

# Precision is a metric used to evaluate the performance of a classification 
# model, specifically focusing on the accuracy of positive predictions. 
# It is defined as the proportion of true positive predictions 
# (correctly predicted "yes" cases) out of all the observations that the model
# predicted as positive (both true positives and false positives).
# 
# In other words, precision answers the question: 
# "Out of all the times the model predicted 'yes,' how many were actually 'yes'?"
# High precision means that when the model predicts a positive class, 
# it is often correct. Precision is particularly important when the cost of a 
# false positive is high, 
# such as in medical diagnosis or fraud detection scenarios.

precision(results, truth = y,
          estimate = .pred_class)

##Recall:

# Recall is a metric used to evaluate a classification model's ability to 
# correctly identify all relevant positive cases. It is defined as the proportion
# of true positive predictions (correctly predicted "yes" cases) out of
# the total actual positive cases. 
# 
# In other words, recall answers the question: 
# "Out of all the actual 'yes' cases, how many 
# did the model successfully identify?" 
# High recall means that the model captures most of the positive cases, 
# which is particularly important in scenarios like disease detection, 
# where missing a positive case can have serious consequences.

recall(results, truth = y,
       estimate = .pred_class)

##ROC curve and AUC:

# The ROC (Receiver Operating Characteristic) curve is a graphical representation
# that illustrates the performance of a classification model 
# at various threshold settings. It plots the True Positive Rate (Recall) against 
# the False Positive Rate, showing the trade-off between 
# sensitivity and specificity.
# 
# The AUC (Area Under the Curve) quantifies the overall ability of the model 
# to distinguish between classes. An AUC of 0.5 indicates no discrimination 
# (equivalent to random guessing), while an AUC of 1.0 represents 
# perfect classification. The higher the AUC, the better the model is at 
# correctly classifying positive and negative cases across all thresholds.

# install.packages("ROCR")
library(ROCR)

# For the ROC curve, we need the probability for the positive class ("yes")
# Convert 'yes' probabilities to a prediction object
predictions <- prediction(pred_proba$.pred_yes, test$y)

# Create the ROC performance object:
roc_perf <- performance(predictions, "tpr", "fpr")

# Plot the ROC curve:
plot(roc_perf, colorize = TRUE, main = "ROC Curve", col = "blue")
# Add a diagonal line for reference
abline(a = 0, b = 1, lty = 2, col = "gray")  

# Calculate the AUC:
auc <- performance(predictions, "auc")
auc_value <- auc@y.values[[1]]
print(paste("AUC:", auc_value))


# Bibliography ------------------------------------------------------------
#1. https://www.datacamp.com/tutorial/logistic-regression-R
#2. Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. 
#UCI Machine Learning Repository. https://doi.org/10.24432/C5K306.
