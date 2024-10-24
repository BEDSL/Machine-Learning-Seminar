
# Decision Tree Classification ---------------------------------------------

# Imagine you’re a bank trying to decide if a customer is likely to 
# subscribe to a new product. 
# Instead of sifting through all the data yourself, you can use a decision tree 
# to make these decisions systematically.
# A decision tree is like a flowchart: it starts with a simple yes-or-no question 
# (or rule) based on one customer attribute, and splits into branches. 
# Each branch leads to more questions until it arrives at a final decision—
# 'yes' or 'no.' 
# The tree learns these rules from data, picking the most informative questions 
# at each step to create a clear, step-by-step decision-making process. 
# Simple, visual, and powerful—let's see how it works in action!

# 1. Read the data -----------------------------------------------------------

# In this lab, we will use the same dataset from the bank's direct marketing 
# campaign to build a decision tree model. 
# By applying this alternative classification method, we can compare its 
# performance to that of the logistic regression model we previously built. 
# This comparison will help us understand the strengths and weaknesses of each 
# approach in predicting customer subscriptions.

## Set the working directory:
setwd("C:/Users/Admin/OneDrive - Academia de Studii Economice din Bucuresti/Documents/Materiale predare/Master Business Analytics/decision trees & random forest/SEMINAR")
#### Please change the line above to your own file path!

# Then install and load the libraries:
#install.packages("readr")
library(readr)
#install.packages("tidymodels")
library(tidymodels)

# Read the dataset and convert the target variable to a factor:
bank_df <- read_csv2("bank-full.csv")
bank_df$y = as.factor(bank_df$y)


# 2. Splitting the Data ---------------------------------------------------

# Split data into train and test
set.seed(421)
split <- initial_split(bank_df, prop = 0.8, strata = y)
train <- split %>% 
  training()
test <- split %>% 
  testing()


# 3. Fitting the Model ----------------------------------------------------

# In R, there are two common ways to fit a decision tree model:

#  Using tidymodels (with rpart as the engine): The tidymodels framework 
# provides a consistent and flexible interface for building and evaluating models.
# By using tidymodels with the rpart engine, you can easily integrate 
# preprocessing, cross-validation, and evaluation steps into a unified workflow.
# This approach is great for managing complex modeling processes 
# and comparing different models.

# Using the rpart Package Directly: The rpart package offers a straightforward 
# way to fit a decision tree using the rpart() function. 
# It requires fewer steps and is ideal for quick analyses or when you only 
# need a basic decision tree model. 
# However, it doesn't include the advanced preprocessing, model tuning or 
# evaluation tools provided by tidymodels, 
# so you may need to handle those steps manually.

# Both approaches allow you to fit a decision tree, but tidymodels provides
# more structure and flexibility, 
# while rpart offers simplicity and direct control.

# We'll be using a decision tree model with the tidymodels workflow.

# Define the decision tree model and specify the engine as "rpart".
?decision_tree()
model <- decision_tree() %>%
  set_engine("rpart") %>%
  set_mode("classification") %>%
  fit(y ~ ., data = train)

# Install and load the rpart.plot package for visualizing the decision tree:
# install.packages("rpart.plot")
library(rpart.plot)

# Visualize the pruned decision tree: 
rpart.plot(model$fit, main = "Decision Tree for Subscription Prediction")

# This decision tree predicts whether a bank customer will subscribe 
# to a term deposit. The tree begins by splitting the data based on the duration 
# of the last contact, with the first split at a duration of 504. If the duration
# is less than 504, the model further examines the poutcome 
# (outcome of the previous marketing campaign) and duration values.
# 
# Each node in the tree displays the predicted outcome ("yes" or "no"), 
# the probability of that outcome, and the percentage of data observations in 
# that node. For example, the root node shows that 12% of customers are predicted
# as "no" subscribers. As the tree progresses to leaf nodes, decisions are made
# based on duration and poutcome, resulting in final predictions with varying
# probabilities. This structure helps identify the most important factors 
# influencing a customer's decision to subscribe, with duration being the 
# most significant factor in this case.

# Create a variable importance plot:
rules <- rpart.rules(model$fit)
print(rules)

#install.packages("vip")
library(vip)

var_importance <- vip::vip(model, num_features = 10)
print(var_importance)

#This plot shows the feature importance for the decision tree model used 
# to predict whether a customer will subscribe to a term deposit. 
# The importance of each feature indicates how much it contributes 
# to the decision-making process in the tree.
# 
# Interpretation:
#   duration is the most important feature, with the highest importance score. 
# This means that the duration of the last contact has the greatest impact on the 
# model's predictions. In the context of a marketing campaign, this suggests that
# longer interactions with customers are strongly correlated with their
# likelihood of subscribing.
#   poutcome (outcome of the previous marketing campaign) is the second most 
# important feature. This implies that the customer's response in the previous
# campaign significantly influences the likelihood of their current subscription.
#   other features, like contact, campaign, pdays, default, previous, and balance,
#   have much lower importance scores. This suggests that they contribute less
#   to the decision-making process in this specific model.
#   
# Overall, this plot indicates that the duration and previous campaign outcome are
# the key drivers in predicting customer subscriptions, while other features play
# a minimal role in this decision tree model.


# 4. Making predictions ---------------------------------------------------

# Class Predictions for the decision tree
pred_class <- predict(model, new_data = test, type = "class")

# Class Probabilities for the decision tree
pred_proba <- predict(model, new_data = test, type = "prob")

results <- test %>%
  select(y) %>%
  bind_cols(pred_class, pred_proba)
print(results)

# 5. Evaluating the Model -------------------------------------------------

# Confusion Matrix:
conf_mat(results, truth = y, estimate = .pred_class)

# Accuracy:
accuracy(results, truth = y, estimate = .pred_class)

# Precision:
precision(results, truth = y, estimate = .pred_class)

# Recall:
recall(results, truth = y, estimate = .pred_class)

# ROC Curve and AUC:
library(ROCR)

# Convert 'yes' probabilities to a prediction object
predictions <- prediction(pred_proba$.pred_yes, test$y)

# Create the ROC performance object
roc_perf <- performance(predictions, "tpr", "fpr")

# Plot the ROC curve
plot(roc_perf, colorize = TRUE, main = "ROC Curve", col = "blue")
abline(a = 0, b = 1, lty = 2, col = "gray")  # Add a diagonal line for reference

# Calculate the AUC
auc <- performance(predictions, "auc")
auc_value <- auc@y.values[[1]]
print(paste("AUC:", auc_value))

# When comparing the logistic regression and decision tree models, 
# we see that both have similar overall performance, 
# but there are some notable differences.
# 
# First, the accuracy of logistic regression is 90.3%, which is slightly better
# than the decision tree's accuracy of 90.0%. This small difference suggests that
# logistic regression is just a bit more reliable 
# in making correct predictions overall.
# 
# Next, looking at precision, which measures how often the model's positive
# predictions are correct, the decision tree has a slight advantage, 
# scoring 92.1% compared to logistic regression's 91.9%. This indicates that 
# the decision tree is marginally better at avoiding false positives.
# 
# However, when we consider recall, which reflects how well the model identifies 
# all actual positive cases, logistic regression outperforms with a recall of 
# 97.6%, compared to the decision tree's 97.0%. This means logistic regression 
# is slightly more effective at capturing the true positive cases in the dataset.
# 
# The most significant difference between the models is seen in the
# AUC (Area Under the ROC Curve). Logistic regression has an AUC of 0.909, 
# indicating a strong ability to distinguish between the positive and negative 
# classes across different thresholds. In contrast, the decision tree has an AUC
# of only 0.7509, suggesting that it is less effective in separating the two classes.
# 
# Overall Conclusion:
# While the decision tree has a slightly higher precision, 
# logistic regression outperforms it in terms of accuracy, recall, 
# and most notably, AUC. The higher AUC value for logistic regression indicates 
# that it is better at distinguishing between positive and negative classes across
# various thresholds. Therefore, logistic regression is the better-performing model
# for this task, offering a more balanced and robust performance 
# across most evaluation metrics.


# Random Forest -----------------------------------------------------------

# Now that we've explored logistic regression and decision trees, let's introduce 
# a more advanced method: Random Forest. 
# A random forest is an ensemble learning method
# that builds multiple decision trees using random subsets of data and features. 
# Instead of relying on a single tree, a random forest combines the predictions of
# many trees to make a final decision, typically by majority vote. This process helps
# to improve accuracy and reduces the risk of overfitting that can occur with 
# a single decision tree.
# 
# In this section, we will apply a random forest model to our dataset to see if it 
# can enhance our prediction accuracy compared to the previous models. 
# Random forests are known for their robustness and ability to handle complex data
# relationships, making them an excellent choice for improving predictive performance.
# Let's dive in and explore how this method can potentially give us more reliable 
# results for predicting customer subscriptions.

# 1. Fitting the Model ----------------------------------------------------

# In R, there are two common ways to fit a random forest model:
#
# Using the randomForest Package: This method uses the randomForest() 
# function for quick and straightforward modeling. It allows you to set parameters
# like the number of trees (ntree) and variables (mtry). It's simple but has limited
# flexibility for preprocessing and evaluation.
# Using tidymodels with "ranger" or "randomForest" Engine: The tidymodels 
# framework provides a flexible workflow with rand_forest(), allowing seamless 
# integration with data preprocessing, cross-validation, and tuning. It requires a
# slightly more complex setup but offers more control.
#
# Use randomForest for simplicity and tidymodels for more advanced modeling needs.

# We'll be using a random forest model with the tidymodels workflow.
# Install ranger if you haven't already:
# install.packages("ranger")
library(ranger)

# Define the random forest model:
model <- rand_forest(trees = 1000, min_n = 5) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification") %>%
  fit(y ~ ., data = train)

# Create the variable importance plot for the random forest:
library(vip)

var_importance <- vip::vip(model, num_features = 10)
print(var_importance)


#This plot shows the feature importance of the random forest model used 
# to predict customer subscriptions.
# 
#   duration is the most influential feature by a large margin, indicating that the
# length of the last contact significantly affects the likelihood of a customer subscribing.
#   balance and age are the next most important features, suggesting that a customer's 
# financial status and age are key factors in their decision-making process.
#   month and day of the contact have moderate importance, implying that when the
# customer was contacted also plays a role.
#   poutcome (previous campaign outcome) and pdays (days since last contact) 
# contribute some information, indicating that past interactions impact the subscription decision.
#   job, campaign, and housing have relatively low importance, suggesting they have 
# a minimal effect on the model's predictions.
# 
# Overall, the plot highlights that contact duration and customer financial attributes are the primary drivers in predicting subscriptions.


# 2. Making predictions ---------------------------------------------------

# Class Predictions for the random forest
pred_class <- predict(model, new_data = test, type = "class")

# Class Probabilities for the random forest
pred_proba <- predict(model, new_data = test, type = "prob")

results <- test %>%
  select(y) %>%
  bind_cols(pred_class, pred_proba)
print(results)

# 3. Evaluating the Model -------------------------------------------------

# Confusion Matrix:
# Step 1: Ensure `y` in results has the same levels as `.pred_class`
results$y <- factor(results$y, levels = levels(results$.pred_class))

# Step 2: Confusion Matrix
confusion_matrix <- conf_mat(results, truth = y, estimate = .pred_class)
print(confusion_matrix)


# Accuracy:
accuracy(results, truth = y, estimate = .pred_class)

# Precision:
precision(results, truth = y, estimate = .pred_class)

# Recall:
recall(results, truth = y, estimate = .pred_class)

# Remove rows if `y` is "Other"
# results_filtered <- results %>% filter(y %in% c("no", "yes")) #ROCR currently supports only evaluation of binary classification tasks

# Step 1: Convert 'yes' probabilities to a prediction object for the ROC curve
library(ROCR)
predictions <- prediction(results$.pred_yes, results$y)

# Step 2: Calculate the ROC curve and AUC
roc_perf <- performance(predictions, "tpr", "fpr")  # True Positive Rate vs False Positive Rate
auc_perf <- performance(predictions, "auc")         # Area Under Curve

# Step 3: Plot the ROC curve
plot(roc_perf, main = "ROC Curve")
abline(a = 0, b = 1, lty = 2, col = "gray")  # Add a diagonal line for reference

# Display AUC
auc_value <- auc_perf@y.values[[1]]
cat("AUC:", auc_value, "\n")

# Comparison Between Decision Trees and Random Forest
# 1. Basic Concept:
#   
#   Decision Trees: A decision tree is a single, interpretable model that splits data based on feature values to make predictions. It works by dividing the dataset into smaller subsets based on specific conditions, forming a tree-like structure. While simple and easy to visualize, decision trees are prone to overfitting, especially with complex datasets.
# Random Forest: Random forest is an ensemble learning method that builds multiple decision trees using random subsets of data and features. The final prediction is made by aggregating the predictions of all trees (e.g., majority voting for classification). This ensemble approach helps improve accuracy and reduces the risk of overfitting.
# 2. Performance:
#   
#   Decision Trees: They tend to perform well on simpler datasets and are easily interpretable. However, their performance can degrade if the dataset has complex relationships or noise, often leading to overfitting.
# Random Forest: By averaging the results of many trees, random forests generally provide higher accuracy and better generalization than a single decision tree. In our example, the random forest model achieved a higher AUC (0.93) compared to the decision tree's AUC (0.75), indicating its superior predictive performance.
# 3. Robustness and Stability:
# 
# Decision Trees: A single decision tree is sensitive to small changes in the data. A minor variation in the dataset can lead to a completely different tree structure, affecting model stability.
# Random Forest: The use of multiple trees and random sampling makes random forests more robust and less sensitive to fluctuations in the training data, resulting in more stable predictions.
# 4. Interpretability:
# 
# Decision Trees: They are highly interpretable because they provide a clear visual representation of decision paths. You can easily understand how predictions are made by tracing the branches.
# Random Forest: Although random forests improve accuracy, they sacrifice some interpretability. Since predictions are based on the consensus of numerous trees, it's harder to extract a straightforward decision rule.
# 5. Feature Importance:
#   
#   Decision Trees: Provide a basic measure of feature importance based on how often a feature is used for splitting. However, this measure can be biased towards features with more levels.
# Random Forest: Offers a more robust measure of feature importance by averaging across many trees. This approach helps to provide a more reliable understanding of which features contribute most to the predictions.


# Bibliography ------------------------------------------------------------
# 1. https://www.datacamp.com/tutorial/decision-trees-R
# 2. https://www.kaggle.com/code/hamelg/intro-to-r-part-30-random-forests
# 3. https://rviews.rstudio.com/2019/06/19/a-gentle-intro-to-tidymodels/