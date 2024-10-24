# Helper packages
library(dplyr)    # for data manipulation
library(ggplot2)  # for awesome graphics

# Modeling packages
library(caret)    # for cross-validation, etc.

# Model interpretability packages
library(vip)      # variable importance


# About the data set
# Property sales information as described in De Cock (2011).
#problem type: supervised regression
#response variable: Sale_Price (i.e., $195,000, $215,000)
#features: 80
#observations: 2,930
#objective: use property attributes to predict the sale price of a home
#access: provided by the AmesHousing package (Kuhn 2017a)
#more details: See ?AmesHousing::ames_raw

# access data
ames <- AmesHousing::make_ames()

# splitting the data
split <- initial_split(ames, prop = 0.7, 
                       strata = "Sale_Price")
ames_train  <- training(split)
ames_test   <- testing(split)

# Simple linear regression

# With the Ames housing data, suppose we wanted to model a linear relationship
# between the total above ground living space of a home (Gr_Liv_Area) and sale
# price (Sale_Price). To perform an OLS regression model in R we can use the 
#lm() function:
model1 <- lm(Sale_Price ~ Gr_Liv_Area, data = ames_train)

# The coef() function extracts the estimated coefficients from the model.
# We can also use summary() to get a more detailed report of the model results
summary(model1) 

# The estimated coefficients from our model are ??0=14796.788 and ??1=111.221 
# To interpret, we estimate that the mean selling price increases by 111.221 
# for each additional one square foot of above ground living space. This simple
# description of the relationship between the sale price and square footage 
# using a single number (i.e., the slope) is what makes linear regression such 
# an intuitive and popular modelling tool.

# RMSE
sigma(model1)    # RMSE
# MSE
sigma(model1)^2  # MSE

# In R, we can construct such (one-at-a-time) confidence intervals for each 
# coefficient using confint(). For example, a 95% confidence intervals for the
# coefficients in our example can be computed using
confint(model1, level = 0.95)
# To interpret, we estimate with 95% confidence that the mean selling price increases
# between 106.39 and 116.05 for each additional one square foot of above ground
# living space. We can also conclude that the slope is significantly different 
# from zero (or any other pre-specified value not included in the interval) 
# at the alpha = 0.05 level. This is also supported by the output from summary()

# Multiple linear regression

# In R, multiple linear regression models can be fit by separating all the 
# features of interest with a +
model2 <- lm(Sale_Price ~ Gr_Liv_Area + Year_Built, data = ames_train)
summary(model2)

# Alternatively, we can use update() to update the model formula used in model1.
# The new formula can use a . as shorthand for keep everything on either the 
# left or right hand side of the formula, and a + or - can be used to add or 
# remove terms from the original model, respectively. In the case of adding
# Year_Built to model1, we could've used:
(model2 <- update(model1, . ~ . + Year_Built))

# The LS estimates of the regression coefficients are ??1 = 96.08 and ??2=1008.00
# (the estimated intercept is -2093000 In other words, every one square foot
# increase to above ground square footage is associated with an additional 96.08
# in mean selling price when holding the year the house was built constant. 
# Likewise, for every year newer a home is there is approximately an increase 
# of 1008.00  in selling price when holding the above ground square footage constant

# An interaction occurs when the effect of one predictor on the response 
# depends on the values of other predictors. In linear regression, interactions
# can be captured via products of features. A model with two main effects can 
# also include a two-way interaction. 

# Note that in R, we use the : operator to include an interaction (technically,
# we could use * as well, but x1 * x2 is shorthand for x1 + x2 + x1:x2 so is 
# slightly redundant):
lm(Sale_Price ~ Gr_Liv_Area + Year_Built + Gr_Liv_Area:Year_Built, data = ames_train)

# include all possible main effects
model3 <- lm(Sale_Price ~ ., data = ames_train) 

# print estimated coefficients in a tidy data frame
broom::tidy(model3)  


# We've fit three main effects models to the Ames housing data: a single 
# predictor, two predictors, and all possible predictors. But the question 
# remains, which model is "best"? To answer this question we have to define 
# what we mean by "best". In our case, we'll use the RMSE metric and
# cross-validation to determine the "best" model. We can use the
# caret::train() function to train a linear model (i.e., method = "lm")
# using cross-validation (or a variety of other validation methods). 
# Train model using 10-fold cross-validation
set.seed(123)  # for reproducibility
(cv_model1 <- train(
  form = Sale_Price ~ Gr_Liv_Area, 
  data = ames_train, 
  method = "lm",
  trControl = trainControl(method = "cv", number = 10)
))

# The resulting cross-validated RMSE is $56,403.83 (this is the average RMSE
# across the 10 CV folds). How should we interpret this? When applied to 
# unseen data, the predictions this model makes are, on average, about 
# $56,403.83 off from the actual sale price.

# We can perform cross-validation on the other two models in a similar 
# fashion, which we do in the code chunk below.
# model 2 CV
set.seed(123)
cv_model2 <- train(
  Sale_Price ~ Gr_Liv_Area + Year_Built, 
  data = ames_train, 
  method = "lm",
  trControl = trainControl(method = "cv", number = 10)
)

# model 3 CV
set.seed(123)
cv_model3 <- train(
  Sale_Price ~ ., 
  data = ames_train, 
  method = "lm",
  trControl = trainControl(method = "cv", number = 10)
)

# Extract out of sample performance measures
summary(resamples(list(
  model1 = cv_model1, 
  model2 = cv_model2, 
  model3 = cv_model3
)))

# Extracting the results for each model, we see that by adding more
# information via more predictors, we are able to improve the out-of-sample 
# cross validation performance metrics. Specifically, our cross-validated RMSE 
# reduces from $48968.43 (the model with two predictors) down to $17241.74 
# (for our full model). In this case, the model with all possible main 
# effects performs the "best" (compared with the other two).

# Model concerns

# 1. Linear regression assumes a linear relationship between the predictor and
# the response variable. The left plot illustrates the non-linear relationship
# that exists. However, we can achieve a near-linear relationship by log 
# transforming sale price, although some non-linearity still exists for older
# homes.

p1 <- ggplot(ames_train, aes(Year_Built, Sale_Price)) + 
  geom_point(size = 1, alpha = .4) +
  geom_smooth(se = FALSE) +
  scale_y_continuous("Sale price", labels = scales::dollar) +
  xlab("Year built") +
  ggtitle(paste("Non-transformed variables with a\n",
                "non-linear relationship."))

p2 <- ggplot(ames_train, aes(Year_Built, Sale_Price)) + 
  geom_point(size = 1, alpha = .4) + 
  geom_smooth(method = "lm", se = FALSE) +
  scale_y_log10("Sale price", labels = scales::dollar, 
                breaks = seq(0, 400000, by = 100000)) +
  xlab("Year built") +
  ggtitle(paste("Transforming variables can provide a\n",
                "near-linear relationship."))

gridExtra::grid.arrange(p1, p2, nrow = 1)

# 2. Constant variance among residuals: Linear regression assumes the variance
# among error terms are constant (this assumption is referred to as
# homoscedasticity). If the error variance is not constant, the p-values and 
# confidence intervals for the coefficients will be invalid. For example,
# plots shows the residuals vs. predicted values for model1 and model3. 
# model1 displays a classic violation of constant variance as indicated by 
# the cone-shaped pattern. However, model3 appears to have near-constant 
# variance.

df1 <- data.frame(model1$fitted.values, model1$residuals)

p1 <- ggplot(df1, aes(model1.fitted.values, model1.residuals)) + 
  geom_point(size = 1, alpha = .4) +
  xlab("Predicted values") +
  ylab("Residuals") +
  ggtitle("Model 1", subtitle = "Sale_Price ~ Gr_Liv_Area")

df2 <- data.frame(model3$fitted.values, model3$residuals)

p2 <- ggplot(df2, aes(model3.fitted.values, model3.residuals)) + 
  geom_point(size = 1, alpha = .4)  +
  xlab("Predicted values") +
  ylab("Residuals") +
  ggtitle("Model 3", subtitle = "Sale_Price ~ .")

gridExtra::grid.arrange(p1, p2, nrow = 1)

# 3 No autocorrelation: Linear regression assumes the errors are independent
# and uncorrelated. If in fact, there is correlation among the errors, then 
# the estimated standard errors of the coefficients will be biased leading 
# to prediction intervals being narrower than they should be. For example,
# the left plot displays the residuals (y-axis) vs. the observation ID (x-axis)
# for model1. A clear pattern exists suggesting that information about error1
# provides information about error2

df1 <- mutate(df1, id = row_number())
df2 <- mutate(df2, id = row_number())

p1 <- ggplot(df1, aes(id, model1.residuals)) + 
  geom_point(size = 1, alpha = .4) +
  xlab("Row ID") +
  ylab("Residuals") +
  ggtitle("Model 1", subtitle = "Correlated residuals.")

p2 <- ggplot(df2, aes(id, model3.residuals)) + 
  geom_point(size = 1, alpha = .4) +
  xlab("Row ID") +
  ylab("Residuals") +
  ggtitle("Model 3", subtitle = "Uncorrelated residuals.")

gridExtra::grid.arrange(p1, p2, nrow = 1)

# 4.  More observations than predictors: Although not an issue with the Ames 
# housing data, when the number of features exceeds the number of observations
# (p> n), the OLS estimates are not obtainable.

# 5. No or little multicollinearity: Collinearity refers to the situation in
# which two or more predictor variables are closely related to one another. 
# The presence of collinearity can pose problems in the OLS, since it can be 
# difficult to separate out the individual effects of collinear variables on 
# the response. In fact, collinearity can cause predictor variables to appear
# as statistically insignificant when in fact they are significant. This 
# obviously leads to an inaccurate interpretation of coefficients and makes
# it difficult to identify influential predictors.

# In ames, for example, Garage_Area and Garage_Cars are two variables that have
# a correlation of 0.89 and both variables are strongly related to our response
# variable (Sale_Price). Looking at our full model where both of these
# variables are included, we see that Garage_Cars is found to be statistically
# significant but Garage_Area is not:
# fit with two strongly correlated variables
summary(cv_model3) %>%
  broom::tidy() %>%
  filter(term %in% c("Garage_Area", "Garage_Cars"))

# However, if we refit the full model without Garage_Cars, the coefficient 
# estimate for Garage_Area increases two fold and becomes statistically 
# significant.

# model without Garage_Area
set.seed(123)
mod_wo_Garage_Cars <- train(
  Sale_Price ~ ., 
  data = select(ames_train, -Garage_Cars), 
  method = "lm",
  trControl = trainControl(method = "cv", number = 10)
)

summary(mod_wo_Garage_Cars) %>%
  broom::tidy() %>%
  filter(term == "Garage_Area")
