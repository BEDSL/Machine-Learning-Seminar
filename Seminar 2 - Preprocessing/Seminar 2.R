
# Helper packages
library(dplyr)    # for data manipulation
library(ggplot2)  # for awesome graphics
library(visdat)   # for additional visualizations

# Feature engineering packages
library(caret)    # for various ML tasks
library(recipes)  # for feature engineering tasks
library(forecast) # for Box Cox transformation

# Modeling process packages
library(rsample)   # for resampling procedures
library(caret)     # for resampling and model training
library(h2o)       # for resampling and model training

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

# initial dimension
dim(ames)

# response variable
head(ames$Sale_Price)


# Simple random sampling

# The simplest way to split the data into training and test sets is to take a
# simple random sample. This does not control for any data attributes, such as
# the distribution of your response variable

# Throughout this seminar we'll often use the seed 123 for reproducibility but
# the number itself has no special meaning.

# Using base R
set.seed(123)  # for reproducibility
index_1 <- sample(1:nrow(ames), round(nrow(ames) * 0.7))
train_1 <- ames[index_1, ]
test_1  <- ames[-index_1, ]

# Using caret package
set.seed(123)  # for reproducibility
index_2 <- createDataPartition(ames$Sale_Price, p = 0.7, 
                               list = FALSE)
train_2 <- ames[index_2, ]
test_2  <- ames[-index_2, ]

# Using rsample package
set.seed(123)  # for reproducibility
split_1  <- initial_split(ames, prop = 0.7)
train_3  <- training(split_1)
test_3   <- testing(split_1)

# Resampling methods provide an alternative approach by allowing us to 
# repeatedly fit a model of interest to parts of the training data and test 
# its performance on other parts. The two most commonly used resampling methods
# include k-fold cross validation and bootstrapping.

# k-fold cross-validation (aka k-fold CV) is a resampling method that randomly
# divides the training data into k groups (aka folds) of approximately equal 
# size. When applying it externally to an ML algorithm as below, we'll need
# a process to apply the ML model to each resample, which we'll also cover
vfold_cv(ames, v = 10)

# A bootstrap sample is a random sample of the data taken with replacement 
# This means that, after a data point is selected for inclusion in the subset,
# its still available for further selection. A bootstrap sample is the same 
# size as the original data set from which it was constructed.
bootstraps(ames, times = 10)

# Putting the process together
split <- initial_split(ames, prop = 0.7, 
                       strata = "Sale_Price")
ames_train  <- training(split)
ames_test   <- testing(split)





# Data pre-processing and feature engineering

# Although not always a requirement, transforming the response variable can lead 
# to predictive improvement, especially with parametric models (which require 
# that certain assumptions about the model be met). For instance, ordinary 
# linear regression models assume that the prediction errors (and hence the 
# response) are normally distributed. This is usually fine, except when the 
# prediction target has heavy tails (i.e., outliers) or is skewed in one 
# direction or the other. In these cases, the normality assumption likely does 
# not hold.

# Using a log (or other) transformation to minimize the response skewness can 
# be used for shaping the business problems.

# log function in R
log(0.5)
log(-0.5) # will give errors because log can be applied only on positive numbers
log1p(-0.5) # log1p is the alternative that can be use when we encounter negative values

# exponential is the inverse operation of log which can be use in order to return
# to original values of a variables when we model with log values
exp(-0.69)

# Create a log variable withing the data frame
ames_log <- ames %>%
  mutate(log_sales_price = log(Sale_Price)) # mutate function helps us to create 
# variables within an existing data frame

# Let's visualize the Sales Price
ggplot(data = ames_log, aes(x=Sale_Price)) +
  geom_histogram() +
  labs(title = "Sale Price") +
  theme_minimal() # the histogram tells us that the variable is skewed

# Let's visualize now the log of Sales Price
ggplot(data = ames_log, aes(x=log_sales_price)) +
  geom_histogram( ) +
  labs(title = "Log transformed Sale Price") +
  theme_minimal() # the transformed variable seems to be closer to the normal distribution

# Another option would be to  transform most right skewed distributions to be 
# approximately normal. One way to do this is to simply log transform the 
# training and test set in a manual, single step manner similar to:
transformed_response <- log(ames_train$Sale_Price)

# However, we should think of the preprocessing as creating a blueprint to be 
# re-applied strategically. For this, you can use the recipe package
ames_recipe <- recipe(Sale_Price ~ ., data = ames_train) %>%
  step_log(all_outcomes())
ames_recipe

# If your response has negative values or zeros then a log transformation will 
# produce NaNs and -Inf. If the non positive response values are small (say 
# between -0.99 and 0) then you can apply a small offset such as in log1p() 
# which adds 1 to the value prior to applying a log transformation (you can do
# the same within step_log() by using the offset argument). If your data 
# consists of values < -1, use the Yeo-Johnson transformation mentioned later.

# A Box Cox transformation is more flexible than (but also includes as a 
# special case) the log transformation and will find an appropriate 
# transformation from a family of power transforms that will transform the 
# variable as close as possible to a normal distribution. At the core of the 
# Box Cox transformation is an exponent, lambda (??), which varies from -5 to 5.
# All values of ?? are considered and the optimal value for the given data is 
# estimated from the training data; The "optimal value" is the one which
# results in the best transformation to an approximate normal distribution.
ames_recipe <- recipe(Sale_Price ~ ., data = ames_train) %>%
  step_BoxCox(all_outcomes())
ames_recipe

# Be sure to compute the lambda on the training set and apply that same lambda
# to both the training and test set to minimize data leakage. The recipes 
# package automates this process for you.

# If your response has negative values, the Yeo-Johnson transformation is very
# similar to the Box-Cox but does not require the input variables to be 
# strictly positive.
ames_recipe <- recipe(Sale_Price ~ ., data = ames_train) %>%
  step_YeoJohnson(all_outcomes())
ames_recipe

# Note that when you model with a transformed response variable, your 
# predictions will also be on the transformed scale. You will likely want to 
# undo (or re-transform) your predicted values back to their normal scale so 
# that decision-makers can more easily interpret the results. This is 
# illustrated in the following code chunk:

# Log transform a value
y <- log(10)

# Undo log-transformation
exp(y)
## [1] 10

# Box Cox transform a value
y <- forecast::BoxCox(10, lambda=5)

# Inverse Box Cox function
inv_box_cox <- function(x, lambda) {
  # for Box-Cox, lambda = 0 --> log transform
  if (lambda == 0) exp(x) else (lambda*x + 1)^(1/lambda) 
}

# Undo Box Cox-transformation
inv_box_cox(y, lambda=5)



# Dealing with missingness

# It is important to understand the distribution of missing values (i.e., NA)
# in any data set. So far, we have been using a pre-processed version of the
# Ames housing data set. However, if we use the raw Ames housing data 
# there are actually 13,997 missing values-there is at least one missing 
# values in each row of the original data!

sum(is.na(AmesHousing::ames_raw))

# Let's visualize the missing values in the raw dataset  
AmesHousing::ames_raw %>%
  is.na() %>%
  reshape2::melt() %>%
  ggplot(aes(Var2, Var1, fill=value)) + 
  geom_raster() + 
  coord_flip() +
  scale_y_continuous(NULL, expand = c(0, 0)) +
  scale_fill_grey(name = "", 
                  labels = c("Present", 
                             "Missing")) +
  xlab("Observation") +
  theme(axis.text.y  = element_text(size = 6))

# Digging a little deeper into these variables, we might notice that 
# Garage_Cars and Garage_Area contain the value 0 whenever the other 
# Garage_xx variables have missing values (i.e. a value of NA). This might be
# because they did not have a way to identify houses with no garages when the 
# data were originally collected, and therefore, all houses with no garage 
# were identified by including nothing. Since this missingness is informative,
# it would be appropriate to impute NA with a new category level (e.g., "None") 
# for these garage variables. Circumstances like this tend to only become
# apparent upon careful descriptive and visual examination of the data!

AmesHousing::ames_raw %>% 
  filter(is.na(`Garage Type`)) %>% 
  select(`Garage Type`, `Garage Cars`, `Garage Area`)

# The vis_miss() function in R package visdat also allows for easy 
# visualization of missing data patterns (with sorting and clustering options).
# The columns of the heat map represent the 82 variables of the raw data and
# the rows represent the observations. Missing values (i.e., NA) are indicated 
# via a black cell. The variables and NA patterns have been clustered by rows 
# (i.e., cluster = TRUE).
vis_miss(AmesHousing::ames_raw, cluster = TRUE)

# Imputation

# Imputation is the process of replacing a missing value with a substituted, 
# "best guess" value. Imputation should be one of the first feature engineering
# steps you take as it will affect any downstream preprocessing

# Estimated statistic - An elementary approach to imputing missing values for
# a feature is to compute descriptive statistics such as the mean, median, or 
# mode (for categorical) and use that value to replace NAs. Although 
# computationally efficient, this approach does not consider any other 
# attributes for a given observation when imputing (e.g., a female patient 
# that is 63 inches tall may have her weight imputed as 175 lbs since that is 
# the average weight across all observations which contains 65% males that 
# average a height of 70 inches).

ames_recipe %>%
  step_medianimpute(Gr_Liv_Area) #Impute the median for Gr_Liv_Area

# Use step_modeimpute() to impute categorical features with the most common value.

# K-nearest neighbor imputes values by identifying observations with missing
# values, then identifying other observations that are most similar based on 
# the other available features, and using the values from these nearest 
# neighbor observations to impute missing values.

ames_recipe %>%
  step_knnimpute(all_predictors(), neighbors = 6)

# Tree-based - Similar to KNN imputation, observations with missing values 
# are identified and the feature containing the missing value is treated as 
# the target and predicted using bagged decision trees.
ames_recipe %>%
  step_bagimpute(all_predictors())

# Standardization

# We must also consider the scale on which the individual features are measured
# What are the largest and smallest values across all features and do they 
# span several orders of magnitude? Models that incorporate smooth functions 
# of input features are sensitive to the scale of the inputs.

# For these models and modeling components, it is often a good idea to 
# standardize the features. Standardizing features includes centering and 
# scaling so that numeric variables have zero mean and unit variance, which
# provides a common comparable unit of measure across all the variables.

ames_recipe %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes())


# Categorical feature engineering

# Most models require that the predictors take numeric form. There are 
# exceptions; for example, tree-based models naturally handle numeric or 
# categorical features. However, even tree-based models can benefit from 
# preprocessing categorical features. 

# Lumping - sometimes features will contain levels that have very few 
# observations. For example, there are 28 unique neighborhoods represented in 
# the Ames housing data but several of them only have a few observations.
count(ames_train, Neighborhood) %>% arrange(n)

# Even numeric features can have similar distributions. For example, 
# Screen_Porch has 92% values recorded as zero (zero square footage meaning 
# no screen porch) and the remaining 8% have unique dispersed values.
count(ames_train, Screen_Porch) %>% arrange(n)

# Sometimes we can benefit from collapsing, or "lumping" these into a lesser 
# number of categories. In the above examples, we may want to collapse all 
# levels that are observed in less than 10% of the training sample into an 
# "other" category. We can use step_other() to do so. However, lumping should
# be used sparingly as there is often a loss in model performance

# Lump levels for two features
lumping <- recipe(Sale_Price ~ ., data = ames_train) %>%
  step_other(Neighborhood, threshold = 0.01, 
             other = "other") %>%
  step_other(Screen_Porch, threshold = 0.1, 
             other = ">0")

# Apply this blue print --> you will learn about this at the end of the seminar
apply_2_training <- prep(lumping, training = ames_train) %>%
  bake(ames_train)

# New distribution of Neighborhood
count(apply_2_training, Neighborhood) %>% arrange(n)
# New distribution of Screen_Porch
count(apply_2_training, Screen_Porch) %>% arrange(n)


# One-hot & dummy encoding

# Many models require that all predictor variables be numeric. Consequently, 
# we need to intelligently transform any categorical variables into numeric 
# representations so that these algorithms can compute.

# Lump levels for two features
recipe(Sale_Price ~ ., data = ames_train) %>%
  step_dummy(all_nominal(), one_hot = TRUE)

# Since one-hot encoding adds new features it can significantly increase the 
# dimensionality of our data. If you have a data set with many categorical 
# variables and those categorical variables in turn have many unique levels, 
# the number of features can explode. In these cases you may want to explore 
# label/ordinal encoding or some other alternative.

# Label encoding is a pure numeric conversion of the levels of a categorical 
# variable. If a categorical variable is a factor and it has pre-specified
# levels then the numeric conversion will be in level order. If no levels are 
# specified, the encoding will be based on alphabetical order. For example, 
# the MS_SubClass variable has 16 levels, which we can recode numerically with
# step_integer().

# Original categories
count(ames_train, MS_SubClass)

# Label encoded
recipe(Sale_Price ~ ., data = ames_train) %>%
  step_integer(MS_SubClass) %>%
  prep(ames_train) %>%
  bake(ames_train) %>%
  count(MS_SubClass)

# We should be careful with label encoding unordered categorical features 
# because most models will treat them as ordered numeric features. If a 
# categorical feature is naturally ordered then label encoding is a natural 
# choice (most commonly referred to as ordinal encoding). For example, the 
# various quality features in the Ames housing data are ordinal in nature
# (ranging from Very_Poor to Very_Excellent).
ames_train %>% select(contains("Qual"))

# Ordinal encoding these features provides a natural and intuitive interpretation 
# and can logically be applied to all models.

# The various xxx_Qual features in the Ames housing are not ordered factors.
# For ordered factors you could also use step_ordinalscore()

# Original categories
count(ames_train, Overall_Qual)

# Label encoded
recipe(Sale_Price ~ ., data = ames_train) %>%
  step_integer(Overall_Qual) %>%
  prep(ames_train) %>%
  bake(ames_train) %>%
  count(Overall_Qual)

# Putting the process together 
blueprint <- recipe(Sale_Price ~ ., data = ames_train) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes())

blueprint # we apply all the wanted transformation

prepare <- prep(blueprint, training = ames_train)
prepare # we prepare the data

baked_train <- bake(prepare, new_data = ames_train)
baked_test <- bake(prepare, new_data = ames_test)
baked_train # in the end we bake it and export the processed data
