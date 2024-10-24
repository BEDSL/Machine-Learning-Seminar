library(ggplot2)
library(dplyr)
library(caret)
library(randomForest)
library(e1071)
library(rpart) 
library(Metrics)

#### Polynomial regression example

#### When the relationship between the variables is better represented by a curve 
####rather than a straight line, polynomial regression can capture the non-linear 
####patterns in the data.

# The relationship between the dependent variable and the independent variable is modeled as an 
# nth-degree polynomial function. When the polynomial is of degree 2, it is called a quadratic model 
# (will produce a parabolic curve); when the degree of a polynomial is 3, it is called a cubic model 
# (will produce an S-shaped curve), and so on.


#make this example reproducible
set.seed(1)

#create dataset
df <- data.frame(hours = runif(50, 5, 15), score=50)
df$score = df$score + df$hours^3/150 + df$hours*runif(50, 1, 2)

#view first six rows of data
head(df)

#### visualize the data
ggplot(df, aes(x=hours, y=score)) +
  geom_point()

##### We can see that the data exhibits a bit of a quadratic relationship,
##### which indicates that polynomial regression could fit the data better than 
##### simple linear regression.

#### Fit the Polynomial Regression Models

# we’ll fit five different polynomial regression models with degrees h = 1…5 and
# use k-fold cross-validation with k=10 folds to calculate the test MSE for each model:

#randomly shuffle data
df.shuffled <- df[sample(nrow(df)),]

#define number of folds to use for k-fold cross-validation
K <- 10 

#define degree of polynomials to fit
degree <- 5

#create k equal-sized folds
folds <- cut(seq(1,nrow(df.shuffled)),breaks=K,labels=FALSE)

#create object to hold MSE's of models
mse = matrix(data=NA,nrow=K,ncol=degree)

#Perform K-fold cross validation
for(i in 1:K){
  
  #define training and testing data
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData <- df.shuffled[testIndexes, ]
  trainData <- df.shuffled[-testIndexes, ]
  
  #use k-fold cv to evaluate models
  for (j in 1:degree){
    fit.train = lm(score ~ poly(hours,j), data=trainData)
    fit.test = predict(fit.train, newdata=testData)
    mse[i,j] = mean((fit.test-testData$score)^2) 
  }
}

#find MSE for each degree 
colMeans(mse)

# Test MSE with degree h = 1: 9.80
# Test MSE with degree h = 2: 8.75
# Test MSE with degree h = 3: 9.60
# Test MSE with degree h = 4: 10.59
# Test MSE with degree h = 5: 13.55

# The model with the lowest test MSE turned out to be the polynomial regression model 
# with degree h =2.

#### Analyze final model

#fit best model
best = lm(score ~ poly(hours,2, raw=T), data=df)

#view summary of best model
summary(best)
 
# From the output we can see that the final fitted model is:
#   Score = 54.00526 – .07904*(hours) + .18596*(hours)2
# We can use this equation to estimate the score that a student will receive based on the number of hours they studied.
# For example, a student who studies for 10 hours is expected to receive a score of 71.81:
#   Score = 54.00526 – .07904*(10) + .18596*(10)2 = 71.81

##### plot the fitted model
ggplot(df, aes(x=hours, y=score)) + 
  geom_point() +
  stat_smooth(method='lm', formula = y ~ poly(x,2), linewidth = 1) + 
  xlab('Hours Studied') +
  ylab('Score')





##### SVR

# Support vector regression (SVR) is a type of support vector machine (SVM) that is used for 
# regression tasks. It tries to find a function that best predicts the continuous output value for a 
# given input value.

# With SVR, instead of a simple line, you'll see a tube. The regression line is in the middle and 
# the tube is around it. The tube itself is called the epsilon-insensitive tube, it can be seen as a
# marginal error that is allowing the model to have and not take into consideration the errors inside it.


# In R, we can implement SVR using the e1071 package, which provides the svm function for fitting support vector machines.



##### EXAMPLE 1

#Importing the dataset
dataset = read.csv('C:/Cursuri ML/Position_Salaries.csv')
dataset = dataset[2:3]

#missing value checking
sum(is.na(dataset$Level))

regressor = svm(formula = Salary ~ .,
                data = dataset,
                type = 'eps-regression',
                kernel = 'radial')

# the Salary is the dependent variable and the dot represents all our independent variables. 
# the type argument is very important 
# it defines for what purpose we are using the SVM model: for classification or for regression (C-classification or eps-regression)
# Kernel = radial means that we are using one non linear regression model
# The choice of kernel depends on the data’s characteristics and the task’s complexity.

# Predicting the new result
y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualising the SVR results

x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (SVR)') +
  xlab('Level') +
  ylab('Salary')


##### EXAMPLE 2

### generating data
x = 1:75
y = cumsum((rnorm(length(x))))

# plot the original data
makePlot <-function(x,y){
  plot(x,y,col="black",pch=5,lwd=1)
  lines(x,y,lty=2, lwd=2)
  grid()}
makePlot(x,y)
title("original data")

### fit a linear regression model

# make data frame named `Data`
Data<-data.frame(cbind(x,y))

# Create a linear regression model
linregress_model <- lm(y ~ x, data=Data)

# make predictions for regression model for each x val
predictYlinregress <- predict(linregress_model,Data)

# show predictions with orignal
makePlot(x,y)
title("original data + linear regression")
abline(linregress_model, col="red")

# The data has a non-linear pattern (with fluctuations around a general upward trend).
# The linear regression line fits poorly because it tries to approximate the relationship with a straight 
# line, which doesn't capture the oscillations in the data. This highlights the limitation of linear models
# for non-linear patterns.

# Error in linear regression model
errval <- linregress_model$residuals  # same as data$Y - predictedY

# RMSE function
rmse <- function(errval)
{
  val = sqrt(mean(errval^2))
  return(val)
}

linregress_RMSE <- rmse(errval)   
print(paste('logregress RMSE = ', 
            linregress_RMSE))

#### support vector regression

svm_model <- svm(y ~ x , Data)
#predicted vals for all X
predictYsvm <- predict(svm_model, Data)
# viz comparison
makePlot(x,y)
title("original data + linear regression + svr")
abline(linregress_model, col="red")
points(Data$x, predictYsvm, col = "blue", pch=4)
points(Data$x, predictYsvm, col = "blue", type="l")

# The red line (linear regression) doesn't capture the non-linear pattern well, as it is a straight line.
# The blue line (SVR model) fits the data much better, capturing the underlying non-linear trend of the points.


### model errors
errval <- Data$y - predictYsvm 
svr_RMSE <- rmse(errval)   
print(paste('svr RMSE = ', 
            svr_RMSE))

# 
# linear RMSE = 1.43136
# SVR RMSE = 1.28203
# linear model has a higher error compared to the SVR model

### tune SVM model regression

# perform a grid search 
# (this might take a few seconds, adjust how fine of grid if taking too long)
tuneResult1 <- tune(svm, y ~ x,  data = Data,
                    ranges = list(epsilon = seq(0,1,0.1), cost = 2^(seq(0.5,8,.5)))
)

## go through different variations of cost and epsilon (parameters of svm function) to get the best model

# Map tuning results
plot(tuneResult1)

# The best model is the one with lowest MSE. The darker the region the lower the RMSE, 
# which means better the model. In our model the lowest RMSE is at epsilon 0.4 and cost 64. 

print(tuneResult1)

tuneResult <- tune(svm, y ~ x,  data = Data,
                   ranges = list(epsilon = seq(tuneResult1$best.model$epsilon-.15,
                                               tuneResult1$best.model$epsilon+.15,
                                               0.01), 
                                 cost = seq(2^(log2(tuneResult1$best.model$cost)-1),
                                            2^(log2(tuneResult1$best.model$cost)+1),
                                            length=6))
)

plot(tuneResult)


#### Optimized tuned parameters
print(tuneResult)
# best parameters:
#   epsilon cost
#     0.27  128

# predict with tuned and vizualize the data

#predicted vals for all X for tuned
tunedVals <-tuneResult$best.model
predictYsvm2 <- predict(tunedVals, Data)

# viz comparison
makePlot(x,y)
title("original data + linear regression + svr + tuned svm")
abline(linregress_model, col="red")
points(Data$x, predictYsvm, col = "blue", pch=4)
points(Data$x, predictYsvm, col = "blue", type="l")
points(Data$x, predictYsvm2, col = "green", pch=5)
points(Data$x, predictYsvm2, col = "green", type="l")
legend("bottomleft", # places a legend at the appropriate place 
       c("Data","Linear regress","SVM regress","tuned SVM regress"))

### compare errors
errval2 <- Data$y - predictYsvm2 
svr_RMSE2 <- rmse(errval2)   

vals <- matrix(c(linregress_RMSE,svr_RMSE,svr_RMSE2),ncol=3,byrow=TRUE)
colnames(vals) <- c("Lin regress  ","SVM model  ","Tuned SVM model ")
rownames(vals) <- c("RMSE of model")
as.table(vals)

#### the tuned model has the lowest RMSE (1.080989)




##### Decision tree regression 

##### Example 1

fit <- rpart(Sepal.Width ~ Sepal.Length + 
               Petal.Length + Petal.Width + Species,  
             method = "anova", data = iris)

# Plot 
plot(fit, uniform = TRUE, 
     main = "Sepal Width Decision Tree using Regression") 
text(fit, use.n = TRUE, cex = .7) 
# 
# The predicted sepal width at each terminal node (leaf) is displayed, along with the number of samples in that leaf.
# 
# Top Node: The root starts with all samples and splits based on the species.
# Left Subtree: The branches corresponding to "versicolor" and "virginica" species split based on petal length and other features like petal width and sepal length.
# Right Subtree: The "setosa" species has its own splits based on petal length and sepal length.
# Key Insights:
#   The first important feature for splitting is the species (whether it is setosa or one of the other two).
# For "versicolor" and "virginica," further splits are determined primarily by petal length, petal width, and sepal length.
# For "setosa," the splits are influenced by petal length and sepal length.

print(fit) 

# Root Node:
#   
#   The root node contains all 150 samples (n = 150).
# The deviance, a measure of error or variance at this node, is 28.30693.
# The predicted sepal width (yval) is 3.05733 (the mean of the response variable in this node).
# Split at Node 2:
#   
#   The first split is based on the species being either "versicolor" or "virginica" (excluding "setosa").
# 
# There are 100 samples in this split, and the predicted sepal width is 2.87200.
# 
# Further Split on Petal Length < 4.05:
#   
#   When the petal length is less than 4.05, the predicted sepal width is 2.487 (n = 16, deviance = 0.7975), and this is a terminal node (denoted by *).
# When petal length is greater than or equal to 4.05, the tree branches out again.
# Further Split on Petal Width < 1.95:
#   
#   When petal width is less than 1.95, the predicted sepal width is 2.806 (terminal node).
# 
# When petal width is greater than 1.95, the prediction further splits on sepal length.
# 
# Further Split on Sepal Length:
#   
#   For sepal length < 6.35, the predicted sepal width is 2.805 (terminal).
# For sepal length ≥ 6.35, the prediction for sepal width is 2.963 (terminal).
# Split at Node 3 (Setosa Species):
#   
#   For the species "setosa," the predicted sepal width is 3.46000.
# The prediction splits based on petal length and sepal length:
#   For petal length < 5.05, the predicted sepal width is 3.204.
# For petal length ≥ 5.05, the split continues with further conditions, resulting in final predicted values of 3.714 and 3.168.



# Create test data 
df  <- data.frame (Species = 'versicolor',  
                   Sepal.Length = 5.1, 
                   Petal.Length = 4.5, 
                   Petal.Width = 1.4) 

# Predicting sepal width 
# using testing data and model 
# method anova is used for regression 
cat("Predicted value:\n") 
predict(fit, df, method = "anova") 






##### Example 2
fit <- rpart(mpg ~ disp + hp + cyl,  
             method = "anova", data = mtcars ) 

# Output to be present as PNG file 
png(file = "decTree2GFG.png", width = 600, 
    height = 600) 

# Plot 
plot(fit, uniform = TRUE, 
     main = "MPG Decision Tree using Regression") 
text(fit, use.n = TRUE, cex = .6) 

# Saving the file 
dev.off() 

# Print model 
print(fit) 


# Create test data 
df  <- data.frame (disp = 351, hp = 250,  
                   cyl = 8) 

# Predicting mpg using testing data and model 
cat("Predicted value:\n") 
predict(fit, df, method = "anova") 


##### Random forest regression
# 
# Random forest is a supervised learning algorithm, meaning that the data on which it operates contains 
# labels or outcomes. It works by creating many decision trees, each built on randomly chosen subsets of
# the data. The model then aggregates the outputs of all of these decision trees to make an overall 
# prediction for unseen data points. In this way, it can process larger datasets and capture more complex
# associations than individual decision trees.

# In this example we will try to forecast the value of diamonds using the Diamonds dataset (part of ggplot2). 

# Import the dataset
diamond <-diamonds
head(diamond)
# The dataset contains information on 54,000 diamonds. It contains the price as well as 9 other attributes. 
# Some features are in the text format, and we need to encode them in numerical format. Let’s also drop 
# the unnamed index column.

# Convert the variables to numerical
diamond$cut <- as.integer(diamond$cut)
diamond$color <-as.integer(diamond$color)
diamond$clarity <- as.integer(diamond$clarity)

head(diamond)

# One of the advantages of the Random Forest algorithm is that it does not require data scaling, 
# as previously stated. To apply this technique, all we need to do is define the features and the target
# we’re attempting to predict. By mixing the available attributes, we might potentially construct a variety 
# of features. 

# Create features and target
X <- diamond %>% 
  select(carat, depth, table, x, y, z, clarity, cut, color)
y <- diamond$price

# At this point, we have to split our data into training and test sets. As a training set, 
# we will take 75% of all rows and use 25% as test data.

# Split data into training and test sets
index <- createDataPartition(y, p=0.75, list=FALSE)
X_train <- X[ index, ]
X_test <- X[-index, ]
y_train <- y[index]
y_test<-y[-index]
# Train the model 
regr <- randomForest(x = X_train, y = y_train , maxnodes = 10, ntree = 10)

# We now have a model that has been pre-trained and can predict values for the test data. 
# The model’s accuracy is then evaluated by comparing the predicted value to the actual values 
# in the test data. We will present this comparison in the form of a table and plot the price and carat
# value to make it more illustrative.

# Make prediction
predictions <- predict(regr, X_test)

result <- X_test
result['price'] <- y_test
result['prediction']<-  predictions

head(result)


# Build scatterplot
ggplot(  ) + 
  geom_point( aes(x = X_test$carat, y = y_test, color = 'red', alpha = 0.5) ) + 
  geom_point( aes(x = X_test$carat , y = predictions, color = 'blue',  alpha = 0.5)) + 
  labs(x = "Carat", y = "Price", color = "", alpha = 'Transperency') +
  scale_color_manual(labels = c( "Predicted", "Real"), values = c("blue", "red")) 

# 
# The figure displays that predicted prices (blue scatters) coincide well with the real ones (red scatters),
# especially in the region of small carat values. But to estimate our model more precisely, we will look at
# Mean absolute error (MAE), Mean squared error (MSE), and R-squared scores.


print(paste0('MAE: ' , mae(y_test,predictions) ))
## [1] "MAE: 742.401258870433"
print(paste0('MSE: ' ,caret::postResample(predictions , y_test)['RMSE']^2 ))
## [1] "MSE: 1717272.6547428"
print(paste0('R2: ' ,caret::postResample(predictions , y_test)['Rsquared'] ))
## [1] "R2: 0.894548902990278"

# We get a couple of errors (MAE and MSE). We should modify the algorithm’s hyperparameters to improve 
# model’s predictive power. We could do it by hand, but it would take a long time.
# 
# We’ll need to build a custom Random Forest model to get the best set of parameters for our model and 
# compare the output for various combinations of the parameters in order to tune the parameters ntrees 
# (number of trees in the forest) and maxnodes (maximum number of terminal nodes trees in the forest can have).


#### Tuning the parameters
# If training the model takes too long try setting up lower value of N

N=500 #length(X_train)
X_train_ = X_train[1:N , ]
y_train_ = y_train[1:N]

seed <-7
metric<-'RMSE'

customRF <- list(type = "Regression", library = "randomForest", loop = NULL)

customRF$parameters <- data.frame(parameter = c("maxnodes", "ntree"), class = rep("numeric", 2), label = c("maxnodes", "ntree"))

customRF$grid <- function(x, y, len = NULL, search = "grid") {}

customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  randomForest(x, y, maxnodes = param$maxnodes, ntree=param$ntree, ...)
}

customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata)
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata, type = "prob")
customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes
# Set grid search parameters
control <- trainControl(method="repeatedcv", number=10, repeats=3, search='grid')

# Outline the grid of parameters
tunegrid <- expand.grid(.maxnodes=c(70,80,90,100), .ntree=c(900, 1000, 1100))
set.seed(seed)

# Train the model
rf_gridsearch <- train(x=X_train_, y=y_train_, method=customRF, metric=metric, tuneGrid=tunegrid, trControl=control)


### visualize the impact of tuned parameters on RMSE

plot(rf_gridsearch)

# the plot shows how the model’s performance develops with different variations of the parameters. 
# For values maxnodes: 80 and ntree: 900, the model seems to perform best. We would now use these 
# parameters in the final model.

rf_gridsearch$bestTune
##   maxnodes ntree
##        70  1000

 
#### Defining and visualizing variables importance

# Let’s build the plot with a features list on the y axis. On the X-axis we’ll have an incremental 
# decrease in node impurities from splitting on the variable, averaged over all trees, it is measured 
# by the residual sum of squares and therefore gives us a rough idea about the predictive power of the 
# feature. 

varImpPlot(rf_gridsearch$finalModel, main ='Feature importance')

# he size of the diamond (x,y,z refer to length, width, depth) and the weight (carat) 
# contains the major part of the predictive power.