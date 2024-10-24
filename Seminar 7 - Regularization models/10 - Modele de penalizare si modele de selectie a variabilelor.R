# Metode de regularizare, algoritmi de selectie a variabilelor

# Cuprins:
#   Regresia Ridge
#   Regresia LASSO
#   Regresia Elastic Net

# Fisiere: 
#   wage1.csv
  
directory <- "C:/Users/40726/Desktop/Econometrie 2023/Seminar 9/"

# Install packages
PackageNames <- c("tidyverse", "stargazer", "magrittr", "car", "strucchange",
                  "ggplot2","caret", "splines","mgcv","glmnet","psych")
for(i in PackageNames){
  if(!require(i, character.only = T)){
    install.packages(i, dependencies = T)
    require(i, character.only = T)
  }
}


# Regresia Ridge - este un model care se foloseste cu precadere atunci cand 
# exista multicoliniaritate in date 

# Regresia liniara clasica incearca sa gaseasca estimari ale coeficientilor 
# care minimizeaza SSR
# SSR = sum((y-yfit)^2)
# Regresia ridge incearca sa minimizeze SSR + lambda*sum(beta^2)
# lambda*sum(beta^2) se mai numeste si shrinkage penalty 
# lambda ia valoarea a.i. sa produca cea mai mica valoare pt MSE 

# Importam setul de date cu privire la salariu
wage1 <- read.csv(paste0(directory, "wage1.csv"))

# Regresie liniara multipla
model0 <- lm(wage ~ educ + exper + tenure + female + married + south, wage1)
summary(model0)
prognoza <- data.frame(educ = c(15),
                       exper = c(5),
                       tenure = c(3),
                       female = c(1),
                       married = c(0),
                       south= c(1))
y_pred_scenariu <- predict(model0, newdata = prognoza)
y_pred_scenariu

# Definim variabila raspuns
y <- wage1$wage

# Definim predictorii
x <- data.matrix(wage1[, c('educ', 'exper', 'tenure', 'female',
                           'married', 'south')])

# Estimam modelul ridge (alpha = 0)
model <- glmnet(x, y, alpha = 0)
summary(model)

# In continuare vom identifica valoarea lui lambda pt care avem MSE minimizat
# utilizand validarea incrucisata (cross validation)
cv_model <- cv.glmnet(x, y, alpha = 0)
best_lambda <- cv_model$lambda.min
best_lambda # 0.15

# testarea valorii lamda 
plot(cv_model) 

# Reimplementam modelul cu valoarea lamda optima
best_model <- glmnet(x, y, alpha = 0, lambda = best_lambda)
coef(best_model) # coeficientii variabilelor 

# Diagrama Trace pentru a vizualiza modul in care estimarile coeficientulilor s-au
# modificat ca urmare a cresterii valorii lui lambda
plot(model, xvar = "lambda")
legend("bottomright", lwd = 1, col = 1:6, legend = colnames(x), cex = .7)

# Prognoze 
y_predicted <- predict(model, s = best_lambda, newx = x)

# Progoza out-of-sample
new <- matrix(c(15, 5, 3, 1,0,1), nrow=1, ncol=6) 
predict(best_model, s = best_lambda, newx = new)

# calcularea lui R2
sst <- sum((y - mean(y))^2)
sse <- sum((y_predicted - y)^2)
rsq <- 1 - sse/sst
rsq # 37.23%

###############################################################

# Regresia LASSO - functioneaza similar cu regresie Ridge doar ca incearca sa minimizeze
# SSR + lambda*sum(|beta|)
model <- glmnet(x, y, alpha = 1)

# Din punct de vedere tehnic, vom seta valoarea alpha = 1 pentru 
# regresia LASSO. 
cv_model <- cv.glmnet(x, y, alpha = 1)

# Valoarea optima a lui lambda
best_lambda <- cv_model$lambda.min
best_lambda # 0.006

# testarea valorii lamda
plot(cv_model) 


# Reimplementam modelul cu valoarea lamda optima
best_model <- glmnet(x, y, alpha = 1, lambda = best_lambda)
coef(best_model) # coeficientii variabilelor  # coeficientii variabilelor 
# daca unul din coeficienti este 0 inseamna ca acea variabila nu este importanta
# si de aceea nu o estimeaza modelul

# Diagrama Trace pentru a vizualiza modul in care estimarile coeficientulilor s-au
# modificat ca urmare a cresterii valorii lui lambda
plot(model, xvar = "lambda",label=T)
legend("bottomright", lwd = 1, col = 1:6, legend = colnames(x), cex = .7)

# Prognoze 
y_predicted <- predict(best_model, s = best_lambda, newx = x)

# Prognoza out-of-sample
# 'educ', 'exper', 'tenure', 'female','married', 'south'
new <- matrix(c(15, 5, 3, 1,0,1), nrow=1, ncol=6) 
predict(best_model, s = best_lambda, newx = new)

# calcularea lui R2
sst <- sum((y - mean(y))^2)
sse <- sum((y_predicted - y)^2)
rsq <- 1 - sse/sst
rsq # 37.27%

#######################################################################

# Elastic net regression - functioneaza similar cu Ridge si LASSO doar ca 
# adauga ambele penalitati SSR + lambda*sum(beta^2) + lambda*sum(|beta|)
model <- cv.glmnet(x, y, alpha = 0.5)
cv_model <- cv.glmnet(x, y, alpha = 0.5)

# Valoarea optima a lui lambda
best_lambda <- cv_model$lambda.min
best_lambda # 0.034


# testarea valorii lamda
plot(cv_model) 

# Reimplementam modelul cu valoarea lamda optima
best_model <- glmnet(x, y, alpha = 0.5, lambda = best_lambda)
coef(best_model) # coeficientii variabilelor 

# Prognoze 
y_predicted <- predict(best_model, s = best_lambda, newx = x)

# Prognoza out-of-sample
# 'educ', 'exper', 'tenure', 'female','married', 'south'
new <- matrix(c(15, 5, 3, 1,0,1), nrow=1, ncol=6) 
predict(best_model, s = best_lambda, newx = new)

# calcularea lui R2
sst <- sum((y - mean(y))^2)
sse <- sum((y_predicted - y)^2)
rsq <- 1 - sse/sst
rsq # 37.26%

# Vom compara valorile lui rsq si in functie de acestea vom alege modelul cu cea
# mai mare bonitate drept modelul optim 

 