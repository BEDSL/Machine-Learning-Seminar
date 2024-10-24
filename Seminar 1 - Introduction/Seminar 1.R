# Script outline

# A. Installation Instructions for R and RStudio
# B. RStudio interface overview
# C. Install and manage packages in RStudio
# D. R's built in datasets
# E. Basic arithmetics with R
# F. Variable assignment
# G. Data types in R
# H. Vector, matrix, data frames and lists
# I. Reading data in R from csv and Excel sheet

# A. Installation Instructions for R and RStudio
# Install R

# Instructions:
# 1. Go to the website https://www.r-project.org/
# 2. click Download R - CRAN (Comprehensive R archive network) Mirros window
# 3. Select a version mirrored in Canada
# 4. Click to Download R for Windows, Mac or Linux
# 5. To launch it once installed double-click on the R icon

# Install RStudio

# Instructions:
# 1. Go to the website https://www.rstudio.com/
# 2. Click Download RStudio 
# 3. Select RStudio Desktop Free version
# 4. Click on the version you need - this will download the package
# 5. Install it
# 6. To launch it once installed double-click on the RStudio icon

# B. RStudio interface overview

#    4 main windows: Script, Console, Environment/History and Plots/Others
#    Script - where the actual writing or programming takes place
#    Console - mainly used for output
#    Environment/History - contains all variables/stores all the commands
#    Files/Plots/Packages/Help 

# C. Install and manage packages in RStudio

# What is a package?
# Packages are collections of R functions, data, and compiled code in a well-defined format.
# An R package is an extension of R
# containing data sets and specific functions to solve specific questions.
# How many packages have been developed so far?
nrow(available.packages()) # 14733 packages in october 2019, 16278 in september 2020, 17866 in september 2021

# Find packages by browsing R package lists https://cran.r-project.org/
browseURL("http://cran.r-project.org/web/views/") # Opens CRAN Task Views in browser categorical sorting of packages
browseURL("http://cran.stat.ucla.edu/web/packages/available_packages_by_name.html") # Opens "Available CRAN Packages By Name" (from UCLA mirror) in browser
browseURL("http://crantastic.org/") # See also CRANtastic external source, most used, latest updates

# See current packages
library()  # Brings up editor list of installed packages
search()   # Shows packages that are currently loaded
.libPaths() # shows where your library is located in your computer

# Install and use packages
# Can use menus: Tools > Install Packages...
# or use Packages tab from the RStudio interface window
# or can use scripts, which can be saved and incorporated in source (recommended)
# the function install.packages("package_name") is used to install a package from CRAN
install.packages("ggplot2", dependencies = TRUE)  # Downloads package called ggplot2 from CRAN and installs in R
library(ggplot2)

?install.packages            # Opens help window for installing packages
library("ggplot2")           # Makes package available; often used for loading in scripts
                             # Take a look at Packages tab
require("ggplot2")           # Preferred for loading in functions; maybe better?
library(help = "ggplot2")    # Brings up documentation in editor window
browseURL("http://cran.stat.ucla.edu/web/packages/ggplot2/index.html")
# ggplot2 - Create Elegant Data Visualisations Using the Grammar of Graphics

# VIGNETTE - guide to your package.
# A vignette is like a book chapter or an academic paper:
# it can describe the problem that your package is designed to solve,
# and then show the reader how to solve it. 
?vignette                          # Opens help window for vignette
vignette(package = "ggplot2")      # Brings up list of vignettes in editor window
browseVignettes(package = "ggplot2")  # Open web page with hyperlinks for vignette PDFs 

# Update packages
# In RStudio, Tools > Check for Package Updates
# or in Packages > Update button
update.packages()  # Checks for updates; do periodically
?update.packages

# Unload/Remove packages
# By default, all loaded packages are unloaded when R quits.
# Can also open Packages window and manually uncheck
# or can use this code to unload packages
detach("package:ggplot2", unload = TRUE) # take a look at Packages tab
remove.packages("ggplot2") # removes or deletes the package

# To permanently remove (delete) package
install.packages("moments", dependencies = TRUE)  # Adds package called moments
library("moments")           # Makes package available; often used for loading in scripts
remove.packages("moments")   # Deletes it
?remove.packages

# Exercise 1 - for package readxl: install, load, read documentation, detach and remove

install.packages("readxl")               # Install package readxl
library (readxl)                         # Load package psych
browseVignettes(package = "readxl")      # Read documentation for the package psych
update.packages("readxl")                # Update package
detach("package:readxl", unload = TRUE)  # Detach package
remove.packages("readxl")                # Remove package


# D. R's built in datasets

install.packages("datasets.load") # Install the datasets.load package, if not yet available
install.packages("datasets", dependencies = TRUE)      # Install the datasets package, if not yet available
?datasets                         # R datasets package description
library(help = "datasets")        # R dataset package description & datasets list & description
data()                            # To see a list of the available datasets of package datasets
library(datasets)                 # Load the datasets package
require(datasets)                 # Load the datasets package
detach(package:datasets)          # To detach the datasets package
# You can see the same list with clickable links to descriptions for each dataset at
browseURL("http://stat.ethz.ch/R-manual/R-devel/library/datasets/html/00Index.html") 
?airmiles       # For information on a specific dataset
library(datasets)

?AirPassengers
data(AirPassengers)  # To load a dataset from the package into the Workspace
AirPassengers        # To see the contents of the dataset (Don't actually need to load for this)
class(AirPassengers) # ts means a time series dataset
start(AirPassengers) # first moment in the time series, in this case 1st month of 1949
end(AirPassengers)   # last moment in the time series, in this case 12th month of 1960
View(AirPassengers)  # To see the contents or the recorded values
str(AirPassengers)   # To see its "structure", it is a time series
# plot the time series using the simplet function called plot
plot(AirPassengers, # the observed values in the data set
     main = "Number of the Monthly International Airline Passengers 1949-1960", # title
     xlab = "Time period", # label for the horizontal axis or x axis
     ylab = "Passengers, in thousands", # label for the vertical axis or y axis
     col = "darkgreen", # color of the line
     lwd = 2) # line width given as multiple of the default, which is 1
# The time evolution of the number of passengers is strongly influenced by the seasonal component
# An additional graph can help to visualize a seasonal decomposition.
# The function seasonplot() in the "forecast" package
# might help the viewer identifying trend and similar seasonal pattern each year
install.packages("forecast", dependencies = TRUE)
library(forecast)
seasonplot(AirPassengers,
           year.labels=TRUE,
           col = rainbow(12),
           lwd = 2,
           ylab = "Passengers, in thousands",
           main="")

rm(AirPassengers) # Removes dataset from the global environment (not from the library)
# rm(list = ls()) # Clean up the workspace
# CTRL+l        to clear console 
# ALT+SHIFT+k   to learn more about shortcut keys in R 


# E. Basic arithmetics with R

# Used as a simple calculator. Consider the following arithmetic operators:
#   Addition: +
#   Subtraction: -
#   Multiplication: *
#   Division: /
5/3
#   Exponentiation: ^ 
#       raises the number to its left to the power of the number to its right (3^2 = 9)
#   Modulo: %% 
#       returns the remainder of the division of the number to the left by the number on its right (5 %% 3 = 2).
5%%3
#   x %/% y  means   x divided by y but rounded down (integer divide) (5 %/% 3 = 1)	
5%/%3


# PRACTICE

# Calculate 3 + 4
3+4
# Calculate 6 + 12
# An addition
# A subtraction
# A multiplication
# A division
# Exponentiation
# Modulo


# F. VARIABLE ASSIGNMENT

# A basic concept in (statistical) programming is called a variable.
# A variable allows you to store a value (e.g. 4) or an object (e.g. a function description) in R.
# You can then later use this variable's name to easily access the value
# or the object that is stored within this variable

my_height_cm <- 175 # Assigns the value 175 to my_height_cm
my_height_cm        # Prints out the value of the variable my_height_cm

my_weight_kg <- 70
my_weight_kg

# BMI means Body Mass Index
BMI <- my_weight_kg/(my_height_cm/100)^2
BMI


# PRACTICE
# Assign the value 42 to x
x <- 42

# Print out the value of the variable x
x

# Assign the value 8 to y
y <- 8

# Print out the value of the variable y
y

# Add these two variables together
x+y

# Assign the result of adding x and y to a new variable named suma
sum <- x+y
sum



# G. BASIC DATA TYPES IN R

# R works with numerous data types. The most basic types are:
# 1. Decimals values like 4.5 are called numerics.
# 2. Natural numbers like 4 are called integers. Integers are also numerics.
# 3. Boolean values (TRUE or FALSE) are called logical.
# 4. Text (or string) values are called characters.

# PRACTICE
# Assign my_numeric the value of 42
my_numeric <- 42
my_numeric
class(my_numeric) # solicitam tipul variabilei sau al obiectului

# Assign my_character the word "universe" - use quotation marks
my_character <- "universe"
my_character
class(my_character)  # solicitam tipul variabilei sau al obiectului

my_name <- "Adriana"
my_name
class(my_name)


# Assign my_logical to be FALSE
my_logical <- FALSE
my_logical
class(my_logical) # solicitam tipul variabilei sau al obiectului

friday_seminar <- FALSE
friday_seminar
class(friday_seminar)



# Add variables my_numeric with my_character
my_numeric + my_character
# Got an error due to a mismatch in data types?
# You can avoid this by checking the data type of a variable beforehand.
# You can do this with the class() function.
# Check class of my_numeric
class(my_numeric)
# Check class of my_character
class(my_character)
# Check class of my_logical
class(my_logical)

class(airmiles) # ts class

# H. VECTORS, matrix, dataframes and lists

# Vectors are arrays that can hold
# numeric data, character data, or logical data.
# In other words, a vector is a simple tool to store data.
# In R, you create a vector with the concatenate function denoted c().
# You place the vector elements separated by a comma between the parentheses.
vector_numeric <- c(1, 10, 49)
vector_numeric
class(vector_numeric) # tipul obiectului
length(vector_numeric) # lungimea obiectului, adica numarul de elemente

vector_numeric[2] # elementul de pe pozitia 2 din vector_numeric


vector_character <- c("a", "b", "c")
vector_character
class(vector_character)
length(vector_character)

vector_nume <- c("Alex", "Bogdan", "Miruna")
vector_nume

vector_boolean <- c(TRUE, FALSE, TRUE)
vector_boolean
class(vector_boolean)

# Let's introduce the following two vectors
w1cashflow <- c(140, -50, 20, -120, 240) # revenues and expenses for week 1
w2cashflow <- c(-24, -50, 100, -350, 10) # revenues and expenses for week 2

w1cashflow
w2cashflow

?transform
transform(w1cashflow)
transform(w2cashflow)

w1cashflow[2] # the 2nd element in vector w1cashflow
w2cashflow[4] # the 4th element in vector w2cashflow

# It is important to have a clear view on the data that you are using.
# Understanding what each element refers to is therefore essential.
# We created a vector with your revenues and expenses over the working days of a week.
# Each vector element refers to a day of the week,
# but it is hard to tell which element belongs to which day.
# It would be nice if you could show that in the vector itself.
# You can give a name to the elements of a vector with the names() function. 

# OPTION 1
# Assign days as names of w1cashflow
names(w1cashflow) <- c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
names(w1cashflow)
w1cashflow
str(w1cashflow)
class(w1cashflow)

nume_w1cashflow <- names(w1cashflow)
nume_w1cashflow
class(nume_w1cashflow)

# Assign days as names of w2cashflow
names(w2cashflow) <- c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")

# OPTION 2
# Create a variable days_vector that contains the days of the week 
days_vector <- c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
days_vector

rm(w1cashflow)
rm(w2cashflow)
rm(nume_w1cashflow)

w1cashflow <- c(140, -50, 20, -120, 240) # revenues and expenses for week 1
w2cashflow <- c(-24, -50, 100, -350, 10) # revenues and expenses for week 2

names(w1cashflow) <- days_vector
w1cashflow

names(w2cashflow) <- days_vector
w2cashflow

# Arithmetic calculations on vectors.

# It is important to know that if you sum two vectors in R,
# it takes the element-wise sum.
# For example, the following three statements are completely equivalent
# c(1, 2, 3) + c(4, 5, 6)
# c(1 + 4, 2 + 5, 3 + 6)
# c(5, 7, 9)

# Practice exercise
# Create a vector A containing numbers 1,2,3 and a Vector B containing 4,5,6.
# Add vectors A and B in vector T and print results of vector T
A <- c(1,2,3)
A
B <- c(4,5,6)
B
T <- A+B
T

# Assign to total_biweekly how much you won/lost on each day
total_biweekly <- w1cashflow + w2cashflow
total_biweekly # :)
str(total_biweekly)
class(total_biweekly)

# Use function sum(), to calculate the sum of all elements of a vector. 
total_w1<-sum(w1cashflow)
total_w1
total_w2<-sum(w2cashflow)
total_w2
# Compare values
total_w1>total_w2 # returns the logical value
total_w1<total_w2

# Compare vectors
w1cashflow > w2cashflow # :)

# SELECT SPECIFIC VECTOR ELEMENTS
# To select elements of a vector (and later matrices, data frames, ...),
# you can use square brackets.
# Between the square brackets, you indicate what elements to select.
# For example, to select the first element of the vector, you type w1cashflow[1].
# To select the second element of the vector, you type w1cashflow[2], etc.
# Notice that the first element in a vector has index 1, not 0 as in many other programming languages.
w1_Wednesday <- w1cashflow[3]
w1_Wednesday # take a look at the global environment

w1_Thursday <- w1cashflow[days_vector[4]]
w1_Thursday

# SELECT MULTIPLE VECTOR ELEMENTS
# To select multiple elements from a vector, you can add square brackets at the end of it.
# You can indicate between the brackets what elements should be selected.
# For example: suppose you want to select the first and the fifth day of the week:
# use the vector c(1, 5) between the square brackets.
w1_midweek_opt1 <- w1cashflow[c(2,3,4)]  # option 1 to select the values in midweek days
w1_midweek_opt1
w1_midweek_opt2 <- w1cashflow[2:4]       # option 2
w1_midweek_opt2
w1_midweek_opt3 <- w1cashflow[c("Tuesday", "Wednesday", "Thursday")] # option 3
w1_midweek_opt3

mean(w1_midweek_opt1)  # calculates average
?var # variance
var(w1_midweek_opt1)
?sd()
sd(w1_midweek_opt1)

# SELECTION BY COMPARISON
# By making use of comparison operators. The comparison operators known to R are:
# <  for less than
# >  for greater than
# <= for less than or equal to
# >= for greater than or equal to
# == for equal to each other
# != not equal to each other

# Comparisons
# 6 > 5 returns TRUE.
6 > 5
# The nice thing about R is that you can use these comparison operators also on vectors
# c(4, 5, 6) > 5 returns [1] FALSE FALSE TRUE.
# This command tests for every element of the vector if the condition stated by the comparison operator is TRUE or FALSE.
c(4, 5, 6) > 5


# PRACTICE
# Check which elements in w1cashflow are positive (i.e. > 0)
# and assign this to selection_vector.
# Print out selection_vector so you can inspect it.
# The printout tells you whether you had an inflow (TRUE) or an outflow (FALSE) for each day.
w1cashflow # let's remind the components of w1cashflow
selection_vector <- w1cashflow > 0
selection_vector # vector of logical data, names are kept

# select only the subset of your vector which has inflows
positive_vector<-w1cashflow[selection_vector] #selects logical data with TRUE value
positive_vector # a new named numerical valued vector

# Check which elements in w1cashflow are negative (i.e. < 0)
nselection_vector <- w1cashflow < 0
w1cashflow
nselection_vector # vector of logical data, names are kept
# select only the subset of your vector which has outflows
negative_vector<-w1cashflow[nselection_vector]
negative_vector #a new named numerical valued vector

# MATRIX

# In R, a matrix is a collection of elements of the same data type (numeric, character, or logical)
# arranged into a fixed number of rows and columns.
# Since you are only working with rows and columns,
# a matrix is called two-dimensional.
# You can construct a matrix in R with the matrix() function. 
#   
?matrix
# matrix(data, # sirul de valori care reprezinta elementele matricei,
#        nrow = numarul de linii al matricei,
#        ncol = numarul de coloane al matricei,
#        byrow = TRUE sau FALSE daca sirul de valori se aseaza pe linie sau pe coloana)
# rm(A)
A <- matrix(data = c(-2, 5, 4, 14, 2, 3, 7, -3, 1),
            nrow = 3,
            ncol = 3,
            byrow = TRUE)
A
class(A) # tipul obiectului
A[2,1] # returneaza elementul de pe pozitia linia 2, coloana 1

A[2,] # returneaza cea de a 2-a linie a matricei
A[,3] # returneaza cea de a 3-a coloana a matricei

B <- matrix(data = c(4, 5, 4, 14, 2, 9, 7, -3, 2),
            nrow = 3,
            byrow = TRUE)
B

# Operatii cu matrice
A+B # operatia obisnuita de adunare a matricelor

A
2 + A # adunarea unui scalar cu o matrice, se adauga scalarul la fiecare element al matricei

3*A # inmultirea obisnuita a unui scalar cu o matrice

A %*% B # produsul obisnuit a doua matrice

A
B
A*B # inmultirea componenta cu componenta a elementelor matricelor
A/B # impartirea componenta cu componenta a elementelor matricelor


C <- matrix(data = c(3, 2, 5, 9),
            nrow = 2,
            byrow = FALSE)
D <- matrix(data = c(4, 6, 5, 0),
            nrow = 2,
            byrow = FALSE)
C
D
C/D

det(A) # determinantul unei matrice patratica

# inversa unei matrice patratice se calculeaza cu functia solve(matrice)
solve(A)
round(solve(A), digits = 4)

A %*% solve(A)
round(A %*% solve(A), digits = 4)


#   matrix(1:9, byrow = TRUE, nrow = 3)
matrix(1:9, # sirul de numere consecutive de la 1 la 9
       byrow = TRUE,
       nrow = 3)
# 
# In the matrix() function:
# the first argument is the collection of elements that R will arrange
# into the rows and columns of the matrix.
# Here, we use 1:9 which is a shortcut for c(1, 2, 3, 4, 5, 6, 7, 8, 9).
# The argument byrow indicates that the matrix is filled by the rows.
# If we want the matrix to be filled by the columns, we just place byrow = FALSE.
# The third argument nrow indicates that the matrix should have three rows.

y1 <- c(460.998, 314.4)
y2 <- c(290.475, 247.900)
y3 <- c(309.306, 165.8)
y1to3 <- c(y1,y2,y3)  # combine the three vectors into one vector
y1to3
y1to3_matrix <- matrix(y1to3, byrow=TRUE, nrow =3) # create matrix
y1to3_matrix     # Print out the matrix
class(y1to3_matrix)

# label columns and rows
year <- c(1970,1980,1990) # create row label vector
year
y1to3_matrix
rownames(y1to3_matrix) <- year # attach row label vector to matrix
y1to3_matrix # each line of the matrix has a name, 1970, 1980, 1990
location <- c("Canada", "non-Canada") # create column label vector
location
y1to3_matrix 
colnames(y1to3_matrix) <- location    # attach column label vector to matrix
y1to3_matrix
class(y1to3_matrix)

rm(y1to3_matrix) # remove the object from environment or workspace
y1to3_matrix     # the object was not found

# Another option to label rows and columns in a matrix
y1to3_matrix <- matrix(y1to3,         # the elements of the matrix
                       nrow = 3,      # number of rows
                       byrow = TRUE,  # the elements are given by row
                       dimnames = list(c(1970,1980,1990),          # labels for raws
                                       c("Canada", "non-Canada"))) # labels for columns
y1to3_matrix # print out the matrix

total_by_row<- rowSums(y1to3_matrix) # calculates sum by each row - inside and outside Canada 
total_by_row
new_matrix <- cbind(y1to3_matrix, total_by_row) # create a new matrix adding a new colum
new_matrix # names are kept and the new column is labeled accordingly

# rbind(A,x) attaches a row containing vector x to a matrix A
# cbind(A,x) attaches a column containing vector x to a matrix A
# colSums(A) sums values of matrix A by columns
# rowSums(A) sums values of matrix A by rows

# Selecting elements in a matrix
# my_matrix[1,2] selects the element at the first row and second column
# my_matrix[1:3,2:4] results in a matrix with the data on the rows 1, 2, 3 and columns 2, 3, 4.
# my_matrix[,1] selects all elements of the first column.
# my_matrix[1,] selects all elements of the first row.
# Similar to what you have learned with vectors,
# the standard operators like +, -, /, *, etc. work in an element-wise way on matrices in R.
# For example, 2 * my_matrix multiplies each element of my_matrix by two.

# FACTORS
# The term factor refers to a statistical data type used to store categorical variables.
# The difference between a categorical variable and a continuous variable
# is that a categorical variable can belong to a limited number of categories.
# A continuous variable, on the other hand, can correspond to an infinite number of values.
# To create factors in R, you make use of the function factor().
# First thing that you have to do is create a vector
# that contains all the observations that belong to a limited number of categories.
# For example, gender_vector contains the gender of 5 different individuals:
gender_vector <- c("Male", "Female", "Female", "Male", "Male")
gender_vector
class(gender_vector) # returns character as type
length(gender_vector) # vector de lungime 5

#The function factor() will encode the vector as a factor:
factor_gender_vector <- factor(gender_vector)
factor_gender_vector
class(factor_gender_vector)   # returns factor as type
levels(factor_gender_vector)  # returns the different categories of this vector

X <- c(-3, 14, 11,2)
X
class(X)
sum(X)


factor_X <- factor(X)
factor_X
class(factor_X)
sum(factor_X)

# There are two types of categorical variables: 
# a nominal categorical variable and an ordinal categorical variable.
# A nominal variable is a categorical variable without an implied order.
# This means that it is impossible to say that 'one is worth more than the other'.
# For example, think of the categorical variable animals_vector
# with the categories "Elephant", "Giraffe", "Donkey" and "Horse".
# Here, it is impossible to say that one stands above or below the other.
# In contrast, ordinal variables do have a natural ordering.
# Consider for example the categorical variable temperature_vector
# with the categories: "Low", "Medium" and "High".
# Here it is obvious that "Medium" stands above "Low", and "High" stands above "Medium".

# create the vector animals_vector
animals_vector <- c("Elephant", "Giraffe", "Donkey", "Horse")
animals_vector
class(animals_vector)
factor_animals_vector <- factor(animals_vector)
factor_animals_vector
class(factor_animals_vector)
levels(factor_animals_vector)

# create the vector temperature_vector
temperature_vector <- c("High", "Low", "High","Low", "Medium")
temperature_vector
class(temperature_vector) # character type
levels(temperature_vector) # does not identify any levels or categories for a character vector
factor_temperature_vector <- factor(temperature_vector,
                                    order = TRUE, # vector woth ordered levels
                                    levels = c("Low", "Medium", "High")) # the ordering
# check out the Environment window for factor_temperature_vector
factor_temperature_vector
class(factor_temperature_vector) # returns "ordered" "factor" 
levels(factor_temperature_vector)

survey_vector <- c("M", "F", "F", "M", "M")
survey_vector
factor_survey_vector <- factor(survey_vector)
factor_survey_vector # the levels are F M

levels(factor_survey_vector) <- c("Female", "Male") # assigns new names for the old levels
factor_survey_vector

summary(survey_vector)   # returns a description of the character vector
summary(factor_survey_vector) # like a frequency distribution 

# Male
male <- factor_survey_vector[1]
male
class(male)

# Female
female <- factor_survey_vector[2]
female
class(female)
table(female)

# What does R think about the variables:  Male 'larger' than Female?
male > female
#Since "Male" and "Female" are unordered (or nominal) factor levels,
# R returns a warning message, telling you that the greater than operator is not meaningful.
# As seen before, R attaches an equal value to the levels for such factors.

# Sometimes you will also deal with factors that do have a natural ordering between its categories.
# If this is the case, we have to make sure that we pass this information to R.
# Let us say that you are leading a research team of five data analysts
# and that you want to evaluate their performance.
# To do this, you track their speed, evaluate each 

# Analyst 1 is fast,
# Analyst 2 is slow,
# Analyst 3 is slow,
# Analyst 4 is fast and
# Analyst 5 is very fast

speed_vector <- c("fast", "slow", "slow", "fast", "very fast")
speed_vector
class(speed_vector)
factor_speed_vector <- factor(speed_vector,
                              ordered = TRUE,
                              levels = c("slow", "fast", "very fast"))
factor_speed_vector
summary(factor_speed_vector)

da2<-factor_speed_vector[2] # the 2nd element of the vector
da2
da5<-factor_speed_vector[5] # the 5th element of the vector
da5
factor_speed_vector
da2>da5 # compare two levels


# DATA FRAME
# You may remember that all the elements that you put in a matrix should be of the same type.
# Back then. When doing a market research survey, however, you often have questions such as:
#   
# 'Are your married?' or 'yes/no' questions (logical)
# 'How old are you?' (numeric)
# 'What is your opinion on this product?' or other 'open-ended' questions (character)
# ...
# 
# The output, namely the respondents' answers to the questions formulated above, 
# is a data set of different data types.
# You will often find yourself working with data sets that contain
# different data types instead of only one.
# 
# A data frame has the variables of a data set as columns and the observations as rows.
# This will be a familiar concept for those coming
# from different statistical software packages such as SAS or SPSS.
# Another method that is often used to get a rapid overview of your data
# is the function str().
# The function str() shows you the structure of your data set.
# For a data frame it tells you:
#   
# The total number of observations (e.g. 32 car types)
# The total number of variables (e.g. 11 car features)
# A full list of the variables names (e.g. mpg, cyl ... )
# The data type of each variable (e.g. num)
# The first observations
# 
# Applying the str() function will often be the first thing
# that you do when receiving a new data set or data frame.
# It is a great way to get more insight in your data set
# before diving into the real analysis.
data("mtcars")
mtcars
?mtcars
View(mtcars)
class(mtcars) # dataframe
warnings()
str(mtcars)    # structure of the data set
names(mtcars)  # returns the names of the variables
head(mtcars)   # first 6 cases
head(mtcars, 4) # first 4 cases
tail(mtcars)   # last 6 cases
# PRACTICE WITH dataset iris

mpg # you get an error, this means you have to ask for that variable in a different manner
mtcars$mpg # first write the name of the data frame, then type $ sign, and then the name of the variable
mean(mtcars$mpg) # media de selectie sau sample mean sau media aritmetica
var(mtcars$mpg) # dispersia de selectie sau sample variance
sd(mtcars$mpg) # sample standard deviation sau abaterea standard de selectie

# there are packages with specific functions returning descriptive statistics
# such a package is called "psych"
install.packages("psych", dependencies = TRUE)
library(psych)
describe(mtcars$mpg) # returns descriptive statistics for a particular variable
describe(mtcars) # # returns descriptive statistics for all variables in the dataframe

# you can CREATE DATA FRAME
# Definition of vectors
planet_name <- c("Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune")
type <- c("Terrestrial planet", "Terrestrial planet", "Terrestrial planet", 
          "Terrestrial planet", "Gas giant", "Gas giant", "Gas giant", "Gas giant")
diameter <- c(0.382, 0.949, 1, 0.532, 11.209, 9.449, 4.007, 3.883)
rotation <- c(58.64, -243.02, 1, 1.03, 0.41, 0.43, -0.72, 0.67)
rings <- c(FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE)
# Create a data frame from the vectors ussing a specific function
planets_df <- data.frame(planet_name, type, diameter, rotation, rings)
planets_df
str(planets_df)
planets_df[1,3] # Prints out diameter of Mercury (row 1, column 3)
planets_df[4, ] # Prints out data for Mars (entire fourth row)
planets_df[ ,5] # Prints out data for rings (entire fifth column)
planets_df[1:5,3] # Prints put data for the first 5 lines and the 3rd column
rings
planets_df$rings 
attach(planets_df)
rings
planets_df[rings, "name"] #logical?, nu inteleg ce reprezinta
planets_df[rings, ]

subset(planets_df, subset = diameter<1) # subsetting data using the dataset

# find the positions of the diameter values when ordering increasingly the values
?order
positions_increasing <- order(planets_df$diameter, decreasing = FALSE)
positions_increasing
planets_df[positions_increasing, ] # sort dataset by increasing positions according to diameter 

# find the positions of the diameter values when ordering decreasingly the values
positions_decreasing <- order(planets_df$diameter, decreasing = TRUE)
positions_decreasing
planets_df[positions_decreasing, ] # sort dataset by decreasing positions according to diameter 




# To recap: 
#   
# Vectors (one dimensional array): can hold numeric, character or logical values.
#      The elements in a vector all have the same data type.
# Matrices (two dimensional array): can hold numeric, character or logical values.
#      The elements in a matrix all have the same data type.
# Data frames (two-dimensional objects): can hold numeric, character or logical values.
#      Within a column all elements have the same data type,
#      but different columns can be of different data type.


#LISTS 
# A list in R is similar to your to-do list at work or school:
#   the different items on that list most likely differ in length, characteristic,
#   type of activity that has to do be done, ...
# 
# A list in R allows you to gather a variety of objects under one name
# (that is, the name of the list) in an ordered way.
# These objects can be matrices, vectors, data frames, even other lists, etc.
# It is not even required that these objects are related to each other in any way.
# 
# You could say that a list is some kind super data type:
#   you can store practically any piece of information in it!

# Vector with numerics from 1 up to 10
my_vector <- 1:10 

# Matrix with numerics from 1 up to 9
my_matrix <- matrix(1:9, ncol = 3)

# First 10 elements (lines) of the built-in data frame mtcars
my_df <- mtcars[1:10,]

# Construct list with these different elements:
my_list <- list(my_vector, my_matrix, my_df)
my_list
names(my_list) <- c("vec", "mat", "df")
my_list


# I. Importing data from csv and excel files in R
# Do not use regular EXCEL FILES to upload the data in R, for reasons follow the link
browseURL("http://cran.r-project.org/doc/manuals/R-data.html#Reading-Excel-spreadsheets")

# Use CSV (comma separated values) files, they are universal and easiest way to read data

# OPTION 1
data <- read.csv(file.choose(), # you can choose the csv file you want to load in the Environment
                 header=TRUE)   # the names of the variables are taken from the first row in the csv file
head(data)
str(data)
View(data)
rm(list = ls())

# OPTION 2
# the same function as above, but the first argument of it is the path to the csv file in your computer
exemplu_econometrie <- read.csv("path/fil",
                                header = TRUE)
exemplu_econometrie
str(exemplu_econometrie)

# OPTION 3
# Main menu File, Import Dataset, then select the type of the file containing the dataset, .....
