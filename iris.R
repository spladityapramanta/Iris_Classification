# Iris flower classification project
# by: SPL Aditya Pramanta
# created on: November 13, 2017

#########################
# DATA PREPARATION PART #
#########################

# load data
iris <- read.csv("bezdekIris.csv")

# create new column name that represent its meaning
names(iris)[1] <- "sepal_length"
names(iris)[2] <- "sepal_width"
names(iris)[3] <- "petal_length"
names(iris)[4] <- "petal_width"
names(iris)[5] <- "flower_class"

# check whether there is NA in data
sepal_length_NA <- which(is.na(iris$sepal_length))
sepal_width_NA <- which(is.na(iris$sepal_width))
petal_length_NA <- which(is.na(iris$petal_length))
petal_width_NA <- which(is.na(iris$petal_width))
# fortunately there is no NA data, so no need to do further NA handling

# create summary and plot of class to know the class distribution
round(prop.table(table(iris$class))*100,1) # the result is the classes are distributed equally (~33% each) so using accuracy as the model evaluator is enough
# or you can use :
summary(iris[5]) # result : 49, 50, 50

#######################
# CLASSIFICATION PART #
#######################

# sampling data as train and test
library(boot)
set.seed(100)
rand.vec <- sample(1:149,119)
iris.train <- iris[rand.vec, ] 
iris.test <- iris[-rand.vec, ]

###########################################
# Multiple logistic regression
iris.logistic <- glm(flower_class~., data = iris.train, family = binomial())

###########################################

###########################################
# SVM
library(e1071)
iris.svm <- svm(flower_class~., data = iris.train)

# check the model result
print(iris.svm) 
summary(iris.svm) # same as above

# testing the model
iris.pred_svm <- predict(iris.svm, iris.test[,-5])
table(iris.pred_svm, iris.test$flower_class)

# acc = 31/31 = 100%
###########################################


###########################################
# KNN
library(class)

# We use k=12, ~sqrt(149) as its also an odd number we eliminate the prob. of getting a tie
iris.pred_kNN <- knn(train = iris.train[,-5], test=iris.test[,-5], cl=iris.train[,5], k=12)

# evaluate the confusion matrix to get the accuracy
library(gmodels)
CrossTable(x = iris.test$flower_class, y = iris.pred_kNN, prop.chisq = F, prop.c = F)

# Acc = 30/31 = 96%
###########################################

###########################################
# Decision Tree
library(C50)

iris.model_C50 <- C5.0(iris.train[,-5], iris.train$flower_class)

# summary of the model
summary(iris.model_C50)
# it said that, the error = 2.5%, thus acc = 97.5%

#plotting the tree
plot(iris.model_C50)

# predict the result of test data
iris.pred_DT <- predict(iris.model_C50, iris.test[,-5])

# confusion matrix
library("gmodels")
CrossTable(iris.pred_DT, iris.test$flower_class, prop.chisq = F, prop.c = F)

# Acc = 31/31 = 100%
###########################################

###########################################
# Random forest
library(caret)
forestcontrol <- trainControl(method="repeatedcv", number=10, repeats = 10)
iris.forest <- train(iris.train[,-5], iris.train[,5], method="rf", trControl=forestcontrol)
iris.pred_forest <- predict(iris.forest, iris.test[,-5])

# confusion matrix
library("gmodels")
CrossTable(iris.pred_forest, iris.test$flower_class, prop.chisq = F, prop.c = F)

# Acc = 31/31 = 100%

# another random forest function
iris.forest2 <- randomForest(flower_class~.,data=iris.train)
iris.pred_forest2 <- predict(iris.forest2, iris.test[,-5])

# confusion matrix
library("gmodels")
CrossTable(iris.pred_forest, iris.test$flower_class, prop.chisq = F, prop.c = F)

# Acc = 31/31 = 100%
##########################################

##########################################
# using Xboost

