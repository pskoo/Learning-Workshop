#Packages that are used
library(plyr)
library(dplyr)
library(ggplot2)
library(caret)
library(rattle)

#setup parallel processing
library(doSNOW)
library(parallel)
coreNumber <- max(detectCores()-1,1)
cl <- makeCluster(coreNumber, type = "SOCK")
registerDoSNOW(cl)

#set working directory
setwd("d:/data")

#Setting the random seed
set.seed(768)

#Preparation of Data
census <- read.csv("./adult.csv",header=FALSE,strip.white=TRUE,sep=",")

names(census) <- c("age","workclass","fnlwgt","education",
                   "education_num", "marital_status", "occupation", "relationship", "race", "sex",
                   "capital_gain","capital_loss","hrs_per_week", 
                   "native_country","target")
#setup for calculation of correlation to understand relationships
census[census$target==">50K","targetB"] <- 1
census[census$target=="<=50K","targetB"] <- 0

#Because of the sign in front, the logistic regression does not work
census[census$target==">50K","targetF"] <- "GT50"
census[census$target=="<=50K","targetF"] <- "LE50"
census$targetF <- as.factor(census$targetF)

#Data Partition
train_c <- createDataPartition (census$target, p=.8,list=FALSE,times=1)  
censusTrain <- census[train_c,] 
censusValid <- census[-train_c,]

#Modeling Starts here
#Logistic regression
myControl <- trainControl(method='repeatedcv', number =10, repeats=3)
census_log <- train(targetF~race+sex+hrs_per_week+education_num+age,data=censusTrain,method='glm', family=binomial(link="logit"),trControl=myControl)
censuslogvalid <- predict(census_log,censusValid,type="prob")
censuslogvalid$pred <- predict(census_log,censusValid)
confusionMatrix(censuslogvalid$pred,censusValid$targetF)
varImp(census_log)

#Decision Trees
census_tree <- train(targetF~race+sex+hrs_per_week+education_num+age,data=censusTrain,method='rpart',trControl=myControl)
fancyRpartPlot(census_tree$finalModel)
predtreeValid <- predict(census_tree, censusValid)
confusionMatrix(predtreeValid,censusValid$targetF) 


census_gbm <- train(targetF~race+sex+hrs_per_week+education_num+age,data=censusTrain,method='gbm',trControl=myControl)
predgbmValid <- predict(census_gbm, censusValid)
confusionMatrix(predgbmValid,censusValid$targetF)

#Random_forest
census_rf <- train(targetF~race+sex+hrs_per_week+education_num+age,data=censusTrain,method='rf',trControl=myControl)
predrfValid <- predict(census_rf, censusValid)
confusionMatrix(predrfValid,censusValid$targetF)

stopCluster(cl)
