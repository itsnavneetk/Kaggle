
#Work space for titanic kaggle prediction competition
#https://www.kaggle.com/c/titanic-gettingStarted
library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)
library(caTools)
library(gbm)


#Read in the data
train <- read.csv("titanic_train.csv")
test <- read.csv("titanic_test.csv")

train$Survived <- as.factor(train$Survived)

#Create new columns for ship section (first letter listed under cabin)
#Most passengers don't have a cabin listed, but those that do tend to
#Have higher survival rates
train_ship_section <- substr(train$Cabin, 1, 1)
test_ship_section <- substr(test$Cabin, 1, 1)

train$section <- as.factor(train_ship_section)
test$section <- as.factor(test_ship_section)

#Split data into cases where age is listed and were age is NA
train_age <- subset(train, is.na(Age)==FALSE)
train_no_age <- subset(train, is.na(Age)==TRUE)

test_age <- subset(test, is.na(Age)==FALSE & is.na(Fare)==FALSE)
test_no_age <- subset(test, is.na(Age)==TRUE | is.na(Fare)==TRUE)


#Basic random forest on entries with and without Age data--------------------------
set.seed(121)
basic_rf <- randomForest(Survived~ Pclass+Sex+Age+SibSp+Parch+Fare+Embarked+section,data=train_age,ntree=15, nodesize= 4)
rf_pred <- predict(basic_rf, newdata=test_age)

basic_rf2 <- randomForest(Survived~ Pclass+Sex+SibSp+Parch+Fare+Embarked+section,data=train_no_age,ntree=15, nodesize= 4)
rf_pred2 <- predict(basic_rf2, newdata=test_no_age)

age_pred_frame <- data.frame(PassengerId= test_age$PassengerId , Survived=rf_pred)

no_pred_frame <- data.frame(PassengerId= test_no_age$PassengerId , Survived=rf_pred2)

submission5 <- rbind(age_pred_frame , no_pred_frame)
  
write.csv(submission5, "titanic_submission5.csv", row.names=FALSE)

#Score = 0.77033 (Not good)--------------------------------------------------------

#Trying CART again-----------------------------------------------------------------

basic_cart = rpart(Survived~ Pclass+Sex+Age+SibSp+Parch+Fare+Embarked+section,data=train, cp=0.009,minbucket=5,method="class")
cart_pred = predict(basic_cart, newdata=test, type="response")

prp(basic_cart)

pred_frame = data.frame(PassengerId= test$PassengerId , Survived=cart_pred)

submission7 = pred_frame

write.csv(submission7, "titanic_submission7.csv", row.names=FALSE)

#Score = 0.77512 (CART and RF seem to do poorly and tend to overfit)--------------

#--Basic Logistic Regression--------------------------------------------------------
basic_log = glm(Survived~ Pclass+Sex+Age+SibSp+Parch+Fare+Embarked+section,data=train_age,family="binomial")
log_pred = predict(basic_log, newdata=test_age,type="response")

basic_log2 = glm(Survived~ Pclass+Sex+SibSp+Parch+Fare+Embarked+section,data=train_no_age,family="binomial")
log_pred2 = predict(basic_log2, newdata=test_no_age,type="response")


age_pred_frame = data.frame(PassengerId= test_age$PassengerId , Survived=as.numeric(log_pred>=0.5))

no_pred_frame = data.frame(PassengerId= test_no_age$PassengerId , Survived=as.numeric(log_pred2>=0.5))

submission8 = rbind(age_pred_frame , no_pred_frame)

write.csv(submission8, "titanic_submission8.csv", row.names=FALSE)
#Score = 0.74163

#---Polynomial Logistic regression---------------------------------------------
basic_log = glm(Survived~ Pclass+Sex+poly(Age, 3, raw=TRUE)+poly(SibSp, 2, raw=TRUE)+Fare+Embarked+section,data=train_age,family="binomial")

log_pred = predict(basic_log, newdata=test_age,type="response")

basic_log2 = glm(Survived~ Pclass+Sex+SibSp+Fare+Embarked+section,data=train_age,family="binomial")
log_pred2 = predict(basic_log2, newdata=test_no_age,type="response")


age_pred_frame = data.frame(PassengerId= test_age$PassengerId , Survived=as.numeric(log_pred>=0.5))

no_pred_frame = data.frame(PassengerId= test_no_age$PassengerId , Survived=as.numeric(log_pred2>=0.5))

submission9 = rbind(age_pred_frame , no_pred_frame)

write.csv(submission9, "titanic_submission9.csv", row.names=FALSE)

#---Score = 0.76555-------------------------------------------------

#Misc plotting workspace-----------------
females = subset(train, Sex =="female")
females$Pclass = as.numeric(females$Pclass)
ggplot(aes(x=Age, y=section), data = subset(females, Pclass==3 & Age>19& SibSp  == 0 & Embarked=="S" & Fare < 15)) + geom_jitter(aes(color=Survived))

deadfem = subset(females, Survived == "0")


table(subset(females, Pclass==3 & Age>19 & SibSp  == 0 )$SibSp,subset(females, Pclass==3 & Age>19 & SibSp  == 0 )$Survived,subset(females, Pclass==3 & Age>19& SibSp  == 0 )$Embarked)
#-------------------------------


#----More CART models with varying levels of complexity------------

basic_cart = rpart(Survived~ Pclass+Sex+Age+SibSp+Parch+Fare+Embarked, data=train, cp=0.008,minbucket=4, method="class")

cart_pred = predict(basic_cart, newdata=test, type="class")

prp(basic_cart)

pred_frame = data.frame(PassengerId= test$PassengerId , Survived=cart_pred)

submission13 = pred_frame
summary(submission13)

write.csv(submission13, "titanic_submission13.csv", row.names=FALSE)

#After several different trees of varying comlexity---Score = 0.78469


#----Try random forests with caret package-----------------------------------

set.seed(121)

rfcontrol = trainControl(method = "repeatedcv",
                         number = 10,
                         repeats = 3
                         )

rfgrid = expand.grid(n.trees=c(80,100,150,200,300),interaction.depth=c(2,3,4,5,6),shrinkage=c(0.08,0.06,0.04))

basic_rf = train(Survived~ Pclass+Sex+Age+SibSp+Fare+Embarked,data=train_age, method='gbm',tuneGrid=rfgrid,trControl = rfcontrol)

rf_pred = predict(basic_rf, newdata=test_age)

basic_rf2 = train(Survived~ Pclass+Sex+SibSp+Fare+Embarked,data=train_no_age, method='gbm',tuneGrid=rfgrid, trControl = rfcontrol)

rf_pred2 = predict(basic_rf2, newdata=test_no_age)


age_pred_frame = data.frame(PassengerId= test_age$PassengerId , Survived=rf_pred)
no_pred_frame = data.frame(PassengerId= test_no_age$PassengerId , Survived=rf_pred2)

submission15 = rbind(age_pred_frame , no_pred_frame)

basic_rf
basic_rf2
summary(submission15 )


write.csv(submission15, "titanic_submission15.csv", row.names=FALSE)
#Score: 0.76555 ... it seems simple models work best------------------


#Simplistic CART------------------------------------------------------
basic_cart = rpart(Survived~ Pclass+Sex+Age+SibSp+Fare+Parch+Embarked+section, data=train, cp=0.02,minbucket=3, method="class")

cart_pred = predict(basic_cart, newdata=test, type="class")

prp(basic_cart)

pred_frame = data.frame(PassengerId= test$PassengerId , Survived=cart_pred)

submission16 = pred_frame
summary(submission16)

write.csv(submission16, "titanic_submission16.csv", row.names=FALSE)
#Tied best model.... Score: 0.78469------------------------------------

#Another CART, higher comlexity and min bucket------------------------------------------------------

basic_cart = rpart(Survived~ Pclass+Sex+Age+SibSp+Fare+Parch+Embarked+section, data=train, cp=0.00000001,minbucket=12, method="class")

cart_pred = predict(basic_cart, newdata=test, type="class")

prp(basic_cart)

pred_frame = data.frame(PassengerId= test$PassengerId , Survived=cart_pred)

submission17 = pred_frame
summary(submission17)
write.csv(submission17, "titanic_submission17.csv", row.names=FALSE)
#Score: 0.79426 Best so far




basic_cart = rpart(Survived~ Pclass+Sex+Embarked+Age+section+SibSp, data=train, cp=0.00001,minbucket=14, method="class")

cart_pred = predict(basic_cart, newdata=test, type="class")

prp(basic_cart)

pred_frame = data.frame(PassengerId= test$PassengerId , Survived=cart_pred)

submission18 = pred_frame
summary(submission18)
write.csv(submission18, "titanic_submission18.csv", row.names=FALSE)
#Score: 0.77990



basic_cart = rpart(Survived~ Pclass+Sex+Age+SibSp+Fare+Parch+Embarked+section, data=train, cp=0.009,minbucket=6, method="class")

cart_pred = predict(basic_cart, newdata=test, type="class")

prp(basic_cart)

pred_frame = data.frame(PassengerId= test$PassengerId , Survived=cart_pred)

submission19 = pred_frame
summary(submission19)
write.csv(submission19, "titanic_submission19.csv", row.names=FALSE)