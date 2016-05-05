#West Nile Virus Kaggle competition 
#https://www.kaggle.com/c/predict-west-nile-virus

library(ggplot2)
library(rpart)
library(caret)
library(randomForest)
library(e1071)
library(glmnet)
library(gbm)
options( java.parameters = "-Xmx8g" )
library(extraTrees)
library(nnet)
library(xgboost)
library(tm)
library(pROC)
library(SnowballC)
library(MASS)
library(NLP)
library(RWeka)
library(plyr)
library(dplyr)
library(data.table)

#Loading data
train = fread("train.csv", stringsAsFactors=FALSE)
test = fread("test.csv", stringsAsFactors=FALSE)

weather = fread("weather.csv", stringsAsFactors=FALSE)

#Separate weather data by station:
#O'Hare, Lat: 41.995 Lon: -87.933
#Midway Lat: 41.786 Lon: -87.752

#Remove some weather variables and conver to numeric
weather = weather[c("Station" ,"Date","Tmax", "Tmin", "DewPoint" ,"WetBulb",  "Heat", "Cool", "PrecipTotal" ,"StnPressure", "SeaLevel","ResultSpeed", "ResultDir" , "AvgSpeed" ) ]

weather[,6:14] = data.matrix(weather[,6:14], rownames.force = NA)
weather[is.na(weather)] <- 0

#Calculate closest weather station and merge weather data into train
train$Station = ifelse( sqrt((train$Latitude-41.995)^2+(train$Longitude+87.933)^2 ) < sqrt((train$Latitude-41.786)^2+(train$Longitude+87.752)^2 ) , 1, 2)

train = merge(train, weather, by.y=c("Date","Station"))

test$Station = ifelse( sqrt((test$Latitude-41.995)^2+(test$Longitude+87.933)^2 ) < sqrt((test$Latitude-41.786)^2+(test$Longitude+87.752)^2 ) , 1, 2)

test = merge(test, weather, by.y=c("Date","Station"))

#It appears there is no spray data for test years so I'm not sure if this data will be useful
spray = read.csv("spray.csv", stringsAsFactors=FALSE)

# #Investigate trap locations vs spray locations...
# spray2011 = subset(spray, Date<2012)
# spray2013 = subset(spray, Date>2012)
# #2011 spray locations
# ggplot(data=train, aes(x=Longitude, y=Latitude)) + geom_point() +geom_point(data= spray2011, aes(x=Longitude, y=Latitude, color="red"), alpha=0.1) + xlim(c(-87.85,-87.5)) + ylim(c(41.6,42.05))
# #2013 spray locations
# ggplot(data=train, aes(x=Longitude, y=Latitude)) + geom_point() +geom_point(data= spray2013, aes(x=Longitude, y=Latitude, color="red"), alpha=0.1) + xlim(c(-87.85,-87.5)) + ylim(c(41.6,42.05))
# #Spraying locations were changed and expanded by 2013. Without test data for spraying it would be difficult to use spray data in a model. Could create a variable for trap proximity to closest spray location in any year.

#Create new features based on distance from various map locations
# train$spray_cluster2013_1 = sqrt((train$Latitude-42)^2+(train$Longitude+87.8)^2)
# 
# test$spray_cluster2013_1 = sqrt((test$Latitude-42)^2+(test$Longitude+87.8)^2)
# 
# train$spray_cluster2013_2 = sqrt((train$Latitude-41.98)^2+(train$Longitude+87.68)^2)
# 
# test$spray_cluster2013_2 = sqrt((test$Latitude-41.98)^2+(test$Longitude+87.68)^2)
# 
# train$spray_cluster2013_3 = sqrt((train$Latitude-41.945)^2+(train$Longitude+87.72)^2)
# 
# test$spray_cluster2013_3 = sqrt((test$Latitude-41.945)^2+(test$Longitude+87.72)^2)
# 
# train$spray_cluster2013_4 = sqrt((train$Latitude-41.925)^2+(train$Longitude+87.775)^2)
# 
# test$spray_cluster2013_4 = sqrt((test$Latitude-41.925)^2+(test$Longitude+87.775)^2)
# 
# train$spray_cluster2013_5 = sqrt((train$Latitude-41.88)^2+(train$Longitude+87.72)^2)
# 
# test$spray_cluster2013_5 = sqrt((test$Latitude-41.88)^2+(test$Longitude+87.72)^2)

train$spray_cluster2013_6 = sqrt((train$Latitude-41.725)^2+(train$Longitude+87.65)^2)

test$spray_cluster2013_6 = sqrt((test$Latitude-41.725)^2+(test$Longitude+87.65)^2)

# train$spray_cluster2013_7 = sqrt((train$Latitude-41.775)^2+(train$Longitude+87.725)^2)
# 
# test$spray_cluster2013_7 = sqrt((test$Latitude-41.775)^2+(test$Longitude+87.725)^2)

train$spray_cluster2011 = sqrt((train$Latitude-41.98)^2+(train$Longitude+87.81)^2)

test$spray_cluster2011 = sqrt((test$Latitude-41.98)^2+(test$Longitude+87.81)^2)

train$city_center_dist = sqrt((train$Latitude-41.85)^2+(train$Longitude+87.6)^2)

test$city_center_dist = sqrt((test$Latitude-41.85)^2+(test$Longitude+87.6)^2)

sampleSubmission = read.csv("sampleSubmission.csv", stringsAsFactors=FALSE)

#Preparing data for modeling------------------
test_id = test$Id
test$Id = NULL

targets = train$WnvPresent
targets = as.factor(targets)
levels(targets) = c("no_virus","virus")
train$WnvPresent = NULL

train_Date = strptime(train$Date, "%Y-%m-%d")
test_Date = strptime(test$Date, "%Y-%m-%d")

train_Date2 = strptime(train$Date, "%Y-%m-%d")
test_Date2 = strptime(test$Date, "%Y-%m-%d")

train_Date3 = strptime(train$Date, "%Y-%m-%d")
test_Date3 = strptime(test$Date, "%Y-%m-%d")


train$year = train_Date$year
#Subract 1 from test for models that require values to match
test$year = test_Date$year-1

train$month = train_Date2$mon
test$month = test_Date2$mon

train$day = train_Date3$wday
test$day = test_Date3$wday

train$Date = NULL
test$Date = NULL

#Since training data contains block, lat/long and a trap number, street/address data seems redundant.
train$Address = NULL
test$Address = NULL
train$Street = NULL
test$Street  = NULL
train$Street = NULL
test$Street  = NULL
train$AddressNumberAndStreet = NULL
test$AddressNumberAndStreet  = NULL

#Accuracy of lat/long measurements doesn't seem to add anything other than noise. Remove it.
train$AddressAccuracy = NULL
test$AddressAccuracy  = NULL

#Turn species variable into 8 level factor
train$Species = as.factor(train$Species)
test$Species = as.factor(test$Species)
levels(test$Species) = c(1,2,3,4,5,6,7,8 )
train$Species = as.numeric(train$Species)
test$Species = as.numeric(test$Species)

levels(train$Species) = levels(test$Species)

#Train has feature Num Mosquitos. Test does not. Remove for initial models.

num_mosquitos = train$NumMosquitos
train$NumMosquitos = NULL

#Convert Trap numbers into numeric for now. Ideally traps should be factor/inidactor variables but there are so many of them it would result in a ton of factor levels/indiactor vars.
train$Trap = as.numeric(substring(train$Trap, 2))
train$Trap = ifelse(is.na(train$Trap),0,train$Trap)

test$Trap = as.numeric(substring(test$Trap, 2))
test$Trap = ifelse(is.na(test$Trap),0,test$Trap)


#Data Preprocessing-------------------------------------

# #Center and scale the data
preped_data = preProcess(train, method=c("center", "scale","BoxCox"))
train = predict(preped_data, train)
test = predict(preped_data, test)

# valid = predict(preped_data, valid[2:94])
# #Remove near zero variance predictors
# near_zero = nearZeroVar(train)
# train = train[-near_zero]
# valid = valid[-near_zero]
# #Remove highly correlated features
# correlations <- cor(train)
# highcor <- findCorrelation(correlations, cutoff= 0.75)
# train = train[-highcor]
# valid = valid[-highcor]



#Validation splitting----------------
set.seed(12)

train_index = createDataPartition(targets, p = .75,
                                  list = FALSE,
                                  times = 1)


train_valid = train[ train_index,]
valid = train[ -train_index,]

train_valid_targets= targets[ train_index]
valid_targets= targets[ -train_index]

train_valid_num = num_mosquitos[ train_index]
valid_num = num_mosquitos[-train_index]
#Validation splitting--------------

#Preparing data for modeling------------------





#Modeling Starts here
#---------------------------------------------------------------
#---------------------------------------------------------------
#Modeling Starts here



#Logistic Regression validation code
control = trainControl(method = "repeatedcv",
                       number = 5,
                       verboseIter = TRUE,
                       repeats = 2,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

set.seed(12)
glm_model_caret = train(train_valid_targets~., data=train_valid, method="glm", trControl=control, metric="ROC")

glm_valid_pred = predict(glm_model_caret, newdata=valid, type="prob")

rocCurve = roc(response=valid_targets, predictor=glm_valid_pred$virus)
lm_auc = auc(rocCurve)
lm_auc
#Validation best AUC: 0.7361 (with 12 weather vars)
#Validation best AUC: 0.7409 (with 12 weather vars and 3 distance vars)


#Logistic Regression submission code
control = trainControl(method = "none",
                       number = 5,
                       verboseIter = TRUE,
                       repeats = 2,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

set.seed(12)
glm_model_caret = train(targets~., data=train, method="glm", trControl=control, metric="ROC")

glm_test_pred = predict(glm_model_caret, newdata=test, type="prob")

#Inital baseline submission.
submission1 = data.frame(Id=test_id,WnvPresent=glm_test_pred$virus)
write.csv(submission1, "wnv_sub1_glm.csv", row.names=FALSE)
#LB AUC: 0.66582

#Submission after imputing num_mosquitos for test set using RF and using degree 2 feature interactions for the logistic regression
submission3 = data.frame(Id=test_id, WnvPresent=glm_test_pred$virus)
write.csv(submission3, "wnv_sub3_glm.csv", row.names=FALSE)
#LB AUC: 0.63970... Imputed num_mosquitos likely too noisy

#Degree 2 logisitic without num_mosquitos
submission4 = data.frame(Id=test_id, WnvPresent=glm_test_pred$virus)
write.csv(submission4, "wnv_sub4glm.csv", row.names=FALSE)
#LB AUC: 0.62466

#Degree 1 logistic with station and 12 numeric weather vars added
submission6 = data.frame(Id=test_id,WnvPresent=glm_test_pred$virus)
write.csv(submission6, "wnv_sub6_glm.csv", row.names=FALSE)
#LB AUC: 0.69895  Adding weather vars was significant improvement




#Random Forest Baseline
control = trainControl(method = "repeatedcv",
                       number = 3,
                       verboseIter = TRUE,
                       repeats = 1,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

grid = expand.grid(  mtry= c(3))

set.seed(12)
rf_model_caret = train(train_valid_targets~., data=train_valid, method="rf", trControl=control, tuneGrid = grid, metric="ROC", ntree=1000, nodesize=10)

rf_valid_pred = predict(rf_model_caret, newdata=valid, type="prob")

rocCurve = roc(response=valid_targets, predictor=rf_valid_pred$virus)
varImp(rf_model_caret)
rf_auc = auc(rocCurve)
rf_auc
#Validation AUC: 0.7542 with ntree=1000, nodesize=10, and 3X1 CV reps.
#Validation AUC: 0.741 with ntree=1000, nodesize=10, and 3X1 CV reps. 12 weather vars, 3 distance vars

#RF Submission code
control = trainControl(method = "none",
                       number = 3,
                       verboseIter = TRUE,
                       repeats = 1,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

grid = expand.grid(  mtry= c(3))

set.seed(12)
rf_model_caret = train(targets~., data=train, method="rf", trControl=control, tuneGrid = grid, metric="ROC", ntree=1000, nodesize=10)

rf_test_pred = predict(rf_model_caret, newdata=test, type="prob")

#Inital rf baseline submission.
submission2 = data.frame(Id=test_id,WnvPresent=rf_test_pred$virus)
write.csv(submission2, "wnv_sub2_rf.csv", row.names=FALSE)
#LB AUC: 0.60

#second rf baseline submission.
submission5 = data.frame(Id=test_id,WnvPresent=rf_test_pred$virus)
write.csv(submission5, "wnv_sub5_rf.csv", row.names=FALSE)
#LB AUC: 0.60581... 

#I need to extrat new features from spray and weather data sets

#rf with station and 12 added weather features. mtry=3
submission7 = data.frame(Id=test_id,WnvPresent=rf_test_pred$virus)
write.csv(submission7, "wnv_sub7_rf.csv", row.names=FALSE)
#LB AUC: 0.67975  Not Good but a huge improvement over other RF attempts.



#GLM NET model Validation
control = trainControl(method = "none",
                       number = 5,
                       verboseIter = TRUE,
                       repeats = 1,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

grid = expand.grid(  alpha= c(0.5), lambda =c(0.0003))

glm_model_caret = train(train_valid_targets~., data=train_valid, method="glmnet", trControl=control, tuneGrid = grid, metric="ROC")

glmnet_valid_pred = predict(glm_model_caret, newdata=valid, type="prob")

rocCurve = roc(response=valid_targets, predictor=glmnet_valid_pred$virus )
glm_net = auc(rocCurve)
glm_net 
#Validation AUC: 0.7398 alpha= c(0.4), lambda =c(0.00035))
#Validation AUC: 0.811  alpha= c(0.025), lambda =c(0.1,0.0001)) with second order features... possible overfit
#Validation AUC: 0.745   alpha= c(0.5), lambda =c(0.0003)) 12 weather vars and 3 map distance vars



#GLM NET model submission code
control = trainControl(method = "none",
                       number = 5,
                       verboseIter = TRUE,
                       repeats = 1,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

grid = expand.grid(  alpha= c(0.5), lambda =c(0.0003))

glm_model_caret = train(targets~., data=train, method="glmnet", trControl=control, tuneGrid = grid, metric="ROC")

glmnet_test_pred = predict(glm_model_caret, newdata=test, type="prob")


#First glmnet sub. second order features with alpha= c(0.025), lambda =c(0.1,0.0001))
submission10 = data.frame(Id=test_id,WnvPresent=glmnet_test_pred$virus)
write.csv(submission10, "wnv_sub10_glmnet.csv", row.names=FALSE)
#LB AUC: 0.57391 Second order features doing some odd overfitting

#First order vars, 12 weather, 3 location, alpha= c(0.5), lambda =c(0.0003))
submission11 = data.frame(Id=test_id,WnvPresent=glmnet_test_pred$virus)
write.csv(submission11, "wnv_sub11_glmnet.csv", row.names=FALSE)
#LB AUC: 0.71062  #Best single model solution so far


#GBM Validation
set.seed(12)

control = trainControl(method = "repeatedcv",
                       number = 3,
                       verboseIter = TRUE,
                       repeats = 1,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

grid = expand.grid( n.trees= c(200), interaction.depth=c(12),shrinkage=c(0.025))


gbm_model_caret = train(train_valid_targets~., data=train_valid, method="gbm", trControl=control, tuneGrid = grid, metric="ROC")

gbm_valid_pred = predict(gbm_model_caret, newdata=valid, type="prob")

rocCurve = roc(response=valid_targets, predictor=gbm_valid_pred$virus )
gbm_auc = auc(rocCurve)
gbm_auc 
#Validation AUC: 0.8184 with 12 weather, 3 distance, c(200), interaction.depth=c(12),shrinkage=c(0.025))  3v1 repeated cv



#GBM Submission
set.seed(12)

control = trainControl(method = "repeatedcv",
                       number = 3,
                       verboseIter = TRUE,
                       repeats = 1,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

grid = expand.grid( n.trees= c(200), interaction.depth=c(12),shrinkage=c(0.025))

gbm_model_caret = train(targets~., data=train, method="gbm", trControl=control, tuneGrid = grid, metric="ROC")

gbm_test_pred = predict(gbm_model_caret, newdata=test, type="prob")

#GBM with 12 weather, 3 distance, c(200), interaction.depth=c(12),shrinkage=c(0.025))  3v1 repeated cv
submission12 = data.frame(Id=test_id,WnvPresent=gbm_test_pred$virus)
write.csv(submission12, "wnv_sub12_gbm.csv", row.names=FALSE)
#LB AUC: 0.67780 Complex models seem to overfit.



#Other caret models
#PLS Model Validation
set.seed(12)

control = trainControl(method = "repeatedcv",
                       number = 3,
                       verboseIter = TRUE,
                       repeats = 1,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

grid = expand.grid( ncomp= c(3))


model_caret = train(train_valid_targets~., data=train_valid, method="pls", trControl=control, tuneGrid = grid, metric="ROC")

valid_pred = predict(model_caret, newdata=valid, type="prob")

rocCurve = roc(response=valid_targets, predictor=valid_pred$virus )
auc = auc(rocCurve)
auc 
#Validation AUC: 0.7382


#RBF SVM Model Validation
set.seed(12)

control = trainControl(method = "none",
                       number = 3,
                       verboseIter = TRUE,
                       repeats = 1,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

grid = expand.grid( sigma= c(0.01), C=c(0.9))


model_caret = train(train_valid_targets~., data=train_valid, method="svmRadial", trControl=control, tuneGrid = grid, metric="ROC")

valid_pred = predict(model_caret, newdata=valid, type="prob")

rocCurve = roc(response=valid_targets, predictor=valid_pred$virus )
auc = auc(rocCurve)
auc 
#Validation AUC:  0.7511 sigma= c(0.01), C=c(0.9))

model_caret = train(targets~., data=train, method="svmRadial", trControl=control, tuneGrid = grid, metric="ROC")

svmRadial_test_pred = predict(model_caret, newdata=test, type="prob")

submission20 = data.frame(Id=test_id,WnvPresent=svmRadial_test_pred$virus)
write.csv(submission20, "wnv_sub20_svmRadial.csv", row.names=FALSE)
#LB AUC: 0.54774



#CART model model Validation
set.seed(12)

control = trainControl(method = "repeatedcv",
                       number = 3,
                       verboseIter = TRUE,
                       repeats = 1,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

grid = expand.grid( cp=c(0.0001))


model_caret = train(train_valid_targets~., data=train_valid, method="rpart", trControl=control, tuneGrid = grid, metric="ROC")

valid_pred = predict(model_caret, newdata=valid, type="prob")

rocCurve = roc(response=valid_targets, predictor=valid_pred$virus )
auc = auc(rocCurve)
auc 
#Validation AUC: 0.7547 cp=c(0.0001)

#CART Submission
model_caret = train(targets~., data=train, method="rpart", trControl=control, tuneGrid = grid, metric="ROC")

rpart_test_pred = predict(model_caret, newdata=test, type="prob")

submission13 = data.frame(Id=test_id,WnvPresent=rpart_test_pred$virus)
write.csv(submission13, "wnv_sub13_cart.csv", row.names=FALSE)
#LB AUC: 0.59201 terrible!



#Single layer nnet model Validation
set.seed(12)

control = trainControl(method = "none",
                       number = 5,
                       verboseIter = TRUE,
                       repeats = 1,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

grid = expand.grid( size=c(10), decay=c(0.3))


model_caret = train(train_valid_targets~., data=train_valid, method="nnet", trControl=control, tuneGrid = grid, metric="ROC", maxit = 100)

valid_pred = predict(model_caret, newdata=valid, type="prob")

rocCurve = roc(response=valid_targets, predictor=valid_pred$virus )
auc = auc(rocCurve)
auc 
#Validation AUC: 0.801 size=c(10), decay=c(0.3)

#NNet submission
model_caret = train(targets~., data=train, method="nnet", trControl=control, tuneGrid = grid, metric="ROC", maxit = 100)

nnet_test_pred = predict(model_caret, newdata=test, type="prob")

submission14 = data.frame(Id=test_id,WnvPresent=nnet_test_pred$virus)
write.csv(submission14, "wnv_sub14_nnet.csv", row.names=FALSE)
#LB AUC: 0.64850


#KNN Model testing
set.seed(12)

control = trainControl(method = "none",
                       number = 5,
                       verboseIter = TRUE,
                       repeats = 1,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

grid = expand.grid( k=c(25))


model_caret = train(train_valid_targets~., data=train_valid, method="knn", trControl=control, tuneGrid = grid, metric="ROC")

valid_pred = predict(model_caret, newdata=valid, type="prob")

rocCurve = roc(response=valid_targets, predictor=valid_pred$virus )
auc = auc(rocCurve)
auc
#Validation AUC: 0.7802 k=c(25))

model_caret = train(targets~., data=train, method="knn", trControl=control, tuneGrid = grid, metric="ROC")

knn_test_pred = predict(model_caret, newdata=test, type="prob")

submission15 = data.frame(Id=test_id,WnvPresent=knn_test_pred$virus)
write.csv(submission15, "wnv_sub15_knn.csv", row.names=FALSE)
#LB AUC: 0.69602 Not terrible for a KNN...



#Gam Splines Model validation
set.seed(12)

control = trainControl(method = "none",
                       number = 5,
                       verboseIter = TRUE,
                       repeats = 1,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

grid = expand.grid( df=c(1))

model_caret = train(train_valid_targets~., data=train_valid, method="gamSpline", trControl=control, tuneGrid = grid, metric="ROC")

valid_pred = predict(model_caret, newdata=valid, type="prob")

rocCurve = roc(response=valid_targets, predictor=valid_pred$virus )
auc = auc(rocCurve)
auc
#Validation AUC:0.8117 df=c(10))

model_caret = train(targets~., data=train, method="gamSpline", trControl=control, tuneGrid = grid, metric="ROC")

gam_test_pred = predict(model_caret, newdata=test, type="prob")

submission23 = data.frame(Id=test_id,WnvPresent=gam_test_pred$virus)
write.csv(submission23, "wnv_sub23_gam.csv", row.names=FALSE)
#LB AUC: 0.55831 (df 25)
#LB AUC: 0.71000 (df 1)


#bagFDAGCV Model validation
set.seed(12)

control = trainControl(method = "none",
                       number = 5,
                       verboseIter = TRUE,
                       repeats = 1,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

grid = expand.grid( degree=c(1))


model_caret = train(train_valid_targets~., data=train_valid, method="bagFDAGCV", trControl=control, tuneGrid = grid, metric="ROC")

valid_pred = predict(model_caret, newdata=valid, type="prob")

rocCurve = roc(response=valid_targets, predictor=valid_pred$virus )
auc = auc(rocCurve)
auc
#Validation AUC: 0.8032 degree=c(2))
#Validation AUC: 0.7897 degree=c(1))

model_caret = train(targets~., data=train, method="bagFDAGCV", trControl=control, tuneGrid = grid, metric="ROC")

bagFDAGCV_test_pred = predict(model_caret, newdata=test, type="prob")

submission21 = data.frame(Id=test_id,WnvPresent=bagFDAGCV_test_pred$virus)
write.csv(submission21, "wnv_sub21_bagFDAGCV.csv", row.names=FALSE)
#LB AUC: 0.64655
#LB AUC: 0.66275  degree 1


#bagEarthGCV Model validation
set.seed(12)

control = trainControl(method = "none",
                       number = 5,
                       verboseIter = TRUE,
                       repeats = 1,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

grid = expand.grid( degree=c(1))


model_caret = train(train_valid_targets~., data=train_valid, method="bagEarthGCV", trControl=control, tuneGrid = grid, metric="ROC")

valid_pred = predict(model_caret, newdata=valid, type="prob")

rocCurve = roc(response=valid_targets, predictor=valid_pred$virus )
auc = auc(rocCurve)
auc
#Validation AUC: 0.8198 degree=c(2))
#Validation AUC: 0.8027 degree=c(1))


model_caret = train(targets~., data=train, method="bagEarthGCV", trControl=control, tuneGrid = grid, metric="ROC")

bagEarthGCV_test_pred = predict(model_caret, newdata=test, type="prob")

submission22 = data.frame(Id=test_id,WnvPresent=bagEarthGCV_test_pred$virus)
write.csv(submission22, "wnv_sub22_bagEarthGCV.csv", row.names=FALSE)
#LB AUC: 0.67438 degree2
#LB AUC: 0.67275  degree1



#NB model
naiveBayes_model = naiveBayes(train_valid_targets~., data=train_valid, laplace=0)

valid_pred = predict(naiveBayes_model, newdata=valid, type="raw")

rocCurve = roc(response=valid_targets, predictor=valid_pred[,2])
auc = auc(rocCurve)
auc
#Valid AUC: 0.7332

naiveBayes_model = naiveBayes(targets~., data=train, laplace=0)

naiveBayes_pred = predict(naiveBayes_model, newdata=test, type="raw")

submission19 = data.frame(Id=test_id,WnvPresent=naiveBayes_pred[,2])
write.csv(submission19, "wnv_sub19_NB.csv", row.names=FALSE)
#LB AUC: 0.73191 #!!! Naive Bayes significantly better than my RF, logistic ensemble. Without better features/data, simple models seem to work best on this problem.






#Ensemble Validation
log_weight = 0.75
rf_weight = 1

ensemble_model_valid =  (glm_valid_pred*log_weight+rf_valid_pred*rf_weight)/sum(log_weight+rf_weight)

rocCurve = roc(response=valid_targets, predictor=ensemble_model_valid$virus)
ensemble_auc = auc(rocCurve)
ensemble_auc
#Best Logistic/RF ensemble AUC: 0.7875  weights: log_weight 0.75 rf_weight 1


#Ensemble Submission code:
log_weight = 0.75
rf_weight = 1

ensemble_model_test =  (glm_test_pred*log_weight+rf_test_pred*rf_weight)/sum(log_weight+rf_weight)

submission8 = data.frame(Id=test_id,WnvPresent=ensemble_model_test$virus)
write.csv(submission8, "wnv_sub8_log_rf.csv", row.names=FALSE)
#LB AUC: 0.71866 #Log = submission 6, RF = submission 7


log_weight = 1
rf_weight = 0.75

ensemble_model_test =  (glm_test_pred*log_weight+rf_test_pred*rf_weight)/sum(log_weight+rf_weight)

submission9 = data.frame(Id=test_id,WnvPresent=ensemble_model_test$virus)
write.csv(submission9, "wnv_sub9_log_rf.csv", row.names=FALSE)
#LB AUC: 0.72000   #Log = submission 6, RF = submission 7
#More weight on the better (logistic) model gives higher LB score


#Ensemble gam, nb, glmnet
gam = read.csv("wnv_sub23_gam.csv")
nb = read.csv("wnv_sub19_NB.csv")
glm = read.csv("wnv_sub11_glmnet.csv")
knn = read.csv("wnv_sub15_knn.csv")

ensemble_model_test =  (gam*0.5+nb+glm*0.5)/2

submission24 = data.frame(Id=test_id, WnvPresent=ensemble_model_test$WnvPresent)
write.csv(submission24, "wnv_sub24_nb_gam_glmnet.csv", row.names=FALSE)
#LB AUC: 0.74037


#Ensemble gam, nb, glmnet, knn
gam = read.csv("wnv_sub23_gam.csv")
nb = read.csv("wnv_sub19_NB.csv")
glm = read.csv("wnv_sub11_glmnet.csv")
knn = read.csv("wnv_sub15_knn.csv")

ensemble_model_test =  (gam*0.3+nb+glm*0.3+knn*0.2)/1.8

submission25 = data.frame(Id=test_id, WnvPresent=ensemble_model_test$WnvPresent)
write.csv(submission25, "wnv_sub25_nb_gam_glmnet.csv", row.names=FALSE)
#LB AUC: 0.74692

#Ensemble gam, nb, glmnet, knn, beat_the_benchmark
gam = read.csv("wnv_sub23_gam.csv")
nb = read.csv("wnv_sub19_NB.csv")
glm = read.csv("wnv_sub11_glmnet.csv")
knn = read.csv("wnv_sub15_knn.csv")
bbm = read.csv("beat_the_benchmark.csv")

ensemble_model_test =  (gam*0.3+nb+glm*0.3+knn*0.2)/1.8

submission32 = data.frame(Id=test_id, WnvPresent=ensemble_model_test$WnvPresent)
write.csv(submission32, "wnv_sub32_ensemble.csv", row.names=FALSE)
#LB AUC: 0.74671

#Ensemble gam, nb, glmnet, knn, more weight to weaker models
gam = read.csv("wnv_sub23_gam.csv")
nb = read.csv("wnv_sub19_NB.csv")
glm = read.csv("wnv_sub11_glmnet.csv")
knn = read.csv("wnv_sub15_knn.csv")
bbm = read.csv("beat_the_benchmark.csv")

ensemble_model_test =  (gam*0.5+nb+glm*0.5+knn*0.5)/2.5

submission33 = data.frame(Id=test_id, WnvPresent=ensemble_model_test$WnvPresent)
write.csv(submission33, "wnv_sub33_ensemble.csv", row.names=FALSE)
#LB AUC: 0.75000

#Ensemble gam, nb, glmnet, knn, more weight to weaker models
gam = read.csv("wnv_sub23_gam.csv")
nb = read.csv("wnv_sub19_NB.csv")
glm = read.csv("wnv_sub11_glmnet.csv")
knn = read.csv("wnv_sub15_knn.csv")
bbm = read.csv("beat_the_benchmark.csv")

ensemble_model_test =  ((gam*0.75+nb+glm*0.75+knn*0.75)/3.25)^0.95

submission34 = data.frame(Id=test_id, WnvPresent=ensemble_model_test$WnvPresent)
write.csv(submission34, "wnv_sub34_ensemble.csv", row.names=FALSE)
#LB AUC: 0.75020


ensemble_model = read.csv("wnv_sub33_ensemble.csv")
ensemble_model$WnvPresent = ensemble_model$WnvPresent^0.95
write.csv(ensemble_model, "wnv_sub34_ensemble.csv", row.names=FALSE)
#LB AUC: 


#Try some H20 nnets!
library(h2o)
localH2O <- h2o.init(nthread=4, max_mem_size="20g")

train$targets = targets

train.hex <- as.h2o(localH2O,train)
test.hex <- as.h2o(localH2O,test)

predictors <- 1:24
response <- 25



#Validation code
set.seed(12)

train_index = createDataPartition(train$target, p = .75,
                                  list = FALSE,
                                  times = 1)

valid = train[ -train_index,]
train_valid = train[ train_index,]

train_valid.hex <- as.h2o(localH2O,train_valid)
valid.hex <- as.h2o(localH2O, valid)

preds = valid_preds

for (i in 2:30){
print(i)
model <- h2o.deeplearning(x=predictors,
                          y=response,
                          data=train_valid.hex,
                          classification=T,
                          activation="RectifierWithDropout",
                          hidden=c(20,10),
                          hidden_dropout_ratio=c(0.2,0.2),
                          input_dropout_ratio=0.01,
                          epochs=30,
                          l1=1e-3,
                          l2=1e-3,
                          rho=0.999,
                          epsilon=1e-8,
                          train_samples_per_iteration=100,
                          max_w2=10,
                          seed=i, 
                          fast_mode=T,
                          shuffle_training_data=T) 

valid_preds <- as.data.frame(h2o.predict(model,valid.hex))

preds$virus = preds$virus+valid_preds$virus

}

preds$virus = preds$virus/30

rocCurve = roc(response=valid_targets, predictor=preds$virus)
auc = auc(rocCurve)
auc




submission = sampleSubmission
#NNet ensemble Submission code
for (i in 1:30){
  print(i)
  model <- h2o.deeplearning(x=predictors,
                            y=response,
                            data=train.hex,
                            classification=T,
                            activation="RectifierWithDropout",
                            hidden=c(20,10),
                            hidden_dropout_ratio=c(0.2,0.2),
                            input_dropout_ratio=0.01,
                            epochs=30,
                            l1=1e-3,
                            l2=1e-3,
                            rho=0.999,
                            epsilon=1e-8,
                            train_samples_per_iteration=100,
                            max_w2=10,
                            seed=i, 
                            fast_mode=T,
                            shuffle_training_data=T) 
  
  test_preds <- as.data.frame(h2o.predict(model,test.hex))
  
  submission$WnvPresent = submission$WnvPresent+test_preds$virus
}

submission$WnvPresent = submission$WnvPresent/30
write.csv(submission, "wnv_H20_ensemble1.csv", row.names=FALSE)
#LB AUC:0.63070 -- As expected, complex model performs poorly