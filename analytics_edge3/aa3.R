#15.071x - The Analytics Edge (Summer 2015)

library(ggplot2)
library(rpart)
library(caret)
library(randomForest)
library(e1071)
library(glmnet)
library(xgboost)
library(tm)
library(pROC)
library(SnowballC)
library(dummies)

train = read.csv("eBayiPadTrain.csv", stringsAsFactors=TRUE)
test = read.csv("eBayiPadTest.csv", stringsAsFactors=TRUE)

sample_sub = read.csv("SampleSubmission.csv")

train$UniqueID = NULL
test$UniqueID = NULL

targets = train$sold
train$sold = NULL
targets = as.factor(paste("X",as.character(targets),sep="_"))


#Add a couple features related to the description
train$description_exists = ifelse( nchar(as.character(train$description)) > 1,1,0)
train$description_length = nchar(as.character(train$description))

test$description_exists = ifelse( nchar(as.character(test$description)) > 1,1,0)
test$description_length = nchar(as.character(test$description))


#Try model with a holdout validation set
set.seed(11)
train_index = createDataPartition(targets,p=0.75,list = FALSE,
                                  times = 1)
train_valid = train[train_index,]
valid = train[-train_index,]

t_valid_targets = targets[train_index]
valid_targets = targets[-train_index]


#RF model to start, don't deal with description yet

control = trainControl(method = "repeatedcv",
                       number = 10,
                       verboseIter = TRUE,
                       repeats = 1,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

grid = expand.grid( mtry=c(6))
  
set.seed(12)
rf_model_caret = train(t_valid_targets~., data=train_valid[2:11], method="rf", trControl=control, tuneGrid=grid, ntree = 250, metric="ROC", nodesize=1)

pred = predict(rf_model_caret, newdata=valid[2:11], type="prob")

rocCurve = roc(response=valid_targets, predictor=pred$X_1)
auc(rocCurve)




#XGB
dummy_train = data.frame(model.matrix(~., train[4:9]))
dummy_train$X.Intercept. = NULL
dummy_train = cbind(train[2:3], dummy_train)
dummy_train = cbind(train[10:11], dummy_train)

set.seed(121)

train_m = matrix(as.numeric(data.matrix(dummy_train)),ncol=34)

num_ttargets = as.numeric(targets)-1

set.seed(13)
xgb_model = xgboost(data=train_m, label=num_ttargets, nrounds=100, verbose=1, eta=0.1, gamma=0.1, max_depth=5, min_child_weight=1, subsample=0.5, objective="binary:logistic", eval_metric="auc")

xgb_preds = predict(xgb_model, train_m)
rocCurve = roc(response=num_ttargets, predictor=xgb_preds)
auc(rocCurve)




#XGB
set.seed(18)
train_index = createDataPartition(targets,p=0.75,list = FALSE,
                                  times = 1)
train_valid = dummy_train[train_index,]
valid = dummy_train[-train_index,]

t_valid_targets = targets[train_index]
valid_targets = targets[-train_index]


set.seed(121)

train_m_v = matrix(as.numeric(data.matrix(train_valid)),ncol=34)

valid_m = matrix(as.numeric(data.matrix(valid)),ncol=34)

num_ttargets_t_v = as.numeric(t_valid_targets)-1
num_ttargets_v = as.numeric(valid_targets)-1


set.seed(1)
xgb_model = xgboost(data=train_m_v, label=num_ttargets_t_v, nrounds=2000, verbose=0, eta=0.01, gamma=0.1, max_depth=5, min_child_weight=1, subsample=0.5, objective="binary:logistic", eval_metric="auc")

xgb_preds = predict(xgb_model, valid_m)
rocCurve = roc(response=num_ttargets_v, predictor=xgb_preds)
print( auc(rocCurve) )
















#Initial XGB submission (no description features)
combined= rbind(train,test)

dummy_combined = data.frame(model.matrix(~., combined[4:9]))
dummy_combined$X.Intercept. = NULL
dummy_combined = cbind(combined[2:3], dummy_combined)

dummy_train = dummy_combined[1:nrow(train),]
dummy_test= dummy_combined[(nrow(train)+1):nrow(combined),]


set.seed(121)

train_m = matrix(as.numeric(data.matrix(dummy_train)),ncol=34)
test_m = matrix(as.numeric(data.matrix(dummy_test)),ncol=34)

num_ttargets = as.numeric(targets)-1

set.seed(13)
xgb_model = xgboost(data=train_m, label=num_ttargets, nrounds=750, verbose=0, eta=0.01, gamma=0.1, max_depth=8, min_child_weight=1, subsample=0.5, objective="binary:logistic", eval_metric="auc")

xgb_preds = predict(xgb_model, test_m)

sample_sub$Probability1 = xgb_preds

write.csv(sample_sub,"aa3_submission1xgb.csv", row.names=FALSE)
#Score: 0.83061

#Initial RF model sub (no description features)
control = trainControl(method = "repeatedcv",
                       number = 10,
                       verboseIter = TRUE,
                       repeats = 1,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

grid = expand.grid( mtry=c(6))

set.seed(12)
rf_model_caret = train(targets~., data=train[2:9], method="rf", trControl=control, tuneGrid=grid, ntree = 250, metric="ROC", nodesize=1)

pred = predict(rf_model_caret, newdata=test[2:9], type="prob")

sample_sub$Probability1 = pred$X_1

write.csv(sample_sub,"aa3_submission2RF.csv", row.names=FALSE)
#Score:0.83348



#Second XGB submission (no description features)
combined= rbind(train,test)

dummy_combined = data.frame(model.matrix(~., combined[4:9]))
dummy_combined$X.Intercept. = NULL
dummy_combined = cbind(combined[2:3], dummy_combined)

dummy_train = dummy_combined[1:nrow(train),]
dummy_test= dummy_combined[(nrow(train)+1):nrow(combined),]


set.seed(121)

train_m = matrix(as.numeric(data.matrix(dummy_train)),ncol=34)
test_m = matrix(as.numeric(data.matrix(dummy_test)),ncol=34)

num_ttargets = as.numeric(targets)-1

set.seed(13)
xgb_model = xgboost(data=train_m, label=num_ttargets, nrounds=2000, verbose=0, eta=0.01, gamma=0.1, max_depth=5, min_child_weight=1, subsample=0.5, objective="binary:logistic", eval_metric="auc")

xgb_preds = predict(xgb_model, test_m)

sample_sub$Probability1 = xgb_preds

write.csv(sample_sub,"aa3_submission3xgb.csv", row.names=FALSE)
#Score: 0.82927
#This appears to be a case where simple models will score better


#Submission 4, combined RF and XGB
xgbmod = read.csv("aa3_submission1xgb.csv")
rfmod = read.csv("aa3_submission2RF.csv")
xgbmod$Probability1 = (xgbmod$Probability1+rfmod$Probability1)/2

write.csv(xgbmod,"aa3_submission4xgb_rf.csv", row.names=FALSE)
#0.83493 Slight improvement, but not huge
