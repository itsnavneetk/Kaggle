#Liberty Mutual Group: Property Inspection Prediction
#https://www.kaggle.com/c/liberty-mutual-group-property-inspection-prediction
#Author: Greg Hamel

library(ggplot2)
library(caret)
library(randomForest)
library(e1071)
library(glmnet)
library(gbm)
library(xgboost)
library(plyr)
library(dplyr)
library(data.table)
library(Metrics)
library(lubridate)

sample_sub = read.csv("sample_submission.csv")

train = read.csv("train.csv")
test = read.csv("test.csv")

#Data set is comprised of only ints and factors, not much munging required.
train$Id = NULL
test$Id =NULL

targets = train$Hazard
train$Hazard = NULL


#Function for scoring:
#"NormalizedGini" is the other half of the metric. This function does most of the work, though
SumModelGini <- function(solution, submission) {
  df = data.frame(solution = solution, submission = submission)
  df <- df[order(df$submission, decreasing = TRUE),]
  df
  df$random = (1:nrow(df))/nrow(df)
  df
  totalPos <- sum(df$solution)
  df$cumPosFound <- cumsum(df$solution) # this will store the cumulative number of positive examples found (used for computing "Model Lorentz")
  df$Lorentz <- df$cumPosFound / totalPos # this will store the cumulative proportion of positive examples found ("Model Lorentz")
  df$Gini <- df$Lorentz - df$random # will store Lorentz minus random
  return(sum(df$Gini))
}

NormalizedGini <- function(solution, submission) {
  SumModelGini(solution, submission) / SumModelGini(solution, solution)
}


#Split into train and validation sets
set.seed(12)

train_index = createDataPartition(targets, p = .75,
                                  list = FALSE,
                                  times = 1)

valid = train[ -train_index,]
train_valid = train[ train_index,]

train_valid_targets = targets[ train_index]
valid_targets =  targets[ -train_index]


#Convert categorical variables into indicator variables
train_valid2 = model.matrix(~.,train_valid)
valid2 = model.matrix(~.,valid)

#Start with an RF model

control = trainControl(method = "none",
                       number = 2,
                       verboseIter = TRUE,
                       repeats = 2
)

grid = expand.grid(  mtry= c(8))

rf_mod_caret = train(train_valid_targets~.,data=train_valid,method="rf", trControl=control, tuneGrid = grid, metric="RMSE", ntree=50,nodesize=3)

valid_preds = predict(rf_mod_caret, newdata=valid)
gini_score = NormalizedGini(valid_targets, valid_preds)
gini_score



#XGB
set.seed(121)

train_m_valid = matrix(as.numeric(data.matrix(train_valid)),ncol=ncol(train_valid))
valid_m = matrix(as.numeric(data.matrix(valid)),ncol=ncol(valid))

num_ttargets_train_valid = as.numeric(train_valid_targets)
num_ttargets_valid = as.numeric(valid_targets)

# train_m_valid = log(train_m_valid+1)
# valid_m = log(valid_m+1)

#Average several xgb models
runs = 10
total_model = rep(0, (nrow(valid)) )

for (r in 1:runs){
print("Run")
print(r)

set.seed(1)
xgb_model = xgboost(data=train_m_valid, label=num_ttargets_train_valid, nrounds=900, verbose=0, eta=0.01, gamma=0.1, max_depth=6, min_child_weight=0.5, subsample=0.7, objective="reg:linear", eval_metric="rmse", lambda=1,alpha=0.5,lambda_bias=0.1)

xgb_preds = predict(xgb_model, valid_m)
error = NormalizedGini(num_ttargets_valid, xgb_preds)
print(error)

total_model = total_model+xgb_preds
print( NormalizedGini(num_ttargets_valid, total_model/r) )
}
#Validation score for 10 runs (for sub1): 0.3806518



#XGB on expanded data (indicator variable matrix)
set.seed(121)

train_m_valid2 = matrix(as.numeric(data.matrix(train_valid2)),ncol=ncol(train_valid2))
valid_m2 = matrix(as.numeric(data.matrix(valid2)),ncol=ncol(valid2))

num_ttargets_train_valid = as.numeric(train_valid_targets)
num_ttargets_valid = as.numeric(valid_targets)

train_m_valid = log(train_m_valid2+1)
valid_m = log(valid_m2+1)

runs = 10
total_model = rep(0, (nrow(valid)) )

for (r in 1:runs){
  print("Run")
  print(r)
  
set.seed(10)
xgb_model2 = xgboost(data=train_m_valid2, label=num_ttargets_train_valid, nrounds=800, verbose=0, eta=0.025, gamma=0.1, max_depth=4, min_child_weight=0.5, subsample=0.7, objective="reg:linear", eval_metric="rmse", lambda=2,alpha=0.5,lambda_bias=0)

xgb_preds2 = predict(xgb_model2, valid_m2)
error2 = NormalizedGini(num_ttargets_valid, xgb_preds2)
print(error2)

total_model = total_model+xgb_preds2
print( NormalizedGini(num_ttargets_valid, total_model/r) )

}



#XGB average of 10 submission

set.seed(121)

train_m = matrix(as.numeric(data.matrix(train)),ncol=ncol(train))
test_m = matrix(as.numeric(data.matrix(test)),ncol=ncol(test))

num_ttargets = as.numeric(targets)


#Average several xgb models
runs = 10
total_model = rep(0, (nrow(test)) )

for (r in 1:runs){
  print("Run")
  print(r)
  
  set.seed(r)
  xgb_model = xgboost(data=train_m, label=num_ttargets, nrounds=900, verbose=0, eta=0.01, gamma=0.1, max_depth=6, min_child_weight=0.5, subsample=0.7, objective="reg:linear", eval_metric="rmse", lambda=0.5,alpha=0.5,lambda_bias=0)
  
  xgb_preds = predict(xgb_model, test_m)
  total_model = total_model+xgb_preds
}

final_preds = total_model/runs

sample_sub$Hazard = final_preds
write.csv(sample_sub, "liberty_sub1_xgb.csv", row.names=FALSE)
#Initial submission score: 0.379993




