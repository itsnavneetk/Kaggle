#Caterpillar Tube Pricing
#https://www.kaggle.com/c/caterpillar-tube-pricing/leaderboard
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
filenames <- list.files("competition_data", pattern="*.csv", full.names=TRUE)

for (f in filenames){
  assign(strsplit(f,"/")[[1]][2] , read.csv(f, quote = "", 
           row.names = NULL, 
           stringsAsFactors = FALSE) )
}

train = train_set.csv
test = test_set.csv

train = left_join(train, tube.csv, by = "tube_assembly_id")
test = left_join(test, tube.csv, by = "tube_assembly_id")

train = left_join(train, tube_end_form.csv, by = c("end_a" = "end_form_id") )
test = left_join(test, tube_end_form.csv, by = c("end_a" = "end_form_id") )

train = left_join(train, tube_end_form.csv, by = c("end_x" = "end_form_id") )
test = left_join(test, tube_end_form.csv, by = c("end_x" = "end_form_id"))

#Get unique special attachments
spec_list = unique(unlist(specs.csv[2:11]))[2:86]

#Create indicator variables for all spec types
for (spec in spec_list){
  train[spec] = NA
  test[spec] = NA
}

for (tube in 1:nrow(train)) {
  row = train[tube, ]
  specs = as.vector(unlist(specs.csv[as.numeric(strsplit(row$tube_assembly_id,"-")[[1]][2]),2:11]))
  for(s in specs){
    if (is.na(s)){ 
    break
    } else {
    train[tube, s] = 1
    }
  }
}

for (tube in 1:nrow(test)) {
  row = test[tube, ]
  specs = as.vector(unlist(specs.csv[as.numeric(strsplit(row$tube_assembly_id,"-")[[1]][2]),2:11]))
  for(s in specs){
    if (is.na(s)){ 
      break
    } else {
      test[tube, s] = 1
    }
  }
}

train[is.na(train)] = 0
test[is.na(test)] = 0

#Save new train and test sets so far
# write.csv(train,"train.csv",row.names=FALSE)
# write.csv(test,"test.csv",row.names=FALSE)

train = read.csv("train.csv", stringsAsFactors = FALSE)
test = read.csv("test.csv", stringsAsFactors = FALSE)



#Get unique components
comp_list = unique(unlist(bill_of_materials.csv[seq(2,17,2)]))
comp_list = comp_list[!is.na(comp_list)]

#Create variables for component counts
#This is going to create over 2000 features, which will likely end up being too much to work with. Will have to reduce later.
for (c in comp_list){
  train[c] = NA
  test[c] = NA
}


for (tube in 1:nrow(train)) {
  row = train[tube, ]
  comps = as.vector(unlist(bill_of_materials.csv[as.numeric(strsplit(row$tube_assembly_id,"-")[[1]][2]),seq(2,17,2)]))
  quantities = as.vector(unlist(bill_of_materials.csv[as.numeric(strsplit(row$tube_assembly_id,"-")[[1]][2]),seq(3,18,2)]))
  for(c in 1:length(comps)){
    if (is.na(comps[c])){ 
      break
    } else {
      train[tube, comps[c]] = quantities[c]
    }
  }
}

for (tube in 1:nrow(test)) {
  row = test[tube, ]
  comps = as.vector(unlist(bill_of_materials.csv[as.numeric(strsplit(row$tube_assembly_id,"-")[[1]][2]),seq(2,17,2)]))
  quantities = as.vector(unlist(bill_of_materials.csv[as.numeric(strsplit(row$tube_assembly_id,"-")[[1]][2]),seq(3,18,2)]))
  for(c in 1:length(comps)){
    if (is.na(comps[c])){ 
      break
    } else {
      test[tube, comps[c]] = quantities[c]
    }
  }
}

train[is.na(train)] = 0
test[is.na(test)] = 0

#Save new train and test with component vars added
# write.csv(train,"train_full_comp.csv",row.names=FALSE)
# write.csv(test,"test_full_comp.csv",row.names=FALSE)


#Read in full data sets
train = fread("train_full_comp.csv")
test = fread("test_full_comp.csv")

train=data.frame(train)
test=data.frame(test)

#Extract date varaibles
train$year = year(train$quote_date)
train$month = month(train$quote_date)
train$day = day(train$quote_date)
train$wday = wday(train$quote_date)
train$yday = yday(train$quote_date)

test$year = year(test$quote_date)
test$month = month(test$quote_date)
test$day = day(test$quote_date)
test$wday = wday(test$quote_date)
test$yday = yday(test$quote_date)


#Throw out vars not needed for prediction
test$id = NULL
train$tube_assembly_id = NULL
test$tube_assembly_id = NULL

train$quote_date = NULL
test$quote_date = NULL


#Remove variables with few non-zero rows. For now, only keep rows with at least 15 non-zero values
for (var in colnames(train[23:ncol(train)])){
  if (length(which(train[var] > 0)) < 15 ){
    train[var] = NULL
    test[var] = NULL
  }
}

# write.csv(train, "reduced15_train.csv")
# write.csv(test, "reduced15_test.csv")


train = read.csv("reduced15_train.csv")
test = read.csv("reduced15_test.csv")

targets = train$cost
train$cost = NULL

train_mod = model.matrix(~.,train)
test_mod = model.matrix(~.,test)

train = data.frame(train_mod)
test = data.frame(test_mod)

#Adjust data sparsity here
cutoff = 50
for (var in colnames(train)){
  if (length(which(train[var] > 0)) < cutoff ){
    train[var] = NULL
    test[var] = NULL
  }
}

for (var in colnames(test)){
  if (!(var %in% colnames(train))){
    test[var] = NULL
  }
}


#Split train into parts for validation
set.seed(12)
train_index = createDataPartition(targets, p = .75,
                                  list = FALSE,
                                  times = 1)

valid = train[ -train_index,]
train_valid = train[ train_index,]

valid_target = targets[ -train_index]
train_valid_target= targets[train_index]


#Try RF
set.seed(10)

rf_mod = randomForest(train_valid_target~., data=train_valid, ntree=10, nodesize=3, mtry=30)

train_preds = predict(rf_mod, newdata=valid)
train_preds = ifelse(train_preds<0,0,train_preds)

error = rmsle(valid_target,train_preds)
error
#Validation score with sparsity cutoff 50 (166 vars) and ntree=100, nodesize=3, mtry=40): 0.2598103


#RF submission code:
set.seed(10)

rf_mod = randomForest(targets~., data=train, ntree=100, nodesize=3, mtry=40)

test_preds = predict(rf_mod, newdata=test)

sample_sub$cost = test_preds
write.csv(sample_sub,"cat_tube_sub1RF.csv", row.names=FALSE)
#Test score with sparsity cutoff 50 (166 vars) and ntree=100, nodesize=3, mtry=40): 0.313247  -- Validation holdout doesn't do a great job tracking test outcome




#Try xgb
set.seed(121)

train_m_valid = matrix(as.numeric(data.matrix(train_valid)),ncol=ncol(train_valid))
valid_m = matrix(as.numeric(data.matrix(valid)),ncol=ncol(valid))

num_ttargets_train_valid = as.numeric(train_valid_target)
num_ttargets_valid = as.numeric(valid_target)

set.seed(13)
xgb_model = xgboost(data=train_m_valid, label=num_ttargets_train_valid, nrounds=500, verbose=1, eta=0.025, gamma=0.1, max_depth=13, min_child_weight=1, subsample=0.8, objective="reg:linear", eval_metric="rmse", lambda=1,alpha=1,lambda_bias=0)

xgb_preds = predict(xgb_model, valid_m)
xgb_preds = ifelse(xgb_preds<0,1.5,xgb_preds)
error = rmsle(num_ttargets_valid, xgb_preds)
error
#Best valid: 0.2192433


#Try xgb sub
set.seed(121)

train_m_full = matrix(as.numeric(data.matrix(train)),ncol=ncol(train))
test_m = matrix(as.numeric(data.matrix(test)),ncol=ncol(test))

num_ttargets = as.numeric(targets)

set.seed(13)
xgb_model = xgboost(data=train_m_full, label=num_ttargets, nrounds=500, verbose=1, eta=0.025, gamma=0.1, max_depth=13, min_child_weight=1, subsample=0.8, objective="reg:linear", eval_metric="rmse", lambda=1,alpha=1,lambda_bias=0)

xgb_preds = predict(xgb_model, test_m)
xgb_preds = ifelse(xgb_preds<0,1.5,xgb_preds)
sample_sub$cost = xgb_preds
write.csv(sample_sub,"cat_tube_sub2XGB.csv", row.names=FALSE)
