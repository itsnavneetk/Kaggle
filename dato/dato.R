#Truly Native?
#https://www.kaggle.com/c/dato-native

library(caret)
library(xgboost)
library(data.table)
library(pROC)
library(randomForest)

sample <- read.csv("data/sampleSubmission_v2.csv", stringsAsFactors = FALSE)
train <- read.csv("train_feats.csv", stringsAsFactors = FALSE)
test <- read.csv("test_feats.csv", stringsAsFactors = FALSE)

targets <- train$sponsored
targets <- as.factor(targets)
levels(targets) <- c("not_s","sponsored")
train$sponsored <- NULL
test$sponsored <- NULL

train$file <- NULL
test_files <- test$file
test$file <- NULL

#Try model with a holdout validation set
set.seed(11)
train_index = createDataPartition(targets,p=0.75,list = FALSE, times = 1)
train_valid = train[train_index,]
valid = train[-train_index,]

t_valid_targets = targets[train_index]
valid_targets = targets[-train_index]


#RF model to start
control = trainControl(method = "none",
                       number = 1,
                       verboseIter = TRUE,
                       repeats = 1,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

grid = expand.grid( mtry=c(4))

set.seed(12)
rf_model_caret = train(t_valid_targets~., data=train_valid, method="rf", trControl=control, tuneGrid=grid, ntree = 200, metric="ROC", nodesize=1)

pred = predict(rf_model_caret, newdata=valid, type="prob")

rocCurve = roc(response=valid_targets, predictor=pred$sponsored)
auc(rocCurve)

# Valid acc with 200 trees: 0.9303



# gbm model
control = trainControl(method = "none",
                       number = 1,
                       verboseIter = TRUE,
                       repeats = 1,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

grid = expand.grid( n.trees=c(150), interaction.depth=c(15), shrinkage=c(0.1), n.minobsinnode=c(1))

set.seed(12)
rf_model_caret = train(t_valid_targets~., data=train_valid, method="gbm", trControl=control, tuneGrid=grid)

pred = predict(rf_model_caret, newdata=valid, type="prob")

rocCurve = roc(response=valid_targets, predictor=pred$sponsored)
auc(rocCurve)



# xgb model raw
set.seed(121)

train_m_v = matrix(as.numeric(data.matrix(train_valid)),ncol=ncol(train_valid))

valid_m = matrix(as.numeric(data.matrix(valid)),ncol=ncol(valid))

num_ttargets_t_v = as.numeric(t_valid_targets)-1
num_ttargets_v = as.numeric(valid_targets)-1


set.seed(1)
xgb_model = xgboost(data=train_m_v, label=num_ttargets_t_v, nrounds=100, verbose=1, eta=0.025, gamma=0.10, max_depth=20, min_child_weight=1, subsample=0.75, objective="binary:logistic", eval_metric="auc")

xgb_preds = predict(xgb_model, valid_m)

rocCurve = roc(response=num_ttargets_v, predictor=xgb_preds)
print( auc(rocCurve) )



# XGB submission
set.seed(121)

train_m = matrix(as.numeric(data.matrix(train)),ncol=ncol(train))

test_m = matrix(as.numeric(data.matrix(test)),ncol=ncol(test))

num_ttargets = as.numeric(targets)-1



set.seed(1)
xgb_model = xgboost(data=train_m, label=num_ttargets, nrounds=100, verbose=1, eta=0.025, gamma=0.10, max_depth=20, min_child_weight=1, subsample=0.75, objective="binary:logistic", eval_metric="auc")

xgb_preds = predict(xgb_model, test_m)

submission <- data.frame(file=test_files, sponsored=xgb_preds)
missing_files <- c(4967,7523,27567,35567)
submission <- rbind(submission, sample[missing_files,])

write.csv(submission,"dato_sub1_xgb.csv", row.names=FALSE)

#Leaderboard score: 0.93254



#RF model sub
control = trainControl(method = "none",
                       number = 1,
                       verboseIter = TRUE,
                       repeats = 1,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

grid = expand.grid( mtry=c(4))

set.seed(12)
rf_model_caret = train(targets~., data=train, method="rf", trControl=control, tuneGrid=grid, ntree = 200, metric="ROC", nodesize=1)

gc()

preds = predict(rf_model_caret, newdata=test, type="prob")

submission <- data.frame(file=test_files, sponsored=preds$sponsored)
missing_files <- c(4967,7523,27567,35567)
submission <- rbind(submission, sample[missing_files,])

write.csv(submission,"dato_sub2_rf.csv", row.names=FALSE)

#Leaderboard score: 0.93548

sub1 <- read.csv("dato_sub1_xgb.csv")
sub2 <- read.csv("dato_sub2_rf.csv")

sub3 <- sub1
sub3$sponsored <- sub1$sponsored*0.5 + sub2$sponsored*0.5

write.csv(sub3,"dato_sub3_rfxgb.csv", row.names=FALSE)

# combined rf xgb score: 0.93958