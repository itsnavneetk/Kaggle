#Rossmann Store Sales
#https://www.kaggle.com/c/rossmann-store-sales

library(caret)
library(xgboost)
library(data.table)
library(pROC)
library(randomForest)
library(Metrics)
library(lubridate)
library(forecast)
library(dplyr)
library(elasticnet)
library(psych)
library(rpart)

# Data preparation--------------------------------------------------
train <- read.csv("train.csv", stringsAsFactors = FALSE)
test <- read.csv("test.csv", stringsAsFactors = FALSE)
store <- read.csv("store.csv")

#Fill NA with 0
store[is.na(store)] <- 0

train$StateHoliday <- as.factor(train$StateHoliday)
test$StateHoliday <- as.factor(test$StateHoliday)

# Extract Date information
# Note: Test set only contains dates in august and september 2015.

train$month <- as.integer(month(train$Date))
train$week <- week(train$Date)
train$year <- year(train$Date)
train$day_of_month <- mday(train$Date)
train$day_of_year <- yday(train$Date)

train$open_yesterday <- c(train$Open[1116:nrow(train)], rep(1,1115))
train$open_tomorrow <- c(rep(1,1115),train$Open[1:(nrow(train)-1115)])

test$month <- month(test$Date)
test$week <- week(test$Date)
test$year <- year(test$Date)
test$day_of_month <- mday(test$Date)
test$day_of_year <- yday(test$Date)

test$open_yesterday <- c(test$Open[857:nrow(test)], rep(1,856))
test$open_tomorrow <- c(rep(1,856),train$Open[1:(nrow(test)-856)])

# All days stores are closed have zero sales and zero sales days are ignored by the evaluation function. Remove closed days from training set.

nz_index <- train$Sales > 0
nz_train <- train[nz_index, ]

# Assume NA days in test are open days
test[is.na(test)] <- 1

nz_train$Date <-NULL
test$Date <- NULL
nz_train$Open <-NULL

nz_train_full <- merge(nz_train,store,by="Store")

test_full <- merge(test,store,by="Store")

# Calculate store sales benchmarks by month and day of week
# (version 4, switching from median to harmonic mean)
hm_store_sales_month_day <- nz_train_full %>%
  group_by(Store, month, DayOfWeek) %>%
  summarize(med_sales_month_day = harmonic.mean(Sales))

hm_store_sales_promo_day <- nz_train_full %>%
  group_by(Store, Promo, DayOfWeek) %>%
  summarize(med_promo_day = harmonic.mean(Sales))

hm_store_cust_promo_day <- nz_train_full %>%
  group_by(Store, Promo, DayOfWeek) %>%
  summarize(cust_promo_day = harmonic.mean(Customers))

hm_store_sales_month <- nz_train_full %>%
  group_by(Store, month) %>%
  summarize(med_promo_month = harmonic.mean(Sales))

store_sales_per_cust <- nz_train_full %>%
  group_by(Store) %>%
  summarize(sales_per_cust = mean(Sales/Customers))

med_store_sales_promo_day <- nz_train_full %>%
  group_by(Store, Promo, DayOfWeek) %>%
  summarize(med_promo_day = median(Sales))

min_store_sales_promo_day <- nz_train_full %>%
  group_by(Store, Promo, DayOfWeek) %>%
  summarize(min_promo_day = min(Sales))

max_store_sales_promo_day <- nz_train_full %>%
  group_by(Store, Promo, DayOfWeek) %>%
  summarize(max_promo_day = max(Sales))

sd_store_sales_promo_day <- nz_train_full %>%
  group_by(Store, Promo, DayOfWeek) %>%
  summarize(sd_promo_day = sd(Sales))


nz_train_full <- left_join(nz_train_full, hm_store_sales_month_day,
                           by=c("Store", "month","DayOfWeek"))

nz_train_full <- left_join(nz_train_full, hm_store_sales_promo_day ,
                           by=c("Store", "Promo", "DayOfWeek"))

nz_train_full <- left_join(nz_train_full, hm_store_cust_promo_day,
                           by=c("Store", "Promo", "DayOfWeek"))

nz_train_full <- left_join(nz_train_full, hm_store_sales_month,
                           by=c("Store", "month"))

nz_train_full <- left_join(nz_train_full, store_sales_per_cust,
                           by=c("Store"))

nz_train_full <- left_join(nz_train_full, med_store_sales_promo_day ,
                           by=c("Store", "Promo", "DayOfWeek"))
nz_train_full <- left_join(nz_train_full, min_store_sales_promo_day ,
                           by=c("Store", "Promo", "DayOfWeek"))
nz_train_full <- left_join(nz_train_full, max_store_sales_promo_day ,
                           by=c("Store", "Promo", "DayOfWeek"))
nz_train_full <- left_join(nz_train_full, sd_store_sales_promo_day ,
                           by=c("Store", "Promo", "DayOfWeek"))


test_full <- left_join(test_full, hm_store_sales_month_day,
                          by=c("Store", "month","DayOfWeek"))

test_full <- left_join(test_full, hm_store_sales_promo_day ,
                          by=c("Store", "Promo", "DayOfWeek"))

test_full <- left_join(test_full, hm_store_cust_promo_day,
                          by=c("Store", "Promo", "DayOfWeek"))

test_full <- left_join(test_full, hm_store_sales_month,
                          by=c("Store", "month"))

test_full <- left_join(test_full, store_sales_per_cust,
                          by=c("Store"))

test_full <- left_join(test_full, med_store_sales_promo_day ,
                       by=c("Store", "Promo", "DayOfWeek"))
test_full <- left_join(test_full, min_store_sales_promo_day ,
                       by=c("Store", "Promo", "DayOfWeek"))
test_full <- left_join(test_full, max_store_sales_promo_day ,
                       by=c("Store", "Promo", "DayOfWeek"))
test_full <- left_join(test_full, sd_store_sales_promo_day ,
                       by=c("Store", "Promo", "DayOfWeek"))


write.csv(nz_train_full,"train_full5.csv",row.names=FALSE)
write.csv(test_full,"test_full5.csv",row.names=FALSE)

# End Initial Data preparation-------------------------------------------


#Reload prepared data
sample <- read.csv("sample_submission.csv", stringsAsFactors = FALSE)

train <- fread("train_full4.csv", stringsAsFactors = TRUE)
train <- data.frame(train)

test <- fread("test_full4.csv", stringsAsFactors = TRUE)
test <- data.frame(test)

test[is.na(test)] <- 0

#Remove some vars that prove bad in validation
# train$StateHoliday <- NULL
# test$StateHoliday <- NULL
# train$day_of_year <- NULL
# test$day_of_year <- NULL

# train$open_tomorrow <- NULL
# test$open_tomorrow <- NULL
# train$open_yesterday <- NULL
# test$open_yesterday <- NULL

train$compsince <- (2015-train$CompetitionOpenSinceYear)+(12-(train$CompetitionOpenSinceMonth-1))/12
test$compsince <- (2015-test$CompetitionOpenSinceYear)+(12-(test$CompetitionOpenSinceMonth-1))/12

train$CompetitionOpenSinceYear <- NULL
train$CompetitionOpenSinceMonth <- NULL
test$CompetitionOpenSinceYear <- NULL
test$CompetitionOpenSinceMonth <- NULL


train$promo2since <- (2015-train$Promo2SinceYear)+(52-(train$Promo2SinceWeek-1))/52
test$promo2since <- (2015-test$Promo2SinceYear)+(52-(test$Promo2SinceWeek-1))/52

train$Promo2SinceYear <- NULL
train$Promo2SinceWeek <- NULL
test$Promo2SinceYear <- NULL
test$Promo2SinceWeek <- NULL

train <- train[,c(1:19,25,26)]
test <- test[,c(1:19,25,26)]


# Evaluation = Root Mean Square Percentage Error
RMSPE <- function(obs, preds){
  return( sqrt(mean(((obs-preds)/obs)^2)) )
}

RMSPEc <- function(data, lev=NULL, model){
  obs <- data[,1]
  preds <- data[,2]
  return( sqrt(mean(((obs-preds)/obs)^2)) )
}

targets <- train$Sales
train$Sales <- NULL
train$Customers <- NULL

test_ids <- test$Id
test$Id <- NULL

test_open <- test$Open
test$Open <- NULL


# Create a hold-out validation set
set.seed(10)
train_index <- createDataPartition(targets,p=0.75,list = FALSE,times = 1)

train_valid <- train[train_index,]
valid <- train[-train_index,]

t_valid_targets <- targets[train_index]
valid_targets <- targets[-train_index]


#Testing a submission... Test submission working...
sample_sub <- data.frame(Id=test_ids, Sales=test$avg_sales_month_day)
sample_sub$Sales[test_open == 0] <- 0
write.csv(sample_sub,"test_sub3.csv", row.names=FALSE)

# Try a simple linear regression model
lm_model <- lm(targets~., data=train)
preds <- predict(lm_model, newdata=train)

RMSPE(preds, targets)
#On train set tested RMSPE: 0.2476473

test_preds <- predict(lm_model, newdata=test)

sample_sub <- data.frame(Id=test_ids, Sales=test_preds)
sample_sub$Sales[test_open == 0] <- 0
write.csv(sample_sub,"test_sub4_lm.csv", row.names=FALSE)
# LB Score: 0.18845





# Try random forest model

control <- trainControl(method = "none",
                       number = 1,
                       verboseIter = TRUE,
                       repeats = 1
)

grid <- expand.grid( mtry=c(10) )

set.seed(9)
rf_model_caret <- train(log(t_valid_targets+1)~., data=train_valid, method="rf", trControl=control, tuneGrid=grid, ntree = 10, metric="rmse", nodesize=50, sampsize=400000)

pred <- (exp(predict(rf_model_caret, newdata=valid))-1)^0.996
RMSPE(valid_targets, pred)

index98 <- pred[valid$month %in% c(8,9)]
pred98 <- pred[index98]
t98 <- valid_targets[index98]
RMSPE(t98, pred98)


# Model with small sample size and trees is performing better on holdout
# Holdout RMSPE: 0.2185676 with mtry=c(7) ntree = 10, nodesize=5, sampsize=50000
# Holdout RMSPE: 0.2132141 with mtry=c(7) ntree = 50, nodesize=10, sampsize=100000

# After changing calulated cols from median to harmonic mean--------
# Holdout RMSPE: 0.1472188 with mtry=c(12) ntree = 100, nodesize=40, sampsize=50000 ^0.998 (seed=12)

# After converting competition since and promo2 sence to single vars:
# Holdout RMSPE: 0.1226156 with mtry=c(11) ntree = 50, nodesize=5, sampsize=50000 ^0.998 (seed=11)

# After converting competition since and promo2 sence to single vars:
# Holdout RMSPE: 0.1215508 with mtry=c(11) ntree = 50, nodesize=20, sampsize=100000 ^0.998 (seed=11)



# Submit initial RF model
control <- trainControl(method = "none",
                       number = 1,
                       verboseIter = TRUE,
                       repeats = 1
)

grid = expand.grid( mtry=c(7) )

set.seed(12)
rf_model_caret <- train(targets~., data=train, method="rf", trControl=control, tuneGrid=grid, ntree = 10, metric="rmse", nodesize=5, sampsize=50000)

test_preds <- predict(rf_model_caret, newdata=test)

sample_sub <- data.frame(Id=test_ids, Sales=test_preds)
sample_sub$Sales[test_open == 0] <- 0
write.csv(sample_sub,"test_sub6_rf.csv", row.names=FALSE)
# LB Score: 0.15081


# Try another submission with new RF model
control <- trainControl(method = "none",
                        number = 1,
                        verboseIter = TRUE,
                        repeats = 1
)

grid = expand.grid( mtry=c(12) )

set.seed(12)
rf_model_caret <- train(log(targets+1)~., data=train, method="rf", trControl=control, tuneGrid=grid, ntree = 50, metric="rmse", nodesize=6, sampsize=10000)

test_preds <- (exp(predict(rf_model_caret, newdata=test))-1)^0.996

sample_sub <- data.frame(Id=test_ids, Sales=test_preds)
sample_sub$Sales[test_open == 0] <- 0
write.csv(sample_sub,"test_sub7_rf.csv", row.names=FALSE)
# LB Score: 0.13699... Not the improvement I had expected...


# Try another submission with new RF model
control <- trainControl(method = "none",
                        number = 1,
                        verboseIter = TRUE,
                        repeats = 1
)

grid = expand.grid( mtry=c(12) )

set.seed(12)
rf_model_caret <- train(log(targets+1)~., data=train, method="rf", trControl=control, tuneGrid=grid, ntree = 100, metric="rmse", nodesize=40, sampsize=50000)

test_preds <- (exp(predict(rf_model_caret, newdata=test))-1)^0.998

sample_sub <- data.frame(Id=test_ids, Sales=test_preds)
sample_sub$Sales[test_open == 0] <- 0
write.csv(sample_sub,"test_sub8_rf.csv", row.names=FALSE)
# LB Score: 0.13441... Not the improvement I had expected...



# Try another submission with new RF model
control <- trainControl(method = "none",
                        number = 1,
                        verboseIter = TRUE,
                        repeats = 1
)

grid = expand.grid( mtry=c(12) )

set.seed(12)
rf_model_caret <- train(log(targets+1)~., data=train, method="rf", trControl=control, tuneGrid=grid, ntree = 200, metric="rmse", nodesize=50, sampsize=200000)

test_preds <- (exp(predict(rf_model_caret, newdata=test))-1)

sample_sub <- data.frame(Id=test_ids, Sales=test_preds)
sample_sub$Sales[test_open == 0] <- 0
write.csv(sample_sub,"test_sub9_rf.csv", row.names=FALSE)
# Score: 0.13404, Train time 1hour 5min


# Try another submission with new RF model only training on months 8,9
control <- trainControl(method = "none",
                        number = 1,
                        verboseIter = TRUE,
                        repeats = 1
)

grid = expand.grid( mtry=c(12) )

set.seed(12)
rf_model_caret <- train(log(targets+1)~., data=train, method="rf", trControl=control, tuneGrid=grid, ntree = 50, metric="rmse", nodesize=6, sampsize=10000)

test_preds <- (exp(predict(rf_model_caret, newdata=test))-1)^0.996

sample_sub <- data.frame(Id=test_ids, Sales=test_preds)
sample_sub$Sales[test_open == 0] <- 0
write.csv(sample_sub,"test_sub10_rf.csv", row.names=FALSE)
#Score:  0.13795 (doesn't help to ingore off months)


# Try another submission with new RF model
control <- trainControl(method = "none",
                        number = 1,
                        verboseIter = TRUE,
                        repeats = 1
)

grid = expand.grid( mtry=c(11) )

set.seed(10)
rf_model_caret <- train(log(targets+1)~., data=train, method="rf", trControl=control, tuneGrid=grid, ntree = 150, metric="rmse", nodesize=5, sampsize=50000)

test_preds <- (exp(predict(rf_model_caret, newdata=test))-1)^0.998

sample_sub <- data.frame(Id=test_ids, Sales=test_preds)
sample_sub$Sales[test_open == 0] <- 0
write.csv(sample_sub,"test_sub11_rf.csv", row.names=FALSE)
# LB Score: 0.13224  ... holdout validation performance not tracking test set performance well


# Try another submission with new RF model
control <- trainControl(method = "none",
                        number = 1,
                        verboseIter = TRUE,
                        repeats = 1
)

grid = expand.grid( mtry=c(11) )

set.seed(10)
rf_model_caret <- train(log(targets+1)~., data=train, method="rf", trControl=control, tuneGrid=grid, ntree = 150, metric="rmse", nodesize=5, sampsize=50000)

test_preds <- (exp(predict(rf_model_caret, newdata=test))-1)^0.998

sample_sub <- data.frame(Id=test_ids, Sales=test_preds)
sample_sub$Sales[test_open == 0] <- 0
write.csv(sample_sub,"test_sub11_rf.csv", row.names=FALSE)
# LB Score: 0.13224  ... holdout validation performance not tracking test set performance well



# Try another submission with new RF model, fewer features
control <- trainControl(method = "none",
                        number = 1,
                        verboseIter = TRUE,
                        repeats = 1
)

grid = expand.grid( mtry=c(10) )

set.seed(10)
rf_model_caret <- train(log(targets+1)~., data=train, method="rf", trControl=control, tuneGrid=grid, ntree = 50, metric="rmse", nodesize=5, sampsize=100000)

test_preds <- (exp(predict(rf_model_caret, newdata=test))-1)

sample_sub <- data.frame(Id=test_ids, Sales=test_preds)
sample_sub$Sales[test_open == 0] <- 0
write.csv(sample_sub,"test_sub12_rf.csv", row.names=FALSE)
# Score: 0.14615 Train time 1hour.. how are the leaderboard random forest examples getting 
# 0.12 with the same features...





# My XGB attempts...

set.seed(121)

train_m = matrix(as.numeric(data.matrix(train_valid)),ncol=ncol(train))
valid_m = matrix(as.numeric(data.matrix(valid)),ncol=ncol(valid))

set.seed(13)

xgb_model = xgboost(data=train_m, label=log(t_valid_targets+1), nrounds=500, verbose=1, eta=0.1, gamma=0.1, max_depth=12, min_child_weight=1, subsample=0.70, colsample_bytree = 0.7, objective="reg:linear", feval=RMSPE)

xgb_preds = exp(predict(xgb_model, valid_m))-1
xgb_preds = ifelse(xgb_preds<0,0,xgb_preds)

xgb_pred2 = (xgb_preds-95)^1.0007
  
RMSPE(valid_targets, xgb_pred2)


# validation: 0.1120573 on nrounds=500, verbose=1, eta=0.1, gamma=0.1 ^ 0.998 with log target adjustment



set.seed(121)

train_m = matrix(as.numeric(data.matrix(train)),ncol=ncol(train))
test_m = matrix(as.numeric(data.matrix(test)),ncol=ncol(valid))

set.seed(13)

xgb_model = xgboost(data=train_m, label=log(targets+1), nrounds=500, verbose=1, eta=0.1, gamma=0.1, max_depth=12, min_child_weight=1, subsample=0.70, colsample_bytree = 0.7, objective="reg:linear", feval=RMSPE)

xgb_preds = exp(predict(xgb_model, test_m))-1
xgb_preds = ifelse(xgb_preds<0,0,xgb_preds)

xgb_pred2 = xgb_preds^0.998

sample_sub <- data.frame(Id=test_ids, Sales=xgb_pred2)
write.csv(sample_sub,"test_xgb1.csv", row.names=FALSE)





#Sub 0.11514, about 1% worse than example