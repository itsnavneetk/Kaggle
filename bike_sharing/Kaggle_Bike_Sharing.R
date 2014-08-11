
#R code for Kaggle Competition-Bike Sharing Demand
#https://www.kaggle.com/c/bike-sharing-demand

#Date submission fomrat in Excel = yyyy-mm-dd hh:mm:ss

library(rpart)
library(rpart.plot)
library(caret)
library(e1071)
library(randomForest)

#Loading in data and preprocessing------------------
train = read.csv("bike_train.csv")
test = read.csv("bike_test.csv")

test$season = as.factor(test$season)
test$holiday= as.factor(test$holiday)
test$weather= as.factor(test$weather)
test$workingday= as.factor(test$workingday)

train$season = as.factor(train$season)
train$holiday= as.factor(train$holiday)
train$weather= as.factor(train$weather)
train$workingday= as.factor(train$workingday)

#Create time variables for the hour of the day and add them to train and test
time = strftime(as.POSIXlt(train$datetime), format = "%H")
time = as.factor(time)
train = cbind(train,time)

time = strftime(as.POSIXlt(test$datetime), format = "%H")
time = as.factor(time)
test = cbind(test,time)

#Create Year Variable and add it
year = strftime(as.POSIXlt(train$datetime), format = "%Y")
year = as.factor(year)
train = cbind(train,year)

year = strftime(as.POSIXlt(test$datetime), format = "%Y")
year = as.factor(year)
test = cbind(test,year)

#Create Month Variable and add it
month = strftime(as.POSIXlt(train$datetime), format = "%m")
month = as.factor(month)
train = cbind(train,month)

month = strftime(as.POSIXlt(test$datetime), format = "%m")
month = as.factor(month)
test = cbind(test,month)

#Create Day of Week Variable and add it
day = strftime(as.POSIXlt(train$datetime), format = "%a")
day = as.factor(day)
train = cbind(train,day)

day = strftime(as.POSIXlt(test$datetime), format = "%a")
day = as.factor(mday)
test = cbind(test,day)

test$humidity = as.double(test$humidity)
train$humidity = as.double(train$humidity)

test$time = as.double(test$time)
train$time = as.double(train$time)

#Set bad atemp values equal to temp for training set
train[8992:9015,]$atemp = train[8992:9015,]$temp+1.5


#----Day 1 (time data not included)--------------------------------------------------------

#Submission 1 Code- Simple Regression
log_model_casual = glm(casual~ season+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train)
log_model_registered = glm(registered~ season+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train)

casual_log_testpred = predict(log_model_casual, newdata=test)
registered_log_testpred = predict(log_model_registered, newdata=test)

casual_log_testpred = pmax(casual_log_testpred,0)
registered_log_testpred = pmax(registered_log_testpred,0)

combined_log_testpred = casual_log_testpred+registered_log_testpred

#Submission 1
submission1 = data.frame(datetime = test$datetime, count = combined_log_testpred)

write.csv(submission1, "bike_submission1.csv", row.names=FALSE)
# Score = 1.37132

#Submission 2 Code - Classification and Regressin Tree(CART)

casual_tree = rpart(casual~ season+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train,minbucket=50)
registered_tree = rpart(registered~ season+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train,minbucket=50)

casual_tree_pred = predict(casual_tree, newdata=test)
registered_tree_pred = predict(registered_tree, newdata=test)

combined_tree_pred = casual_tree_pred+registered_tree_pred

#Submission 2
submission2 = data.frame(datetime = test$datetime, count = combined_tree_pred)

write.csv(submission2, "bike_submission2.csv", row.names=FALSE)
# Score = 1.38580


#Submission 3 Code - Better Classification and Regressin Tree(CART)(Cross-validation)

tr.control = trainControl(method = "cv", number = 10)

# cp values
cp.grid = expand.grid( .cp = (0:10)*0.001)

# Cross-validation
tr_casual = train(casual~ season+holiday+workingday+weather+temp+atemp+humidity+windspeed, data = train, method = "rpart", trControl = tr.control, tuneGrid = cp.grid)
tr_registered = train(registered~ season+holiday+workingday+weather+temp+atemp+humidity+windspeed, data = train, method = "rpart", trControl = tr.control, tuneGrid = cp.grid)

# Extract tree
casual_tree2 = tr_casual$finalModel
registered_tree2 = tr_registered$finalModel

casual_tree_pred2 = predict(casual_tree2, newdata=test)
registered_tree_pred2 = predict(registered_tree2, newdata=test)

combined_tree_pred2 = casual_tree_pred2+registered_tree_pred2

#Submission 3
submission3 = data.frame(datetime = test$datetime, count = combined_tree_pred2)

write.csv(submission3, "bike_submission3.csv", row.names=FALSE)
# Score = 1.33194


#Submission 4 - Basic Random Forest Model 

casual_forest = randomForest(casual~ season+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train, ntree=200, nodesize=25 )
registered_forest = randomForest(registered~ season+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train, ntree=200, nodesize=25 )

casual_forest_pred = predict(casual_forest, newdata=test)
registered_forest_pred = predict(registered_forest, newdata=test )

combined_forestpred = casual_forest_pred+registered_forest_pred

#Submission 4
submission4 = data.frame(datetime = test$datetime, count = combined_forestpred)

write.csv(submission4, "bike_submission4.csv", row.names=FALSE)
# Score = 1.33776


#----------End First day of Submissions--------------------------------------------------

#---------Start of Second day------------------------------------------------------------

#Some Exploratory analysis on year shows that ridership increased by an average of almost
#100 per hour from 2011 to 2012:
#mean(subset(train, year == "2011")$count)
#[1] 144.2233
#> mean(subset(train, year == "2012")$count)
#[1] 238.5609
#Adding year and time data should improve model accuracy.
#It may improve accuracy to group data into 2 clusters.


#Submission 5 Code- Simple Regression with time variable as factor
log_model_casual = glm(casual~ time+season+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train)
log_model_registered = glm(registered~ time+season+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train)

casual_log_testpred = predict(log_model_casual, newdata=test)
registered_log_testpred = predict(log_model_registered, newdata=test)

casual_log_testpred = pmax(casual_log_testpred,0)
registered_log_testpred = pmax(registered_log_testpred,0)

combined_log_testpred = casual_log_testpred+registered_log_testpred

#Submission 5
submission5 = data.frame(datetime = test$datetime, count = combined_log_testpred)

write.csv(submission5, "bike_submission5.csv", row.names=FALSE)
#Score 0.95570


#Submission 6 - Classification and Regressin Tree with Time added

casual_tree = rpart(casual~ time+season+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train,minbucket=20,cp=0.0001)
registered_tree = rpart(registered~ time+season+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train,minbucket=20,cp=0.0001)

casual_tree_pred = predict(casual_tree, newdata=test)
registered_tree_pred = predict(registered_tree, newdata=test)

combined_tree_pred = casual_tree_pred+registered_tree_pred


#Submission 6
submission6 = data.frame(datetime = test$datetime, count = combined_tree_pred)

write.csv(submission6, "bike_submission6.csv", row.names=FALSE)
#Score 0.57013


#Submission 7 - Random Forest Model + Time

casual_forest = randomForest(casual~ time+season+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train, ntree=200, nodesize=20 )
registered_forest = randomForest(registered~ time+season+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train, ntree=200, nodesize=20 )

casual_forest_pred = predict(casual_forest, newdata=test)
registered_forest_pred = predict(registered_forest, newdata=test )

combined_forestpred = casual_forest_pred+registered_forest_pred

#Submission 7
submission7 = data.frame(datetime = test$datetime, count = combined_forestpred)

write.csv(submission7, "bike_submission7.csv", row.names=FALSE)
#Score 0.60781


#Submission 8 - Classification and Regressin Tree with Time, Year and Month added

casual_tree = rpart(casual~ year+month+time+season+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train,minbucket=20,cp=0.0001)
registered_tree = rpart(registered~ year+month+time+season+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train,minbucket=20,cp=0.0001)

casual_tree_pred = predict(casual_tree, newdata=test)
registered_tree_pred = predict(registered_tree, newdata=test)

combined_tree_pred = casual_tree_pred+registered_tree_pred


#Submission 8
submission8 = data.frame(datetime = test$datetime, count = combined_tree_pred)

write.csv(submission8, "bike_submission8.csv", row.names=FALSE)
#Score 0.54666 (BEST SO FAR)


#Submission 9 - Random Forest Model with year month and hour

casual_forest = randomForest(casual~ year+month+time+season+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train, ntree=500, nodesize=5 )
registered_forest = randomForest(registered~ year+month+time+season+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train, ntree=500, nodesize=5 )

casual_forest_pred = predict(casual_forest, newdata=test)
registered_forest_pred = predict(registered_forest, newdata=test )

combined_forestpred = casual_forest_pred+registered_forest_pred

#Submission 9
submission9 = data.frame(datetime = test$datetime, count = combined_forestpred)

write.csv(submission9, "bike_submission9.csv", row.names=FALSE)
#Score0.64129

#----------End Second day of Submissions--------------------------------------------------

#---------Start of Third day------------------------------------------------------------

#I realized I hadn't added month data correctly. (uppercase M=minutes)
#I have corrected this error.

# Starting today with some exploratory Analysis:
# splom(train[ ,6:9])
# subset(train, (temp-atemp) > 7)
# It is clear that the training set has several points where temp and atemp don't match up
# as they should. Likely bad data on the atemp variable.
# splom(test[ ,6:9]) shows no such outliers in the test set

# I will make a rough correction to these values by setting them equal to temp+1.5
# train[8992:9015,]$atemp = train[8992:9015,]$temp+1.5
# atemp and temp have 99 percent correlation:
# cor(train[ ,6:9])
# I wil consider dropping one.
# Season and month are also overlapping variables.

#Submission 10 - Classification and Regressin Tree with better data

casual_tree = rpart(casual~ year+month+time+season+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train,minbucket=20,cp=0.0001)
registered_tree = rpart(registered~ year+month+time+season+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train,minbucket=20,cp=0.0001)

casual_tree_pred = predict(casual_tree, newdata=test)
registered_tree_pred = predict(registered_tree, newdata=test)

combined_tree_pred = casual_tree_pred+registered_tree_pred


#Submission 10
submission10 = data.frame(datetime = test$datetime, count = combined_tree_pred)

write.csv(submission10, "bike_submission10.csv", row.names=FALSE)
#Score= 0.58481 Cleaning data and including month made results worse...


#Submission 11 - Classification and Regressin Tree dropping temp and season

casual_tree = rpart(casual~ year+month+time+holiday+workingday+weather+atemp+humidity+windspeed, data=train,minbucket=20,cp=0.0001)
registered_tree = rpart(registered~ year+month+time+holiday+workingday+weather+atemp+humidity+windspeed, data=train,minbucket=20,cp=0.0001)

casual_tree_pred = predict(casual_tree, newdata=test)
registered_tree_pred = predict(registered_tree, newdata=test)

combined_tree_pred = casual_tree_pred+registered_tree_pred


#Submission 11
submission11 = data.frame(datetime = test$datetime, count = combined_tree_pred)

write.csv(submission11, "bike_submission11.csv", row.names=FALSE)
#Score= 0.58692 Excluding temp and season had small negative effect.


#Submission 12, 13 - CART on 2 clusters, one for each year

train2011 = subset(train, year=="2011")
train2012 = subset(train, year=="2012")

test2011 = subset(test, year=="2011")
test2012 = subset(test, year=="2012")

casual_tree2011 = rpart(casual~ season+time+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train2011,minbucket=10,cp=0.00001)
registered_tree2011 = rpart(registered~ season+time+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train2011,minbucket=10,cp=0.00001)
casual_tree2012 = rpart(casual~ season+time+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train2012,minbucket=10,cp=0.00001)
registered_tree2012 = rpart(registered~ season+time+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train2012,minbucket=10,cp=0.00001)

casual_tree_pred2011 = predict(casual_tree2011, newdata=test2011)
registered_tree_pred2011 = predict(registered_tree2011, newdata=test2011)
casual_tree_pred2012 = predict(casual_tree2012, newdata=test2012)
registered_tree_pred2012 = predict(registered_tree2012, newdata=test2012)

combined_tree_pred2011 = data.frame(datetime = test2011$datetime, count = casual_tree_pred2011+registered_tree_pred2011)
combined_tree_pred2012 = data.frame(datetime = test2012$datetime, count = casual_tree_pred2012+registered_tree_pred2012)


#Submission 12
submission12 = rbind(combined_tree_pred2011,combined_tree_pred2012)

write.csv(submission12, "bike_submission12.csv", row.names=FALSE)
#Score 0.53182 (Best so far)

#tweaked CART cp and minbucket values for a closer fit to the training set.
submission13 = rbind(combined_tree_pred2011,combined_tree_pred2012)
write.csv(submission13, "bike_submission13.csv", row.names=FALSE)
#Score 0.46072 (Best so far Moved up 82 positions. Closer fit seems to work better...)


#Checkng importance of variables to models...
casual_tree2011$variable.importance/sum(casual_tree2011$variable.importance)
casual_tree2012$variable.importance/sum(casual_tree2012$variable.importance)
registered_tree2011$variable.importance/sum(registered_tree2011$variable.importance)
registered_tree2012$variable.importance/sum(registered_tree2012$variable.importance)
#Temp and atemp are most important for casual riders followed closely by Time
#Time is the most important measure for registered users.

#---------End of Third day------------------------------------------------------------

#---------Start of fourth day---------------------------------------------------------



#Submission 14. Using same clusters with different min terminal node size

train2011 = subset(train, year=="2011")
train2012 = subset(train, year=="2012")

test2011 = subset(test, year=="2011")
test2012 = subset(test, year=="2012")

casual_tree2011 = rpart(casual~ season+time+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train2011,minbucket=5,cp=0.00001)
registered_tree2011 = rpart(registered~ season+time+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train2011,minbucket=10,cp=0.00001)
casual_tree2012 = rpart(casual~ season+time+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train2012,minbucket=5,cp=0.00001)
registered_tree2012 = rpart(registered~ season+time+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train2012,minbucket=10,cp=0.00001)

casual_tree_pred2011 = predict(casual_tree2011, newdata=test2011)
registered_tree_pred2011 = predict(registered_tree2011, newdata=test2011)
casual_tree_pred2012 = predict(casual_tree2012, newdata=test2012)
registered_tree_pred2012 = predict(registered_tree2012, newdata=test2012)

combined_tree_pred2011 = data.frame(datetime = test2011$datetime, count = casual_tree_pred2011+registered_tree_pred2011)
combined_tree_pred2012 = data.frame(datetime = test2012$datetime, count = casual_tree_pred2012+registered_tree_pred2012)


submission14 = rbind(combined_tree_pred2011,combined_tree_pred2012)

write.csv(submission14, "bike_submission14.csv", row.names=FALSE)
#Score 0.45974  (Slight improvement)

#Submission 15 - Trying Random Forest again...

train2011 = subset(train, year=="2011")
train2012 = subset(train, year=="2012")

test2011 = subset(test, year=="2011")
test2012 = subset(test, year=="2012")

set.seed(121)
casual_tree2011 = randomForest(casual~ season+time+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train2011,ntree=200, nodesize=5)
registered_tree2011 = randomForest(registered~ season+time+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train2011,ntree=200, nodesize=5 )
casual_tree2012 = randomForest(casual~ season+time+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train2012,ntree=200, nodesize=5 )
registered_tree2012 = randomForest(registered~ season+time+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train2012,ntree=200, nodesize=5  )

casual_tree_pred2011 = predict(casual_tree2011, newdata=test2011)
registered_tree_pred2011 = predict(registered_tree2011, newdata=test2011)
casual_tree_pred2012 = predict(casual_tree2012, newdata=test2012)
registered_tree_pred2012 = predict(registered_tree2012, newdata=test2012)

combined_tree_pred2011 = data.frame(datetime = test2011$datetime, count = casual_tree_pred2011+registered_tree_pred2011)
combined_tree_pred2012 = data.frame(datetime = test2012$datetime, count = casual_tree_pred2012+registered_tree_pred2012)


#Submission 15
submission15 = rbind(combined_tree_pred2011,combined_tree_pred2012)

write.csv(submission15, "bike_submission15.csv", row.names=FALSE)
#Score 0.56998

#Submission 16. Using same clusters with different smaller cp value and smaller end node size

train2011 = subset(train, year=="2011")
train2012 = subset(train, year=="2012")

test2011 = subset(test, year=="2011")
test2012 = subset(test, year=="2012")

casual_tree2011 = rpart(casual~ season+time+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train2011,minbucket=5,cp=0.00022)
registered_tree2011 = rpart(registered~ season+time+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train2011,minbucket=5,cp=0.000022)
casual_tree2012 = rpart(casual~ season+time+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train2012,minbucket=5,cp=0.000022)
registered_tree2012 = rpart(registered~ season+time+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train2012,minbucket=5,cp=0.000022)

casual_tree_pred2011 = predict(casual_tree2011, newdata=test2011)
registered_tree_pred2011 = predict(registered_tree2011, newdata=test2011)
casual_tree_pred2012 = predict(casual_tree2012, newdata=test2012)
registered_tree_pred2012 = predict(registered_tree2012, newdata=test2012)

combined_tree_pred2011 = data.frame(datetime = test2011$datetime, count = casual_tree_pred2011+registered_tree_pred2011)
combined_tree_pred2012 = data.frame(datetime = test2012$datetime, count = casual_tree_pred2012+registered_tree_pred2012)


submission16 = rbind(combined_tree_pred2011,combined_tree_pred2012)

write.csv(submission16, "bike_submission16.csv", row.names=FALSE)
#Score 0.44835 (Another improvement, moved up to #45 on leaderboard)

#Submission 17. Using smaller cp and nodesize (let's overfit!)

train2011 = subset(train, year=="2011")
train2012 = subset(train, year=="2012")

test2011 = subset(test, year=="2011")
test2012 = subset(test, year=="2012")

casual_tree2011 = rpart(casual~ season+time+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train2011,minbucket=3,cp=0.000000001)
registered_tree2011 = rpart(registered~ season+time+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train2011,minbucket=3,cp=0.000000001)
casual_tree2012 = rpart(casual~ season+time+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train2012,minbucket=3,cp=0.000000001)
registered_tree2012 = rpart(registered~ season+time+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train2012,minbucket=3,cp=0.000000001)

casual_tree_pred2011 = predict(casual_tree2011, newdata=test2011)
registered_tree_pred2011 = predict(registered_tree2011, newdata=test2011)
casual_tree_pred2012 = predict(casual_tree2012, newdata=test2012)
registered_tree_pred2012 = predict(registered_tree2012, newdata=test2012)

combined_tree_pred2011 = data.frame(datetime = test2011$datetime, count = casual_tree_pred2011+registered_tree_pred2011)
combined_tree_pred2012 = data.frame(datetime = test2012$datetime, count = casual_tree_pred2012+registered_tree_pred2012)


submission17 = rbind(combined_tree_pred2011,combined_tree_pred2012)

write.csv(submission17, "bike_submission17.csv", row.names=FALSE)
#Score 0.46366 (worse, overfit likely)

#Submission 18. Using smaller cp and nodesize (let's overfit!)

train2011 = subset(train, year=="2011")
train2012 = subset(train, year=="2012")

test2011 = subset(test, year=="2011")
test2012 = subset(test, year=="2012")

casual_tree2011 = rpart(casual~ season+time+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train2011,minbucket=5,cp=0.00000001)
registered_tree2011 = rpart(registered~ season+time+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train2011,minbucket=5,cp=0.00000001)
casual_tree2012 = rpart(casual~ season+time+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train2012,minbucket=5,cp=0.000000001)
registered_tree2012 = rpart(registered~ season+time+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train2012,minbucket=5,cp=0.00000001)

casual_tree_pred2011 = predict(casual_tree2011, newdata=test2011)
registered_tree_pred2011 = predict(registered_tree2011, newdata=test2011)
casual_tree_pred2012 = predict(casual_tree2012, newdata=test2012)
registered_tree_pred2012 = predict(registered_tree2012, newdata=test2012)

combined_tree_pred2011 = data.frame(datetime = test2011$datetime, count = casual_tree_pred2011+registered_tree_pred2011)
combined_tree_pred2012 = data.frame(datetime = test2012$datetime, count = casual_tree_pred2012+registered_tree_pred2012)


submission18 = rbind(combined_tree_pred2011,combined_tree_pred2012)

write.csv(submission18, "bike_submission18.csv", row.names=FALSE)

#---------End of Fourth day------------------------------------------------------------

#---------Start of Fifth day----------------------------------------------------------
#Going to start trying some different methods.

#function to noramlize data. Transforms data frame double values
#into normalized versions 
normalize_doubles <- function(dframe){
  lframe = as.list(dframe)
  for (column in names(lframe)){

    if (typeof(lframe[[column]]) =="double"){

      colMAX = max(lframe[[column]])
      colMin = min(lframe[[column]])

      newcolumn = (lframe[[column]]-colMin)/(colMAX-colMin)
      lframe[[column]] = newcolumn
      }
  }
  return (data.frame(lframe));
}

test = normalize_doubles(test)
train = normalize_doubles(train)


#Submission 19-21. K-Nearest Neighbors--

casual_tree2011 = knnreg(casual~ time+temp, data=train2011, k = 2)
registered_tree2011 = knnreg(registered~ time+temp, data=train2011, k = 2)
casual_tree2012 = knnreg(casual~ time+temp, data=train2012, k = 2)
registered_tree2012 = knnreg(registered~ time+temp, data=train2012, k = 2)

casual_tree_pred2011 = predict(casual_tree2011, newdata=test2011)
registered_tree_pred2011 = predict(registered_tree2011, newdata=test2011)
casual_tree_pred2012 = predict(casual_tree2012, newdata=test2012)
registered_tree_pred2012 = predict(registered_tree2012, newdata=test2012)

predictscas2011 = casual_tree_pred2011 %*% c(1:ncol(casual_tree_pred2011))
predictsreg2011 = registered_tree_pred2011 %*% c(1:ncol(registered_tree_pred2011))
predictscas2012 = casual_tree_pred2012 %*% c(1:ncol(casual_tree_pred2012))
predictsreg2012 = registered_tree_pred2012 %*% c(1:ncol(registered_tree_pred2012))

combined_tree_pred2011 = data.frame(datetime = test2011$datetime, count = predictscas2011+predictsreg2011)
combined_tree_pred2012 = data.frame(datetime = test2012$datetime, count = predictscas2012+predictsreg2012)


submission21 = rbind(combined_tree_pred2011,combined_tree_pred2012)

write.csv(submission21, "bike_submission21.csv", row.names=FALSE)

#KNN best result of 3 attempts = 0.72329. KNN does not seem to be able to handle many
#variables/ high complexity well

#---------End of Fifth day------------------------------------------------------------

#---------Start of Sixth day----------------------------------------------------------


#Submission 22- Expirmenting with caret package...
set.seed(121)


rGrid = data.frame(neurons=c(5,10,15))

casual_tree2011 = train(casual~ day+season+time+holiday+workingday+weather+
                              temp+atemp+humidity+windspeed, data=train2011, method = 'brnn',
                            tuneGrid = rGrid)
casual_tree2011

registered_tree2011 = train(registered~ day+season+time+holiday+workingday+weather+
                          temp+atemp+humidity+windspeed, data=train2011, method = 'rpart',
                        tuneGrid = rGrid)


casual_tree2012 = train(casual~ day+season+time+holiday+workingday+weather+
                          temp+atemp+humidity+windspeed, data=train2012, method = 'rpart',
                        tuneGrid = rGrid)

registered_tree2012 = train(registered~ day+season+time+holiday+workingday+weather+
                              temp+atemp+humidity+windspeed, data=train2012, method = 'rpart',
                            tuneGrid = rGrid)

casual_tree_pred2011 = predict(casual_tree2011, newdata=test2011)
registered_tree_pred2011 = predict(registered_tree2011,   newdata=test2011)
casual_tree_pred2012 = predict(casual_tree2012,   newdata=test2012)
registered_tree_pred2012 = predict(registered_tree2012,   newdata=test2012)

combined_tree_pred2011 = data.frame(datetime = test2011$datetime, count = casual_tree_pred2011+registered_tree_pred2011)
combined_tree_pred2012 = data.frame(datetime = test2012$datetime, count = casual_tree_pred2012+registered_tree_pred2012)


submission22 = rbind(combined_tree_pred2011,combined_tree_pred2012)
summary(submission22)
summary(train$count)

write.csv(submission22, "bike_submission22.csv", row.names=FALSE)





#Trying best CART model again with day of week added--------------------

casual_tree2011 = rpart(casual~ day+season+time+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train2011,minbucket=8,cp=0.0000005)
registered_tree2011 = rpart(registered~ day+season+time+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train2011,minbucket=8,cp=0.0000005)
casual_tree2012 = rpart(casual~ day+season+time+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train2012,minbucket=8,cp=0.0000005)
registered_tree2012 = rpart(registered~ day+season+time+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=train2012,minbucket=8,cp=0.0000005)

casual_tree_pred2011 = predict(casual_tree2011, newdata=test2011)
registered_tree_pred2011 = predict(registered_tree2011, newdata=test2011)
casual_tree_pred2012 = predict(casual_tree2012, newdata=test2012)
registered_tree_pred2012 = predict(registered_tree2012, newdata=test2012)

combined_tree_pred2011 = data.frame(datetime = test2011$datetime, count = casual_tree_pred2011+registered_tree_pred2011)
combined_tree_pred2012 = data.frame(datetime = test2012$datetime, count = casual_tree_pred2012+registered_tree_pred2012)


submission27 = rbind(combined_tree_pred2011,combined_tree_pred2012)
summary(submission27 )
summary(train$count)

write.csv(submission27, "bike_submission27.csv", row.names=FALSE)
# Score 0.43215 (Moved up 7 spots to #40/315)

---------------------------------------------


