#Predicting elo Kaggle competition
library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)
library(gbm)
options( java.parameters = "-Xmx3g" )
library(extraTrees)
library(ggplot2)

#Data munging and feature extraction performed with python

chess_data = read.csv("score_features3.csv")
train_elo = read.csv("chess_ELO.csv")

#Split into train and test
train = chess_data[1:25000, ]
test =  chess_data[25001:50000, ]

#Turn opening moves into opening strength estimates for both players
#For every 2 moves create an elo estimate (mean ELO of players making those moves)

#First 2 moves
white_open2 = tapply(train_elo$WhiteElo , train$opening_2_moves, mean) 
black_open2 = tapply(train_elo$BlackElo , train$opening_2_moves, mean)

train["white_opening_str2"] = white_open2[train$opening_2_moves]
train["black_opening_str2"] = black_open2[train$opening_2_moves]

test["white_opening_str2"] = white_open2[test$opening_2_moves]
test["black_opening_str2"] = black_open2[test$opening_2_moves]

test$white_opening_str2[is.na(test$white_opening_str2)] =  median(test$white_opening_str2,na.rm = TRUE)
test$black_opening_str2[is.na(test$black_opening_str2)] =  median(test$black_opening_str2,na.rm = TRUE)

#Moves 3 and 4
white_open4 = tapply(train_elo$WhiteElo , train$opening_4_moves, mean) 
black_open4 = tapply(train_elo$BlackElo , train$opening_4_moves, mean)

train["white_opening_str4"] = white_open4[train$opening_4_moves]
train["black_opening_str4"] = black_open4[train$opening_4_moves]

train$white_opening_str4[is.na(train$white_opening_str4)] =  median(train$white_opening_str4)
train$black_opening_str4[is.na(train$black_opening_str4)] =  median(train$black_opening_str4)

test["white_opening_str4"] = white_open4[test$opening_4_moves]
test["black_opening_str4"] = black_open4[test$opening_4_moves]

test$white_opening_str4[is.na(test$white_opening_str4)] =  median(test$white_opening_str4,na.rm = TRUE)
test$black_opening_str4[is.na(test$black_opening_str4)] =  median(test$black_opening_str4,na.rm = TRUE)

#Moves 5 and 6
white_open6 = tapply(train_elo$WhiteElo , train$opening_6_moves, mean) 
black_open6 = tapply(train_elo$BlackElo , train$opening_6_moves, mean)

train["white_opening_str6"] = white_open6[train$opening_6_moves]
train["black_opening_str6"] = black_open6[train$opening_6_moves]

train$white_opening_str6[is.na(train$white_opening_str6)] =  median(train$white_opening_str6)
train$black_opening_str6[is.na(train$black_opening_str6)] =  median(train$black_opening_str6)

test["white_opening_str6"] = white_open6[test$opening_6_moves]
test["black_opening_str6"] = black_open6[test$opening_6_moves]

test$white_opening_str6[is.na(test$white_opening_str6)] =  median(test$white_opening_str6,na.rm = TRUE)
test$black_opening_str6[is.na(test$black_opening_str6)] =  median(test$black_opening_str6,na.rm = TRUE)

#8-9
white_open8 = tapply(train_elo$WhiteElo , train$opening_8_moves, mean) 
black_open8 = tapply(train_elo$BlackElo , train$opening_8_moves, mean)

train["white_opening_str8"] = white_open8[train$opening_8_moves]
train["black_opening_str8"] = black_open8[train$opening_8_moves]

train$white_opening_str8[is.na(train$white_opening_str8)] =  median(train$white_opening_str8)
train$black_opening_str8[is.na(train$black_opening_str8)] =  median(train$black_opening_str8)

test["white_opening_str8"] = white_open8[test$opening_8_moves]
test["black_opening_str8"] = black_open8[test$opening_8_moves]

test$white_opening_str8[is.na(test$white_opening_str8)] =  median(test$white_opening_str8,na.rm = TRUE)
test$black_opening_str8[is.na(test$black_opening_str8)] =  median(test$black_opening_str8,na.rm = TRUE)

#3-4
white_open3 = tapply(train_elo$WhiteElo , train$opening_3_moves, mean)
black_open3 = tapply(train_elo$BlackElo , train$opening_3_moves, mean)

train["white_opening_str3"] = white_open3[train$opening_3_moves]
train["black_opening_str3"] = black_open3[train$opening_3_moves]

train$white_opening_str3[is.na(train$white_opening_str3)] =  median(train$white_opening_str3)
train$black_opening_str3[is.na(train$black_opening_str3)] =  median(train$black_opening_str3)

test["white_opening_str3"] = white_open3[test$opening_3_moves]
test["black_opening_str3"] = black_open3[test$opening_3_moves]

test$white_opening_str3[is.na(test$white_opening_str3)] =  median(test$white_opening_str3,na.rm = TRUE)
test$black_opening_str3[is.na(test$black_opening_str3)] =  median(test$black_opening_str3,na.rm = TRUE)
  
#5-6
white_open5 = tapply(train_elo$WhiteElo , train$opening_5_moves, mean)
black_open5 = tapply(train_elo$BlackElo , train$opening_5_moves, mean)

train["white_opening_str5"] = white_open5[train$opening_5_moves]
train["black_opening_str5"] = black_open5[train$opening_5_moves]

train$white_opening_str5[is.na(train$white_opening_str5)] =  median(train$white_opening_str5)
train$black_opening_str5[is.na(train$black_opening_str5)] =  median(train$black_opening_str5)

test["white_opening_str5"] = white_open5[test$opening_5_moves]
test["black_opening_str5"] = black_open5[test$opening_5_moves]

test$white_opening_str5[is.na(test$white_opening_str5)] =  median(test$white_opening_str5,na.rm = TRUE)
test$black_opening_str5[is.na(test$black_opening_str5)] =  median(test$black_opening_str5,na.rm = TRUE)

#7-8
white_open7 = tapply(train_elo$WhiteElo , train$opening_7_moves, mean)
black_open7 = tapply(train_elo$BlackElo , train$opening_7_moves, mean)

train["white_opening_str7"] = white_open7[train$opening_7_moves]
train["black_opening_str7"] = black_open7[train$opening_7_moves]

train$white_opening_str7[is.na(train$white_opening_str7)] =  median(train$white_opening_str7)
train$black_opening_str7[is.na(train$black_opening_str7)] =  median(train$black_opening_str7)

test["white_opening_str7"] = white_open7[test$opening_7_moves]
test["black_opening_str7"] = black_open7[test$opening_7_moves]

test$white_opening_str7[is.na(test$white_opening_str7)] =  median(test$white_opening_str7,na.rm = TRUE)
test$black_opening_str7[is.na(test$black_opening_str7)] =  median(test$black_opening_str7,na.rm = TRUE)

#1-4
white_open14 = tapply(train_elo$WhiteElo , train$opening_14_moves, mean)
black_open14 = tapply(train_elo$BlackElo , train$opening_14_moves, mean)

train["white_opening_str14"] = white_open14[train$opening_14_moves]
train["black_opening_str14"] = black_open14[train$opening_14_moves]

train$white_opening_str14[is.na(train$white_opening_str14)] =  median(train$white_opening_str14)
train$black_opening_str14[is.na(train$black_opening_str14)] =  median(train$black_opening_str14)

test["white_opening_str14"] = white_open14[test$opening_14_moves]
test["black_opening_str14"] = black_open14[test$opening_14_moves]

test$white_opening_str14[is.na(test$white_opening_str14)] =  median(test$white_opening_str14,na.rm = TRUE)
test$black_opening_str14[is.na(test$black_opening_str14)] =  median(test$black_opening_str14,na.rm = TRUE)

#2-5
white_open25 = tapply(train_elo$WhiteElo , train$opening_25_moves, mean)
black_open25 = tapply(train_elo$BlackElo , train$opening_25_moves, mean)

train["white_opening_str25"] = white_open25[train$opening_25_moves]
train["black_opening_str25"] = black_open25[train$opening_25_moves]

train$white_opening_str25[is.na(train$white_opening_str25)] =  median(train$white_opening_str25)
train$black_opening_str25[is.na(train$black_opening_str25)] =  median(train$black_opening_str25)

test["white_opening_str25"] = white_open25[test$opening_25_moves]
test["black_opening_str25"] = black_open25[test$opening_25_moves]

test$white_opening_str25[is.na(test$white_opening_str25)] =  median(test$white_opening_str25,na.rm = TRUE)
test$black_opening_str25[is.na(test$black_opening_str25)] =  median(test$black_opening_str25,na.rm = TRUE)


train_partial = train[0:15000,]
validation = train[15001:25000,]

train_partial_elo = train_elo[0:15000,]
valid_elo = train_elo[15001:25000,]



#Exploratory analsysis
summary(train)
summary(train_elo)

ggplot(data=train,aes(average_score,game_length,colour = result))+geom_point(alpha = 0.2)

ggplot(data=train,aes(ending_score,game_length,colour = result))+geom_point(alpha = 0.2)

ggplot(data=train_elo,aes(WhiteElo,BlackElo,colour = train$result))+geom_point(alpha = 0.2)

table(train$result,train$opening_2_moves)

ggplot(data=train_elo,aes(WhiteElo,train$opening_2_moves))+geom_boxplot()

cor(train[ ,c(3:9)])
splom(train[ ,c(3:9)])

#Basic regression - Doesn't appear to improve upon mean score benchmark
linear_mod1 = lm(train_elo$WhiteElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result,data=train)
linear_mod2 = lm(train_elo$BlackElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result,data=train)

summary(linear_mod1)
summary(linear_mod2)

pred1 = predict(linear_mod1, data=train)
mean(abs(train_elo$WhiteElo-pred1))

pred2 = predict(linear_mod1, data=train)
mean(abs(train_elo$BlackElo-pred2))
#---------------------------

#Basic CART 
set.seed(12)
cart_mod1 = rpart(train_elo$WhiteElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result+opening_2_moves+opening_4_moves,data=train, cp=0.00001,maxdepth = 4)
cart_mod2 = rpart(train_elo$BlackElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result+opening_2_moves+opening_4_moves,data=train,cp=0.00001,maxdepth = 4)


pred1 = predict(cart_mod1, newdata=validation)
mean(abs(valid_elo$WhiteElo-pred1))

pred2 = predict(cart_mod2, newdata=validation)
mean(abs(valid_elo$BlackElo-pred2))

#Validation score use this model as baseline submission

pred1 = predict(cart_mod1, newdata=test)

pred2 = predict(cart_mod2, newdata=test)

submission1 = data.frame(Event= test$Event, WhiteElo=pred1 ,BlackElo=pred2 )
write.csv(submission1, "chess_elo_submission1.csv", row.names=FALSE)
#Model scored 205.85236


#Retry with maxdepth = 4 and use all 25000 examples:
set.seed(12)
  
cart_mod1 = rpart(train_elo$WhiteElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result+opening_2_moves+opening_4_moves,data=train, cp=0.00001,maxdepth = 4)
cart_mod2 = rpart(train_elo$BlackElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result+opening_2_moves+opening_4_moves,data=train,cp=0.00001,maxdepth = 4)

pred1 = predict(cart_mod1, newdata=test)

pred2 = predict(cart_mod2, newdata=test)

submission2 = data.frame(Event= test$Event, WhiteElo=pred1 ,BlackElo=pred2 )
write.csv(submission2, "chess_elo_submission2.csv", row.names=FALSE)
#Score improved to 203.24299 with all data and maxdepth = 4

#Testing Random Forests---------------------------------------------
set.seed(12)

rf_mod1 = randomForest(train_partial_elo$WhiteElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result+white_opening_str2+black_opening_str2+white_opening_str4+black_opening_str4+white_opening_str6+black_opening_str6+white_opening_str3+black_opening_str3+white_opening_str5+black_opening_str5+white_opening_str7+black_opening_str7+white_opening_str8+black_opening_str8+white_opening_str14+black_opening_str14,data=train_partial, mtry=4, nodesize=35,ntree=50)

rf_mod2 = randomForest(train_partial_elo$BlackElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result+white_opening_str2+black_opening_str2+white_opening_str4+black_opening_str4+white_opening_str6+black_opening_str6+white_opening_str3+black_opening_str3+white_opening_str5+black_opening_str5+white_opening_str7+black_opening_str7+white_opening_str8+black_opening_str8+white_opening_str14+black_opening_str14,data=train_partial, mtry=4, nodesize=35, ntree=50)

pred1 = predict(rf_mod1, newdata=validation)
mean(abs(valid_elo$WhiteElo-pred1))

pred2 = predict(rf_mod2, newdata=validation)
mean(abs(valid_elo$BlackElo-pred2))

importance(rf_mod1)
importance(rf_mod2)
#Testing Random Forests---------------------------------------------

#Scored 195.5934 and 198.5039 on validation set with ntree 100
#With full training data and ntree 1000 expecting slight improvement on test set Expect submission around 196

set.seed(12)

rf_mod1 = randomForest(train_elo$WhiteElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result,data=train, mtry=2, nodesize=25,ntree=1000)

rf_mod2 = randomForest(train_elo$BlackElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result,data=train, mtry=2, nodesize=25, ntree=1000)

pred1 = predict(rf_mod1, newdata=test)

pred2 = predict(rf_mod2, newdata=test)

submission3 = data.frame(Event= test$Event, WhiteElo=pred1 ,BlackElo=pred2 )
write.csv(submission3, "chess_elo_submission3.csv", row.names=FALSE)

#Runtime:12.5 minutes  
#Score: 197.44481  a bit worse than expected...



#Rerun random Forest Model adding new features of opening strength estimates
#Small validation trials show errors of 188.8904 and 192.2238
#Expecting slighly worse performance on test since not all opening moves
#in test set were in the training set.

set.seed(12)

rf_mod1 = randomForest(train_elo$WhiteElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result+white_opening_str2+black_opening_str2+white_opening_str4+black_opening_str4+white_opening_str6+black_opening_str6,data=train, mtry=3, nodesize=25,ntree=250)

rf_mod2 = randomForest(train_elo$BlackElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result+white_opening_str2+black_opening_str2+white_opening_str4+black_opening_str4+white_opening_str6+black_opening_str6,data=train, mtry=3, nodesize=25, ntree=250)

pred1 = predict(rf_mod1, newdata=test)
summary(pred1)

pred2 = predict(rf_mod2, newdata=test)
summary(pred2)

submission4 = data.frame(Event= test$Event, WhiteElo=pred1 ,BlackElo=pred2 )
write.csv(submission4, "chess_elo_submission4.csv", row.names=FALSE)
#Runtime: 3 minutes
#Score: 192.97163  4th place!

#Adding more game move player skill estimate features...
set.seed(12)

rf_mod1 = randomForest(train_elo$WhiteElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result+white_opening_str2+black_opening_str2+white_opening_str4+black_opening_str4+white_opening_str6+black_opening_str6+white_opening_str3+black_opening_str3+white_opening_str5+black_opening_str5+white_opening_str7+black_opening_str7+white_opening_str8+black_opening_str8+white_opening_str14+black_opening_str14,data=train, mtry=4, nodesize=35,ntree=250)

rf_mod2 = randomForest(train_elo$BlackElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result+white_opening_str2+black_opening_str2+white_opening_str4+black_opening_str4+white_opening_str6+black_opening_str6+white_opening_str3+black_opening_str3+white_opening_str5+black_opening_str5+white_opening_str7+black_opening_str7+white_opening_str8+black_opening_str8+white_opening_str14+black_opening_str14,data=train, mtry=4, nodesize=35, ntree=250)

pred1 = predict(rf_mod1, newdata=test)

pred2 = predict(rf_mod2, newdata=test)


submission9 = data.frame(Event= test$Event, WhiteElo=pred1 ,BlackElo=pred2 )
write.csv(submission9, "chess_elo_submission9.csv", row.names=FALSE)
#Runtime: 4.5 minutes
#Score: 192.57385  slight improvement. opening strength estimates start to overfit training data significantly past move 5-6




#Try gradient boosting
set.seed(12)


tunecontrol = trainControl(method = "repeatedcv",
                           number = 2,
                           repeats = 1
)

tgrid = expand.grid(n.trees = c(100),interaction.depth=c(8) ,shrinkage=c(0.107) )


rf_mod1 = train(train_elo$WhiteElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result+white_opening_str2+black_opening_str2+white_opening_str4+black_opening_str4+white_opening_str6+black_opening_str6+white_opening_str3+black_opening_str3+white_opening_str5+black_opening_str5+white_opening_str7+black_opening_str7+white_opening_str8+black_opening_str8+white_opening_str14+black_opening_str14+white_opening_str25+black_opening_str25, data=train, method= 'gbm',trControl=tunecontrol, tuneGrid=tgrid)


rf_mod2 = train(train_elo$BlackElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result+white_opening_str2+black_opening_str2+white_opening_str4+black_opening_str4+white_opening_str6+black_opening_str6+white_opening_str3+black_opening_str3+white_opening_str5+black_opening_str5+white_opening_str7+black_opening_str7+white_opening_str8+black_opening_str8+white_opening_str14+black_opening_str14+white_opening_str25+black_opening_str25, data=train, method= 'gbm',trControl=tunecontrol, tuneGrid=tgrid)

pred1 = predict(rf_mod1, newdata=test)

pred2 = predict(rf_mod2, newdata=test)

submission11 = data.frame(Event= test$Event, WhiteElo=pred1 ,BlackElo=pred2 )
write.csv(submission11, "chess_elo_submission11.csv", row.names=FALSE)
#Runtime: 2min
#Score: 193.47330


#extraTrees model
set.seed(12)


tunecontrol = trainControl(method = "repeatedcv",
                           number = 2,
                           repeats = 1
)

tgrid = expand.grid(mtry = c(3), numRandomCuts= c(30) )


rf_mod1 = train(train_elo$WhiteElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result+white_opening_str2+black_opening_str2+white_opening_str4+black_opening_str4+white_opening_str6+black_opening_str6+white_opening_str3+black_opening_str3+white_opening_str5+black_opening_str5+white_opening_str7+black_opening_str7+white_opening_str8+black_opening_str8+white_opening_str14+black_opening_str14+white_opening_str25+black_opening_str25, data=train, method= 'extraTrees',trControl=tunecontrol, tuneGrid=tgrid, nodesize =c(30), ntree=c(1000))



rf_mod2 = train(train_elo$BlackElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result+white_opening_str2+black_opening_str2+white_opening_str4+black_opening_str4+white_opening_str6+black_opening_str6+white_opening_str3+black_opening_str3+white_opening_str5+black_opening_str5+white_opening_str7+black_opening_str7+white_opening_str8+black_opening_str8+white_opening_str14+black_opening_str14+white_opening_str25+black_opening_str25, data=train, method= 'extraTrees',trControl=tunecontrol, tuneGrid=tgrid, nodesize =c(50), ntree=c(1000))


pred1 = predict(rf_mod1, newdata=test)

pred2 = predict(rf_mod2, newdata=test)

submission12 = data.frame(Event= test$Event, WhiteElo=pred1 ,BlackElo=pred2 )
write.csv(submission12, "chess_elo_submission12.csv", row.names=FALSE)
#Runtime: 
#Score: 


#Added features for average and median game score gains for each player rerun best RF model with new features
set.seed(12)

rf_mod1 = randomForest(train_elo$WhiteElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result+white_opening_str2+black_opening_str2+white_opening_str4+black_opening_str4+white_opening_str6+black_opening_str6+white_opening_str3+black_opening_str3+white_opening_str5+black_opening_str5+white_opening_str7+black_opening_str7+white_opening_str8+black_opening_str8+white_opening_str14+black_opening_str14+white_opening_str25+black_opening_str25+white_avg_improve+black_avg_improve +white_median_improve + black_median_improve, data= train, mtry=3, ntree=2500, nodesize=30)

rf_mod2 = randomForest(train_elo$BlackElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result+white_opening_str2+black_opening_str2+white_opening_str4+black_opening_str4+white_opening_str6+black_opening_str6+white_opening_str3+black_opening_str3+white_opening_str5+black_opening_str5+white_opening_str7+black_opening_str7+white_opening_str8+black_opening_str8+white_opening_str14+black_opening_str14+white_opening_str25+black_opening_str25+white_avg_improve+black_avg_improve +white_median_improve + black_median_improve, data= train, mtry=3, ntree=2500, nodesize=30)

pred1 = predict(rf_mod1, newdata=test)

pred2 = predict(rf_mod2, newdata=test)


submission14 = data.frame(Event= test$Event, WhiteElo=pred1 ,BlackElo=pred2 )
write.csv(submission14, "chess_elo_submission14.csv", row.names=FALSE)
#Score: 190.60450  3rd place. Would have been 1st yesterday...


#Added features for average score gains for each player for each quarter of the game playe. Added castling features.
set.seed(12)

rf_mod1 = randomForest(train_elo$WhiteElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result+white_opening_str2+black_opening_str2+white_opening_str4+black_opening_str4+white_opening_str6+black_opening_str6+white_opening_str3+black_opening_str3+white_opening_str5+black_opening_str5+white_opening_str7+black_opening_str7+white_opening_str8+black_opening_str8+white_opening_str14+black_opening_str14+white_opening_str25+black_opening_str25+white_avg_improve+black_avg_improve +white_median_improve + black_median_improve +white_q1_improve +
                         white_q2_improve +
                         white_q3_improve +
                         white_q4_improve +
                         black_q1_improve +
                         black_q2_improve +
                         black_q3_improve +
                         black_q4_improve , data= train, mtry=3, ntree=1000, nodesize=30)

rf_mod2 = randomForest(train_elo$BlackElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result+white_opening_str2+black_opening_str2+white_opening_str4+black_opening_str4+white_opening_str6+black_opening_str6+white_opening_str3+black_opening_str3+white_opening_str5+black_opening_str5+white_opening_str7+black_opening_str7+white_opening_str8+black_opening_str8+white_opening_str14+black_opening_str14+white_opening_str25+black_opening_str25+white_avg_improve+black_avg_improve +white_median_improve + black_median_improve +white_q1_improve +
                         white_q2_improve +
                         white_q3_improve +
                         white_q4_improve +
                         black_q1_improve +
                         black_q2_improve +
                         black_q3_improve +
                         black_q4_improve, data= train, mtry=3, ntree=1000, nodesize=30)

pred1 = predict(rf_mod1, newdata=test)

pred2 = predict(rf_mod2, newdata=test)

submission16 = data.frame(Event= test$Event, WhiteElo=pred1 ,BlackElo=pred2 )
write.csv(submission16, "chess_elo_submission16.csv", row.names=FALSE)
#Score: 189.88015

best_rf= read.csv("chess_elo_submission16.csv")
pred1 = best_rf$WhiteElo
pred2 = best_rf$BlackElo

set.seed(12)
gbm1 = gbm(train_elo$WhiteElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result+white_opening_str2+black_opening_str2+white_opening_str4+black_opening_str4+white_opening_str6+black_opening_str6+white_opening_str3+black_opening_str3+white_opening_str5+black_opening_str5+white_opening_str7+black_opening_str7+white_opening_str8+black_opening_str8+white_opening_str14+black_opening_str14+white_opening_str25+black_opening_str25+white_avg_improve+black_avg_improve +white_median_improve + black_median_improve+white_q1_improve +
                white_q2_improve +
                white_q3_improve +
                white_q4_improve +
                black_q1_improve +
                black_q2_improve +
                black_q3_improve +
                black_q4_improve + white_castle_side +
                black_castle_side, data= train, n.trees=70, interaction.depth = 10, shrinkage = 0.1)


pred3 = predict(gbm1, n.trees=70, newdata=test)

set.seed(12)
gbm2 = gbm(train_elo$BlackElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result+white_opening_str2+black_opening_str2+white_opening_str4+black_opening_str4+white_opening_str6+black_opening_str6+white_opening_str3+black_opening_str3+white_opening_str5+black_opening_str5+white_opening_str7+black_opening_str7+white_opening_str8+black_opening_str8+white_opening_str14+black_opening_str14+white_opening_str25+black_opening_str25+white_avg_improve+black_avg_improve +white_median_improve + black_median_improve+white_q1_improve +
                white_q2_improve +
                white_q3_improve +
                white_q4_improve +
                black_q1_improve +
                black_q2_improve +
                black_q3_improve +
                black_q4_improve + white_castle_turn_num +
                black_castle_turn_num +
                white_castle_side +
                black_castle_side, data= train, n.trees=70, interaction.depth = 10, shrinkage = 0.1)

pred4 = predict(gbm2, n.trees=70, newdata=test)

#Ensemble, 50% weight to RF model 50% to gbm
pred5 = (pred1*0.5)+(pred3*0.5)
pred6 = (pred2*0.5)+(pred4*0.5)
  
submission17 = data.frame(Event= test$Event, WhiteElo=pred5 ,BlackElo=pred6 )
write.csv(submission17, "chess_elo_submission17.csv", row.names=FALSE)
#Score: 189.34253      0.025 above 2nd place




#Added several new features. Validation scores with new features approx 186.7
set.seed(12)

#Random Forest white
rf_mod1 = randomForest(train_elo$WhiteElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result+white_opening_str2+black_opening_str2+white_opening_str4+black_opening_str4+white_opening_str6+black_opening_str6+white_opening_str3+black_opening_str3+white_opening_str5+black_opening_str5+white_opening_str7+black_opening_str7+white_opening_str8+black_opening_str8+white_opening_str14+black_opening_str14+white_opening_str25+black_opening_str25+white_avg_improve+black_avg_improve +white_median_improve + black_median_improve+white_q1_improve +white_q2_improve+white_q3_improve+white_q4_improve+black_q1_improve+black_q2_improve+black_q3_improve+black_q4_improve+white_castle_side+black_castle_side+score_stdev+white_5_improve+white_10_improve+white_15_improve+white_20_improve+white_25_improve+white_30_improve+white_35_improve+white_40_improve+white_45_improve+white_50_improve+black_5_improve+black_10_improve+black_15_improve+black_20_improve+black_25_improve+black_30_improve+black_35_improve+black_40_improve+black_45_improve+black_50_improve+white_55_improve+black_55_improve+white_q1_max +white_q2_max +white_q3_max +white_q4_max +black_q1_max +black_q2_max +black_q3_max + black_q4_max+white_q1_min +white_q2_min +white_q3_min+ white_q4_min +black_q1_min + black_q2_min + black_q3_min + black_q4_min, data= train, mtry=4, ntree=1000, nodesize=25)

wpred1 = predict(rf_mod1, newdata=test)


#GBM white
set.seed(12)

gbm_mod1 = gbm(train_elo$WhiteElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result+white_opening_str2+black_opening_str2+white_opening_str4+black_opening_str4+white_opening_str6+black_opening_str6+white_opening_str3+black_opening_str3+white_opening_str5+black_opening_str5+white_opening_str7+black_opening_str7+white_opening_str8+black_opening_str8+white_opening_str14+black_opening_str14+white_opening_str25+black_opening_str25+white_avg_improve+black_avg_improve +white_median_improve + black_median_improve+white_q1_improve +white_q2_improve+white_q3_improve+white_q4_improve+black_q1_improve+black_q2_improve+black_q3_improve+black_q4_improve+white_castle_side+black_castle_side+score_stdev+white_5_improve+white_10_improve+white_15_improve+white_20_improve+white_25_improve+white_30_improve+white_35_improve+white_40_improve+white_45_improve+white_50_improve+black_5_improve+black_10_improve+black_15_improve+black_20_improve+black_25_improve+black_30_improve+black_35_improve+black_40_improve+black_45_improve+black_50_improve+white_55_improve+black_55_improve+white_q1_max +white_q2_max +white_q3_max +white_q4_max +black_q1_max +black_q2_max +black_q3_max + black_q4_max +white_q1_min +white_q2_min +white_q3_min+ white_q4_min +black_q1_min + black_q2_min + black_q3_min + black_q4_min, data= train, n.trees=80, interaction.depth = 10, shrinkage = 0.1)


wpred3 = predict(gbm_mod1, n.trees=80, newdata=test)


#composition of 50% random forest and 50% gmb
wpred4 = (wpred1*0.5)+(wpred3*0.5)



#Models for black elo

set.seed(12)

#Random Forest black
rf_mod2 = randomForest(train_elo$BlackElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result+white_opening_str2+black_opening_str2+white_opening_str4+black_opening_str4+white_opening_str6+black_opening_str6+white_opening_str3+black_opening_str3+white_opening_str5+black_opening_str5+white_opening_str7+black_opening_str7+white_opening_str8+black_opening_str8+white_opening_str14+black_opening_str14+white_opening_str25+black_opening_str25+white_avg_improve+black_avg_improve +white_median_improve + black_median_improve+white_q1_improve +white_q2_improve+white_q3_improve+white_q4_improve+black_q1_improve+black_q2_improve+black_q3_improve+black_q4_improve+white_castle_side+black_castle_side+score_stdev+white_5_improve+white_10_improve+white_15_improve+white_20_improve+white_25_improve+white_30_improve+white_35_improve+white_40_improve+white_45_improve+white_50_improve+black_5_improve+black_10_improve+black_15_improve+black_20_improve+black_25_improve+black_30_improve+black_35_improve+black_40_improve+black_45_improve+black_50_improve+white_55_improve+black_55_improve+white_q1_max +white_q2_max +white_q3_max +white_q4_max +black_q1_max +black_q2_max +black_q3_max + black_q4_max+white_q1_min +white_q2_min +white_q3_min+ white_q4_min +black_q1_min + black_q2_min + black_q3_min + black_q4_min, data= train, mtry=4, ntree=1000, nodesize=25)

bpred1 = predict(rf_mod2, newdata=test)


#Regression model black
linear_mod2 = lm(train_elo$BlackElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result+white_opening_str2+black_opening_str2+white_opening_str4+black_opening_str4+white_opening_str6+black_opening_str6+white_opening_str3+black_opening_str3+white_opening_str5+black_opening_str5+white_opening_str7+black_opening_str7+white_opening_str8+black_opening_str8+white_opening_str14+black_opening_str14+white_opening_str25+black_opening_str25+white_avg_improve+black_avg_improve +white_median_improve + black_median_improve+white_q1_improve + white_q2_improve +white_q3_improve +white_q4_improve +black_q1_improve +black_q2_improve +black_q3_improve +black_q4_improve + white_castle_side + black_castle_side+I(average_score^2)+I(ending_score^2)+I(largest_drop^2)+I(largest_gain^2)+I(max_score^2)+I(min_score^2)+I(white_opening_str25^2)+I(black_opening_str25^2)+I(max_score^2)+I(min_score^2)+I(white_opening_str14^2)+I(black_opening_str14^2)+I(white_opening_str8^2)+I(black_opening_str8^2)+I(average_score*game_length)+I(largest_drop*game_length)+I(min_score*game_length)+I(white_avg_improve^3)+I(black_avg_improve^3)+I(white_q2_improve^2)+I(white_q3_improve^2)+I(black_q2_improve^2)+I(log(game_length))+score_stdev+I(score_stdev^2)+I(score_stdev^3)+I(score_stdev^4)+I(score_stdev^5)+I(score_stdev^6)+I(score_stdev^2*average_score^2)+I(log(game_length)*score_stdev)+white_check_rate+white_5_improve+white_10_improve+white_15_improve+white_20_improve+white_25_improve+white_30_improve+white_35_improve+white_40_improve+white_45_improve+white_50_improve+black_5_improve+black_10_improve+black_15_improve+black_20_improve+black_25_improve+black_30_improve+black_35_improve+black_40_improve+black_45_improve+black_50_improve+white_55_improve+black_55_improve+white_q1_max +white_q2_max +white_q3_max +white_q4_max +black_q1_max +black_q2_max +black_q3_max + black_q4_max+white_q1_min +white_q2_min +white_q3_min+ white_q4_min +black_q1_min + black_q2_min + black_q3_min + black_q4_min, data= train)

bpred2 = predict(linear_mod2, newdata=test)
#Set outlier predictions equal to a fixed max and min
bpred2 = ifelse(bpred2<1700,1700,bpred2)
bpred2 = ifelse(bpred2>2500,2500,bpred2)


#GBM black
set.seed(12)
gbm_mod2 = gbm(train_elo$BlackElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result+white_opening_str2+black_opening_str2+white_opening_str4+black_opening_str4+white_opening_str6+black_opening_str6+white_opening_str3+black_opening_str3+white_opening_str5+black_opening_str5+white_opening_str7+black_opening_str7+white_opening_str8+black_opening_str8+white_opening_str14+black_opening_str14+white_opening_str25+black_opening_str25+white_avg_improve+black_avg_improve +white_median_improve + black_median_improve+white_q1_improve +white_q2_improve+white_q3_improve+white_q4_improve+black_q1_improve+black_q2_improve+black_q3_improve+black_q4_improve+white_castle_side+black_castle_side+score_stdev+white_5_improve+white_10_improve+white_15_improve+white_20_improve+white_25_improve+white_30_improve+white_35_improve+white_40_improve+white_45_improve+white_50_improve+black_5_improve+black_10_improve+black_15_improve+black_20_improve+black_25_improve+black_30_improve+black_35_improve+black_40_improve+black_45_improve+black_50_improve+white_55_improve+black_55_improve+white_q1_max +white_q2_max +white_q3_max +white_q4_max +black_q1_max +black_q2_max +black_q3_max + black_q4_max+white_q1_min +white_q2_min +white_q3_min+ white_q4_min +black_q1_min + black_q2_min + black_q3_min + black_q4_min, data= train, n.trees=80, interaction.depth = 10, shrinkage = 0.1)

bpred3 = predict(gbm_mod2, n.trees=80, newdata=test)


#composition of 45% random forest and 45% gmb and 10% linear reg
bpred4 = (bpred1*0.475)+(bpred3*0.475)+(bpred2*0.05)

submission18 = data.frame(Event= test$Event, WhiteElo=wpred4 ,BlackElo=bpred4 )
write.csv(submission18, "chess_elo_submission18.csv", row.names=FALSE)
#Total runtime: 20min
#Score: 186.464  2nd place. 0.13 below first place.




#Added new variables--game score every 10 moves
set.seed(12)

rf_mod1 = randomForest(train_elo$WhiteElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result+white_opening_str2+black_opening_str2+white_opening_str4+black_opening_str4+white_opening_str6+black_opening_str6+white_opening_str3+black_opening_str3+white_opening_str5+black_opening_str5+white_opening_str7+black_opening_str7+white_opening_str8+black_opening_str8+white_opening_str14+black_opening_str14+white_opening_str25+black_opening_str25+white_avg_improve+black_avg_improve +white_median_improve + black_median_improve+white_q1_improve +white_q2_improve+white_q3_improve+white_q4_improve+black_q1_improve+black_q2_improve+black_q3_improve+black_q4_improve+white_castle_side+black_castle_side+score_stdev+white_5_improve+white_10_improve+white_15_improve+white_20_improve+white_25_improve+white_30_improve+white_35_improve+white_40_improve+white_45_improve+white_50_improve+black_5_improve+black_10_improve+black_15_improve+black_20_improve+black_25_improve+black_30_improve+black_35_improve+black_40_improve+black_45_improve+black_50_improve+white_55_improve+black_55_improve+white_q1_max +white_q2_max +white_q3_max +white_q4_max +black_q1_max +black_q2_max +black_q3_max + black_q4_max+white_q1_min +white_q2_min +white_q3_min+ white_q4_min +black_q1_min + black_q2_min + black_q3_min + black_q4_min+                   game_score10 + game_score20 + game_score30 + game_score40 + game_score50 + game_score60 + game_score70 + game_score80 +  game_score90 + game_score100, data= train, mtry=4, ntree=1000, nodesize=25)

wpred1 = predict(rf_mod1, newdata=test)

set.seed(12)
gbm_mod1 = gbm(train_elo$WhiteElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result+white_opening_str2+black_opening_str2+white_opening_str4+black_opening_str4+white_opening_str6+black_opening_str6+white_opening_str3+black_opening_str3+white_opening_str5+black_opening_str5+white_opening_str7+black_opening_str7+white_opening_str8+black_opening_str8+white_opening_str14+black_opening_str14+white_opening_str25+black_opening_str25+white_avg_improve+black_avg_improve +white_median_improve + black_median_improve+white_q1_improve +white_q2_improve+white_q3_improve+white_q4_improve+black_q1_improve+black_q2_improve+black_q3_improve+black_q4_improve+white_castle_side+black_castle_side+score_stdev+white_5_improve+white_10_improve+white_15_improve+white_20_improve+white_25_improve+white_30_improve+white_35_improve+white_40_improve+white_45_improve+white_50_improve+black_5_improve+black_10_improve+black_15_improve+black_20_improve+black_25_improve+black_30_improve+black_35_improve+black_40_improve+black_45_improve+black_50_improve+white_55_improve+black_55_improve+white_q1_max +white_q2_max +white_q3_max +white_q4_max +black_q1_max +black_q2_max +black_q3_max + black_q4_max +white_q1_min +white_q2_min +white_q3_min+ white_q4_min +black_q1_min + black_q2_min + black_q3_min + black_q4_min+                   game_score10 + game_score20 + game_score30 + game_score40 + game_score50 + game_score60 + game_score70 + game_score80 +  game_score90 + game_score100, data= train, n.trees=80, interaction.depth = 10, shrinkage = 0.1)

wpred3 = predict(gbm_mod1, n.trees=80, newdata=test)
mean(abs(test_elo$WhiteElo-wpred3))

wpred5 = (wpred1*0.5)+(wpred3*0.5)
mean(abs(test_elo$WhiteElo-wpred5))


#Models for black elo

set.seed(12)

rf_mod2 = randomForest(train_elo$BlackElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result+white_opening_str2+black_opening_str2+white_opening_str4+black_opening_str4+white_opening_str6+black_opening_str6+white_opening_str3+black_opening_str3+white_opening_str5+black_opening_str5+white_opening_str7+black_opening_str7+white_opening_str8+black_opening_str8+white_opening_str14+black_opening_str14+white_opening_str25+black_opening_str25+white_avg_improve+black_avg_improve +white_median_improve + black_median_improve+white_q1_improve +white_q2_improve+white_q3_improve+white_q4_improve+black_q1_improve+black_q2_improve+black_q3_improve+black_q4_improve+white_castle_side+black_castle_side+score_stdev+white_5_improve+white_10_improve+white_15_improve+white_20_improve+white_25_improve+white_30_improve+white_35_improve+white_40_improve+white_45_improve+white_50_improve+black_5_improve+black_10_improve+black_15_improve+black_20_improve+black_25_improve+black_30_improve+black_35_improve+black_40_improve+black_45_improve+black_50_improve+white_55_improve+black_55_improve+white_q1_max +white_q2_max +white_q3_max +white_q4_max +black_q1_max +black_q2_max +black_q3_max + black_q4_max+white_q1_min +white_q2_min +white_q3_min+ white_q4_min +black_q1_min + black_q2_min + black_q3_min + black_q4_min+                   game_score10 + game_score20 + game_score30 + game_score40 + game_score50 + game_score60 + game_score70 + game_score80 +  game_score90 + game_score100, data= train, mtry=4, ntree=1000, nodesize=25)

bpred1 = predict(rf_mod2, newdata=test)

linear_mod2 = lm(train_elo$BlackElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result+white_opening_str2+black_opening_str2+white_opening_str4+black_opening_str4+white_opening_str6+black_opening_str6+white_opening_str3+black_opening_str3+white_opening_str5+black_opening_str5+white_opening_str7+black_opening_str7+white_opening_str8+black_opening_str8+white_opening_str14+black_opening_str14+white_opening_str25+black_opening_str25+white_avg_improve+black_avg_improve +white_median_improve + black_median_improve+white_q1_improve + white_q2_improve +white_q3_improve +white_q4_improve +black_q1_improve +black_q2_improve +black_q3_improve +black_q4_improve + white_castle_side + black_castle_side+I(average_score^2)+I(ending_score^2)+I(largest_drop^2)+I(largest_gain^2)+I(max_score^2)+I(min_score^2)+I(white_opening_str25^2)+I(black_opening_str25^2)+I(max_score^2)+I(min_score^2)+I(white_opening_str14^2)+I(black_opening_str14^2)+I(white_opening_str8^2)+I(black_opening_str8^2)+I(average_score*game_length)+I(largest_drop*game_length)+I(min_score*game_length)+I(white_avg_improve^3)+I(black_avg_improve^3)+I(white_q2_improve^2)+I(white_q3_improve^2)+I(black_q2_improve^2)+I(log(game_length))+score_stdev+I(score_stdev^2)+I(score_stdev^3)+I(score_stdev^4)+I(score_stdev^5)+I(score_stdev^6)+I(score_stdev^2*average_score^2)+I(log(game_length)*score_stdev)+white_check_rate+white_5_improve+white_10_improve+white_15_improve+white_20_improve+white_25_improve+white_30_improve+white_35_improve+white_40_improve+white_45_improve+white_50_improve+black_5_improve+black_10_improve+black_15_improve+black_20_improve+black_25_improve+black_30_improve+black_35_improve+black_40_improve+black_45_improve+black_50_improve+white_55_improve+black_55_improve+white_q1_max +white_q2_max +white_q3_max +white_q4_max +black_q1_max +black_q2_max +black_q3_max + black_q4_max+white_q1_min +white_q2_min +white_q3_min+ white_q4_min +black_q1_min + black_q2_min + black_q3_min + black_q4_min+  game_score10 + game_score20 + game_score30 + game_score40 + game_score50 + game_score60 + game_score70 + game_score80 +  game_score90 + game_score100, data= train)

bpred2 = predict(linear_mod2, newdata=test)
#Set outlier predictions equal to a fixed max and min
bpred2 = ifelse(bpred2<1700,1700,bpred2)
bpred2 = ifelse(bpred2>2500,2500,bpred2)


set.seed(12)
gbm_mod2 = gbm(train_elo$BlackElo~average_score+ending_score+game_length+largest_drop+largest_gain+max_score+min_score+result+white_opening_str2+black_opening_str2+white_opening_str4+black_opening_str4+white_opening_str6+black_opening_str6+white_opening_str3+black_opening_str3+white_opening_str5+black_opening_str5+white_opening_str7+black_opening_str7+white_opening_str8+black_opening_str8+white_opening_str14+black_opening_str14+white_opening_str25+black_opening_str25+white_avg_improve+black_avg_improve +white_median_improve + black_median_improve+white_q1_improve +white_q2_improve+white_q3_improve+white_q4_improve+black_q1_improve+black_q2_improve+black_q3_improve+black_q4_improve+white_castle_side+black_castle_side+score_stdev+white_5_improve+white_10_improve+white_15_improve+white_20_improve+white_25_improve+white_30_improve+white_35_improve+white_40_improve+white_45_improve+white_50_improve+black_5_improve+black_10_improve+black_15_improve+black_20_improve+black_25_improve+black_30_improve+black_35_improve+black_40_improve+black_45_improve+black_50_improve+white_55_improve+black_55_improve+white_q1_max +white_q2_max +white_q3_max +white_q4_max +black_q1_max +black_q2_max +black_q3_max + black_q4_max+white_q1_min +white_q2_min +white_q3_min+ white_q4_min +black_q1_min + black_q2_min + black_q3_min + black_q4_min+                   game_score10 + game_score20 + game_score30 + game_score40 + game_score50 + game_score60 + game_score70 + game_score80 +  game_score90 + game_score100, data= train, n.trees=80, interaction.depth = 10, shrinkage = 0.1)


bpred3 = predict(gbm_mod2, n.trees=80, newdata=test)

bpred4 = (bpred1*0.475)+(bpred3*0.475)+(bpred2*0.05)

submission19 = data.frame(Event= test$Event, WhiteElo=wpred5 ,BlackElo=bpred4 )
write.csv(submission19, "chess_elo_submission19.csv", row.names=FALSE)

#Score=

