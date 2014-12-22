
#Tradeshift Text Classification
library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)
library(e1071)
library(gbm)
library(kernlab)
library(kknn)
options( java.parameters = "-Xmx3g" )
library(extraTrees)




train_labels = read.csv("trainLabels.csv")

train = read.csv("train.csv")

#To start, throw out training features with thousands of levels
#train = train[ ,c(1,2,3,6:34,37:61,63,64,67:91,93,94,97:146)]

#Save the first 1/10 of the data to a new training file. 
#17million by 146 is too unwieldy for a single laptop
write.csv(train[1:170000, ], "tenth_train.csv", row.names=FALSE)
write.csv(train_labels[1:170000, ], "tenth_labels.csv", row.names=FALSE)

train = read.csv("tenth_train.csv")
train_labels = read.csv("tenth_labels.csv")

#Save first 100th into a smaller training set for initial tests
write.csv(train[1:17000, ], "hundreth_train.csv", row.names=FALSE)
write.csv(train_labels[1:17000, ], "hundreth_labels.csv", row.names=FALSE)

#Save second 100th into a validation set for initial tests
write.csv(train[17001:(17000*2), ], "hundreth_validation.csv", row.names=FALSE)
write.csv(train_labels[17001:(17000*2), ], "validation_labels.csv", row.names=FALSE)

train = read.csv("hundreth_train.csv")
train_labels = read.csv("hundreth_labels.csv")

test = read.csv("hundreth_validation.csv")
test_labels = read.csv("validation_labels.csv")

#test = read.csv("test.csv")

#Remove features of hash values
train = train[ ,c(1,2,3,6:34,37:61,63,64,67:91,93,94,97:146)]
test = test[ ,c(1,2,3,6:34,37:61,63,64,67:91,93,94,97:146)]


#For use when using final test data and making submissions----------

#Bad testing set rows, no data for several variables
shitty_test = subset(test, x126=="")

#Predict 0 for all 8 bad rows.
bad_data_preds = data.frame(id_label=shitty_combs, pred= rep(0,length(shitty_combs)))

#Remove bad rows from test set
test = test[-c(shitty_test$id-1700000), ]

#Initialize submission data frame with y14 predictions set to zero
submission = data.frame(id_label=(paste(1700001:(1700000+545082),"y14",sep="_")), pred=rep(0,545082))

shitty = c('1724268_y1', '1724268_y2', '1724268_y3', '1724268_y4', '1724268_y5', '1724268_y6', '1724268_y7', '1724268_y8', '1724268_y9', '1724268_y10', '1724268_y11', '1724268_y12', '1724268_y13', '1724268_y14', '1724268_y15', '1724268_y16', '1724268_y17', '1724268_y18', '1724268_y19', '1724268_y20', '1724268_y21', '1724268_y22', '1724268_y23', '1724268_y24', '1724268_y25', '1724268_y26', '1724268_y27', '1724268_y28', '1724268_y29', '1724268_y30', '1724268_y31', '1724268_y32', '1724268_y33', '1724364_y1', '1724364_y2', '1724364_y3', '1724364_y4', '1724364_y5', '1724364_y6', '1724364_y7', '1724364_y8', '1724364_y9', '1724364_y10', '1724364_y11', '1724364_y12', '1724364_y13', '1724364_y14', '1724364_y15', '1724364_y16', '1724364_y17', '1724364_y18', '1724364_y19', '1724364_y20', '1724364_y21', '1724364_y22', '1724364_y23', '1724364_y24', '1724364_y25', '1724364_y26', '1724364_y27', '1724364_y28', '1724364_y29', '1724364_y30', '1724364_y31', '1724364_y32', '1724364_y33', '1860995_y1', '1860995_y2', '1860995_y3', '1860995_y4', '1860995_y5', '1860995_y6', '1860995_y7', '1860995_y8', '1860995_y9', '1860995_y10', '1860995_y11', '1860995_y12', '1860995_y13', '1860995_y14', '1860995_y15', '1860995_y16', '1860995_y17', '1860995_y18', '1860995_y19', '1860995_y20', '1860995_y21', '1860995_y22', '1860995_y23', '1860995_y24', '1860995_y25', '1860995_y26', '1860995_y27', '1860995_y28', '1860995_y29', '1860995_y30', '1860995_y31', '1860995_y32', '1860995_y33', '2003561_y1', '2003561_y2', '2003561_y3', '2003561_y4', '2003561_y5', '2003561_y6', '2003561_y7', '2003561_y8', '2003561_y9', '2003561_y10', '2003561_y11', '2003561_y12', '2003561_y13', '2003561_y14', '2003561_y15', '2003561_y16', '2003561_y17', '2003561_y18', '2003561_y19', '2003561_y20', '2003561_y21', '2003561_y22', '2003561_y23', '2003561_y24', '2003561_y25', '2003561_y26', '2003561_y27', '2003561_y28', '2003561_y29', '2003561_y30', '2003561_y31', '2003561_y32', '2003561_y33', '2041394_y1', '2041394_y2', '2041394_y3', '2041394_y4', '2041394_y5', '2041394_y6', '2041394_y7', '2041394_y8', '2041394_y9', '2041394_y10', '2041394_y11', '2041394_y12', '2041394_y13', '2041394_y14', '2041394_y15', '2041394_y16', '2041394_y17', '2041394_y18', '2041394_y19', '2041394_y20', '2041394_y21', '2041394_y22', '2041394_y23', '2041394_y24', '2041394_y25', '2041394_y26', '2041394_y27', '2041394_y28', '2041394_y29', '2041394_y30', '2041394_y31', '2041394_y32', '2041394_y33', '2091271_y1', '2091271_y2', '2091271_y3', '2091271_y4', '2091271_y5', '2091271_y6', '2091271_y7', '2091271_y8', '2091271_y9', '2091271_y10', '2091271_y11', '2091271_y12', '2091271_y13', '2091271_y14', '2091271_y15', '2091271_y16', '2091271_y17', '2091271_y18', '2091271_y19', '2091271_y20', '2091271_y21', '2091271_y22', '2091271_y23', '2091271_y24', '2091271_y25', '2091271_y26', '2091271_y27', '2091271_y28', '2091271_y29', '2091271_y30', '2091271_y31', '2091271_y32', '2091271_y33', '2140951_y1', '2140951_y2', '2140951_y3', '2140951_y4', '2140951_y5', '2140951_y6', '2140951_y7', '2140951_y8', '2140951_y9', '2140951_y10', '2140951_y11', '2140951_y12', '2140951_y13', '2140951_y14', '2140951_y15', '2140951_y16', '2140951_y17', '2140951_y18', '2140951_y19', '2140951_y20', '2140951_y21', '2140951_y22', '2140951_y23', '2140951_y24', '2140951_y25', '2140951_y26', '2140951_y27', '2140951_y28', '2140951_y29', '2140951_y30', '2140951_y31', '2140951_y32', '2140951_y33', '2242660_y1', '2242660_y2', '2242660_y3', '2242660_y4', '2242660_y5', '2242660_y6', '2242660_y7', '2242660_y8', '2242660_y9', '2242660_y10', '2242660_y11', '2242660_y12', '2242660_y13', '2242660_y14', '2242660_y15', '2242660_y16', '2242660_y17', '2242660_y18', '2242660_y19', '2242660_y20', '2242660_y21', '2242660_y22', '2242660_y23', '2242660_y24', '2242660_y25', '2242660_y26', '2242660_y27', '2242660_y28', '2242660_y29', '2242660_y30', '2242660_y31', '2242660_y32', '2242660_y33')

bad_data_preds = data.frame(id_label=shitty, pred= rep(0,length(shitty)))

#Add bad testing rows to submission
submission = rbind(submission,bad_data_preds)


#-------------------------------------------------------------------


#Basic Cart #Scored 0.08xxx on first attempt
#Function trains a different CART model to each of the target variables in a dataframe

allCART = function(data, labels, cp, maxdep){
  CART_models = list()
  for (label in names(labels)){
    
    CART_mod = rpart(labels[[label]] ~ .-id, data=data, method="class", cp=cp, minbucket=1, maxdepth= maxdep)
    
    #prp(CART_mod)
    print(label)
    
    CART_models[[label]] = CART_mod
  }
  CART_models
}

#For some reason label 14 (all zeros) is causing an error with rpart. Handle label 14 separately.
CART_models = allCART(train, train_labels[ ,c(2:14,16:34)], 0.001, 4)


#Function to apply a list of CART models to a test set
makePreds = function(model_list, test_data){
  
  predictions = list()
  for (model in names(model_list)){
    print(model)
    prediction = predict(model_list[[model]], newdata=test_data, type="prob")
    predictions[[model]] = prediction
  }
  predictions
}

#All predictions except those for y14 and bad test rows
predictions = makePreds(CART_models, test)


#Format Predictions and add them to the submission data frame
for (prediction in names(predictions)){
  print(prediction) #Print for progress updates... takes a while to run
  formatted_pred = data.frame(id_label=(paste(test$id,prediction,sep="_")), pred= predictions[[prediction]][ ,2])
  submission = rbind(submission,formatted_pred)
}

#Write submission to CSV
write.csv(submission, "tradeshift2.csv", row.names=FALSE)


#Scored 0.08xxx on first attempt with depth=4 cp=0.001 on first 100th of training data

#Retry with more data/different parameters.
#2nd attempt 100th of training data cp=0.001, tree depth 10
#Tree depth 10 scored worse...



#Summary of labels shows y33 is the only target variable with a high percentage of appearance (> 0.5)
summary(train_labels)

#Tables of y33 vs other labels show that if y33 is true(1) all other labels
#are always zero:
for (target in 2:33){
  print ( table(train_labels[,34],train_labels[,target]) )
}
#Predicting y33 seems to be the key, since if y33 has high probability all other labels should be 0

#For now focus on models to predict y33


#Log loss function provided by competition:
llfun <- function(actual, prediction) {
  epsilon <- .000000000000001
  yhat <- pmin(pmax(prediction, epsilon), 1-epsilon)
  logloss <- -mean(actual*log(yhat)
                   + (1-actual)*log(1 - yhat))
  return(logloss)
}


##Total average Log loss guessing mean values
total_log_loss = c()
for (target in 2:34){
  validation_logloss =llfun(test_labels[,target], mean(train_labels[,target]))
  print (paste("y",target-1))
  print(validation_logloss)
  total_log_loss = append(total_log_loss, validation_logloss)
}
mean(total_log_loss)
#Mean values perform well for many of the variables, but worst for y33.



#Cart model achieves ~87% accuracy classifying y33
#CART Log loss = 0.3588316
y33_cart = rpart(train_labels$y33~ .-id, data=train, method="class", cp=0.001, minbucket=1, maxdepth= 10)

valid_pred = predict(y33_cart, newdata=test, type="prob")

confusionMatrix(test_labels$y33, as.numeric(valid_pred[,2]>0.5))

validation_logloss =llfun(test_labels$y33, valid_pred[,2])
validation_logloss




#Logistic regression model achieves ~80% accuracy classifying y33
#Log loss of regrssion model is 0.4414204... lower than CART
y33_glm = glm(train_labels$y33~ .-id, data=train, family="binomial")

valid_pred = predict(y33_glm, newdata=test, type="response")

validation_logloss =llfun(test_labels$y33, valid_pred)
validation_logloss

confusionMatrix(test_labels$y33, as.numeric(valid_pred >0.5))




#RF model 0.8939% accurate on y33
#Logloss 0.3010616
y33_rf = randomForest(train_labels$y33~ .-id, data=train, ntree=100, mtry=4)

predy33 = predict(y33_rf, newdata=test)

predy33[predy33==1] = 0.99

validation_logloss =llfun(test_labels$y33, predy33)
validation_logloss

confusionMatrix(test_labels$y33, as.numeric(predy33 >0.5))


#Add y33 to train predictions of y33 to test
train_y33= cbind(train, train_labels$y33)
colnames(train_y33)[137] = "y33"
test_y33 = cbind(test, predy33)
colnames(test_y33)[137] = "y33"



y32_rf = randomForest(train_labels$y32 ~ .-id, data=train, ntree=100, mtry=4)

predy32 = predict(y32_rf, newdata=test)

predy32[predy32==1] = 0.99

validation_logloss =llfun(test_labels$y32, predy32)
validation_logloss

confusionMatrix(test_labels$y32, as.numeric(predy32 >0.5))



allRF = function(train, test, labels, test_labels, tries, trees){
  RF_preds = list()

  for (label in names(labels)){
    
    RF_mod = randomForest(labels[[label]] ~ .-id, data=train, ntree=trees, mtry=tries)
    
    pred = predict(RF_mod, newdata=test)
    
    test_logloss= llfun(test_labels[[label]], pred)
    print(label)
    print(test_logloss)
    
    RF_preds[[label]] = pred
    
  }
  RF_preds
}

RF_preds = allRF(train_y33, test_y33, train_labels[ ,c(2:14,16:34)], test_labels,  4, 20)

log_losses = function(preds,labels){
  losses = c()
  for (label in names(labels)){
    loss = llfun(labels[[label]], preds[label])
    
  }
  losses
}

RF_losses = log_losses(RF_preds,train_labels[ ,c(2:14,16:34)])










