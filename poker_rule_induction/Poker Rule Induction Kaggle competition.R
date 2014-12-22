#Poker Rule Induction Kaggle competition
#https://www.kaggle.com/c/poker-rule-induction
library(caret)
library(randomForest)
library(gbm)
options( java.parameters = "-Xmx3g" )
library(extraTrees)




train = read.csv("train.csv")
test = read.csv("test.csv")
test = test[,2:11]

labels = as.factor(train$hand)
train = train[,1:10]

part_train = train[1:18000,]
valid = train[-1:-18000,]

labels_part = labels[1:18000]
valid_labels = labels[-1:-18000]

set.seed(12)
tree = randomForest(labels_part~., data=part_train, nodesize=1, ntree=500, mtry=4)

tree_pred = predict(tree, newdata=valid, type="class")

confusionMatrix(tree_pred,valid_labels)

set.seed(12)
xtrees = extraTrees(y=labels_part, x=part_train, nodesize=1, ntree=500, mtry=4, numRandomCuts = 3)

xtrees_pred = predict(xtrees, newdata=valid)

confusionMatrix(xtrees_pred,valid_labels)



#GBM model
set.seed(12)

tunecontrol = trainControl(method = "none")

tgrid = expand.grid(n.trees = c(2000), interaction.depth=c(15) ,shrinkage=c(0.107) )

gbm_mod = train(labels_part~., data=part_train, method="gbm", trControl=tunecontrol, tuneGrid=tgrid)

pred_gbm = predict(gbm_mod, newdata=valid)
confusionMatrix(pred_gbm ,valid_labels)

#GBM accuracy 0.6439 on 100 trees,  on 2000 trees


#Full training model and submission code
set.seed(12)

tunecontrol = trainControl(method = "none")

tgrid = expand.grid(n.trees = c(2000), interaction.depth=c(15) ,shrinkage=c(0.107) )

gbm_mod = train(labels~., data=train, method="gbm", trControl=tunecontrol, tuneGrid=tgrid)

pred_gbm = predict(gbm_mod, newdata=test[1:1000000,])

submission1 = data.frame(id=1:1000000, hand=pred_gbm)
write.csv(submission1, "poker_induction_submission_firsthalf.csv", row.names=FALSE)
