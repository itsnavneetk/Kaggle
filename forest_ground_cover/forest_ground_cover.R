
#Forest ground cover Kaggle competition
library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)
library(caTools)
library(mice)
library(gbm)



#Load the data
train = read.csv("train.csv")
test = read.csv("test.csv")

train$Cover_Type = as.factor(train$Cover_Type)



#start with high complxity CART model, nodesize = 10

cart_model = rpart(Cover_Type~ .-Id, data=train, cp=0.00000000001, minbucket=10, method="class")

pred = predict(cart_model, newdata=test, type="class")

submission1 = data.frame(Id= test$Id , Cover_Type=pred)
summary(submission1)

write.csv(submission1, "forest_submission1.csv", row.names=FALSE)
#prp(cart_model)


cart_model = rpart(Cover_Type~ .-Id, data=train, cp=0.00000000001, minbucket=4, method="class")

pred = predict(cart_model, newdata=test, type="class")

submission2 = data.frame(Id= test$Id , Cover_Type=pred)
summary(submission2)

write.csv(submission2, "forest_submission2.csv", row.names=FALSE)


cart_model = rpart(Cover_Type~ .-Id, data=train, cp=0.001, minbucket=5, method="class")

pred = predict(cart_model, newdata=test, type="class")

submission3 = data.frame(Id= test$Id , Cover_Type=pred)
summary(submission3)

write.csv(submission3, "forest_submission3.csv", row.names=FALSE)


#Super complex CART model
cart_model = rpart(Cover_Type~ .-Id, data=train, cp=0.00000000000001, minbucket=2, method="class")

pred = predict(cart_model, newdata=test, type="class")

submission4 = data.frame(Id= test$Id , Cover_Type=pred)
summary(submission4)

write.csv(submission4, "forest_submission4.csv", row.names=FALSE)
