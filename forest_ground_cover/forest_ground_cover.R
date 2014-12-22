
#Forest ground cover Kaggle competition
library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)
library(e1071)
library(caTools)
library(mice)
library(gbm)
library(kernlab)
library(kknn)
options( java.parameters = "-Xmx3g" )
library(extraTrees)
library(deepnet)


#First I used python to add a Soil_Type variable that combines the 40 soil type factors into a single
#factor with 40 levels

#Load the data
train = read.csv("altered_train.csv")
test = read.csv("altered_test.csv")

train$Cover_Type = as.factor(train$Cover_Type)

#Convert Wilderness area and soil type into single factors with multiple levels

train$Wilderness_Area = (train$Wilderness_Area1+(train$Wilderness_Area2*2)+(train$Wilderness_Area3*3)+(train$Wilderness_Area4*4))

test$Wilderness_Area = (test$Wilderness_Area1+(test$Wilderness_Area2*2)+(test$Wilderness_Area3*3)+(test$Wilderness_Area4*4))





#start with high complxity CART model, nodesize = 10

cart_model = rpart(Cover_Type~ .-Id, data=train, cp=0.00000000001, minbucket=10, method="class")

pred = predict(cart_model, newdata=test, type="class")

submission1 = data.frame(Id= test$Id , Cover_Type=pred)
summary(submission1)

write.csv(submission1, "forest_submission1.csv", row.names=FALSE)
#prp(cart_model)


cart_model = rpart(Cover_Type~ .-Id, data=train, cp=0.00000000001, minbucket=4, method="class")

pred = predict(cart_model, newdata=test, type="class")

submission2 = data.frame(Id= test$Id , Cover_Type=pre)
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
#Best score: 0.65940

#------End Basic cart models-------------------------------------------------------

#Exploring the data...
ggplot(aes(x=Elevation,y= Wilderness_Area1),data=train) + geom_jitter(aes(color=Cover_Type))
ggplot(aes(x=Elevation,y= Wilderness_Area2),data=train) + geom_jitter(aes(color=Cover_Type))
ggplot(aes(x=Elevation,y= Wilderness_Area3),data=train) + geom_jitter(aes(color=Cover_Type))
ggplot(aes(x=Elevation,y= Wilderness_Area4),data=train) + geom_jitter(aes(color=Cover_Type))


ggplot(aes(x=Soil_Type10,y= Wilderness_Area1),data=train) + geom_jitter(aes(color=Cover_Type))

# Wilderness area, Elevation and soil type seem most important.
summary(train$Wilderness_Area)
summary(test)

set.seed(121)

rfcontrol = trainControl(method = "repeatedcv",
                         number = 5,
                         repeats = 1
)

rfgrid = expand.grid(mtry=c(10))

rf_model = train(Cover_Type~ Elevation+Wilderness_Area+Soil_Type, data=train, method='rf', tune.grid=rfgrid)

rf_model
summary(rf_model)

pred = predict(rf_model, newdata=test)

submission5 = data.frame(Id= test$Id , Cover_Type=pred)
summary(submission5)
write.csv(submission5, "forest_submission5.csv", row.names=FALSE)

#Random forest with caret takes forever... Try basic rf


set.seed(121)
basic_rf = randomForest(Cover_Type~ Elevation+Slope+Hillshade_9am+Hillshade_Noon+Hillshade_3pm+Horizontal_Distance_To_Fire_Points+Wilderness_Area+Soil_Type, data=train,ntree=450, nodesize= 5)

basic_rf 
summary(basic_rf)

pred = predict(basic_rf, newdata=test)
submission6 = data.frame(Id= test$Id , Cover_Type=pred)
summary(submission6)

write.csv(submission6, "forest_submission6.csv", row.names=FALSE)

#Score = 0.68250

set.seed(121)
basic_rf = randomForest(Cover_Type~ Elevation+Slope+Aspect+Horizontal_Distance_To_Hydrology+Vertical_Distance_To_Hydrology+Horizontal_Distance_To_Roadways + Hillshade_9am+Hillshade_Noon+Hillshade_3pm+Horizontal_Distance_To_Fire_Points+Wilderness_Area+Soil_Type, data=train,ntree=1500, nodesize= 10, mtry= 8)

basic_rf 
summary(basic_rf)

pred = predict(basic_rf, newdata=test)

submission7 = data.frame(Id= test$Id , Cover_Type=pred)
summary(submission7)

write.csv(submission7, "forest_submission7.csv", row.names=FALSE)


#Score = 0.74289 Much better. High ntree, mtry and nodesize seem to have helped. Also adding in all varaibles may have helped.

set.seed(121)
basic_rf = randomForest(Cover_Type~ Elevation+Slope+Aspect+Horizontal_Distance_To_Hydrology+Vertical_Distance_To_Hydrology+Horizontal_Distance_To_Roadways + Hillshade_9am+Hillshade_Noon+Hillshade_3pm+Horizontal_Distance_To_Fire_Points+Wilderness_Area+Soil_Type, data=train,ntree=1500, nodesize= 30, mtry= 15)

basic_rf 
summary(basic_rf)

pred = predict(basic_rf, newdata=test)

submission8 = data.frame(Id= test$Id , Cover_Type=pred)
summary(submission8)

write.csv(submission8, "forest_submission8.csv", row.names=FALSE)

#-Larger node size and mtry... did not work as well


set.seed(121)
basic_rf = randomForest(Cover_Type~ Elevation+Slope+Aspect+Horizontal_Distance_To_Hydrology+Vertical_Distance_To_Hydrology+Horizontal_Distance_To_Roadways + Hillshade_9am+Hillshade_Noon+Hillshade_3pm+Horizontal_Distance_To_Fire_Points+Wilderness_Area+Soil_Type, data=train,ntree=1500, nodesize= 1, mtry= 5)

basic_rf 
summary(basic_rf)

pred = predict(basic_rf, newdata=test)

submission9 = data.frame(Id= test$Id , Cover_Type=pred)
summary(submission9)

write.csv(submission9, "forest_submission9.csv", row.names=FALSE)

#-Try nodesize 1 and lower mtry. Best so far: 0.75688



#Try Stochasitc Gradient Boosting
gbm_model1 = gbm(Cover_Type~ Elevation+Slope+Aspect+Horizontal_Distance_To_Hydrology+Vertical_Distance_To_Hydrology+Horizontal_Distance_To_Roadways + Hillshade_9am+Hillshade_Noon+Hillshade_3pm+Horizontal_Distance_To_Fire_Points+Wilderness_Area+Soil_Type, data=train, n.trees=1000, interaction.depth = 10, shrinkage = 0.1)

best = gbm.perf(gbm_model1,method="OOB")
print(best)

pred = predict(gbm_model1, newdata=test, n.trees = best, type="response")

submission10 = data.frame(Id= test$Id , Cover_Type_Pred=pred)
submission10$Cover_Type = apply(submission10[ ,c(2:8)], 1, which.max)
submission10 = submission10[ ,c(1,9)]

table(submission10$Cover_Type)

write.csv(submission10, "forest_submission10.csv", row.names=FALSE)
#Score=  0.68779

#Stochasitc Gradient Boosting attempt 2
gbm_model1 = gbm(Cover_Type~ Elevation+Slope+Aspect+Horizontal_Distance_To_Hydrology+Vertical_Distance_To_Hydrology+Horizontal_Distance_To_Roadways + Hillshade_9am+Hillshade_Noon+Hillshade_3pm+Horizontal_Distance_To_Fire_Points+Wilderness_Area+Soil_Type, data=train, n.trees=1000, interaction.depth = 5, shrinkage = 0.01)

best = gbm.perf(gbm_model1,method="OOB")
print(best)

pred = predict(gbm_model1, newdata=test, n.trees = best, type="response")

submission11 = data.frame(Id= test$Id , Cover_Type_Pred=pred)
submission11$Cover_Type = apply(submission11[ ,c(2:8)], 1, which.max)
submission11 = submission11[ ,c(1,9)]

table(submission11$Cover_Type)

write.csv(submission11, "forest_submission11.csv", row.names=FALSE)
#Score went down


#Attempt Support Vector Machine
svm_model1 = ksvm(Cover_Type~ Elevation+Slope+Aspect+Horizontal_Distance_To_Hydrology+Vertical_Distance_To_Hydrology+Horizontal_Distance_To_Roadways + Hillshade_9am+Hillshade_Noon+Hillshade_3pm+Horizontal_Distance_To_Fire_Points+Wilderness_Area+Soil_Type, data=train)


pred = predict(svm_model1, newdata=test, type="response")
summary(pred)

submission12 = data.frame(Id= test$Id , Cover_Type=pred)
summary(submission12)

write.csv(submission12, "forest_submission12.csv", row.names=FALSE)
#Not that good: 0.63450


#K Nearest Neighbors - k =5

set.seed(121)
knn_model = kknn(Cover_Type~ Elevation+Slope+Aspect+Horizontal_Distance_To_Hydrology+Vertical_Distance_To_Hydrology+Horizontal_Distance_To_Roadways + Hillshade_9am+Hillshade_Noon+Hillshade_3pm+Horizontal_Distance_To_Fire_Points+Wilderness_Area+Soil_Type, train=train, test=test, k=5)

knn_model 
summary(knn_model)


submission13 = data.frame(Id= test$Id , Cover_Type=knn_model$fitted.values)
summary(submission13)

write.csv(submission13, "forest_submission13.csv", row.names=FALSE)
#Score = 0.66447

set.seed(121)
knn_model = kknn(Cover_Type~ Elevation+Slope+Aspect+Horizontal_Distance_To_Hydrology+Vertical_Distance_To_Hydrology+Horizontal_Distance_To_Roadways + Hillshade_9am+Hillshade_Noon+Hillshade_3pm+Horizontal_Distance_To_Fire_Points+Wilderness_Area+Soil_Type, train=train, test=test, k=5)

knn_model 
summary(knn_model)


submission13 = data.frame(Id= test$Id , Cover_Type=knn_model$fitted.values)
summary(submission13)

write.csv(submission13, "forest_submission13.csv", row.names=FALSE)


set.seed(121)

knn_model2 = kknn(Cover_Type~ Elevation+Slope+Aspect+Horizontal_Distance_To_Hydrology+Vertical_Distance_To_Hydrology+Horizontal_Distance_To_Roadways + Hillshade_9am+Hillshade_Noon+Hillshade_3pm+Horizontal_Distance_To_Fire_Points+Wilderness_Area+Soil_Type, train=train, test=test, k=1)


submission14 = data.frame(Id= test$Id , Cover_Type=knn_model2$fitted.values)
summary(submission14)

write.csv(submission14, "forest_submission14.csv", row.names=FALSE)
#Score = 0.67255


#Extra trees random forest

set.seed(121)
extra_trees_rf = extraTrees(x= train[ ,c(2:11,56,58)], y= train$Cover_Type, ntree=500, nodesize= 1, mtry= 5, numRandomCuts=3)

extra_trees_rf
summary(extra_trees_rf)

pred = predict(extra_trees_rf, newdata=test[ ,c(2:11,56,57)])

submission15 = data.frame(Id= test$Id , Cover_Type=pred)
summary(submission15)

write.csv(submission15, "forest_submission15.csv", row.names=FALSE)
#Score 0.78564 Significant improvement over my best basic RF


#split data into train and validation sets
set.seed(121)
split1 = createDataPartition(train$Cover_Type,p=0.8,list = FALSE,times = 1)
train2 = train[ split1,]
validation = train[-split1,]
  
#Try to find best accuracy using validation set
extra_trees_rf = extraTrees(x= train2[ ,c(2:11,56,58)], y= train2$Cover_Type, ntree=100, nodesize= 1, mtry= 5, numRandomCuts=4)

pred = predict(extra_trees_rf, newdata=validation[ ,c(2:11,56,58)])

confusionMatrix(validation$Cover_Type, pred)
  

#100 ntree seems adequate
set.seed(121)
extra_trees_rf = extraTrees(x= train[ ,c(2:11,56,58)], y= train$Cover_Type, ntree=100, nodesize= 1, mtry= 5, numRandomCuts=4)

pred = predict(extra_trees_rf, newdata=test[ ,c(2:11,56,57)])

submission16 = data.frame(Id= test$Id , Cover_Type=pred)
summary(submission16)

write.csv(submission16, "forest_submission16.csv", row.names=FALSE)

#Scored slightly less.

set.seed(121)
extra_trees_rf = extraTrees(x= train[ ,c(2:11,56,58)], y= train$Cover_Type, ntree=1000, nodesize= 1, mtry= 5, numRandomCuts=4)

pred = predict(extra_trees_rf, newdata=test[ ,c(2:11,56,57)])

submission17 = data.frame(Id= test$Id , Cover_Type=pred)
summary(submission17)

write.csv(submission17, "forest_submission17.csv", row.names=FALSE)
#Score: 0.78600

#One more with high ntree
set.seed(13)

extra_trees_rf = extraTrees(x= train[ ,c(2:11,56,58)], y= train$Cover_Type, ntree=1500, nodesize= 1, mtry= 5, numRandomCuts=4)

pred = predict(extra_trees_rf, newdata=test[ ,c(2:11,56,57)])

submission18 = data.frame(Id= test$Id , Cover_Type=pred)
summary(submission18)

write.csv(submission18, "forest_submission18.csv", row.names=FALSE)


#Cluster by Wilderness Area and then run extra trees

trains = split(train, f= train[ ,58])
tests = split(test, f= test[ ,57])

set.seed(144)
extra_trees1 = extraTrees(x= trains[[1]][ ,c(2:11,56,58)], y= trains[[1]][ ,c(57)], ntree=1500, nodesize= 1, mtry= 5, numRandomCuts=4)

extra_trees2 = extraTrees(x= trains[[2]][ ,c(2:11,56,58)], y= trains[[2]][ ,c(57)], ntree=1500, nodesize= 1, mtry= 5, numRandomCuts=4)

extra_trees3 = extraTrees(x= trains[[3]][ ,c(2:11,56,58)], y= trains[[3]][ ,c(57)], ntree=1500, nodesize= 1, mtry= 5, numRandomCuts=4)

extra_trees4 = extraTrees(x= trains[[4]][ ,c(2:11,56,58)], y= trains[[4]][ ,c(57)], ntree=1500, nodesize= 1, mtry= 5, numRandomCuts=4)


pred1 = predict(extra_trees1, newdata=tests[[1]][ ,c(2:11,56,57)])
summary(pred1)
pred2 = predict(extra_trees2, newdata=tests[[2]][ ,c(2:11,56,57)])
summary(pred2)
pred3 = predict(extra_trees3, newdata=tests[[3]][ ,c(2:11,56,57)])
summary(pred3)
pred4 = predict(extra_trees4, newdata=tests[[4]][ ,c(2:11,56,57)])
summary(pred4)

sub1 = data.frame(Id= tests[[1]]$Id , Cover_Type=pred1)
sub2 = data.frame(Id= tests[[2]]$Id , Cover_Type=pred2)
sub3 = data.frame(Id= tests[[3]]$Id , Cover_Type=pred3)
sub4 = data.frame(Id= tests[[4]]$Id , Cover_Type=pred4)

submission19 = rbind(sub1,sub2,sub3,sub4)

summary(submission19)

write.csv(submission19, "forest_submission19.csv", row.names=FALSE)
#Score = 0.78612 slightly lower than non-clustered version with ntree 1500.. retry with ntree 1500

submission20 = rbind(sub1,sub2,sub3,sub4)

summary(submission20)

write.csv(submission20, "forest_submission20.csv", row.names=FALSE)
#Score = 0.78699 slight improvement on best score.



#Neural Nets
train$Cover_Type = as.numeric(train$Cover_Type)

nnet1 = nn.train(x= as.matrix(train[ , c(2:11,56,58)]), y= as.matrix(train$Cover_Type),  hidden = c(12,25), activationfun = "sigm", learningrate = 0.8,
                 momentum = 0.5, learningrate_scale = 1, output = "sigm", numepochs = 3,
                 batchsize = 10, hidden_dropout = 0, visible_dropout = 0)


nnpred = nn.predict(nnet1, as.matrix(test[ , c(2:11,56,57)] ) )


summary(nnpred)

