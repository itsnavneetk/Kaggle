#digit recognizer competition
library(caret)
library(randomForest)
library(gbm)
library(kknn)
library(ggplot2)
library(deepnet)

#Read the data
train = read.csv("train.csv")
test= read.csv("test.csv")

#Separate training labels from training data
labels= as.factor(train[ ,1])
train = train[ ,2:785]

#Combine train and test sets temporarily when creating features
train = rbind(train,test)


#Create some potentially useful features

#Total pixel intensity
total_inensity = rowSums(train)
#Number of pixels with non-zero intensity
non_zero = rowSums(train!=0)
#Average pixel intensity
average_inensity = total_inensity/non_zero
#Number of low inensity pixels
between_0_100 = rowSums(train>0 & train<100)
#Proportion of low inensity pixels
prop_between_0_100 = between_0_100/non_zero


new_train=data.frame("total_inensity"=total_inensity,"non_zero"=non_zero,"average_inensity"=average_inensity,"between_0_100"=between_0_100,"prop_between_0_100"=prop_between_0_100)



vertical_seq = seq(1,784,by=28)

#Features for proportion of total pixel intensity in horizontal strips
for (x in 1:28){
  row_intensity = rowSums(train[,(1+(28*(x-1))):(28+(28*(x-1)))])
  new_train[paste("h",x,sep="")] = row_intensity/total_inensity
}


#Features for proportion of total pixel intensity in vertical strips
for (x in 1:28){
  new_train[paste("v",x,sep="")] = rowSums(train[,vertical_seq+(x-1)])/total_inensity
}

#Features for proportion of total pixel intensity in large patches (splits the image into a 4x4 grid and finds the proportion of total pixel inensity in each section)
for (x in 1:4){
  for (y in 1:4){
    line = 1:7 + ((x-1)*7) + ((y-1)*196)
    for (z in 1:6){
      line=c(line,line[1:7]+(28*z))
    }
    new_train[paste("section",x,y,sep="")] = rowSums(train[line])/total_inensity
  }
}

#Features for proportion of total pixel intensity in small patches (splits the image into a 7x7 grid and finds the proportion of total pixel inensity in each section)
for (x in 1:7){
  for (y in 1:7){
    line = 1:4 + ((x-1)*4) + ((y-1)*112)
    for (z in 1:3){
      line=c(line,line[1:4]+(28*z))
    }
    new_train[paste("smallsection",x,y,sep="")] = rowSums(train[line])/total_inensity
  }
}

#Separate testing from training
test = new_train[42001:70000, ]
new_train= new_train[1:42000,]

#Training and validation sets
part_train = new_train[1:30000,]
valid = new_train[30001:42000,]



#Validation code------------------------------------------
#Random forest model
rf_mod1= randomForest(labels[1:30000]~., data=part_train, nodesize=1, ntree=50, mtry=5)

rf_pred = predict(rf_mod1,newdata=valid)

confusionMatrix(rf_pred,labels[30001:42000])
#Validation accuracy of 0.96


#KNN model
knnpred = kknn(labels[1:30000]~., train=part_train, test=valid, k=3)

kpred = fitted.values(knnpred)

confusionMatrix(fitted.values(knnpred),labels[30001:42000])
#Validation Accuracy 0.9385



#GBM model
tunecontrol = trainControl(method = "repeatedcv",
                           number = 2,
                           repeats = 1
)

tgrid = expand.grid(n.trees = c(10),interaction.depth=c(7) ,shrinkage=c(0.107) )


gbm_mod = train(labels[1:30000]~., data=part_train, method= 'gbm', trControl=tunecontrol, tuneGrid=tgrid)

pred_gbm = predict(gbm_mod, newdata=valid)

confusionMatrix(pred_gbm,labels[30001:42000])
#Validation Accuracy 0.95916


#Ensemble prediction
comb_pred = as.factor(ifelse(pred_gbm==kpred,kpred,rf_pred))
levels(comb_pred) = 0:9
confusionMatrix(comb_pred,labels[30001:42000])
#Validation code------------------------------------------



#MISC code----------------
#Average pixel intensities of each number
avg_list = list()

for (x in 0:9){
  avg_list[[length(avg_list)+1]] <- as.integer((colMeans(subset(train, labels==as.character(x)))))
}

avg_d = data.frame(matrix(unlist(avg_list), nrow=10, byrow=T))
names(avg_d) = names(train)


#Code to plot individual rows(images) in the data set
plotter = function(img_num){
  image(t(matrix(as.matrix(train[img_num,]),nrow=28,byrow=TRUE)[28:1,]), axes = FALSE, col = grey(seq(1, 0, length = 256)))
}
  
first_10_missclassified = head(which(comb_pred!=labels[30001:42000]), 10)
for (error in first_10_missclassified){
  print(paste("Estimate: ",comb_pred[error]))
  print(paste("Actual: ",labels[30001:42000][error]))
  plotter(30000+error)
}
#MISC code----------------





#Final prediction code---------------
#Random forest model
rf_mod1= randomForest(labels~., data=new_train, nodesize=1, ntree=500, mtry=5)

rf_pred = predict(rf_mod1,newdata=test)


#KNN model
knnpred = kknn(labels~., train=new_train, test=test, k=3)

kpred = fitted.values(knnpred)

#GBM model
tunecontrol = trainControl(method = "repeatedcv",
                           number = 2,
                           repeats = 1
)

tgrid = expand.grid(n.trees = c(100),interaction.depth=c(7) ,shrinkage=c(0.107) )


gbm_mod = train(labels~., data=new_train, method= 'gbm', trControl=tunecontrol, tuneGrid=tgrid)

pred_gbm = predict(gbm_mod, newdata=test)


#Ensemble prediction
comb_pred = as.factor(ifelse(pred_gbm==kpred,kpred,rf_pred))
levels(comb_pred) = 0:9


image_comp_submission1 = data.frame(ImageId=c(1:nrow(test)) ,  Label=comb_pred)
write.csv(image_comp_submission1, "image_comp_submission1.csv", row.names=FALSE)

