#Analytics Edge Kaggle competition #2


library(ggplot2)
library(rpart)
library(caret)
library(randomForest)
library(e1071)
library(glmnet)
library(gbm)
options( java.parameters = "-Xmx8g" )
library(extraTrees)
library(nnet)
library(xgboost)
library(tm)
library(pROC)
library(SnowballC)

train = read.csv("NYTimesBlogTrain.csv", stringsAsFactors=FALSE)
test = read.csv("NYTimesBlogTest.csv", stringsAsFactors=FALSE)

sample_sub = read.csv("SampleSubmission.csv")


train$PubDate = strptime(train$PubDate, "%Y-%m-%d %H:%M:%S")
test$PubDate = strptime(test$PubDate, "%Y-%m-%d %H:%M:%S")

train$Weekday =train$PubDate$wday
test$Weekday = test$PubDate$wday

train$Hour =train$PubDate$hour
test$Hour = test$PubDate$hour

train$PubDate = NULL
test$PubDate = NULL

targets = train$Popular
targets = as.factor(targets)
levels(targets) = c("notpop","popular")

train_ID = train$UniqueID
test_ID = test$UniqueID

train$UniqueID = NULL
test$UniqueID = NULL
train$Popular = NULL

#Convert headlines using TDIDF weighting
CorpusHeadline = Corpus(VectorSource(c(train$Headline, test$Headline)))
CorpusHeadline = tm_map(CorpusHeadline, tolower)
CorpusHeadline = tm_map(CorpusHeadline, PlainTextDocument)
CorpusHeadline = tm_map(CorpusHeadline, removePunctuation)
CorpusHeadline = tm_map(CorpusHeadline, removeWords, stopwords("english"))
CorpusHeadline = tm_map(CorpusHeadline, stemDocument)

headline_dtm = DocumentTermMatrix(CorpusHeadline, control = list(weighting = function(x) weightTfIdf(x, normalize = FALSE), stopwords = TRUE))
headline_dtm_sparse = removeSparseTerms(headline_dtm, 0.98)
HeadlineWords = as.data.frame(as.matrix(headline_dtm_sparse))
colnames(HeadlineWords) = paste("headline", make.names(colnames(HeadlineWords)), sep="_")

HeadlineWordsTrain = head(HeadlineWords, nrow(train))
HeadlineWordsTest = tail(HeadlineWords, nrow(test))

train = cbind(train,HeadlineWordsTrain)
test = cbind(test,HeadlineWordsTest)

train$Headline = NULL
test$Headline = NULL

#Abstract and snippet are the same for over 99% of articles. No reason to use snippet. Do TDIDF on abstract
CorpusAbstract = Corpus(VectorSource(c(train$Abstract, test$Abstract)))
CorpusAbstract = tm_map(CorpusAbstract, tolower)
CorpusAbstract = tm_map(CorpusAbstract, PlainTextDocument)
CorpusAbstract = tm_map(CorpusAbstract, removePunctuation)
CorpusAbstract = tm_map(CorpusAbstract, removeWords, stopwords("english"))
CorpusAbstract = tm_map(CorpusAbstract, stemDocument)

abs_dtm = DocumentTermMatrix(CorpusAbstract, control = list(weighting = function(x) weightTfIdf(x, normalize = FALSE), stopwords = TRUE))

abs_dtm_sparse = removeSparseTerms(abs_dtm, 0.99)
abs_dtm_sparse 
AbstractWords = as.data.frame(as.matrix(abs_dtm_sparse))
colnames(AbstractWords) = make.names(colnames(AbstractWords))

AbstractWordsTrain = head(AbstractWords, nrow(train))
AbstractWordsTest = tail(AbstractWords, nrow(test))

train = cbind(train,AbstractWordsTrain)
test = cbind(test,AbstractWordsTest)

train$Abstract= NULL
test$Abstract = NULL
train$Snippet = NULL
test$Snippet = NULL

#Basic text preprocessing done

#Try a simple logistic regression model
glm_model = glm(targets~., data=train, family=binomial)
train_pred = predict(glm_model, data=train)

rocCurve = roc(response=targets, predictor=train_pred)
auc(rocCurve)
#AUC on only train 0.949. probably overfit. USE caret and cross validate

control = trainControl(method = "repeatedcv",
                         number = 5,
                         verboseIter = TRUE,
                         repeats = 1
)


set.seed(12)
glm_model_caret = train(targets~., data=train, method="glm", trControl=control)

train_pred = predict(glm_model_caret, newdata=train)

rocCurve = roc(response=targets, predictor=train_pred)
auc(rocCurve)
#AUC 0.9436. Still probably overfitting. Try submitting just to get a baseline prediction in.

test_pred = predict(glm_model_caret, newdata=test)
test_pred = ifelse(test_pred < 0, 0, test_pred)
test_pred = ifelse(test_pred > 1, 1, test_pred)

submission_1 = data.frame(UniqueID=test_ID, Probability1=test_pred)
write.csv(submission_1,"aa2_sub1_logistic.csv", row.names=FALSE)
#LB AUC 0.91889. 9th place. Better result than I expected.


#Try logisitc again with some data center and scaling
#Center and scale the data
preped_data = preProcess(train[4:236], method=c("center", "scale"))
train = cbind(train[1:3], predict(preped_data, train[4:236]))
test = cbind(test[1:3], predict(preped_data, test[4:236]))

#Glm_net model with caret tuned for ROC
control = trainControl(method = "repeatedcv",
                       number = 2,
                       verboseIter = TRUE,
                       repeats = 2,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

grid = expand.grid(  alpha= c(0.95), lambda =c(0.005))

glm_model_caret = train(targets~., data=train, method="glmnet", trControl=control, tuneGrid = grid, metric="ROC")

train_pred = predict(glm_model_caret, newdata=train, type="prob")

rocCurve = roc(response=targets, predictor=train_pred$popular)
auc(rocCurve)

#In-train AUC on cross validated model: 0.9394

test_pred = predict(glm_model_caret, newdata=test, type="prob")

submission_2 = data.frame(UniqueID=test_ID, Probability1=test_pred$popular)

write.csv(submission_2,"aa2_sub2_glmnet.csv", row.names=FALSE)
#LB AUC 0.92464. With alpha=0.95 Third place!



#Try random forests.

control = trainControl(method = "repeatedcv",
                       number = 2,
                       verboseIter = TRUE,
                       repeats = 2,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

set.seed(12)
grid = expand.grid(  mtry= c(25))

rf_model_caret = train(targets~., data=train, method="rf", trControl=control, tuneGrid = grid, metric="ROC", ntree=50, nodesize=1)

train_pred = predict(rf_model_caret, newdata=train, type="prob")

rocCurve = roc(response=targets, predictor=train_pred$popular)
auc(rocCurve)
#In-train AUC on cross validated model: 0.9979. Must be overfittig... Try sumbission anyway

test_pred = predict(rf_model_caret, newdata=test, type="prob")

submission_3 = data.frame(UniqueID=test_ID, Probability1=test_pred$popular)

write.csv(submission_3,"aa2_sub3_rf.csv", row.names=FALSE)
#LB Score: 0.91928.



#GBM
set.seed(12)

control = trainControl(method = "repeatedcv",
                       number = 2,
                       verboseIter = TRUE,
                       repeats = 2,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

grid = expand.grid(  n.trees= c(200), interaction.depth=c(6),shrinkage=c(0.025))

rf_model_caret = train(targets~., data=train, method='gbm', trControl=control, tuneGrid = grid, metric="ROC")

train_pred = predict(rf_model_caret, newdata=train, type="prob")

rocCurve = roc(response=targets, predictor=train_pred$popular)
auc(rocCurve)
#In-train AUC on cross validated model: 0.9436

test_pred = predict(rf_model_caret, newdata=test, type="prob")

submission_4 = data.frame(UniqueID=test_ID, Probability1=test_pred$popular)
write.csv(submission_4,"aa2_sub4_gbm.csv", row.names=FALSE)
#LB Score: 0.92186


#Ensemble of all 4 models: logistic, glm_net, rf, and gbm
ensemble_pred = (submission_1$Probability1+submission_2$Probability1+submission_3$Probability1+submission_4$Probability1)/4

submission_5 = data.frame(UniqueID=test_ID, Probability1=ensemble_pred)
write.csv(submission_5,"aa2_sub5_ensemble.csv", row.names=FALSE)
#LB Score: 0.93320  First place! Won't last but fun for now!



#Try to get a random forest that does not overfit so much
control = trainControl(method = "repeatedcv",
                       number = 5,
                       verboseIter = TRUE,
                       repeats = 5,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

set.seed(12)
grid = expand.grid(mtry= c(30))

rf_model_caret = train(targets~., data=train, method="rf", trControl=control, tuneGrid = grid, metric="ROC", ntree=50, nodesize=1, maxnodes=250)

train_pred = predict(rf_model_caret, newdata=train, type="prob")

rocCurve = roc(response=targets, predictor=train_pred$popular)
auc(rocCurve)

test_pred = predict(rf_model_caret, newdata=test, type="prob")

submission_6 = data.frame(UniqueID=test_ID, Probability1=test_pred$popular)
write.csv(submission_6,"aa2_sub6_rf.csv", row.names=FALSE)
#LB: 0.90872. Overfit model was better...



#Try a single-layer nueral network
control = trainControl(method = "repeatedcv",
                       number = 2,
                       verboseIter = TRUE,
                       repeats = 2,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

set.seed(12)
grid = expand.grid(size= c(10),decay=c(0.01))

rf_model_caret = train(targets~., data=train, method='nnet', trControl=control, tuneGrid = grid, metric="ROC", MaxNWts = 10000, maxit = 100)

train_pred = predict(rf_model_caret, newdata=train, type="prob")

rocCurve = roc(response=targets, predictor=train_pred$popular)
auc(rocCurve)
#In-train AUC:0.9914. More overfitting with only 10 hidden units...

test_pred = predict(rf_model_caret, newdata=test, type="prob")

submission_7 = data.frame(UniqueID=test_ID, Probability1=test_pred$popular)

write.csv(submission_7,"aa2_sub7_nn.csv", row.names=FALSE)
#LB: 0.85254 Vast overfit...


#SVM
control = trainControl(method = "repeatedcv",
                       number = 2,
                       verboseIter = TRUE,
                       repeats = 2,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

set.seed(12)
grid = expand.grid(sigma= c(1), C=c(1))

rf_model_caret = train(targets~., data=train, method='svmRadial', trControl=control, tuneGrid = grid, metric="ROC")

train_pred = predict(rf_model_caret, newdata=train, type="prob")

rocCurve = roc(response=targets, predictor=train_pred$popular)
auc(rocCurve)
#In-train AUC:0.9974. Seems almost every complex model will overfit

test_pred = predict(rf_model_caret, newdata=test, type="prob")

submission_8 = data.frame(UniqueID=test_ID, Probability1=test_pred$popular)

write.csv(submission_8,"aa2_sub8_rbfsvm.csv", row.names=FALSE)
#0.64698 horrid!


#CART
control = trainControl(method = "repeatedcv",
                       number = 5,
                       verboseIter = TRUE,
                       repeats = 5,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

set.seed(12)
grid = expand.grid(cp=c(0.0001))

rf_model_caret = train(targets~., data=train, method='rpart', trControl=control, tuneGrid = grid, metric="ROC", control=as.list(c(maxdepth=12, minbucket=1)))

train_pred = predict(rf_model_caret, newdata=train, type="prob")

rocCurve = roc(response=targets, predictor=train_pred$popular)
auc(rocCurve)
#In-train AUC: 0.9517

test_pred = predict(rf_model_caret, newdata=test, type="prob")

submission_9 = data.frame(UniqueID=test_ID, Probability1=test_pred$popular)

write.csv(submission_9,"aa2_sub9_cart.csv", row.names=FALSE)
#0.90394... Not too good!
