#Analytics Edge Kaggle competition #2
#Retry models with more terms and holdout data for validation.

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
library(MASS)
library(NLP)
library(RWeka)

train = read.csv("NYTimesBlogTrain.csv", stringsAsFactors=FALSE)
test = read.csv("NYTimesBlogTest.csv", stringsAsFactors=FALSE)

sample_sub = read.csv("SampleSubmission.csv")


train$PubDate = strptime(train$PubDate, "%Y-%m-%d %H:%M:%S")
test$PubDate = strptime(test$PubDate, "%Y-%m-%d %H:%M:%S")

train$Weekday =train$PubDate$wday
test$Weekday = test$PubDate$wday

train$Hour = train$PubDate$hour 
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

BigramTokenizer <- function(x) {RWeka::NGramTokenizer(x, RWeka::Weka_control(min = 1, max = 2))} 

#Add: tokenize = BigramTokenizer, to list of contol arguments to document term matrix to compute n-grams

#Convert headlines using TDIDF weighting
CorpusHeadline = Corpus(VectorSource(c(train$Headline, test$Headline)))
CorpusHeadline = tm_map(CorpusHeadline, tolower)
CorpusHeadline = tm_map(CorpusHeadline, PlainTextDocument)
CorpusHeadline = tm_map(CorpusHeadline, removePunctuation)
CorpusHeadline = tm_map(CorpusHeadline, removeWords, stopwords("english"))
CorpusHeadline = tm_map(CorpusHeadline, stemDocument)

headline_dtm = DocumentTermMatrix(CorpusHeadline, control = list(weighting = function(x) weightTfIdf(x, normalize = FALSE), tokenize = BigramTokenizer, stopwords = TRUE))
headline_dtm_sparse = removeSparseTerms(headline_dtm, 0.991)
headline_dtm_sparse
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

abs_dtm = DocumentTermMatrix(CorpusAbstract, control = list(weighting = function(x) weightTfIdf(x, normalize = FALSE), tokenize = BigramTokenizer, stopwords = TRUE))

abs_dtm_sparse = removeSparseTerms(abs_dtm, 0.984)
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


#Center and scale the data
preped_data = preProcess(train[4:ncol(train)], method=c("center", "scale"))
train = cbind(train[1:3], predict(preped_data, train[4:ncol(train)]))
test = cbind(test[1:3], predict(preped_data, test[4:ncol(train)]))


#For validation, use this code to split train into train and validation sets:
train$pop= targets
set.seed(12)
in_train = createDataPartition(train$pop, p = .75,
                               list = FALSE,
                               times = 1)

in_train2 = createDataPartition(train$pop, p = .75,
                               list = FALSE,
                               times = 1)
valid2 = train[ -in_train2,]
train2 = train[ in_train2,]

valid = train[ -in_train,]
train = train[ in_train,]

targets2 = train2$pop
valid_targets2 = valid2$pop
train2$pop = NULL
valid2$pop = NULL

targets = train$pop 
valid_targets = valid$pop
train$pop = NULL
valid$pop = NULL




#Text preprocessing done

#Try a simple logistic regression model
control = trainControl(method = "none",
                       number = 5,
                       verboseIter = TRUE,
                       repeats = 1,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

set.seed(12)
glm_model_caret = train(targets~., data=train, method="glm", trControl=control, metric="ROC")

glm_valid_pred = predict(glm_model_caret, newdata=valid, type="prob")

rocCurve = roc(response=valid_targets, predictor=glm_valid_pred$popular )
lm_auc = auc(rocCurve)
lm_auc


#AUC 0.9217 on validation holdout with 16 headline variables and 142 abstract variables. (headline 0.985, abstract 0.9875)

#AUC 0.9101 on validation holdout with 8 headline variables and 222 abstract variables. (headline 0.98, abstract 0.99)

#AUC 0.9274 on validation holdout with 16 headline variables and 89 abstract variables. (headline 0.985, abstract 0.984)

#AUC 0.9284 on validation holdout with 29 headline variables and 89 abstract variables. (headline 0.988, abstract 0.984)

#AUC 0.9312 on validation holdout with 42 headline variables and 89 abstract variables. (headline 0.99, abstract 0.984)

#AUC 0.9317 on validation holdout with 52 headline variables and 89 abstract variables. (headline 0.991, abstract 0.984)



#Glm_net model with caret tuned for ROC
control = trainControl(method = "none",
                       number = 5,
                       verboseIter = TRUE,
                       repeats = 1,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

grid = expand.grid(  alpha= c(0.98), lambda =c(0.0025))

glm_model_caret = train(targets~., data=train, method="glmnet", trControl=control, tuneGrid = grid, metric="ROC")

glmnet_valid_pred = predict(glm_model_caret, newdata=valid, type="prob")

rocCurve = roc(response=valid_targets, predictor=glmnet_valid_pred$popular )
glm_net = auc(rocCurve)
glm_net 
#AUC 0.9333 on vh 16 headline 142 abstract variables. (headline 0.985, abstract 0.9875) alpha= c(0.5), lambda =c(0.004)

#AUC 0.9261 on vh 8 headline variables and 222 abstract variables. (headline 0.98, abstract 0.99)

#AUC 0.9333 on vh  16 headline variables and 89 abstract variables. (headline 0.985, abstract 0.984)  alpha= c(0.98), lambda =c(0.003)

#AUC 0.9349 on vh  29 headline variables and 89 abstract variables. (headline 0.988, abstract 0.984) alpha= c(0.98), lambda =c(0.003)

#AUC 0.9379 on vh 42 headline variables and 89 abstract variables. (headline 0.99, abstract 0.984) alpha= c(0.98), lambda =c(0.0025)

#AUC 0.9376 on vh 52 headline variables and 89 abstract variables. (headline 0.991, abstract 0.984) alpha= c(0.98), lambda =c(0.0025)

#AUC 0.9399 with (headline 0.991, abstract 0.984) 2-grams and removed uninformed features (cutoff-0.94352). alpha= c(0.98), lambda =c(0.0025)


#Try random forests.
control = trainControl(method = "none",
                       number = 2,
                       verboseIter = TRUE,
                       repeats = 2,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

set.seed(12)
grid = expand.grid(  mtry= c(14))

rf_model_caret = train(targets~., data=train, method="rf", trControl=control, tuneGrid = grid, metric="ROC", ntree=800, nodesize=1)

varImp(rf_model_caret)

rf_valid_pred = predict(rf_model_caret, newdata=valid, type="prob")

rocCurve = roc(response=valid_targets, predictor=rf_valid_pred$popular )
rf_auc= auc(rocCurve)
rf_auc

tpred = predict(rf_model_caret, newdata=test, type="prob")
sub = data.frame(UniqueID=test_ID, Probability1=tpred$popular)
write.csv(sub, "lessrf.csv", row.names=FALSE) 


#AUC  0.9338 vh 16 headline variables and 142 abstract variables. (headline 0.985, abstract 0.9875)  mtry= c(20), ntree=50

#AUC 0.9274 vh 8 headline variables and 222 abstract variables. (headline 0.98, abstract 0.99) mtry= c(20), ntree=50

#AUC 0.9361 vh 16 headline variables and 89 abstract variables. (headline 0.985, abstract 0.984)  mtry= c(18), ntree=300

#AUC 0.9387 vh 29 headline variables and 89 abstract variables. (headline 0.988, abstract 0.984)mtry= c(15), ntree=300

#AUC 0.9402 on vh 42 headline variables and 89 abstract variables. (headline 0.99, abstract 0.984) mtry= c(15), ntree=300

#AUC 0.942 on vh 52 headline variables and 89 abstract variables. (headline 0.991, abstract 0.984) mtry= c(12), ntree=300

#AUC 0.9432 with (headline 0.991, abstract 0.984) 2-grams and removing uniformative features (cutoff 0.94352).  mtry= c(14), ntree=800


#GBM
set.seed(12)

control = trainControl(method = "none",
                       number = 2,
                       verboseIter = TRUE,
                       repeats = 2,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

grid = expand.grid(  n.trees= c(300), interaction.depth=c(30),shrinkage=c(0.0125))

rf_model_caret = train(targets~., data=train, method='gbm', trControl=control, tuneGrid = grid, metric="ROC")

gbm_valid_pred = predict(rf_model_caret, newdata=valid, type="prob")

rocCurve = roc(response=valid_targets, predictor=gbm_valid_pred$popular )
auc(rocCurve)
#AUC 0.937 on vh 16 headline variables and 142 abstract variables. (headline 0.985, abstract 0.9875) n.trees= c(150), interaction.depth=c(25),shrinkage=c(0.025)

#AUC 0.9312 on vh 8 headline variables and 222 abstract variables. (headline 0.98, abstract 0.99) n.trees= c(150), interaction.depth=c(30),shrinkage=c(0.025)

#AUC 0.9378 vh 16 headline variables and 89 abstract variables. (headline 0.985, abstract 0.984)  n.trees= c(300), interaction.depth=c(30),shrinkage=c(0.0125)

#AUC 0.9372 on vh 29 headline variables and 89 abstract variables. (headline 0.988, abstract 0.984) n.trees= c(300), interaction.depth=c(30),shrinkage=c(0.0125)

#AUC 0.9382 on vh 42 headline variables and 89 abstract variables. (headline 0.99, abstract 0.984) n.trees= c(300), interaction.depth=c(30),shrinkage=c(0.0125)

#AUC 0.9383 on vh 52 headline variables and 89 abstract variables. (headline 0.991, abstract 0.984)  n.trees= c(300), interaction.depth=c(30),shrinkage=c(0.0125)


#Potentially uninformative features= ['headline_billion', 'headline_say', 'headline_test', 'appear', 'book', 'deal', 'famili', 'make', 'now', 'offer', 'peopl', 'plan', 'presid', 'public', 'republican', 'said', 'two', 'use', 'way', 'work']

#Keep informative features (determined by building n leave 1 out models with xgb)
informative1 = c('headline_X2014', 'headline_X2015', 'headline_X6.qs', 'headline_agenda', 'headline_art', 'headline_bank', 'headline_big', 'headline_busi', 'headline_can', 'headline_china', 'headline_clip', 'headline_clip.report', 'headline_daili', 'headline_daili.clip', 'headline_daili.report', 'headline_deal', 'headline_ebola', 'headline_fashion', 'headline_fashion.week', 'headline_first', 'headline_get', 'headline_hous', 'headline_london', 'headline_make', 'headline_million', 'headline_morn', 'headline_morn.agenda', 'headline_music', 'headline_new', 'headline_new.york', 'headline_news', 'headline_obama', 'headline_offer', 'headline_pari', 'headline_pictur', 'headline_plan', 'headline_qs.news', 'headline_read', 'headline_report', 'headline_senat', 'headline_show', 'headline_small', 'headline_small.busi', 'headline_springsumm', 'headline_springsumm.2015', 'headline_take', 'headline_talk', 'headline_time', 'headline_today', 'headline_video', 'headline_week', 'headline_will', 'headline_word', 'headline_word.day', 'headline_year', 'headline_york', 'headline_york.today', 'american', 'archiv', 'articl', 'back', 'bank', 'best', 'busi', 'can', 'chang', 'china', 'collect', 'come', 'design', 'diari', 'discuss', 'editor', 'fashion', 'fashion.week', 'first', 'former', 'fund', 'get', 'group', 'herald', 'herald.tribun', 'highlight', 'hous', 'includ', 'intern', 'intern.herald', 'just', 'last', 'like', 'look', 'mani', 'market', 'million', 'nation', 'new', 'new.york', 'news', 'open', 'photo', 'presid.obama', 'report', 'say', 'scene', 'servic', 'share', 'stori', 'take', 'talk', 'tribun', 'tribun.archiv', 'unit.state', 'will', 'word', 'world', 'write', 'year', 'york', 'york.time')

informative2 = c('headline_X2014', 'headline_X2015', 'headline_X6.qs', 'headline_art', 'headline_bank', 'headline_big', 'headline_billion', 'headline_busi', 'headline_can', 'headline_china', 'headline_clip', 'headline_clip.report', 'headline_daili', 'headline_daili.clip', 'headline_daili.report', 'headline_deal', 'headline_ebola', 'headline_fashion', 'headline_fashion.week', 'headline_first', 'headline_hous', 'headline_london', 'headline_make', 'headline_million', 'headline_morn', 'headline_morn.agenda', 'headline_music', 'headline_new.york', 'headline_news', 'headline_offer', 'headline_pari', 'headline_pictur', 'headline_plan', 'headline_polit', 'headline_qs.news', 'headline_rais', 'headline_read', 'headline_report', 'headline_say', 'headline_senat', 'headline_show', 'headline_small', 'headline_small.busi', 'headline_springsumm', 'headline_springsumm.2015', 'headline_take', 'headline_test', 'headline_time', 'headline_today', 'headline_video', 'headline_week', 'headline_will', 'headline_word', 'headline_word.day', 'headline_year', 'headline_york', 'headline_york.today', 'appear', 'archiv', 'articl', 'best', 'billion', 'book', 'busi', 'china', 'collect', 'come', 'deal', 'design', 'editor', 'fashion', 'fashion.week', 'former', 'fund', 'govern', 'group', 'herald', 'herald.tribun', 'highlight', 'hous', 'includ', 'intern', 'intern.herald', 'just', 'last', 'manag', 'market', 'may', 'new', 'new.york', 'news', 'obama', 'offer', 'one', 'open', 'peopl', 'photo', 'plan', 'presid.obama', 'public', 'republican', 'say', 'scene', 'senat', 'servic', 'show', 'take', 'time', 'tribun', 'tribun.archiv', 'two', 'unit', 'unit.state', 'use', 'week', 'work', 'world', 'write', 'york', 'york.time')

#Combined vector of informative features(full join)
informative3 = c('just', 'show', 'photo', 'headline_word', 'govern', 'headline_week', 'headline_fashion.week', 'headline_report', 'headline_talk', 'group', 'headline_clip', 'articl', 'busi', 'headline_big', 'senat', 'presid.obama', 'diari', 'get', 'headline_qs.news', 'nation', 'tribun', 'headline_china', 'new', 'report', 'world', 'republican', 'discuss', 'bank', 'headline_make', 'scene', 'like', 'headline_X6.qs', 'headline_york.today', 'headline_take', 'herald', 'collect', 'manag', 'week', 'work', 'headline_springsumm.2015', 'deal', 'headline_art', 'back', 'intern', 'headline_daili.report', 'design', 'headline_pictur', 'year', 'headline_small.busi', 'best', 'headline_polit', 'headline_bank', 'headline_today', 'headline_new', 'headline_year', 'unit', 'headline_get', 'public', 'headline_senat', 'million', 'headline_pari', 'headline_busi', 'headline_offer', 'herald.tribun', 'china', 'york', 'headline_news', 'new.york', 'come', 'headline_plan', 'last', 'intern.herald', 'headline_ebola', 'headline_york', 'mani', 'chang', 'first', 'headline_obama', 'headline_springsumm', 'headline_first', 'one', 'fashion', 'headline_billion', 'news', 'open', 'market', 'headline_deal', 'use', 'headline_daili.clip', 'two', 'stori', 'includ', 'headline_X2014', 'headline_X2015', 'editor', 'headline_show', 'headline_read', 'offer', 'peopl', 'headline_rais', 'headline_clip.report', 'fund', 'headline_new.york', 'headline_time', 'headline_daili', 'former', 'obama', 'headline_test', 'word', 'look', 'hous', 'servic', 'york.time', 'say', 'will', 'headline_hous', 'headline_word.day', 'archiv', 'can', 'highlight', 'headline_morn.agenda', 'share', 'headline_video', 'headline_can', 'unit.state', 'headline_million', 'american', 'headline_morn', 'write', 'headline_london', 'book', 'take', 'tribun.archiv', 'may', 'headline_fashion', 'headline_agenda', 'fashion.week', 'plan', 'headline_will', 'appear', 'billion', 'headline_small', 'time', 'headline_say', 'headline_music', 'talk')

train_temp = train
valid_temp = valid

train_temp2 = train2
valid_temp2 = valid2


#Code to keep only informative vars
train = cbind(train_temp[1:6], train_temp[informative3])
valid= cbind(valid_temp[1:6], valid_temp[informative3])

train2 = cbind(train_temp2[1:6], train_temp2[informative3])
valid2 = cbind(valid_temp2[1:6], valid_temp2[informative3])


features = labels(train2)[2][[1]][7:167]
valid_1_results = c()
valid_2_results = c()

for (label in features){
  train = train_temp 
  valid= valid_temp
  
  train2 = train_temp2
  valid2 = valid_temp2
  
  print(label)
  train[label] = NULL  
  valid[label] = NULL

  train2[label] = NULL  
  valid2[label] = NULL
  
# #XGB
set.seed(121)
combined = rbind(train,valid)
combined_m = data.frame(lapply(combined[c("NewsDesk","SectionName","SubsectionName")] , factor))
combined_m = model.matrix(~ ., combined_m) 
combined_m = data.frame(combined_m)

train_m = cbind(combined[1:4900, 4:ncol(combined)], combined_m[1:4900,])
valid_m = cbind(combined[4901:nrow(combined), 4:ncol(combined)], combined_m[4901:nrow(combined),])

train_m = matrix(as.numeric(data.matrix(train_m)),ncol=ncol(train_m))
valid_m = matrix(as.numeric(data.matrix(valid_m)),ncol=ncol(valid_m))

num_ttargets = as.numeric(targets)-1
num_v_ttargets = as.numeric(valid_targets)-1

set.seed(13)
xgb_model = xgboost(data=train_m, label=num_ttargets, nrounds=2000, verbose=0, eta=0.01, gamma=0.105, max_depth=4, min_child_weight=0.95, subsample=0.5, objective="binary:logistic", eval_metric="auc")

xgb_preds = predict(xgb_model, valid_m)

rocCurve = roc(response=valid_targets, predictor=xgb_preds)
xgb_auc = auc(rocCurve)[1]
print (xgb_auc)
# 
# valid_1_results = c(valid_1_results,xgb_auc)



#Secondary validation
set.seed(121)
combined2 = rbind(train2,valid2)
combined_m2 = data.frame(lapply(combined2[c("NewsDesk","SectionName","SubsectionName")] , factor))
combined_m2 = model.matrix(~ ., combined_m2) 
combined_m2 = data.frame(combined_m2)

train_m2 = cbind(combined2[1:4900, 4:ncol(combined2)], combined_m2[1:4900,])
valid_m2 = cbind(combined2[4901:nrow(combined2), 4:ncol(combined2)], combined_m2[4901:nrow(combined2),])

train_m2 = matrix(as.numeric(data.matrix(train_m2)),ncol=ncol(train_m2))
valid_m2 = matrix(as.numeric(data.matrix(valid_m2)),ncol=ncol(valid_m2))

num_ttargets2 = as.numeric(targets2)-1
num_v_ttargets2 = as.numeric(valid_targets2)-1

set.seed(13)
xgb_model2 = xgboost(data=train_m2, label=num_ttargets2, nrounds=2000, verbose=0, eta=0.01, gamma=0.105, max_depth=4, min_child_weight=0.95, subsample=0.5, objective="binary:logistic", eval_metric="auc")

xgb_preds2 = predict(xgb_model2, valid_m2)

rocCurve2 = roc(response=valid_targets2, predictor=xgb_preds2)
xgb_auc2 = auc(rocCurve2)[1]
print (xgb_auc2)

valid_2_results = c(valid_2_results,xgb_auc2)

}


#AUC  0.9417 on vh 16 headline variables and 142 abstract variables. nrounds=260, verbose=1, eta=0.05, gamma=0.1, max_depth=6, min_child_weight=1, subsample=0.5

#AUC 0.9394 on vh 8 headline variables and 222 abstract variables. (headline 0.98, abstract 0.99) nrounds=260, verbose=1, eta=0.05, gamma=0.1, max_depth=6, min_child_weight=1, subsample=0.5

#AUC  0.9422 on vh 16 headline variables and 89 abstract variables. nrounds=600, verbose=1, eta=0.025, gamma=0.105, max_depth=4, min_child_weight=0.95, subsample=0.5

#AUC 0.94 on vh 29 headline variables and 89 abstract variables. (headline 0.988, abstract 0.984) nrounds=600, verbose=1, eta=0.025, gamma=0.105, max_depth=4, min_child_weight=0.95, subsample=0.5

#AUC 0.9437 on vh 42 headline variables and 89 abstract variables. (headline 0.99, abstract 0.984) nrounds=1000, verbose=1, eta=0.015, gamma=0.105, max_depth=4, min_child_weight=0.95, subsample=0.5

#AUC 0.9441 on vh 52 headline variables and 89 abstract variables. (headline 0.991, abstract 0.984) nrounds=1500, verbose=1, eta=0.01, gamma=0.105, max_depth=4, min_child_weight=0.95, subsample=0.5

#AUC 0.9477597 with (headline 0.991, abstract 0.984) 2-grams and removing uniformative features (cutoff-0.94352).  nrounds=2700, verbose=1, eta=0.01, gamma=0.105, max_depth=4, min_child_weight=0.95, subsample=0.5



set.seed(13)
xgb_model2 = xgboost(data=train_m, label=num_ttargets, nrounds=1000, verbose=1, eta=0.01, gamma=0.105, max_depth=5, min_child_weight=0.95, subsample=0.5, objective="reg:logistic", eval_metric="auc", booster="gblinear", alpha=5)

xgb_preds2 = predict(xgb_model2, valid_m)

rocCurve = roc(response=valid_targets, predictor=xgb_preds2)
auc(rocCurve)

#Using linear booster: 0.9372 on vh 52 headline variables and 89 abstract variables. nrounds=1000, verbose=1, eta=0.01, gamma=0.105, max_depth=5, min_child_weight=0.95, subsample=0.5, objective="reg:logistic", eval_metric="auc", booster="gblinear", alpha=5)


#Validation shows 52 headline vars and 89 abstract vars gives good performance across all 5 models on a holdout validation set. Try running and submitting each model using 52 headline vars and 89 abstract vars on the test set.


#FDA model
control = trainControl(method = "none",
                       number = 5,
                       verboseIter = TRUE,
                       repeats = 3
)

grid = expand.grid( degree = c(1), nprune=c(25))

fda_model_caret = train(targets~., data=train, method='fda', trControl=control, tuneGrid = grid)

fda_valid_pred = predict(fda_model_caret, newdata=valid, type="prob")

rocCurve = roc(response=valid_targets, predictor=fda_valid_pred$popular)
auc(rocCurve)
#AUC 0.9301




#Ensembling

glmnet_weight = 0.425
rf_weight = 0.425
xgb_weight = 1


ensemble_pred = ((glmnet_valid_pred$popular*glmnet_weight)+(rf_valid_pred$popular*rf_weight)+(xgb_preds*xgb_weight))/sum(glmnet_weight+rf_weight+xgb_weight)


ensemble_roc = roc(response=valid_targets, predictor=ensemble_pred)
ensemble_auc = auc(ensemble_roc)
ensemble_auc 
#Best ensemble AUC = 0.95 (all models using the same features: 2-grams and removing uniformative features (cutoff-0.94352))




control = trainControl(method = "none",
                       number = 2,
                       verboseIter = TRUE,
                       repeats = 2,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

set.seed(12)
grid = expand.grid( xgenes= c(1))

prac_model_caret = train(targets~., data=train, method="rocc", trControl=control, tuneGrid = grid, metric="ROC")

prac_valid_pred = predict(prac_model_caret, newdata=valid, type="prob")

rocCurve = roc(response=valid_targets, predictor=prac_valid_pred$popular )
auc(rocCurve)


