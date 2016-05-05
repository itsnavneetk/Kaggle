#Analytics Edge Kaggle competition #2
#Submission code for tuned features and models

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
test = cbind(test[1:3], predict(preped_data, test[4:ncol(test)]))

#Text preprocessing done

#Remove uninformative features (determined in validation)
informative = c('just', 'show', 'photo', 'headline_word', 'govern', 'headline_week', 'headline_fashion.week', 'headline_report', 'headline_talk', 'group', 'headline_clip', 'articl', 'busi', 'headline_big', 'senat', 'presid.obama', 'diari', 'get', 'headline_qs.news', 'nation', 'tribun', 'headline_china', 'new', 'report', 'world', 'republican', 'discuss', 'bank', 'headline_make', 'scene', 'like', 'headline_X6.qs', 'headline_york.today', 'headline_take', 'herald', 'collect', 'manag', 'week', 'work', 'headline_springsumm.2015', 'deal', 'headline_art', 'back', 'intern', 'headline_daili.report', 'design', 'headline_pictur', 'year', 'headline_small.busi', 'best', 'headline_polit', 'headline_bank', 'headline_today', 'headline_new', 'headline_year', 'unit', 'headline_get', 'public', 'headline_senat', 'million', 'headline_pari', 'headline_busi', 'headline_offer', 'herald.tribun', 'china', 'york', 'headline_news', 'new.york', 'come', 'headline_plan', 'last', 'intern.herald', 'headline_ebola', 'headline_york', 'mani', 'chang', 'first', 'headline_obama', 'headline_springsumm', 'headline_first', 'one', 'fashion', 'headline_billion', 'news', 'open', 'market', 'headline_deal', 'use', 'headline_daili.clip', 'two', 'stori', 'includ', 'headline_X2014', 'headline_X2015', 'editor', 'headline_show', 'headline_read', 'offer', 'peopl', 'headline_rais', 'headline_clip.report', 'fund', 'headline_new.york', 'headline_time', 'headline_daili', 'former', 'obama', 'headline_test', 'word', 'look', 'hous', 'servic', 'york.time', 'say', 'will', 'headline_hous', 'headline_word.day', 'archiv', 'can', 'highlight', 'headline_morn.agenda', 'share', 'headline_video', 'headline_can', 'unit.state', 'headline_million', 'american', 'headline_morn', 'write', 'headline_london', 'book', 'take', 'tribun.archiv', 'may', 'headline_fashion', 'headline_agenda', 'fashion.week', 'plan', 'headline_will', 'appear', 'billion', 'headline_small', 'time', 'headline_say', 'headline_music', 'talk')

train = cbind(train[1:6], train[informative])
test = cbind(test[1:6], test[informative])


# #Logistic regression
# control = trainControl(method = "none",
#                        number = 5,
#                        verboseIter = TRUE,
#                        repeats = 1,
#                        classProbs=TRUE,
#                        summaryFunction= twoClassSummary
# )
# 
# set.seed(12)
# 
# glm_model_caret = train(targets~., data=train, method="glm", trControl=control, metric="ROC")
# 
# log_test_pred = predict(glm_model_caret, newdata=test, type="prob")
# 
# submission_11 = data.frame(UniqueID=test_ID, Probability1=log_test_pred$popular)
# 
# write.csv(submission_11,"aa2_sub11_logistic.csv", row.names=FALSE) 
# #LB AUC: 0.90086 much worse than expected based on validation results


#Glm_net
control = trainControl(method = "none",
                       number = 5,
                       verboseIter = TRUE,
                       repeats = 1,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

grid = expand.grid(  alpha= c(0.98), lambda =c(0.0025))

glmnet_model_caret = train(targets~., data=train, method="glmnet", trControl=control, tuneGrid = grid, metric="ROC")

glmnet_test_pred = predict(glmnet_model_caret, newdata=test, type="prob")

submission_40 = data.frame(UniqueID=test_ID, Probability1=glmnet_test_pred$popular)

write.csv(submission_40,"aa2_sub40_glmnet_new.csv", row.names=FALSE)
#LB AUC: 0.92543 Not much different than my original glmnet

#Random forests
control = trainControl(method = "none",
                       number = 2,
                       verboseIter = TRUE,
                       repeats = 2,
                       classProbs=TRUE,
                       summaryFunction= twoClassSummary
)

set.seed(12)
grid = expand.grid(  mtry= c(14))

rf_model_caret = train(targets~., data=train, method="rf", trControl=control, tuneGrid = grid, metric="ROC", ntree=500, nodesize=1)

rf_test_pred = predict(rf_model_caret, newdata=test, type="prob")

submission_41 = data.frame(UniqueID=test_ID, Probability1=rf_test_pred$popular)

write.csv(submission_41,"aa2_sub41_rf.csv", row.names=FALSE)
#0.93053

# #GBM
# set.seed(12)
# 
# control = trainControl(method = "none",
#                        number = 2,
#                        verboseIter = TRUE,
#                        repeats = 2,
#                        classProbs=TRUE,
#                        summaryFunction= twoClassSummary
# )
# 
# grid = expand.grid(  n.trees= c(300), interaction.depth=c(30),shrinkage=c(0.0125))
# 
# gbm_model_caret = train(targets~., data=train, method='gbm', trControl=control, tuneGrid = grid, metric="ROC")
# 
# gbm_test_pred = predict(gbm_model_caret, newdata=test, type="prob")
# 
# submission_14 = data.frame(UniqueID=test_ID, Probability1=gbm_test_pred$popular)
# 
# write.csv(submission_14,"aa2_sub14_gbm.csv", row.names=FALSE)
# #0.93078


#XGB
set.seed(121)
combined = rbind(train, test)
combined_m = data.frame(lapply(combined[c("NewsDesk","SectionName","SubsectionName")] , factor))
combined_m = model.matrix(~ ., combined_m) 
combined_m = data.frame(combined_m)

train_m = cbind(combined[1:6532, 4:ncol(combined)], combined_m[1:6532,])
test_m = cbind(combined[6533:nrow(combined), 4:ncol(combined)], combined_m[6533:nrow(combined),])

train_m = matrix(as.numeric(data.matrix(train_m)),ncol=ncol(train_m))
test_m = matrix(as.numeric(data.matrix(test_m)),ncol=ncol(test_m))

num_ttargets = as.numeric(targets)-1

set.seed(13)
xgb_model = xgboost(data=train_m, label=num_ttargets, nrounds=2000, verbose=1, eta=0.01, gamma=0.105, max_depth=4, min_child_weight=0.95, subsample=0.5, objective="binary:logistic", eval_metric="auc")

xgb_preds = predict(xgb_model, newdata=test_m)

submission_42 = data.frame(UniqueID=test_ID, Probability1=xgb_preds)

write.csv(submission_42,"aa2_sub42_xgb.csv", row.names=FALSE)
#LB score: 0.93463. with nrounds=2700 and reduced features
#LB: 0.93478 with nrounds=1700 and reduced features (sub 25)









#Ensemble code--------------------------------------------------------
#Ensemble of best XGB, GLMNET and RF
rf = read.csv("aa2_sub41_rf.csv")
glmnet = read.csv("aa2_sub40_glmnet_new.csv")
xgb = read.csv("aa2_sub42_xgb.csv")

ensemble = rf$Probability1*0.4+ glmnet$Probability1*0.4 + xgb$Probability1
ensemble_sub = data.frame(UniqueID=rf$UniqueID, Probability1=ensemble/1.8)

write.csv(ensemble_sub, "aa2_sub43_rf_glmnet_xgb.csv", row.names=FALSE)
#LB Score: 


#Ensemble of best XGB, GLMNET and RF
rf = read.csv("aa2_sub21_rf.csv")
glmnet = read.csv("aa2_sub20_glmnet_new.csv")
xgb = read.csv("aa2_sub22_xgb.csv")

ensemble = rf$Probability1*0.4+ glmnet$Probability1*0.4 + xgb$Probability1
ensemble_sub = data.frame(UniqueID=rf$UniqueID, Probability1=ensemble/1.8)

write.csv(ensemble_sub, "aa2_sub23_rf_glmnet_xgb.csv", row.names=FALSE)
#LB Score: 



#Ensemble of best XGB, GBM and RF
rf = read.csv("aa2_sub13_rf.csv")
gbm = read.csv("aa2_sub14_gbm.csv")
xgb = read.csv("aa2_sub15_xgb.csv")

ensemble = rf$Probability1 + gbm$Probability1 + xgb$Probability1
ensemble_sub = data.frame(UniqueID=rf$UniqueID, Probability1=ensemble/3)

write.csv(ensemble_sub, "aa2_sub16_rf_gbm_xgb.csv", row.names=FALSE)
#LB Score: 0.93474 Not much of an improvement over single xgb model


#Ensemble of best XGB, GLMNET and RF
rf = read.csv("aa2_sub13_rf.csv")
glmnet = read.csv("aa2_sub12_glmnet.csv")
xgb = read.csv("aa2_sub25_xgb.csv")

ensemble = rf$Probability1*0.3 + glmnet$Probability1*0.2 + xgb$Probability1
ensemble_sub = data.frame(UniqueID=rf$UniqueID, Probability1=ensemble/1.5)

write.csv(ensemble_sub, "aa2_sub28_rf_glmnet_xgb.csv", row.names=FALSE)
#LB Score: 0.93599 A bit of improvment giving more weight to XGB and mixing in glmnet


#Ensemble of best XGB, GLMNET and RF
rf = read.csv("aa2_sub13_rf.csv")
glmnet = read.csv("aa2_sub12_glmnet.csv")
xgb = read.csv("aa2_sub15_xgb.csv")

ensemble = rf$Probability1*0.3+ glmnet$Probability1*0.3 + xgb$Probability1
ensemble_sub = data.frame(UniqueID=rf$UniqueID, Probability1=ensemble/2)

write.csv(ensemble_sub, "aa2_sub18_rf_glmnet_xgb.csv", row.names=FALSE)
#LB Score: 0.93590