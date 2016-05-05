# Shelter Animal Outcomes
# https://www.kaggle.com/c/shelter-animal-outcomes

# Read the data
train <- read.csv("train.csv")
test <- read.csv("test.csv")
sample <- read.csv("sample_submission.csv")

# Data exploration & preparation
summary(train)
summary(test)

# train$Name <- NULL
# test$Name <- NULL
train$AnimalID <- NULL
test_ID <- test$ID
test$ID <- NULL

# Convert dates
train$DateTime <- as.POSIXct(train$DateTime)
test$DateTime <- as.POSIXct(test$DateTime)

# Add some date/time-related variables
library(lubridate)
train$year <- year(train$DateTime)
train$month <- month(train$DateTime)
train$wday <- wday(train$DateTime)
train$hour <- hour(train$DateTime)

test$year <- year(test$DateTime)
test$month <- month(test$DateTime)
test$wday <- wday(test$DateTime)
test$hour <- hour(test$DateTime)

train$DateTime <- as.numeric(train$DateTime)
test$DateTime <- as.numeric(test$DateTime)

# Write a function to convert age outcome to numeric age in days
convert <- function(age_outcome){
  split <- strsplit(as.character(age_outcome), split=" ")
  period <- split[[1]][2]
  if (grepl("year", period)){
    per_mod <- 356
  } else if (grepl("month", period)){ 
    per_mod <- 30
  } else if (grepl("week", period)){
    per_mod <- 7
  } else
    per_mod <- 1
  age <- as.numeric(split[[1]][1]) * per_mod
  return(age)
}

train$AgeuponOutcome <- sapply(train$AgeuponOutcome, FUN=convert)
test$AgeuponOutcome <- sapply(test$AgeuponOutcome, FUN=convert)
train[is.na(train)] <- 0  # Fill NA with 0
test[is.na(test)] <- 0

# Remove row with missing sex label and drop the level
train <- train[-which(train$SexuponOutcome == ""),]
train$SexuponOutcome <- droplevels(train$SexuponOutcome)

# Explore the data
# Outcome by animal type
t1 <- table(train$AnimalType, train$OutcomeType)
t1

prop.table(t1, margin=1)
# Notable differences: 
# 1.3% of cats die, while 0.3% of dogs die
# 24.5% of dogs are returned to owner, 4.5% of cats are returned
# 50% of cats are transferred vs. 25% for dogs

# Animal type vs. subtype
t2 <- table(train$AnimalType, train$OutcomeSubtype)
t2
# Almost all animals flagged as "Aggressive" or "Behavior" are dogs
# 1599 cats are "SCRP" (Stray Cat Return Program) animals

# Subtype is not in test set so remove it
train$OutcomeSubtype <- NULL

# Outcome by sex type
t3 <- table(train$OutcomeType, train$SexuponOutcome)
t3

prop.table(t3, margin=2)


# Initial model on holdout validation set
# Remove breed and color for now (many factor levels)
# train$Breed <- NULL
# test$Breed <- NULL
# train$Color <- NULL
# test$Color <- NULL

targets <- train$OutcomeType
train$OutcomeType <- NULL

library(caret)
library(xgboost)
library(MLmetrics)
library(dummies)

# Split into train and validation sets
set.seed(14)
train_index <- createDataPartition(targets,p=0.75,list = FALSE,
                                  times = 1)

train_valid <- train[train_index,]
valid <- train[-train_index,]

t_valid_targets <- targets[train_index]
valid_targets <- targets[-train_index]

# Format for xgboost
set.seed(121)
train_matrix <- matrix(as.numeric(data.matrix(train_valid)),ncol=306)
valid_matrix <- matrix(as.numeric(data.matrix(valid)),ncol=306)

num_targets_train <- as.numeric(t_valid_targets)-1
num_targets_valid <- as.numeric(valid_targets)-1

# Run xgb validation
xgb_model_valid <- xgboost(data=train_matrix, 
                      label=num_targets_train, 
                      nrounds=250, 
                      verbose=1, 
                      eta=0.1, 
                      max_depth=7, 
                      subsample=0.75, 
                      colsample_bytree=0.85,
                      objective="multi:softprob", 
                      eval_metric="mlogloss",
                      num_class=5)


xgb_preds <- predict(xgb_model_valid, valid_matrix)

xgb_preds_matrix <- matrix(xgb_preds, ncol = 5, byrow=TRUE)
target_matrix <- dummy(valid_targets)

#Check validation logloss
MultiLogLoss(xgb_preds_matrix , target_matrix)
#Validation logloss: 0.7754385 (8 simple vars)
#Validation logloss: 0.7409206 (with name length added)
#Validation logloss: 0.7363757 (with breeds and crosses vars)
#Validation logloss: 0.7356038 (color vars added)
#Validation logloss: 0.736373 (color vars and color counts)
#After tuning with all vars approx 0.733

#Submission code
set.seed(121)
full_train_matrix <- matrix(as.numeric(data.matrix(train)),ncol=8)
test_matrix <- matrix(as.numeric(data.matrix(test)),ncol=8)

full_targets_train <- as.numeric(targets)-1

# Run xgb on full train set
xgb_model_test = xgboost(data=full_train_matrix, 
                    label=full_targets_train, 
                    nrounds=200, 
                    verbose=1, 
                    eta=0.05, 
                    max_depth=8, 
                    subsample=0.85, 
                    colsample_bytree=0.85,
                    objective="multi:softprob", 
                    eval_metric="mlogloss",
                    num_class=5)


test_preds <- predict(xgb_model_test, test_matrix)
test_preds_frame <- data.frame(matrix(test_preds, ncol = 5, byrow=TRUE))
colnames(test_preds_frame) <- levels(targets)

submission <- cbind(data.frame(ID=test_ID), test_preds_frame)

write.csv(submission , "shelter_animals_3.csv", row.names=FALSE)
# Submission LB score: 0.75018


# --------------------------------------------------------------------
# Feature and model exploration

# Rerun data cleaning code above but comment out lines that remove name, breed and color

# Rerun validation code above with name length variable added
train$name_len <- sapply(as.character(train$Name),nchar)
test$name_len <- sapply(as.character(test$Name),nchar)

train$Name <- NULL
test$Name <- NULL

# Create indicator vars for breeds and mix
train_breeds <- as.character(train$Breed)
test_breeds <- as.character(test$Breed)
all_breeds <- unique(c(train_breeds,test_breeds))
breed_words <- unique(unlist(strsplit(all_breeds, c("/| Mi")))) 

for (breed in breed_words){
  train[breed] <- as.numeric(grepl(breed, train_breeds))
  test[breed] <- as.numeric(grepl(breed, test_breeds))
}

library(stringr)

train["crosses"] <- str_count(train$Breed, pattern="/")
test["crosses"] <- str_count(test$Breed, pattern="/")

train$Breed <- NULL
test$Breed <- NULL


# Create indicator vars for color
train_colors <- as.character(train$Color)
test_colors <- as.character(test$Color)
all_colors <- unique(c(train_colors,test_colors))
color_words <- unique(unlist(strsplit(all_colors, c("/")))) 

for (color in color_words){
  train[color] <- as.numeric(grepl(color, train_colors))
  test[color] <- as.numeric(grepl(color, test_colors))
}

train["color_count"] <- str_count(train$Color, pattern="/")+1
test["color_count"] <- str_count(test$Color, pattern="/")+1

train$Color <- NULL
test$Color <- NULL



# Submission code
set.seed(121)
full_train_matrix <- matrix(as.numeric(data.matrix(train)),ncol=306)
test_matrix <- matrix(as.numeric(data.matrix(test)),ncol=306)

full_targets_train <- as.numeric(targets)-1

# Run xgb on full train set
xgb_model_test = xgboost(data=full_train_matrix, 
                         label=full_targets_train, 
                         nrounds=400, 
                         verbose=1, 
                         eta=0.05, 
                         max_depth=9, 
                         subsample=0.85, 
                         colsample_bytree=0.85,
                         objective="multi:softprob", 
                         eval_metric="mlogloss",
                         num_class=5)


test_preds <- predict(xgb_model_test, test_matrix)
test_preds_frame <- data.frame(matrix(test_preds, ncol = 5, byrow=TRUE))
colnames(test_preds_frame) <- levels(targets)

submission <- cbind(data.frame(ID=test_ID), test_preds_frame)

write.csv(submission , "shelter_animals_5.csv", row.names=FALSE)
# Submission LB score: 