# Kobe Bryant Shot Selection
# https://www.kaggle.com/c/kobe-bryant-shot-selection

library(ggplot2)
library(lubridate)
library(caret)
library(xgboost)
library(Metrics)
library(plyr)
library(dplyr)
library(zoo)

# Read in the shot data and the sample submission:
data <- read.csv("data.csv")
sample <- read.csv("sample_submission.csv")

# Explore the data
summary(data)

# Check action type vs made
t1 <- table(data$action_type, data$shot_made_flag)
t1 

# Check action type shot percentage
prop.table(t1, margin=1)

# Condense rare bad and good shots into new levels
new_levs <- ifelse(rowSums(t1) < 50 & prop.table(t1, margin=1)[,2] >= 0.5, "rare_good_shot", 
            ifelse(rowSums(t1) < 50 & prop.table(t1, margin=1)[,2] < 0.5,
            "rare_bad_shot",
            levels(data$action_type)))

levels(data$action_type) = new_levs

# Fill 2 NA values as bad shots
data[is.na(data$action_type),"action_type"] <- "rare_bad_shot"

# Check action type again with new levels
t1 <- table(data$action_type, data$shot_made_flag)
t1 

# Check action type shot percentage
prop.table(t1, margin=1)


# Check simple shot type vs made
t2 <- table(data$combined_shot_type, data$shot_made_flag)
t2 

# Check simple shot type shot percentage
prop.table(t2, margin=1)


# Remove some vars:
data$game_id <- NULL            # can determine game by date
data$lat <- NULL                # can determine by matchup
data$lon <- NULL                # can determine by matchup
data$team_name <- NULL          # unused label
data$team_id <- NULL            # unused label

# Checks shooting by period
t3 <- table(data$period,data$shot_made_flag)
t3

# Shooting percentage lowest in 4th quarter:
prop.table(t3, margin=1)

# Convert period, mins remaining, seconds remaining into a single game time var
data$shot_time <- ifelse(data$period < 5, 
                         (data$period-1)*12, 
                         48 + (data$period-5)*5 ) + 
                         (11 - data$minutes_remaining) + 
                         (60 - data$seconds_remaining)/100

# Add indicator vars for clutch and overtime shots
# data$clutch_reg <- ifelse(data$shot_time >= 43 & data$shot_time <= 48,1,0)
# data$overtime <- ifelse(data$shot_time > 48,1,0)

data$period <- NULL
data$minutes_remaining <- NULL
data$seconds_remaining <- NULL

# Explore impact of playoffs on shooting
t4 <- table(data$playoffs,data$shot_made_flag)
t4

# No difference bewteen reg season shooting percentage and playoffs:
prop.table(t4, margin=1)

data$playoffs <- NULL   # No difference and determined by game month

# Explore impact of season on shooting
t5 <- table(data$season,data$shot_made_flag)
t5

# shooting drops off at the end of career:
prop.table(t5, margin=1)

# Create a new variable for home/away and remove matchup (creates extra levels for team-rebrands)
data$home_or_away <- ifelse(grepl("@", as.character(data$matchup)), 1, 0)

data$matchup <- NULL

# Check home advantage
t6 <- table(data$home_or_away, data$shot_made_flag)
t6

prop.table(t6, margin=1)

# About 2 percenage points better at home over his career


# Convert date to date object
data$game_date <- as.POSIXlt(data$game_date)

# Add vars for month and day of week
data$month <- month(data$game_date)
data$day_of_week <- wday(data$game_date)

# Check shooting by month/day
t7 <- table(data$month, data$shot_made_flag)
t7

prop.table(t7, margin=1)

# Shooting improves through regular season
t8 <- table(data$day_of_week, data$shot_made_flag)
t8

prop.table(t8, margin=1)
# Shoots wost on Monday.

# Remove shot zone information for now. These variable are arbitrary ranges made on top of x/y shot coordinates and distance.
data$shot_zone_range <- NULL
data$shot_zone_basic <- NULL
data$shot_zone_area <- NULL

# Add varaible for shot angle (straight on shot = 0, corner shot = 90 degrees)
# data$shot_angle <- ifelse(data$loc_x == 0 & data$loc_y >= 0, 0,
#                    ifelse(data$loc_x == 0, 180,
#                          (atan(abs(data$loc_x)/data$loc_y)*180)/pi))
# 
# data$shot_angle <- ifelse(data$shot_angle<0, abs(data$shot_angle)+90,data$shot_angle)


# Convert season to numeric indicator
data$season <- as.numeric(data$season)

# Sort data into event order (sort by date and then event_id)
data$num_date <- as.numeric(data$game_date)
data$game_date <- NULL
data <- arrange(data, num_date, game_event_id)
data$game_event_id <- NULL
data$num_date <- NULL

# Add some variables related to "hot streaks"
avg_na <- function(x){
  return ( mean(x[!is.na(x)]) )
}
  
# data$last_25 <- c(rep(NA,25), rollapply(data$shot_made_flag, 
#                                        width=25, 
#                                        FUN= avg_na)[1:(30697-25)])
  
# data$last_10 <-c(rep(NA,10), rollapply(data$shot_made_flag, 
#                                      width=10, 
#                                      FUN= avg_na)[1:(30697-10)])
# 
# data$last_5 <- c(rep(NA,5), rollapply(data$shot_made_flag, 
#                            width=5, 
#                            FUN= avg_na)[1:(30697-5)])
#   
# data$last_1 <- c(NA, data$shot_made_flag[1:(30697-1)])


data$game_event_id <- NULL


# Separate train data from test data
mask <- is.na(data$shot_made_flag)
train <- data[! mask,]
test <- data[mask,]

test$shot_made_flag <- NULL
targets <- as.factor(train$shot_made_flag)
levels(targets) <- c("missed","made")


# # Create a shot chart
# shot_chart <- ggplot(data=train, aes(x=loc_x, y=loc_y, 
#                                      color=targets)) +
#               geom_point(size=2, alpha=0.5) +
#               scale_color_manual(values=c(missed="red", made="dark green"))
#               
# shot_chart
# 
# 
# # Missed shots
# miss_chart <- ggplot(data=train[targets=="missed",], aes(x=loc_x, y=loc_y)) +
#   geom_point(size=2, alpha=0.5, color="dark red") +
#   ylim(-50,450)
# 
# miss_chart
# 
# # Made shots
# made_chart <- ggplot(data=train[targets=="made",], aes(x=loc_x, y=loc_y)) +
#   geom_point(size=2, alpha=0.5, color="blue") +
#   ylim(-50,450)
# 
# made_chart

# Chart shot type over time




# Remove and save any more unnecessary vars
train$shot_id <- NULL
test_ids <- test$shot_id
test$shot_id <- NULL
train$shot_made_flag <- NULL
train_dates <- train$game_date
test_dates <- test$game_date
train$game_date <-NULL
test$game_date <- NULL

# Fill NA with a baseline percentage
train[is.na(train)] <- 0.45
test[is.na(test)] <- 0.45

# The problem calls for time-sensitive training.
# you can't train on shots that haven't happened yet to predict older shots. 
# It is not clear why this would lead to problematic data leakage and 
# the problem doesn't lend itself too well to time series methods because the 
# last few shots you took don't have much if any more bearing on shooting than 
# the shot you took 100 or 500 shots ago. (Hot hand theory has not held up when analyzed.). 
# Online learning methods or creating many different models might give better performance. 
# I'll start by making a single model that includes all data as a baseline and then attempt 
# to make many models that incrementally add more data and predict shots from the next month or so.

#Create a validation set for initial model testing
set.seed(12)
train_index = createDataPartition(targets,p=0.75,list = FALSE,
                                  times = 1)

train_valid = train[train_index,]
valid = train[-train_index,]

t_valid_targets = targets[train_index]
valid_targets = targets[-train_index]


# Initial baseline

set.seed(121)
train_m_v = matrix(as.numeric(data.matrix(train_valid)),ncol=12)

valid_m = matrix(as.numeric(data.matrix(valid)),ncol=12)

num_ttargets_t_v = as.numeric(t_valid_targets)-1
num_ttargets_v = as.numeric(valid_targets)-1

set.seed(11)

xgb_preds <- rep(0,length(valid_targets))
iters <- 1

for (mod in 1:iters){

xgb_model = xgboost(data=train_m_v, 
                    label=num_ttargets_t_v, 
                    nrounds=500, 
                    verbose=1, 
                    eta=0.0125, 
                    max_depth=6, 
                    subsample=0.85, 
                    colsample_bytree=0.65,
                    objective="binary:logistic", 
                    eval_metric="logloss")

xgb_pred = predict(xgb_model, valid_m)

xgb_preds <- xgb_preds + xgb_pred
}

xgb_preds <- xgb_preds/iters

logLoss(as.numeric(valid_targets)-1, xgb_preds)


# Baseline validation logloss: 0.6018101. 
# Comparable to top leaderboard scores. 
# I suspect most top LB scorers are not making models that avoid leakage.


# Construct a loop that incrementally trains new models on more and more data and makes predictions. 
# Over 20 years, the number of models built could approach the 100-200 range depending on increment used.

# Set up model parameters
control <- trainControl(method = "none",
                        verboseIter = FALSE,
                        repeats = 1,                        
                        classProbs= TRUE,
                        summaryFunction= twoClassSummary)

grid <- expand.grid(nrounds=c(500), max_depth=c(6), eta=c(0.0125))


test_preds <- c()
test_pred_ids <- c()

first_test <- min(test_dates)

model_no <- 1

while ( length(test_preds) < 5000){
print("model number:")
print( model_no)
print (first_test)
set.seed(12)

if (first_test < "1997-11-08 CST"){   #First season, 20 day increments
    next_test <- first_test + 86400*10
} else if(first_test < "1998-11-08 CST") { #Second season 30 day 
    next_test <- first_test + 86400*20
} else if(first_test < "1999-11-08 CST") { #Third season 40 day 
    next_test <- first_test + 86400*30
} else { #After third season 50 day 
    next_test <- first_test + 86400*30
}

test_mask <- test_dates >= first_test & test_dates < next_test

if (length(which(test_mask == TRUE)) == 0){
  first_test <- next_test
  next
}

xgb_model_caret <- train(targets[train_dates < first_test]~., 
                         data=train[train_dates < first_test,], 
                         method="xgbTree", 
                         trControl=control, 
                         tuneGrid=grid, 
                         metric="ROC",
                         subsample=0.85, 
                         colsample_bytree=0.65)


test_split <- test[test_mask,]
round_preds <- predict(xgb_model_caret, newdata=test_split, type="prob")

test_preds <- c(test_preds, round_preds[,2])
test_pred_ids <- c(test_pred_ids, test_ids[test_mask])

first_test <- next_test

model_no <- model_no + 1

}


sub <- sample
sub$shot_id <- test_pred_ids
sub$shot_made_flag <- test_preds

write.csv(sub, "kobe_sub5.csv", row.names=FALSE)

# LB Score (sub4): 0.61279
# LB Score (sub5): 0.60887 nrounds=c(200), max_depth=c(7), eta=c(0.025)



