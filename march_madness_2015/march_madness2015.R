#March Machine Learning Mania 2015 Kaggle Competition
#https://www.kaggle.com/c/march-machine-learning-mania-2015


#PART 1 Code -- Predicting 2011-2014 vv
detailed <- read.csv("regular_season_detailed_results.csv")
detailed <- subset(detailed, season > 2010 & season < 2015)
detailed$wteam <- as.character(detailed$wteam)
detailed$lteam <- as.character(detailed$lteam)

teams <- read.csv("teams.csv")

tourn_detailed <- read.csv("tourney_detailed_results.csv")
tourn_detailed <- subset(tourn_detailed, season > 2010 & season < 2015)

tourney_seeds = read.csv("tourney_seeds.csv",stringsAsFactors =FALSE)
tourney_seeds <- subset(tourney_seeds, season > 2010 & season < 2015)

tourney_slots <- read.csv("tourney_slots.csv")
tourney_slots <- subset(tourney_slots, season > 2010 & season < 2015)

sample_sub <- read.csv("sample_submission.csv",stringsAsFactors =FALSE)
#PART 1 Code -- Predicting 2011-2014 ^^

#PART 2 Code -- Predicting 2015 vv
detailed <- read.csv("regular_season_detailed_results_2015.csv")
detailed$wteam <- as.character(detailed$wteam)
detailed$lteam <- as.character(detailed$lteam)

teams <-read.csv("teams.csv")

tourney_seeds <- read.csv("tourney_seeds_2015.csv",stringsAsFactors =FALSE)

tourney_slots <- read.csv("tourney_slots_2015.csv")

sample_sub <- read.csv("sample_submission_2015.csv",stringsAsFactors =FALSE)
#PART 2 Code -- Predicting 2015 ^^



#Custom game weighting function. Return 1 for unweighted games
get_game_weight <- function(game){
  mov <- ((game[,"wscore"])-game[,"lscore"])
  season_mod <- ifelse(game[,"daynum"]>118,1.25,1)
  loc_mod <- ifelse(game[,"wloc"]=="N", 1,1)
     if (mov<6){
         return(0.95*season_mod*loc_mod)
       }
     if (mov <11){
         return (1*season_mod*loc_mod)
       }
     return (1.05*season_mod*loc_mod)
}

#Create team rankings using the Colley Matrix method. 
#Ranks teams based on weighted wins/losses and adjusts for strength of schedule
get_colley_rankings <- function(year){
  colley_matrix <- diag(364)*2
  team_names <- as.character(teams$team_id)
  rownames(colley_matrix) <- team_names
  colnames(colley_matrix) <- team_names
  regular_season_games <- subset(detailed, season == year)
  win_diff = rep(1,364)
  names(win_diff) <- team_names
  for (g in 1:nrow(regular_season_games)){
    game <- regular_season_games[g,]
    winner <- game$wteam
    loser <- game$lteam
    game_weight <- get_game_weight(game)
    colley_matrix[winner,loser]<- colley_matrix[winner,loser]-game_weight
    colley_matrix[loser,winner]<- colley_matrix[loser,winner]-game_weight
    colley_matrix[winner,winner]<- colley_matrix[winner,winner]+(1*game_weight)
    colley_matrix[loser,loser]<- colley_matrix[loser,loser]+(1*game_weight)
    win_diff[winner] <- win_diff[winner]+((1/2)*game_weight)
    win_diff[loser] <- win_diff[loser]-((1/2)*game_weight)
  }
  rankings <- solve(colley_matrix) %*% win_diff
  rownames(rankings) <- paste(year, team_names, sep="_" )
  rankings
}

#Part 1 rankings
# rankings = rbind(get_colley_rankings(2011),get_colley_rankings(2012),get_colley_rankings(2013),get_colley_rankings(2014))

#Part 2 2015 rankings
rankings2015 <- get_colley_rankings(2015)
  
  
#Takes two team ratings and outputs the probability of team 1 beating team 2.
#Uses a sigmoid function on the difference in team rankings to output a probability. 
#Outputs a fixed win probability of 0.55 for the winning team if rankings are close and 
#0.95 at the most to minimize logloss.

get_prob <- function(team1, team2){
  diff <- team1-team2
  if((abs(diff)<0.125) & (diff>0)){
    prob <- 0.55
  }
  if ((abs(diff)<0.125) & (diff<0)){
    prob <- 0.45
  } else{
    prob <- 1/(1+exp(-6*(diff)))
  }
  prob <- (ifelse(prob>0.50, min(prob,0.95), max(prob,0.05)) )
  return (prob)
}


#Function takes a vector of tourney games in YEAR_TEAM1_TEAM2 format and outputs predictions 
#for each game based on team rankings for each year computed earlier
get_predictions <- function(games, rankings){
  matchups <- c()
  
  for (game in games){
    year_teams <- strsplit(game,"_")
    team1 <- paste(year_teams[[1]][1],year_teams[[1]][2],sep="_")
    team2 <- paste(year_teams[[1]][1],year_teams[[1]][3],sep="_")
    win_prob <- get_prob(rankings[team1,],rankings[team2,])
    matchups <- c(matchups,win_prob)
  }
  
  names(matchups) <- games
  return (matchups)
}


#Evaluation/Submission code for PART 1 vv----------------
preds_2011_2014 <- get_predictions(sample_sub$id, rankings)

tourney_games <- tourn_detailed[c("season","wteam","lteam")]

#Log loss function for evaluation of preditions
log_loss <- function(preds,tourney_games){
  total_loss <- 0
  for (g in 1:nrow(tourney_games)){
    game <- tourney_games[g,]
    team1 <- min(game[2],game[3])
    team2 <- max(game[2],game[3])
    team1_win_prob <- preds[paste(game[1],team1,team2,sep="_")]
    if (team1 == game[2]){
      total_loss <- total_loss + log(team1_win_prob)
    } else{
      total_loss <- total_loss + log(1-team1_win_prob)
    }
  }
  return (total_loss*-1*(1/nrow(tourney_games)))
}

log_loss_sub1 <- log_loss(preds_2011_2014,tourney_games)
print(log_loss_sub1)
summary(rankings)

submission5 <-data.frame(id = names(preds_2011_2014), pred = preds_2011_2014)

write.csv(submission5, "MM2015_sub5.csv", row.names=FALSE)
#Evaluation/Submission code for PART 1 ^^-----------------


#Evaluation/Submission code for PART 2 vv-----------------
preds_2015 <- get_predictions(sample_sub$id, rankings2015)

summary(rankings2015)

submission_2015 <- data.frame(id = names(preds_2015), pred = preds_2015)

write.csv(submission_2015, "submission_2015.csv", row.names=FALSE)


#For second solution, use same solution as #1 but make a few huge gambles
# to attempt to pick up points. (Kentucky wins the whole thing with 99% probability. etc.

submission_2015_2 <- submission_2015

for (matchup in rownames(submission_2015_2)){

   year_teams <- strsplit(matchup,"_")

   if (year_teams[[1]][2] == "1458"){  #Wisconsin is 4th
     submission_2015_2[matchup,"pred"] <- 0.95
   }
   if (year_teams[[1]][3] == "1458"){  #Wisconsin is 4th
     submission_2015_2[matchup,"pred"] <- 0.05
   }
   
   if (year_teams[[1]][2] == "1181"){  #Duke is third
     submission_2015_2[matchup,"pred"] <- 0.95
   }
   if (year_teams[[1]][3] == "1181"){  #Duke is third
     submission_2015_2[matchup,"pred"] <- 0.05
   }
   
   if (year_teams[[1]][2] == "1437"){  #Villanova is second
     submission_2015_2[matchup,"pred"] <- 0.95
   }
   if (year_teams[[1]][3] == "1437"){  #Villanova is second
     submission_2015_2[matchup,"pred"] <- 0.05
   }

   if (year_teams[[1]][2] == "1246"){  #Kentucky wins all games
     submission_2015_2[matchup,"pred"] <- 0.999
   }
   if (year_teams[[1]][3] == "1246"){  #Kentucky wins all games
     submission_2015_2[matchup,"pred"] <- 0.001
   }
   
}

write.csv(submission_2015_2, "submission_2015_2.csv", row.names=FALSE)
#Evaluation/Submission code for PART 2 ^^-----------------
