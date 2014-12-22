#Helping Santa's Helpers competition
#https://www.kaggle.com/c/helping-santas-helpers

#Some code (primarily time formatting functions) taken from/inspired by starter code posted by forum user skwalas
#https://www.kaggle.com/c/helping-santas-helpers/forums/t/11181/starter-naive-r-code-for-cookies-and-milk

#competition Rules

#Function to maximize:
#ojective_function <- minutes_to_complete * log(1 + number_of_elves)
#Min and max productivity scores c(0.25,4.0)
#Santioned hours are 9:00 to 19:00 (10 hours per day)


#Load training data
toys <- read.csv("toys_rev2.csv", stringsAsFactors=FALSE)

elves <- as.matrix(data.frame("id"=1:900,"productivity"=rep(1,900),"time_free"=rep(9*60,900)))

#Convert toy arrival times to ints (in seconds from starting time of Jan 1 2014)
reference_time <- as.POSIXct('2014 1 1 0 0', '%Y %m %d %H %M', tz = 'UTC')

convert_to_minute <- function(arrival) {
  arrive_time <- as.POSIXct(arrival, '%Y %m %d %H %M', tz = 'UTC')
  age <- as.integer(difftime(arrive_time, reference_time, units = 'mins', tz = 'UTC'))
  return(age)
}

toys[,'Arrival_time'] <- convert_to_minute(toys[,'Arrival_time'])

toys = as.matrix(toys)

#Convert time back to date
convert_to_chardate <- function(arrive_int) {
  char_date <- format(reference_time + arrive_int * 60, format = '%Y %m %d %H %M', tz = 'UTC')
  return(char_date)
}

#Update productivity function
productivity_update <- function(old_p, normal_hours, overtime_hours){
  new_p <- old_p *(1.02)^normal_hours*(0.9)^overtime_hours
  
  if (new_p > 4) {
    return (4)
    } else if (new_p < 0.25) {
    return(0.25)} else { 
    return (new_p)}
}

#Function to find normal time worked and overtime worked
normal_vs_overtime <- function(start_time, work_time){
  adj_time <- start_time %% 1440
  overtime <- 0
  normal_time <- 0
  while(work_time > 0) {
    if (adj_time < 540) { 
      early_time <- 540-adj_time
      overtime <- overtime + min(work_time,early_time)
      work_time <- work_time - early_time
    }
    if (adj_time < 1140 & work_time > 0) {
      norm_hours_left <- min(1140-adj_time,600)
      normal_time <- normal_time + min(work_time, norm_hours_left)
      work_time <- work_time - norm_hours_left  
    }
    if (work_time > 0) { 
      overtime <- overtime + min(work_time, 300)
      work_time <- work_time - 300    
    }
    adj_time <- 0
  }
  return ( c(normal_time,overtime) )
}

#Helper function to apply work penalty
apply_penalty <- function(next_normal, overtime_worked, past_9Oclock){
  day_left<- 600-past_9Oclock
  if (overtime_worked<day_left){ return(next_normal+overtime_worked)}
  next_normal <- next_normal+day_left+840
  return (apply_penalty(next_normal, overtime_worked-day_left, 0))
}

#Function applies overtime rest penalty and returns the soonest time and elf can work during normal hours
next_available <- function(end_time, overtime_worked){
   adj_end <- end_time %% 1440
   next_normal <- ifelse(adj_end<540,540,ifelse(adj_end>=1140,540+1440,adj_end))
   past_9Oclock <- (next_normal %% 1440)-540
   after_penalty <- apply_penalty(next_normal, overtime_worked, past_9Oclock)
   return (end_time+after_penalty-adj_end)
}

solution <- matrix(0, nrow=nrow(toys), ncol = 4, dimnames = list(NULL, c('ToyId', 'ElfId', 'StartTime', 'Duration')))

#Main for loop. Loops through all toys, assigns first available elf and updates elf status
system.time(
for (toy in 1:nrow(toys)){

  toy_arrival <- toys[toy,'Arrival_time']
  toy_duration <- toys[toy,'Duration']
  
  #Rules for selecting elves vvv

  elf <- which.min(elves[,'time_free'])

  #Rules for selecting elves ^^^
  
  time_available <- elves[elf, 'time_free']
  rating <- elves[elf, 'productivity']

  start_time <- ifelse(toy_arrival > time_available, toy_arrival, time_available)
  work_time <- as.integer(ceiling(toy_duration/rating))
  
  normal_and_overtime <- normal_vs_overtime(start_time, work_time)
  
  end_time <- start_time+work_time


  soonest_available <- next_available(end_time, normal_and_overtime[2])

  elves[elf, 'time_free'] <- soonest_available
  
  elves[elf, 'productivity'] <- productivity_update(rating, normal_and_overtime[1]/60,
                                                            normal_and_overtime[2]/60)  
  solution[toy,] <- c(toy, elves[elf,'id'], start_time, work_time)
  
  if(toy%%100000 == 0) {
    print(paste("Current toy:", toy))
  }
  if(toy==1000000){ break }
}
)

#Submission code
submission3 <- data.frame(ToyId = as.integer(solution[,1]), 
                         ElfId = as.integer(solution[,2]), 
                         StartTime = convert_to_chardate(solution[,3]), 
                         Duration = as.integer(solution[,4]), stringsAsFactors = FALSE)

write.csv(submission3, 'toys_submission3.csv', row.names = FALSE)



#Baseline start_times of simple naive model
baseline100k = 7913
baseline500k = 39443
baseline1mil = 78316


test100k <- solution[100000,3]
test500k <- solution[500000,3]
test1mil <- solution[1000000,3]