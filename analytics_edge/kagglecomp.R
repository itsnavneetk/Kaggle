
#Kaggle Competition
#First remove NA YOB values and oddball YOB values (1900, 2011 and 2039)
Ktrain = read.csv("Kaggletrain.csv")
Ktest = read.csv("Kaggletest.csv")

library(mice)
set.seed(144)
Ktest2 = complete(mice(Ktest))

Ktrain2 = subset(Ktrain, YOB!=1900 &YOB!=2011&YOB!=2039)
#Add age variable
Ktrain2$age = 2014-Ktrain2$YOB
Ktest2$age = 2014-Ktest2$YOB

Ktrain2$ageunder10 = as.numeric(Ktrain2$age < 10)
Ktrain2$age10to15 = as.numeric(Ktrain2$age > 10 & Ktrain2$age < 16)
Ktrain2$age16to19 = as.numeric(Ktrain2$age > 15 & Ktrain2$age < 20)
Ktrain2$age20to25 = as.numeric(Ktrain2$age > 19 & Ktrain2$age < 26)
Ktrain2$age26to30 = as.numeric(Ktrain2$age > 25 & Ktrain2$age < 31)
Ktrain2$age31to40 = as.numeric(Ktrain2$age > 30 & Ktrain2$age < 41)
Ktrain2$age41to50 = as.numeric(Ktrain2$age > 40 & Ktrain2$age < 51)
Ktrain2$age51to60 = as.numeric(Ktrain2$age > 50 & Ktrain2$age < 61)
Ktrain2$age61to70 = as.numeric(Ktrain2$age > 60 & Ktrain2$age < 71)
Ktrain2$age71to80 = as.numeric(Ktrain2$age > 70 & Ktrain2$age < 81)
Ktrain2$age81to90 = as.numeric(Ktrain2$age > 80 & Ktrain2$age < 91)
Ktrain2$age91plus = as.numeric(Ktrain2$age > 90)


Ktest2$ageunder10 = as.numeric(Ktest2$age < 10)
Ktest2$age10to15 = as.numeric(Ktest2$age > 10 & Ktest2$age < 16)
Ktest2$age16to19 = as.numeric(Ktest2$age > 15 & Ktest2$age < 20)
Ktest2$age20to25 = as.numeric(Ktest2$age > 19 & Ktest2$age < 26)
Ktest2$age26to30 = as.numeric(Ktest2$age > 25 & Ktest2$age < 31)
Ktest2$age31to40 = as.numeric(Ktest2$age > 30 & Ktest2$age < 41)
Ktest2$age41to50 = as.numeric(Ktest2$age > 40 & Ktest2$age < 51)
Ktest2$age51to60 = as.numeric(Ktest2$age > 50 & Ktest2$age < 61)
Ktest2$age61to70 = as.numeric(Ktest2$age > 60 & Ktest2$age < 71)
Ktest2$age71to80 = as.numeric(Ktest2$age > 70 & Ktest2$age < 81)
Ktest2$age81to90 = as.numeric(Ktest2$age > 80 & Ktest2$age < 91)
Ktest2$age91plus = as.numeric(Ktest2$age > 90)

KtrainLogModel2 = glm(Happy~YOB+Income+HouseholdStatus+Q98869+Q99716+Q101162+Q102289+Q102089+Q102289+Q102674+Q102687+Q106389+Q106388+Q107869+Q115899+Q115611+Q115777+Q118237+Q119334+Q120014+Q120194+Q122769,data=Ktrain2, family="binomial")

Ktrain2pred = predict(KtrainLogModel2, type="response", data=Ktrain2)

table(Ktrain2$Happy, Ktrain2pred >= 0.5)

FALSE TRUE
0  1004  711
1   487 1729

> (1004+1729)/nrow(Ktrain2)
[1] 0.6952429

Ktest2pred = predict(KtrainLogModel2, type="response", newdata=Ktest2)

submission = data.frame(UserID = Ktest2$UserID, Probability1 = Ktest2pred)

write.csv(submission, "submission.csv", row.names=FALSE) #submission1


#KtrainLogModel2 = glm(Happy~ .-UserID-YOB-Q96024-Q101163-Q112270-Q120978,data=Ktrain2, family="binomial")
FALSE TRUE
0  1087  628
1   466 1750

KtrainLogModel2 = glm(Happy~ .-age-Party-EducationLevel-Income-UserID-YOB-Q96024-Q101163-Q112270-Q120978-Q105840-Q109244-Q98078-Q124122-Q124122-Q123621-Q120650-Q118892-Q108617-Q108342-Q110740-Q100010-Q106993-Q108754-Q115390-Gender,data=Ktrain2, family="binomial")



# Attempt #2
ktrain = read.csv("Kaggletrain.csv")
ktest = read.csv("Kaggletest.csv")

#Remove bogus ages and turn Year of Birth into a vairable called age
#Divide age by 100 to get a value bewteen 1 and 0 for use in creating distances
ktrain = subset(ktrain, YOB > 1930 & YOB < 2002)
ktrain$YOB = ((2014-ktrain$YOB)+100)/100
ktest$YOB = ((2014-ktest$YOB)+100)/100

library(mice)
set.seed(144)

#Create versions of both data sets with imputed values for year of birth
ktestimputed = complete(mice(ktest))


library(caTools)
set.seed(88)

#Split train data into two groups for testing purposes.
split = sample.split(ktrainimputed$Happy, SplitRatio = 0.75)

ktrainimputed.trainset = subset(ktrainimputed, split == TRUE)
ktrainimputed.testset = subset(ktrainimputed, split == FALSE)

#Calculate distances between training training set, training test set, and real test set
#for clustering.
distances.ktrainimputed.trainset = dist(ktrainimputed.trainset[3:110], method = "euclidean")
distances.ktrainimputed.testset = dist(ktrainimputed.testset[3:110], method = "euclidean")
distances.ktestimputed = dist(ktestimputed[3:109], method = "euclidean")

#Create clusters for each of the 3 groups.
cluster.ktrainimputed.trainset = hclust(distances.ktrainimputed.trainset, method = "ward") 
cluster.ktrainimputed.testset = hclust(distances.ktrainimputed.testset, method = "ward") 
cluster.ktestimputed = hclust(distances.ktestimputed, method = "ward") 

#Plot the cluster dendrograms
#plot(cluster.ktrainimputed.trainset)
#plot(cluster.ktrainimputed.testset)
#plot(cluster.ktestimputed)

#Dendrogram shows clustering of 2 appears most viable. 3 and 4 are possibilities
#for future tests.

#Assign values to 2 cluster groups
clustergroups.ktrainimputed.trainset = cutree(cluster.ktrainimputed.trainset, k = 2)
clustergroups.ktrainimputed.testset = cutree(cluster.ktrainimputed.testset, k = 2)
clustergroups.ktestimputed  = cutree(cluster.ktestimputed, k = 2)

#Split each data set into groups 1 and 2 based on the assignment
clustergroups.ktrainimputed.trainset.G1 = subset(ktrainimputed.trainset, clustergroups.ktrainimputed.trainset==1 )
clustergroups.ktrainimputed.trainset.G2 = subset(ktrainimputed.trainset, clustergroups.ktrainimputed.trainset==2 )

clustergroups.ktrainimputed.testset.G1 = subset(ktrainimputed.testset, clustergroups.ktrainimputed.testset==1 )
clustergroups.ktrainimputed.testset.G2 = subset(ktrainimputed.testset, clustergroups.ktrainimputed.testset==2 )

clustergroups.ktestimputed.G1 = subset(ktestimputed, clustergroups.ktestimputed==1 )
clustergroups.ktestimputed.G2 = subset(ktestimputed, clustergroups.ktestimputed==2 )


#Create and test models based on the cluster groupings-------------------

#Logistisc Regression with backward selection for each group

log.model.ktrain.train.G1 = glm(formula = Happy ~ HouseholdStatus  + 
                                  Q122769 + Q120978 + 
                                    + Q118237 +  
                                  Q116797 + Q116441 + Q116197 + 
                                  Q113584  + Q108343 + Q107869 + Q107491 + 
                                    Q101162 + 
                                  Q96024, data = clustergroups.ktrainimputed.trainset.G1)

log.model.ktrain.train.G2 = glm(Happy~.-UserID, data = clustergroups.ktrainimputed.trainset.G2)

summary(log.model.ktrain.train.G1)
summary(log.model.ktrain.train.G2)

prediction.against.ktrain.testset.G1 = predict(log.model.ktrain.train.G1, type="response", newdata=clustergroups.ktrainimputed.testset.G1)
prediction.against.ktrain.testset.G2 = predict(log.model.ktrain.train.G2, type="response", newdata=clustergroups.ktrainimputed.testset.G2)

table(clustergroups.ktrainimputed.testset.G1$Happy, prediction.against.ktrain.testset.G1 >= 0.5)

table(clustergroups.ktrainimputed.testset.G2$Happy, prediction.against.ktrain.testset.G2 >= 0.5)




# Attempt #3
ktrain = read.csv("Kaggletrain.csv")
ktest = read.csv("Kaggletest.csv")

#Remove bogus ages and turn Year of Birth into a vairable based on age.
#Divide age by 100 to get a value bewteen 1 and 2.
ktrain = subset(ktrain, YOB > 1930 & YOB < 2002)
ktrain$YOB = ((2014-ktrain$YOB)+100)/100
ktest$YOB = ((2014-ktest$YOB)+100)/100



library(mice)
set.seed(122)
ktest = complete(mice(ktest))
ktrain = complete(mice(ktrain))

logmodel = glm(Happy~.-UserID, data=ktrain)

summary(logmodel)

pred = predict(logmodel , type="response", data=ktrain)

table(ktrain$Happy, pred >= 0.5)


StevensForest = randomForest(Reverse ~ Circuit + Issue + Petitioner + Respondent + LowerCourt + Unconst, data = Train, ntree=200, nodesize=25 )

library(rpart)
library(rpart.plot)
library(randomForest)

kaggleforest= randomForest(Happy~.-UserID, data = ktrain, ntree=25, nodesize=100 )

PredictForest = predict(kaggleforest, newdata = ktrain)
tab = table(ktrain$Happy, PredictForest > 0.5)
(tab[[1]]+tab[[2,2]])/nrow(ktrain)
plot(kaggleforest)



Ktest2pred = predict(kaggleforest, newdata=ktest)

submission = data.frame(UserID = ktest$UserID, Probability1 = Ktest2pred)

write.csv(submission, "submission.csv", row.names=FALSE) #submission1

subset(ktest, YOB>1.74)$UserID


ggplot(data=ktrain, aes(x=YOB ,y=Happy ))+
  geom_boxplot()




