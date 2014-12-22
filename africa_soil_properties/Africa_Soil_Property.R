#Africa Soil Property Kaggle competition
library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)
library(e1071)
library(gbm)
library(kernlab)
library(kknn)
options( java.parameters = "-Xmx3g" )
library(extraTrees)
library(deepnet)
library(prospectr)


#Original data sets
train = read.csv("training.csv")
test = read.csv("sorted_test.csv")


train = read.csv("africa_train2.csv")
test = read.csv("africa_test2.csv")



#Start attempt 4 at paring data (automated paring with prospectr-----
Xtrain <- train[,2:3595]
Ytrain <- train[,3596:3600]
Xtest <- test[,2:3595]
IDtest <- test[,1]

#Extract the spectral testing and training data
all_spectral_data = rbind(Xtrain[ ,2:3578],Xtest[ ,2:3578])

plot(as.numeric(sapply(colnames(all_spectral_data), substring, first=2)), all_spectral_data[1, ], type = "l", xlab = "Wavelength",  ylab = "Absorbance")

#Create a moving average to reduce signal noise
mov_average = movav(all_spectral_data, w = 1) 

lines(as.numeric(sapply(colnames(mov_average), substring, first=2)), mov_average[1, ], col = "red")

#Compress signal via binning
signal_bins = binning(mov_average,bin.size=3)

#combine highly correlated (>0.975) bins.
highly_correlated_bins <- findCorrelation(cor(signal_bins), .975)

#Remove unnecessary bins
trimmed_bins = data.frame(signal_bins)[ ,-highly_correlated_bins]

#Create new training and test sets with comprssed spectral data
train_trim = trimmed_bins[1:nrow(train), ]
test_trim = trimmed_bins[(nrow(train)+1):nrow(trimmed_bins), ]
train_trim$PIDN = train$PIDN
test_trim$PIDN = test$PIDN
train_trim = cbind(train_trim,train[ ,3578:3600])
test_trim = cbind(test_trim,test[ ,3578:3595])

write.csv(train_trim, "africa_train5.csv", row.names=FALSE)

write.csv(test_trim, "africa_test5.csv", row.names=FALSE)

train = read.csv("africa_train5.csv")
test = read.csv("africa_test5.csv")

#End attempt 4 at paring data (automated paring with prospectr-----

#Start attempt 3 at paring data (automated paring with prospectr-----
Xtrain <- train[,2:3595]
Ytrain <- train[,3596:3600]
Xtest <- test[,2:3595]
IDtest <- test[,1]

#Extract the spectral testing and training data
all_spectral_data = rbind(Xtrain[ ,2:3578],Xtest[ ,2:3578])

#combine highly correlated (>0.998) bins.
highly_correlated_bins <- findCorrelation(cor(all_spectral_data), 0.9975)

#Remove unnecessary bins
trimmed_bins = all_spectral_data[ ,-highly_correlated_bins]

#Create new training and test sets with comprssed spectral data
train_trim = trimmed_bins[1:nrow(train), ]
test_trim = trimmed_bins[(nrow(train)+1):nrow(trimmed_bins), ]
train_trim$PIDN = train$PIDN
test_trim$PIDN = test$PIDN
train_trim = cbind(train_trim,train[ ,3578:3600])
test_trim = cbind(test_trim,test[ ,3578:3595])

write.csv(train_trim, "africa_train4.csv", row.names=FALSE)

write.csv(test_trim, "africa_test4.csv", row.names=FALSE)

train = read.csv("africa_train4.csv")
test = read.csv("africa_test4.csv")
#End attempt 3 at paring data (automated paring with prospectr-----



#Start attempt 2 at paring data (automated paring with prospectr-----
Xtrain <- train[,2:3595]
Ytrain <- train[,3596:3600]
Xtest <- test[,2:3595]
IDtest <- test[,1]

#Extract the spectral testing and training data
all_spectral_data = rbind(Xtrain[ ,2:3578],Xtest[ ,2:3578])

plot(as.numeric(sapply(colnames(all_spectral_data), substring, first=2)), all_spectral_data[1, ], type = "l", xlab = "Wavelength",  ylab = "Absorbance")

#Create a moving average to reduce signal noise
mov_average = movav(all_spectral_data, w = 10) 

lines(as.numeric(sapply(colnames(mov_average), substring, first=2)), mov_average[1, ], col = "red")

#Compress signal via binning
signal_bins = binning(mov_average,bin.size=5)

#combine highly correlated (>0.975) bins.
highly_correlated_bins <- findCorrelation(cor(signal_bins), .975)

#Remove uneceessary bins
trimmed_bins = data.frame(signal_bins)[ ,-highly_correlated_bins]

#Create new training and test sets with comprssed spectral data
train_trim = trimmed_bins[1:nrow(train), ]
test_trim = trimmed_bins[(nrow(train)+1):nrow(trimmed_bins), ]
train_trim$PIDN = train$PIDN
test_trim$PIDN = test$PIDN
train_trim = cbind(train_trim,train[ ,3578:3600])
test_trim = cbind(test_trim,test[ ,3578:3595])

write.csv(train_trim, "africa_train3.csv", row.names=FALSE)

write.csv(test_trim, "africa_test3.csv", row.names=FALSE)

train = read.csv("africa_train3.csv")
test = read.csv("africa_test3.csv")

#End attempt 2 at paring data (automated paring with prospectr-----




#Start attempt 1 at paring down data(manual paring-----------------------)
#Pared down data sets----
train = read.csv("africa_train2.csv")
test = read.csv("africa_test2.csv")
train$Depth = as.numeric(train$Depth)
test$Depth = as.numeric(test$Depth)
#-----------------------

summary(train[ ,c(3580:3600)])
summary(test[ ,c(3580:3595)])


#splom(train[ ,c(3580:3591)])

#cor(train[ ,seq(2,3679,250)])

#Far too many spectral variables to use in a model. Need to reduce the number of parameters by
#Combining/eliminating spectral data variables. Perhaps combine groups with high correlation.

#First 2:1100 variables have min correlation of 0.9757352. Combine into a single variable or #use only 1 of them.
cor2_1100 = cor(train[ ,c(2:1100)])
min(cor2_1100)

#Variables 1101:1450, min correlation: 0.949764. Combine
cor1101_1450 = cor(train[ ,c(1101:1450)])
min(cor1101_1450)

#Variables 1451:1750, min correlation: 0.9582676. Combine
cor1451_1750 = cor(train[ ,c(1451:1750)])
min(cor1451_1750)

#Variables 1751:1900, min correlation: 0.981061. Combine
cor1751_1900 = cor(train[ ,c(1751:1900)])
min(cor1751_1900)

#Variables 1901:1945, min correlation: 0.9841781. Combine
cor1901_1945 = cor(train[ ,c(1901:1945)])
min(cor1901_1945)

#Variables 1946:1955, min correlation: 0.9903429. Combine
cor1946_1955 = cor(train[ ,c(1946:1955)])
min(cor1946_1955)

#Variables 1956:1962, min correlation: 0.9817752. Combine
cor1956_1962 = cor(train[ ,c(1956:1962)])
min(cor1956_1962)

#Variables 1963:1975, min correlation: 0.9517245. Combine
cor1963_1975 = cor(train[ ,c(1963:1975)])
min(cor1963_1975)

#Variables 1976:1985, min correlation: 0.9614633. Combine
cor1976_1985 = cor(train[ ,c(1976:1985)])
min(cor1976_1985)

#Variables 1986:2000, min correlation: 0.9641776. Combine
cor1986_2000 = cor(train[ ,c(1986:2000)])
min(cor1986_2000)

#Variables 2001:2015, min correlation: 0.9562656. Combine
cor2001_2015 = cor(test[ ,c(2001:2015)])
min(cor2001_2015)

#0.9370824
cor2016_2050 = cor(test[ ,c(2016:2050)])
min(cor2016_2050)

#0.9550563
cor2051_2150 = cor(test[ ,c(2051:2150)])
min(cor2051_2150)

#0.9511684
cor2151_2275 = cor(test[ ,c(2151:2275)])
min(cor2151_2275)

#0.9399128
cor2276_2376 = cor(test[ ,c(2276:2376)])
min(cor2276_2376)

#0.9669492
cor2377_2576 = cor(test[ ,c(2377:2576)])
min(cor2377_2576)

#0.9582912
cor2577_2676 = cor(test[ ,c(2577:2676)])
min(cor2577_2676)

#0.9413439
cor2677_2825 = cor(test[ ,c(2677:2825)])
min(cor2677_2825)

#0.9391739
cor2826_2832 = cor(test[ ,c(2826:2832)])
min(cor2826_2832)

#0.9305608
cor2833_2843 = cor(test[ ,c(2833:2843)])
min(cor2833_2843)

#0.9403897
cor2834_2900 = cor(test[ ,c(2844:2900)])
min(cor2834_2900)

#0.9464079
cor2901_2930 = cor(test[ ,c(2901:2930)])
min(cor2901_2930)

#0.9338439
cor2931_2955 = cor(test[ ,c(2931:2955)])
min(cor2931_2955)

#0.9240767
cor2956_2972 = cor(test[ ,c(2956:2972)])
min(cor2956_2972)

#0.9408056print(item)
cor2972_2977 = cor(test[ ,c(2972:2977)])
min(cor2972_2977)

#0.9353462
cor2978_3000 = cor(test[ ,c(2978:3000)])
min(cor2978_3000)

#0.9367055
cor3001_3020 = cor(test[ ,c(3001:3020)])
min(cor3001_3020)

#0.9488123
cor3021_3047 = cor(test[ ,c(3021:3047)])
min(cor3021_3047)

#0.9156737
cor3048_3056 = cor(test[ ,c(3048:3056)])
min(cor3048_3056)

#0.9212022
cor3057_3078 = cor(test[ ,c(3057:3078)])
min(cor3057_3078)

#0.9227557
cor3079_3093 = cor(test[ ,c(3079:3093)])
min(cor3079_3093)

#0.9226525
cor3094_3113 = cor(test[ ,c(3094:3113)])
min(cor3094_3113)


#0.910226
cor3114_3153 = cor(test[ ,c(3114:3153)])
min(cor3114_3153)

#0.9218084
cor3154_3170 = cor(test[ ,c(3154:3170)])
min(cor3154_3170)

#0.9238652
cor3171_3205 = cor(test[ ,c(3171:3205)])
min(cor3171_3205)

#0.9226847
cor3206_3217 = cor(test[ ,c(3206:3217)])
min(cor3206_3217)

#0.904639
cor3218_3227 = cor(test[ ,c(3218:3227)])
min(cor3218_3227)

#0.9112608
cor3228_3235 = cor(test[ ,c(3228:3235)])
min(cor3228_3235)

#0.9015272
cor3236_3250 = cor(test[ ,c(3236:3250)])
min(cor3236_3250)

#0.922615
cor3251_3305 = cor(test[ ,c(3251:3305)])
min(cor3251_3305)

#0.9457398
cor3306_3337 = cor(test[ ,c(3306:3337)])
min(cor3306_3337)

#0.9078334
cor3338_3358 = cor(test[ ,c(3338:3358)])
min(cor3338_3358)

#0.9317399
cor3359_3388 = cor(test[ ,c(3359:3388)])
min(cor3359_3388)

#0.9249653
cor3389_3458 = cor(test[ ,c(3389:3458)])
min(cor3389_3458)

#0.9168629
cor3459_3465 = cor(test[ ,c(3459:3465)])
min(cor3459_3465)

#0.9344318
cor3466_3470 = cor(test[ ,c(3466:3470)])
min(cor3466_3470)

#0.9084517
cor3471_3480 = cor(test[ ,c(3471:3480)])
min(cor3471_3480)

#0.9037902
cor3481_3559 = cor(test[ ,c(3481:3559)])
min(cor3481_3559)

#0.9243934
cor3560_3574 = cor(test[ ,c(3560:3574)])
min(cor3560_3574)

#0.9844115
cor3575_3579 = cor(test[ ,c(3575:3579)])
min(cor3575_3579)

#train$m2_1100 = rowSums(train[ ,2:1100])/(1100-2)

corlist = c('1101_1450', '1451_1750', '1751_1900', '1901_1945', '1946_1955', '1956_1962', '1963_1975', '1976_1985', '1986_2000', '2001_2015', '2016_2050', '2051_2150', '2151_2275', '2276_2376', '2377_2576', '2577_2676', '2677_2825', '2826_2832', '2833_2843', '2834_2900', '2901_2930', '2931_2955', '2956_2972', '2973_2977', '2978_3000', '2_1100', '3001_3020', '3021_3047', '3048_3056', '3057_3078', '3079_3093', '3094_3113', '3114_3153', '3154_3170', '3171_3205', '3206_3217', '3218_3227', '3228_3235', '3236_3250', '3251_3305', '3306_3337', '3338_3358', '3359_3388', '3389_3458', '3459_3465', '3466_3470', '3471_3480', '3481_3559', '3560_3574', '3575_3579')

#Create 50 new variables that are averages of spectral ranges with high correlation
for (item in corlist){
  name = paste("m",item, sep="")
  nums = strsplit(item,"_")
  n1 = as.numeric(nums[[1]][1])
  n2 = as.numeric(nums[[1]][2])
  train[[name]] = rowSums(train[ ,n1:n2])/(n2-n1)
}

#Create 50 new variables that are averages of spectral ranges with high correlation
for (item in corlist){
  name = paste("m",item, sep="")
  nums = strsplit(item,"_")
  n1 = as.numeric(nums[[1]][1])
  n2 = as.numeric(nums[[1]][2])
  test[[name]] = rowSums(test[ ,n1:n2])/(n2-n1)
}

#Save pared down data sets for easier loading and use------------------
train2 = train[ ,c(1,3580:3650)]
test2 = test[ ,c(1,3580:3645)]

write.csv(train2, "africa_train2.csv", row.names=FALSE)

write.csv(test2, "africa_test2.csv", row.names=FALSE)

train = read.csv("africa_train2.csv")
test = read.csv("africa_test2.csv")
#End attempt 1 at paring down data(manual paring-----------------------)






#Data ready for use in models start with basic regression on each variable to get a baseline prediction and make sure submission works

#Linear model for sand
regmod_Sand = glm(Sand~m1101_1450+m1451_1750+m1751_1900+m1901_1945+m1946_1955+m1956_1962+m1963_1975+m1976_1985+m1986_2000+m2001_2015+m2016_2050+m2051_2150+m2151_2275+m2276_2376+m2377_2576+m2577_2676+m2677_2825+m2826_2832+m2833_2843+m2834_2900+m2901_2930+m2931_2955+m2956_2972+m2973_2977+m2978_3000+m2_1100+m3001_3020+m3021_3047+m3048_3056+m3057_3078+m3079_3093+m3094_3113+m3114_3153+m3154_3170+m3171_3205+m3206_3217+m3218_3227+m3228_3235+m3236_3250+m3251_3305+m3306_3337+m3338_3358+m3359_3388+m3389_3458+m3459_3465+m3466_3470+m3471_3480+m3481_3559+m3560_3574+m3575_3579+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth, data=train)

pred_Sand = predict(regmod_Sand, data=train)

#Training RMSE
sum(sqrt((train$Sand-pred_Sand)^2))/nrow(train)

#Linear model for SOC
regmod_SOC = glm(SOC~m1101_1450+m1451_1750+m1751_1900+m1901_1945+m1946_1955+m1956_1962+m1963_1975+m1976_1985+m1986_2000+m2001_2015+m2016_2050+m2051_2150+m2151_2275+m2276_2376+m2377_2576+m2577_2676+m2677_2825+m2826_2832+m2833_2843+m2834_2900+m2901_2930+m2931_2955+m2956_2972+m2973_2977+m2978_3000+m2_1100+m3001_3020+m3021_3047+m3048_3056+m3057_3078+m3079_3093+m3094_3113+m3114_3153+m3154_3170+m3171_3205+m3206_3217+m3218_3227+m3228_3235+m3236_3250+m3251_3305+m3306_3337+m3338_3358+m3359_3388+m3389_3458+m3459_3465+m3466_3470+m3471_3480+m3481_3559+m3560_3574+m3575_3579+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth, data=train)

pred_SOC = predict(regmod_SOC, data=train)

#Training RMSE
sum(sqrt((train$SOC-pred_SOC)^2))/nrow(train)

#Linear model for pH
regmod_pH = glm(pH~m1101_1450+m1451_1750+m1751_1900+m1901_1945+m1946_1955+m1956_1962+m1963_1975+m1976_1985+m1986_2000+m2001_2015+m2016_2050+m2051_2150+m2151_2275+m2276_2376+m2377_2576+m2577_2676+m2677_2825+m2826_2832+m2833_2843+m2834_2900+m2901_2930+m2931_2955+m2956_2972+m2973_2977+m2978_3000+m2_1100+m3001_3020+m3021_3047+m3048_3056+m3057_3078+m3079_3093+m3094_3113+m3114_3153+m3154_3170+m3171_3205+m3206_3217+m3218_3227+m3228_3235+m3236_3250+m3251_3305+m3306_3337+m3338_3358+m3359_3388+m3389_3458+m3459_3465+m3466_3470+m3471_3480+m3481_3559+m3560_3574+m3575_3579+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth, data=train)

pred_pH = predict(regmod_pH, data=train)

#Training RMSE
sum(sqrt((train$pH-pred_pH)^2))/nrow(train)

#Linear model for Ca
regmod_Ca = glm(Ca~m1101_1450+m1451_1750+m1751_1900+m1901_1945+m1946_1955+m1956_1962+m1963_1975+m1976_1985+m1986_2000+m2001_2015+m2016_2050+m2051_2150+m2151_2275+m2276_2376+m2377_2576+m2577_2676+m2677_2825+m2826_2832+m2833_2843+m2834_2900+m2901_2930+m2931_2955+m2956_2972+m2973_2977+m2978_3000+m2_1100+m3001_3020+m3021_3047+m3048_3056+m3057_3078+m3079_3093+m3094_3113+m3114_3153+m3154_3170+m3171_3205+m3206_3217+m3218_3227+m3228_3235+m3236_3250+m3251_3305+m3306_3337+m3338_3358+m3359_3388+m3389_3458+m3459_3465+m3466_3470+m3471_3480+m3481_3559+m3560_3574+m3575_3579+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth, data=train)

pred_Ca = predict(regmod_Ca, data=train)

#Training RMSE
sum(sqrt((train$Ca-pred_Ca)^2))/nrow(train)

#Linear model for P
regmod_P = glm(P~m1101_1450+m1451_1750+m1751_1900+m1901_1945+m1946_1955+m1956_1962+m1963_1975+m1976_1985+m1986_2000+m2001_2015+m2016_2050+m2051_2150+m2151_2275+m2276_2376+m2377_2576+m2577_2676+m2677_2825+m2826_2832+m2833_2843+m2834_2900+m2901_2930+m2931_2955+m2956_2972+m2973_2977+m2978_3000+m2_1100+m3001_3020+m3021_3047+m3048_3056+m3057_3078+m3079_3093+m3094_3113+m3114_3153+m3154_3170+m3171_3205+m3206_3217+m3218_3227+m3228_3235+m3236_3250+m3251_3305+m3306_3337+m3338_3358+m3359_3388+m3389_3458+m3459_3465+m3466_3470+m3471_3480+m3481_3559+m3560_3574+m3575_3579+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth, data=train)

pred_P = predict(regmod_P, data=train)

#Training RMSE
sum(sqrt((train$P-pred_P)^2))/nrow(train)

tpred_Sand = predict(regmod_Sand, newdata=test)
tpred_pH = predict(regmod_pH, newdata=test)
tpred_Ca = predict(regmod_Ca, newdata=test)
tpred_SOC = predict(regmod_SOC, newdata=test)
tpred_P = predict(regmod_P, newdata=test)


submission9 = data.frame(PIDN= test$PIDN, Ca=tpred_Ca	 ,P=tpred_P	,pH=tpred_pH	,SOC=tpred_SOC	,Sand=tpred_Sand )

write.csv(submission9, "africa_submission9.csv", row.names=FALSE)
#Score=0.55931 (average RMSE across the 5 variables)
#Rerun model of basic model with data data trimmed using trim method #2
#Score =


#Try KNN model, k=1
#Linear model for sand
kknn_Sand = kknn(Sand~m1101_1450+m1451_1750+m1751_1900+m1901_1945+m1946_1955+m1956_1962+m1963_1975+m1976_1985+m1986_2000+m2001_2015+m2016_2050+m2051_2150+m2151_2275+m2276_2376+m2377_2576+m2577_2676+m2677_2825+m2826_2832+m2833_2843+m2834_2900+m2901_2930+m2931_2955+m2956_2972+m2973_2977+m2978_3000+m2_1100+m3001_3020+m3021_3047+m3048_3056+m3057_3078+m3079_3093+m3094_3113+m3114_3153+m3154_3170+m3171_3205+m3206_3217+m3218_3227+m3228_3235+m3236_3250+m3251_3305+m3306_3337+m3338_3358+m3359_3388+m3389_3458+m3459_3465+m3466_3470+m3471_3480+m3481_3559+m3560_3574+m3575_3579+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth, train=train, test=test, k=1)

#KNN model for SOC
kknn_SOC = kknn(SOC~m1101_1450+m1451_1750+m1751_1900+m1901_1945+m1946_1955+m1956_1962+m1963_1975+m1976_1985+m1986_2000+m2001_2015+m2016_2050+m2051_2150+m2151_2275+m2276_2376+m2377_2576+m2577_2676+m2677_2825+m2826_2832+m2833_2843+m2834_2900+m2901_2930+m2931_2955+m2956_2972+m2973_2977+m2978_3000+m2_1100+m3001_3020+m3021_3047+m3048_3056+m3057_3078+m3079_3093+m3094_3113+m3114_3153+m3154_3170+m3171_3205+m3206_3217+m3218_3227+m3228_3235+m3236_3250+m3251_3305+m3306_3337+m3338_3358+m3359_3388+m3389_3458+m3459_3465+m3466_3470+m3471_3480+m3481_3559+m3560_3574+m3575_3579+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth, train=train, test=test, k=1)

#KNN model for pH
kknn_pH = kknn(pH~m1101_1450+m1451_1750+m1751_1900+m1901_1945+m1946_1955+m1956_1962+m1963_1975+m1976_1985+m1986_2000+m2001_2015+m2016_2050+m2051_2150+m2151_2275+m2276_2376+m2377_2576+m2577_2676+m2677_2825+m2826_2832+m2833_2843+m2834_2900+m2901_2930+m2931_2955+m2956_2972+m2973_2977+m2978_3000+m2_1100+m3001_3020+m3021_3047+m3048_3056+m3057_3078+m3079_3093+m3094_3113+m3114_3153+m3154_3170+m3171_3205+m3206_3217+m3218_3227+m3228_3235+m3236_3250+m3251_3305+m3306_3337+m3338_3358+m3359_3388+m3389_3458+m3459_3465+m3466_3470+m3471_3480+m3481_3559+m3560_3574+m3575_3579+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth, train=train, test=test, k=1)

#KNN model for Ca
kknn_Ca = kknn(Ca~m1101_1450+m1451_1750+m1751_1900+m1901_1945+m1946_1955+m1956_1962+m1963_1975+m1976_1985+m1986_2000+m2001_2015+m2016_2050+m2051_2150+m2151_2275+m2276_2376+m2377_2576+m2577_2676+m2677_2825+m2826_2832+m2833_2843+m2834_2900+m2901_2930+m2931_2955+m2956_2972+m2973_2977+m2978_3000+m2_1100+m3001_3020+m3021_3047+m3048_3056+m3057_3078+m3079_3093+m3094_3113+m3114_3153+m3154_3170+m3171_3205+m3206_3217+m3218_3227+m3228_3235+m3236_3250+m3251_3305+m3306_3337+m3338_3358+m3359_3388+m3389_3458+m3459_3465+m3466_3470+m3471_3480+m3481_3559+m3560_3574+m3575_3579+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth, train=train, test=test, k=1)

#KNN model for P
kknn_P = kknn(P~m1101_1450+m1451_1750+m1751_1900+m1901_1945+m1946_1955+m1956_1962+m1963_1975+m1976_1985+m1986_2000+m2001_2015+m2016_2050+m2051_2150+m2151_2275+m2276_2376+m2377_2576+m2577_2676+m2677_2825+m2826_2832+m2833_2843+m2834_2900+m2901_2930+m2931_2955+m2956_2972+m2973_2977+m2978_3000+m2_1100+m3001_3020+m3021_3047+m3048_3056+m3057_3078+m3079_3093+m3094_3113+m3114_3153+m3154_3170+m3171_3205+m3206_3217+m3218_3227+m3228_3235+m3236_3250+m3251_3305+m3306_3337+m3338_3358+m3359_3388+m3389_3458+m3459_3465+m3466_3470+m3471_3480+m3481_3559+m3560_3574+m3575_3579+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth, train=train, test=test, k=1)


submission2 = data.frame(PIDN= test$PIDN, Ca=kknn_Ca$fitted.values   ,P=kknn_P$fitted.values	,pH=kknn_pH$fitted.values	,SOC=kknn_SOC$fitted.values	,Sand=kknn_Sand$fitted.values )

write.csv(submission2, "africa_submission2.csv", row.names=FALSE)
#Score: 0.86557... bad

#Try CART model

CART_Sand = rpart(Sand~m1101_1450+m1451_1750+m1751_1900+m1901_1945+m1946_1955+m1956_1962+m1963_1975+m1976_1985+m1986_2000+m2001_2015+m2016_2050+m2051_2150+m2151_2275+m2276_2376+m2377_2576+m2577_2676+m2677_2825+m2826_2832+m2833_2843+m2834_2900+m2901_2930+m2931_2955+m2956_2972+m2973_2977+m2978_3000+m2_1100+m3001_3020+m3021_3047+m3048_3056+m3057_3078+m3079_3093+m3094_3113+m3114_3153+m3154_3170+m3171_3205+m3206_3217+m3218_3227+m3228_3235+m3236_3250+m3251_3305+m3306_3337+m3338_3358+m3359_3388+m3389_3458+m3459_3465+m3466_3470+m3471_3480+m3481_3559+m3560_3574+m3575_3579+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth, data=train,cp=0.00001, minbucket=5)

pred_Sand = predict(CART_Sand, data=train)

#Training RMSE
sum(sqrt((train$Sand-pred_Sand)^2))/nrow(train)

#CART model for SOC
CART_SOC = rpart(SOC~m1101_1450+m1451_1750+m1751_1900+m1901_1945+m1946_1955+m1956_1962+m1963_1975+m1976_1985+m1986_2000+m2001_2015+m2016_2050+m2051_2150+m2151_2275+m2276_2376+m2377_2576+m2577_2676+m2677_2825+m2826_2832+m2833_2843+m2834_2900+m2901_2930+m2931_2955+m2956_2972+m2973_2977+m2978_3000+m2_1100+m3001_3020+m3021_3047+m3048_3056+m3057_3078+m3079_3093+m3094_3113+m3114_3153+m3154_3170+m3171_3205+m3206_3217+m3218_3227+m3228_3235+m3236_3250+m3251_3305+m3306_3337+m3338_3358+m3359_3388+m3389_3458+m3459_3465+m3466_3470+m3471_3480+m3481_3559+m3560_3574+m3575_3579+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth, data=train,cp=0.00001, minbucket=5)

pred_SOC = predict(CART_SOC, data=train)

#Training RMSE
sum(sqrt((train$SOC-pred_SOC)^2))/nrow(train)

#CART model for pH
CART_pH = rpart(pH~m1101_1450+m1451_1750+m1751_1900+m1901_1945+m1946_1955+m1956_1962+m1963_1975+m1976_1985+m1986_2000+m2001_2015+m2016_2050+m2051_2150+m2151_2275+m2276_2376+m2377_2576+m2577_2676+m2677_2825+m2826_2832+m2833_2843+m2834_2900+m2901_2930+m2931_2955+m2956_2972+m2973_2977+m2978_3000+m2_1100+m3001_3020+m3021_3047+m3048_3056+m3057_3078+m3079_3093+m3094_3113+m3114_3153+m3154_3170+m3171_3205+m3206_3217+m3218_3227+m3228_3235+m3236_3250+m3251_3305+m3306_3337+m3338_3358+m3359_3388+m3389_3458+m3459_3465+m3466_3470+m3471_3480+m3481_3559+m3560_3574+m3575_3579+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth, data=train,cp=0.00001, minbucket=5)

pred_pH = predict(CART_pH, data=train)

#Training RMSE
sum(sqrt((train$pH-pred_pH)^2))/nrow(train)

#Linear model for Ca
CART_Ca = rpart(Ca~m1101_1450+m1451_1750+m1751_1900+m1901_1945+m1946_1955+m1956_1962+m1963_1975+m1976_1985+m1986_2000+m2001_2015+m2016_2050+m2051_2150+m2151_2275+m2276_2376+m2377_2576+m2577_2676+m2677_2825+m2826_2832+m2833_2843+m2834_2900+m2901_2930+m2931_2955+m2956_2972+m2973_2977+m2978_3000+m2_1100+m3001_3020+m3021_3047+m3048_3056+m3057_3078+m3079_3093+m3094_3113+m3114_3153+m3154_3170+m3171_3205+m3206_3217+m3218_3227+m3228_3235+m3236_3250+m3251_3305+m3306_3337+m3338_3358+m3359_3388+m3389_3458+m3459_3465+m3466_3470+m3471_3480+m3481_3559+m3560_3574+m3575_3579+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth, data=train,cp=0.00001, minbucket=5)

pred_Ca = predict(CART_Ca, data=train)

#Training RMSE
sum(sqrt((train$Ca-pred_Ca)^2))/nrow(train)

#Linear model for P
CART_P = rpart(P~m1101_1450+m1451_1750+m1751_1900+m1901_1945+m1946_1955+m1956_1962+m1963_1975+m1976_1985+m1986_2000+m2001_2015+m2016_2050+m2051_2150+m2151_2275+m2276_2376+m2377_2576+m2577_2676+m2677_2825+m2826_2832+m2833_2843+m2834_2900+m2901_2930+m2931_2955+m2956_2972+m2973_2977+m2978_3000+m2_1100+m3001_3020+m3021_3047+m3048_3056+m3057_3078+m3079_3093+m3094_3113+m3114_3153+m3154_3170+m3171_3205+m3206_3217+m3218_3227+m3228_3235+m3236_3250+m3251_3305+m3306_3337+m3338_3358+m3359_3388+m3389_3458+m3459_3465+m3466_3470+m3471_3480+m3481_3559+m3560_3574+m3575_3579+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth, data=train,cp=0.00001, minbucket=5)

pred_P = predict(CART_P, data=train)

#Training RMSE
sum(sqrt((train$P-pred_P)^2))/nrow(train)

tpred_Sand = predict(CART_Sand, newdata=test)
tpred_pH = predict(CART_pH, newdata=test)
tpred_Ca = predict(CART_Ca, newdata=test)
tpred_SOC = predict(CART_SOC, newdata=test)
tpred_P = predict(CART_P, newdata=test)


submission3 = data.frame(PIDN= test$PIDN, Ca=tpred_Ca   ,P=tpred_P	,pH=tpred_pH	,SOC=tpred_SOC	,Sand=tpred_Sand )

summary(submission3)


write.csv(submission3, "africa_submission3.csv", row.names=FALSE)
#Score: 0.93687, Overfitting?

#Try Caret package with GBM and cross validation...

tunecontrol = trainControl(method = "repeatedcv",
                         number = 2,
                         repeats = 1
)

tgrid = expand.grid( n.trees=c(150),interaction.depth=c(20),shrinkage=c(0.107))

gbm_Sand = train(Sand~m1101_1450+m1451_1750+m1751_1900+m1901_1945+m1946_1955+m1956_1962+m1963_1975+m1976_1985+m1986_2000+m2001_2015+m2016_2050+m2051_2150+m2151_2275+m2276_2376+m2377_2576+m2577_2676+m2677_2825+m2826_2832+m2833_2843+m2834_2900+m2901_2930+m2931_2955+m2956_2972+m2973_2977+m2978_3000+m2_1100+m3001_3020+m3021_3047+m3048_3056+m3057_3078+m3079_3093+m3094_3113+m3114_3153+m3154_3170+m3171_3205+m3206_3217+m3218_3227+m3228_3235+m3236_3250+m3251_3305+m3306_3337+m3338_3358+m3359_3388+m3389_3458+m3459_3465+m3466_3470+m3471_3480+m3481_3559+m3560_3574+m3575_3579+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth, data=train, method='gbm', trControl=tunecontrol, tuneGrid=tgrid)

gbm_Sand

pred_Sand = predict(gbm_Sand, newdata=test)


tunecontrol = trainControl(method = "repeatedcv",
                           number = 3,
                           repeats = 1
)

tgrid = expand.grid( n.trees=c(150),interaction.depth=c(20),shrinkage=c(0.107))

gbm_SOC = train(SOC~m1101_1450+m1451_1750+m1751_1900+m1901_1945+m1946_1955+m1956_1962+m1963_1975+m1976_1985+m1986_2000+m2001_2015+m2016_2050+m2051_2150+m2151_2275+m2276_2376+m2377_2576+m2577_2676+m2677_2825+m2826_2832+m2833_2843+m2834_2900+m2901_2930+m2931_2955+m2956_2972+m2973_2977+m2978_3000+m2_1100+m3001_3020+m3021_3047+m3048_3056+m3057_3078+m3079_3093+m3094_3113+m3114_3153+m3154_3170+m3171_3205+m3206_3217+m3218_3227+m3228_3235+m3236_3250+m3251_3305+m3306_3337+m3338_3358+m3359_3388+m3389_3458+m3459_3465+m3466_3470+m3471_3480+m3481_3559+m3560_3574+m3575_3579+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth, data=train, method='gbm', trControl=tunecontrol, tuneGrid=tgrid)

gbm_SOC

pred_SOC = predict(gbm_SOC, newdata=test)

tunecontrol = trainControl(method = "repeatedcv",
                           number = 3,
                           repeats = 1
)

tgrid = expand.grid( n.trees=c(100),interaction.depth=c(20),shrinkage=c(0.0.107))

gbm_P = train(P~m1101_1450+m1451_1750+m1751_1900+m1901_1945+m1946_1955+m1956_1962+m1963_1975+m1976_1985+m1986_2000+m2001_2015+m2016_2050+m2051_2150+m2151_2275+m2276_2376+m2377_2576+m2577_2676+m2677_2825+m2826_2832+m2833_2843+m2834_2900+m2901_2930+m2931_2955+m2956_2972+m2973_2977+m2978_3000+m2_1100+m3001_3020+m3021_3047+m3048_3056+m3057_3078+m3079_3093+m3094_3113+m3114_3153+m3154_3170+m3171_3205+m3206_3217+m3218_3227+m3228_3235+m3236_3250+m3251_3305+m3306_3337+m3338_3358+m3359_3388+m3389_3458+m3459_3465+m3466_3470+m3471_3480+m3481_3559+m3560_3574+m3575_3579+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth, data=train, method='gbm', trControl=tunecontrol, tuneGrid=tgrid)

gbm_P

pred_P = predict(gbm_P, newdata=test)



tunecontrol = trainControl(method = "repeatedcv",
                           number = 3,
                           repeats = 1
)

tgrid = expand.grid( n.trees=c(150),interaction.depth=c(20),shrinkage=c(0.107))



gbm_pH = train(pH~m1101_1450+m1451_1750+m1751_1900+m1901_1945+m1946_1955+m1956_1962+m1963_1975+m1976_1985+m1986_2000+m2001_2015+m2016_2050+m2051_2150+m2151_2275+m2276_2376+m2377_2576+m2577_2676+m2677_2825+m2826_2832+m2833_2843+m2834_2900+m2901_2930+m2931_2955+m2956_2972+m2973_2977+m2978_3000+m2_1100+m3001_3020+m3021_3047+m3048_3056+m3057_3078+m3079_3093+m3094_3113+m3114_3153+m3154_3170+m3171_3205+m3206_3217+m3218_3227+m3228_3235+m3236_3250+m3251_3305+m3306_3337+m3338_3358+m3359_3388+m3389_3458+m3459_3465+m3466_3470+m3471_3480+m3481_3559+m3560_3574+m3575_3579+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth, data=train, method='gbm', trControl=tunecontrol, tuneGrid=tgrid)

gbm_pH

pred_pH = predict(gbm_pH, newdata=test)


tunecontrol = trainControl(method = "repeatedcv",
                           number = 3,
                           repeats = 1
)

tgrid = expand.grid( n.trees=c(150),interaction.depth=c(10),shrinkage=c(0.11))



gbm_Ca = train(Ca~m1101_1450+m1451_1750+m1751_1900+m1901_1945+m1946_1955+m1956_1962+m1963_1975+m1976_1985+m1986_2000+m2001_2015+m2016_2050+m2051_2150+m2151_2275+m2276_2376+m2377_2576+m2577_2676+m2677_2825+m2826_2832+m2833_2843+m2834_2900+m2901_2930+m2931_2955+m2956_2972+m2973_2977+m2978_3000+m2_1100+m3001_3020+m3021_3047+m3048_3056+m3057_3078+m3079_3093+m3094_3113+m3114_3153+m3154_3170+m3171_3205+m3206_3217+m3218_3227+m3228_3235+m3236_3250+m3251_3305+m3306_3337+m3338_3358+m3359_3388+m3389_3458+m3459_3465+m3466_3470+m3471_3480+m3481_3559+m3560_3574+m3575_3579+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth, data=train, method='gbm', trControl=tunecontrol, tuneGrid=tgrid)

gbm_Ca

pred_Ca = predict(gbm_Ca, newdata=test)



submission4 = data.frame(PIDN= test$PIDN, Ca=pred_Ca   ,P=pred_P  ,pH=pred_pH	,SOC=pred_SOC	,Sand=pred_Sand )

write.csv(submission4, "africa_submission4.csv", row.names=FALSE)
#Score: 0.68 ... Need methods better at estimating continuous data?

#Testing other methods---------

set.seed(121)

tunecontrol = trainControl(method = "repeatedcv",
                           number = 3,
                           repeats = 1
)

tgrid = expand.grid(neurons=c(4))

m_Sand = train(Sand~m1101_1450+m1451_1750+m1751_1900+m1901_1945+m1946_1955+m1956_1962+m1963_1975+m1976_1985+m1986_2000+m2001_2015+m2016_2050+m2051_2150+m2151_2275+m2276_2376+m2377_2576+m2577_2676+m2677_2825+m2826_2832+m2833_2843+m2834_2900+m2901_2930+m2931_2955+m2956_2972+m2973_2977+m2978_3000+m2_1100+m3001_3020+m3021_3047+m3048_3056+m3057_3078+m3079_3093+m3094_3113+m3114_3153+m3154_3170+m3171_3205+m3206_3217+m3218_3227+m3228_3235+m3236_3250+m3251_3305+m3306_3337+m3338_3358+m3359_3388+m3389_3458+m3459_3465+m3466_3470+m3471_3480+m3481_3559+m3560_3574+m3575_3579+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth, data=train, method='brnn', trControl=tunecontrol, tuneGrid=tgrid)

m_Sand

tunecontrol = trainControl(method = "repeatedcv",
                           number = 3,
                           repeats = 1
)

tgrid = expand.grid(C = 12)

m_pH = train(pH~m1101_1450+m1451_1750+m1751_1900+m1901_1945+m1946_1955+m1956_1962+m1963_1975+m1976_1985+m1986_2000+m2001_2015+m2016_2050+m2051_2150+m2151_2275+m2276_2376+m2377_2576+m2577_2676+m2677_2825+m2826_2832+m2833_2843+m2834_2900+m2901_2930+m2931_2955+m2956_2972+m2973_2977+m2978_3000+m2_1100+m3001_3020+m3021_3047+m3048_3056+m3057_3078+m3079_3093+m3094_3113+m3114_3153+m3154_3170+m3171_3205+m3206_3217+m3218_3227+m3228_3235+m3236_3250+m3251_3305+m3306_3337+m3338_3358+m3359_3388+m3389_3458+m3459_3465+m3466_3470+m3471_3480+m3481_3559+m3560_3574+m3575_3579+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth, data=train, method='svmRadialCost', trControl=tunecontrol, tuneGrid=tgrid)

m_pH

tunecontrol = trainControl(method = "repeatedcv",
                           number = 3,
                           repeats = 1
)

tgrid = expand.grid(C = 9)

m_Ca = train(Ca~m1101_1450+m1451_1750+m1751_1900+m1901_1945+m1946_1955+m1956_1962+m1963_1975+m1976_1985+m1986_2000+m2001_2015+m2016_2050+m2051_2150+m2151_2275+m2276_2376+m2377_2576+m2577_2676+m2677_2825+m2826_2832+m2833_2843+m2834_2900+m2901_2930+m2931_2955+m2956_2972+m2973_2977+m2978_3000+m2_1100+m3001_3020+m3021_3047+m3048_3056+m3057_3078+m3079_3093+m3094_3113+m3114_3153+m3154_3170+m3171_3205+m3206_3217+m3218_3227+m3228_3235+m3236_3250+m3251_3305+m3306_3337+m3338_3358+m3359_3388+m3389_3458+m3459_3465+m3466_3470+m3471_3480+m3481_3559+m3560_3574+m3575_3579+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth, data=train, method='svmRadialCost', trControl=tunecontrol, tuneGrid=tgrid)

m_Ca

tunecontrol = trainControl(method = "repeatedcv",
                           number = 3,
                           repeats = 1
)

tgrid = expand.grid(C = 9)

m_SOC = train(SOC~m1101_1450+m1451_1750+m1751_1900+m1901_1945+m1946_1955+m1956_1962+m1963_1975+m1976_1985+m1986_2000+m2001_2015+m2016_2050+m2051_2150+m2151_2275+m2276_2376+m2377_2576+m2577_2676+m2677_2825+m2826_2832+m2833_2843+m2834_2900+m2901_2930+m2931_2955+m2956_2972+m2973_2977+m2978_3000+m2_1100+m3001_3020+m3021_3047+m3048_3056+m3057_3078+m3079_3093+m3094_3113+m3114_3153+m3154_3170+m3171_3205+m3206_3217+m3218_3227+m3228_3235+m3236_3250+m3251_3305+m3306_3337+m3338_3358+m3359_3388+m3389_3458+m3459_3465+m3466_3470+m3471_3480+m3481_3559+m3560_3574+m3575_3579+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth, data=train, method='svmRadialCost', trControl=tunecontrol, tuneGrid=tgrid)

m_SOC

tunecontrol = trainControl(method = "repeatedcv",
                           number = 3,
                           repeats = 1
)

tgrid = expand.grid(C = 20)

m_P = train(P~m1101_1450+m1451_1750+m1751_1900+m1901_1945+m1946_1955+m1956_1962+m1963_1975+m1976_1985+m1986_2000+m2001_2015+m2016_2050+m2051_2150+m2151_2275+m2276_2376+m2377_2576+m2577_2676+m2677_2825+m2826_2832+m2833_2843+m2834_2900+m2901_2930+m2931_2955+m2956_2972+m2973_2977+m2978_3000+m2_1100+m3001_3020+m3021_3047+m3048_3056+m3057_3078+m3079_3093+m3094_3113+m3114_3153+m3154_3170+m3171_3205+m3206_3217+m3218_3227+m3228_3235+m3236_3250+m3251_3305+m3306_3337+m3338_3358+m3359_3388+m3389_3458+m3459_3465+m3466_3470+m3471_3480+m3481_3559+m3560_3574+m3575_3579+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth, data=train, method='svmRadialCost', trControl=tunecontrol, tuneGrid=tgrid)

m_P

#extraTrees RMSE scores
#Sand - 0.393   (mtry=c(5),numRandomCuts=c(3))
#Ca - 0.362  (mtry=c(7),numRandomCuts=c(3))
#pH - 0.453   (mtry=c(5),numRandomCuts=c(3))
#P - 0.85  (mtry=c(5),numRandomCuts=c(3))
#SOC - 0.46  (mtry=c(5),numRandomCuts=c(3))

#cubist RMSE -- seems to perform ok on everything other than P
#Sand - 0.335   (committees=c(6),neighbors=c(4))
#Ca - 0.311  (committees=c(6),neighbors=c(6))
#pH - 0.365   (committees=c(6),neighbors=c(6))
#P - 0.974  (committees=c(6),neighbors=c(6))  ##Nothing performs well on P.
#SOC - 0.35  (committees=c(6),neighbors=c(5))

#'glmboost'--worse results...

#lasso RMSE scores
#Sand - 0.387  (fraction=c(0.65)
#P - 0.99 (fraction=c(0.65)

#'spls'
#Sand - 0.386  K = 40, eta = 0.8 and kappa = 0.1.
#P - 0.95 K = 60, eta = 0.8 and kappa = 0.1.

#'svmRadialCost' RMSE scores - Radial Basis Kernel SVM --does almost as well as cubist
#Sand - 0.348   C = 9
#Ca - 0.336   C = 10
#pH - 0.39    C=12
#P - 0.929   C=20
#SOC - 0.338   C = 10

tpred_Sand = predict(m_Sand, newdata=test)
tpred_pH = predict(m_pH, newdata=test)
tpred_Ca = predict(m_Ca, newdata=test)
tpred_SOC = predict(m_SOC, newdata=test)
tpred_P = predict(m_P, newdata=test)

submission5 = data.frame(PIDN= test$PIDN, Ca=tpred_Ca   ,P=tpred_P  ,pH=tpred_pH	,SOC=tpred_SOC	,Sand=tpred_Sand )

summary(submission5)

write.csv(submission5, "africa_submission5.csv", row.names=FALSE)




#-------------------------------------------------------
#Trying Bayes NN, using linear model for P


set.seed(121)

tunecontrol = trainControl(method = "repeatedcv",
                           number = 3,
                           repeats = 1
)

tgrid = expand.grid(neurons=c(4))

m_Sand = train(Sand~m1101_1450+m1451_1750+m1751_1900+m1901_1945+m1946_1955+m1956_1962+m1963_1975+m1976_1985+m1986_2000+m2001_2015+m2016_2050+m2051_2150+m2151_2275+m2276_2376+m2377_2576+m2577_2676+m2677_2825+m2826_2832+m2833_2843+m2834_2900+m2901_2930+m2931_2955+m2956_2972+m2973_2977+m2978_3000+m2_1100+m3001_3020+m3021_3047+m3048_3056+m3057_3078+m3079_3093+m3094_3113+m3114_3153+m3154_3170+m3171_3205+m3206_3217+m3218_3227+m3228_3235+m3236_3250+m3251_3305+m3306_3337+m3338_3358+m3359_3388+m3389_3458+m3459_3465+m3466_3470+m3471_3480+m3481_3559+m3560_3574+m3575_3579+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth, data=train, method='brnn', trControl=tunecontrol, tuneGrid=tgrid)

m_Sand

tunecontrol = trainControl(method = "repeatedcv",
                           number = 3,
                           repeats = 1
)

tgrid = expand.grid(neurons=c(4))

m_pH = train(pH~m1101_1450+m1451_1750+m1751_1900+m1901_1945+m1946_1955+m1956_1962+m1963_1975+m1976_1985+m1986_2000+m2001_2015+m2016_2050+m2051_2150+m2151_2275+m2276_2376+m2377_2576+m2577_2676+m2677_2825+m2826_2832+m2833_2843+m2834_2900+m2901_2930+m2931_2955+m2956_2972+m2973_2977+m2978_3000+m2_1100+m3001_3020+m3021_3047+m3048_3056+m3057_3078+m3079_3093+m3094_3113+m3114_3153+m3154_3170+m3171_3205+m3206_3217+m3218_3227+m3228_3235+m3236_3250+m3251_3305+m3306_3337+m3338_3358+m3359_3388+m3389_3458+m3459_3465+m3466_3470+m3471_3480+m3481_3559+m3560_3574+m3575_3579+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth, data=train, method='brnn', trControl=tunecontrol, tuneGrid=tgrid)

m_pH

tunecontrol = trainControl(method = "repeatedcv",
                           number = 3,
                           repeats = 1
)

tgrid = expand.grid(neurons=c(1))

m_Ca = train(Ca~m1101_1450+m1451_1750+m1751_1900+m1901_1945+m1946_1955+m1956_1962+m1963_1975+m1976_1985+m1986_2000+m2001_2015+m2016_2050+m2051_2150+m2151_2275+m2276_2376+m2377_2576+m2577_2676+m2677_2825+m2826_2832+m2833_2843+m2834_2900+m2901_2930+m2931_2955+m2956_2972+m2973_2977+m2978_3000+m2_1100+m3001_3020+m3021_3047+m3048_3056+m3057_3078+m3079_3093+m3094_3113+m3114_3153+m3154_3170+m3171_3205+m3206_3217+m3218_3227+m3228_3235+m3236_3250+m3251_3305+m3306_3337+m3338_3358+m3359_3388+m3389_3458+m3459_3465+m3466_3470+m3471_3480+m3481_3559+m3560_3574+m3575_3579+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth, data=train, method='brnn', trControl=tunecontrol, tuneGrid=tgrid)

m_Ca

tunecontrol = trainControl(method = "repeatedcv",
                           number = 3,
                           repeats = 1
)

tgrid = expand.grid(neurons=c(1))

m_SOC = train(SOC~m1101_1450+m1451_1750+m1751_1900+m1901_1945+m1946_1955+m1956_1962+m1963_1975+m1976_1985+m1986_2000+m2001_2015+m2016_2050+m2051_2150+m2151_2275+m2276_2376+m2377_2576+m2577_2676+m2677_2825+m2826_2832+m2833_2843+m2834_2900+m2901_2930+m2931_2955+m2956_2972+m2973_2977+m2978_3000+m2_1100+m3001_3020+m3021_3047+m3048_3056+m3057_3078+m3079_3093+m3094_3113+m3114_3153+m3154_3170+m3171_3205+m3206_3217+m3218_3227+m3228_3235+m3236_3250+m3251_3305+m3306_3337+m3338_3358+m3359_3388+m3389_3458+m3459_3465+m3466_3470+m3471_3480+m3481_3559+m3560_3574+m3575_3579+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth, data=train, method='brnn', trControl=tunecontrol, tuneGrid=tgrid)

m_SOC

tunecontrol = trainControl(method = "repeatedcv",
                           number = 3,
                           repeats = 1
)

tgrid = expand.grid(neurons=c(1))

m_P = train(P~m1101_1450+m1451_1750+m1751_1900+m1901_1945+m1946_1955+m1956_1962+m1963_1975+m1976_1985+m1986_2000+m2001_2015+m2016_2050+m2051_2150+m2151_2275+m2276_2376+m2377_2576+m2577_2676+m2677_2825+m2826_2832+m2833_2843+m2834_2900+m2901_2930+m2931_2955+m2956_2972+m2973_2977+m2978_3000+m2_1100+m3001_3020+m3021_3047+m3048_3056+m3057_3078+m3079_3093+m3094_3113+m3114_3153+m3154_3170+m3171_3205+m3206_3217+m3218_3227+m3228_3235+m3236_3250+m3251_3305+m3306_3337+m3338_3358+m3359_3388+m3389_3458+m3459_3465+m3466_3470+m3471_3480+m3481_3559+m3560_3574+m3575_3579+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth, data=train, method='brnn', trControl=tunecontrol, tuneGrid=tgrid)

m_P


tpred_Sand = predict(m_Sand, newdata=test)
tpred_pH = predict(m_pH, newdata=test)
tpred_Ca = predict(m_Ca, newdata=test)
tpred_SOC = predict(m_SOC, newdata=test)
tpred_P = predict(m_P, newdata=test)

submission8 = data.frame(PIDN= test$PIDN, Ca=tpred_Ca   ,P=tpred_P  ,pH=tpred_pH  ,SOC=tpred_SOC	,Sand=tpred_Sand )

summary(submission8)

write.csv(submission8, "africa_submission8.csv", row.names=FALSE)





#Rerun linear model using new data(data trimmed using method 2)
#Linear model for sand
regmod_Sand = glm(Sand~m7474.82+m5237.78+m3694.98+m3453.92+m2518.61+m2055.77+m2046.13+m2036.48+m1901.49+m1795.42+m1766.5+m1756.85+m1621.86+m1602.57+m1544.72+m1486.86+m1419.37+m1303.66+m1294.02+m1284.37+m1274.73+m1265.09+m1255.45+m1197.59+m1110.81+m1024.03+m937.246+m831.179+m821.536+m811.894+m696.184+m611.331+m601.688+m599.76+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI,  data=train)

pred_Sand = predict(regmod_Sand, data=train)

#Training RMSE
sum(sqrt((train$Sand-pred_Sand)^2))/nrow(train)

#Linear model for SOC
regmod_SOC = glm(SOC~m7474.82+m5237.78+m3694.98+m3453.92+m2518.61+m2055.77+m2046.13+m2036.48+m1901.49+m1795.42+m1766.5+m1756.85+m1621.86+m1602.57+m1544.72+m1486.86+m1419.37+m1303.66+m1294.02+m1284.37+m1274.73+m1265.09+m1255.45+m1197.59+m1110.81+m1024.03+m937.246+m831.179+m821.536+m811.894+m696.184+m611.331+m601.688+m599.76+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI,  data=train)

pred_SOC = predict(regmod_SOC, data=train)

#Training RMSE
sum(sqrt((train$SOC-pred_SOC)^2))/nrow(train)

#Linear model for pH
regmod_pH = glm(pH~m7474.82+m5237.78+m3694.98+m3453.92+m2518.61+m2055.77+m2046.13+m2036.48+m1901.49+m1795.42+m1766.5+m1756.85+m1621.86+m1602.57+m1544.72+m1486.86+m1419.37+m1303.66+m1294.02+m1284.37+m1274.73+m1265.09+m1255.45+m1197.59+m1110.81+m1024.03+m937.246+m831.179+m821.536+m811.894+m696.184+m611.331+m601.688+m599.76+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI,  data=train)

pred_pH = predict(regmod_pH, data=train)

#Training RMSE
sum(sqrt((train$pH-pred_pH)^2))/nrow(train)

#Linear model for Ca
regmod_Ca = glm(Ca~m7474.82+m5237.78+m3694.98+m3453.92+m2518.61+m2055.77+m2046.13+m2036.48+m1901.49+m1795.42+m1766.5+m1756.85+m1621.86+m1602.57+m1544.72+m1486.86+m1419.37+m1303.66+m1294.02+m1284.37+m1274.73+m1265.09+m1255.45+m1197.59+m1110.81+m1024.03+m937.246+m831.179+m821.536+m811.894+m696.184+m611.331+m601.688+m599.76+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI,  data=train)

pred_Ca = predict(regmod_Ca, data=train)

#Training RMSE
sum(sqrt((train$Ca-pred_Ca)^2))/nrow(train)

#Linear model for P
regmod_P = glm(P~m7474.82+m5237.78+m3694.98+m3453.92+m2518.61+m2055.77+m2046.13+m2036.48+m1901.49+m1795.42+m1766.5+m1756.85+m1621.86+m1602.57+m1544.72+m1486.86+m1419.37+m1303.66+m1294.02+m1284.37+m1274.73+m1265.09+m1255.45+m1197.59+m1110.81+m1024.03+m937.246+m831.179+m821.536+m811.894+m696.184+m611.331+m601.688+m599.76+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI,  data=train)

pred_P = predict(regmod_P, data=train)

#Training RMSE
sum(sqrt((train$P-pred_P)^2))/nrow(train)

tpred_Sand = predict(regmod_Sand, newdata=test)
tpred_pH = predict(regmod_pH, newdata=test)
tpred_Ca = predict(regmod_Ca, newdata=test)
tpred_SOC = predict(regmod_SOC, newdata=test)
tpred_P = predict(regmod_P, newdata=test)


submission9 = data.frame(PIDN= test$PIDN, Ca=tpred_Ca   ,P=tpred_P	,pH=tpred_pH	,SOC=tpred_SOC	,Sand=tpred_Sand )

write.csv(submission9, "africa_submission9.csv", row.names=FALSE)
#Score =0.577

#-----------------------------------------------------------------------

#Rerun linear model using new data(data trimmed using method 3)
#Linear model for sand
regmod_Sand = glm(Sand~m7482.54+m5237.78+m4526.16+m3698.84+m3450.07+m2512.82+m2044.2+m1899.56+m1795.42+m1766.5+m1598.72+m1488.79+m1419.37+m1303.66+m1193.73+m1112.74+m1037.53+m1020.17+m944.959+m823.465+m811.894+m696.184+m599.76+m601.688+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth
,  data=train)

pred_Sand = predict(regmod_Sand, data=train)

#Training RMSE
sum(sqrt((train$Sand-pred_Sand)^2))/nrow(train)

#Linear model for SOC
regmod_SOC = glm(SOC~m7482.54+m5237.78+m4526.16+m3698.84+m3450.07+m2512.82+m2044.2+m1899.56+m1795.42+m1766.5+m1598.72+m1488.79+m1419.37+m1303.66+m1193.73+m1112.74+m1037.53+m1020.17+m944.959+m823.465+m811.894+m696.184+m599.76+m601.688+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth
,  data=train)

pred_SOC = predict(regmod_SOC, data=train)

#Training RMSE
sum(sqrt((train$SOC-pred_SOC)^2))/nrow(train)

#Linear model for pH
regmod_pH = glm(pH~m7482.54+m5237.78+m4526.16+m3698.84+m3450.07+m2512.82+m2044.2+m1899.56+m1795.42+m1766.5+m1598.72+m1488.79+m1419.37+m1303.66+m1193.73+m1112.74+m1037.53+m1020.17+m944.959+m823.465+m811.894+m696.184+m599.76+m601.688+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth
,  data=train)

pred_pH = predict(regmod_pH, data=train)

#Training RMSE
sum(sqrt((train$pH-pred_pH)^2))/nrow(train)

#Linear model for Ca
regmod_Ca = glm(Ca~m7482.54+m5237.78+m4526.16+m3698.84+m3450.07+m2512.82+m2044.2+m1899.56+m1795.42+m1766.5+m1598.72+m1488.79+m1419.37+m1303.66+m1193.73+m1112.74+m1037.53+m1020.17+m944.959+m823.465+m811.894+m696.184+m599.76+m601.688+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth
,  data=train)

pred_Ca = predict(regmod_Ca, data=train)

#Training RMSE
sum(sqrt((train$Ca-pred_Ca)^2))/nrow(train)

#Linear model for P
regmod_P = glm(P~m7482.54+m5237.78+m4526.16+m3698.84+m3450.07+m2512.82+m2044.2+m1899.56+m1795.42+m1766.5+m1598.72+m1488.79+m1419.37+m1303.66+m1193.73+m1112.74+m1037.53+m1020.17+m944.959+m823.465+m811.894+m696.184+m599.76+m601.688+BSAN+BSAS+BSAV+CTI+ELEV+EVI+LSTD+LSTN+REF1+REF2+REF3+REF7+RELI+TMAP+TMFI+Depth
,  data=train)

pred_P = predict(regmod_P, data=train)

#Training RMSE
sum(sqrt((train$P-pred_P)^2))/nrow(train)

tpred_Sand = predict(regmod_Sand, newdata=test)
tpred_pH = predict(regmod_pH, newdata=test)
tpred_Ca = predict(regmod_Ca, newdata=test)
tpred_SOC = predict(regmod_SOC, newdata=test)
tpred_P = predict(regmod_P, newdata=test)


submission10 = data.frame(PIDN= test$PIDN, Ca=tpred_Ca   ,P=tpred_P  ,pH=tpred_pH	,SOC=tpred_SOC	,Sand=tpred_Sand )

write.csv(submission10, "africa_submission10.csv", row.names=FALSE)
#Score = 




#Trymodel without eliminating any features

#Rerun linear model using new data(data trimmed using method 3)
#Linear model for sand
strain =train[ ,c(2:3595,3600)]
regmod_Sand = glm(Sand~.,  data=strain)

soctrain =train[ ,c(2:3595,3599)]
#Linear model for SOC
regmod_SOC = glm(SOC~.,  data=soctrain)

#Linear model for pH
phtrain = train[ ,c(2:3595,3598)]
regmod_pH = glm(pH~.,  data=phtrain)

#Linear model for Ca
catrain = train[ ,c(2:3595,3596)]
regmod_Ca = glm(Ca~.,  data=catrain)

#Linear model for P
ptrain = train[ ,c(2:3595,3597)]
regmod_P = glm(P~.
               ,  data=ptrain)


tpred_Sand = predict(regmod_Sand, newdata=Xtest)
tpred_pH = predict(regmod_pH, newdata=Xtest)
tpred_Ca = predict(regmod_Ca, newdata=Xtest)
tpred_SOC = predict(regmod_SOC, newdata=Xtest)
tpred_P = predict(regmod_P, newdata=Xtest)


submission11 = data.frame(PIDN= test$PIDN, Ca=tpred_Ca   ,P=tpred_P  ,pH=tpred_pH  ,SOC=tpred_SOC	,Sand=tpred_Sand )

write.csv(submission11, "africa_submission11.csv", row.names=FALSE)