#Denoising Dirty Documents
#https://www.kaggle.com/c/denoising-dirty-documents


library(ggplot2)
library(rpart)
library(caret)
library(randomForest)
library(e1071)
library(glmnet)
library(xgboost)
library(deepnet)
library(dplyr)
library(data.table)
library(bit64)
library(pROC)
library(png)
library(grid)
library(Metrics)

sample_sub = fread("sampleSubmission.csv")


train_files <- list.files("train", pattern="*.png", full.names=TRUE)

#Read training images into a list
train_images = list()

for (img in train_files){
  train_images[[img]] = readPNG(img, native = FALSE, info = FALSE)
}

train_cleaned <- list.files("train_cleaned", pattern="*.png", full.names=TRUE)

#Read training images into a list
train_cleaned_images = list()

for (img in train_cleaned){
  train_cleaned_images[[img]] = readPNG(img, native = FALSE, info = FALSE)
}

test_files <- list.files("test", pattern="*.png", full.names=TRUE)

#Read test images into a list
test_images = list()

for (img in test_files){
  test_images[[img]] = readPNG(img, native = FALSE, info = FALSE)
}


#There are 2 image sizes: 540x258 and 540x420  with 139320 and 226800 pixels respectively.


#Function to view images in R:
# grid.raster(img_matrix)

#Exploration area----------------------------------
test_image = "test/1.png"


#Try some basic thresholding on 1 image
image1 = test_images[[test_image]]
avg_pixel = mean(image1)
# grid.raster(image1)
dark_ratio = length(which(image1 < 0.35))/length(image1)

#Start with a strong treshold to remove most/all nontext pixels
thres_img = ifelse(image1>(avg_pixel-(4*dark_ratio)), 0, 1)
grid.raster(thres_img)

#Spread pixel intensity in vertical direction by 1 pixel
cols = rep(0,ncol(thres_img))

smear_vertical_by =1
for (x in 1:smear_vertical_by){
  thres_d_1 = rbind(thres_img, cols )[2:(nrow(thres_img)+1), ]
  thres_u_1 = rbind(cols,thres_img )[1:(nrow(thres_img)), ]
  thres_img = (thres_img+thres_d_1+thres_u_1)
}

rows = rep(0,nrow(thres_img))

smear_laterally_by =1
for (x in 1:smear_laterally_by){
  thres_r_1 = cbind(thres_img, rows )[,2:(ncol(thres_img)+1)]
  thres_l_1 = cbind(rows, thres_img)[,1:ncol(thres_img)]
  thres_img = (thres_img+thres_r_1+thres_l_1)
}


#Create new image matrix using filter matrix
filter_matrix = ifelse(thres_img ==0,-1,1)

flitered_image = ifelse(filter_matrix ==-1,1, image1)
grid.raster(flitered_image)
#Exploration area----------------------------------


#First threshold model validation--------
avg_rmse = 0

for (image in names(train_images)){
print(image)

#Calculate average img pixel intensity
img = train_images[[image]]
avg_pixel = mean(img)


#Create simple threshold
thres_img = ifelse(img>0.95, 1, 
            ifelse(img<0.025, 0, 
            ifelse(img>avg_pixel-0.15, 0.985,
                   img)))

#Actual cleaned image:
test_image =paste("train_cleaned/", strsplit(image,"/")[[1]][2], sep="")
print(test_image)

cleaned_img = train_cleaned_images[[test_image]]

img_rmse = rmse(thres_img, cleaned_img)
print ( img_rmse )

avg_rmse = avg_rmse+img_rmse
}
avg_rmse= avg_rmse/144
print(avg_rmse)
#First threshold model validation--------


#First threshold model submission--------
#Run threshold on all test_images and submit
cleaned_test = list()

for (image in names(test_images)){
  img = test_images[[image]]
  avg_pixel = mean(img)
  thres_img = thres_img = ifelse(img>avg_pixel-0.225, 0.985, img)
  cleaned_test[[image]] = thres_img
}

solution_vector = c()

for (image in cleaned_test){
  solution_vector = c(solution_vector, as.vector(image))
}

submission1 = sample_sub
submission1$value = solution_vector
write.csv(submission1, "denoising_sub1_thresh.csv", row.names=FALSE)
#Submission score: 0.08036  First place! (out of 5...)
#First threshold model submission--------



#Look at test output images:
test_image = "test/154.png"
test_img = test_images[[test_image]]

avg_pixel = mean(test_img)

thres_img = ifelse(test_img>avg_pixel-0.225, 0.985, test_img)
grid.raster(thres_img )



#Training filter matrix model

avg_rmse = 0

for (image in names(train_images)){
  print(image)
  
  #Try some basic thresholding on 1 image
  image1 = train_images[[image]]
  avg_pixel = mean(image1)
  # grid.raster(image1)
  dark_ratio = length(which(image1 < 0.35))/length(image1)
  
  #Start with a strong treshold to remove most/all nontext pixels
  thres_img = ifelse(image1>(avg_pixel-(4*dark_ratio)), 0, 1)
  
  #Smear pixel intensity vertically and laterally
  cols = rep(0,ncol(thres_img))
  
  smear_vertical_by =1
  for (x in 1:smear_vertical_by){
    thres_d_1 = rbind(thres_img, cols )[2:(nrow(thres_img)+1), ]
    thres_u_1 = rbind(cols,thres_img )[1:(nrow(thres_img)), ]
    thres_img = (thres_img+thres_d_1+thres_u_1)
  }
  
  rows = rep(0,nrow(thres_img))
  
  smear_laterally_by =1
  for (x in 1:smear_laterally_by){
    thres_r_1 = cbind(thres_img, rows )[,2:(ncol(thres_img)+1)]
    thres_l_1 = cbind(rows, thres_img)[,1:ncol(thres_img)]
    thres_img = (thres_img+thres_r_1+thres_l_1)
  }
  
  
  #Create new image matrix using smeared filter matrix
  filter_matrix = ifelse(thres_img ==0,-1,1)
  
  flitered_image = ifelse(filter_matrix ==-1,1, image1)
  
#   grid.raster(flitered_image)
  
  #Calculate average non-white pixel intensity of flitered image
  white_space = (flitered_image==1)
  non_white_vec = flitered_image[!white_space]
  avg_non_zero = mean(non_white_vec)
  
  #Use a simple threshold
  final_img = ifelse(flitered_image>avg_non_zero+0.10, 0.99, flitered_image)
  

#   final_img = ifelse(final_img<0.05, 0, final_img)
#   grid.raster(final_img)
  
  #Actual cleaned image:
  test_image =paste("train_cleaned/", strsplit(image,"/")[[1]][2], sep="")
  
  cleaned_img = train_cleaned_images[[test_image]]
  
  img_rmse = rmse(final_img, cleaned_img)
  print ( img_rmse )
  
  avg_rmse = avg_rmse+img_rmse

}

avg_rmse= avg_rmse/144
print(avg_rmse)


#Filter with threshold model submission--------
#Run threshold on all test_images and submit
cleaned_test = list()

for (image in names(test_images)){
  print(image)
  
  #Try some basic thresholding on 1 image
  image1 = test_images[[image]]
  avg_pixel = mean(image1)
  # grid.raster(image1)
  dark_ratio = length(which(image1 < 0.35))/length(image1)
  
  #Start with a strong treshold to remove most/all nontext pixels
  thres_img = ifelse(image1>(avg_pixel-(4*dark_ratio)), 0, 1)
  
  #Smear pixel intensity vertically and laterally
  cols = rep(0,ncol(thres_img))
  
  smear_vertical_by =1
  for (x in 1:smear_vertical_by){
    thres_d_1 = rbind(thres_img, cols )[2:(nrow(thres_img)+1), ]
    thres_u_1 = rbind(cols,thres_img )[1:(nrow(thres_img)), ]
    thres_img = (thres_img+thres_d_1+thres_u_1)
  }
  
  rows = rep(0,nrow(thres_img))
  
  smear_laterally_by =1
  for (x in 1:smear_laterally_by){
    thres_r_1 = cbind(thres_img, rows )[,2:(ncol(thres_img)+1)]
    thres_l_1 = cbind(rows, thres_img)[,1:ncol(thres_img)]
    thres_img = (thres_img+thres_r_1+thres_l_1)
  }
  
  
  #Create new image matrix using smeared filter matrix
  filter_matrix = ifelse(thres_img ==0,-1,1)
  
  flitered_image = ifelse(filter_matrix ==-1,1, image1)
  
  #   grid.raster(flitered_image)
  
  #Calculate average non-white pixel intensity of flitered image
  white_space = (flitered_image==1)
  non_white_vec = flitered_image[!white_space]
  avg_non_zero = mean(non_white_vec)
  
  #Use a simple threshold
  final_img = ifelse(flitered_image>avg_non_zero+0.1, 0.99, flitered_image)
  
  cleaned_test[[image]] = final_img
}

solution_vector = c()

for (image in cleaned_test){
  solution_vector = c(solution_vector, as.vector(image))
}

submission2 = sample_sub
submission2$value = solution_vector
write.csv(submission2, "denoising_sub2_thresh.csv", row.names=FALSE)
#Submission score: 0.08036  First place! (out of 5...)
#First threshold model submission--------