## Starting with loading csv data in R
install.packages("tm") # Text Mining package
library(tm) # Text Mining package
library(caret)
library(corrplot) # package for correlation plot
install.packages("SnowballC")
library(SnowballC) # Snowball Stemmer package
library(elasticnet) # Package for Ridge and Lasso
library(earth) # Package for MARS
library(kernal) # Package for SVM


#**************************************************************************
#Step 1 
#Creating aggregated data: 
#Loading data set containing product title and search terms
title_HD <- read.csv("/Users/divya/OneDrive/Data Analytics/Spring 2016/OR 568/Project/Data/train.csv", 
                    header= TRUE,stringsAsFactors = FALSE, fileEncoding="latin1") 
#Loading data set containing product description and search terms
desc_HD <- read.csv("/Users/divya/OneDrive/Data Analytics/Spring 2016/OR 568/Project/Data/product_descriptions.csv", 
                 header= TRUE,stringsAsFactors = FALSE, fileEncoding="latin1")
#Merging the description and title data to enable better predictor creation
dataHD <- merge(title_HD,desc_HD, by.x = "product_uid", by.y = "product_uid", all.x = TRUE, all.y = FALSE)

#**************************************************************************

#Step 2

#Cleaning data and creating predictors
product_title <- dataHD$product_title
product_descrip <- dataHD$product_description
search_term <- dataHD$search_term

#Creating empty predictor matrix for 6 predictors
Np = 6
x_predictor = matrix(0*seq(1:Np*nrow(dataHD)),nrow(dataHD),Np)

#Using text mining for data cleaning 
#Cleaning Step 1: Convert all the text to lowercase for ease of pre-processing
#Cleaning Step 2: Break down the title, description and search text into vectors
#Cleaning Step 3: Remove all punctuations from the text as they won't help much in the analysis
#Cleaning Step 4: Removing blank spaces which resulted from converting text into vectors
#Cleaning Step 5: Apply stemming function to get the root word for removing redundant words & better computational speeds

word_stem = function(x) wordStem(x,language = "porter") # defining the stemming function

for(i in 1:nrow(dataHD)){
# Initially cleaning the data and pre-processing it
  # 1. For the product title
  prod_title <- tolower(product_title[i])#Step 1
  prod_title <- unlist(strsplit(prod_title, "\\s+"))#Step 2
  prod_title <- gsub("[[:punct:]]", "", prod_title)#Step 3
  prod_title = prod_title[prod_title != ""]#Step 4
  prod_title = as.character(lapply(prod_title, word_stem))#Step 5
  
  # 2. For the product description
  prod_desc <- tolower(product_descrip[i])#Step 1
  prod_desc <- unlist(strsplit(prod_desc, "\\s+"))#Step 2
  prod_desc <- gsub("[[:punct:]]", "", prod_desc)#Step 3
  prod_desc <- prod_desc[prod_desc != ""]#Step 4
  prod_desc <- as.character(lapply(prod_desc, word_stem))#Step 5
  
  # 3. For the search term
  search_vec <- tolower(search_term[i])#Step 1
  search_vec <- unlist(strsplit(search_vec, "\\s+"))#Step 2
  search_vec <- gsub("[[:punct:]]", "", search_vec)#Step 3
  search_vec = search_vec[search_vec != ""]#Step 4
  search_vec = as.character(lapply(search_vec, word_stem))#Step 5
  
#Now creating the predictors from the data
  Nsearch = length(search_vec) # length of the search vector
  num_match_words_title = 0 # match between search word and title
  num_match_words_desc = 0 # match between search word and description
  
  for (j in 1:Nsearch) # Looping over the number of search terms in the search vector
  {
    # PREDICTOR 1
    if (match(search_vec[j], prod_title, nomatch = 0))
    {
      num_match_words_title = num_match_words_title + 1
    }
    # PREDICTOR 4
    if (match(search_vec[j], prod_desc, nomatch = 0))
    {
      num_match_words_desc = num_match_words_desc + 1
    }
  } # end j loop
  
  # Create rest of the predictors
  x_predictor[i,1] <- num_match_words_title
  x_predictor[i,2] <- num_match_words_title / Nsearch # PREDICTOR 2
  x_predictor[i,3] <- Nsearch / length(prod_title) # PREDICTOR 3 (num_match_words_title)
  
  x_predictor[i,4] <- num_match_words_desc
  x_predictor[i,5] <- num_match_words_desc / Nsearch # PREDICTOR 5
  x_predictor[i,6] <- Nsearch / length(prod_desc) # PREDICTOR 6 (num_match_words_title)
}

x1 <- x_predictor[,1] # Number of matched terms in product title and search term
x2 <- x_predictor[,2] # Percentage of matched terms from the title in the search term 
x3 <- x_predictor[,3] # Proportion of the search term vector in the title
x4 <- x_predictor[,4] # Number of matched terms in product description and search term
x5 <- x_predictor[,5] # Percentage of matched terms from the description in the search term 
x6 <- x_predictor[,6] # Proportion of the search term vector in the description

threePred <- cbind(x1,x2,x3) # Considering only first 3 predictors
allPred <- cbind(x1,x2,x3,x4,x5,x6) # Considering all predictors

# Exploratory Data Analysis


allPredPP <- preProcess(allPred, method = c("center", "scale"))
allPred_cs <- predict(allPredPP,allPred)

par(mfrow=c(1,3))
hist(allPred[,1], xlab = "x1 values", ylab="Frequency", main="Predictor x1 in original form", col="cornsilk")
hist(allPred[,2], xlab = "x2 values", ylab="Frequency", main="Predictor x2 in original form", col="cornsilk")
hist(allPred[,3], xlab = "x3 values", ylab="Frequency", main="Predictor x3 in original form", col="cornsilk")

hist(allPred[,4], xlab = "x4 values", ylab="Frequency", main="Predictor x4 in original form", col="cornsilk")
hist(allPred[,5], xlab = "x5 values", ylab="Frequency", main="Predictor x5 in original form", col="cornsilk")
hist(allPred[,6], xlab = "x6 values", ylab="Frequency", main="Predictor x6 in original form", col="cornsilk")


hist(allPred_cs[,1], xlab = "x1 values", ylab="Frequency", main="x1 centered and scaled", col="cornsilk")
hist(allPred_cs[,2], xlab = "x2 values", ylab="Frequency", main="x2 centered and scaled", col="cornsilk")
hist(allPred_cs[,3], xlab = "x3 values", ylab="Frequency", main="x3 centered and scaled", col="cornsilk")
hist(allPred_cs[,4], xlab = "x4 values", ylab="Frequency", main="x4 centered and scaled", col="cornsilk")
hist(allPred_cs[,5], xlab = "x5 values", ylab="Frequency", main="x5 centered and scaled", col="cornsilk")
hist(allPred_cs[,6], xlab = "x6 values", ylab="Frequency", main="x6 centered and scaled", col="cornsilk")

#**************************************************************************
#Step 3
# Create training and test set with 3 predictors
indexes = sample(1:nrow(threePred), size=0.75*nrow(threePred))
train_threePred = threePred[indexes,]
dim(train_threePred)  # 55550 3 # Training data 75%
test_threePred = threePred[-indexes,]
dim(test_threePred)   # 18517 3 # Test data 25%

# Create training and test set with all predictors
indexes = sample(1:nrow(allPred), size=0.75*nrow(allPred))
train_allPred = allPred[indexes,]
dim(train_allPred)  # 55550 6 # Training data 75%
test_allPred = allPred[-indexes,]
dim(test_allPred)   # 18517 6 # Test data 25%

y_train <- dataHD[indexes,]$relevance # Training data class variable
y_test <- dataHD[-indexes,]$relevance # Test data class variable

three_data_HD <- cbind(test_threePred, y_test)
all_data_HD <- cbind(test_allPred, y_test)

# Exploratory Data Analysis

data_set <- cbind(allPred_cs,dataHD$relevance)
View(data_set)

HD_6Corr <- cor(all_data_HD)
corrplot(HD_6Corr, order = "hclust", tl.cex = .95) # Correlation plot for all predictors

#**************************************************************************
#Step 4: Applying various algorithms

# Using Linear regression
ctrl <- trainControl(method = "repeatedcv", repeats = 5)
#With 3 predictors
set.seed(100)
lmFitTrain1 <- train(x = train_threePred, y = y_train, method = "lm",preProcess = c("center","scale"), trControl = ctrl)
lmFitTrain1 # RMSE =  0.5335015
lmFitTest1 <- predict(lmFitTrain1,test_threePred)
rlmValues = data.frame(obs = y_test, pred = lmFitTest1) # Will test with test model
defaultSummary(rlmValues) # RMSE = 0.5352964918
#With all predictors
set.seed(100)
lmFitTrain2 <- train(x = train_allPred, y = y_train, method = "lm",preProcess = c("center","scale"), trControl = ctrl)
lmFitTrain2 # RMSE = 0.4964688
lmFitTest2 <- predict(lmFitTrain2,test_allPred)
rlmValues2 = data.frame(obs = y_test, pred = lmFitTest2) # Will test with test model
defaultSummary(rlmValues2) # RMSE = 0.4991941
varImp(lmFitTrain2)

# Using PLS
ctrl1 <- trainControl(method = "cv", number = 10)
#With 3 predictors
set.seed(100)
PLSFitTrain1 <- train(x = train_threePred, y = y_train, method = "pls",preProcess = c("center","scale"), trControl = ctrl1, tuneGrid = expand.grid(ncomp = 1:2))
PLSFitTrain1  #RMSE = 0.5335082
PLSFitTest1 <- predict(PLSFitTrain1,test_threePred)
plsValues = data.frame(obs = y_test, pred = PLSFitTest1) # Will test with test model
defaultSummary(plsValues) # RMSE = 0.5352533429
#With all predictors
set.seed(100)
PLSFitTrain2 <- train(x = train_allPred, y = y_train, method = "pls",preProcess = c("center","scale"), trControl = ctrl1, tuneGrid = expand.grid(ncomp = 1:2))
PLSFitTrain2  #RMSE =  0.4986622
PLSFitTest2 <- predict(PLSFitTrain2,test_allPred)
plsValues2 = data.frame(obs = y_test, pred = PLSFitTest2) # Will test with test model
defaultSummary(plsValues2) # RMSE = 0.5020126
varImp(PLSFitTrain2)

# Using Ridge Regression
ctrl1 <- trainControl(method = "cv", number = 10)
#With 3 predictors
set.seed(100)
RidgeTrain1 <- train(x = train_threePred, y = y_train, method = "ridge",preProcess = c("center","scale"),
                     tuneGrid = data.frame(.lambda = seq(0, .1, length = 5)),trControl = ctrl1)
RidgeTrain1  #RMSE = 0.5335116
RidgeTest1 <- predict(RidgeTrain1,test_threePred)
plsValues = data.frame(obs = y_test, pred = RidgeTest1) # Will test with test model
defaultSummary(plsValues) # RMSE = 0.5352847675
#With all predictors
set.seed(100)
RidgeTrain2 <- train(x = train_allPred, y = y_train, method = "ridge",preProcess = c("center","scale"), 
                     tuneGrid = data.frame(.lambda = seq(0, .1, length = 5)),trControl = ctrl1)
RidgeTrain2  #RMSE = 0.4964928 
RidgeTest2 <- predict(RidgeTrain2,test_allPred)
ridgeValues2 = data.frame(obs = y_test, pred = RidgeTest2) # Will test with test model
defaultSummary(ridgeValues2) # RMSE = 0.4991941
varImp(RidgeTrain2)

# Using Lasso Regression
ctrl1 <- trainControl(method = "cv", number = 10)
#With 3 predictors
set.seed(100)
LassoTrain1 <- train(x = train_threePred, y = y_train, method = "enet", preProcess = c("center","scale"),
                     tuneGrid = expand.grid(.lambda = c(0, 0.01, .1),.fraction = seq(.05, 1, length = 15)),trControl = ctrl1)
LassoTrain1  #RMSE = 0.5335043
LassoTest1 <- predict(LassoTrain1,test_threePred)
lassoValues = data.frame(obs = y_test, pred = LassoTest1) # Will test with test model
defaultSummary(lassoValues) # RMSE = 0.535298578
#With all predictors
LassoTrain2 <- train(x = train_allPred, y = y_train,method = "enet", preProcess = c("center","scale"),
                     tuneGrid = expand.grid(.lambda = c(0, 0.01, .1),.fraction = seq(.05, 1, length = 15)),trControl = ctrl1)
LassoTrain2  #RMSE = 0.4964634
LassoTest2 <- predict(LassoTrain2,test_allPred)
lassoValues2 = data.frame(obs = y_test, pred = LassoTest2) # Will test with test model
defaultSummary(lassoValues2) # RMSE = 0.4991941
varImp(LassoTrain2)

#####################################################################
# Using MARS
set.seed(100)
#With 3 predictors
marsTrain1 <- train(x = train_threePred, y = y_train, method = "earth",preProcess = c("center","scale"),
                    tuneGrid = expand.grid(.degree=1:2, .nprune=1:5),trControl = ctrl)
marsTrain1  #RMSE =  
marsTest1 <- predict(marsTrain1,test_threePred)
marsValues = data.frame(obs = y_test, pred = marsTest1) # Will test with test model
defaultSummary(marsValues) # RMSE = 
#With all predictors
marsTrain2 <- train(x = train_allPred, y = y_train,method = "earth", preProcess = c("center","scale"),
                     tuneGrid = expand.grid(.degree=1:2, .nprune=1:5),trControl = ctrl)
marsTrain2  #RMSE = 
marsTest2 <- predict(marsTrain2,test_allPred)
marsValues2 = data.frame(obs = y_test, pred = marsTest2) # Will test with test model
defaultSummary(marsValues2) # RMSE = 
varImp(marsTrain2)
#####################################################################
# Using SVM
set.seed(100)
#With 3 predictors
svmTrain1 <- train(x = train_threePred, y = y_train, method = "svmLinear", preProcess = c("center","scale"),
                    tuneLength=5, trControl = ctrl)
svmTrain1  #RMSE =  0.5338877
svmTest1 <- predict(svmTrain1,test_threePred)
svmValues = data.frame(obs = y_test, pred = svmTest1) # Will test with test model
defaultSummary(svmValues) # RMSE = 0.534230551
#With all predictors
svmTrain2 <- train(x = train_allPred, y = y_train,method = "svmLinear", preProcess = c("center","scale"),
                    tuneLength=5, trControl = ctrl)
svmTrain2  #RMSE = 
svmTest2 <- predict(svmTrain2,test_allPred)
svmValues2 = data.frame(obs = y_test, pred = svmTest2) # Will test with test model
defaultSummary(svmValues2) # RMSE = 
