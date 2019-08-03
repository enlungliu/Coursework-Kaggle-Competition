rm(list=ls())
set.seed(5072)
#######################################
####### loaded the origina data #######
#######################################

ks <- read.csv("C:/Users/user/Desktop/W&M BA Fall/Course/MachingLearning/HW/group project/kickstarter-projects/ks-projects-201801.csv")
str(ks)

#######################################
####### import function needed ########
#######################################
installIfAbsentAndLoad <- function(neededVector) {
  for(thispackage in neededVector) {
    if( ! require(thispackage, character.only = T) )
    { install.packages(thispackage)}
    require(thispackage, character.only = T)
  }
}

needed <- c("leaps", "ISLR", "glmnet", "glmulti", "pROC"
            ,"dplyr","class","stats","randomForest","verification"
            ,"tidyr","ggplot2","class",'FNN',"MASS","audio")  

installIfAbsentAndLoad(needed)

######################################################################
####### import data from Alteryx and clean the data in R #############
######################################################################
ks <- read.csv("C:/Users/user/Desktop/W&M BA Fall/Course/MachingLearning/HW/group project/kickstarter-projects/CleanedDataML.csv", as.is = T)
str(ks)

# Removing the outliers - top 1% and bottom 1%
xq <- quantile(ks$multiple, probs = c(0.01, 0.99), na.rm = T)
ks <- ks[ks$multiple > xq[1],]
ks <- ks[ks$multiple < xq[2],]
# make dependent variable binary variable 
status_num1 = ifelse(ks$status_num == 1,1,0)
ks = data.frame(ks, status_num1)
ks = ks[,-16]

# remove the meaningful variables
ks = ks[,-18:-20]

# delete values that are null
ks[is.na(ks)] <- 0

# Change the categorical variables into numeric type
# for running classifiers
ks$country = as.factor(ks$country)
ks$country = as.numeric(ks$country)

ks$main_category =as.factor(ks$main_category)
ks$main_category =as.numeric(ks$main_category)

ks$usd_pledged_real =as.numeric(ks$usd_pledged_real)

ks$backers =as.numeric(ks$backers)

ks$year =as.integer(ks$year)

###############################################
####### Select predictors for regression ######
###############################################

# use select() to choose the variables that are available to use
ks.sub = dplyr::select(ks, main_category
                       , backers , country,usd_goal_real, usd_pledged_real
                       , timeperiod, year, status_num1)


# select subset for regression 
regfit.full1 <- regsubsets(usd_pledged_real ~ ., ks.sub)
summary(regfit.full1)

regfit.full1 <- regsubsets(usd_pledged_real ~ ., 
                           data=ks.sub, 
                           nvmax=8)
reg.summary1 <- summary(regfit.full1)
reg.summary1$adjr2

# rsq & adjr2
par(mfrow=c(1, 2))
plot(reg.summary1$rsq, 
     xlab="Number of Variables", 
     ylab="RSS", 
     type="l")
plot(reg.summary1$adjr2, 
     xlab="Number of Variables", 
     ylab="Adjusted RSq", 
     type="l")
(max<-which.max(reg.summary1$adjr2))
points(max, reg.summary1$adjr2[max],  
       col="red", 
       cex=2, 
       pch=20)

# cp & bic
plot(reg.summary1$cp, 
     xlab="Number of Variables", 
     ylab="Cp", 
     type='l')
(min <- which.min(reg.summary1$cp))
points(min, reg.summary1$cp[min], 
       col="red", 
       cex=2, 
       pch=20)
(min <- which.min(reg.summary1$bic))
plot(reg.summary1$bic, 
     xlab="Number of Variables", 
     ylab="BIC", 
     type='l')
points(min, reg.summary1$bic[min], 
       col="red", 
       cex=2, 
       pch=20)
par(mfrow=c(1, 1))

# The regsubsets() function has a built-in plot() command
# which can be used to display the selected variables for
# the best model with a given number of predictors, ranked
# according to the BIC, Cp, adjusted R2, or AIC.

# The top row of each plot contains a black square for each 
# variable selected according to the optimal model 
# associated with that statistic. For instance, we see that 
# several models share a BIC close to -150. However, the 
# model with the lowest BIC is the six-variable model 
# (confirming we saw from our analysis above)  that contains
# only AtBat, Hits, Walks, CRBI, DivisionW, and PutOuts.
plot(regfit.full1, scale="r2")
plot(regfit.full1, scale="adjr2")
plot(regfit.full1, scale="Cp")
plot(regfit.full1, scale="bic")


###################################################
####### Select predictors for classification ######
###################################################

# use select() to choose the variables that are available to use
ks.sub1 = dplyr::select(ks, main_category
                        , backers , country, usd_goal_real
                        , timeperiod, year, status_num1)

# In the glmulti() function call, the following are the 
# important new parameters:
# 
# method parameter determines the type of analysis: 
# h is exhaustive, g is genetic algorithm
# 
# crit is the information criteria used to compare models 
# with different numbers of predictors. Possible values are 
# aic, bic, aicc 

#For classification models fit with MLE, 
# aic is the preferred method.
# 
# confsetsize is the number of candidate models to
# report on (including the best model)
# 
# fitfunction is the function to use when fitting the models
num.to.keep.glm = 7
glmulti.glm <- glmulti(ks.sub1$status_num1 ~ .,
                       data=ks.sub1,
                       method="g",
                       crit = "aic",       
                       confsetsize = num.to.keep.glm,
                       fitfunction = "glm",
                       family = binomial,
                       level = 1)                          
glmulti.summaryglm <- summary(glmulti.glm)

# Retrieve the coefficients of the best model
glmulti.summaryglm$bestmodel

# Display the vector of aic values for the winner and the 9
# runners-up (recall confsetsize models were retained)
glmulti.summaryglm$icvalues

# Plot these values
plot(glmulti.summaryglm$icvalues, 
     type='b', 
     xlab='Item',
     ylab='AIC',
     main='AIC for Candidate Set')

# Display the number of predictors in each of these
glmulti.glm@K - 1    #This value includes the intercept


##############################
####### Classification #######
##############################

# select variables for logistic regression, LDA and QDA
ks_logistic = dplyr::select(ks, main_category, country, backers, usd_goal_real
                     , timeperiod, status_num1)

# observe the histograms of the predictors we choose for classification 

par(mfrow = c(2,3))
hist(ks_logistic$main_category)
hist(ks_logistic$backers)
hist(ks_logistic$country)
hist(ks_logistic$usd_goal_real)
hist(ks_logistic$timeperiod)
par(mfrow = c(1,1))

# split into train set and test set
auto.train = sample(1:nrow(ks_logistic), nrow(ks_logistic)*.8, replace = F)
train = ks_logistic[auto.train,]
test = ks_logistic[-auto.train,]
status_num1.train = ks_logistic$status_num1[auto.train]
status_num1.test = ks_logistic$status_num1[-auto.train]

################################################
############ logistic Regression ###############
################################################

# run logistic regression model
glm.fit = glm(status_num1~., data = train, family = binomial)
glm.probs = predict(glm.fit, test, type = "response")

# observe the first few predict results and prepare to do ensemble
glm.probs[1:5]
success.glm = ifelse(glm.probs > .5 , '1' , '0')

# plot ROC curve and calculate AUC
roc.plot(status_num1.test,glm.probs, main="ROC Curve for Logistic Regression")
aucc <- roc.area(status_num1.test,glm.probs)$A
paste("AUC:",aucc)

# create confusion matrix and calculate statistical index
glm.conf = table(Actual = status_num1.test , Predictions = ifelse(glm.probs > .5 , '1' , '0'))
glm.conf

paste('The overall fraction of correct predictions:' , sum(diag(glm.conf)) / sum(glm.conf))
paste('The overall error rate:',(glm.conf[2] + glm.conf[3]) / sum(glm.conf))
paste('The Type I error rates:',glm.conf[1, 2] / sum(glm.conf[1, ]))
paste('The Type II error rates:',glm.conf[2, 1] / sum(glm.conf[2, ]))
paste('The Power of the model:',glm.conf[2, 2] / sum(glm.conf[2, ]))
paste('The Precision of the model:', glm.conf["1", "1"] / sum(glm.conf[, "1"]))

# create tables to compare index between different classifiers 
Acc_models = list()
power_models = list()
prec_models = list()
Acc_models['Logistic'] = sum(diag(glm.conf)) / sum(glm.conf)
power_models['Logistic'] = glm.conf[2, 2] / sum(glm.conf[2, ])
prec_models['Logistic'] = glm.conf["1", "1"] / sum(glm.conf[, "1"])

Terror = list()
T1error = list()
T2error = list()
Terror['Logistic'] = (glm.conf[2] + glm.conf[3]) / sum(glm.conf)
T1error['Logistic'] = glm.conf[1, 2] / sum(glm.conf[1, ])
T2error['Logistic'] = glm.conf[2, 1] / sum(glm.conf[2, ])

################################
############ LDA ###############
################################

# run lda model
lda.fit = lda(status_num1~., data = train)
lda.pred = predict(lda.fit, test)
summary(lda.fit)

# plot ROC curve and calculate AUC 
roc.plot(status_num1.test,lda.pred$posterior[,2], main="ROC Curve for LDA")
aucc1 <- roc.area(status_num1.test,lda.pred$posterior[,2])$A
paste("AUC:",aucc1)

# observe the first few predict results and prepare to do ensemble
head(lda.pred$posterior)
success.lda = ifelse(lda.pred$posterior[,2] > .5 , '1' , '0')

# create confusion matrix and calculate statistical index
lda.conf = table(Actual = status_num1.test, Predictions = lda.pred$class)
lda.conf

paste('The overall fraction of correct predictions:' , sum(diag(lda.conf)) / sum(lda.conf))
paste('The overall error rate:',(lda.conf[2] + lda.conf[3]) / sum(lda.conf))
paste('The Type I error rates:',lda.conf[1, 2] / sum(lda.conf[1, ]))
paste('The Type I error rates:',lda.conf[2, 1] / sum(lda.conf[2, ]))
paste('The Power of the model:',lda.conf[2, 2] / sum(lda.conf[2, ]))
paste('The Precision of the model:', lda.conf["1", "1"] / sum(lda.conf[, "1"]))

# create tables to compare index between different classifiers 
Acc_models['LDA'] = sum(diag(lda.conf)) / sum(lda.conf)
power_models['LDA'] = lda.conf[2, 2] / sum(lda.conf[2, ])
prec_models['LDA'] = lda.conf["1", "1"] / sum(lda.conf[, "1"])

Terror['LDA'] = (lda.conf[2] + lda.conf[3]) / sum(lda.conf)
T1error['LDA'] = lda.conf[1, 2] / sum(lda.conf[1, ])
T2error['LDA'] = lda.conf[2, 1] / sum(lda.conf[2, ])


################################
############# QDA ##############
################################

# run qda model
qda.fit = qda(status_num1~., data = train)
qda.pred = predict(qda.fit, test)
summary(qda.fit)

# plot ROC curve and calculate AUC 
roc.plot(status_num1.test,qda.pred$posterior[,2], main="ROC Curve for QDA")
aucc2 <- roc.area(status_num1.test,qda.pred$posterior[,2])$A
paste("AUC:",aucc2)

# observe the first few predict results and prepare to do ensemble
success.qda = ifelse(qda.pred$posterior[,1] > .5 , '1' , '0')

# create confusion matrix and calculate statistical index
qda.conf = table(Actual = status_num1.test, Predictions = qda.pred$class)
qda.conf

paste('The overall fraction of correct predictions:' , sum(diag(qda.conf)) / sum(qda.conf))
paste('The overall error rate:',(qda.conf[2] + qda.conf[3]) / sum(qda.conf))
paste('The Type I error rates:',qda.conf[1, 2] / sum(qda.conf[1, ]))
paste('The Type I error rates:',qda.conf[2, 1] / sum(qda.conf[2, ]))
paste('The Power of the model:',qda.conf[2, 2] / sum(qda.conf[2, ]))
paste('The Precision of the model:', qda.conf["1", "1"] / sum(qda.conf[, "1"]))

# create tables to compare index between different classifiers 
Acc_models['QDA'] = sum(diag(qda.conf)) / sum(qda.conf)
power_models['QDA'] = qda.conf[2, 2] / sum(qda.conf[2, ])
prec_models['QDA'] = qda.conf["1", "1"] / sum(qda.conf[, "1"])

Terror['QDA'] = (qda.conf[2] + qda.conf[3]) / sum(qda.conf)
T1error['QDA'] = qda.conf[1, 2] / sum(qda.conf[1, ])
T2error['QDA'] = qda.conf[2, 1] / sum(qda.conf[2, ])


##################################
############# KNN ################
##################################

# use select() to keeps only the variables we mention
ks.sub.class = dplyr::select(ks, main_category
                             , backers , country,usd_goal_real
                             ,timeperiod, status_num1)

# ks = select(ks, state, main_category, backers, country, usd_pledged_real
#            , usd_goal_real, timeperiod, year, multiple)
n <- nrow(ks.sub.class)
p <- ncol(ks.sub.class)-1
# scale and center the numeric and integer predictors
ks.sub.class[1:p] <- scale(ks.sub.class[1:p]) 
# Display the first 6 rows of your data frame
head(ks.sub.class)
str(ks.sub.class)

# Create training, validate and test data frames 
# from the scaled data frame.
# Use a 75/15/10 split.
trainprop <- 0.75
validateprop <- 0.15
train = sample(n, trainprop * n)
validate = sample(setdiff(1:n, train), validateprop * n) 
# create a vector of the integers not in either training or
# validate
test <- setdiff(setdiff(1:n, train), validate)
# Create the data frames using the indices created in the
# three vectors above
trainset <- ks.sub.class[train,]
validateset <- ks.sub.class[validate,]
testset <- ks.sub.class[test,]
# Display the first row of each of your three data frames
head(trainset,1)
head(validateset,1)
head(testset,1)

### Create the following 6 data frames ###
y.name='status_num1'
# set x axis variables (predictors)
train.x <- trainset[setdiff(names(trainset), y.name)]
validate.x <- validateset[setdiff(names(validateset), y.name)]
test.x <- testset[setdiff(names(testset), y.name)]
# set y axis variable (the one we are predicting)
train.y <- trainset$status_num1
validate.y <- validateset$status_num1
test.y <- testset$status_num1

biggestk = 15
validate.errors <- rep(0, biggestk %/% 2 + 1)
train.errors <- rep(0, biggestk %/% 2 + 1)
kset <- seq(1, biggestk, by=2) 

time.knn = system.time(for(k in kset) {
  knn.pred <- knn(train.x,validate.x,train.y,k = k)
  validate.errors[k %/% 2 + 1] <- mean(validate.y != knn.pred)
  knn.pred <- knn(train.x,train.x,train.y,k = k)
  train.errors[k %/% 2 + 1] <- mean(train.y != knn.pred)    
})

time.knn

plot(kset, validate.errors, 
     xlim=c(biggestk, 1), 
     ylim=c(0,max(c(validate.errors, train.errors))),      
     type='n',  
     xlab='Increasing Flexibility (Decreasing k)', 
     ylab='Error Rates', 
     main='Error Rates as a Function of \n Flexibility for KNN Classification')

lines(seq(biggestk, 1, by=-2), 
      validate.errors[order(length(validate.errors):1)], 
      type='b',             
      col=2,                
      pch=16)

lines(seq(biggestk, 1, by=-2), 
      train.errors[order(length(train.errors):1)], 
      type='b', 
      col=1,                 
      pch=16)

legend("bottomleft", legend = c("Validation Error Rate", "Train Error Rate"), 
       col=c(2, 1),
       pch=16,
       lty = c(1,1))

# Print the k 
# and associated error rate that produced the lowest training error rate.
# Do the same for validate error rate
print(paste("My 'best' training error rate occurred with k =", kset[which.min(train.errors)], "and produced a training error rate of", train.errors[which.min(train.errors)]))
print(paste("My 'best' validate error rate occurred with k =", kset[which.min(validate.errors)], "and produced a validate error rate of", validate.errors[which.min(validate.errors)]))

# Finally, predict loan.repaid for the test set using the optimal value of k 
# that you found for the validate set, and compute (and print as above) 
# the associated error rate.

knn.pred <- knn(train.x,test.x,train.y,k = 5)
test.errors <- mean(test.y != knn.pred)
print(paste("My 'best' test error rate occurred with k =", 5, "and produced a a test error rate of", test.errors))

# plot the roc curve and calculate the AUC
knn.pred1 = as.numeric(knn.pred) - 1
roc.plot(test.y, knn.pred1, main="ROC Curve for KNN")
aucc4 <- roc.area(test.y,knn.pred1)$A
paste("AUC:",aucc4)

# observe the first few predict results and prepare to do ensemble
success.knn = knn.pred

# Confusion Matrix of test set
knn.conf = table(Actual = test.y, Predictions = knn.pred)
knn.conf

paste('The overall fraction of correct predictions:' , sum(diag(knn.conf)) / sum(knn.conf))
paste('The overall error rate:',(knn.conf[2] + knn.conf[3]) / sum(knn.conf))
paste('The Type I error rates:',knn.conf[1, 2] / sum(knn.conf[1, ]))
paste('The Type I error rates:',knn.conf[2, 1] / sum(knn.conf[2, ]))
paste('The Power of the model:',knn.conf[2, 2] / sum(knn.conf[2, ]))
paste('The Precision of the model:', knn.conf["1", "1"] / sum(knn.conf[, "1"]))

# create tables to compare index between different classifiers 
Acc_models['KNN'] = sum(diag(knn.conf)) / sum(knn.conf)
power_models['KNN'] = knn.conf[2, 2] / sum(knn.conf[2, ])
prec_models['KNN'] = knn.conf["1", "1"] / sum(knn.conf[, "1"])

Terror['KNN'] = (knn.conf[2] + knn.conf[3]) / sum(knn.conf)
T1error['KNN'] = knn.conf[1, 2] / sum(knn.conf[1, ])
T2error['KNN'] = knn.conf[2, 1] / sum(knn.conf[2, ])


######################################
####### Random Forest Classifier######
######################################

# select 
ks_randomForest = dplyr::select(ks, backers, usd_goal_real, main_category, country
                                ,timeperiod, status_num1)

ks_randomForest$status_num1 = as.factor(ks_randomForest$status_num1)
status_num1 = as.factor(status_num1)
# split into train set and test set
auto.train.forest = sample(1:nrow(ks_randomForest), nrow(ks_randomForest)*.8, replace = F)
train.forest = ks_randomForest[auto.train.forest,]
test.forest  = ks_randomForest[-auto.train.forest,]
status_num1.train.forest  = ks_randomForest$status_num1[auto.train.forest]
status_num1.test.forest  = ks_randomForest$status_num1[-auto.train.forest]

# run randomforest model for classification
rf.class = randomForest(status_num1 ~ . , train.forest[sample(dim(train.forest)[1] , 50000) , ] , ntree = 45)
plot(rf.class)
rf.preds = predict(rf.class , test.forest)

# plot the roc curve and calculate the AUC
rf.preds1 = as.numeric(rf.preds) - 1
status_num1.test.forest1 = as.numeric(status_num1.test.forest) - 1
roc.plot(status_num1.test.forest1,rf.preds1, 
         main="ROC Curve for Random Forest")
aucc3 <- roc.area(status_num1.test.forest1,rf.preds1)$A
paste("AUC:",aucc3)

# observe the first few predict results and prepare to do ensemble
rf.preds[1:5]
success.rf = rf.preds

# Confusion Matrix of test set
rf.class.conf = table(Actual = test.forest$status_num1, Predictions = rf.preds)
rf.class.conf
paste('The overall fraction of correct predictions:' , sum(diag(rf.class.conf)) / sum(rf.class.conf))
paste('The overall error rate:',(rf.class.conf[2] + rf.class.conf[3]) / sum(rf.class.conf))
paste('The Type I error rates:',rf.class.conf[1, 2] / sum(rf.class.conf[1, ]))
paste('The Type I error rates:',rf.class.conf[2, 1] / sum(rf.class.conf[2, ]))
paste('The Power of the model:',rf.class.conf[2, 2] / sum(rf.class.conf[2, ]))
paste('The Precision of the model:', rf.class.conf["1", "1"] / sum(rf.class.conf[, "1"]))

# create tables to compare index between different classifiers 
Acc_models['Random Forest'] = sum(diag(rf.class.conf)) / sum(rf.class.conf)
power_models['Random Forest'] = rf.class.conf[2, 2] / sum(rf.class.conf[2, ])
prec_models['Random Forest'] = rf.class.conf["1", "1"] / sum(rf.class.conf[, "1"])

Terror['Random Forest'] = (rf.class.conf[2] + rf.class.conf[3]) / sum(rf.class.conf)
T1error['Random Forest'] = rf.class.conf[1, 2] / sum(rf.class.conf[1, ])
T2error['Random Forest'] = rf.class.conf[2, 1] / sum(rf.class.conf[2, ])

#################################################
############### Comparison Table ################
#################################################

# print the comparison table of Accuracy and Error Rates
indexTable = cbind(Acc_models, power_models, prec_models)
colnames(indexTable) = c('Accuracy', 'Power', 'Precision')
indexTable = data.frame(indexTable)

errorTable = cbind(Terror, T1error, T2error)
colnames(errorTable) = c('Total.Error', 'Type1.Error', 'Type2.Error')
errorTable = data.frame(errorTable)

#################################################
######## Prediction of classification ###########
#################################################

# Do the prediction of classification with data from classmates (msbapreds.csv)

msba.pred <- read.csv("C://Users//user//Desktop//W&M BA Fall//Course//MachingLearning//HW//group project//kickstarter-projects//msbapreds.csv", as.is = T)
msba.pred = dplyr::select(msba.pred, main_category
                          , backers, country, timeperiod
                          , usd_goal_real)

rf.preds.msba = predict(rf.class , msba.pred)
success.rf.msba = rf.preds.msba
success.rf.msba

# import music wav. 
xmas.sorrow <-load.wave("C:\\Users\\user\\Desktop\\Last_Christmas_long.wav")
xmas.happy <-load.wave("C:\\Users\\user\\Desktop\\Jingle_Bell_Rock_long.wav")

# create xmas function to recognize the prediction 
# and play related music accordingly
xmas <- function(success.or.not) {
  if (success.or.not == 1){
    return(play(xmas.happy))
  }else{
    return(play(xmas.sorrow))
  }
}

# recognize the prediction
xmas(success.rf.msba[1])

##############################
############Ensemble##########
##############################

success.glm = as.numeric(success.glm)
success.lda = as.numeric(success.lda)
success.qda = as.numeric(success.qda)
success.knn = as.character(success.knn)
success.knn = as.numeric(success.knn)
success.rf  = as.character(success.rf)
success.rf  = as.numeric(success.rf)


# select only the good classifiers, which are logistic, knn, randomforest
en= list()
en= cbind(success.glm, success.lda, success.rf)
(mean(en[preds,])) > 0.34

enlist= c()

for (preds in 1:nrow(en)){
  if (mean(en[preds,]) > 0.34){
    enlist[preds] = 1
  } else {
    enlist[preds] = 0
  }
}

as.data.frame(table(enlist))

str(enlist)

rf.preds1 = as.numeric(rf.preds) - 1
status_num1.test.forest1 = as.numeric(status_num1.test.forest) - 1

roc.plot(status_num1.test,enlist1, 
         main="ROC Curve for ensemble")
aucc5 <- roc.area(status_num1.test,enlist1)$A
paste("AUC:",aucc5)

##############################
######### Regression #########
##############################

# Split into successful campaigns and unsuccessful campaigns
ks1 <- split(ks, ks$status)
ks.success <- ks1$'1'
ks.nosuccess <- ks1$'0'
str(ks.success)
str(ks.nosuccess)

#################################################
#######Split successful campaigns by category####
#################################################
ks.success <-split(ks.success, ks.success$main_category)

## Successful Technology Campaigns
success.tech <- ks.success$`31`
success.tech <- dplyr::select(success.tech, backers, usd_goal_real
                              , timeperiod, usd_pledged_real)
lm.tech<- lm(usd_pledged_real ~ success.tech$backers
             +success.tech$usd_goal_real
             +success.tech$timeperiod
             , data = success.tech)
summary(lm.tech)
# calculate confidence interval and prediction interval
df.tech = data.frame(success.tech$backers,success.tech$usd_goal_real,success.tech$timeperiod)

predict(lm.tech, df.tech, interval="confidence")    

mean.predictAmountofMoeny.tech = 
  mean(predict(lm.tech, df.tech, interval="prediction"))

pred.table = list()
pred.table["Technology"] = mean.predictAmountofMoeny.tech

## Successful Design Campaigns
success.design <-ks.success$`10`
success.design <- dplyr::select(success.design, backers, usd_goal_real
                                , timeperiod, usd_pledged_real)
lm.design<- lm(usd_pledged_real ~ success.design$backers
               +success.design$usd_goal_real
               +success.design$timeperiod
               , data = success.design)
summary(lm.design)

# calculate confidence interval and prediction interval
df.design = data.frame(success.design$backers,success.design$usd_goal_real,success.design$timeperiod)

predict(lm.design, df.design, interval="confidence")    

mean.predictAmountofMoeny.design = 
  mean(predict(lm.design, df.design, interval="prediction"))

pred.table["Design"] = mean.predictAmountofMoeny.design

## Successful Art Campaigns
success.art <- ks.success$`3`
success.art <- dplyr::select(success.art, backers, usd_goal_real
                             , timeperiod, usd_pledged_real)
lm.art<- lm(usd_pledged_real ~ success.art$backers
            +success.art$usd_goal_real
            +success.art$timeperiod
            , data = success.art)
summary(lm.art)
# calculate confidence interval and prediction interval
df.art = data.frame(success.art$backers
                    ,success.art$usd_goal_real
                    ,success.art$timeperiod)

predict(lm.art, df.art, interval="confidence")    

mean.predictAmountofMoeny.art = 
  mean(predict(lm.art, df.art, interval="prediction"))

pred.table["Art"] = mean.predictAmountofMoeny.art

## Successful Comic Campaigns
success.comics <- ks.success$`7`
success.comics <- dplyr::select(success.comics, backers, usd_goal_real
                                , timeperiod, usd_pledged_real)
lm.comics<- lm(usd_pledged_real ~ success.comics$backers
               +success.comics$usd_goal_real
               +success.comics$timeperiod
               , data = success.comics)
summary(lm.comics)
# calculate confidence interval and prediction interval
df.comics = data.frame(success.comics$backers
                       ,success.comics$usd_goal_real
                       ,success.comics$timeperiod)

predict(lm.comics, df.comics, interval="confidence")    

mean.predictAmountofMoeny.comics = 
  mean(predict(lm.comics, df.comics, interval="prediction"))

pred.table["comics"] = mean.predictAmountofMoeny.comics

## Successful Film & Video Campaigns
success.film <- ks.success$`13`
success.film <- dplyr::select(success.film, backers, usd_goal_real
                              , timeperiod, usd_pledged_real)
lm.film<- lm(usd_pledged_real ~ success.film$backers
             +success.film$usd_goal_real
             +success.film$timeperiod
             , data = success.film)
summary(lm.film)
# calculate confidence interval and prediction interval
df.film = data.frame(success.film$backers
                     ,success.film$usd_goal_real
                     ,success.film$timeperiod)

predict(lm.film, df.film, interval="confidence")    

mean.predictAmountofMoeny.film = 
  mean(predict(lm.film, df.film, interval="prediction"))

pred.table["film"] = mean.predictAmountofMoeny.film


## Successful Publishing Campaigns
success.publishing <- ks.success$`25`
success.publishing <- dplyr::select(success.publishing, backers, usd_goal_real
                                    , timeperiod, usd_pledged_real)
lm.publishing<- lm(usd_pledged_real ~ success.publishing$backers
                   +success.publishing$usd_goal_real
                   +success.publishing$timeperiod
                   , data = success.publishing)
summary(lm.publishing)
# calculate confidence interval and prediction interval
df.publishing = data.frame(success.publishing$backers
                           ,success.publishing$usd_goal_real
                           ,success.publishing$timeperiod)

predict(lm.publishing, df.publishing, interval="confidence")    

mean.predictAmountofMoeny.publishing = 
  mean(predict(lm.publishing, df.publishing, interval="prediction"))

pred.table["Publishing"] = mean.predictAmountofMoeny.publishing

## Successful Music Campaigns
success.music <- ks.success$`19`
success.music <- dplyr::select(success.music, backers, usd_goal_real
                               , timeperiod, usd_pledged_real)
lm.music<- lm(usd_pledged_real ~ success.music$backers
              +success.music$usd_goal_real
              +success.music$timeperiod
              , data = success.music)
summary(lm.music)
# calculate confidence interval and prediction interval
df.music = data.frame(success.music$backers
                      ,success.music$usd_goal_real
                      ,success.music$timeperiod)

predict(lm.music, df.music, interval="confidence")    

mean.predictAmountofMoeny.music = 
  mean(predict(lm.music, df.music, interval="prediction"))
pred.table["music"] = mean.predictAmountofMoeny.music

## Successful Food Campaigns
success.food <- ks.success$`14`
success.food <- dplyr::select(success.food, backers, usd_goal_real
                              , timeperiod, usd_pledged_real)
lm.food<- lm(usd_pledged_real ~ success.food$backers
             +success.food$usd_goal_real
             +success.food$timeperiod
             , data = success.food)
summary(lm.food)
# calculate confidence interval and prediction interval
df.food = data.frame(success.food$backers
                     ,success.food$usd_goal_real
                     ,success.food$timeperiod)

predict(lm.food, df.food, interval="confidence")    

mean.predictAmountofMoeny.food = 
  mean(predict(lm.food, df.food, interval="prediction"))
pred.table["Food"] = mean.predictAmountofMoeny.food

## Successful Crafts Campaigns
success.crafts <- ks.success$`8`
success.crafts <- dplyr::select(success.crafts, backers, usd_goal_real
                                , timeperiod, usd_pledged_real)
lm.crafts<- lm(usd_pledged_real ~ success.crafts$backers
               +success.crafts$usd_goal_real
               +success.crafts$timeperiod
               , data = success.crafts)
summary(lm.crafts)
# calculate confidence interval and prediction interval
df.crafts = data.frame(success.crafts$backers
                       ,success.crafts$usd_goal_real
                       ,success.crafts$timeperiod)

predict(lm.crafts, df.crafts, interval="confidence")    

mean.predictAmountofMoeny.crafts = 
  mean(predict(lm.crafts, df.crafts, interval="prediction"))
pred.table["Crafts"] = mean.predictAmountofMoeny.crafts

## Successful Games Campaigns
success.games <- ks.success$`15`
success.games <- dplyr::select(success.games, backers, usd_goal_real
                               , timeperiod, usd_pledged_real)
lm.games <- lm(success.games$usd_pledged_real ~ ., data = success.games)
summary(lm.games)
lm.games<- lm(usd_pledged_real ~ success.games$backers
              +success.games$usd_goal_real
              +success.games$timeperiod
              , data = success.games)
summary(lm.games)
# calculate confidence interval and prediction interval
df.games = data.frame(success.games$backers
                      ,success.games$usd_goal_real
                      ,success.games$timeperiod)

predict(lm.games, df.games, interval="confidence")    

mean.predictAmountofMoeny.games = 
  mean(predict(lm.games, df.games, interval="prediction"))
pred.table["Games"] = mean.predictAmountofMoeny.games

## Successful Fashion Campaigns
success.fashion <- ks.success$`12`
success.fashion <- dplyr::select(success.fashion, backers, usd_goal_real
                                 , timeperiod, usd_pledged_real)
lm.fashion<- lm(usd_pledged_real ~ success.fashion$backers
                +success.fashion$usd_goal_real
                +success.fashion$timeperiod
                , data = success.fashion)
summary(lm.fashion)
# calculate confidence interval and prediction interval
df.fashion = data.frame(success.fashion$backers
                        ,success.fashion$usd_goal_real
                        ,success.fashion$timeperiod)

predict(lm.fashion, df.fashion, interval="confidence")    

mean.predictAmountofMoeny.fashion = 
  mean(predict(lm.fashion, df.fashion, interval="prediction"))
pred.table["Fashion"] = mean.predictAmountofMoeny.fashion

## Successful Theater Campaigns
success.theater <- ks.success$`32`
success.theater <- dplyr::select(success.theater, backers, usd_goal_real
                                 , timeperiod, usd_pledged_real)
lm.theater<- lm(usd_pledged_real ~ success.theater$backers
                +success.theater$usd_goal_real
                +success.theater$timeperiod
                , data = success.theater)
summary(lm.theater)
# calculate confidence interval and prediction interval
df.theater = data.frame(success.theater$backers
                        ,success.theater$usd_goal_real
                        ,success.theater$timeperiod)

predict(lm.theater, df.theater, interval="confidence")    

mean.predictAmountofMoeny.theater = 
  mean(predict(lm.theater, df.theater, interval="prediction"))
pred.table["Theater"] = mean.predictAmountofMoeny.theater

## Successful Photography Campaigns
success.photography <- ks.success$`22`
success.photography <- dplyr::select(success.photography, backers, usd_goal_real
                                     , timeperiod, usd_pledged_real)
lm.photography<- lm(usd_pledged_real ~ success.photography$backers
                    +success.photography$usd_goal_real
                    +success.photography$timeperiod
                    , data = success.film)
summary(lm.photography)
# calculate confidence interval and prediction interval
df.photography = data.frame(success.photography$backers
                            ,success.photography$usd_goal_real
                            ,success.photography$timeperiod)

predict(lm.photography, df.photography, interval="confidence")    

mean.predictAmountofMoeny.photography = 
  mean(predict(lm.photography, df.photography, interval="prediction"))
pred.table["Photography"] = mean.predictAmountofMoeny.photography

## Successful Journalism Campaigns
success.journalism <- ks.success$`18`
success.journalism <- dplyr::select(success.journalism, backers, usd_goal_real
                                    , timeperiod, usd_pledged_real)
lm.journalism<- lm(usd_pledged_real ~ success.journalism$backers
                   +success.journalism$usd_goal_real
                   +success.journalism$timeperiod
                   , data = success.journalism)
summary(lm.journalism)
# calculate confidence interval and prediction interval
df.journalism = data.frame(success.journalism$backers
                           ,success.journalism$usd_goal_real
                           ,success.journalism$timeperiod)

predict(lm.journalism, df.journalism, interval="confidence")    

mean.predictAmountofMoeny.journalism = 
  mean(predict(lm.journalism, df.journalism, interval="prediction"))
pred.table["Journalism"] = mean.predictAmountofMoeny.journalism

## Successful Dance Campaigns
success.dance <- ks.success$`9`
success.dance <- dplyr::select(success.dance, backers, usd_goal_real
                               , timeperiod, usd_pledged_real)
lm.dance<- lm(usd_pledged_real ~ success.dance$backers
              +success.dance$usd_goal_real
              +success.dance$timeperiod
              , data = success.dance)
summary(lm.dance)
# calculate confidence interval and prediction interval
df.dance = data.frame(success.dance$backers
                      ,success.dance$usd_goal_real
                      ,success.dance$timeperiod)

predict(lm.dance, df.dance, interval="confidence")    

mean.predictAmountofMoeny.dance = 
  mean(predict(lm.dance, df.dance, interval="prediction"))

pred.table["Dance"] = mean.predictAmountofMoeny.dance

# print the prediction table of regression
pred.table = as.matrix(pred.table, bycol = T)
colnames(pred.table) = "Average amount of money would be raised"
pred.table = data.matrix(pred.table)
pred.table = sort(pred.table[,1])
pred.table = data.frame(pred.table)