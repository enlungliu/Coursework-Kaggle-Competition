rm(list=ls())
set.seed(5072)
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
ks <- read.csv("CleanedDataML.csv", as.is = T)
str(ks)

# Removing the outliers - top 1% and bottom 1%
x <- quantile(x = ks$multiple, probs = c(0.99, 0.01), na.rm = T)
ks <- ks[!ks$multiple > x[1],]
ks<- ks[!ks$multiple < x[2],]
ks <- ks[which(ks$multiple != 0),]

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

ks$main_category =as.factor(ks$main_category)

ks$usd_pledged_real =as.numeric(ks$usd_pledged_real)

ks$backers =as.numeric(ks$backers)

ks$year =as.integer(ks$year)

##############################
####### Classification #######
##############################

# select variables for logistic regression, LDA and QDA
ks_logistic = dplyr::select(ks, main_category, country, backers, usd_goal_real
                            , timeperiod, status_num1)

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
indexTable

errorTable = cbind(Terror, T1error, T2error)
colnames(errorTable) = c('Total.Error', 'Type1.Error', 'Type2.Error')
errorTable = data.frame(errorTable)
errorTable
