rm(list=ls())
installIfAbsentAndLoad <- function(neededVector) {
  for(thispackage in neededVector) {
    if( ! require(thispackage, character.only = T) )
    { install.packages(thispackage)}
    require(thispackage, character.only = T)
  }
}
needed <- c('rattle', 'ada','gbm','verification','randomForest'
            ,"ggplot2", "ggthemes", "tree", "gdata", "readxl"
            , "rpart","rpart.plot","e1071","sparcl","cluster"
            , "partykit")  
library(caret)
installIfAbsentAndLoad(needed)


#######################################################
################### Data exporation & cleaning ########
#######################################################

data <- read.csv("C:\\Users\\user\\Desktop\\W&M BA Fall\\Course\\Spring Semester\\Machine Learning II\\HW\\Final Project\\UCI_Credit_Card.csv")
head(data)
summary(data)
str(data)

# Make Categorical variables into factors
factor_vars <- c('SEX','EDUCATION','MARRIAGE','default.payment.next.month')
data[factor_vars] <- lapply(data[factor_vars], function(x) as.factor(x))
str(data)


# take a look at the distribution of some variables and the target
plot(data$PAY_0)
plot(data$SEX)
plot(data$EDUCATION)
plot(data$MARRIAGE)
plot(data$default.payment.next.month)

# drop out those levels that reprent only a tini amount of the data
data$EDUCATION[data$EDUCATION== 4] = 0 
data$EDUCATION[data$EDUCATION== 5] = 0 
data$EDUCATION[data$EDUCATION== 6] = 0 
data$MARRIAGE[data$MARRIAGE== 3] = 0 

myData = myData[!myData$A > 4,]
data = data[!data$EDUCATION == 0,]
data = data[!data$MARRIAGE == 0,]

plot(data$EDUCATION)
plot(data$MARRIAGE)

data$paystate <- ""
data$genderNew <- ""
data$educationNew <- ""
data$maritalNew <- ""

for (i in 1:nrow(data)) {
  if ((data[i,7] + data[i,8] +data[i,9]+data[i,10] +data[i,11]+data[i,12]) <= 0){
    data[i,26] <- "YES"  
  }
  else {
    data[i,26] <- "NO"         
  }
}

for (i in 1:nrow(data)) {
  if (data[i,3] == 1) {
    data[i,27] <- "Male"  
  }
  else {
    data[i,27] <- "Female"         
  }
}

for (i in 1:nrow(data)) {
  if (data[i,4] == 1) {
    data[i,28] <- "Graduate"
  } else if (data [i,4] == 2) {
    data[i,28] <- "University" 
  } else if (data [i,4] == 3) {
    data[i,28] <- "High School" 
  } else {
    data[i,28] <- "Unknown" 
  }
}

for (i in 1:nrow(data)) {
  if(data[i,5] == 1) {
    data[i,29] <- "Married"
  } else if (data[i,5] == 2) {
    data[i,29] <- "Single"
  } else {
    data[i,29] <- "Other"
  }
}

factor_vars <- c('genderNew','educationNew','maritalNew','paystate')
data[factor_vars] <- lapply(data[factor_vars], function(x) as.factor(x))

data$AGE.group<-cut(data$AGE,c(20,30,40,50,60,70,80))

str(data)
# remove ID, education, gender, marital
data = data[,-c(1,3:5)]
str(data)

#######################################################
################### Variable Selection     ############
#######################################################
# split into train and test set 
n <- dim(data)[1]
set.seed(5082)
train = sample(n, .8 * n)
test = -train
data.train = data[train, ]
data.test = data[test, ]
default.test = data.test$default.payment.next.month


# run logistic regression to select important variables
set.seed(1)
glm.fit = glm(default.payment.next.month~., data = data.train, family = binomial)
glm.probs = predict(glm.fit, data.test, type = "response")
summary(glm.fit)
# filter out those variables that their Pr(>|z|) larger than 0.1
data = data[,-c(2,5,7,10:14,17,19,20,26)]
str(data)


# split into train and test set after selecting important variables
n <- dim(data)[1]
set.seed(5082)
train = sample(n, .8 * n)
test = -train
data.train = data[train, ]
data.test = data[test, ]
default.test = data.test$default.payment.next.month

#######################################################
################### Unsurpervised Learning ############
#######################################################

#######################################################
###################    Clustering #####################
#######################################################
str(data)
data.x=data[,-10]
str(data.x)
data.x[c(1,6:9)]=scale(data.x[c(1,6:9)])

#####################################################
####complete
#####################################################

hc.complete = hclust(dist(data.x), method = "complete") #complete linkage
#plot(hc.complete, main = "complete Linkage", xlab = "", sub = "", cex = 0.9)

k=20
hc.complete.cut=cutree(hc.complete, k=k)

data.hc.complete=data
data.hc.complete$complete.Group=NA

for(i in 1:nrow(data.hc.complete)){
  data.hc.complete$complete.Group[i]=hc.complete.cut[i]
}

for (i in 1:k){
  temp=mean(as.numeric(data.hc.complete$default.payment.next.month[data.hc.complete$complete.Group==i])-1)
  print(paste(i, sum(hc.complete.cut==i), temp))
}

means_complete=data.frame(matrix(nrow=k, ncol=24))
colnames(means_complete)=colnames(data[2:25])
rownames(means_complete)=c(1:k)

for (j in 1:k){
  for(i in 2:25){
    means_complete[j,i-1]=mean(as.numeric(as.character(unlist(data.hc.complete[data.hc.complete$complete.Group==j,i]))))
  }
  for(i in 7:12){
    means_complete[j,i-1]=mean(as.numeric(as.character(unlist(data.hc.complete[data.hc.complete$complete.Group==j,i])))==-1)
  }
}

#####################################################
####Single
#####################################################

hc.single = hclust(dist(data.x), method = "single")
#plot(hc.single, main = "Single Linkage", xlab = "", sub = "", cex = 0.9)

k=50
hc.single.cut=cutree(hc.single, k=k)

#for(i in 1:k){
#  print(paste(i, (sum(hc.single.cut==i))))
#}

data.hc.single=data
data.hc.single$single.Group=NA

for(i in 1:nrow(data.hc.single)){
  data.hc.single$single.Group[i]=hc.single.cut[i]
}

for (i in 1:k){
  temp=mean(as.numeric(data.hc.single$default.payment.next.month[data.hc.single$single.Group==i])-1)
  print(paste(i, sum(hc.single.cut==i), temp))
}

means_single=data.frame(matrix(nrow=k, ncol=24))
colnames(means_single)=colnames(data[2:25])
rownames(means_single)=c(1:k)

for (j in 1:k){
  for(i in c(2:6,13:25)){
    means_single[j,i-1]=mean(as.numeric(as.character(unlist(data.hc.single[data.hc.single$single.Group==j,i]))))
  }
  for(i in 7:12){
    means_single[j,i-1]=mean(as.numeric(as.character(unlist(data.hc.single[data.hc.single$single.Group==j,i])))==-1)
  }
}

#####################################################
####average
#####################################################

hc.average = hclust(dist(data.x), method = "average")
#plot(hc.average, main = Average Linkage", xlab = "", sub = "", cex = 0.9)

k=50
hc.average.cut=cutree(hc.average, k=k)

data.hc.average=data
data.hc.average$average.Group=NA

for(i in 1:nrow(data.hc.average)){
  data.hc.average$average.Group[i]=hc.average.cut[i]
}

for (i in 1:k){
  temp=mean(as.numeric(data.hc.average$default.payment.next.month[data.hc.average$average.Group==i])-1)
  print(paste(i, sum(hc.average.cut==i), temp))
}

means_average=data.frame(matrix(nrow=k, ncol=24))
colnames(means_average)=colnames(data[2:25])
rownames(means_average)=c(1:k)

for (j in 1:k){
  for(i in 2:25){
    means_average[j,i-1]=mean(as.numeric(as.character(unlist(data.hc.average[data.hc.average$average.Group==j,i]))))
  }
  for(i in 7:12){
    means_average[j,i-1]=mean(as.numeric(as.character(unlist(data.hc.average[data.hc.average$average.Group==j,i])))==-1)
  }
}
################################################################################




hc.complete.part1 = hclust(dist(data.x[,2:5]), method = "complete") #complete linkage
#plot(hc.complete, main = "complete Linkage", xlab = "", sub = "", cex = 0.9)

k=20
hc.complete.part1.cut=cutree(hc.complete.part1, k=k)

data.hc.complete.part1=data
data.hc.complete.part1$complete.Group=NA

for(i in 1:nrow(data.hc.complete.part1)){
  data.hc.complete.part1$complete.Group[i]=hc.complete.part1.cut[i]
}

for (i in 1:k){
  temp=mean(as.numeric(data.hc.complete.part1$default.payment.next.month[data.hc.complete.part1$complete.Group==i])-1)
  print(paste(i, sum(hc.complete.part1.cut==i), temp))
}

means_part1=data.frame(matrix(nrow=k, ncol=24))
colnames(means_part1)=colnames(data.3[2:25])
rownames(means_part1)=c(1:k)

for (j in 1:k){
  for(i in 2:25){
    means_part1[j,i-1]=mean(as.numeric(as.character(unlist(data.hc.complete.part1[data.hc.complete.part1$complete.Group==j,i]))))
  }
  for(i in 7:12){
    means_part1[j,i-1]=mean(as.numeric(as.character(unlist(data.hc.complete.part1[data.hc.complete.part1$complete.Group==j,i])))==-1)
  }
}

###############################################

hc.complete.part = hclust(dist(data.x[,1:5]), method = "complete") #complete linkage

k=9
hc.complete.part.cut=cutree(hc.complete.part, k=k)

data.hc.complete.part=data
data.hc.complete.part$complete.Group=NA

for(i in 1:nrow(data.hc.complete.part)){
  data.hc.complete.part$complete.Group[i]=hc.complete.part.cut[i]
}

for (i in 1:k){
  temp=mean(as.numeric(data.hc.complete.part$default.payment.next.month[data.hc.complete.part$complete.Group==i])-1)
  print(paste(i, sum(hc.complete.part.cut==i), temp))
}

means_part=data.frame(matrix(nrow=k, ncol=24))
colnames(means_part)=colnames(data.3[2:25])
rownames(means_part)=c(1:k)

for (j in 1:k){
  for(i in 2:25){
    means_part[j,i-1]=mean(as.numeric(as.character(unlist(data.hc.complete.part[data.hc.complete.part$complete.Group==j,i]))))
  }
  for(i in 7:12){
    means_part[j,i-1]=mean(as.numeric(as.character(unlist(data.hc.complete.part[data.hc.complete.part$complete.Group==j,i])))==-1)
  }
  means_part[j,4]=mean(as.numeric(as.character(unlist(data.hc.complete.part[data.hc.complete.part$complete.Group==j,4])))==1)
}

colors=data.frame(colors=c("black_left", "red","green","blue","teal","pink","yellow","grey","black_right"))
means_part=cbind(means_part, colors)
melted_means_part=melt(means_part, id.vars = "colors")
write.csv(melted_means_part, "melted_means_part.csv")

plot(data.hc.complete.part$AGE,data.hc.complete.part$LIMIT_BAL, col=data.hc.complete.part$complete.Group, lwd=3, ylab="Balance Limit", xlab="Age")

#1 is black
#2 is red
#3 is green
#4 is blue
#5 is teal
#6 is pink
#7 is yellow
#8 is grey
#9 is black again

for (j in 1:k){
  print(mean(as.numeric(as.character(unlist(data.hc.complete.part[data.hc.complete.part$complete.Group==j,4])))==1))
}


#######################################################
################### Surpervised Learning ##############
#######################################################


#######################################################
################### SVM ###############################
#######################################################

#Linear
svmfit <- svm(default.payment.next.month~., data=data.train, kernel="linear", cost=1)
svmpred=predict(svmfit, data.test)

(svm_1_table=table(actual=data.test$default.payment.next.month, predict=svmpred))

# plot the roc curve and calculate the AUC
svmpred1 = as.numeric(svmpred) - 1
default.test1 = as.numeric(default.test) - 1
roc.plot(default.test1,svmpred1, 
         main="ROC Curve for SVM (linear)")
aucc1 <- roc.area(default.test1,svmpred1)$A
paste("AUC for SVM (linear):",aucc1)

#Radial
svmfit_radial<- svm(default.payment.next.month~., data=data.train, kernel="radial", cost=1, gamma=1)

svmpred_radial=predict(svmfit_radial, data.test)

(svm_radial_1_table=table(actual=data.test$default.payment.next.month, predict=svmpred_radial))

# plot the roc curve and calculate the AUC
svmpred_radial1 = as.numeric(svmpred_radial) - 1
default.test1 = as.numeric(default.test) - 1
roc.plot(default.test1,svmpred_radial1, 
         main="ROC Curve for SVM (radial)")
aucc2 <- roc.area(default.test1,svmpred_radial1)$A
paste("AUC for SVM (radial):",aucc2)

#######################################################
################### Logistic Regression ###############
#######################################################

# run logistic regression model
set.seed(1)
glm.fit = glm(default.payment.next.month~., data = data.train, family = binomial)
glm.probs = predict(glm.fit, data.test, type = "response")
summary(glm.fit)
# observe the first few predict results and prepare to do ensemble
glm.probs[1:5]
success.glm = ifelse(glm.probs > .5 , '1' , '0')
# create confusion matrix and calculate statistical index
glm.conf = table(Actual = default.test , Predictions = ifelse(glm.probs > .5 , '1' , '0'))
glm.conf

# plot ROC curve and calculate AUC
success.glm1 = as.numeric(success.glm) 
default.test1 = as.numeric(default.test) - 1
roc.plot(default.test1,success.glm1
         , main="ROC Curve for Logistic Regression")
aucc3 <- roc.area(default.test1,success.glm1)$A
paste("AUC for Logistic Regression:",aucc3)

#####################################
######### Decision Tree #############
#####################################

# fit tree model by using rpart()
dtreeM <- rpart(formula = default.payment.next.month ~ .
                , data = data.train, method = "class"
                , control = rpart.control(cp = 0.001))
dtreeM$cptable

# use rattle to plot tree
fancyRpartPlot(dtreeM)
prp(dtreeM)
rpart.plot(dtreeM)

# predict
dtreeM_preds <- predict(dtreeM, newdata = data.test, type = "class")
# confusion matrix
dtreeM_cm <- table(default.test, dtreeM_preds
                   , dnn = c("actual", "preds"))
dtreeM_cm
# prediction accuracy
(accuracy <- sum(diag(dtreeM_cm)) / sum(dtreeM_cm))


# plot the roc curve and calculate the AUC
dtreeM_preds1 = as.numeric(dtreeM_preds) - 1
default.test1 = as.numeric(default.test) - 1
roc.plot(default.test1,dtreeM_preds1, 
         main="ROC Curve for Decision Tree")
aucc3 <- roc.area(default.test1,dtreeM_preds1)$A
paste("AUC for Decision Tree:",aucc3)


#####################################
######### Random Forest #############
#####################################
# without sampling 
set.seed(1)
randomForest_original = randomForest(default.payment.next.month~., data.train
                                  , mtry = 4, importance = T
                                  , ntree=200)

randomForest_original$confusion
# predict
randomForest_preds_original  <- predict(randomForest_original, newdata = data.test, type = "class")
# confusion matrix
randomForest_cm_original  <- table(default.test, randomForest_preds_original 
                          , dnn = c("actual", "preds"))
randomForest_cm_original
# prediction accuracy
(accuracy <- sum(diag(randomForest_cm_original )) / sum(randomForest_cm_original ))

round(100* table(default.test, randomForest_preds_original,
                 dnn=c("% Actual", "% Predicted"))/length(randomForest_preds_original),2)

# plot the roc curve and calculate the AUC
randomForest_preds_original1 = as.numeric(randomForest_preds_original) - 1
default.test1 = as.numeric(default.test) - 1
roc.plot(default.test1,randomForest_preds_original1, 
         main="ROC Curve for Random Forest without sampling")
aucc4 <- roc.area(default.test1,randomForest_preds_original1)$A
paste("AUC for Random Forest without sampling:",aucc4)






# with sampling 
# note that default of ntree is 500 
str(data.train)
set.seed(1)
randomForest_sample35.35  = randomForest(default.payment.next.month~., data.train
                         , mtry = 4, importance = T
                         , ntree=200, sampsize=c(35,35))

randomForest_sample35.35$confusion
# Error: caculate by OOB(Out Of Bag)
par(mar=c(5,4,4,0)) #No margin on the right side
plot(randomForest_sample35.35)
par(mar=c(5,0,4,2)) #No margin on the left side
legend("topright", colnames(randomForest$err.rate),col=1:3,cex=0.9,fill=1:3)

# predict
yhat.bag = predict(randomForest, newdata = data.test)
plot(yhat.bag, default.test)
# view and plot the importance of each variable
varImpPlot(randomForest)
# predict
randomForest_preds <- predict(randomForest, newdata = data.test, type = "class")
# confusion matrix
randomForest_cm <- table(default.test, randomForest_preds
                , dnn = c("actual", "preds"))
randomForest_cm
# prediction accuracy
(accuracy <- sum(diag(randomForest_cm)) / sum(randomForest_cm))

# plot the roc curve and calculate the AUC
randomForest_preds1 = as.numeric(randomForest_preds) - 1
default.test1 = as.numeric(default.test) - 1
roc.plot(default.test1,randomForest_preds1, 
         main="ROC Curve for Random Forest with sampling")

aucc5 <- roc.area(default.test1,randomForest_preds1)$A
paste("AUC for Random Forest with sampling:",aucc5)
#####################################
######### Comparison Table ##########
#####################################
Acc_models = list()
power_models = list()
prec_models = list()

Terror = list()
T1error = list()
T2error = list()

# Decision Tree
Acc_models['tree'] = round(sum(diag(dtreeM_cm)) / sum(dtreeM_cm),3)
power_models['tree'] = round(dtreeM_cm[2, 2] / sum(dtreeM_cm[2, ]),3)
prec_models['tree'] = round(dtreeM_cm["1", "1"] / sum(dtreeM_cm[, "1"]),3)

Terror['tree'] = round((dtreeM_cm[2] + dtreeM_cm[3]) / sum(dtreeM_cm),3)
T1error['tree'] = round(dtreeM_cm[1, 2] / sum(dtreeM_cm[1, ]),3)
T2error['tree'] = round(dtreeM_cm[2, 1] / sum(dtreeM_cm[2, ]),3)

# Random Forest without sampling
Acc_models['RForest'] = round(sum(diag(randomForest_cm_original)) / sum(randomForest_cm_original),3)
power_models['RForest'] = round(randomForest_cm_original[2, 2] / sum(randomForest_cm_original[2, ]),3)
prec_models['RForest'] = round(randomForest_cm_original["1", "1"] / sum(randomForest_cm_original[, "1"]),3)

Terror['RForest'] = round((randomForest_cm_original[2] + randomForest_cm_original[3]) / sum(randomForest_cm_original),3)
T1error['RForest'] = round(randomForest_cm_original[1, 2] / sum(randomForest_cm_original[1, ]),3)
T2error['RForest'] = round(randomForest_cm_original[2, 1] / sum(randomForest_cm_original[2, ]),3)

# SVM_linear
Acc_models['svm_linear'] = round(sum(diag(svm_1_table)) / sum(svm_1_table),3)
power_models['svm_linear'] = round(svm_1_table[2, 2] / sum(svm_1_table[2, ]),3)
prec_models['svm_linear'] = round(svm_1_table["1", "1"] / sum(svm_1_table[, "1"]),3)

Terror['svm_linear'] = round((svm_1_table[2] + svm_1_table[3]) / sum(svm_1_table),3)
T1error['svm_linear'] = round(svm_1_table[1, 2] / sum(svm_1_table[1, ]),3)
T2error['svm_linear'] = round(svm_1_table[2, 1] / sum(svm_1_table[2, ]),3)

# SVM_radial
Acc_models['svm_radial'] = round(sum(diag(svm_radial_1_table)) / sum(svm_radial_1_table),3)
power_models['svm_radial'] = round(svm_radial_1_table[2, 2] / sum(svm_radial_1_table[2, ]),3)
prec_models['svm_radial'] = round(svm_radial_1_table["1", "1"] / sum(svm_radial_1_table[, "1"]),3)

Terror['svm_radial'] = round((svm_radial_1_table[2] + svm_radial_1_table[3]) / sum(svm_radial_1_table),3)
T1error['svm_radial'] = round(svm_radial_1_table[1, 2] / sum(svm_radial_1_table[1, ]),3)
T2error['svm_radial'] = round(svm_radial_1_table[2, 1] / sum(svm_radial_1_table[2, ]),3)

# Logistic Regression
Acc_models['Logistic'] = round(sum(diag(glm.conf)) / sum(glm.conf),3)
power_models['Logistic'] = round(glm.conf[2, 2] / sum(glm.conf[2, ]),3)
prec_models['Logistic'] = round(glm.conf["1", "1"] / sum(glm.conf[, "1"]),3)

Terror['Logistic'] = round((glm.conf[2] + glm.conf[3]) / sum(glm.conf),3)
T1error['Logistic'] = round(glm.conf[1, 2] / sum(glm.conf[1, ]),3)
T2error['Logistic'] = round(glm.conf[2, 1] / sum(glm.conf[2, ]),3)

# Random Forest with sampling (35, 35)
Acc_models['RForest_sample'] = round(sum(diag(randomForest_cm)) / sum(randomForest_cm),3)
power_models['RForest_sample'] = round(randomForest_cm[2, 2] / sum(randomForest_cm[2, ]),3)
prec_models['RForest_sample'] = round(randomForest_cm["1", "1"] / sum(randomForest_cm[, "1"]),3)

Terror['RForest_sample'] = round((randomForest_cm[2] + randomForest_cm[3]) / sum(randomForest_cm),3)
T1error['RForest_sample'] = round(randomForest_cm[1, 2] / sum(randomForest_cm[1, ]),3)
T2error['RForest_sample'] = round(randomForest_cm[2, 1] / sum(randomForest_cm[2, ]),3)


AccTable = cbind(Acc_models, power_models, prec_models)
colnames(AccTable) = c('Accuracy', 'Power', 'Precision')
(AccTable = data.frame(AccTable))

errorTable = cbind(Terror, T1error, T2error)
colnames(errorTable) = c('Total.Error', 'Type1.Error', 'Type2.Error')
(errorTable = data.frame(errorTable))


