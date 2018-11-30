library(readr)
library(MASS)
library(BBmisc)

#Read data
results <- read_csv("~/Documents/Olympic-Results-Regression/olympic-track-field-results/results.csv")

#Change categorical varibales to factors
results$Gender<-as.factor(results$Gender)
results$Event<-as.factor(results$Event)
results$Location<-as.factor(results$Location)
results$Medal<-as.factor(results$Medal)
results$Nationality<-as.factor(results$Nationality)

#Create list of the event names we are working with
focus<-c('100M Men', '100M Women', '200M Men', '200M Women', '1500M Men', '1500M Women', 'Long Jump Men', 'Long Jump Women', 'Shot Put Men', 'Shot Put Women' )

#Create new data frames for each event seperately 
for (event in focus){
  name<- paste("data_",event, sep="")
  assign(name,results[results$Event==event,])
}

#Change the results of each new dataset to numeric
`data_100M Men`$Result<-as.numeric(`data_100M Men`$Result)
`data_100M Women`$Result<-as.numeric(`data_100M Women`$Result)
`data_200M Men`$Result<-as.numeric(`data_200M Men`$Result)
`data_200M Women`$Result<-as.numeric(`data_200M Women`$Result)
`data_Long Jump Men`$Result<-as.numeric(`data_Long Jump Men`$Result)
`data_Long Jump Women`$Result<-as.numeric(`data_Long Jump Women`$Result)
`data_Shot Put Men`$Result<-as.numeric(`data_Shot Put Men`$Result)
`data_Shot Put Women`$Result<-as.numeric(`data_Shot Put Women`$Result)

#1500M is in the format of xx:xx, so need to change into seconds and make numeric
for (i in 1:length(`data_1500M Men`$Result)){
  `data_1500M Men`$Result[i]<-as.numeric(strsplit(`data_1500M Men`$Result,':')[[i]][1])*60+as.numeric(strsplit(`data_1500M Men`$Result,':')[[i]][2])
}
for (i in 1:length(`data_1500M Women`$Result)){
  `data_1500M Women`$Result[i]<-as.numeric(strsplit(`data_1500M Women`$Result,':')[[i]][1])*60+as.numeric(strsplit(`data_1500M Women`$Result,':')[[i]][2])
}
`data_1500M Men`$Result<-as.numeric(`data_1500M Men`$Result)
`data_1500M Women`$Result<-as.numeric(`data_1500M Women`$Result)

#Create a list of the individual event data frames
WholeData<-list(`data_100M Men`, `data_100M Women`, `data_200M Men`, `data_200M Women`, `data_1500M Men`, `data_1500M Women`, `data_Long Jump Men`, `data_Long Jump Women`, `data_Shot Put Men`, `data_Shot Put Women`)

#linear regression
#Start with simple linear regression with model Y=b1X+b0
i<-1
for (dataset in WholeData){
  name<- paste("model",i, sep="")
  model<-lm(Result~Year, data=dataset)
  assign(name,model)
  i=i+1
}

#List of the models in order of focus
modelList1<-list(model1,model2,model3,model4,model5,model6,model7,model8,model9,model10)

#Plot fitted line with data points to see fit
i<-1
for (dataset in WholeData){
  plot(dataset$Year,dataset$Result,main=focus[i], col=dataset$Medal)
  lines(dataset$Year,modelList1[[i]]$fitted.values)
  i<-i+1
}

#plot models to get inference on residuals and normality
for (model in modelList1){
  plot(model)
}
#Linear model is okay for some of the events and terrible for other. However a linear model is also very unrealistic.

#Test for higher order regression up to cubic
#use step regression for each event starting with an empty model and ending with a full model of Y=b0+b1X+b2X^2+b3X^3.
i<-1
for (dataset in WholeData){
  name<- paste("model",i, sep="")
  name2<-paste("model_full",i, sep="")
  name3<-paste("model_final",i, sep="")
  model<-lm(Result~Year, data=dataset)
  model_full<-lm(Result~Year+I(Year^2)+I(Year^3), data=dataset)
  model_final<-step(model,scope = list(lower=formula(model),upper=formula(model_full)), direction = 'both')
  assign(name,model)
  assign(name2,model_full)
  assign(name3,model_final)
  i=i+1
}

#Create list of the polynomial regression for the best polynomial fits.
modelList2<-list(model_final1,model_final2,model_final3,model_final4,model_final5,model_final6,model_final7,model_final8,model_final9,model_final10)

#Plot the data points with the polynomial line fits.
i<-1
for (dataset in WholeData){
  plot(dataset$Year,dataset$Result,main=focus[i], col=dataset$Medal)
  lines(dataset$Year,modelList2[[i]]$fitted.values)
  i<-i+1
}

#plot models to get inference on residuals and normality
for (model in modelList2){
  plot(model)
}
#Similar to linear, polynomial fit seems to work well for some of the events but terrible for other. For both, the QQ-plots were not perfect, so lets try boxcox transformation.

#BoxCox Transformation, assume transformation of form (Y^lambda-1)/lambda=b0+b1X
#create list for the lambda values
Lambda<-c()
#Find the maximum likelihood lambda for each event and create a model with it.
i<-1
for (dataset in WholeData){
  name<- paste("model_boxcox",i, sep="")
  model<-lm(Result~Year, data=dataset)
  bc<-boxcox(model, lambda = seq(-20, 20, 1/10))
  lambda <-bc$x[which.max(bc$y)]
  Lambda<-c(Lambda,lambda)
  boxModel <- lm(((Result^(lambda-1))/lambda)~Year,data = dataset)
  assign(name, boxModel)
  title(main = focus[[i]])
  i=i+1
}

#Create list of the box cox models
modelList3<-list(model_boxcox1,model_boxcox2,model_boxcox3,model_boxcox4,model_boxcox5,model_boxcox6, model_boxcox7, model_boxcox8, model_boxcox9, model_boxcox10)

#Plot box cox line with the data points
i<-1
for (dataset in WholeData){
  name<- paste("model_boxcox",i, sep="")
  model<-lm(Result~Year, data=dataset)
  bc<-boxcox(model, lambda = seq(-20, 20, 1/10))
  lambda <-bc$x[which.max(bc$y)]
  boxModel <- lm(((Result^(lambda-1))/lambda)~Year,data = dataset)
  plot(dataset$Year,(dataset$Result^(lambda-1))/lambda, col = dataset$Medal)
  lines(dataset$Year, boxModel$fitted.values)
  assign(name, boxModel)
  title(main = focus[[i]])
  i=i+1
}

#plot models to get inference on residuals and normality
for (model in modelList3){
  plot(model)
}
#Most of the QQ plots seem to be better and others are still not super great.

#Normalize data for log transformation
avg<-mean(results$Year)
std<-sd(results$Year)
results$Year<-normalize(results$Year)

focus<-c('100M Men', '100M Women', '200M Men', '200M Women', '1500M Men', '1500M Women', 'Long Jump Men', 'Long Jump Women', 'Shot Put Men', 'Shot Put Women' )
for (event in focus){
  name<- paste("data_",event, sep="")
  assign(name,results[results$Event==event,])
}
`data_100M Men`$Result<-as.numeric(`data_100M Men`$Result)
`data_100M Women`$Result<-as.numeric(`data_100M Women`$Result)
`data_200M Men`$Result<-as.numeric(`data_200M Men`$Result)
`data_200M Women`$Result<-as.numeric(`data_200M Women`$Result)
`data_Long Jump Men`$Result<-as.numeric(`data_Long Jump Men`$Result)
`data_Long Jump Women`$Result<-as.numeric(`data_Long Jump Women`$Result)
`data_Shot Put Men`$Result<-as.numeric(`data_Shot Put Men`$Result)
`data_Shot Put Women`$Result<-as.numeric(`data_Shot Put Women`$Result)
for (i in 1:length(`data_1500M Men`$Result)){
  `data_1500M Men`$Result[i]<-as.numeric(strsplit(`data_1500M Men`$Result,':')[[i]][1])*60+as.numeric(strsplit(`data_1500M Men`$Result,':')[[i]][2])
}
for (i in 1:length(`data_1500M Women`$Result)){
  `data_1500M Women`$Result[i]<-as.numeric(strsplit(`data_1500M Women`$Result,':')[[i]][1])*60+as.numeric(strsplit(`data_1500M Women`$Result,':')[[i]][2])
}
`data_1500M Men`$Result<-as.numeric(`data_1500M Men`$Result)
`data_1500M Women`$Result<-as.numeric(`data_1500M Women`$Result)
WholeData2<-list(`data_100M Men`, `data_100M Women`, `data_200M Men`, `data_200M Women`, `data_1500M Men`, `data_1500M Women`, `data_Long Jump Men`, `data_Long Jump Women`, `data_Shot Put Men`, `data_Shot Put Women`)

#Log transformation assume Y=1/exp(X)+c

#plot Results vs. 1/exp(Year) to see if the data looks more linear
i<-1
for (dataset in WholeData2){
  plot(1/exp(dataset$Year),dataset$Result,main=paste(focus[i],' e^-x transform'), col=dataset$Medal)
  plot(dataset$Year,dataset$Result,main=paste(focus[i], 'original data'), col=dataset$Medal)
  i<-i+1
}

#Create a linear model by transforming Results with -log
i<-1
for (dataset in WholeData2){
  name<- paste("model_log",i, sep="")
  model<-lm(I(-log(Result))~Year, data=dataset)
  assign(name,model)
  i=i+1
}

#Create a list of the log transformation models
modelList4<-list(model_log1,model_log2,model_log3,model_log4,model_log5,model_log6, model_log7, model_log8, model_log9, model_log10)

#Plot the fitted values with the lines
i<-1
for (dataset in WholeData2){
  plot(1/exp(dataset$Year),dataset$Result,main=paste(focus[i],' e^-x transform'), col=dataset$Medal)
  lines(1/exp(dataset$Year), 1/exp(modelList4[[i]]$fitted.values))
  plot(dataset$Year,dataset$Result,main=paste(focus[i], 'original data'), col=dataset$Medal)
  lines(dataset$Year,1/exp(modelList4[[i]]$fitted.values))
  i<-i+1
}


#Compare Model summaries for each of the different events
for (i in 1:length(modelList1)){
  print(focus[i])
  print(summary(modelList1[[i]]))
  print(summary(modelList2[[i]]))
  print(summary(modelList3[[i]]))
  print(summary(modelList4[[i]]))
  print('##########################################################')
}

#Compare Model adj R^2 for each of the different events
for (i in 1:length(modelList1)){
  print(focus[i])
  print('linear')
  print(summary(modelList1[[i]])[[9]])
  print('polynomial')
  print(summary(modelList2[[i]])[[9]])
  print('boxcox')
  print(summary(modelList3[[i]])[[9]])
  print('log')
  print(summary(modelList4[[i]])[[9]])
  print('##########################################################')
}

#Use best model to try to predict the best that someone can ever do
for (i in 1:length(modelList1)){
  print(focus[i])
  model<-modelList1[[i]]
  Transform<-FALSE
  if (summary(model)[[9]]<summary(modelList2[[i]])[[9]]){
    model<-modelList2[[i]]
  }
  if (summary(model)[[9]]<summary(modelList3[[i]])[[9]]){
    model<-modelList3[[i]]
    Transform<-TRUE
  }
  if (summary(model)[[9]]<summary(modelList4[[i]])[[9]]){
    model<-modelList4[[i]]
  }
  if (Transform==TRUE){
    print((predict(model,newdata = data.frame(Year=1:3000), type = 'response' )[2020]*Lambda[i])^(1/(Lambda[i]-1)))
    print((predict(model,newdata = data.frame(Year=1:3000), type = 'response' )[3000]*Lambda[i])^(1/(Lambda[i]-1)))
  }
  else{
    print(predict(model,newdata = data.frame(Year=1:3000), type = 'response' )[2020])
    print(predict(model,newdata = data.frame(Year=1:3000), type = 'response' )[3000])
  }
  print('##########################################################')
}
#polynomial may have the best adjusted R^2 for most of the events but it is not good at all for the far future.

#Prediction for all models for 2020 and 3000
for (i in 1:length(modelList1)){
  print(focus[i])
  print('linear')
  print(predict(modelList1[[i]],newdata = data.frame(Year=1:3000), type = 'response' )[2020])
  print(predict(modelList1[[i]],newdata = data.frame(Year=1:3000), type = 'response' )[3000])
  print('polynomial')
  print(predict(modelList2[[i]],newdata = data.frame(Year=1:3000), type = 'response' )[2020])
  print(predict(modelList2[[i]],newdata = data.frame(Year=1:3000), type = 'response' )[3000])
  print('boxcox transform')
  print((predict(modelList3[[i]],newdata = data.frame(Year=1:3000), type = 'response' )[2020]*Lambda[i])^(1/(Lambda[i]-1)))
  print((predict(modelList3[[i]],newdata = data.frame(Year=1:3000), type = 'response' )[3000]*Lambda[i])^(1/(Lambda[i]-1)))
  print('log transform')
  print(1/exp(predict(modelList4[[i]],newdata = data.frame(Year=((1:3000)-avg)/std), type = 'response' )[2020]))
  print(1/exp(predict(modelList4[[i]],newdata = data.frame(Year=((1:3000)-avg)/std), type = 'response' )[3000]))
  print('##########################################################')
}
#Box Cox seems to be pretty realistic for near and far future.

#Plot lines of prediction from 2000 to 3000 to see how the models perform in the next 1000 years.
for (i in 1:length(modelList1)){
  ymin<-0
  ymax<-max(WholeData[[i]]$Result)
  if (i>5){
    ymax<-max(WholeData[[i]]$Result*10)
  }
  plot(2000:3000,predict(modelList1[[i]],newdata = data.frame(Year=2000:3000), type = 'response' ), type = 'l', main = focus[i], col = 'red', xlab = "Year", ylab = 'Time', ylim=c(ymin, ymax))
  par(new=TRUE)
  plot(2000:3000,predict(modelList2[[i]],newdata = data.frame(Year=2000:3000), type = 'response' ), type = 'l', main = focus[i], col = 'blue', xlab = "Year", ylab = 'Time', ylim=c(ymin, ymax))
  par(new=TRUE)
  plot(2000:3000,(predict(modelList3[[i]],newdata = data.frame(Year=2000:3000), type = 'response' )*Lambda[i])^(1/(Lambda[i]-1)), type = 'l', main = focus[i], col = 'green', xlab = "Year", ylab = 'Time', ylim=c(ymin, ymax))
  par(new=TRUE)
  plot(2000:3000,1/exp(predict(modelList4[[i]],newdata = data.frame(Year=((2000:3000)-avg)/std), type = 'response' )), type= 'l', main = focus[i], xlab = "Year", ylab = 'Time', ylim=c(ymin, ymax))
  legend('topright', legend=c("linear", "polynomial", "boxcox", "log"),col=c("red", "blue", "green", "black"), lty=1, cex=0.4)
}
#Again we can see that linear and polynomial are very unrealistic, whereas logarithmic transformation is somewhat realistic and the box cox transformation is the most realistic and should be used to predict the general trend of event scores. 


