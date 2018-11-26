library(readr)

#Read data
results <- read_csv("~/Documents/Olympic-Results-Regression/olympic-track-field-results/results.csv")
View(results)

#Change categorical varibales to factors
results$Gender<-as.factor(results$Gender)
results$Event<-as.factor(results$Event)
results$Location<-as.factor(results$Location)
results$Medal<-as.factor(results$Medal)
results$Nationality<-as.factor(results$Nationality)

focus<-c('100M Men', '100M Women', '200M Men', '200M Women', '1500M Men', '1500M Women', 'Long Jump Men', 'Long Jump Women', 'Shot Put Men', 'Shot Put Women' )
#Create new data frames for each event seperately 
for (event in focus){
  name<- paste("data_",event, sep="")
  assign(name,results[results$Event==event,])
}

`data_100M Men`$Result<-as.numeric(`data_100M Men`$Result)
`data_100M Women`$Result<-as.numeric(`data_100M Women`$Result)
`data_200M Men`$Result<-as.numeric(`data_100M Men`$Result)
`data_200M Women`$Result<-as.numeric(`data_100M Women`$Result)
`data_Long Jump Men`$Result<-as.numeric(`data_100M Men`$Result)
`data_Long Jump Women`$Result<-as.numeric(`data_100M Women`$Result)
`data_Shot Put Men`$Result<-as.numeric(`data_100M Men`$Result)
`data_Shot Put Women`$Result<-as.numeric(`data_100M Women`$Result)
for (i in 1:length(`data_1500M Men`$Result)){
  `data_1500M Men`$Result[i]<-as.numeric(strsplit(`data_1500M Men`$Result,':')[[i]][1])*60+as.numeric(strsplit(`data_1500M Men`$Result,':')[[i]][2])
}
for (i in 1:length(`data_1500M Women`$Result)){
  `data_1500M Women`$Result[i]<-as.numeric(strsplit(`data_1500M Women`$Result,':')[[i]][1])*60+as.numeric(strsplit(`data_1500M Women`$Result,':')[[i]][2])
}

WholeData<-list(`data_100M Men`, `data_100M Women`, `data_200M Men`, `data_200M Women`, `data_1500M Men`, `data_1500M Women`, `data_Long Jump Men`, `data_Long Jump Women`, `data_Shot Put Men`, `data_Shot Put Women`)

#linear regression
i=1
for (dataset in WholeData){
  name<- paste("model",i, sep="")
  model<-lm(Result~Year, data=dataset)
  assign(name,model)
  i=i+1
}

modelList1<-list(model1,model2,model3,model4,model5,model6,model7,model8,model9,model10)

#Plot data points
i=1
for (dataset in WholeData){
  plot(dataset$Year,dataset$Result,main=focus[i], col=dataset$Medal)
  i<-i+1
}

#Plot fitted values with data points
i=1
for (dataset in WholeData){
  plot(dataset$Year,dataset$Result,main=focus[i], col=dataset$Medal)
  lines(dataset$Year,modelList1[[i]]$fitted.values)
  i<-i+1
}

