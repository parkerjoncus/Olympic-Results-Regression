library(readr)
library(MASS)
library(BBmisc)
library(leaps)
library(corrplot)

#Read data
results <- read_csv("~/Documents/Olympic-Results-Regression/olympic-track-field-results/TF_results_phys_char.csv")

#Change categorical varibales to factors
results$Gender<-as.factor(results$Gender)
results$Event<-as.factor(results$Event)
results$Location<-as.factor(results$Location)
results$Medal<-as.factor(results$Medal)
results$Nationality<-as.factor(results$Nationality)
results[is.na(results$seconds),'seconds']<-as.numeric(results[is.na(results$seconds),'Result'][[1]])

#Create list of the event names we are working with
focus<-c('100M Men', '100M Women', '200M Men', '200M Women', '1500M Men', '1500M Women', 'Long Jump Men', 'Long Jump Women', 'Shot Put Men', 'Shot Put Women' )

#Create new data frames for each event seperately 
for (event in focus){
  name<- paste("data_",event, sep="")
  assign(name,results[results$Event==event,])
}

#Create a list of the individual event data frames
WholeData<-list(`data_100M Men`, `data_100M Women`, `data_200M Men`, `data_200M Women`, `data_1500M Men`, `data_1500M Women`, `data_Long Jump Men`, `data_Long Jump Women`, `data_Shot Put Men`, `data_Shot Put Women`)

#Correlation plots
i<-1
for (dataset in WholeData){
  cor(dataset[,c(4,9,10,11,12)])
  corrplot.mixed(cor(dataset[,c(4,9,10,11,12)]), main = paste('Correlation Plot',focus[i]))
  i<-i+1
}

#step regression with single order
i<-1
for (dataset in WholeData){
  name<- paste("model",i, sep="")
  name2<-paste("model_full",i, sep="")
  name3<-paste("model_final",i, sep="")
  model<-lm(seconds~1, data=dataset)
  model_full<-lm(seconds~Year+Medal+Nationality+Age+Weight+Height, data=dataset)
  model_final<-step(model,scope = list(lower=formula(model),upper=formula(model_full)), direction = 'both')
  assign(name,model)
  assign(name2,model_full)
  assign(name3,model_final)
  i=i+1
}

modelList1<-list(model_final1,model_final2,model_final3,model_final4,model_final5,model_final6,model_final7,model_final8,model_final9,model_final10)

for (model in modelList1){
  print(summary(model))
}

#Test for higher order and interaction regression up to squared
#use step regression for each event starting with an empty model and ending with a full model of Y=b0+b1X+b2X^2+b3X^3.
i<-1
for (dataset in WholeData){
  name<- paste("model",i, sep="")
  name2<-paste("model_full",i, sep="")
  name3<-paste("model_final",i, sep="")
  model<-lm(seconds~1, data=dataset)
  model_full<-lm(seconds~Year+Medal+Nationality+Age+Weight+Height+I(Year*Age)+I(Year*Weight)+I(Year*Height)+I(Age*Weight)+I(Age*Height)+I(Height*Weight)+I(Year^2)+I(Age^2)+I(Weight^2)+I(Height^2), data=dataset)
  model_final<-step(model,scope = list(lower=formula(model),upper=formula(model_full)), direction = 'both')
  assign(name,model)
  assign(name2,model_full)
  assign(name3,model_final)
  i=i+1
}

modelList2<-list(model_final1,model_final2,model_final3,model_final4,model_final5,model_final6,model_final7,model_final8,model_final9,model_final10)

for (model in modelList2){
  print(summary(model))
}

#List adj R^2 values for each event
for (i in 1:length(modelList2)){
  print(focus[i])
  print('single order')
  print(summary(modelList1[[i]])[[9]])
  print('higher order')
  print(summary(modelList2[[i]])[[9]])
  print('##########################################################')
}
