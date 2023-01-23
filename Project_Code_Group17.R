library(caret)
library(ggplot2)
library(GGally)
library(forcats)
library(recipes)
library(factoextra)
library(kernlab)

autoinsurance <- read.csv("AutoInsurance.csv")
dim(autoinsurance)
summary(autoinsurance)

## EDA
autoinsurance %>%
  select_if(is.numeric) %>%
  ggpairs(progress = FALSE)

pacman::p_load(dplyr, psych, ggplot2, recipes,kableExtra)

# overall pie chart
pacman::p_load(ggstatsplot)
piechart <- ggpiestats(autoinsurance, 'Response', 
                       results.subtitle = F, 
                       factor.levels = c('Yes', 'No'),
                       label = "both", 
                       perc.k = 0, 
                       direction = 1, 
                       palette = 'Pastel2',
                       size = 1,
                       title = 'Number of response')
piechart


# view the situations between genders using bar plots
response_gender <- autoinsurance %>%
  group_by(Gender, Response)%>%
  summarise(n=n())


ggplot(response_gender, aes(x = Response, y = n))+
  labs(x = NULL, y = NULL, fill = NULL, title = "Offering Renew Offer Type") +
  geom_bar(
    aes(fill = Response), stat = "identity", color = "white",
    position = position_dodge(0.9)
  )+
  theme_classic() +
  facet_wrap(~Gender)


# box plot considering Total Claim Amount
ggplot(autoinsurance, aes(x=Response, y=Total.Claim.Amount)) + 
  geom_boxplot(fill='#A4A4A4', color="darkblue") +
  theme_minimal() +
  labs(title = "Boxplot of Customers Decision")


# we check the trend between "income" and "Response"
ggplot(data = autoinsurance, mapping = aes(x = Income, y = ..count..)) + 
  geom_freqpoly(mapping = aes(colour = Response), binwidth = 2000)


response_offer <- autoinsurance %>%
  group_by(Response,Renew.Offer.Type)%>%
  summarise(n=n())

pacman::p_load(ggthemes)
ggplot(response_offer,aes(Response,n,fill=Renew.Offer.Type))+
  geom_bar(stat="identity",position="fill")+
  ggtitle("Constitution of Offer Type ")+
  theme_wsj()+
  scale_fill_wsj("rgby", "")+
  theme(axis.ticks.length=unit(0.5,'cm'))+
  guides(fill=guide_legend(title=NULL))+
  coord_flip()


## Data preprocessing before data spliting
### Dealing with Missingness 
# check missing data
sum(is.na(autoinsurance))

### Feature Filtering
# delete customer ID
autoinsurance$Customer <- NULL

# check nzv
nearZeroVar(autoinsurance,saveMetrics = TRUE)

### Categorical Feature Engineering
### 1) Dummy Encoding
# categorical variable as factor
col.names <- c("State","Response","Coverage","Education","EmploymentStatus","Gender","Location.Code","Marital.Status","Policy.Type","Policy","Renew.Offer.Type","Sales.Channel","Vehicle.Class","Vehicle.Size")
autoinsurance[,col.names] <- lapply(autoinsurance[,col.names], as.factor)


### 2) Label Encoding
# ordinal variable
autoinsurance$Coverage <- factor(autoinsurance$Coverage, order = TRUE, 
                                 levels = c("Basic", "Extended", "Premium"))
autoinsurance$Education <- factor(autoinsurance$Education, order = TRUE, 
                                  levels = c("High School or Below", "College","Bachelor","Master","Doctor"))
autoinsurance$Vehicle.Size <- factor(autoinsurance$Vehicle.Size, order = TRUE, 
                                     levels = c("Small", "Medsize", "Large"))


### 3) Lumping
# Effective.To.Date -> month
to.month = c()
for (i in autoinsurance$Effective.To.Date) {
  # append(to.month,paste(unlist(strsplit(i,"/"))[1],"2011",sep = "/"))
  to.month <- append(to.month,unlist(strsplit(i,"/"))[1])
}

autoinsurance$Effective.To.Month <- as.factor(to.month)

# delete Effective.To.Date
autoinsurance$Effective.To.Date <- NULL

# check if some variables need lumping
autoinsurance$Policy %>% fct_lump_prop(0.01) %>% table()
autoinsurance$Effective.To.Month %>% fct_lump_prop(0.01) %>% table()

# check outlier
summary(dplyr::select_if(autoinsurance, is.numeric))

### Data spliting
# split data,stratified sampling
set.seed("123")

split <- initial_split(autoinsurance,prop = 0.7,strata = "Response")

train <- training(split)
test <- testing(split)

### Data preprocessing after data spliting
blueprint <- recipe(Response ~ .,data = train) %>% 
  step_center(all_numeric(),-all_outcomes()) %>% 
  step_scale(all_numeric(),-all_outcomes()) %>% 
  step_normalize(all_numeric(),-all_outcomes())

prepare <- prep(blueprint, training = train)
baked_train <- bake(prepare, new_data = train)
names(baked_train)

# numbers of features
n_features <- length(setdiff(names(baked_train),"Response"))
n_features


rf <- ranger(
  Response ~ .,
  data = baked_train,
  mtry = floor(n_features / 3),
  respect.unordered.factors = "order",
  seed = 123
)

rf$num.trees
rf$num.independent.variables
rf$mtry
rf$min.node.size

defaut_rf_rmse <- sqrt(rf$prediction.error)
defaut_rf_rmse

# create hyperparameter grid
# hyper.grid <- expand.grid(
#   mtry = floor(n_features*c(.09,.27,.333,.64,.82)), # 2-22
#   min.node.size = c(1,3,5,10),
#   replace = c(TRUE,FALSE),
#   sample.fraction = c(.44,.63,.81,0.9),
#   rmse = NA
# )
hyper.grid <- expand.grid(num.trees = floor(n_features * c(5, 10, 15)),
                          mtry = c(2, 7, 12, 17, 22),
                          min.node.size = c(1, 5, 10),
                          replace = c(TRUE, FALSE),
                          sample.fraction = c(.5, .63, .8),
                          rmse = NA)

# execute full cartesian grid search
for(i in seq_len(nrow(hyper.grid))){
  # fit model for ith hyperparameter combination
  fit <- ranger(
    formula = Response ~ .,
    data = train,
    num.trees = n_features*10,
    mtry = hyper.grid$mtry[i],
    min.node.size = hyper.grid$min.node.size[i],
    replace = hyper.grid$replace[i],
    sample.fraction = hyper.grid$sample.fraction[i],
    verbose = FALSE,
    seed = 123,
    respect.unordered.factors = "order"
  )
  hyper.grid$rmse[i] <- sqrt(fit$prediction.error)
}

hyper.grid %>% 
  arrange(rmse) %>% 
  mutate(perc_gain = (defaut_rf_rmse - rmse)/defaut_rf_rmse*100) %>% 
  head(50)

# rf model with best hyperparameter
rf.best <- ranger(
  Response ~ .,
  data = train,
  num.trees = 110,
  mtry = 7,
  min.node.size = 1,
  replace = FALSE,
  sample.fraction = 0.80,
  importance = "permutation",
  verbose = FALSE,
  seed = 123,
  respect.unordered.factors = "order",
  method = "rpart",
  probability = TRUE
)

vip(rf.best,num_features = 22,scale = TRUE)

prob_yes <- function(object, newdata) {
  predict(object, newdata = newdata, type = "prob")[, "Yes"]
}

features <- c("Income", "Renew.Offer.Type", 
              "Total.Claim.Amount", "Customer.Lifetime.Value")
pdps <- lapply(features, function(x) {
  partial(rf.best, pred.var = x, which.class = 2,  
          prob = TRUE, plot = TRUE, plot.engine = "ggplot2") 
})
grid.arrange(grobs = pdps,  ncol = 2)


pd.rf <- partial(rf.best, pred.var = c("Income", "EmploymentStatus"))
# Default PDP
pdp1 <- plotPartial(pd.rf)

pdp1


dplyr::select_if(train, is.numeric)
 
 
summary(train)
 
## SVM
 
ctrl <- trainControl(
  method = "cv", 
  number = 10, 
  classProbs = TRUE,                 
  summaryFunction = twoClassSummary  # also needed for AUC/ROC
)
 

 
set.seed(123)  # for reproducibility
svm_auc <- train(
  Response ~ ., 
  data = baked_train,
  method = "svmRadial",               
  metric = "ROC",  # area under ROC curve (AUC)       
  trControl = ctrl,
  tuneLength = 10
)
 

 
confusionMatrix(svm_auc)
 
 
prob_yes <- function(object, newdata) {
  predict(object, newdata = newdata, type = "prob")[, "Yes"]
}
 

 
set.seed(123)  # for reproducibility
vip(svm_auc, method = "permute", nsim = 5, train = baked_train, 
    target = "Response", metric = "auc", reference_class = "Yes", 
    pred_wrapper = prob_yes,num_features = 22)

features <- c("Effective.To.Month", "EmploymentStatus", 
              "Renew.Offer.Type", "Education")
pdps <- lapply(features, function(x) {
  partial(svm_auc, pred.var = x, which.class = 2,  
          prob = TRUE, plot = TRUE, plot.engine = "ggplot2") 
})
grid.arrange(grobs = pdps,  ncol = 2)
 


vint(
  object= svm_auc,
  feature_names = c("Income", "Education"),
  progress = "none",
  parallel = FALSE,
  paropts = NULL,
)

vint(
  object= svm_auc,
  feature_names = c("Income", "EmploymentStatus"),
  progress = "none",
  parallel = FALSE,
  paropts = NULL,
)


## Classifcation
  
train_Y <- as.matrix(baked_train[, 23])
train_X <- as.matrix(baked_train[, -23])
test_Y <- as.matrix(baked_test[, 23])
test_X <- as.matrix(baked_test[, -23])
 

### K-Nearest Neighbor
  
# prepare data for KNN
KNN_recipe <- recipe(Response ~ .,data = train) %>%
  step_center(all_numeric(),-all_outcomes()) %>%
  step_scale(all_numeric(),-all_outcomes()) %>%
  step_normalize(all_numeric(),-all_outcomes()) %>%
  step_integer(matches("Basic", "Extended", "Premium")) %>%
  step_integer(matches("High School or Below", "College", "Bachelor", "Master")) %>%
  step_integer(matches("Small", "Medsize", "Large")) %>%
  step_dummy(all_nominal(), -all_outcomes())
 

  
#head(KNN_recipe)
summary(train)
KNN_prepare <- prep(KNN_recipe, training = train)
KNN_train <- bake(KNN_prepare, new_data = train)
KNN_test <- bake(KNN_prepare, new_data = test)
head(KNN_train)

dim(KNN_train)
dim(KNN_test)

#summary(train)
 

  
# Setting up train controls
numbers = 10
tunel = 10

set.seed(123)
x = trainControl(method = "cv",
                 number = numbers,
                 classProbs = TRUE,
                 summaryFunction = twoClassSummary)

KNN.tune <- train(Response ~ ., 
                  data = KNN_train, method = "knn",
                  trControl = x,
                  tuneLength = tunel)

# Summary of model
KNN.tune
 

  
plot(KNN.tune)
 

  
# make predictions
KNN.pred <- predict(KNN.tune$finalModel, 
                    KNN_test[setdiff(names(KNN_test), 'Response')], 
                    type = "class")

# calculate accuracy
KNN.table <- table(true = KNN_test$Response, pred = KNN.pred)
KNN.accuracy <- (KNN.table[1] + KNN.table[4]) / length(KNN.pred)
KNN.accuracy
KNN.table
 

  
vip(KNN.tune,scale = TRUE)
 

  
prob_yes <- function(object, newdata) {
  predict(object, newdata = newdata, type = "prob")[, "Yes"]
}
 

  
KNN.table[2:1, 2:1]
 

  
levels(KNN_test$Response) <- c("Yes","No")
levels(KNN.pred) <- c("Yes","No")
knn.tb <- confusionMatrix(KNN_test$Response,KNN.pred)
knn.tb
 
### SVM
  
set.seed(123)

# tune linear SVM model
SVM.linear.tune <- tune(svm, Response ~ ., data = baked_train, kernel = "linear",
                        ranges = list(cost = c(0.01, 0.1, 1, 5, 10)))
summary(SVM.linear.tune)
 

  
# select best 'cost' and refit model using whole training data
SVM.linear = svm(Response ~ ., data = baked_train, kernel = 'linear', cost = 1)

summary(SVM.linear)

# make predictions
SVM.linear.pred <- predict(SVM.linear, newdata = baked_test)

# calculate accuracy
SVM.linear.table <- table(true = baked_test$Response, pred = SVM.linear.pred)
SVM.linear.accuracy <- (SVM.linear.table[1] + SVM.linear.table[4]) / length(SVM.linear.pred)
SVM.linear.accuracy
SVM.linear.table
 
  
levels(baked_test$Response) <- c("Yes","No")
levels(SVM.linear.pred) <- c("Yes","No")
svm.linear.tb <- confusionMatrix(baked_test$Response,SVM.linear.pred)
svm.linear.tb
 

  
set.seed(123)

# tune polynomial SVM model
SVM.poly.tune <- tune(svm, Response ~ ., data = baked_train, kernel = "polynomial",
                      ranges = list(cost = c(0.01, 0.1, 1, 5, 10), 
                                    degree = c(3, 4, 5, 6, 7, 8)))
summary(SVM.poly.tune)
 

  
# refit model using whole training data
SVM.poly = svm(Response ~ ., data = baked_train, kernel = 'polynomial', 
               cost = 10, degree = 3)

summary(SVM.poly)

# make predictions
SVM.poly.pred <- predict(SVM.poly, newdata = baked_test)

# calculate accuracy
SVM.poly.table <- table(true = baked_test$Response, pred = SVM.poly.pred)
SVM.poly.accuracy <- (SVM.poly.table[1] + SVM.poly.table[4]) / length(SVM.poly.pred)
SVM.poly.accuracy
SVM.poly.table
 
  
svm.poly.tb <- confusionMatrix(baked_test$Response,SVM.poly.pred)
svm.poly.tb
 

  
set.seed(123)

# tune radial SVM model
SVM.radial.tune <- tune(svm, Response ~ ., data = baked_train, kernel = "radial",
                        ranges = list(cost = c(0.01, 0.1, 1, 5, 10), 
                                      gamma = c(0.5, 1, 2, 3)))
summary(SVM.radial.tune)
 

  
# refit model using whole training data
SVM.radial = svm(Response ~ ., data = baked_train, kernel = 'radial', 
                 cost = 5, gamma = 0.5)

summary(SVM.radial)

# make predictions
SVM.radial.pred <- predict(SVM.radial, newdata = baked_test)

# calculate accuracy
SVM.radial.table <- table(true = baked_test$Response, pred = SVM.radial.pred)
SVM.radial.accuracy <- (SVM.radial.table[1] + SVM.radial.table[4]) / length(SVM.radial.pred)
SVM.radial.accuracy
SVM.radial.table
 
  
svm.radial.tb <- confusionMatrix(baked_test$Response,SVM.radial.pred)
svm.radial.tb
 
  
vip(SVM.radial.pred,scale = TRUE)
vip(churn_svm_auc, method = "permute", nsim = 5, train = churn_train, 
    target = "Attrition", metric = "auc", reference_class = "Yes", 
    pred_wrapper = prob_yes)
 

### Random Forest
  
# no tuning

# number of features
n_features <- length(setdiff(names(baked_train), 'Response'))

# default model
RF_default <- ranger(Response ~ ., 
                     data = baked_train, 
                     mtry = floor(n_features/3),
                     respect.unordered.factors = 'order',
                     seed = 123)

# get baseline RMSE
RF_default_error <- sqrt(RF_default$prediction.error)
RF_default_error
 

  
# tuning parameters
# parameters: #Trees, Mtry, Min node size/Max depth, Sampling scheme

# create hyperparameter grid
RF_hyper_grid <- expand.grid(num.trees = floor(n_features * c(5, 10, 15)),
                             mtry = c(2, 7, 12, 17, 22),
                             min.node.size = c(1, 5, 10),
                             replace = c(TRUE, FALSE),
                             sample.fraction = c(.5, .63, .8),
                             error = NA)

for (i in seq_len(nrow(RF_hyper_grid))) {
  fit <- ranger(
    formula = Response ~ .,
    data = baked_train,
    num.trees = RF_hyper_grid$num.trees[i],
    mtry = RF_hyper_grid$mtry[i],
    min.node.size = RF_hyper_grid$min.node.size[i],
    replace = RF_hyper_grid$replace[i],
    sample.fraction = RF_hyper_grid$sample.fraction[i],
    verbose = FALSE,
    seed = 123,
    respect.unordered.factors = 'order'
  )
  RF_hyper_grid$error[i] <- sqrt(fit$prediction.error)
}

RF_hyper_grid %>%
  arrange(error) %>%
  mutate(perc_gain = (RF_default_error - error) / RF_default_error * 100) %>%
  head(10)
 

  
# rerun model with permutation-based variable importance
RF.final <- ranger(formula = Response ~ .,
                   data = baked_train,
                   num.trees = 220,
                   mtry = 7,
                   min.node.size = 1,
                   replace = FALSE,
                   sample.fraction = 0.8,
                   importance = 'permutation',
                   respect.unordered.factors = 'order',
                   verbose = FALSE,
                   seed = 123)

vip(RF.final, num_features = 22, scale = TRUE)

# make predictions
RF.pred <- predict(RF.final, data = baked_test)

# calculate accuracy
RF.table <- table(true = baked_test$Response, pred = RF.pred$predictions)
RF.accuracy <- (RF.table[1] + RF.table[4]) / length(RF.pred$predictions)
RF.accuracy

RF.table
 

  
rf.tb <- confusionMatrix(baked_test$Response,RF.pred$predictions)
rf.tb
 

### Decision Tree
  
# fit the model
DT = train(Response ~ ., 
           data = baked_train, 
           method = "rpart", 
           trControl = trainControl(method = "cv"))

# plot the model
fancyRpartPlot(DT$finalModel)
 

  
# make predictions
DT.pred = predict(DT, newdata = baked_test)

# calculate accuracy
DT.table <- table(true = baked_test$Response, pred = DT.pred)
DT.accuracy <- (DT.table[1] + DT.table[4]) / length(DT.pred)
DT.accuracy

DT.table
 
  
dt.tb <- confusionMatrix(baked_test$Response,DT.pred)
dt.tb
 
  
set.seed(123)  # for reproducibility
vip(DT, method = "permute", nsim = 5, train = baked_train, 
    target = "Response", metric = "auc", reference_class = "Yes", 
    pred_wrapper = prob_yes)
 


 
##logistic
  
set.seed(7027)
cv <- trainControl(method = "repeatedcv",number = 10,repeats=5)
 

  
logit_cv <- train(
  Response~.,
  data = baked_train,
  method = 'glm',
  family = 'binomial',
  trControl = cv,
)
 
  
logit_test <- predict(logit_cv,baked_test)
logistic.tb <- table(logit_test,baked_test$Response)
logit_acccuracy <- (2312+75)/2741 #= 87.09%
 
  
logistic.accuracy <- (logistic.tb[1] + logistic.tb[4]) / length(logit_test)
logistic.accuracy
 


logit_test2 <- predict(logit_cv,baked_test,type = 'prob')
logit.roc <- roc(baked_test$Response, logit_test2[,1], plot = T, col = 'blue')
 
  
# 1st
logit_p <-vip::vip(logit_cv,num_features = 23, scale = TRUE)
logit_p
 

# 2nd
pred<-function(object,newdata){
  results <-as.vector(predict(object,newdata))
  return(results)
}
vip(
  logit_cv,
  train = baked_train,
  method = 'permute',
  target = 'Response',
  metric = 'accuracy',
  nsim = 5,
  sample_frac = 0.5,
  pred_wrapper = pred
)
 
  
confusionMatrix(baked_test$Response,logit_test)
 

##LDA
lda_cv <- train(
  Response~.,
  data = baked_train,
  method = 'lda',
  trControl = cv,
  
)
 

  
lda_test <- predict(lda_cv,baked_test)
lda.tb <- table(true = baked_test$Response, pred = lda_test)
lda_acccuracy <- (2318+72)/2741 #= 87.19%
lda.tb
 
  
lda.accuracy <- (lda.tb[1] + lda.tb[4]) / length(lda_test)
lda.accuracy
 
  
confusionMatrix(baked_test$Response,lda_test)
 




lda_test2 <- predict(lda_cv,baked_test,type = 'prob')
lda.roc <- roc(baked_test$Response, lda_test2[,1], plot = T, col = 'blue')
 


  
pred <-function(object,newdata){
  results <-as.vector(predict(object,newdata))
  return(results)
}
vip(
  lda_cv,
  train = baked_train,
  method = 'permute',
  target = 'Response',
  metric = 'accuracy',
  nsim = 5,
  sample_frac = 0.5,
  pred_wrapper = pred
)
 

##GBM
  
library(gbm)
library(survival)
 
  
#gbm -> numeric
baked_train2 <- baked_train
baked_test2 <- baked_test
baked_train2$Response<-as.numeric(baked_train2$Response)-1
baked_test2$Response<-as.numeric(baked_test2$Response)-1
 

  
#start with learning rate = 0.1,tree = 10000, tree depth =3, min node size = 10, 10-fold cv
set.seed(7027)
gbm <- gbm(
  formula = Response~.,
  data = baked_train2,
  distribution = 'bernoulli',
  n.trees = 10000,
  shrinkage = 0.1,
  interaction.depth = 3,
  n.minobsinnode = 10,
  cv.folds = 10,
)
 
  
gbm_iter = gbm.perf(gbm,method='cv')
gbm_iter  #6503
 

  
gbm_test <- predict(gbm,baked_test2,n.trees = gbm_iter)
 

gbm.roc <- roc(baked_test2$Response, gbm_test, n.trees = gbm_iter,plot = T, col = 'blue')
 

  
#get the best threshold
coords(gbm.roc,'best')
 
  
gbm_test_class=c()
for (i in gbm_test){
  gbm_test_class <-cbind(gbm_test_class,if_else(i>coords(gbm.roc,'best')['threshold'],'Yes','No'))}

 
  
table(baked_test$Response,gbm_test_class)
 
  
gbm_acccuracy <- 1-20/2741 # 0.993
 

  
#fix tree hyperparameter and tune learning rate
GBM_hyper_grid1 <- expand.grid(
  learning_rate = c(0.3,0.1,0.05,0.01)
)
 
   
for(i in seq_len(nrow(GBM_hyper_grid1))){
  set.seed(7027)
  train_time <-system.time({
    gbm2 <- gbm(
      formula = Response~.,
      data = baked_train2,
      distribution = 'bernoulli',
      n.trees = 8000,
      shrinkage = GBM_hyper_grid1$learning_rate[i],
      interaction.depth = 3,
      n.minobsinnode = 10,
      cv.folds = 10,
    )
  })
  GBM_hyper_grid1$error[i] <-sqrt(mean(gbm2$cv.error))
  GBM_hyper_grid1$trees[i] <-which.min(gbm2$cv.error)
  GBM_hyper_grid1$time[i] <- train_time[['elapsed']]
}
 
  
#results
arrange(GBM_hyper_grid1,error)
 
  
GBM_hyper_grid2 <- expand.grid(
  interaction.depth =c(3,5,7),
  n.minobsinnode = c(5,10,15)
)
 
   
for(i in seq_len(nrow(GBM_hyper_grid2))){
  set.seed(7027)
  train_time <-system.time({
    gbm3 <- gbm(
      formula = Response~.,
      data = baked_train2,
      distribution = 'bernoulli',
      n.trees = 8000,
      shrinkage = 0.3,
      interaction.depth = GBM_hyper_grid2$interaction.depth[i],
      n.minobsinnode = GBM_hyper_grid2$n.minobsinnode[i],
      cv.folds = 10,
    )
  })
  GBM_hyper_grid2$error[i] <-sqrt(mean(gbm3$cv.error))
  GBM_hyper_grid2$trees[i] <-which.min(gbm3$cv.error)
  GBM_hyper_grid2$time[i] <- train_time[['elapsed']]
}
 
  
#results
arrange(GBM_hyper_grid2,error)
 

  
#best model
set.seed(123)
gbm_best <- gbm(
  formula = Response~.,
  data = baked_train2,
  distribution = 'bernoulli',
  n.trees = 3000,
  shrinkage = 0.3,
  interaction.depth = 3,
  n.minobsinnode = 5,
  cv.folds = 10,
)
 

  
gbm.best.model <- GBM.train(X.train, Y.train, s_f = 0.3, s_s = 1, lf =1, M.train = 5000, nu = 0.001)
 


  
gbm_iter2 = gbm.perf(gbm_best,method='cv')
gbm_iter2  #2621
 

  
gbm_test2 <- predict(gbm_best,baked_test2,n.trees = gbm_iter2, type="response")
 


gbm.roc2 <- roc(baked_test2$Response, gbm_test2, n.trees = gbm_iter,plot = T, col = 'blue')
 
   
#get the best threshold
coords(gbm.roc2,'best')
 
  
gbm_test_class2=c()
for (i in gbm_test2){
  gbm_test_class2 <-cbind(gbm_test_class2,if_else(i>coords(gbm.roc2,'best')['threshold'],'Yes','No'))}
 
  
table(baked_test$Response,gbm_test_class2)
 
  
confusionMatrix(baked_test$Response,gbm_test2)
 
  
vip(gbm_best,num_features = 22,scale = TRUE)
 

  
gbm_acccuracy <- 1-28/2741 # 0.990
 

  
vi(gbm_best,scale = TRUE)
 

  
features <- c("Customer.Lifetime.Value", "Income", 
              "Effective.To.Month","EmploymentStatus","Total.Claim.Amount","Renew.Offer.Type")
pdps <- lapply(features, function(x) {
  partial(gbm_best, pred.var = x, which.class = 2,n.trees = 3000,  
          prob = TRUE, plot = TRUE, plot.engine = "ggplot2") 
})
grid.arrange(grobs = pdps,  ncol = 2)
 
