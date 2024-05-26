
# p is number of independent variables and nn is the sample size
set.seed(123)
library(car)
library(nnet)
library(GGally)
library(lmtest)
require(hash)
library(Hmisc)
library(hier.part)
library(MASS)
library(rockchalk)
library(olsrr)
library(caret)
library(MASS)
library(glmnet)
library(GlmSimulatoR)
library(GLMMadaptive)
library(faraway)
library(ggfortify)#For graphics

p = 45
nn = 30
X.base<-data.frame(mvrnorm(nn,mu =(1/1:p),Sigma = diag(p)))



Y.base <- 1 + 2* X.base [,1] + X.base [ ,3] - 2.5 * X.base [ ,6] -
  2* X.base [ ,8] + 7/8* X.base [ ,12] + 2/5 * X.base [ ,13] + X.base [ ,14] + X.base [ ,16] + 2* X.base [ ,18] -3.0* X.base [ ,19]+
  X.base [ ,20] +2* X.base [ ,21] + X.base [ ,23] + 1.5* X.base [ ,30] - X.base [ ,31] + X.base [ ,32] + X.base [ ,33]+ 2* X.base [ ,34] - 0.25* X.base [ ,35]+ 
  0.85* X.base [ ,36] +4* X.base [ ,44]+ 3* rnorm ( nn)

Data.matrix.base <- data.frame ( Y.base , X.base )

# create empty list object for further usage
model.outcome = list()
X<- data.matrix(Data.matrix.base[c(2:46)])
Y <- Data.matrix.base$Y.base
n <- nrow(X)
train_rows <- sample(1:n, .70*n) ## 70% training, 30% test
X.train <- X[train_rows, ]
X.test <- X[-train_rows, ]
y.train <- Y[train_rows]
y.test <- Y[-train_rows]
train.set <- data.frame(y.train ,X.train)

test.set <- data.frame(y.test,X.test)
rmse2 = function(obs, preds){
  sqrt(sum((obs-preds)^2)/length(obs))
}
rmse = function(model){
  sqrt(sum((model$residual)^2)/nrow(model$model))
}
#_____________________________________________________________________________________
# Train Data set
train.model <- lm(formula = y.train ~1, data = train.set)
train.scope <- as.formula(lm(y.train ~., data = train.set))
forward_train <- step(train.model, scope = train.scope, direction = "forward", trace = 0)


# calculate rmse on training set
print(paste("RMSE on training set:", rmse(forward_train)))
summary(forward_train)

#Test Data set
test.model <- lm(formula = y.test ~1, data = test.set)
test.scope <- as.formula(lm(y.test ~., data = test.set))
step(test.model, scope = test.scope, direction = "forward")

m.forward.test <- lm(formula = y.test ~ X24 + X41 + X25 + X45 + X34 + X29 + X20 + 
                       X1, data = test.set)
print(paste("RMSE on testing set:", rmse(m.forward.test)))
summary(m.forward.test)


#_______________________________________________________________________________
                       # Ridge regression
#_______________________________________________________________________________
### RIDGE REGRESSION ###
m.ridge <- glmnet(X.train, y.train, alpha=0)
coef(m.ridge)
# plot model coefficients vs. shrinkage parameter lambda
plot(m.ridge, xvar = "lambda", label = TRUE)
plot(m.ridge, xvar = "dev", label = TRUE)


grid = 10^seq(10,-1,length=30) ## set lambda sequence
m.ridge.cv <- cv.glmnet(X.train, y.train, alpha = 0, lambda = grid)
plot(m.ridge.cv)
m.ridge.cv$lambda.min

m.ridge.cv$lambda.1se
# prediction for the training set
m.ridge.cv.pred.train = predict(m.ridge.cv, X.train, s = "lambda.min")
# calculate rmse on training set
print(paste('RMSE on training set:', rmse2(m.ridge.cv.pred.train, y.train)))
# prediction for the test set
m.ridge.cv.pred.test = predict(m.ridge.cv, X.test, s = "lambda.min")
# calculate RMSE for the test data set
print(paste('RMSE on test set:', rmse2(m.ridge.cv.pred.test, y.test)))
# store model object and results of rmse in the `list` object named `model.outcome`
model.outcome[['m.ridge.cv']] = list('model' = m.ridge.cv, 
                                     'rmse' =  data.frame('name'= 'ridge regression',
                                                          'train.RMSE' = rmse2(m.ridge.cv.pred.train, y.train),
                                                          'test.RMSE' = rmse2(m.ridge.cv.pred.test, y.test)))




#R_squared for train data
m.Ridge.cv_Tr <- cv.glmnet(X.train, y.train, alpha = 0)
Ridge_model_Tr <- glmnet(X.train,y.train ,  family = "gaussian",  alpha=0,  lambda = m.Ridge.cv_Tr$lambda.min )
Ridge_model_Tr$dev.ratio

#R-squared for test data
m.Ridge.cv_T <- cv.glmnet(X.test, y.test, alpha = 0)
Ridge_model_T <- glmnet(X.test,y.test ,  family = "gaussian",  alpha=0,  lambda = m.Ridge.cv_T$lambda.min )
Ridge_model_T$dev.ratio


#_______________________________________________________________________________________________________________
                         #LASSO Regression
#_______________________________________________________________________________________________________________

### LASSO REGRESSION ###
y= train.set$y.train

model=model.matrix(y.train~ ., data= train.set)
lasso.reg=cv.glmnet(model, y, type.measure='mse', alpha=0)
names(lasso.reg)
mse=lasso.reg$cvm[lasso.reg$lambda == lasso.reg$lambda.min]
mse
rmse = sqrt(mse)
rmse

m.lasso <- glmnet(X.train, y.train, alpha = 1)

# plot model coefficients vs. shrinkage parameter lambda

plot(m.lasso, xvar = "lambda", label = TRUE)

plot(m.lasso, xvar = "dev", label = TRUE)

m.lasso.cv <- cv.glmnet(X.train, y.train, alpha = 1)

plot(m.lasso.cv)

m.lasso.cv$lambda.min

m.lasso.cv$lambda.1se
# prediction for the training set
m.lasso.cv.pred.train = predict(m.lasso.cv, X.train, s = m.lasso.cv$lambda.min)
# calculate rmse on training set
print(paste('RMSE on training set:', rmse2(m.lasso.cv.pred.train, y.train)))
# prediction for the test set
m.lasso.cv.pred.test = predict(m.lasso.cv, X.test, s = m.lasso.cv$lambda.min)
# calculate RMSE for the test data set
print(paste('RMSE on test set:', rmse2(m.lasso.cv.pred.test, y.test)))
# store model object and results of rmse in the `list` object named `model.outcome`
model.outcome[['m.lasso.cv']] = list('model' = m.lasso.cv, 
                                     'rmse' =  data.frame('name'= 'LASSO regression',
                                                          'train.RMSE' = rmse2(m.lasso.cv.pred.train, y.train),
                                                          'test.RMSE' = rmse2(m.lasso.cv.pred.test, y.test)))


#R_squared for train data
m.lasso.cv_Tr <- cv.glmnet(X.train, y.train, alpha = 1)
lasso_model_Tr <- glmnet(X.train,y.train ,  family = "gaussian",  alpha=1,  lambda = m.lasso.cv_Tr$lambda.min )
lasso_model_Tr$dev.ratio

#R-squared for test data
m.lasso.cv_T <- cv.glmnet(X.test, y.test, alpha = 1)
lasso_model_T <- glmnet(X.test,y.test ,  family = "gaussian",  alpha=1,  lambda = m.lasso.cv_T$lambda.min )
lasso_model_T$dev.ratio

# extract matrix object
res.matrix = do.call(rbind, model.outcome)
# extract data frame object
res.df = do.call(rbind, res.matrix[,2]) 

library(ggplot2)
library(reshape2)

# melt to tidy data frame
df = melt(res.df, id.vars = "name", measure.vars = c("train.RMSE", "test.RMSE"))

# plot
ggplot(data=df, aes(x=name, y=value, fill=variable)) +
  geom_bar(stat="identity", position=position_dodge()) +
  scale_fill_hue(name="") +
  xlab("") + ylab("RMSE") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#_________________________________________________________________________________________________________________________________
mmmodel <- cv.glmnet(X.train, y.train, alpha = 1, lambda = )
m.lasso.cv$lambda.min

lasso.model <- glmnet(X.train,y.train,  family = "gaussian",  alpha=1,  lambda = m.lasso.cv$lambda.min
 )
lasso.model$dev.ratio


#______________________________________________________________________________________________________________
library(ensr)
x_matrix <- as.matrix(train.set[2:46])
y_matrix <- as.matrix(train.set$y.train)
ensr_obj <- ensr(x_matrix ,y_matrix,  standardize = FALSE)
ensr_obj
ensr_obj_summary <- summary(object = ensr_obj)
par(mfrow = c(1, 3))
plot(preferable(ensr_obj), xvar = "norm")
plot(preferable(ensr_obj), xvar = "lambda")
plot(preferable(ensr_obj), xvar = "dev")
ensr_obj_summary[cvm == min(cvm)]

elastic_net <- preferable(ensr_obj)

elastic_ne = predict(ensr_obj, X.train, s = "lambda")
print(paste('RMSE on test set:', rmse2(elastic_net, y.test)))


E_net <- glmnet(X.train, y.train, alpha = 0.5)

m.Enet.cv <- cv.glmnet(X.train, y.train, alpha = 0.5)
m.Enet.cv$lambda.min

m.Enet.cv$lambda.1se
# prediction for the training set
m.Enet.cv.pred.train = predict(m.Enet.cv, X.train, s = m.Enet.cv$lambda.min)
print(paste('RMSE on training set:', rmse2(m.Enet.cv.pred.train, y.train)))
m.Enet.cv.pred.test = predict(m.Enet.cv, X.test, s = m.Enet.cv$lambda.min)
print(paste('RMSE on test set:', rmse2(m.Enet.cv.pred.test, y.test)))
model.outcome[['m.Enet.cv']] = list('model' = m.lasso.cv, 
                                     'rmse' =  data.frame('name'= 'Elastic net',
                                                          'train.RMSE' = rmse2(m.Enet.cv.pred.train, y.train),
                                                                       'test.RMSE' = rmse2(m.Enet.cv.pred.test, y.test)))
#R-squared for test data
m.Enet.cv_T <- cv.glmnet(X.test, y.test, alpha = 0.5)
Elastic_model_T <- glmnet(X.test,y.test ,  family = "gaussian",  alpha=0.5,  lambda = m.Enet.cv_T$lambda.min )
Elastic_model_T$dev.ratio

#R_squared for train data
m.Enet.cv_Tr <- cv.glmnet(X.train, y.train, alpha = 0.5)
Elastic_model_Tr <- glmnet(X.train,y.train ,  family = "gaussian",  alpha=0.5,  lambda = m.Enet.cv_Tr$lambda.min )
Elastic_model_Tr$dev.ratio
# extract matrix object
res.matrix = do.call(rbind, model.outcome)
# extract data frame object
res.df = do.call(rbind, res.matrix[,2]) 

library(ggplot2)
library(reshape2)


# melt to tidy data frame
df = melt(res.df, id.vars = "name", measure.vars = c("train.RMSE", "test.RMSE"))

# plot
ggplot(data=df, aes(x=name, y=value, fill=variable)) +
  geom_bar(stat="identity", position=position_dodge()) +
  scale_fill_hue(name="") +
  xlab("") + ylab("RMSE") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#___________________________________________________________________________________________________________________
