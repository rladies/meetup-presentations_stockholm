# Date:     2019-02-18
# Author:   Annika Tillander
# Purpose:  LAB for R-ladies

my_lda <- function(x, y, sigma_inv, tx)
{
  n <- dim(x)[1]
  p <- dim(x)[2]	
  sy <- sort((y), decreasing = FALSE, index.return = TRUE)$ix	# Sort the values of y
  n_k <- table(y)					# number of oberservation in each class	
  pi <- n_k/n						# proportion in each class
  sx <- x[sy,]					# sort x in the order of y
  my <- matrix(NaN, 2, p)			# matrix to put the meanvalues
  my[1,] <- apply(sx[1:n_k[1],], 2, mean) # mean values for the first class
  my[2,] <- apply(sx[(n_k[1]+1):n,], 2, mean)# mean values for the second class
  lim_value <- (1/2)*(t(my[2,])%*%sigma_inv%*%my[2,]) - (1/2)*(t(my[1,])%*%sigma_inv%*%my[1,]) + log(pi[1]) - log(pi[2])
  test_value <- tx%*%sigma_inv%*%(my[2,]-my[1,])
  ty <- as.numeric(test_value > as.numeric(lim_value))
  return(ty)
}

# Load data
load(file="C:/***/sim_dat.Rdata")

# The packages needed
require(ggplot2)
require(pROC) 
require(ModelMetrics)
require(caret)
require(DAAG)
require(cvTools)
require(class) 
require(randomForest)
require(glasso)
require(nnet)

n <- length(y) # Number of observations
p <- dim(x)[2] # Number of features

# Random sample 
test_urval <- sample(1:n, 20)

#Test data
test_y <- y[test_urval]
test_x <- x[test_urval, ]

# Check the size
length(test_y)
dim(test_x)

# Train data 
train_y <- y[-test_urval]
train_x <- x[-test_urval,]

# Check the size
length(train_y)
dim(train_x)

# Cross validation
# Numer of folds
my_k <- 4

# Subset data into folds
# In caret 
flds <- createFolds(train_y, k = my_k, list = TRUE, returnTrain = FALSE) 

# In cvTools
folds <- cvFolds(NROW(train_x), K=my_k) 

# Paramter space use only 6 possible 
ps <- 6
lambda <-  (3:8)/10 # Lambda for gLASSO
layers <- 2:7 # hidden layers for nn
nb <- 2:7 # neighbors for knn
nf <- 3*(2:7) # Number of features randomly sampled as candidates at each split

# Scaling and Centering advised for neural networks
s_train_x <- scale(train_x, center = TRUE, scale = TRUE)
xy <- cbind(train_y, s_train_x)

# Turning outcome into factor for random forest
rf_y <- as.factor(train_y)
rf_xy <- data.frame(rf_y, train_x)

# Collect outcomes
temp_auc <- matrix(NaN, 4, my_k)
# For the result
result_auc <- matrix(NaN, 4, ps)
rownames(result_auc) <- c("LDA", "NN", "kNN", "RF")
colnames(result_auc) <- 1:ps
for(i in 1:ps) # Loop over parameter space
{
  for (j in 1:my_k) # Loop over folds
  {
    true_y <- train_y[flds[[j]]]
    s <- cov(train_x[-flds[[j]],])
    is <- glasso(s, rho=lambda[i])$wi 
    pred_lda <- my_lda(x = train_x[-flds[[j]],], y = train_y[-flds[[j]]], 
                       sigma_inv=is, tx=as.matrix(train_x[flds[[j]],]))
    mod_nn <- nnet(train_y ~ ., data=xy[-flds[[j]],], size = layers[i], trace = FALSE)
    pred_nn <- as.numeric(predict(mod_nn, xy[flds[[j]],])>0.1)
    pred_knn <- knn(train = train_x[-flds[[j]],], test = train_x[flds[[j]],], 
                    cl = train_y[-flds[[j]]], k = nb[i])
    mod_rf <- randomForest(rf_y  ~ ., data = rf_xy[-flds[[j]],], mtry = nf[i])
    pred_rf <- as.numeric(predict(mod_rf, rf_xy[flds[[j]],]))-1
    # Using auc from ModelMetrics
    temp_auc[, j] <- c(auc(true_y, pred_lda), auc(true_y, pred_nn), auc(true_y, pred_knn), auc(true_y, pred_rf))
  }
  result_auc[,i] <- apply(temp_auc, 1, mean)
}

result_auc

# Get parameter values that generate best AUC
lda_max <- as.numeric(colnames(result_auc)[result_auc[1,]==max(result_auc[1,])])
nn_max <- as.numeric(colnames(result_auc)[result_auc[2,]==max(result_auc[2,])])
knn_max <- as.numeric(colnames(result_auc)[result_auc[3,]==max(result_auc[3,])])
rf_max <- as.numeric(colnames(result_auc)[result_auc[4,]==max(result_auc[4,])])

# Final test
s <- cov(train_x)
is <- glasso(s, rho=lambda[lda_max])$wi 
pred_lda <- my_lda(x = train_x, y = train_y, 
                   sigma_inv=is, tx=as.matrix(test_x))
mod_nn <- nnet(train_y ~ ., data=xy, size = layers[nn_max], trace = FALSE)
c_test_x <-  scale(test_x, center = TRUE, scale = TRUE)
pred_nn <- as.numeric(predict(mod_nn, c_test_x)>0.1)
pred_knn <- knn(train = train_x, test = test_x, cl = train_y, k = nb[knn_max])
mod_rf <- randomForest(rf_y  ~ ., data = rf_xy, mtry = nf[rf_max])
rf_test_x <- test_x
colnames(rf_test_x) <- colnames(rf_xy)[-1]
pred_rf <- as.numeric(predict(mod_rf, rf_test_x))-1

# Calculate ROC
lda_roc <- roc(test_y, pred_lda)
nn_roc <- roc(test_y, pred_nn)
pred_knn <- as.numeric(pred_knn)-1
knn_roc <- roc(test_y, pred_knn)
rf_roc <- roc(test_y, pred_rf)

# The result
par(mfrow=c(2,2))
plot.roc(lda_roc, main="LDA")
plot.roc(nn_roc, main="NN")
plot.roc(knn_roc, main="kNN")
plot.roc(rf_roc, main="RF")
