#==============================================================
#                       CLASSIFICATION TREE
#
#                    TARGET VARIABLE: RECOMMENDED
#==============================================================


#==============================================================
#                          using rpart()
#==============================================================

#-------------------- Test Train Splitting --------------------
df_model = tfidf_df
df_model$Recommended = df$Recommended

set.seed(123)
df_tag0 = df_model[df_model$Recommended == "no", ]
df_tag1 = df_model[df_model$Recommended == "yes", ]
tag0_idx = sample(nrow(df_tag0), size=0.7*nrow(df_tag0))
tag1_idx = sample(nrow(df_tag1), size=0.7*nrow(df_tag1))
df.train = rbind(df_tag0[ tag0_idx,],df_tag1[ tag1_idx,])
df.test  = rbind(df_tag0[-tag0_idx,],df_tag1[-tag1_idx,])

#--------------------------- Model ---------------------------
#install.packages("rpart")
#install.packages("rpart.plot")

library(rpart) 
library(rpart.plot)

# create decision tree
tree_model = rpart(Recommended ~ ., data = df.train, method = "class")

printcp(tree_model)
print(tree_model)

dev.new()
plot(tree_model, main = "Decision Tree")
text(tree_model, use.n = TRUE, all = TRUE, cex = 0.8)

# Alternative plotting
dev.new()
prp(x = tree_model, extra = 2, main = "Decision Tree") 


# Train Performance
tree.train = predict(tree_model,df.train,type="class")
cf.mat_train = table(tree.train,df.train$Recommended)
train_performance = performance(cf.mat_train, "\nClassification Tree using rpart() Train Performance:")

# Test Performance
tree.pred = predict(tree_model,df.test,type="class")
cf.mat_test = table(tree.pred,df.test$Recommended)
test_performance = performance(cf.mat_test, "\nClassification Tree using rpart() Test Performance:")

#----------------------- Prune tree model -----------------------
# Examine the complexity plot
dev.new()
plotcp(tree_model)

#------------------------ Plot ROC Curve ----------------------------
#install.packages("pROC")
suppressWarnings(library(pROC))

# Get class probabilities using Platt scaling
tree_probs_train = predict(tree_model, newdata = df.train, type = "prob")
tree_probs_test = predict(tree_model, newdata = df.test, type = "prob")

# Create ROC objects
roc_train = roc(df.train$Recommended, tree_probs_train[, "no"])
roc_test = roc(df.test$Recommended, tree_probs_test[, "no"])

# Plot ROC curves
plot(roc_train, col = "blue", main = "ROC Curve for rpart()", print.auc = TRUE)
plot(roc_test, col = "red", add = TRUE, print.auc = TRUE, lty = 2)
legend("bottomright", c("Train", "Test"), col = c("blue", "red"), lty = c(1, 2))

auc_train = auc(roc_train)
auc_test = auc(roc_test)
cat("Train data: ", auc_train, "\n")
cat("Test data: ", auc_test, "\n")





#==============================================================
#                          using tree()
#==============================================================
#-------------------- Test Train Splitting --------------------
df_model = tfidf_df
colnames(df_model)[colnames(df_model) == "next"] = "nex"
df_model$Recommended = df$Recommended

set.seed(123)
df_tag0 = df_model[df_model$Recommended == "no", ]
df_tag1 = df_model[df_model$Recommended == "yes", ]
tag0_idx = sample(nrow(df_tag0), size=0.7*nrow(df_tag0))
tag1_idx = sample(nrow(df_tag1), size=0.7*nrow(df_tag1))
df.train = rbind(df_tag0[ tag0_idx,],df_tag1[ tag1_idx,])
df.test  = rbind(df_tag0[-tag0_idx,],df_tag1[-tag1_idx,])


#--------------------------- Model --------------------------
#install.packages("tree")

library(tree)
tree_model_1 = tree(Recommended~., data=df.train)
summary(tree_model_1)

plot(tree_model_1)
text(tree_model_1,cex=0.8)
title("Classification tree using tree()")
print(tree_model_1)

# Train Performance
tree.train_1 = predict(tree_model_1,df.train,type="class")
cf.mat_train = table(tree.train_1,df.train$Recommended)
train_performance = performance(cf.mat_train, "\nClassification Tree using tree() Train Performance:")

# Test Performance
tree.pred_1 = predict(tree_model_1,df.test,type="class")
cf.mat = table(tree.pred_1,df.test$Recommended)
performance(cf.mat, "\nClassification Tree using tree() Test Performance")

#----------------------- Prune tree model -----------------------
seat_tree_cv = cv.tree(tree_model_1, FUN=prune.misclass) 

min_idx = max(which(seat_tree_cv$dev==min(seat_tree_cv$dev)))
num_terminal_nodes = seat_tree_cv$size[min_idx]

dev.new()
par(mfrow = c(1, 2))
plot(seat_tree_cv)
plot(seat_tree_cv$size, seat_tree_cv$dev / nrow(df.train), 
     type="b", xlab=paste("Tree Size (minimum @", num_terminal_nodes, ")"),
     ylab="CV Misclassification Rate")

#----------------- Prune with minimum index  ------------------
pruneTree_model_1 = prune.misclass(tree_model_1,best=num_terminal_nodes)

summary(pruneTree_model_1)
print(pruneTree_model_1)

# Compare the pruned tree and the original tree
dev.new()
par(mfrow = c(1, 2))
plot(pruneTree_model_1)   
text(pruneTree_model_1,cex=0.8)
title("Pruned Tree")
plot(tree_model_1)
text(tree_model_1,cex=0.8)
title("Original Tree")

# Train Performance
pruneTree.train_1 = predict(pruneTree_model_1,df.train,type="class")
cf.mat_train = table(pruneTree.train_1,df.train$Recommended)
train_performance = performance(cf.mat_train, "\nPrune Classification Tree using tree() Train Performance:")

# Test Performance
pruneTree.pred_1 = predict(pruneTree_model_1,df.test,type="class")
cf.mat.prune = table(pruneTree.pred_1,df.test$Recommended)
performance(cf.mat.prune, "\nPrune Classification Tree using tree() Test Performance:")


#------------------------ Plot ROC Curve ----------------------------
#install.packages("pROC")
suppressWarnings(library(pROC))

# Predict probabilities for the positive class "No"
tree_probs_train = predict(pruneTree_model_1, df.train, probability = TRUE)[, "no"]
tree_probs_test = predict(pruneTree_model_1, df.test, probability = TRUE)[, "no"]

# Create ROC objects
roc_train = roc(df.train$Recommended, tree_probs_train)
roc_test = roc(df.test$Recommended, tree_probs_test)

# Plot ROC curves
dev.new()
plot(roc_train, col = "blue", main = "ROC Curve for tree()", print.auc = TRUE)
plot(roc_test, col = "red", add = TRUE, print.auc = TRUE, lty = 2)
legend("bottomright", c("Train", "Test"), col = c("blue", "red"), lty = c(1, 2))

auc_train = auc(roc_train)
auc_test = auc(roc_test)
cat("Train data: ", auc_train, "\n")
cat("Test data: ", auc_test, "\n")




#==============================================================
#                           OVERSAMPLED
#
#                          using rpart()
#==============================================================

#--------------------------- Train Test -----------------------
df_model = oversampled_data
colnames(df_model)[colnames(df_model) == "next"] = "nex"
df_model$Recommended = oversampled_data$Recommended

df.test = data.frame(test)
df.test$Recommended = Y.test

#--------------------------- Model --------------------------
#install.packages("rpart")
#install.packages("rpart.plot")

library(rpart) #for fitting decision trees
library(rpart.plot) #for plotting decision trees

# create decision tree
tree_model = rpart(Recommended ~ ., data = df_model, method = "class")

printcp(tree_model)
print(tree_model)

plot(tree_model, main = "Classification Tree using rpart()")
text(tree_model, use.n = TRUE, all = TRUE, cex = 0.8)

## Alternative ploting
dev.new()
prp(x = tree_model, extra = 2, main = "Classification Tree using rpart()") 

# Train Performance
tree.train = predict(tree_model,df_model,type="class")
cf.mat_train = table(tree.train,df_model$Recommended)
train_performance = performance(cf.mat_train, "\nClassification Tree using rpart() Train Performance:")

# Test performance
tree.pred = predict(tree_model,df.test,type="class")
cf.mat_test = table(tree.pred,df.test$Recommended)
test_performance = performance(cf.mat_test, "\nClassification Tree using rpart() Test Performance:")

#----------------------- Prune tree model -----------------------
# Examine the complexity plot
dev.new()
plotcp(tree_model)





#==============================================================
#                           OVERSAMPLED
#
#				    using tree()
#==============================================================
#--------------------------- Train Test -----------------------
df.test = data.frame(test)
df.test$Recommended = Y.test

#------------------------------ Model -------------------------
#install.packages("tree")

library(tree)
tree_model_1 = tree(Recommended~., data=oversampled_data)
summary(tree_model_1)

plot(tree_model_1)
text(tree_model_1,cex=0.8)
title("Classification tree using tree()")
print(tree_model_1)

# Train Performance
tree.train_1 = predict(tree_model_1,oversampled_data,type="class")
cf.mat_train = table(tree.train_1,oversampled_data$Recommended)
train_performance = performance(cf.mat_train, "\nClassification Tree using tree() Train Performance:")

# Test Performance
tree.pred_1 = predict(tree_model_1,df.test,type="class")
cf.mat = table(tree.pred_1,df.test$Recommended)
performance(cf.mat, "\nClassification Tree using tree() Test Performance")

#----------------------- Prune tree model -----------------------
seat_tree_cv = cv.tree(tree_model_1, FUN=prune.misclass) 

min_idx = max(which(seat_tree_cv$dev==min(seat_tree_cv$dev)))
num_terminal_nodes = seat_tree_cv$size[min_idx]

dev.new()
par(mfrow = c(1, 2))
plot(seat_tree_cv)
plot(seat_tree_cv$size, seat_tree_cv$dev / nrow(oversampled_data), 
     type="b", xlab=paste("Tree Size (minimum @", num_terminal_nodes, ")"),
     ylab="CV Misclassification Rate")

#----------------------- Prune with index 6 ---------------------
pruneTree_model_1 = prune.misclass(tree_model_1,best=6)
summary(pruneTree_model_1)
print(pruneTree_model_1)

# Compare the pruned tree and the original tree
dev.new()
par(mfrow = c(1, 2))
plot(pruneTree_model_1)   
text(pruneTree_model_1,cex=0.8)
title("Pruned Tree with 6")
plot(tree_model_1)
text(tree_model_1,cex=0.8)
title("Original Tree")

# Train Performance
pruneTree.train_1 = predict(pruneTree_model_1,oversampled_data,type="class")
cf.mat_train = table(pruneTree.train_1,oversampled_data$Recommended)
train_performance = performance(cf.mat_train, "\nPrune Classification Tree using tree() Train Performance:")

# Test Performance
pruneTree.pred_1 = predict(pruneTree_model_1,df.test,type="class")
cf.mat.prune = table(pruneTree.pred_1,df.test$Recommended)
performance(cf.mat.prune, "\nPrune Classification Tree using tree() Test Performance:")


#------------------------ Plot ROC Curve ----------------------------
#install.packages("pROC")
suppressWarnings(library(pROC))

# Predict probabilities for the positive class "No"
tree_probs_train = predict(pruneTree_model_1, oversampled_data, probability = TRUE)[, "no"]
tree_probs_test = predict(pruneTree_model_1, df.test, probability = TRUE)[, "no"]

# Create ROC objects
roc_train = roc(oversampled_data$Recommended, tree_probs_train)
roc_test = roc(df.test$Recommended, tree_probs_test)

# Plot ROC curves
dev.new()
plot(roc_train, col = "blue", main = "ROC Curve for tree()", print.auc = TRUE)
plot(roc_test, col = "red", add = TRUE, print.auc = TRUE, lty = 2)
legend("bottomright", c("Train", "Test"), col = c("blue", "red"), lty = c(1, 2))

auc_train = auc(roc_train)
auc_test = auc(roc_test)
cat("Train data: ", auc_train, "\n")
cat("Test data: ", auc_test, "\n")



