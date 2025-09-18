#==============================================================
#                          RANDOM FOREST
#                   
#                   TARGET VARIABLE: RECOMMENDED
#
#      **WARNING**: It required around 5 minutes to execute
#==============================================================

#-------------------- Test Train Splitting --------------------
df_model = tfidf_df
df_model$Recommended = df$Recommended
colnames(df_model)[colnames(df_model) == "next"] = "nex"

set.seed(123)
df_tag0 = df_model[df_model$Recommended == "no", ]
df_tag1 = df_model[df_model$Recommended == "yes", ]
tag0_idx = sample(nrow(df_tag0), size=0.7*nrow(df_tag0))
tag1_idx = sample(nrow(df_tag1), size=0.7*nrow(df_tag1))
df.train = rbind(df_tag0[ tag0_idx,],df_tag1[ tag1_idx,])
df.test  = rbind(df_tag0[-tag0_idx,],df_tag1[-tag1_idx,])

#------------------------------ Model -------------------------
#install.packages("randomForest")
library(randomForest)

# Training with Random forest model
rf_model = randomForest(Recommended ~. , data=df.train, ntree = 30)
print(rf_model)

# Train Performance
rf.train = predict(rf_model,df.train,type="class")
cf.mat = table(rf.train,df.train$Recommended)
test_performance = performance(cf.mat, "\nRandom Forest Train Performance:")

# Test Performance
rf.pred = predict(rf_model, df.test, type = "class")
cf.mat = table(rf.pred,df.test$Recommended)
test_performance = performance(cf.mat, "\nRandom Forest Test Performance:")

plot(rf_model)

dev.new()
varImpPlot(rf_model)


#------------------------ Plot ROC Curve ----------------------------
#install.packages("pROC")
suppressWarnings(library(pROC))

# Predict probabilities for the positive class "No"
tree_probs_train = predict(rf_model, df.train, type = "prob")[, "no"]
tree_probs_test = predict(rf_model, df.test, type = "prob")[, "no"]

# Create ROC objects
roc_train = roc(df.train$Recommended, tree_probs_train)
roc_test = roc(df.test$Recommended, tree_probs_test)

# Plot ROC curves
dev.new()
plot(roc_train, col = "blue", main = "ROC Curve for Random Forest", print.auc = TRUE)
plot(roc_test, col = "red", add = TRUE, print.auc = TRUE, lty = 2)
legend("bottomright", c("Train", "Test"), col = c("blue", "red"), lty = c(1, 2))

auc_train = auc(roc_train)
auc_test = auc(roc_test)
cat("Train data: ", auc_train, "\n")
cat("Test data: ", auc_test, "\n")
