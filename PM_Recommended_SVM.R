#==============================================================
#                       SUPPORT VECTOR MACHINE
#
#                    TARGET VARIABLE: RECOMMENDED
#
#	**Warning: It requires around 5-10 minutes to execute
#		     model with larger cost value
#==============================================================

#---------------------- Train Test Spliting -------------------
df_model = tfidf_df
df_model$Recommended = df$Recommended

set.seed(123)
# Determine the number of samples for the training set
train_size = floor(0.7 * nrow(df_model))

# Split the indices into training and testing sets directly
train_indices = seq_len(train_size)
test_indices = (train_size + 1):nrow(df_model)

df.train = df_model[train_indices, ]
df.test = df_model[test_indices, ]


#------------------------------ Model -------------------------
#install.packages("e1071")
library(e1071)

svm_model = svm(Recommended ~., data = df.train, type = "C-classification", 
	cost=1, kernel = "linear", scale = FALSE, probability = TRUE)

# Try with other cost values
#svm_model = svm(Recommended ~., data = df.train, type = "C-classification", 
#	cost=10, kernel = "linear", scale = FALSE, probability = TRUE)
#svm_model = svm(Recommended ~., data = df.train, type = "C-classification", 
#	cost=50, kernel = "linear", scale = FALSE, probability = TRUE)
 
print(svm_model)

# Train Performance
svm.predict = predict(svm_model,df.train)
cf.mat = table(svm.predict,df.train$Recommended)
performance(cf.mat, "\nSVM Model Train Performance:")

# Test Performance
svm.predict = predict(svm_model,df.test)
cf.mat = table(svm.predict,df.test$Recommended)
performance(cf.mat, "\nSVM Model Test Performance:")


#------------------------ Plot ROC Curve ----------------------------
#install.packages("pROC")
suppressWarnings(library(pROC))

# Predict probabilities for the positive class "No"
svm_probs_train = attributes(predict(svm_model, df.train, probability = TRUE))$probabilities[, "no"]
svm_probs_test = attributes(predict(svm_model, df.test, probability = TRUE))$probabilities[, "no"]

# Create ROC objects
roc_train = roc(df.train$Recommended, svm_probs_train)
roc_test = roc(df.test$Recommended, svm_probs_test)

# Plot ROC curves
plot(roc_train, col = "blue", main = "ROC Curve for SVM", print.auc = TRUE)
plot(roc_test, col = "red", add = TRUE, print.auc = TRUE, lty = 2)
legend("bottomright", c("Train", "Test"), col = c("blue", "red"), lty = c(1, 2))

auc_train = auc(roc_train)
auc_test = auc(roc_test)
cat("Train data: ", auc_train, "\n")
cat("Test data: ", auc_test, "\n")




#==============================================================
#                            OVERSAMPLED
#==============================================================
#--------------------------- Train Test -----------------------
df.test = data.frame(test)
df.test$Recommended = Y.test

#------------------------------ Model -------------------------
#install.packages("e1071")
library(e1071)
set.seed(123)

svm_model = svm(Recommended~., data = oversampled_data, type = "C-classification", 
	cost = 1, kernel = "linear", scale = FALSE, probability = TRUE)

# Try with other cost values
#svm_model = svm(Recommended~., data = oversampled_data, type = "C-classification", 
#	cost = 10, kernel = "linear", scale = FALSE, probability = TRUE)
#svm_model = svm(Recommended~., data = oversampled_data, type = "C-classification", 
#	cost = 50, kernel = "linear", scale = FALSE, probability = TRUE)

print(svm_model)

# Train Performance
svm.predict = predict(svm_model,oversampled_data)
cf.mat = table(svm.predict,oversampled_data$Recommended)
performance(cf.mat, "\nSVM Model Train Performance:")

# Test Performance
svm.predict = predict(svm_model,df.test)
cf.mat = table(svm.predict,df.test$Recommended)
performance(cf.mat, "\nSVM Model Test Performance:")


#------------------------ Plot ROC Curve ----------------------------
#install.packages("pROC")
suppressWarnings(library(pROC))

# Predict probabilities for the positive class "No"
svm_probs_train = attributes(predict(svm_model, oversampled_data, probability = TRUE))$probabilities[, "no"]
svm_probs_test = attributes(predict(svm_model, df.test, probability = TRUE))$probabilities[, "no"]

# Create ROC objects
roc_train = roc(oversampled_data$Recommended, svm_probs_train)
roc_test = roc(df.test$Recommended, svm_probs_test)

# Plot ROC curves
plot(roc_train, col = "blue", main = "ROC Curve for SVM", print.auc = TRUE)
plot(roc_test, col = "red", add = TRUE, print.auc = TRUE, lty = 2)
legend("bottomright", c("Train", "Test"), col = c("blue", "red"), lty = c(1, 2))

auc_train = auc(roc_train)
auc_test = auc(roc_test)
cat("Train data: ", auc_train, "\n")
cat("Test data: ", auc_test, "\n")

