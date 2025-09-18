#==============================================================
#                     SUPPORT VECTOR MACHINE
#
#                    TARGET VARIABLE: RATING
# 
#    **Warning: It might take 5 - 10 minutes to execute
#==============================================================

#---------------------- Train Test Spliting -------------------
set.seed(123)
n = length(df$Rating) 
idx.train = sample(1:n, size = n * 0.7, replace = FALSE)
idx.test = setdiff(1:n, idx.train)

train = as.matrix(dtm[idx.train,])  
test  = as.matrix(dtm[idx.test,])
Y.train = df$Rating [idx.train]
Y.test = df$Rating [idx.test]

df.train = as.data.frame(train)
df.train$Rating = Y.train
df.test = as.data.frame(test)
df.test$Rating = Y.test

#------------------------------ Model -------------------------
#install.packages("e1071")
library(e1071)

svm_model = svm(Rating ~., data = df.train, type = "C-classification", 
	cost=1, kernel = "linear", scale = FALSE)

# Try with other cost values
#svm_model = svm(Rating ~., data = df.train, type = "C-classification", 
#	cost=50, kernel = "linear", scale = FALSE)

print(svm_model)

# Train Performance
svm.train = predict(svm_model,df.train)
cf.mat = table(svm.train,df.train$Rating)
print(cf.mat)
cat("Accuracy = ", sum(diag(cf.mat))/sum(cf.mat), "\n")

# Test Performance
svm.predict = predict(svm_model,df.test)
cf.mat = table(svm.predict,df.test$Rating)
print(cf.mat)
cat("Accuracy = ", sum(diag(cf.mat))/sum(cf.mat), "\n")
