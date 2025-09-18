#==============================================================
#                          RANDOM FOREST
#
#                    TARGET VARIABLE: RATING
#
#    **WARNING**: It required around 5 minutes to execute
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
colnames(df.train)[colnames(df.train) == "next"] = "nex"
df.test = as.data.frame(test)
df.test$Rating = Y.test
colnames(df.test)[colnames(df.test) == "next"] = "nex"



#------------------------------ Model -------------------------
#install.packages("randomForest")
library(randomForest)

set.seed(123)
# Training with Random forest model
rf_model = randomForest(Rating ~. , data=df.train, ntree = 100)
print(rf_model)

# Train Performance
rf.train = predict(rf_model,df.train,type="class")
cf.mat = table(rf.train,df.train$Rating)
print(cf.mat)
cat("Accuracy = ", sum(diag(cf.mat))/sum(cf.mat), "\n")

# Test Performance
rf.pred = predict(rf_model, df.test, type = "class")
cf.mat = table(rf.pred,df.test$Rating)
print(cf.mat)
cat("Accuracy = ", sum(diag(cf.mat))/sum(cf.mat), "\n")

plot(rf_model)

dev.new()
varImpPlot(rf_model)
