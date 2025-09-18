#==============================================================
#                PRINCIPAL COMPONENT ANALYSIS(PCA) 
#
#                   TARGET VARIABLE: RECOMMENDED         
#==============================================================

#---------------------- Train Test Spliting -------------------
y = df$Recommended
X = tfidf_df

# Determine the number of samples for the training set
train_size = floor(0.7 * nrow(X))

# Split the indices into training and testing sets directly
train_indices = seq_len(train_size)
test_indices = (train_size + 1):nrow(X)

# Create the training and testing sets based on the indices
X_train = X[train_indices, ]
X_test = X[test_indices, ]
y_train = y[train_indices]
y_test = y[test_indices]

# Print the dimensions of the split datasets
cat("X_train shape:", dim(X_train), "\n")
cat("X_test shape:", dim(X_test), "\n")
cat("y_train shape:", length(y_train), "\n")
cat("y_test shape:", length(y_test), "\n")

#------------------------------ Model -------------------------
#install.packages("caret")
#install.packages("ggplot2")

library(caret)
library(ggplot2)

# Perform PCA on the feature matrix
pca_result = prcomp(X_train, center = TRUE, scale. = TRUE)

# PCA results
print(pca_result)

print(summary(pca_result))

plot(pca_result)

# PVEs
pve= pca_result$sdev^2 / sum(pca_result$sdev^2)
cpve = cumsum(pve)

#Plot of the PVEs
dev.new()
plot(pve,type="o",ylab="PVE",xlav="Principal Component",col="blue")
dev.new()
plot(cpve,type="o",ylab="Cumulative PVE", xlab="Principal Component")

# Number of components that explain 95% of variance
num_components = which(cpve >= 0.95)[1]
# Keep only the selected number of components
X_train_pca = as.data.frame(pca_result$x[, 1:num_components])
# Dimensions of the PCA-transformed data
cat("X_train_pca shape:", dim(X_train_pca), "\n")

# PCA-transformed features and the target variable
pca_df = cbind(X_train_pca, Recommended = y_train)

# scatter plot of the first two principal components
dev.new()
ggplot(pca_df, aes(x = PC1, y = PC2, color = Recommended)) +
  geom_point() +
  labs(x = "Principal Component 1", y = "Principal Component 2", color = "Recommended") +
  ggtitle("Scatter Plot of PCA-transformed Data")


#==============================================================
#                          OVERSAMPLED
#==============================================================

#------------------------------ Model -------------------------
#install.packages("caret")
#install.packages("ggplot2")

library(caret)
library(ggplot2)

# Perform PCA on the feature matrix
pca_result = prcomp(X_train_rose, center = TRUE, scale. = TRUE)

# PCA results
print(pca_result)

print(summary(pca_result))

plot(pca_result)

# PVEs
pve= pca_result$sdev^2 / sum(pca_result$sdev^2)
cpve = cumsum(pve)

#Plot of the PVEs
dev.new()
plot(pve,type="o",ylab="PVE",xlav="Principal Component",col="blue")
dev.new()
plot(cpve,type="o",ylab="Cumulative PVE", xlab="Principal Component")

# Number of components that explain 95% of variance
num_components = which(cpve >= 0.95)[1]
# Keep only the selected number of components
X_train_pca = as.data.frame(pca_result$x[, 1:num_components])
# Dimensions of the PCA-transformed data
cat("X_train_pca shape:", dim(X_train_pca), "\n")

# PCA-transformed features and the target variable
pca_df = cbind(X_train_pca, Recommended = y_train)

# scatter plot of the first two principal components
dev.new()
ggplot(pca_df, aes(x = PC1, y = PC2, color = Recommended)) +
  geom_point() +
  labs(x = "Principal Component 1", y = "Principal Component 2", color = "Recommended") +
  ggtitle("Scatter Plot of PCA-transformed Data")
