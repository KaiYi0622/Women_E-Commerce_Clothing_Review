#==============================================================
#                      Data Preprocessing
#==============================================================

#------------------------ Read dataset ------------------------
df = read.csv("Womens Clothing E-Commerce Reviews.csv")
names(df)
sapply(df,class)
dim(df)
head(df)

#==============================================================
#                        Data Cleaning
#==============================================================

# Rename 'Recommended IND' column to 'Recommended'
# and 'Review.Text' column to 'Review'
colnames(df)[colnames(df) == "Recommended.IND"] = "Recommended"
colnames(df)[colnames(df) == "Review.Text"] = "Review"

# change df$Recommended to "yes" and "no"
df$Recommended = ifelse(df$Recommended == 0,"no","yes")

# change data type from integer to categorical
col_fac = c("Rating", "Recommended", "Division.Name", "Department.Name", "Class.Name")
df[col_fac] = lapply(df[col_fac], factor)
df$Title = NULL
df$Clothing.ID = NULL
df$X = NULL

# Remove rows with missing values (NA = Not Available)
df[apply(is.na(df),1,sum)>0,]
df = na.omit(df)
# Remove rows of review with empty string ("")
sum(df$Review == "") # number of rows with empty strings
df = df[!df$Review == "",]
dim(df)      # check the dimension again to see if some rows are removed
summary(df)


#==============================================================
# 			                FEATURE IMPORTANCE
# 			               LOGISTIC REGRESSION
#
# 		            TARGET VARIABLE: RECOMMENDED
#
#==============================================================
# OBJECTIVE: To determine target variable

# Remove rows with empty strings for division, department, class features
sum(df$Division.Name == "")
sum(df$Department.Name == "")
sum(df$Class.Name == "")
all(df[df$Department.Name == "", ] == df[df$Class.Name == "", ])
all(df[df$Department.Name == "", ] == df[df$Division.Name == "", ])
df = df[df$Department.Name != "",]
col_fac = c("Division.Name", "Department.Name", "Class.Name")
df[col_fac] = lapply(df[col_fac], factor)
summary(df)
# remove Review column
df_filtered = df[, -c(2,8)]
head(df_filtered)
set.seed(123)
rec_0 = df_filtered[df_filtered$Recommended=="no", ]
rec_1 = df_filtered[df_filtered$Recommended=="yes", ]
rec0_idx = sample(nrow(rec_0), size=round(0.7*nrow(rec_0)))
rec1_idx = sample(nrow(rec_1), size=round(0.7*nrow(rec_1)))
df.train = rbind(rec_0[ rec0_idx,], rec_1[ rec1_idx,])
df.test = rbind(rec_0[-rec0_idx,], rec_1[-rec1_idx,])
rm(rec_0)
rm(rec_1)
rm(rec0_idx)
rm(rec1_idx)
log_reg_model = glm(Recommended ~., data = df.train, family = binomial)
summary(log_reg_model)

# -------------------------------------------------------------
# 			                using ElasticNet
# -------------------------------------------------------------
library(glmnet) # install.packages("glmnet")
X = model.matrix(Recommended ~., data=df.train)[,-1]
glmmod = glmnet(x=X, y=df.train$Recommended , alpha=1, family=binomial, lambda=1e-5)
print(coef(glmmod))
glmmod = glmnet(x=X, y=df.train$Recommended, alpha=1, family="binomial")
plot(glmmod, lwd=5) # The x-axis defaults to L1-Norm
plot(glmmod, lwd=5, xvar="lambda")
legend("topright", legend = colnames(X), col = 1:length(coef(glmmod)), lwd = 4,
cex = 0.8)



#==============================================================
#                    Review Text Processing
#==============================================================
#install.packages("tm")
#install.packages("SnowballC")

library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(df$Review))

# Keep words with length > 2
remove_short_words = function(x) {
  words = unlist(strsplit(x, "\\s+"))
  words = words[nchar(words) > 2]  
  paste(words, collapse = " ")
}
# Remove mentions, urls, emojis, numbers, punctuations, etc
clean_text = function(text) {
  text = gsub("@\\w+", "", text)
  text = gsub("https?://.+", "", text)
  text = gsub("\\d+\\w*\\d*", "", text)
  text = gsub("#\\w+", "", text)
  text = gsub("[^\x01-\x7F]", "", text)
  text = gsub("[[:punct:]]", " ", text)
  return(text)
}
corpus = tm_map(corpus, content_transformer(remove_short_words))
corpus = tm_map(corpus, stripWhitespace) 
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords("en"))
corpus = tm_map(corpus, content_transformer(clean_text))
corpus = tm_map(corpus, stemDocument, language="english")

df$clean_text = sapply(corpus, as.character) # Check the review text after cleaning

dtm = DocumentTermMatrix(corpus, control = list(
  weighting = weightTfIdf
))
dtm = removeSparseTerms(dtm, .995)
inspect(dtm)

tfidf_df = as.data.frame(as.matrix(dtm))

#==============================================================
#                     Term Frequency Analysis
#==============================================================
library(dplyr)
library(ggplot2)
tdm = TermDocumentMatrix(corpus, control = list())
inspect(tdm)

term_freq = rowSums(as.matrix(tdm))
term_freq = subset(term_freq, term_freq >= 20)
df_term = data.frame(term = names(term_freq), freq = term_freq)

# Get the top 20 frequent words
top_words = df_term[order(df_term$freq, decreasing = TRUE), ][1:20,]

# Plot word frequency
ggplot(top_words, aes(x = reorder(term, freq), y = freq, fill = freq)) +
  geom_bar(stat = "identity") +
  scale_colour_gradientn(colors = terrain.colors(10)) +
  xlab("Terms") +
  ylab("Count") +
  coord_flip() +
  ggtitle("Top 20 Most Frequent Words") +
  theme_minimal()

# Create Wordcloud
# Option 1
#install.packages("wordcloud2")
library(wordcloud2)
wordcloud2(df_term, color = "random-dark", backgroundColor = "white")
# Option 2
#install.packages("wordcloud")
library(wordcloud)
library(RColorBrewer)
wordcloud(words = df_term$term, freq = df_term$freq, min.freq = 5,
          max.words=100, random.order=FALSE, rot.per=0.40, 
          colors=brewer.pal(8, "Dark2"))


# Find associations 
findAssocs(tdm, terms = c("dress","fit","love"), corlimit = 0.10)	

# Sentiment 
library(syuzhet)
syuzhet_vector = get_sentiment(d.f$clean_text, method="syuzhet")
head(syuzhet_vector)
# see summary statistics of the vector
summary(syuzhet_vector)
# bing
bing_vector = get_sentiment(d.f$clean_text, method="bing")
head(bing_vector)
summary(bing_vector)
#affin
afinn_vector = get_sentiment(d.f$clean_text, method="afinn")
head(afinn_vector)
summary(afinn_vector)

#compare the first row of each vector using sign function
rbind(
  sign(head(syuzhet_vector)),
  sign(head(bing_vector)),
  sign(head(afinn_vector))
)


# Emotion
# Takes a long time to execute
d = get_nrc_sentiment(d.f$clean_text)
# to see top 10 lines of the get_nrc_sentiment dataframe
head (d,10)

#transpose
td = data.frame(t(d))
#The function rowSums computes column sums across rows for each level of a grouping variable.
td_new = data.frame(rowSums(td[2:253]))
#Transformation and cleaning
names(td_new)[1] = "count"
td_new = cbind("sentiment" = rownames(td_new), td_new)
rownames(td_new) = NULL
td_new2 = td_new[1:8,]
#Plot One - count of words associated with each sentiment
quickplot(sentiment, data=td_new2, weight=count, geom="bar", fill=sentiment, ylab="count")+ggtitle("Survey sentiments")

#Plot two - count of words associated with each sentiment, expressed as a percentage
barplot(
  sort(colSums(prop.table(d[, 1:8]))), 
  horiz = TRUE, 
  cex.names = 0.7, 
  las = 1, 
  main = "Emotions in Text", xlab="Percentage"
)

#==============================================================
#                          Performance
#==============================================================
performance = function(xtab, desc=""){
    cat("\n", desc,"\n", sep="")
    print(xtab)

    ACR = sum(diag(xtab))/sum(xtab)
    CI  = binom.test(sum(diag(xtab)), sum(xtab))$conf.int
    cat("\n        Accuracy :", ACR)
    cat("\n          95% CI : (", CI[1], ",", CI[2], ")\n")

    if(nrow(xtab)>2){
        # e1071's classAgreement() in matchClasses.R
        # Ref: https://stats.stackexchange.com/questions/586342/measures-to-compare-classification-partitions
        n  = sum(xtab)
        ni = apply(xtab, 1, sum)
        nj = apply(xtab, 2, sum)
        p0 = sum(diag(xtab))/n
        pc = sum(ni * nj)/n^2
        Kappa = (p0 - pc)/(1 - pc)
        cat("\n           Kappa :", Kappa, "\n")
        cat("\nStatistics by Class:\n")
        # Levels of the actual data
        lvls = dimnames(xtab)[[2]]
        sensitivity = c()
        specificity = c()
        ppv         = c()
        npv         = c()
        for(i in 1:length(lvls)) {
            sensitivity[i] = xtab[i,i]/sum(xtab[,i])
            specificity[i] = sum(xtab[-i,-i])/sum(xtab[,-i])
            ppv[i]         = xtab[i,i]/sum(xtab[i,])
            npv[i]         = sum(xtab[-i,-i])/sum(xtab[-i,])
        }
        b = data.frame(rbind(sensitivity,specificity,ppv,npv))
        names(b) = lvls
        print(b)
    } else {
         #names(dimnames(xtab)) = c("Prediction", "Actual")
         TPR = xtab[1,1]/sum(xtab[,1]); TNR = xtab[2,2]/sum(xtab[,2])
         PPV = xtab[1,1]/sum(xtab[1,]); NPV = xtab[2,2]/sum(xtab[2,])
         FPR = 1 - TNR                ; FNR = 1 - TPR
         # https://standardwisdom.com/softwarejournal/2011/12/confusion-matrix-another-single-value-metric-kappa-statistic/
         RandomAccuracy = (sum(xtab[,2])*sum(xtab[2,]) +
           sum(xtab[,1])*sum(xtab[1,]))/(sum(xtab)^2)
         Kappa = (ACR - RandomAccuracy)/(1 - RandomAccuracy)
         cat("\n           Kappa :", Kappa, "\n")
         cat("\n     Sensitivity :", TPR)
         cat("\n     Specificity :", TNR)
         cat("\n  Pos Pred Value :", PPV)
         cat("\n  Neg Pred Value :", NPV)
         cat("\n             FPR :", FPR)
         cat("\n             FNR :", FNR, "\n")
         cat("\n'Positive' Class :", dimnames(xtab)[[1]][1], "\n")
    }
}


#==============================================================
#                        Oversampling
#==============================================================
# Check class distribution 
class_distribution = table(df$Recommended)
barplot(class_distribution, main="Class Distribution", xlab="Class", ylab="Frequency", col="blue")

set.seed(123)
n = length(df$Recommended) 
idx.train = sample(1:n, size = n * 0.7, replace = FALSE)
idx.test = setdiff(1:n, idx.train)

train = as.matrix(dtm[idx.train,])  
test  = as.matrix(dtm[idx.test,])
Y.train = df$Recommended[idx.train]
Y.test = df$Recommended[idx.test]

# Random Oversampling using ROSE
#install.packages("ROSE")
library(ROSE)
oversampled_data = ovun.sample(Recommended ~ ., data = data.frame(train, Recommended = Y.train), method = "both", seed = 123, N = length(Y.train))$data
oversampled_data$Recommended = relevel(oversampled_data$Recommended , ref = "no")

# Check class distribution after oversampling
class_distribution_rose = table(oversampled_data$Recommended)
barplot(class_distribution_rose, main="Class Distribution after Oversampling", xlab="Class", ylab="Frequency", col="blue")

X_train_rose = as.matrix(oversampled_data[, -ncol(oversampled_data)])  # Exclude the last column (Recommended)
y_train_rose = oversampled_data$Recommended

