# Women_E-Commerce_Clothing_Review
## 1.0 Introduction
The dataset of Women's E-Commerce Clothing Review revolves around the reviews written by customers. 
The dataset is contributed by Nick Brooks who is currently a Machine Learning Engineer in London. 
The purpose of this dataset is to understand the correlation of different variables in customer reviews on a women's clothing e-commerce, and to classify each review whether it recommends the reviewed product or not and whether it consists of positive, negative, or neutral sentiment.

Data Source: https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews

In this project, the target variable are Rating and Recommended indicator.
Besides, both unsupervised and supervised learning techniques are used:
1. Unsupervised Learning:
   - Exploratory Data Analysis (EDA)
   - Principal Component Analysis (PCA)
2. Supervised learning:
   - Naive Bayes
   - Logistic Regression
   - Classification Tree
   - Support Vector Machine (SVM)
   - Random Forest

<br>

## 2.0 Data Dictionary
| No | Feature | Description | Values | Unique Count |
| ---| --- | --- | --- | --- | 
| 1. | Clothing ID | Integer categorical variable that refers to the specific piece being reviewed. | 0-1,205 | 1,172 |
| 2. | Age | Positive integer variable of the reviewer’s age. | 18-99 | 77 |
| 3. | Tittle | String variable for the title of the review | (String Text) | 13,984 |
| 4. | Review Text | String variable for the review body | (String Text) | 22,621 |
| 5. | Rating | Positive ordinal integer variable for the product score granted by the customer from 1 (worst) to 5 (best) |1-5 | 5 |
| 6. | Recommended IND | Binary variable stating where the customer recommends the product where 1 is recommended, 0 is not recommended. | 0,1 | 2 |
| 7. | Positive Feedback Count | Positive integer documenting the number of other customers who found this review positive. | 0-122 | 82 |
| 8. | Division Name | Categorical name of the product high level division. | Intimates, General, General Petite | 3 |
| 9. | Department Name | Categorical name of the product department name. | Intimate, Dresses, Bottoms, Tops, Jackets, Trend | 6 |
| 10. | Class Name | Categorical name of the product class name. | Intimates, Dresses, Pants, Blouses, Knits, Outerwear, Lounge, Sweaters, Skirts, Fine gauge, Jackets, Swim, Sleep, Trend, Jeans, Legwear, Layering, Shorts | 20 |

<br>

## 3.0 Pre-Processing
### 3.1 Data Cleaning
1. Features “Recommended.IND” and “Review.Text” are renamed with
“Recommended” and “Review” respectively.
   > colnames(df)[colnames(df) == "Recommended.IND"] = "Recommended"

   > colnames(df)[colnames(df) == "Review.Text"] = "Review"

2. Value in feature “Recommended” is changed from 0 and 1 to “no” and “yes”
respectively.
   > df$Recommended = ifelse(df$Recommended == 0,"no","yes")

3. Data type of feature “Rating”, “Recommended”, “Division.Name”, “Department.Name”, “Class.Name” are changed from integer to categorical type.
   > col_fac = c("Rating", "Recommended", "Division.Name", "Department.Name", "Class.Name")

   > df[col_fac] = lapply(df[col_fac], factor)

4. Remove unnecessary columns 
   > df$Title = NULL

   > df$Clothing.ID = NULL

   > df$X = NULL
   
5. Rows with missing values and empty string (“”) are removed.
   > df[apply(is.na(df),1,sum)>0,]

   > df = df[!df$Review == "",]

<br>

### 3.2 Determine Target Variable based on Feature Importance
Based on the result from Logistic Regression and ElasticNet graph, it can be obviously observed that the Rating variable is highly correlated to Recommended, which we initially pick as the target variable in this project. 
![Feature_Importance](images/Feature_Importance_Result.png)
<img src = "images/Elastic_Net.png" style="width:60%;height:auto;">

In the perspective of business, we could analyze the review in different views:
- Recommended (Yes/No): To evaluate whether the customers are recommending the clothes
- Rating (1-5): To evaluate the range of satisfaction towards the clothes

**Hence, we choose both Recommeded and Rating as the target variables in this project.**

<br>

### 3.3 Text Processing
Libraries “tm” and “SnowballC” are used.
Summary of actions took in the stage of processing review text:
1. Text documentation collection is done by using VCorpus(VectorSource()).
2. Short words with length smaller than or equals to 2 in the review text are removed.
   > remove_short_words = function(x) {
   > words = unlist(strsplit(x, "\\s+"))
   > words = words[nchar(words) > 2]
   > paste(words, collapse = " ")}

   > corpus = tm_map(corpus, content_transformer(remove_short_words))
3. Extra whitespaces are removed in the review text.
   > corpus = tm_map(corpus, stripWhitespace) 
4. All the words in the review text are converted to lowercase.
   > corpus = tm_map(corpus, content_transformer(tolower))
5. Numeric characters in the review text are removed.
   > corpus = tm_map(corpus, removeNumbers)
6. Punctuations in the review text are removed.
   > corpus = tm_map(corpus, removePunctuation)
7. English stopwords (eg. articles:a, an, the; conjunctions:and,but,or) in the review text are removed.
    > corpus = tm_map(corpus, removeWords, stopwords("en"))
8. A clean text function which removes mentions, urls, emojis, numbers, punctuation etc. is applied towards the review text.
    > clean_text = function(text) {
    > text = gsub("@\\w+", "", text)
    > text = gsub("https?://.+", "", text)
    > text = gsub("\\d+\\w*\\d*", "", text)
    > text = gsub("#\\w+", "", text)
    > text = gsub("[^\x01-\x7F]", "", text)
    > text = gsub("[[:punct:]]", " ", text)
    > return(text)}

    > corpus = tm_map(corpus, content_transformer(clean_text))
9. Stemming is applied towards review text to reduce words to their root form.
    > corpus = tm_map(corpus, stemDocument, language="english")
10. All the review text is then converted to Document Term Matrix and weighted with TF-IDF.
    > dtm = DocumentTermMatrix(corpus, control = list(weighting = weightTfIdf))
11. Sparse words, which are the lower occurrence words in the Document Term Matrix are removed to reduce the dimensionality of the matrix.
    > dtm = removeSparseTerms(dtm, .995)
12. The Document Term Matrix eventually is saved as a data frame.
    > tfidf_df = as.data.frame(as.matrix(dtm))

<br>

## 4.0 Data Understanding
### 4.1 Term Frequency analysis
<p align="Center"><img src = "images/Most_Frequent_Word.png" style="width:60%;height:auto;"><br>
Bar Chart of the Top 20 Most Frequent Words</p>

We could interpret the following from this bar chart:
- The most frequently occurring word is “dress”. This indicates customers give a lot offeedback on the dress of the online clothing store.
- “love”, “fit” and “size” are the next three most frequently occurring words, which indicate that most people possibly love and feel good about the fit and size of the clothes.
- We could see that positive words like “love”, “like”, “great”, “perfect”, “nice” appearreally frequently, indicating most reviews could be positive ones.

Next, a word cloud is plotted to show a better visualisation on frequent words in the dataset.
<p align="Center"><img src = "images/Word_Cloud.png" style="width:50%;height:auto;"></p>

#### **Word Association**
Correlation able to show whether and how strongly pairs of variables are related. This method can be used effectively to determine which words are most frequently used in association with the most common terms in the survey responses, which aids in understanding the context around these words. (Mhatre, 2020).

For example, we do a word association analysis for the top 3 most frequent terms
> findAssocs(tdm, terms = c("dress","fit","love"), corlimit = 0.10)
<p><img src = "images/Word_Association_Example.png" style="width:50%;height:auto;"></p>

- The output indicates that “slip”, “wed”, “knee”, “can” occur within the range of 11% to 16% of the time with the word “dress”. We could interpret it to mean that probably a lot of people comment on the slip dress. 
- Similarly, the root of the words “perfect”, “size”, “loos”(root for word “loose”), “well” are highly correlated with the word “fit”. This indicates that most responses are saying that the size of the clothes is perfect, well and true and maybe some find it to be loose.

Hence, let us plot the word assoctiation network for the Top 20 Frequent Words.
<p align="Center"><img src = "images/Word_Association_Network.png" style="width:60%;height:auto;"><br>
Word Association Network of the Top 20 Most Frequent Words</p>

- The thicker the edges, the stronger the correlation between the words.
- We could found that:
   - The mojor discussion of the customer is about size, as words such as fit, true, usual, normal, order, small, medium, chest, waist, lbs are tightly clustered.
   - Majority of the customer review the clothing (probably is the glove) is really look like the model (probable same as the picture). However, the thin edge with the word "dont" suggest there are fewer negative comments.
   - Customers frequently discuss colors, often with positive connotations (vibrant, rich, bright).

<br>

### 4.2 Sentiment Analysis


### 4.3 Emotional Analysis

### 4.3 Exploratory Data Analysis



### References
Mhatre. S. (2020). Text Mining and Sentiment Analysis: Analysis with R. Redgate. https://www.red-gate.com/simple-talk/databases/sql-server/bi-sql-server/text-mining-a nd-sentiment-analysis-with-r/

