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

Hence, we choose both Recommeded and Rating as the target variables in this project.

<br> 

### 3.2 Review Text Processing
Libraries “tm” and “SnowballC” are used.
Summary of actions took in the stage of processing review text:
1. Text documentation collection is done by using VCorpus(VectorSource()).
2. Short words with length smaller than or equals to 2 in the review text are removed.
3. Extra whitespaces are removed in the review text.
4. All the words in the review text are converted to lowercase.
5. Numeric characters in the review text are removed.
6. Punctuations in the review text are removed.
7. English stopwords in the review text are removed
8. A clean text function which removes mentions, urls, emojis, numbers, punctuation
etc. is applied towards the review text.
9. Stemming is applied towards review text to reduce words to their root form.
10. All the review text is then converted to Document Term Matrix and weighted with
TF-IDF.
11. Sparse words, which are the lower occurrence words in the Document Term Matrix
are removed to reduce the dimensionality of the matrix.
12. The Document Term Matrix eventually is saved as a data frame.


## 4.0 Data Understanding
### 4.1 Feature Importance

### 4.2 Term Frequency analysis


### 4.3 Exploratory Data Analysis

