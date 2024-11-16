---
author: "Daniela Rios"
output:
  html_document:
    mathjax: true
    keep_md: true
    highlight: zenburn
    theme:  spacelab
  pdf_document:
always_allow_html: true
---




****

<span><h1 style = "font-family: verdana; font-size: 26px; font-style: normal; letter-spacing: 3px; background-color: #f8f9fa; color: #000000; border-radius: 100px 100px; text-align:center"> 🧠 Machine Learning Pipeline for Stroke Prediction </h1></span>

<b><span style='color:#E888BB; font-size: 16px;'> |</span> <span style='color:#000;'>Evaluation of Neural Network for Stroke Prediction</span> </b>

<div style="background-color:#b2f7ef; color:black; border-radius:5px; padding: 0.2em 0.5em; display: inline-block;"><strong>Introduction</strong></div>
<p style='color:#000;'>This pipeline aims to predict stroke occurrences based on patient data using a neural network model. The dataset includes several health-related attributes such as age, gender, glucose levels, and body mass index (BMI). The goal is to leverage these features to build a deep learning model that can accurately predict which patients are at higher risk of stroke.</p>

<div style="background-color:#b2f7ef; color:black; border-radius:5px; padding: 0.2em 0.5em; display: inline-block;"><strong>Data Loading and Cleaning</strong></div>
<p style='color:#000;'>The data is loaded and cleaned to prepare it for the neural network. This includes handling missing values, converting categorical variables to appropriate data types, and scaling numerical features. Missing values in numerical columns are handled using median imputation to retain data without introducing bias.</p>


```r
# Load necessary libraries
library(tidyverse)
library(caret)
library(nnet)

# Load the dataset
stroke_data <- read.csv("stroke.csv")

# Inspect the data
str(stroke_data)
```

```
'data.frame':	5110 obs. of  12 variables:
 $ id               : int  9046 51676 31112 60182 1665 56669 53882 10434 27419 60491 ...
 $ gender           : chr  "Male" "Female" "Male" "Female" ...
 $ age              : num  67 61 80 49 79 81 74 69 59 78 ...
 $ hypertension     : int  0 0 0 0 1 0 1 0 0 0 ...
 $ heart_disease    : int  1 0 1 0 0 0 1 0 0 0 ...
 $ ever_married     : chr  "Yes" "Yes" "Yes" "Yes" ...
 $ work_type        : chr  "Private" "Self-employed" "Private" "Private" ...
 $ Residence_type   : chr  "Urban" "Rural" "Rural" "Urban" ...
 $ avg_glucose_level: num  229 202 106 171 174 ...
 $ bmi              : chr  "36.6" "N/A" "32.5" "34.4" ...
 $ smoking_status   : chr  "formerly smoked" "never smoked" "never smoked" "smokes" ...
 $ stroke           : int  1 1 1 1 1 1 1 1 1 1 ...
```

```r
sum(stroke_data == "N/A", na.rm = TRUE) # Check for missing values
```

```
[1] 201
```

```r
# Preprocess the data
# Handling missing values using Median Imputation
numerical_columns <- c("age", "avg_glucose_level", "bmi")

# Convert 'bmi' to numeric, handling "N/A" values
stroke_data$bmi <- as.numeric(replace(stroke_data$bmi, stroke_data$bmi == "N/A", NA))

# Remove rows with any remaining non-numeric or infinite values in numerical columns
stroke_data <- stroke_data %>% filter(if_all(all_of(numerical_columns), ~ !is.na(.) & . != "N/A" & is.finite(.)))

# Check if numerical columns are empty after filtering
if (nrow(stroke_data) == 0) {
  stop("Numerical columns are empty after filtering. Please check your data.")
}

for (col in numerical_columns) {
  if (sum(stroke_data[[col]] == "N/A", na.rm = TRUE) > 0) {
    stroke_data[[col]][stroke_data[[col]] == "N/A"] <- median(as.numeric(stroke_data[[col]]), na.rm = TRUE)
  }
}

# Convert categorical variables to factors
stroke_data$gender <- as.factor(stroke_data$gender)
stroke_data$ever_married <- as.factor(stroke_data$ever_married)
stroke_data$work_type <- as.factor(stroke_data$work_type)
stroke_data$Residence_type <- as.factor(stroke_data$Residence_type)
stroke_data$smoking_status <- as.factor(stroke_data$smoking_status)
stroke_data$stroke <- as.factor(stroke_data$stroke)

# Apply Min-Max Scaling to numerical features
preproc <- preProcess(stroke_data[, numerical_columns], method = c("range"))
stroke_data[, numerical_columns] <- predict(preproc, stroke_data[, numerical_columns])

# Apply one-hot encoding for categorical variables
categorical_columns <- c("gender", "ever_married", "work_type", "Residence_type", "smoking_status")
categorical_columns <- categorical_columns[sapply(stroke_data[, categorical_columns], function(x) length(unique(x))) > 1]

if (length(categorical_columns) > 0) {
  dummy_model <- dummyVars(~ ., data = stroke_data[, categorical_columns], fullRank = TRUE)
  dummy_encoded <- predict(dummy_model, newdata = stroke_data) %>% as.data.frame()
  
  # Combine encoded categorical columns with the rest of the dataset
  stroke_data <- cbind(stroke_data %>% select(-all_of(categorical_columns)), dummy_encoded)
  colnames(stroke_data) <- make.names(colnames(stroke_data), unique = TRUE)
} else {
  warning("No categorical columns to encode.")
}
```

<div style="background-color:#b2f7ef; color:black; border-radius:5px; padding: 0.2em 0.5em; display: inline-block;"><strong>Splitting the Dataset</strong></div>
<p style='color:#000;'>The dataset is split into training, testing, and validation sets in a 70-20-10 ratio. This ensures the model has a significant portion of data to learn from, while keeping separate sets for unbiased evaluation and validation of its performance.</p>


```r
# Split the dataset into training, validation, and testing sets (70-20-10 split)
set.seed(123)
train_index <- createDataPartition(stroke_data$stroke, p = 0.7, list = FALSE)
train_data <- stroke_data[train_index, ]
remaining_data <- stroke_data[-train_index, ]

validation_index <- createDataPartition(remaining_data$stroke, p = 2/3, list = FALSE)
validation_data <- remaining_data[validation_index, ]
test_data <- remaining_data[-validation_index, ]
```

<div style="background-color:#b2f7ef; color:black; border-radius:5px; padding: 0.2em 0.5em; display: inline-block;"><strong>Model Preparation and Training</strong></div>
<p style='color:#000;'>The training data is prepared for the neural network model, with features and labels separated. The model is trained using the 'nnet' package, which provides a straightforward interface for training neural networks in R.</p>


```r
# Prepare the data for the neural network model
train_features <- train_data %>% select(-stroke)
train_labels <- train_data$stroke
validation_features <- validation_data %>% select(-stroke)
validation_labels <- validation_data$stroke
test_features <- test_data %>% select(-stroke)
test_labels <- test_data$stroke

# Build the neural network model using nnet
model <- nnet(
  stroke ~ .,
  data = train_data,
  size = 8,
  decay = 0.1,
  maxit = 200
)
```

```
# weights:  153
initial  value 3893.242304 
iter  10 value 607.340107
iter  20 value 607.311631
iter  30 value 607.294464
iter  40 value 607.289185
iter  50 value 607.276302
iter  60 value 606.753600
iter  70 value 538.481799
iter  80 value 495.151251
iter  90 value 491.647979
iter 100 value 490.823650
final  value 490.803746 
converged
```

<div style="background-color:#b2f7ef; color:black; border-radius:5px; padding: 0.2em 0.5em; display: inline-block;"><strong>Model Evaluation</strong></div>
<p style='color:#000;'>The trained model is evaluated using the validation and test datasets. Confusion matrices are used to evaluate the model's performance, indicating how well it predicted stroke occurrences compared to the true labels.</p>


```r
# Make predictions on the validation dataset
validation_predictions <- predict(model, validation_features, type = "class")

# Ensure predictions and labels are factors with the same levels
validation_predictions <- factor(validation_predictions, levels = levels(validation_labels))

# Evaluate the model on the validation dataset
validation_conf_matrix <- confusionMatrix(validation_predictions, validation_labels)
cat("
Validation Confusion Matrix:
")
```

```

Validation Confusion Matrix:
```

```r
print(validation_conf_matrix)
```

```
Confusion Matrix and Statistics

          Reference
Prediction   0   1
         0 940  42
         1   0   0
                                         
               Accuracy : 0.9572         
                 95% CI : (0.9426, 0.969)
    No Information Rate : 0.9572         
    P-Value [Acc > NIR] : 0.5409         
                                         
                  Kappa : 0              
                                         
 Mcnemar's Test P-Value : 2.509e-10      
                                         
            Sensitivity : 1.0000         
            Specificity : 0.0000         
         Pos Pred Value : 0.9572         
         Neg Pred Value :    NaN         
             Prevalence : 0.9572         
         Detection Rate : 0.9572         
   Detection Prevalence : 1.0000         
      Balanced Accuracy : 0.5000         
                                         
       'Positive' Class : 0              
                                         
```

```r
# Display formatted output for paper
cat("
Accuracy:", validation_conf_matrix$overall['Accuracy'], "
")
```

```

Accuracy: 0.9572301 
```

```r
cat("Sensitivity:", validation_conf_matrix$byClass['Sensitivity'], "
")
```

```
Sensitivity: 1 
```

```r
cat("Specificity:", validation_conf_matrix$byClass['Specificity'], "
")
```

```
Specificity: 0 
```

```r
# Make predictions on the test dataset
test_predictions <- predict(model, test_features, type = "class")

# Ensure predictions and labels are factors with the same levels
test_predictions <- factor(test_predictions, levels = levels(test_labels))

# Evaluate the model on the test dataset
test_conf_matrix <- confusionMatrix(test_predictions, test_labels)
cat("
Test Confusion Matrix:
")
```

```

Test Confusion Matrix:
```

```r
print(test_conf_matrix)
```

```
Confusion Matrix and Statistics

          Reference
Prediction   0   1
         0 470  20
         1   0   0
                                          
               Accuracy : 0.9592          
                 95% CI : (0.9377, 0.9749)
    No Information Rate : 0.9592          
    P-Value [Acc > NIR] : 0.5591          
                                          
                  Kappa : 0               
                                          
 Mcnemar's Test P-Value : 2.152e-05       
                                          
            Sensitivity : 1.0000          
            Specificity : 0.0000          
         Pos Pred Value : 0.9592          
         Neg Pred Value :    NaN          
             Prevalence : 0.9592          
         Detection Rate : 0.9592          
   Detection Prevalence : 1.0000          
      Balanced Accuracy : 0.5000          
                                          
       'Positive' Class : 0               
                                          
```

```r
# Display formatted output for paper
cat("
Accuracy:", test_conf_matrix$overall['Accuracy'], "
")
```

```

Accuracy: 0.9591837 
```

```r
cat("Sensitivity:", test_conf_matrix$byClass['Sensitivity'], "
")
```

```
Sensitivity: 1 
```

```r
cat("Specificity:", test_conf_matrix$byClass['Specificity'], "
")
```

```
Specificity: 0 
```

<div style="background-color:#b2f7ef; color:black; border-radius:5px; padding: 0.2em 0.5em; display: inline-block;"><strong>Conclusion</strong></div>
<p style='color:#000;'>The neural network model was trained and evaluated on the dataset to predict stroke occurrences. The performance metrics show that the model's accuracy and other evaluation measures need further optimization. This pipeline provides a foundation for predicting strokes, which could be further refined with additional data and feature engineering.</p>


```r
# Display model weights to assess variable importance
print(summary(model))
```

```
a 17-8-1 network with 153 weights
options were - entropy fitting  decay=0.1
  b->h1  i1->h1  i2->h1  i3->h1  i4->h1  i5->h1  i6->h1  i7->h1  i8->h1  i9->h1 
   0.00   -0.02    0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00 
i10->h1 i11->h1 i12->h1 i13->h1 i14->h1 i15->h1 i16->h1 i17->h1 
   0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00 
  b->h2  i1->h2  i2->h2  i3->h2  i4->h2  i5->h2  i6->h2  i7->h2  i8->h2  i9->h2 
   0.00    0.03    0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00 
i10->h2 i11->h2 i12->h2 i13->h2 i14->h2 i15->h2 i16->h2 i17->h2 
   0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00 
  b->h3  i1->h3  i2->h3  i3->h3  i4->h3  i5->h3  i6->h3  i7->h3  i8->h3  i9->h3 
   0.00    0.01    0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00 
i10->h3 i11->h3 i12->h3 i13->h3 i14->h3 i15->h3 i16->h3 i17->h3 
   0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00 
  b->h4  i1->h4  i2->h4  i3->h4  i4->h4  i5->h4  i6->h4  i7->h4  i8->h4  i9->h4 
   0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00 
i10->h4 i11->h4 i12->h4 i13->h4 i14->h4 i15->h4 i16->h4 i17->h4 
   0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00 
  b->h5  i1->h5  i2->h5  i3->h5  i4->h5  i5->h5  i6->h5  i7->h5  i8->h5  i9->h5 
   0.00    0.02    0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00 
i10->h5 i11->h5 i12->h5 i13->h5 i14->h5 i15->h5 i16->h5 i17->h5 
   0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00 
  b->h6  i1->h6  i2->h6  i3->h6  i4->h6  i5->h6  i6->h6  i7->h6  i8->h6  i9->h6 
   0.00   -0.01    0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00 
i10->h6 i11->h6 i12->h6 i13->h6 i14->h6 i15->h6 i16->h6 i17->h6 
   0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00 
  b->h7  i1->h7  i2->h7  i3->h7  i4->h7  i5->h7  i6->h7  i7->h7  i8->h7  i9->h7 
   2.15    0.00   -3.87   -0.37   -0.25   -0.62   -0.47    0.08    0.04    0.07 
i10->h7 i11->h7 i12->h7 i13->h7 i14->h7 i15->h7 i16->h7 i17->h7 
   0.52    0.27    0.45    0.76    0.01    0.15   -0.04    0.24 
  b->h8  i1->h8  i2->h8  i3->h8  i4->h8  i5->h8  i6->h8  i7->h8  i8->h8  i9->h8 
   0.00    0.01    0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00 
i10->h8 i11->h8 i12->h8 i13->h8 i14->h8 i15->h8 i16->h8 i17->h8 
   0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00 
 b->o h1->o h2->o h3->o h4->o h5->o h6->o h7->o h8->o 
-0.03  0.00 -0.03  0.02 -0.03 -0.02  0.02 -6.76  0.01 
```

```r
# Plotting confusion matrix
table(test_predictions, test_labels)
```

```
                test_labels
test_predictions   0   1
               0 470  20
               1   0   0
```

---

<p style='color:#000;'>This concludes the analysis and evaluation of the neural network model for stroke prediction. Further improvements can be made through hyperparameter tuning and experimentation with different model architectures.</p>

<br><br><br><br>
