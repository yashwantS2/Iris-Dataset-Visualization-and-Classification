# Load necessary libraries
library(caret)
library(e1071)
library(randomForest)
library(ggplot2)
install.packages("naivebayes") 
library(naivebayes)
# Load the iris dataset
data(iris)

# Step 1: Check for missing values
if (anyNA(iris)) {
  print("Data contains missing values.")
  iris <- na.omit(iris)  # Remove missing values if any
} else {
  print("No missing values found.")
}

# Step 2: Ensure the target variable is a factor
iris$Species <- as.factor(iris$Species)

# Step 3: Split the data into training (70%) and testing (30%) sets
set.seed(123)  # Set seed for reproducibility
trainIndex <- createDataPartition(iris$Species, p = 0.7, list = FALSE)
trainData <- iris[trainIndex, ]
testData <- iris[-trainIndex, ]

# Step 4: Set up cross-validation
train_control <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = multiClassSummary)

# Step 5: Train models with different algorithms

# K-Nearest Neighbors (KNN)
knn_model <- train(Species ~ ., data = trainData, method = "knn", trControl = train_control)

# Naive Bayes
nb_model <- train(Species ~ ., data = trainData, method = "naive_bayes", trControl = train_control)

# Random Forest
rf_model <- train(Species ~ ., data = trainData, method = "rf", trControl = train_control)

# Support Vector Machine (SVM)
svm_model <- train(Species ~ ., data = trainData, method = "svmRadial", trControl = train_control)

# Step 6: Make predictions on the test set
knn_pred <- predict(knn_model, testData)
nb_pred <- predict(nb_model, testData)
rf_pred <- predict(rf_model, testData)
svm_pred <- predict(svm_model, testData)

# Step 7: Evaluate the accuracy and other classification metrics for each model

# Accuracy
knn_accuracy <- mean(knn_pred == testData$Species)
nb_accuracy <- mean(nb_pred == testData$Species)
rf_accuracy <- mean(rf_pred == testData$Species)
svm_accuracy <- mean(svm_pred == testData$Species)

# Precision, Recall, F1-Score
knn_metrics <- confusionMatrix(knn_pred, testData$Species)
nb_metrics <- confusionMatrix(nb_pred, testData$Species)
rf_metrics <- confusionMatrix(rf_pred, testData$Species)
svm_metrics <- confusionMatrix(svm_pred, testData$Species)

# Print the classification metrics (Accuracy, Precision, Recall, F1)
cat("KNN Accuracy: ", knn_accuracy, "\n")
print(knn_metrics)

cat("\nNaive Bayes Accuracy: ", nb_accuracy, "\n")
print(nb_metrics)

cat("\nRandom Forest Accuracy: ", rf_accuracy, "\n")
print(rf_metrics)

cat("\nSVM Accuracy: ", svm_accuracy, "\n")
print(svm_metrics)

# Step 8: Compare the models' accuracies in a table
model_accuracies <- data.frame(
  Model = c("Naive Bayes", "KNN", "Random Forest", "SVM"),
  Accuracy = c(nb_accuracy ,knn_accuracy , rf_accuracy, svm_accuracy)
)

# Print the accuracy comparison table
print(model_accuracies)

# Step 9: Visualize the comparison of model performances using a bar plot
ggplot(model_accuracies, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity") +
  labs(title = "Model Accuracy Comparison", x = "Model", y = "Accuracy") +
  theme_minimal()

