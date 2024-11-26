install.packages("ggplot2")
install.packages("GGally")
install.packages("corrplot")

# Load necessary libraries
library(ggplot2)
library(datasets)

# Load the iris dataset
data(iris)

# Step 1: Visualize the distribution of each numeric variable (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width)

# Histogram for Sepal Length
ggplot(iris, aes(x = Sepal.Length)) +
  geom_histogram(binwidth = 0.2, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Sepal Length", x = "Sepal Length", y = "Frequency")

# Histogram for Sepal Width
ggplot(iris, aes(x = Sepal.Width)) +
  geom_histogram(binwidth = 0.2, fill = "green", color = "black") +
  labs(title = "Distribution of Sepal Width", x = "Sepal Width", y = "Frequency")

# Histogram for Petal Length
ggplot(iris, aes(x = Petal.Length)) +
  geom_histogram(binwidth = 0.2, fill = "orange", color = "black") +
  labs(title = "Distribution of Petal Length", x = "Petal Length", y = "Frequency")

# Histogram for Petal Width
ggplot(iris, aes(x = Petal.Width)) +
  geom_histogram(binwidth = 0.2, fill = "purple", color = "black") +
  labs(title = "Distribution of Petal Width", x = "Petal Width", y = "Frequency")

# Step 2: Boxplot to see the distribution across species

# Boxplot for Sepal Length by Species
ggplot(iris, aes(x = Species, y = Sepal.Length, fill = Species)) +
  geom_boxplot() +
  labs(title = "Sepal Length by Species", x = "Species", y = "Sepal Length")

# Boxplot for Petal Length by Species
ggplot(iris, aes(x = Species, y = Petal.Length, fill = Species)) +
  geom_boxplot() +
  labs(title = "Petal Length by Species", x = "Species", y = "Petal Length")

# Step 3: Scatter plot to visualize the relationships between two variables

# Scatter plot for Sepal Length vs Sepal Width
ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) +
  geom_point() +
  labs(title = "Sepal Length vs Sepal Width", x = "Sepal Length", y = "Sepal Width")

# Scatter plot for Petal Length vs Petal Width
ggplot(iris, aes(x = Petal.Length, y = Petal.Width, color = Species)) +
  geom_point() +
  labs(title = "Petal Length vs Petal Width", x = "Petal Length", y = "Petal Width")

# Step 4: Pair plot for all variables
# This requires the GGally package for pair plot functionality
library(GGally)
ggpairs(iris, aes(color = Species))

# Step 5: Correlation plot
library(corrplot)
cor_matrix <- cor(iris[, 1:4])
corrplot(cor_matrix, method = "circle", type = "upper", order = "hclust", addCoef.col = "black")


#=====>>>>>

# Load necessary libraries
library(caret)
library(e1071)
library(randomForest)
library(rpart)
library(ggplot2)

# Load the iris dataset
data(iris)
View(iris)

# Step 1: Split the data into training (70%) and testing (30%) sets
set.seed(123)  # Set seed for reproducibility
trainIndex <- createDataPartition(iris$Species, p = 0.7, list = FALSE)
trainData <- iris[trainIndex, ]
testData <- iris[-trainIndex, ]

# Step 2: Train models with different algorithms


# Decision Tree
tree_model <- train(Species ~ ., data = trainData, method = "rpart")

# Random Forest
rf_model <- train(Species ~ ., data = trainData, method = "rf")

# Support Vector Machine (SVM)
svm_model <- train(Species ~ ., data = trainData, method = "svmRadial")

# Step 3: Make predictions on the test set

tree_pred <- predict(tree_model, testData)
rf_pred <- predict(rf_model, testData)
svm_pred <- predict(svm_model, testData)

# Step 4: Evaluate the accuracy and other classification metrics for each model

# Accuracy

tree_accuracy <- mean(tree_pred == testData$Species)
rf_accuracy <- mean(rf_pred == testData$Species)
svm_accuracy <- mean(svm_pred == testData$Species)

# Precision, Recall, F1-Score

tree_metrics <- confusionMatrix(tree_pred, testData$Species)
rf_metrics <- confusionMatrix(rf_pred, testData$Species)
svm_metrics <- confusionMatrix(svm_pred, testData$Species)

# Print the classification metrics (Accuracy, Precision, Recall, F1)


cat("\nDecision Tree Accuracy: ", tree_accuracy, "\n")
print(tree_metrics)

cat("\nRandom Forest Accuracy: ", rf_accuracy, "\n")
print(rf_metrics)

cat("\nSVM Accuracy: ", svm_accuracy, "\n")
print(svm_metrics)

# Step 5: Compare the models' accuracies in a table
model_accuracies <- data.frame(
  Model = c( "Decision Tree", "Random Forest", "SVM"),
  Accuracy = c( tree_accuracy, rf_accuracy, svm_accuracy)
)

# Print the accuracy comparison table
print(model_accuracies)

# Step 6: Visualize the comparison of model performances using a bar plot
ggplot(model_accuracies, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity") +
  labs(title = "Model Accuracy Comparison", x = "Model", y = "Accuracy") +
  theme_minimal()

