#' Standard k-NN classifier.
#'
#' @param train A numeric matrix or data frame of the training data.
#' @param test A numeric matrix or data frame of the test data.
#' @param train_target A numeric or factor vector of class labels for the training data.
#' @param nn The number of neighbors.
#' @return A numeric or factor vector of predicted class labels.
#' @export
testa_KNN <- function( train, test, train_target, nn ) {
  labels <- class::knn( train = train, test = test, cl = train_target, k = nn, prob = FALSE )
  return( as.numeric( as.character( labels ) ) )
}

#' Computes balanced accuracy.
#'
#' This function requires the 'caret' package.
#'
#' @param true_labels The true class labels.
#' @param predicted_labels The predicted class labels.
#' @return The balanced accuracy score.
#' @export
balanced_accuracy_score <- function( true_labels, predicted_labels ) {
  cm <- caret::confusionMatrix( factor( predicted_labels ), factor( true_labels ) )
  return( mean( as.data.frame( cm$byClass )$`Balanced Accuracy` ) )
}

#' Computes the F1-score.
#'
#' @param true_labels The true class labels.
#' @param predicted_labels The predicted class labels.
#' @param average The type of averaging ('weighted').
#' @return The F1-score.
#' @export
f1_score <- function( true_labels, predicted_labels, average = 'weighted' ) {
  if ( requireNamespace( "MLmetrics", quietly = TRUE ) ) {
    return( MLmetrics::F1_Score( y_pred = predicted_labels, y_true = true_labels, positive = 1 ) )
  } else {
    warning( "MLmetrics package not found. F1_score will return NA. Please install MLmetrics for proper F1 calculation." )
    return( NA )
  }
}
