#' Quantizes real values to integer levels.
#'
#' @description This function quantizes real values in the interval \code{[a, b]} to integer levels from 0 to k-1.
#'
#' @param arr A numeric vector in the interval \code{[a, b]}.
#' @param a The lower bound of the interval.
#' @param b The upper bound of the interval.
#' @param k The number of quantization levels (default is 10).
#' @return A vector of quantized integers in \code{0, ..., k - 1}.
#' @export vector of quantized integers in (0, ..., k - 1).
#' @export
quantize <- function( arr, a, b, k = 10 ) {
  # clip values to [a, b]
  arr_clipped <- pmax( pmin( arr, b ), a )
  # scale to [0, 1]
  normalized <- ( arr_clipped - a ) / ( b - a )
  # scale to [0, k) and convert to integer
  quantized <- as.integer( normalized * k )
  # clip values to [0, k-1]
  quantized <- pmax( pmin( quantized, k - 1 ), 0 )
  return( quantized )
}

#' Computes the curvatures of all samples in the training set.
#'
#' @param data A numeric matrix or data frame of the training data.
#' @param k The number of neighbors for the initial k-NN graph.
#' @return A numeric vector of curvatures for each sample.
#' @export
curvature_estimation <- function( data, k ) {
  n <- nrow( data )
  m <- ncol( data )
  curvatures <- numeric( n )

  # Generate k-NN graph using FNN::get.knn
  knn_results <- FNN::get.knn( data, k = k )

  for ( i in 1:n ) {
    neighbors <- knn_results$nn.index[ i, ]
    amostras <- rbind( data[ i, , drop = FALSE], data[ neighbors, , drop = FALSE ] )
    ni <- nrow( amostras )

    if ( ni > 1 ) {
      I <- stats::cov( amostras )
    } else {
      I <- diag( m )
    }

    eig_results <- eigen( I )
    v <- eig_results$values
    w <- eig_results$vectors
    sorted <- order( v, decreasing = TRUE )
    wpca <- w[ , sorted ]

    squared <- wpca^2
    ncol_cross <- ( m * ( m - 1 ) ) / 2

    cross <- matrix( 0, nrow = m, ncol = ncol_cross )
    col <- 1
    for ( j in 1:m ) {
      for ( l in j:m ) {
        if ( j != l ) {
          cross[ , col ] <- wpca[ , j ] * wpca[ , l ]
          col <- col + 1
        }
      }
    }

    Q <- cbind( matrix( 1, m, 1 ), wpca, squared, cross )
    H <- Q[ , ( m + 2 ):ncol( Q ) ]
    II <- H %*% t( H )

    S <- II %*% I
    curvatures[i] <- sum( diag( S ) )
  }
  return( curvatures )
}

#' Computes the curvature of a single test sample's neighborhood.
#'
#' @param data A numeric matrix or data frame representing the neighborhood (test point + its neighbors).
#' @return A single numeric value for the curvature.
#' @export
point_curvature_estimation <- function( data ) {
  n <- nrow( data )
  m <- ncol( data )

  if ( n > 1 ) {
    I <- stats::cov( data )
  } else {
    I <- diag( m )
  }

  eig_results <- eigen( I )
  v <- eig_results$values
  w <- eig_results$vectors
  sorted <- order( v, decreasing = TRUE )
  wpca <- w[ , sorted ]

  squared <- wpca^2
  ncol_cross <- ( m * ( m - 1 ) ) / 2
  cross <- matrix( 0, nrow = m, ncol = ncol_cross )
  col <- 1
  for ( j in 1:m ) {
    for ( l in j:m ) {
      if ( j != l ) {
        cross[ , col ] <- wpca[ , j ] * wpca[ , l ]
        col <- col + 1
      }
    }
  }

  Q <- cbind( matrix( 1, m, 1 ), wpca, squared, cross )
  H <- Q[, ( m + 2 ):ncol( Q ) ]
  II <- H %*% t( H )

  S <- II %*% I
  curvature <- sum( diag( S ) )
  return( curvature )
}
