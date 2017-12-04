library(tensorflow)

##########################################
#### multivariate line fitting example
##########################################

    # Create 100 phony x1, x2, y data points, y = x1 * 0.1 + x1 * 0.5 + 0.3
    x_data <- cbind(runif(100, min=0, max=1) , runif(100, min=0, max=1))
    y_data <- x_data %*% c(0.1, 0.5) + 0.3
    
    # Try to find values for W and b that compute y_data = W * x_data + b
    W <- tf$Variable(tf$random_uniform(shape(2,1), -1.0, 1.0))
    b <- tf$Variable( tf$zeros(shape(1)) )
    y <-  tf$matmul( x_data, tf$to_double(W))  + tf$to_double(b)
    

    # Minimize the mean squared errors.
    loss <- tf$reduce_mean((y - y_data) ^ 2)
      # optimizer <- tf$train$GradientDescentOptimizer(0.5)
      # train <- optimizer$minimize(loss)
      train <- tf$train$GradientDescentOptimizer(0.5)$minimize(loss)
    
    # Launch the graph and initialize the variables.
    sess = tf$Session()
    sess$run(tf$global_variables_initializer())
    
    # Fit the line (Learns best fit is W: (0.1, 0.5), b: 0.3)
    # for (step in 1:50) {
    #   sess$run(train)
    #   if (step %% 5 == 0)
    #     cat(step, "-", sess$run(W), sess$run(b), "\n")
    # }

    
    # Fit the line (Learns best fit is W: (0.1, 0.5), b: 0.3)
    for (step in 1:201) {
      sess$run(train)
      if (step %% 20 == 0)
        cat(step, "-", sess$run(W), sess$run(b), "\n")
    }


    # data.frame(head( sess$run(y) ), c( sess$run(W), sess$run(b)) )
    # data.frame(head( sess$run(y) ) == c( head(x_data) %*% sess$run(W) + as.numeric(sess$run(b))) )


    sess$close()
    
