library(tensorflow)

#################################
#### n net
#################################

# 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron) implementation with TensorFlow. 
# This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/).
    
    datasets <- tf$contrib$learn$datasets
    mnist <- datasets$mnist$read_data_sets("MNIST-data", one_hot = TRUE)
    
    # Parameters
    learning_rate = 0.1
    num_steps = 500
    batch_size = 128L
    display_step = 100
    
    # Network Parameters
    n_hidden_1 = 256 # 1st layer number of neurons
    n_hidden_2 = 256 # 2nd layer number of neurons
    num_input = 784 # MNIST data input (img shape: 28*28)
    num_classes = 10 # MNIST total classes (0-9 digits)
    
    # tf Graph input
    X = tf$placeholder("float")
    Y = tf$placeholder("float")
    
    # Store layers weight & bias
    weights = c(
      h1=tf$Variable(tf$random_normal(shape(num_input, n_hidden_1))),
      h2=tf$Variable(tf$random_normal(shape(n_hidden_1, n_hidden_2))),
      out=tf$Variable(tf$random_normal(shape(n_hidden_2, num_classes)))
    )
    
    biases = c(
      b1=tf$Variable(tf$random_normal(shape(n_hidden_1))),
      b2=tf$Variable(tf$random_normal(shape(n_hidden_2))),
      out=tf$Variable(tf$random_normal(shape(num_classes)))
    )
    
    
    # Create model
    neural_net <- function(x){
      # Hidden fully connected layer with 256 neurons
      layer_1 = tf$add(tf$matmul(x, weights$h1), biases$b1)
      # Hidden fully connected layer with 256 neurons
      layer_2 = tf$add(tf$matmul(layer_1, weights$h2), biases$b2)
      # Output fully connected layer with a neuron for each class
      out_layer = tf$matmul(layer_2, weights$out) + biases$out
      return(out_layer)
    }
    
    # Construct model
    logits = neural_net(X)
    prediction = tf$nn$softmax(logits)
    
    # Define loss and optimizer
    loss_op = tf$reduce_mean(tf$nn$softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf$train$AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer$minimize(loss_op)
    
    # Evaluate model
    correct_pred = tf$equal(tf$argmax(prediction, 1L), tf$argmax(Y, 1L))
    accuracy = tf$reduce_mean(tf$cast(correct_pred, tf$float32))
    
    # Initialize the variables (i$e$ assign their default value)
    init = tf$global_variables_initializer()
    
    
          # for (step in 1:201) {
          #   sess$run(train)
          #   if (step %% 20 == 0)
          #     cat(step, "-", sess$run(W), sess$run(b), "\n")
          # }
    
    
    # Start training
    with(tf$Session() %as% sess, {
      
          # Run the initializer
          sess$run(init)
          
          for (step in 1:num_steps+1){
              # batch_x, batch_y = mnist$train$next_batch(batch_size)
              batches <- mnist$train$next_batch(batch_size)
                batch_x <- batches[[1]]
                batch_y <- batches[[2]]
                
     
              # Run optimization op (backprop)
              sess$run(train_op, feed_dict=dict(X= batch_x, Y= batch_y))
              if (step %% display_step == 0 | step == 1) {
                  # Calculate batch loss and accuracy
                  out_batch= sess$run( c(loss_op, accuracy), feed_dict=dict(X= batch_x, Y= batch_y))
                      loss = out_batch[[1]]
                      acc = out_batch[[2]]
              
                  cat("Step " , format(step) 
                      , ", Minibatch Loss= " , format(loss, nsmall = 3) 
                      , ", Training Accuracy= " ,  format(acc, nsmall = 3) 
                      , '\n')
                  
                  # print("Step " + str(step) + ", Minibatch Loss= " + \
                  #       "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  #       "{:.3f}".format(acc))        
              }
          }
      
          cat("Optimization Finished!")
          
          # Calculate accuracy for MNIST test images
          cat("Testing Accuracy:"
              , sess$run(accuracy , feed_dict=dict(X=mnist$test$images, Y=mnist$test$labels))
              )
    
          # print(sess$run(tf$argmax(Y, 17L), feed_dict={X=mnist$test$images}))
                
    })
