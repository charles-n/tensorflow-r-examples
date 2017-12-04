library(tensorflow)

### version 1
sess = tf$Session()
a <- tf$constant( 1+1 )
hello <- tf$constant('Hello, TensorFlow!')
sess$run( c(hello, a) )
sess$close()

### version 2
tf$Session()$run(tf$constant('Hello, TensorFlow!'))
