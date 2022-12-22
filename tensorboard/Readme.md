# Index
- [a plus b](#a-plus-b)
- [Mnist](#mnist)
# a plus b
```
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./log/a_plus_b',sess.graph)    
```
![image](https://user-images.githubusercontent.com/2216970/52910306-88ba2e00-32d0-11e9-82dd-aa0b2c078586.png)
# mnist
```
tf.summary.image('input',image_reshaped_input,10)
tf.summary.scalar("cross_entropy",cross_entropy)
tf.summary.histogram("prediction_result",prediction_result)
tf.summary.scalar("accuracy",accuracy)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./log/mnist",sess.graph)
```
![image](https://user-images.githubusercontent.com/2216970/52911008-f9b21380-32d9-11e9-802a-ce6e085db2d5.png)
