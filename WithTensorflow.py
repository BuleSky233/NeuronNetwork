import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
print("在测试集中")
model.evaluate(x_test,  y_test, verbose=2)
plt.imshow(x_test[0], cmap="binary")
plt.show()
prediction = model.predict(x_test)
print("输出层的输出\n",prediction[0])
print("模型的预测结果为(即输出的10个数中最大数的索引为）",np.argmax(prediction[0]))
print("该测试样本的真实标签为",y_test[0])