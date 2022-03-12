import os
import numpy as np
import tensorflow as tf
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Traversing the dataset folder
# 遍历数据集文件夹
for dirname, _, filenames in os.walk('./Kaggel_dataset'):
    for filename in filenames:
        os.path.join(dirname, filename)

# Read the training set and generate the image generator
# 读取训练集，生成图片生成器
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('./Kaggel_dataset/training_set', target_size = (108, 108), batch_size = 32, classes=['cats', 'dogs'])

# Read the testing set and generate the image generator
# 读取测试集，生成图片生成器
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('./Kaggel_dataset/test_set', target_size = (108, 108), batch_size = 32, classes=['cats', 'dogs'])

# Build the model
# 建立模型
model = Sequential()

# Add 1st Convolutional layer
# 增加第一个卷积层
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), input_shape=(108, 108, 3), activation='relu'))
# Batch Normalization
# 批量标准化
model.add(tf.keras.layers.BatchNormalization())
# Add 2nd Convolutional layer
# 添加第二个卷积层
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), input_shape=(108, 108, 3), activation='relu'))
# Batch Normalization
# 批量标准化
model.add(tf.keras.layers.BatchNormalization())
# Add pooling layer
# 添加池化层
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'))
# Add 3rd Convolutional layer
# 添加第三个卷积层
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
# Batch Normalization
# 批量标准化
model.add(tf.keras.layers.BatchNormalization())
# Add 4nd Convolutional layer
# 添加第四个卷积层
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
# Batch Normalization
# 批量标准化
model.add(tf.keras.layers.BatchNormalization())
# Add pooling layer
# 添加池化层
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'))
# Flattening
# 压平数据
model.add(tf.keras.layers.Flatten())
# Output layer
# 输出层
model.add(tf.keras.layers.Dense(2, activation='sigmoid'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Using callbacks for avoiding overfitting
# 防止过拟合
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(patience=5, 
                               min_delta=0.001,
                               restore_best_weights=True)

# Model training
# 训练模型
history = model.fit(x = training_set, validation_data = test_set, epochs = 10, callbacks=[early_stopping])

# Store the models
# 储存模型
model.save_weights('./model.h5')

# Test model
# 测试模型
test_imgs, test_labels = next(test_set)

# Get some photoes(400+) to test
# 获取测试图像
for i in range(14):
    test_imgs_i, test_labels_i = next(test_set)
    test_imgs = np.concatenate((test_imgs, test_imgs_i),axis=0)
    test_labels = np.concatenate((test_labels, test_labels_i),axis=0)

# Get predictions
# 获取预测值
pred = np.round(model.predict(x=test_imgs, verbose=0))

# Get test results
# 得到测试结果
from sklearn import metrics
metrics.classification_report(test_labels, pred)