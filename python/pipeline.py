import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def load_and_preprocess_data(data_dir, target_size=(224, 224), batch_size=32):
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    return train_generator, validation_generator

def create_model(input_shape=(224, 224, 3), num_classes=29):
    vgg16_base = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    for layer in vgg16_base.layers:
        layer.trainable = False

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = vgg16_base(inputs, training=False)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model_vgg16 = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    model_vgg16.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model_vgg16.summary()

    return model_vgg16

def train_model(model, train_generator, validation_generator, epochs=10):
    history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)
    return model, history

def predict(model, img_path):
    pass