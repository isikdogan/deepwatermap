import tensorflow as tf

def model():
    inputs = tf.keras.layers.Input(shape=[None, None, 6])
    x = tf.keras.layers.Conv2D(filters=30, kernel_size=1, padding='same')(inputs)
    x = tf.keras.layers.Activation('sigmoid')(x)
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same')(x)
    x = tf.keras.layers.Activation('sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model