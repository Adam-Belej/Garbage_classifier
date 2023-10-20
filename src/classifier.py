import tensorflow as tf

network = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(512, 512, 3)),
        tf.keras.layers.Dense(units=25, activation='relu'),
        tf.keras.layers.Dense(units=15, activation='relu'),
        tf.keras.layers.Dense(units=10, activation='softmax')

    ])

network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))