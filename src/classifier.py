import tensorflow as tf
import utils.network_utils as nu
import os


def main(input_dir: str,
         num_of_epochs: int = 10,
         num_of_classes: int = 3,
         img_width: int = 512,
         img_height: int = 512,
         batch_size: int = 3,
         validation_split: int = 0.2,
         learning_rate: int = 0.01,
         pretrained: bool = False,
         pretrained_dir: str = None,
         export: bool = False,
         export_dir: str = None):
    training_set = nu.load_dataset(data_dir=input_dir,
                                   validation_split=validation_split,
                                   batch_size=batch_size,
                                   img_width=img_width,
                                   img_height=img_height,
                                   subset="training")
    validation_set = nu.load_dataset(data_dir=input_dir,
                                     validation_split=validation_split,
                                     batch_size=batch_size,
                                     img_width=img_width,
                                     img_height=img_height,
                                     subset="validation")

    if pretrained:
        model = nu.pretrained_model(pretrained_dir)

    else:
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Rescaling(1. / 255),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=128, activation='relu'),
                tf.keras.layers.Dense(units=64, activation='relu'),
                tf.keras.layers.Dense(units=32, activation='relu'),
                tf.keras.layers.Dense(units=num_of_classes)
            ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy())

    model.fit(training_set,
              validation_data=validation_set,
              epochs=num_of_epochs)

    if export:
        nu.export_model(model, export_dir)


if __name__ == "__main__":
    input_folder = input("Input directory: ")
    main(input_dir=input_folder)
