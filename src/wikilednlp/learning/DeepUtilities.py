from tensorflow_core.python.keras.models import model_from_json


def store(model, file_name):
    """Store to disk a model along with its weights
    # Arguments:
        model: the Keras model to be stored
    """
    # serialize model to JSON
    model_json = model.to_json()
    with open(file_name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(file_name + ".h5")
    print("Saved model to disk")


def restore(file_name):
    """Restore from disk a model along with its weights
    # Returns:
        model: the Keras model which was restored
    """
    # load json and create model
    json_file = open(file_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(file_name + '.h5')
    print("Loaded model from disk")
    return model
