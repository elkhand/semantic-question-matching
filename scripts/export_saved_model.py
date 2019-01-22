import tensorflow as tf
import sys

PROJECT_ROOT_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(str(PROJECT_ROOT_DIR))
import sqmutils.data_utils as du

# The export path contains the name and the version of the model
tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference

custom_objects= {"f1": du.f1, "recall" : du.recall, "precision" : du.precision}
model = tf.keras.models.load_model('../models/best_val_f1_model.h5', custom_objects = custom_objects)
export_path = '../semantic_question_classifier/1'

print("\nmodel.input:", model.input,"\n")
print("\nmodel.outputs", model.outputs,"\n")

# Fetch the Keras session and save the model
# The signature definition is defined by the input and output tensors
# And stored with the default serving key
with tf.keras.backend.get_session() as sess:
    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={'q1': model.input[0], 'q2': model.input[1]},
		outputs={t.name: t for t in model.outputs})