import tensorflow as tf
import pandas as pd

CSV_COLUMN_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G']


def load_data(y_name='G'):
	"""Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
	train_path, test_path = train_dataset_fp, test_dataset_fp

	train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
	train_x, train_y = train, train.pop(y_name)

	test = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
	test_x, test_y = test, test.pop(y_name)

	return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
	"""An input function for training"""
	# Convert the inputs to a Dataset.
	dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

	# Shuffle, repeat, and batch the examples.
	dataset = dataset.shuffle(1000).repeat().batch(batch_size)

	# Return the dataset.
	return dataset


def eval_input_fn(features, labels, batch_size):
	"""An input function for evaluation or prediction"""
	features = dict(features)
	if labels is None:
		# No labels, use only features.
		inputs = features
	else:
		inputs = (features, labels)

	# Convert the inputs to a Dataset.
	dataset = tf.data.Dataset.from_tensor_slices(inputs)

	# Batch the examples
	assert batch_size is not None, "batch_size must not be None"
	dataset = dataset.batch(batch_size)

	# Return the dataset.
	return dataset


train_dataset_fp = 'speech_train.csv'
test_dataset_fp = 'speech_test.csv'


def main(argv):
	# Fetch the data
	(train_x, train_y), (test_x, test_y) = load_data()

	# Feature columns describe how to use the input.
	my_feature_columns = []
	for key in train_x.keys():
		my_feature_columns.append(tf.feature_column.numeric_column(key=key))

	# Build 4hidden layer DNN with 10,20,10，10 units respectively.
	classifier = tf.estimator.DNNClassifier(
		feature_columns=my_feature_columns,
		# Two hidden layers of 10 nodes each.
		# hidden_units=[10, 20, 10, 10],
		hidden_units=[1024, 512, 256, 128],
		optimizer=tf.train.ProximalAdagradOptimizer(
			learning_rate=0.01,
			l1_regularization_strength=0.001
		),
		# The model must choose between 2 classes.
		n_classes=2)

	# Train the Model.
	classifier.train(input_fn=lambda: train_input_fn(train_x, train_y, 100), steps=20000)

	# Evaluate the model.
	eval_result = classifier.evaluate(
		input_fn=lambda: eval_input_fn(test_x, test_y, 50))

	print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run(main)
