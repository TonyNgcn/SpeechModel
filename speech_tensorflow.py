import tensorflow as tf
import pandas as pd
import numpy as np


CSV_COLUMN_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
CSV_COLUMN_NAMES_PREDICT = ['A', 'B', 'C', 'D', 'E', 'F']


def load_data(y_name='G'):
	"""Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
	train_path, test_path = train_dataset_fp, test_dataset_fp

	train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
	train_x, train_y = train, train.pop(y_name)

	test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
	test_x, test_y = test, test.pop(y_name)

	return (train_x, train_y), (test_x, test_y)

def load_predict_data():
	predict_path=predict_dataset_fp
	predict_x=pd.read_csv(predict_path,names=CSV_COLUMN_NAMES_PREDICT, header=0)
	return predict_x

def train_input_fn(features, labels, batch_size):
	"""An input function for training"""
	# Convert the inputs to a Dataset.
	dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

	# Shuffle, repeat, and batch the examples.
	dataset = dataset.shuffle(1000).repeat().batch(batch_size)

	# Return the dataset.
	return dataset

def predict_input_fn(features, batch_size):
	"""An input function for training"""
	# Convert the inputs to a Dataset.
	dataset = tf.data.Dataset.from_tensor_slices((dict(features)))

	# Shuffle, repeat, and batch the examples.
	dataset = dataset.repeat().batch(batch_size)

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
predict_dataset_fp='speech_topredict.csv'



def main(argv):
	# Fetch the data
	(train_x, train_y), (test_x, test_y) = load_data()
	predict_x=load_predict_data()
	# Feature columns describe how to use the input.
	my_feature_columns = []
	for key in train_x.keys():
		my_feature_columns.append(tf.feature_column.numeric_column(key=key))
	predict_x=predict_x.T.get_values()

	predict_x_list={}

	for o,x in zip(CSV_COLUMN_NAMES_PREDICT,predict_x):
		predict_x_list[o]=x
	predict_x = predict_x.T
	expected = ['-1', '1']
	# Build 4hidden layer DNN with 10,20,10，10 units respectively.
	classifier = tf.estimator.DNNClassifier(
		feature_columns=my_feature_columns,
		# Two hidden layers of 10 nodes each.
		# hidden_units=[10, 20, 10, 10],
		hidden_units=[1024, 512, 256, 128, 64],
		optimizer=tf.train.ProximalAdagradOptimizer(
			learning_rate=0.01,
			l1_regularization_strength=0.001
		),
		model_dir='models/speech',
		# The model must choose between 2 classes.
		n_classes=2)

	# Train the Model.
	classifier.train(input_fn=lambda: train_input_fn(train_x, train_y, 100), steps=20000)

	# Evaluate the model.
	eval_result = classifier.evaluate(
		input_fn=lambda: eval_input_fn(test_x, test_y, 50))

	print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

	#predict the result
	predictions = classifier.predict(
		input_fn=lambda: eval_input_fn(predict_x_list,
		                                         labels=None,
		                                         batch_size=100))

	template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

	predict_y=[]
	for pred_dict in predictions:
		if pred_dict['class_ids'][0] == 0:
			predict_y.append(-1)
		else:
			predict_y.append(1)

	predict_y = np.array(predict_y).reshape(len(predict_y),1)
	result = np.hstack([predict_x, predict_y])
	result = pd.DataFrame(result)
	result.to_csv('result.csv')

if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run(main)
