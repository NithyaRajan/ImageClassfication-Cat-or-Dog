import flask
from flask import Flask, request, render_template
from sklearn.externals import joblib
import numpy as np
from keras.preprocessing import image
from scipy import misc


app = Flask(__name__)

@app.route("/")
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def make_prediction():
	if request.method=='POST':

		# get uploaded image file if it exists
		file = request.files['image']
		if not file: return render_template('index.html', label="No file")
		
		# read in file as raw pixels values
		# (ignore extra alpha channel and reshape as its a single image)
		image = misc.imread(file)
		
		image = image.img_to_array(image)
		image = np.expand_dims(image, axis = 0)

		# make prediction on new image
		result = classifier.predict(image)
		#training_set.class_indices

		if result[0][0] == 1:
			prediction = 'dog'
    	
		else:
			prediction = 'cat'
		label = prediction
	
		# # squeeze value from 1D array and convert to string for clean return
		# label = str(np.squeeze(result))

		# # switch for case where label=10 and number=0
		# if label=='10': label='0'

		return render_template('index.html', label=label)


if __name__ == '__main__':
	global classifier
	from keras.models import load_model
	# load ml model
	classifier = load_model('my_model.h5')
	# start api
	app.run(host='0.0.0.0', port=8000, debug=True)
