# based on https://flask.palletsprojects.com/en/2.1.x/patterns/fileuploads/

import os
import pathlib

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
# allow Cross Origin Resource Sharing for everything
CORS(app)
# define path for image upload directory
app.config['UPLOAD_FOLDER'] = './uploads'
# create upload folder if it doesn't exist
pathlib.Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)


@app.route('/cat-predictions', methods=['POST'])
def predict_image_labels():
    """
    Expects a multipart/formdata request where the file is attached under the `image` key
    You can use Postman to try it out (see, e.g., https://stackoverflow.com/questions/16015548/how-to-send-multipart-form-data-request-using-postman)
    """
    img = request.files["image"]
    filename = secure_filename(img.filename)
    # save the file in the upload directory
    img.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # TODO: use the functionality from `src/CatPredictor.py` to get the label results for the uploaded image

    # TODO: implement and call the decision function that uses the results of the CatPredictor
    decision = decide_if_image_contains_cat(None, None)

    # TODO: change the response to something useful for image classification that contains the decision (this is just an example)
    return jsonify({'message': 'Image successfully uploaded!'})


def decide_if_image_contains_cat(label1, label2):
    """
    Decides if the image contains a cat based on the two model predictions
    Arguments:
        label1: prediction from the 1st imagenet model
        label2: prediction from the 2nd imagenet model
    Returns:
        # TODO
    """
    # TODO: use both label results to decide if the image is cat

    # TODO: create a reasonable response based on the decision, e.g., using a Dictionary
    return True
