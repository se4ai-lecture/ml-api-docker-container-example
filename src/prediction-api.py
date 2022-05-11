# based on https://flask.palletsprojects.com/en/2.1.x/patterns/fileuploads/

import os
import pathlib

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from CatPredictor import CatPredictor

app = Flask(__name__)
# allow Cross Origin Resource Sharing for everything
CORS(app)
# define path for image upload directory
app.config['UPLOAD_FOLDER'] = './uploads'
# create upload folder if it doesn't exist
pathlib.Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

# instantiate CatPredictor object
cat_pred = CatPredictor()


@app.route('/cat-predictions', methods=['POST'])
def predict_image_labels():
    """
    Expects a multipart/formdata request where the file is attached under the `image` key
    You can use Postman to try it out (see, e.g., https://stackoverflow.com/questions/16015548/how-to-send-multipart-form-data-request-using-postman)
    """
    img = request.files["image"]
    filename = secure_filename(img.filename)
    # save the file in the upload directory
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img.save(img_path)

    # predict labels for image
    results = cat_pred.predict(img_path)

    decision = decide_if_image_contains_cat(
        results['vgg16'], results['mobilenet']
    )

    return jsonify(decision)


def decide_if_image_contains_cat(label1, label2):
    """
    Decides if the image contains a cat based on the two model predictions
    Arguments:
        label1: prediction from the 1st imagenet model
        label2: prediction from the 2nd imagenet model
    Returns:
        Dictionary with a boolean `is_cat`, a float `score` (0 to 1), and a string `message`
    """
    # calculated mean between highest cat label scores
    aggregated_score = (
        get_top_cat_score_for_model(label1[0]) +
        get_top_cat_score_for_model(label2[0])
    ) / 2
    is_cat = True if aggregated_score >= 0.35 else False

    return {
        'is_cat': is_cat,
        'score': aggregated_score,
        'message': 'The image contains a cat.' if is_cat else 'The image does not contain a cat.'
    }


def get_top_cat_score_for_model(labels):
    """
    Checks if the top 2 predicted labels are cat-related, returns the highest label score or 0
    Arguments:
        labels: an array of labels
    Returns:
        A float indicating the highest score for cat-related labels
    """
    if label_is_cat_related(labels[0][1]):
        return labels[0][2]
    elif label_is_cat_related(labels[1][1]):
        return labels[1][2]
    else:
        return 0


def label_is_cat_related(label):
    """
    Checks if the label is cat-related ('*cat*', 'tabby', 'lynx')
    Returns:
        A boolean
    """
    if 'cat' in label:
        return True
    if 'tabby' in label:
        return True
    if 'lynx' in label:
        return True
    return False
