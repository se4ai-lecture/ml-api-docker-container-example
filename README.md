# Creating a Custom Inference API

(You can work in pairs, if you want to.)

Your task is to create an inference Web API that checks if an image contains a cat.
This starting project already provides the ML functionality for this (`src/CatPredictor.py`), namely a class that uses two pretrained models from the [ImageNet](https://www.image-net.org/) family (VGG16 and MobileNet).
You can see an example of its usage in `src/usage-example.py`.
An advanced skeleton for the API you should build is provided based on the [Flask framework](https://flask.palletsprojects.com/en/2.1.x/quickstart/) in `src/prediction-api.py`.
You can use the HTTP client of your choice to test it, e.g., [Postman](https://www.postman.com/downloads/) or [curl](https://curl.se/download.html).
Once your API is ready, encapsulate it into a [Docker image](https://docs.docker.com/language/python/build-images/), and [build and run a container](https://docs.docker.com/language/python/run-containers/) with it.

**Bonus task if you are quick:** change your client project from [last tutorial](https://github.com/se4ai-lecture/google-vision-api-example) to use your newly created API instead of the Google Cloud Vision API for identifying all cats from the image set.

## Prerequisites

1. Install [Python 3](https://www.python.org/downloads/), preferably >= 3.9
2. Install dependencies: `pip install -r requirements.txt`
3. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for the container part)

## Usage

```bash
# using the prediction example (you can change the used filename to a different file under `images`)
python src/usage-example.py

# using the REST API
# set an ENV variable for the Flask entrypoint file
export FLASK_APP=src/prediction-api.py
# start the API --> API offers the endpoint `POST http://localhost:5000/cat-predictions`
flask run
# for enabling hot reload (execute before starting)
export FLASK_ENV=development
```
