# Import the required libraries and modules
import tensorflow as tf
import keras
import openai
import flask
import requests
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from flask import Flask, render_template, request, redirect, url_for, flash, send_file

# Set up the environment and the dependencies for the project
os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # Replace with your own OpenAI API key
GAN_MODEL = "openai-dall-e-0.5" # The name of the GAN model for image generation
CNN_MODEL = "vgg19" # The name of the CNN model for image classification
IMAGE_SIZE = 256 # The size of the output image in pixels
IMAGE_QUALITY = 95 # The quality of the output image in percentage
IMAGE_FORMAT = "png" # The format of the output image
DATA_FOLDER = "data" # The folder that contains the data and the resources for the project
TEXT_FILE = "text_prompts.txt" # The file that contains the text prompts for the input
IMAGE_FILE = "images.npy" # The file that contains the images for the output
LABEL_FILE = "labels.npy" # The file that contains the labels for the output
SOURCE_FILE = "sources.json" # The file that contains the sources for the classification results

# Build and train the GAN model for image generation
def generate_image(text):
  """
  This function takes a text query as input, and outputs an image that shows the astronomical object or phenomenon described by the text.
  It uses the OpenAI DALL·E model to create images from scratch based on a text prompt.
  """
  # Check if the text query is valid and not empty
  if not text or not isinstance(text, str):
    return None, "Invalid input. Please enter a valid text query."
  
  # Query the OpenAI API with the text prompt and the GAN model name
  response = openai.Completion.create(
    engine = GAN_MODEL,
    prompt = text,
    max_tokens = 1,
    temperature = 0.9,
    top_p = 1,
    frequency_penalty = 0,
    presence_penalty = 0,
    logprobs = 0
  )

  # Check if the response is successful and contains an image
  if response and "choices" in response and len(response["choices"]) > 0 and "text" in response["choices"][0]:
    image_data = response["choices"][0]["text"]
    # Decode the image data from base64 to bytes
    image_bytes = base64.b64decode(image_data)
    # Convert the image bytes to a PIL image object
    image = Image.open(BytesIO(image_bytes))
    # Resize the image to the desired size
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    # Return the image and a success message
    return image, "Image generated successfully."
  else:
    # Return None and an error message
    return None, "Image generation failed. Please try again with a different text query."

# Build and train the CNN model for image classification
def classify_image(image):
  """
  This function takes an image as input, and outputs a label that describes the type, size, distance, etc. of the astronomical object or phenomenon shown by the image.
  It uses the VGG-19 model to learn a mapping from input images to output labels.
  """
  # Check if the image is valid and not None
  if not image or not isinstance(image, Image.Image):
    return None, "Invalid input. Please enter a valid image."
  
  # Load the VGG-19 model with pretrained weights
  model = keras.applications.vgg19.VGG19(weights="imagenet", include_top=False, pooling="avg")
  # Load the text prompts, the images, the labels, and the sources from the data folder
  text_prompts = np.loadtxt(os.path.join(DATA_FOLDER, TEXT_FILE), dtype=str, delimiter="\n")
  images = np.load(os.path.join(DATA_FOLDER, IMAGE_FILE))
  labels = np.load(os.path.join(DATA_FOLDER, LABEL_FILE))
  sources = json.load(open(os.path.join(DATA_FOLDER, SOURCE_FILE)))
  # Convert the input image to a numpy array
  image_array = np.array(image)
  # Preprocess the image array for the model
  image_array = keras.applications.vgg19.preprocess_input(image_array)
  # Expand the dimensions of the image array to match the model input shape
  image_array = np.expand_dims(image_array, axis=0)
  # Predict the features of the image using the model
  image_features = model.predict(image_array)
  # Compute the cosine similarity between the image features and the images in the data
  similarities = np.dot(image_features, images.T) / (np.linalg.norm(image_features) * np.linalg.norm(images, axis=1))
  # Find the index of the most similar image in the data
  index = np.argmax(similarities)
  # Find the corresponding text prompt, label, and source in the data
  text_prompt = text_prompts[index]
  label = labels[index]
  source = sources[text_prompt]
  # Return the label and the source as a tuple
  return (label, source), "Image classified successfully."

# Create the user interface for the plugin
app = Flask(__name__) # Create a Flask app object
app.secret_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # Set a secret key for the app
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024 # Set the maximum size of the uploaded file to 16 MB

# Define the route for the home page
@app.route("/")
def index():
  """
  This function renders the home page of the plugin, where the user can enter a text query or a voice command, or upload an image file, and get the output image and the classification results.
  """
  return render_template("index.html") # Render the index.html template

# Define the route for the text input
@app.route("/text", methods=["POST"])
def text():
  """
  This function handles the text input from the user, and calls the generate_image and classify_image functions to get the output image and the classification results.
  It also saves the output image to a temporary file, and displays the image and the results on the web page.
  """
  # Get the text query from the form
  text = request.form.get("text")
  # Check if the text query is not empty
  if text:
    # Call the generate_image function with the text query
    image, message = generate_image(text)
    # Check if the image is not None
    if image:
      # Save the image to a temporary file
      image.save("temp." + IMAGE_FORMAT, quality=IMAGE_QUALITY)
      # Call the classify_image function with the image
      label, source = classify_image(image)
      # Check if the label is not None
      if label:
        # Render the result.html template with the image, the label, and the source
        return render_template("result.html", image="temp." + IMAGE_FORMAT, label=label, source=source)
      else:
        # Flash an error message and redirect to the home page
        flash("Image classification failed. Please try again with a different image.")
        return redirect(url_for("index"))
    else:
      # Flash an error message and redirect to the home page
      flash(message)
      return redirect(url_for("index"))
  else:
    # Flash an error message and redirect to the home page
    flash("Please enter a valid text query.")
    return redirect(url_for("index"))

# Define the route for the voice input
@app.route("/voice", methods=["POST"])
def voice():
  """
  This function handles the voice input from the user, and converts it to a text query using the SpeechRecognition library.
  It then calls the generate_image and classify_image functions to get the output image and the classification results.
  It also saves the output image to a temporary file, and displays the image and the results on the web page.
  """
  # Import the SpeechRecognition library
  import speech_recognition as sr
  # Create a Recognizer object
  r = sr.Recognizer()
  # Get the voice file from the form
  voice = request.files.get("voice")
  # Check if the voice file is not None
  if voice:
    # Save the voice file to a temporary file
    voice.save("temp.wav")
    # Open the voice file as a source
    with sr.AudioFile("temp.wav") as source:
      # Adjust the noise level
      r.adjust_for_ambient_noise(source)
      # Recognize the speech from the source
      try:
        text = r.recognize_google(source)
      except sr.UnknownValueError:
        # Flash an error message and redirect to the home page
        flash("Google Speech Recognition could not understand the audio.")
        return redirect(url_for("index"))
      except sr.RequestError as e:
        # Flash an error message and redirect to the home page
        flash("Could not request results from Google Speech Recognition service; {0}".format(e))
        return redirect(url_for("index"))
    # Check if the text query is not empty
    if text:
      # Call the generate_image function with the text query
      image, message = generate_image(text)
      # Check if the image is not None
      if image:
        # Save the image to a temporary file
        image.save("temp." + IMAGE_FORMAT, quality=IMAGE_QUALITY)
        # Call the classify_image function with the image
        label, source = classify_image(image)
        # Check if the label is not None
        if label:
          # Render the result.html template with the image, the label, and the source
          return render_template("result.html", image="temp." + IMAGE_FORMAT, label=label, source=source)
        else:
          # Flash an error message and redirect to the home page
          flash("Image classification failed. Please try again with a different image.")
          return redirect(url_for("index"))
      else:
        # Flash an error message and redirect to the home page
        flash(message)
        return redirect(url_for("index"))
    else:
      # Flash an error message and redirect to the home page
      flash("Please enter a valid voice command.")
      return redirect(url_for("index"))
  else:
    # Flash an error message and redirect to the home page
    flash("Please upload a valid voice file.")
    return redirect(url_for("index"))

# Define the route for the image input
@app.route("/image", methods=["POST"])
def image():
  """
  This function handles the image input from the user, and calls the classify_image function to get the classification results.
  It also saves the image to a temporary file, and displays the image and the results on the web page.
  """
  # Get the image file from the form
  image = request.files.get("image")
  # Check if the image file is not None
  if image:
    # Save the image to a temporary file
    image.save("temp." + IMAGE_FORMAT)
    # Open the image as a PIL image object
    image = Image.open("temp." + IMAGE_FORMAT)
    # Call the classify_image function with the image
    label, source = classify_image(image)
    # Check if the label is not None
    if label:
      # Render the result.html template with the image, the label, and the source
      return render_template("result.html", image="temp." + IMAGE_FORMAT, label=label, source=source)
    else:
      # Flash an error message and redirect to the home page
      flash("Image classification failed. Please try again with a different image.")
      return redirect(url_for("index"))
  else:
    # Flash an error message and redirect to the home page
    flash("Please upload a valid image file.")
    return redirect(url_for("index"))

# Define the route for the image download
@app.route("/download")
def download():
  """
  This function allows the user to download the output image as a file.
  """
  # Return the output image as a file
  return send_file("temp." + IMAGE_FORMAT, as_attachment=True)

# Run the app on the local server
if __name__ == "__main__":
  app.run(debug=True)
