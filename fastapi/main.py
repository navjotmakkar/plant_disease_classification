from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from enum import Enum

app = FastAPI()

MODEL_POTATO = tf.keras.models.load_model("../models/potato_disease_classifier.h5")
MODEL_PEPPERBELL = tf.keras.models.load_model("../models/pepperbell_disease_classifier.h5")

CLASS_NAMES_POTATO = ["Early Blight", "Late Blight", "Healthy"]
CLASS_NAMES_PEPPERBELL = ["Pepper__bell___Bacterial_spot", "Pepper__bell___healthy"]

class AvailablePlants(str, Enum):
    potato = "potato"
    pepperbell = "bell pepper"

@app.get("/ping")
async def ping():
    return "Web service is working"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict/")
async def predict(
    plant: AvailablePlants,
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    if plant == "bell pepper":
        predictions = MODEL_PEPPERBELL.predict(img_batch)
        predicted_class = CLASS_NAMES_PEPPERBELL[np.argmax(predictions[0])]
    else:
        predictions = MODEL_POTATO.predict(img_batch)
        predicted_class = CLASS_NAMES_POTATO[np.argmax(predictions[0])]

    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8001)