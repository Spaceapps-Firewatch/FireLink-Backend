from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from source import preprocess_input_image, batch_predict, conv_float_int, combine_image, load_trained_model, burn_area
from PIL import Image
import numpy as np
from keras.models import load_model

app = FastAPI()

# Mount a static files directory for serving images
app.mount("/static", StaticFiles(directory="static"), name="static")

model = load_trained_model("temp_model.h5")

@app.post("/upload/")
async def upload_image(file: UploadFile):
    uploaded_image = Image.open(file.file).convert("RGB")
    input_image_array = np.array(uploaded_image)
    original_width, original_height, pix_num = input_image_array.shape
    new_image_array, row_num, col_num = preprocess_input_image(input_image_array)

    # Make Prediction
    preds = batch_predict(new_image_array, model)
    output_pred = conv_float_int(combine_image(preds, row_num, col_num, original_width, original_height, remove_ghost=True)[:,:,0])

    # Save the predicted image to the static directory
    predicted_image_path = Path("static/predicted_image.jpg")
    output_image = Image.fromarray(output_pred)
    output_image.save(predicted_image_path)

    # Prepare the data for response
    response_data = {
        "image_url": f"/static/{predicted_image_path.name}",
        "coordinates": {
            "north": 40.773941,
            "south": 40.712216,
            "east": -74.12544,
            "west": -74.22655,
        }
    }
    return response_data

@app.get("/image/")
async def read_image():
    image_path = Path("static/image.jpg")  # Adjust this path to your image file
    return FileResponse(image_path)

@app.get("/coordinates/")
async def read_coordinates():
    return {
        "north": 40.773941,
        "south": 40.712216,
        "east": -74.12544,
        "west": -74.22655,
    }
