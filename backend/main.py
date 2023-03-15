from fastapi import FastAPI, File, UploadFile
from ocr import ocr
from recognition import Recognition
from PIL import Image
import io
import numpy as np
from pydantic import BaseModel
import base64
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

paddle_recog = Recognition()


@app.get("/")
def hello_world():
    return "Hello World!"


@app.post("/ocr/")
async def convert_image_to_text(file: UploadFile = File(...)):
    """
    Converts an image file to text and returns the result.
    """
    try:
        image_data = file.file.read()
    except Exception:
        return {"message": "There was an error uploading the file"}
    # Read the image data
    img = Image.open(io.BytesIO(image_data))
    np_img = np.array(img)
    text_in_img = ocr(np_img, paddle_recog)

    # Return the text file as a response
    return {"text": text_in_img}
