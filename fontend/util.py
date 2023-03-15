import requests
from io import BytesIO


def ocr(img):
    url = 'http://127.0.0.1:8000/ocr'

    buffered = BytesIO()
    img.save(buffered, format="PNG")
    image_bytes = buffered.getvalue()

    # Send the image bytes as a binary file in the request body
    response = requests.post(url,  files={"file": image_bytes})
    return response.json()["text"]
