import requests
from io import BytesIO
from PIL import Image

# Define the text description
description = input("put your decription")

# Send a request to the DeepAI API to generate an image based on the text description
resp = requests.post('https://api.deepai.org/api/text2img', data={
    'text': description
}, headers={'api-key': '133fc02c-5726-4b9c-b5f2-fa28a2313729'})

# Get the image data from the response
if resp.status_code == 200:
    response_data = resp.json()
    image_url = response_data['output_url']
    # Load the image from the URL
    img = Image.open(BytesIO(requests.get(image_url).content))
    # Display the generated image
    img.show()
else:
    print("Error:", resp.text)
