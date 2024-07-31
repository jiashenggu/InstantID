import requests
import base64
import json
import os
import time

# Function to encode an image to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# The URL of your FastAPI img2img endpoint
url = "http://172.25.248.111:8010/img2img"
# url = "http://0.0.0.0:8000/img2img"

# Encoding the image to base64

# image_path = "https://so1.360tres.com/t0130ee245793fdc291.png"
# image_path = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSoKKTWznZtEpdVhDY9GN2eOFmkuTyipnwWqUH-IWYw5l0YY4RSTAHXi_YLdLC8-nd68TY&usqp=CAU"
import glob
image_paths = glob.glob("/ML-A100/team/mm/gujiasheng/InstantID/examples/*.jpg")
image_paths = ["/ML-A100/team/mm/gujiasheng/InstantID/assets/1721781060-d41d8cd98f00b204e9800998ecf8427e-00ae5dec495411efa5805aae10d2a7c2.jpg"]
for image_path in image_paths:
    # image_path = "/ML-A100/team/mm/gujiasheng/InstantID/examples/dramatic_face.jpg"
    if os.path.exists(image_path):
        image_base64 = encode_image_to_base64(image_path)
    response_format = "file_path"
    # Form data to be sent to the API/ML-A100/team/mm/gujiasheng/InstantID/examples/yann-lecun_resize (1).jpg
    data = {
        "image_url": image_path,  # Assuming the API expects a base64 string in this field
        # "size": "768x1344",
        "size": "1280x1280",
        "prompt": "student",  # The prompt to be used for image generation
        "n_prompt": "",  # The negative prompt to be used for image generation
        "style": "",  # The style to be used for image generation
        "response_format": response_format,
        "can_empty": False,
    }

    import concurrent.futures
    from sdxl_styles import apply_style
    data["prompt"], data["n_prompt"] = apply_style("Japanese Anime Style 2", data["prompt"])

    # Function to make a POST request to the API
    def make_post_request(url, data):
        start = time.time()
        response = requests.post(url, json=data)
        end = time.time()
        print("request wall time: ", end-start)
        return response

    # Making parallel POST requests to the API
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submitting the POST requests
        response_futures = [executor.submit(make_post_request, url, data) for _ in range(6)]

        # Processing the responses
        for i, future in enumerate(concurrent.futures.as_completed(response_futures)):
            response = future.result()
            print(response.headers)
            print("request time: ", float(response.headers['x-request-end-time']) - float(response.headers['x-request-start-time']))
            # Checking if the request was successful
            if response.status_code == 200:
                response_data = response.json()
                print(json.loads(response_data["info"]))
                if data["response_format"] == "b64_json":
                    if "images" not in response_data:
                        print("Image processing failed or no image returned.")
                    if response_data.get("images", []) == []:
                        print("No image returned.")
                        continue
                    for j, output_image in enumerate(response_data.get("images", [])):
                        save_image_path = image_path.split("/")[-1].replace(".jpg", f"_output_{i}_{j}.jpg")
                        with open(f"output/{save_image_path}", "wb") as output_file:
                            output_file.write(base64.b64decode(output_image))
                        print(f"output_image_{i}.jpg, Image processed and saved successfully.")
                            
                elif data["response_format"] == "file_path":
                    if "images" not in response_data:
                        print(f"Image {image_path} processing failed or no image returned.")
                    for output_image_path in response_data.get("images", []):
                        print(f"Image processed and saved successfully at: {output_image_path}")
            else:
                print(f"Failed to process image {image_path}: {response.status_code}")
                print(response.text)
    break
