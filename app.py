
import numpy as np
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

from flask import Flask, render_template, request, redirect, url_for
import os
#import google.generativeai as genai
#import Image
app = Flask(__name__)


# Replace with your API key
#genai.configure(api_key="AIzaSyDAocjymh058Ll_YEwA-YUPXWIAdIF1QSw")

# Use gemini-pro-vision for image input
#model = genai.GenerativeModel("gemini-1.5-flash")
UPLOAD_FOLDER = 'static/uploads'  # Directory to save uploaded images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = YOLO("./runs/content/runs/detect/train2/weights/best.pt")  # Change this to your model path


@app.route("/", methods=["GET", "POST"])
def upload_file():
    detected_products = None  # Initialize as None to prevent errors
    product_percentages = None
    other_percentage = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("3rdpage.html", error="No file part", detected_products=None)

        file = request.files["file"]

        if file.filename == "":
            return render_template("3rdpage.html", error="No selected file", detected_products=None)

        if file:
            # Convert file to an image
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Get image dimensions
            img_height, img_width, _ = img.shape
            total_image_area = img_width * img_height

            # Initialize dictionary for product areas
            product_areas = defaultdict(int)
            detected_products = []  # Store detected product names

            # Run inference
            results = model(img)

            # Process results
            for r in results:
                for box in r.boxes:
                    # Get bounding box coordinates and product name
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates (integers)
                    class_id = int(box.cls[0])  # Get class index
                    product_name = model.names[class_id]  # Get product name

                    # Append product name to detected products
                    detected_products.append(product_name)

                    # Calculate the area occupied by the detected product
                    bbox_area = (x2 - x1) * (y2 - y1)
                    product_areas[product_name] += bbox_area

                    # Black out the detected product regions on the image
                    img[y1:y2, x1:x2] = (0, 0, 0)  # Set the pixels in the bounding box to black

            # Count occurrences of each product
            product_counts = Counter(detected_products)

            # Calculate total occupied area
            total_occupied_area = sum(product_areas.values())

            # Calculate empty space area
            empty_area = total_image_area - total_occupied_area

            # Convert occupied areas to percentages
            product_percentages = {prod: (area / total_image_area) * 100 for prod, area in product_areas.items()}
            other_percentage = (empty_area / total_image_area) * 100

            # Format results for rendering
            detected_products = ", ".join([f"{k}: {v}" for k, v in product_counts.items()])
            area_results = "\n".join([f"{prod}: {percent:.2f}% of the image" for prod, percent in product_percentages.items()])
            area_results += f"\nOther (empty space): {other_percentage:.2f}% of the image"

            # Save the modified image or pass it to be displayed in the template
            modified_img_path = "static/modified_image.jpg"  # Save image to static folder for displaying
            cv2.imwrite(modified_img_path, img)

    else:
        return render_template("3rdpage.html", error=None)

    """# Example of generating additional content (such as improvements for product placement)
    prompt = "Suggest improvements to the placement of Ramy juices on the shelf to improve sales."
    response = model.generate_content([prompt, Image.open(request.files["file"])])  # Assuming your model can handle this input

    if response:
        result = response.text
    else:
        print("Error: No response received from Gemini.")
        result = "Error generating content."
    """
    # Return the result page with the modified image and analysis
    return render_template("3rdpage.html", 
                           detected_products=detected_products, 
                           p=product_percentages,
                           o=other_percentage,
                           area_results=area_results, 
                           error=None,
                           modified_image=modified_img_path)

if __name__ == "__main__":
    app.run(debug=True)
