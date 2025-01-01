import numpy as np
import cv2
import os
import pymupdf  # PyMuPDF
from PIL import Image
import pytesseract

# Specify the folder containing PDF files
pdf_folder = 'input'  # Update this path

# Ensure output directory exists
output_dir = './detected_faces'
os.makedirs(output_dir, exist_ok=True)

# Load the cascade classifier
face_classifier = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')

# Specify Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to extract images from a PDF
def extract_images_from_pdf(pdf_path):
    images = []
    pdf_document = pymupdf.open(pdf_path)

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)
        images.append(cv2.cvtColor(img, cv2.COLOR_RGBA2BGR))  # Convert to BGR format
    return images

# Function to perform OCR and extract the name
def extract_name_from_image(image):
    text = pytesseract.image_to_string(image, lang='eng+ara')
    lines = text.splitlines()
    for line in lines:
        if "Name:" in line:
            return line.split("Name:")[-1].strip()
    return "Unknown"


def process_input():
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

    # Process each PDF file
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        images = extract_images_from_pdf(pdf_path)

        # Iterate over each extracted image
        for page_num, image in enumerate(images):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)

            # Perform OCR to extract the name from the page
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            name = extract_name_from_image(pil_image)

            # Check if any faces are detected
            if len(faces) == 0:
                print(f"No faces found on page {page_num + 1} of {pdf_file}")
            else:
                # Iterate through the detected faces
                for i, (x, y, w, h) in enumerate(faces):
                    # Add padding to the detected face region
                    x = max(x - 25, 0)
                    y = max(y - 40, 0)
                    face_roi = image[y:y + h + 70, x:x + w + 50]  # Extract the face region

                    # Save the extracted face to disk
                    face_filename = os.path.join(output_dir, f'{name}_page_{page_num + 1}_face_{i + 1}.jpg')
                    cv2.imwrite(face_filename, face_roi)

                    print(f"Saved face: {face_filename}")


if __name__ == "__main__":
    process_input()
