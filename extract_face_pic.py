import cv2
import os
from PIL import Image
import pytesseract

# Input folder where ID image exists
input_folder = 'input'  # Update this path

# Ensure output directory exists
output_dir = './detected_faces'
os.makedirs(output_dir, exist_ok=True)

# Load the cascade classifier
face_classifier = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def process_image(file_path):
    # Load the image and convert it to grayscale
    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    # Check if any faces are detected
    if len(faces) == 0:
        print(f"No faces found in {file_path}")
    else:
        # Iterate through the detected faces
        for i, (x, y, w, h) in enumerate(faces):
            # Add padding to the detected face region
            x = max(x - 25, 0)  # Ensure we don't go out of bounds
            y = max(y - 40, 0)  # Ensure we don't go out of bounds
            face_roi = image[y:y + h + 70, x:x + w + 50]  # Extract the face region

            # Get Name from Emirates ID
            name = ""
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image, lang='eng+ara')
            lines = text.splitlines()
            for line in lines:
                if "Name:" in line:
                    name = line.split("Name:")[-1].strip()
                    print(f"Extracted Name: {name}")
                    break
            else:
                print("Name field not found!")

            # Save the extracted face to disk
            face_filename = os.path.join(output_dir, f'{name} {i + 1}.jpg')
            cv2.imwrite(face_filename, face_roi)


def process_folder():
    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            process_image(file_path)
        else:
            print(f"Unsupported file type: {filename}")


if __name__ == "__main__":
    process_folder()

