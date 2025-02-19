import face_recognition 
import os 
import numpy as np 
import cv2
import imutils
import csv
import pandas as pd
import faiss


class ImageEncodings():

    def __init__(self):
        self.persons = {}
        self.encodedPersons = {}  
        self.index = None
    
    def loadImages(self,path):
        '''Loads all the images from the directory
        args: path to directory
        '''

        for folder in os.listdir(path):  # Iterate over person folders (name_id)
            person_path = os.path.join(path, folder)

            if os.path.isdir(person_path):  # Ensure it's a directory
                name, person_id = folder.rsplit("_", 1)  # Split name and id
                self.persons[person_id] = {name:[]}  # Initialize empty list for this person
                #self.person_info.append((name, person_id))  # Store name & ID

                for pic in os.listdir(person_path):  # Iterate over images inside
                    img_path = os.path.join(person_path, pic)
                    current_image = cv2.imread(img_path)

                    if current_image is not None:  # Check if image is valid
                        self.persons[person_id][name].append(current_image)  # Store images
            
    def generateEncodings(self):
        '''Generate encodings of the loaded images'''
        if not self.persons:
            print("There are no images loaded or the specified directory is empty")
            return

        

        for emp_id, person_data in self.persons.items():  # Iterate over persons dictionary
            for name, images in person_data.items():  # Iterate over names under emp_id
                encodings = []  # Store encodings for this person
                
                for img in images:  # Iterate over images (numpy arrays)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                    encoding = face_recognition.face_encodings(img_rgb)

                    if encoding:  # Ensure encoding is not empty
                        encodings.append(encoding[0])  # Append first encoding
                
                if encodings:  # If any encodings were found, store them
                    if emp_id not in self.encodedPersons:
                        self.encodedPersons[emp_id] = {}  # Initialize emp_id entry
                    
                    self.encodedPersons[emp_id][name] = encodings  # Store encodings

        print("Encodings generated successfully.")

    def saveEncodings(self, path):
        '''Saves the encodings to the specified file.
        Args:
            path (str): Path to a CSV file
        '''
        enc_id = 1  # Default starting Enc_Id
        file_exists = os.path.exists(path)

        # Load existing data to find the highest Enc_Id
        if file_exists:
            df = pd.read_csv(path)
            if not df.empty and "Enc_Id" in df.columns:
                enc_id = df["Enc_Id"].max() + 1  # Start from the next available ID

        # Open the file in write mode only if it’s empty or doesn’t exist, otherwise append
        mode = "a" if file_exists and not df.empty else "w"

        with open(path, mode=mode, newline="") as file:
            writer = csv.writer(file)

            # Write headers only if file is newly created or empty
            if mode == "w":
                writer.writerow(["Enc_Id", "Emp_Id", "Name", "Encodings"])

            # Iterate over persons dictionary
            for emp_id, person_data in self.encodedPersons.items():
                for name, encodings in person_data.items():
                    for encoding in encodings:
                        if isinstance(encoding, (list, np.ndarray)) and encoding.size > 0:  # Ensure encoding is valid
                            writer.writerow([enc_id, emp_id, name, ",".join(map(str, encoding))])
                            enc_id += 1  # Increment unique ID for each encoding

        print(f"Encodings saved successfully to {path}")

if __name__ == "__main__":
    enc = ImageEncodings()
    enc.loadImages("images")
    enc.generateEncodings()
    enc.saveEncodings("person_encodings.csv")