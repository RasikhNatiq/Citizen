
import cv2
import face_recognition
import faiss
import numpy as np
import pandas as pd
import threading
import queue
import time
import ast
import argparse
import imutils
from camerathread import CameraThread

# Configuration
DEFAULT_BATCH_SIZE = 10
DEFAULT_FRAME_SKIP = 2
DEFAULT_DISTANCE_THRESHOLD = 0.6

def process_batch(batch_data):
    batch_frames = [data['frame'] for data in batch_data]
    cam_names = [data['cam_name'] for data in batch_data]
    face_locations = []
    face_encodings = []
    cam_indices = []

    for i, frame in enumerate(batch_frames):
        locations = face_recognition.face_locations(frame)
        face_locations.extend(locations)
        encodings = face_recognition.face_encodings(frame, locations)
        face_encodings.extend(encodings)
        cam_indices.extend([i] * len(locations)) # Keep track of which camera each face belongs to

    if face_encodings:
        xq = np.array(face_encodings).astype('float32')
        D, I = index.search(xq, 1)

        for i, (dist, idx) in enumerate(zip(D.flatten(), I.flatten())):
            if dist < DISTANCE_THRESHOLD:
                cam_index = cam_indices[i]
                top, right, bottom, left = face_locations[i]
                cv2.rectangle(batch_frames[cam_index], (left, top), (right, bottom), (0, 0, 255), 2)
                name = embeddings_df.loc[idx, 'emp_id']
                cv2.putText(batch_frames[cam_index], str(name) + f"({dist:.2f})", (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)

    for i, frame in enumerate(batch_frames):
        cv2.imshow(cam_names[i], frame)

def main():
    parser = argparse.ArgumentParser(description="Multi-camera face recognition.")
    parser.add_argument("--camera_file", default="camera.csv", help="Path to camera CSV file.")
    parser.add_argument("--embeddings_file", default="encodings.csv", help="Path to embeddings CSV file.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for processing.")
    parser.add_argument("--frame_skip", type=int, default=DEFAULT_FRAME_SKIP, help="Process every Nth frame.")
    parser.add_argument("--distance_threshold", type=float, default=DEFAULT_DISTANCE_THRESHOLD, help="Distance threshold for face matching.")
    args = parser.parse_args()

 
    cameras_df = pd.read_csv('cameras.csv')

    camera_threads = []
    for _, row in cameras_df.iterrows():
        thread = CameraThread(row['cam_id'], row['rtsp'], row['cam_name'], args.distance_threshold,1)
        camera_threads.append(thread)
        thread.start()


    try:
        while True:
            # Get camera name input from the user
            cam_name_input = input("Enter camera name to view (or 'q' to quit): ")

            if cam_name_input.lower() == 'q':
                break

            # Find the corresponding camera thread
            selected_thread = None
            for thread in camera_threads:
                if thread.cam_name.lower() == cam_name_input.lower():
                    selected_thread = thread
                    break

            if selected_thread:
                cv2.namedWindow(selected_thread.cam_name) # Create a named window
                while True:
                    try:
                        frame = selected_thread.frame_queue.get(block=False)
                        cv2.imshow(selected_thread.cam_name, frame)
                    except queue.Empty:
                        pass

                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q') or key & 0xFF == ord('b'): #Press 'b' to go back
                        cv2.destroyWindow(selected_thread.cam_name) # Destroy the window on exit
                        break
            else:
                print("Camera not found.")

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        for thread in camera_threads:
            thread.stop_capture()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()