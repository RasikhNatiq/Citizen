import cv2
import threading
import face_recognition
import queue
import time
import numpy as np
import imutils
import faiss_index


class CameraThread(threading.Thread):
    def __init__(self, camera_id, rtsp_url, cam_name,thershold,frame_skip): 
        super().__init__()
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id
        self.cam_name = cam_name
        self.frame_queue = queue.Queue(maxsize=30)  # Limit queue size
        self.current_frame = None
        self.previous_frame = None
        self.frame_skip = frame_skip
        self.active = False



        self.index = faiss_index.FaissIndex('person_encodings.csv',3)
        #self.known_faces = known_faces
        #self.names = list(known_faces.keys())

        protopath = "models/deploy.prototxt"
        modelpath = "models/res10_300x300_ssd_iter_140000.caffemodel"
        self.detector = cv2.dnn.readNetFromCaffe(protopath, modelpath)

        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {cam_name} at {rtsp_url}")
            exit()
        self.frame_count = 0

    def run(self):
        '''Main running function of the camera, captures the frame and put them in the queue.
        This function calls the other functions during running.'''
        try:
            self.active = True
            while self.active:
                ret, frame = self.cap.read()
                if ret:
                    
                    self.frame_count += 1
                    self.current_frame = frame

                    if self.frame_count % self.frame_skip == 0:
                        try:                      
                            self.analyse_frame()
                            #pass
                        except queue.Full:
                            print("Unable to analyse frame")
                    
                    if self.frame_queue.full():
                         _ = self.frame_queue.get(block=False)

                    self.frame_queue.put(self.current_frame, block=False)
                    self.previous_frame = self.current_frame
                else:
                    print(f"Camera {self.cam_name} disconnected. Reconnecting...")
                    self.cap.release()

                    # Attempt to reconnect to the camera
                    self.cap = cv2.VideoCapture(self.rtsp_url)
                    if not self.cap.isOpened():
                        print(f"Failed to reconnect to {self.cam_name}. Retrying in 5 seconds...")
                        time.sleep(5)
                    else:
                        print(f"Reconnected to {self.cam_name}.")
                        self.frame_count = 0  # Reset frame count

                # Exit condition (e.g., on a specific key press)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                    print(f"Exiting camera {self.cam_name} processing loop.")
                    break
        finally:
            # Clean up resources on exit
            self.cap.release()
            cv2.destroyAllWindows()
            print(f"Resources for camera {self.cam_name} released.")

    def get_face_locations_res(self):
        '''Returns the locations of the faces in the frame.
        Uses the resnet model to detect faces.'''

        #frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        frame = imutils.resize(self.current_frame, width=600)
        (H, W) = frame.shape[:2]
        face_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
        self.detector.setInput(face_blob)
        face_detections = self.detector.forward()
        face_boxes = []
        for i in np.arange(0, face_detections.shape[2]):
            confidence = face_detections[0, 0, i, 2]
            if confidence > 0.2:
                face_boxes.append(face_detections[0, 0, i, 3:7] * np.array([W, H, W, H]))
        
        return face_boxes

    def get_face_locations_fr(self):
        '''Returns the locations of the faces in the frame.
        Uses the resnet model to detect faces.'''
        imgS = cv2.resize(np.array(self.current_frame), (0, 0), None, 0.5, 0.5)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        all_imgLoc = face_recognition.face_locations(imgS)  #getting image locations on the screen

        return all_imgLoc
    
    def create_bounding_boxes(self,all_imgLoc):
        '''create bounding boxes around the given face locations.
        args: all face locations in the image'''
        for imgLoc in all_imgLoc:
            if imgLoc == []:
                continue
            top, right, bottom, left = imgLoc
            cv2.rectangle(self.current_frame, (left*2, top*2), (right*2, bottom*2), (0, 255, 0), 2)

    def create_encodings_fr(self,all_imgLoc):
        '''generate encodings for the faces detected in the image using face recognition library
        args: all face locations in the image'''
        imgencs = []
        for imgLoc in all_imgLoc:
            if imgLoc == []:
                continue
            name = "Unknown"
            start = time.time()
            
            imgencs.append(face_recognition.face_encodings(self.current_frame, [imgLoc]))

        return imgencs

    def compare_face(self,query_vector):
        '''compare two faces by searching in a faiss vector database.
        args: face to be compared'''
        #print(query_vector)

        top = self.index.find_top_matches(query_vector)
        return top[0][0],top[0][1]
    
        #distances, indices = self.index.search(query_vector, k=1)
        # emp_id = indices[0][0]
        # emp_name = self.names[indices[0][0]]
        #return emp_id,emp_name

    def compare_faces(self,query_vectors):
        '''compare multiple faces with the faces in the faiss database
        args: list of embeddings of faces in the image.'''
        emps={}

        # for query in query_vectors:
        #     query = query[0].reshape(1,-1)
        #     distances, indices = self.index.search(query, k=1)
        #     emps[indices[0][0]] = self.names[indices[0][0]]

        for query in query_vectors:
            top = self.index.find_top_matches(query)
            emps[top[0]] = top[1]
        return emps

    def label_faces(self,emp_data,all_imgLoc):
        '''creates a bounding box around the faces and put name under it.
        args: (emp_data) matched embeddings in the faiss database
        (all_imgLoc) location of faces in the image'''

        if (len(emp_data) != len(all_imgLoc)):
            print("Error additional encodings added")
        else:
            i = 0
            for name in emp_data.values():
                top, right, bottom, left = all_imgLoc[i]
                top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2  
                cv2.rectangle(self.current_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(self.current_frame, name, (left, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                i += 1

    def analyse_frame(self):
        '''analyses the frame to recognise the person in the frame'''
        face_locations = self.get_face_locations_fr()
        face_encodings = self.create_encodings_fr(face_locations)
        matched_encodings = self.compare_faces(face_encodings)
        if matched_encodings:
            self.label_faces(matched_encodings,face_locations)


    def compare_face_fr(self,imgenc):
        '''compare a face with all the faces in the known_faces using face_recognition library
        args: encoding of the face to be compared
        returns: name of the employee closest to the given encoding'''

        for name,encoding in self.known_faces.items():
            match = face_recognition.compare_faces(encoding, imgenc,0.56)    
            
            if match[0] == True:
                name = name.upper()
                break
            else:
                name = "Unknown"
        
        return name
    
    def compare_faces_fr(self,imgencs):
        emps = {}
        for imgenc in imgencs:
            for name,encoding in self.known_faces.items():
                match = face_recognition.compare_faces(encoding, imgenc,0.56)    
                
                if match[0] == True:
                    name = name.upper()
                    break
                else:
                    name = "Unknown"
            emps.append(name)

        return emps

    def show_stream(self):
        while True:     
            sct_img = self.queue.get()           
            cv2.imshow("Results", sct_img)
            k = cv2.waitKey(10)& 0xff
            if k == 27:
                break

    def stop_capture(self):
        self.cap.release()
        self.active = False
        cv2.destroyAllWindows()