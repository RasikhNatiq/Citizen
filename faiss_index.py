import faiss
import pandas as pd
import numpy as np
import os
from collections import Counter



class FaissIndex():
    def __init__(self,csv_path,top_k):
        self.index = None

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        self.top_k = top_k
        self.build_faiss_index()
        
    def build_faiss_index(self):
        '''Builds a Faiss Index from the encodings stored in the CSV file.
        
        Args:
            csv_path (str): Path to the CSV file containing embeddings.
        
        Returns:
            faiss.IndexFlatL2: The FAISS index containing all embeddings.
        '''
        # Ensure the required column exists
        if "Encodings" not in self.df.columns:
            raise ValueError("CSV file does not contain 'Encodings' column.")

        # Convert encodings from strings to NumPy array
        embeddings = []
        for enc in self.df["Encodings"]:
            try:
                emb_array = np.array(list(map(float, enc.split(","))))  # Convert string to float array
                embeddings.append(emb_array)
            except ValueError:
                print(f"Skipping invalid encoding: {enc}")

        # Convert list to NumPy array
        if not embeddings:
            raise ValueError("No valid embeddings found in CSV file.")

        embeddings = np.array(embeddings, dtype=np.float32)

        # Ensure embeddings are of correct shape
        if len(embeddings.shape) != 2:
            raise ValueError(f"Embeddings should have shape (n_samples, n_features), but got {embeddings.shape}")

        # Create FAISS index
        index = faiss.IndexFlatL2(embeddings.shape[1])  
        index.add(embeddings)

        self.index = index

    def find_top_matches(self,query_encoding):
        '''Finds the top K closest matches for a given query encoding.
        
        Args:
            query_encoding (numpy.ndarray): The query face encoding (1D NumPy array).
            faiss_index (faiss.IndexFlatL2): The FAISS index containing stored encodings.
            csv_path (str): Path to the CSV file containing names and embeddings.
            top_k (int): Number of top matches to return.
        
        Returns:
            List[Tuple[str, str, float]]: A list of (Emp_Id, Name, Distance) for top matches.
        '''
        
        # Ensure query encoding is in the correct shape
        query_encoding = np.array(query_encoding, dtype=np.float32).reshape(1, -1)
        

        # Search for the top K nearest neighbors
        distances, indices = self.index.search(query_encoding, 3)

        # Ensure the correct columns exist
        if "Enc_Id" not in self.df.columns or "Emp_Id" not in self.df.columns or "Name" not in self.df.columns:
            raise ValueError("CSV file is missing required columns: 'Enc_Id', 'Emp_Id', 'Name'.")

        # Extract matching results
        matches = []
        for i in range(self.top_k):
            index = indices[0][i]
            distance = distances[0][i]

            if index < len(self.df):  # Ensure index is within range
                emp_id = self.df.iloc[index]["Emp_Id"]
                name = self.df.iloc[index]["Name"]
                matches.append((emp_id, name, distance))

        top_matches = self.most_occour_name(matches)

        return top_matches

    def average_top_matches(self, top_matches):
        '''Averages the occurrences of the returned persons and returns the top 2 matches.
        
        Args:
            top_matches (list of tuples): List of (Emp_Id, Name, Distance) from FAISS search.
        
        Returns:
            List of tuples [(Emp_Id1, Name1, Probability1), (Emp_Id2, Name2, Probability2)]
        '''
        
        if not top_matches:
            return []

        # Convert FAISS distances to similarity scores (higher is better)
        max_distance = max([dist for _, _, dist in top_matches]) + 1e-6  # Avoid division by zero
        scores = [(emp_id, name, 1 - (dist / max_distance)) for emp_id, name, dist in top_matches]

        # Aggregate probabilities for each (Emp_Id, Name)
        id_name_counts = Counter()
        id_name_scores = {}

        for emp_id, name, score in scores:
            key = (emp_id, name)  # Unique identifier for employee
            id_name_counts[key] += 1
            id_name_scores[key] = id_name_scores.get(key, 0) + score  # Sum scores for averaging

        # Calculate average probability for each person
        avg_probabilities = {key: id_name_scores[key] / id_name_counts[key] for key in id_name_scores}

        # Sort by probability in descending order
        sorted_results = sorted(avg_probabilities.items(), key=lambda x: x[1], reverse=True)

        # Return the top 2 matches as (Emp_Id, Name, Probability)
        return [(emp_id, name, prob) for (emp_id, name), prob in sorted_results[:2]]
    
    

    def most_occour_name(self,top_matches):
        '''Averages the occurrences of the returned persons and returns the most frequent match with its ID and probability.
        
        Args:
            top_matches (list of tuples): List of (Emp_Id, Name, Distance) from FAISS search.
        
        Returns:
            Tuple (Emp_Id, Name, Probability)
        '''
        
        if not top_matches:
            return None

        # Convert FAISS distances to similarity scores (higher is better)
        max_distance = max([dist for _, _, dist in top_matches]) + 1e-6  # Avoid division by zero
        scores = [(emp_id, name, 1 - (dist / max_distance)) for emp_id, name, dist in top_matches]

        # Count occurrences of each (Emp_Id, Name) pair
        person_counts = Counter((emp_id, name) for emp_id, name, _ in scores)

        # Find the most frequent (Emp_Id, Name) pair
        (most_frequent_id, most_frequent_name), max_count = person_counts.most_common(1)[0]

        # Aggregate probabilities for the most frequent person
        total_score = sum(score for emp_id, name, score in scores if emp_id == most_frequent_id and name == most_frequent_name)

        # Calculate the average probability
        most_frequent_probability = total_score / max_count

        # Return the most frequent Emp_Id, Name, and Probability
        return most_frequent_id, most_frequent_name, most_frequent_probability

