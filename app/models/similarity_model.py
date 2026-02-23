import os
import json
import numpy as np
from typing import Dict, Any, List, Optional
from sentence_transformers import SentenceTransformer

class EmailClassifierModel:
    """Email classifier model using embedding similarity"""

    def __init__(self):
        self.topic_data = self._load_topic_data()
        self.topics = list(self.topic_data.keys())

        # Load sentence transformer model (same model as feature generator)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Pre-compute embeddings for all topic descriptions
        self.topic_embeddings = self._compute_topic_embeddings()
        
        #stored emails
        self.emailspath='/home/ec2-user/environment/lab2_factories/data/emails.json'
        
    
    def _load_topic_data(self) -> Dict[str, Dict[str, Any]]:
        """Load topic data from data/topic_keywords.json"""
        data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'topic_keywords.json')
        with open(data_file, 'r') as f:
            return json.load(f)

    def _compute_topic_embeddings(self) -> Dict[str, np.ndarray]:
        """Pre-compute embeddings for all topic descriptions"""
        topic_embeddings = {}
        for topic, data in self.topic_data.items():
            description = data['description']
            embedding = self.model.encode(description, convert_to_numpy=True)
            topic_embeddings[topic] = embedding
        return topic_embeddings
    
    
    #new function for similarity check, low theshold to trigger other method
    def _check_stored_similarlity(self, email_embedding: np.ndarray, threshold:float=.1) -> Optional[str]:
        """compare current email against stored emails"""
        if not os.path.exists(self.emailspath):
            return None
            
        with open(self.emailspath, 'r') as f:
            storedemails=json.load(f)
            
        for item in storedemails:
            storedembedding=self.model.encode(item['content'], convert_to_numpy=True)
            
            #I haven't learned how to do this in AI class yet but I know what it is
            #in many ways, this class is teaching more about AI than my AI class
            # Manual cosine similarity
            dot = np.dot(email_embedding, storedembedding)
            norm = np.linalg.norm(email_embedding) * np.linalg.norm(storedembedding)
            sim = dot / norm if norm != 0 else 0

            if sim > threshold:
                return item.get('groundtruth')
        
        return None
    
    
    def predict(self, features: Dict[str, Any], use_similarity: bool=True) -> Dict[str, Any]:
        """Classify email into one of the topics using most similar emails from stored emails. Defaults back to feature similarity"""
        
        #current email embedding
        email_embedding = features.get("email_embeddings_average_embedding", None)
        if isinstance(email_embedding, list):
            email_embedding = np.array(email_embedding)
            
        #similarity check
        if use_similarity and email_embedding is not None:
            similarlabel=self._check_stored_similarlity(email_embedding)
            if similarlabel:
                return {"topic": similarlabel, "method": 'similar email'}
    
        scores = {}
        
        # Calculate similarity scores for each topic based on features
        for topic in self.topics:
            score = self._calculate_topic_score(features, topic)
            scores[topic] = score
        
        predicted_topic = max(scores, key=scores.get)
        return {"topic": predicted_topic, "method": 'topic_model'}
        
        
    
    def get_topic_scores(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Get classification scores for all topics"""
        scores = {}
        
        for topic in self.topics:
            score = self._calculate_topic_score(features, topic)
            scores[topic] = float(score)
        
        return scores
    
    def _calculate_topic_score(self, features: Dict[str, Any], topic: str) -> float:
        """Calculate cosine similarity between email and topic embeddings"""
        # Get email embedding from features (now a list/array)
        email_embedding = features.get("email_embeddings_average_embedding", None)

        if email_embedding is None:
            return 0.0

        # Convert to numpy array if it's a list
        if isinstance(email_embedding, list):
            email_embedding = np.array(email_embedding)

        # Get pre-computed topic embedding
        topic_embedding = self.topic_embeddings[topic]

        # Calculate cosine similarity
        # cosine_similarity = dot(A, B) / (||A|| * ||B||)
        dot_product = np.dot(email_embedding, topic_embedding)
        email_norm = np.linalg.norm(email_embedding)
        topic_norm = np.linalg.norm(topic_embedding)

        if email_norm == 0 or topic_norm == 0:
            return 0.0

        cosine_similarity = dot_product / (email_norm * topic_norm)

        # Cosine similarity is between -1 and 1, but for text it's usually positive
        # Normalize to 0-1 range for better interpretability
        normalized_score = (cosine_similarity + 1) / 2

        return float(normalized_score)
    
    def get_topic_description(self, topic: str) -> str:
        """Get description for a specific topic"""
        return self.topic_data[topic]['description']
    
    def get_all_topics_with_descriptions(self) -> Dict[str, str]:
        """Get all topics with their descriptions"""
        return {topic: self.get_topic_description(topic) for topic in self.topics}