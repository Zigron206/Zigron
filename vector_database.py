# vector_database.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from typing import Dict, List
import json
import os
from datetime import datetime

class DepressionKnowledgeBase:
    def __init__(self, index_path: str = "depression_index"):
        """Initialize the knowledge base"""
        self.index_path = index_path
        self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.dimension = 768
        
        # Load or initialize index and metadata
        if os.path.exists(f"{index_path}.index"):
            try:
                self.index = faiss.read_index(f"{index_path}.index")
                with open(f"{index_path}.json", 'r') as f:
                    data = json.load(f)
                    self.documents = data.get('documents', [])
                    self.metadatas = data.get('metadatas', [])
            except Exception as e:
                print(f"Error loading existing index: {e}")
                self._initialize_empty()
        else:
            self._initialize_empty()

    def _initialize_empty(self):
        """Initialize empty index and storage"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.metadatas = []

    def initialize_knowledge_base(self, excel_file: str) -> Dict:
        """One-time initialization of the knowledge base"""
        try:
            df = pd.read_excel(excel_file)
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            # Clear existing data
            self._initialize_empty()
            
            # Process and categorize content
            for _, row in df.iterrows():
                for col in df.columns:
                    if pd.notna(row[col]):
                        content = str(row[col]).strip()
                        if content:
                            # Determine content category
                            category = self._categorize_content(col, content)
                            
                            # Add to index
                            embedding = self.embedding_model.encode([content])
                            self.index.add(np.array(embedding, dtype=np.float32))
                            
                            # Store document and metadata
                            self.documents.append(content)
                            self.metadatas.append({
                                'section': category,
                                'source': col,
                                'added_at': datetime.now().isoformat()
                            })

            # Save index and metadata
            self._save_knowledge_base()
            
            return {
                "status": "success",
                "documents_processed": len(self.documents),
                "index_path": self.index_path
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def _categorize_content(self, column: str, content: str) -> str:
        """Categorize content based on column name and content"""
        column_lower = column.lower()
        content_lower = content.lower()
        
        categories = {
            'diagnostic_criteria': ['criteria', 'diagnosis', 'diagnostic'],
            'symptoms': ['symptom', 'mood', 'feeling', 'depression'],
            'treatment': ['treatment', 'therapy', 'intervention', 'therapeutic'],
            'assessment': ['assessment', 'evaluation', 'scale', 'measure'],
            'differential_diagnosis': ['differential', 'distinguish', 'versus', 'vs']
        }
        
        # Check both column name and content for category matches
        for category, keywords in categories.items():
            if any(keyword in column_lower for keyword in keywords) or \
               any(keyword in content_lower for keyword in keywords):
                return category
                
        return 'general'

    def _save_knowledge_base(self):
        """Save index and metadata"""
        try:
            faiss.write_index(self.index, f"{self.index_path}.index")
            with open(f"{self.index_path}.json", 'w') as f:
                json.dump({
                    'documents': self.documents,
                    'metadatas': self.metadatas
                }, f)
        except Exception as e:
            print(f"Error saving knowledge base: {e}")

    def query_knowledge_base(self, query: str) -> Dict:
        """Query the knowledge base and return comprehensive JSON response"""
        try:
            # Encode query
            query_embedding = self.embedding_model.encode([query])
            
            # Search in index
            k = min(5, len(self.documents)) if self.documents else 0
            if k == 0:
                return self._format_empty_response(query)
                
            distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), k)
            
            # Process results
            results = {}
            for idx, distance in zip(indices[0], distances[0]):
                if idx != -1 and idx < len(self.documents):
                    metadata = self.metadatas[idx]
                    section = metadata.get('section', 'general')
                    
                    if section not in results:
                        results[section] = []
                        
                    if distance < 1.5:  # Relevance threshold
                        results[section].append({
                            'content': self.documents[idx],
                            'source': metadata.get('source', 'unknown'),
                            'relevance_score': float(1 / (1 + distance))
                        })
            
            # Format response
            return {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'results': {k: v for k, v in results.items() if v},
                'total_matches': sum(len(v) for v in results.values())
            }
            
        except Exception as e:
            return {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'results': {}
            }

    def _format_empty_response(self, query: str) -> Dict:
        """Format response when no results are found"""
        return {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'results': {},
            'total_matches': 0
        }