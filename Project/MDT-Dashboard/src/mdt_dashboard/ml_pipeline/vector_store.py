"""
Vector Database Integration for ML Feature Storage and Similarity Search
Supports multiple vector databases for scalable ML operations
"""

import logging
import time
import hashlib
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import json
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class VectorConfig:
    """Configuration for vector database operations."""
    
    # Database selection
    db_type: str = "faiss"  # faiss, chroma, pinecone, weaviate
    
    # Connection settings
    host: str = "localhost"
    port: int = 6333
    api_key: Optional[str] = None
    environment: str = "development"
    
    # Vector settings
    dimension: int = 768
    metric: str = "cosine"  # cosine, euclidean, dot_product
    index_type: str = "HNSW"  # HNSW, IVF, Flat
    
    # Performance settings
    ef_construction: int = 200
    m: int = 16
    batch_size: int = 100
    
    # Storage settings
    persist_directory: str = "./vector_store"
    collection_name: str = "ml_features"


class VectorDatabase(ABC):
    """Abstract base class for vector database implementations."""
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the vector database."""
        pass
    
    @abstractmethod
    def create_collection(self, name: str, dimension: int) -> bool:
        """Create a new collection."""
        pass
    
    @abstractmethod
    def insert_vectors(self, vectors: List[List[float]], 
                      metadata: List[Dict[str, Any]] = None,
                      ids: List[str] = None) -> bool:
        """Insert vectors into the database."""
        pass
    
    @abstractmethod
    def search_similar(self, query_vector: List[float], 
                      top_k: int = 10, 
                      filter_dict: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors by IDs."""
        pass
    
    @abstractmethod
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        pass


class FAISSVectorDB(VectorDatabase):
    """FAISS vector database implementation."""
    
    def __init__(self, config: VectorConfig):
        self.config = config
        self.index = None
        self.metadata_store = {}
        self.id_to_index = {}
        self.index_to_id = {}
        self.current_index = 0
        
    def connect(self) -> bool:
        """Connect to FAISS (local implementation)."""
        try:
            import faiss
            
            # Create index based on configuration
            if self.config.index_type == "HNSW":
                if self.config.metric == "cosine":
                    # Normalize vectors for cosine similarity
                    self.index = faiss.IndexHNSWFlat(self.config.dimension, self.config.m)
                else:
                    self.index = faiss.IndexHNSWFlat(self.config.dimension, self.config.m)
            elif self.config.index_type == "IVF":
                quantizer = faiss.IndexFlatL2(self.config.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.config.dimension, 100)
            else:
                self.index = faiss.IndexFlatL2(self.config.dimension)
            
            logger.info(f"Connected to FAISS with {self.config.index_type} index")
            return True
            
        except ImportError:
            logger.error("FAISS not available. Install with: pip install faiss-cpu")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to FAISS: {e}")
            return False
    
    def create_collection(self, name: str, dimension: int) -> bool:
        """Create a new collection (reinitialize index)."""
        self.config.collection_name = name
        self.config.dimension = dimension
        return self.connect()
    
    def insert_vectors(self, vectors: List[List[float]], 
                      metadata: List[Dict[str, Any]] = None,
                      ids: List[str] = None) -> bool:
        """Insert vectors into FAISS index."""
        try:
            import faiss
            import numpy as np
            
            if self.index is None:
                logger.error("Index not initialized. Call connect() first.")
                return False
            
            # Convert to numpy array
            vectors_np = np.array(vectors, dtype=np.float32)
            
            # Normalize for cosine similarity
            if self.config.metric == "cosine":
                faiss.normalize_L2(vectors_np)
            
            # Generate IDs if not provided
            if ids is None:
                ids = [f"vec_{self.current_index + i}" for i in range(len(vectors))]
            
            # Train index if needed
            if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                self.index.train(vectors_np)
            
            # Add vectors
            start_index = self.current_index
            self.index.add(vectors_np)
            
            # Store metadata and ID mappings
            for i, (vector_id, meta) in enumerate(zip(ids, metadata or [{}] * len(vectors))):
                index_pos = start_index + i
                self.id_to_index[vector_id] = index_pos
                self.index_to_id[index_pos] = vector_id
                self.metadata_store[vector_id] = meta
            
            self.current_index += len(vectors)
            
            logger.info(f"Inserted {len(vectors)} vectors into FAISS index")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert vectors: {e}")
            return False
    
    def search_similar(self, query_vector: List[float], 
                      top_k: int = 10, 
                      filter_dict: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors in FAISS."""
        try:
            import faiss
            import numpy as np
            
            if self.index is None:
                return []
            
            # Convert query to numpy array
            query_np = np.array([query_vector], dtype=np.float32)
            
            # Normalize for cosine similarity
            if self.config.metric == "cosine":
                faiss.normalize_L2(query_np)
            
            # Search
            distances, indices = self.index.search(query_np, top_k)
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:  # Invalid index
                    continue
                
                vector_id = self.index_to_id.get(idx, f"unknown_{idx}")
                metadata = self.metadata_store.get(vector_id, {})
                
                # Apply filters if specified
                if filter_dict:
                    if not self._matches_filter(metadata, filter_dict):
                        continue
                
                results.append({
                    'id': vector_id,
                    'score': float(dist),
                    'metadata': metadata
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors by IDs (FAISS doesn't support deletion, would need rebuild)."""
        logger.warning("FAISS doesn't support vector deletion. Index would need to be rebuilt.")
        return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get FAISS index information."""
        if self.index is None:
            return {}
        
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.config.dimension,
            'index_type': self.config.index_type,
            'metric': self.config.metric,
            'is_trained': getattr(self.index, 'is_trained', True)
        }
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True


class ChromaVectorDB(VectorDatabase):
    """ChromaDB vector database implementation."""
    
    def __init__(self, config: VectorConfig):
        self.config = config
        self.client = None
        self.collection = None
    
    def connect(self) -> bool:
        """Connect to ChromaDB."""
        try:
            import chromadb
            
            # Create client
            if self.config.host == "localhost":
                self.client = chromadb.PersistentClient(path=self.config.persist_directory)
            else:
                self.client = chromadb.HttpClient(
                    host=self.config.host,
                    port=self.config.port
                )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"dimension": self.config.dimension}
            )
            
            logger.info(f"Connected to ChromaDB collection: {self.config.collection_name}")
            return True
            
        except ImportError:
            logger.error("ChromaDB not available. Install with: pip install chromadb")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            return False
    
    def create_collection(self, name: str, dimension: int) -> bool:
        """Create a new ChromaDB collection."""
        try:
            if self.client is None:
                return False
            
            self.collection = self.client.get_or_create_collection(
                name=name,
                metadata={"dimension": dimension}
            )
            
            self.config.collection_name = name
            self.config.dimension = dimension
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False
    
    def insert_vectors(self, vectors: List[List[float]], 
                      metadata: List[Dict[str, Any]] = None,
                      ids: List[str] = None) -> bool:
        """Insert vectors into ChromaDB."""
        try:
            if self.collection is None:
                return False
            
            # Generate IDs if not provided
            if ids is None:
                ids = [f"vec_{int(time.time() * 1000000)}_{i}" for i in range(len(vectors))]
            
            # Prepare metadata
            if metadata is None:
                metadata = [{}] * len(vectors)
            
            # Insert in batches
            batch_size = self.config.batch_size
            for i in range(0, len(vectors), batch_size):
                batch_end = min(i + batch_size, len(vectors))
                
                self.collection.add(
                    embeddings=vectors[i:batch_end],
                    metadatas=metadata[i:batch_end],
                    ids=ids[i:batch_end]
                )
            
            logger.info(f"Inserted {len(vectors)} vectors into ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert vectors: {e}")
            return False
    
    def search_similar(self, query_vector: List[float], 
                      top_k: int = 10, 
                      filter_dict: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors in ChromaDB."""
        try:
            if self.collection is None:
                return []
            
            # Build where clause for filtering
            where_clause = filter_dict if filter_dict else None
            
            # Query
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where_clause
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'score': results['distances'][0][i] if 'distances' in results else 0.0,
                    'metadata': results['metadatas'][0][i] if 'metadatas' in results else {}
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors from ChromaDB."""
        try:
            if self.collection is None:
                return False
            
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} vectors from ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get ChromaDB collection information."""
        try:
            if self.collection is None:
                return {}
            
            count = self.collection.count()
            return {
                'total_vectors': count,
                'collection_name': self.config.collection_name,
                'dimension': self.config.dimension
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}


class PineconeVectorDB(VectorDatabase):
    """Pinecone vector database implementation."""
    
    def __init__(self, config: VectorConfig):
        self.config = config
        self.index = None
    
    def connect(self) -> bool:
        """Connect to Pinecone."""
        try:
            import pinecone
            
            # Initialize Pinecone
            pinecone.init(
                api_key=self.config.api_key,
                environment=self.config.environment
            )
            
            # Connect to index
            self.index = pinecone.Index(self.config.collection_name)
            
            logger.info(f"Connected to Pinecone index: {self.config.collection_name}")
            return True
            
        except ImportError:
            logger.error("Pinecone not available. Install with: pip install pinecone-client")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {e}")
            return False
    
    def create_collection(self, name: str, dimension: int) -> bool:
        """Create a new Pinecone index."""
        try:
            import pinecone
            
            # Create index
            pinecone.create_index(
                name=name,
                dimension=dimension,
                metric=self.config.metric
            )
            
            self.config.collection_name = name
            self.config.dimension = dimension
            
            # Connect to new index
            return self.connect()
            
        except Exception as e:
            logger.error(f"Failed to create Pinecone index: {e}")
            return False
    
    def insert_vectors(self, vectors: List[List[float]], 
                      metadata: List[Dict[str, Any]] = None,
                      ids: List[str] = None) -> bool:
        """Insert vectors into Pinecone."""
        try:
            if self.index is None:
                return False
            
            # Generate IDs if not provided
            if ids is None:
                ids = [f"vec_{int(time.time() * 1000000)}_{i}" for i in range(len(vectors))]
            
            # Prepare upsert data
            upsert_data = []
            for i, vector in enumerate(vectors):
                item = {
                    'id': ids[i],
                    'values': vector
                }
                if metadata and i < len(metadata):
                    item['metadata'] = metadata[i]
                
                upsert_data.append(item)
            
            # Upsert in batches
            batch_size = self.config.batch_size
            for i in range(0, len(upsert_data), batch_size):
                batch = upsert_data[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            logger.info(f"Inserted {len(vectors)} vectors into Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert vectors: {e}")
            return False
    
    def search_similar(self, query_vector: List[float], 
                      top_k: int = 10, 
                      filter_dict: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors in Pinecone."""
        try:
            if self.index is None:
                return []
            
            # Query
            response = self.index.query(
                vector=query_vector,
                top_k=top_k,
                filter=filter_dict,
                include_metadata=True
            )
            
            # Format results
            results = []
            for match in response.matches:
                results.append({
                    'id': match.id,
                    'score': float(match.score),
                    'metadata': match.metadata or {}
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors from Pinecone."""
        try:
            if self.index is None:
                return False
            
            self.index.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} vectors from Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get Pinecone index information."""
        try:
            if self.index is None:
                return {}
            
            stats = self.index.describe_index_stats()
            return {
                'total_vectors': stats.total_vector_count,
                'dimension': stats.dimension,
                'index_fullness': stats.index_fullness
            }
            
        except Exception as e:
            logger.error(f"Failed to get index info: {e}")
            return {}


class VectorFeatureStore:
    """High-level feature store using vector databases."""
    
    def __init__(self, config: VectorConfig):
        self.config = config
        self.db = self._create_database()
        self.connected = False
    
    def _create_database(self) -> VectorDatabase:
        """Create the appropriate vector database instance."""
        if self.config.db_type == "faiss":
            return FAISSVectorDB(self.config)
        elif self.config.db_type == "chroma":
            return ChromaVectorDB(self.config)
        elif self.config.db_type == "pinecone":
            return PineconeVectorDB(self.config)
        else:
            raise ValueError(f"Unsupported database type: {self.config.db_type}")
    
    def initialize(self) -> bool:
        """Initialize the vector feature store."""
        self.connected = self.db.connect()
        return self.connected
    
    def store_model_embeddings(self, model_name: str, embeddings: List[List[float]], 
                             features: List[Dict[str, Any]]) -> bool:
        """Store model embeddings with associated features."""
        if not self.connected:
            logger.error("Database not connected")
            return False
        
        # Generate IDs
        ids = [f"{model_name}_{self._hash_features(feat)}" for feat in features]
        
        # Add model name to metadata
        metadata = []
        for feat in features:
            meta = feat.copy()
            meta['model_name'] = model_name
            meta['timestamp'] = time.time()
            metadata.append(meta)
        
        return self.db.insert_vectors(embeddings, metadata, ids)
    
    def find_similar_features(self, query_embedding: List[float], 
                            model_name: str = None, 
                            top_k: int = 10) -> List[Dict[str, Any]]:
        """Find similar features based on embedding similarity."""
        if not self.connected:
            return []
        
        # Build filter
        filter_dict = {}
        if model_name:
            filter_dict['model_name'] = model_name
        
        return self.db.search_similar(query_embedding, top_k, filter_dict)
    
    def get_feature_drift_candidates(self, current_embeddings: List[List[float]], 
                                   model_name: str, 
                                   threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Find features that might indicate drift."""
        drift_candidates = []
        
        for embedding in current_embeddings:
            similar = self.find_similar_features(embedding, model_name, top_k=1)
            
            if similar and similar[0]['score'] > threshold:
                drift_candidates.append({
                    'current_embedding': embedding,
                    'historical_match': similar[0],
                    'drift_score': similar[0]['score']
                })
        
        return drift_candidates
    
    def cleanup_old_features(self, model_name: str, days_old: int = 30) -> bool:
        """Clean up old feature embeddings."""
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        
        # This would require querying by timestamp and then deleting
        # Implementation depends on specific database capabilities
        logger.info(f"Cleanup for features older than {days_old} days not fully implemented")
        return True
    
    def get_store_statistics(self) -> Dict[str, Any]:
        """Get feature store statistics."""
        info = self.db.get_collection_info()
        
        # Add custom statistics
        stats = {
            'database_type': self.config.db_type,
            'collection_name': self.config.collection_name,
            'vector_dimension': self.config.dimension,
            'total_vectors': info.get('total_vectors', 0),
            'connected': self.connected
        }
        
        return stats
    
    def _hash_features(self, features: Dict[str, Any]) -> str:
        """Create a hash of feature dictionary for ID generation."""
        feature_str = json.dumps(features, sort_keys=True)
        return hashlib.md5(feature_str.encode()).hexdigest()[:16]


class VectorSearchEngine:
    """Advanced vector search engine with caching and optimization."""
    
    def __init__(self, feature_store: VectorFeatureStore):
        self.feature_store = feature_store
        self.search_cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    def semantic_search(self, query: str, embedding_function: callable, 
                       filters: Dict[str, Any] = None, 
                       top_k: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search using text queries."""
        # Convert query to embedding
        query_embedding = embedding_function(query)
        
        # Search cache key
        cache_key = self._generate_cache_key(query_embedding, filters, top_k)
        
        # Check cache
        if self._is_cache_valid(cache_key):
            logger.debug("Returning cached search results")
            return self.search_cache[cache_key]['results']
        
        # Perform search
        results = self.feature_store.db.search_similar(query_embedding, top_k, filters)
        
        # Cache results
        self.search_cache[cache_key] = {
            'results': results,
            'timestamp': time.time()
        }
        
        return results
    
    def batch_search(self, queries: List[List[float]], 
                    top_k: int = 10) -> List[List[Dict[str, Any]]]:
        """Perform batch vector search for efficiency."""
        results = []
        
        for query in queries:
            result = self.feature_store.db.search_similar(query, top_k)
            results.append(result)
        
        return results
    
    def _generate_cache_key(self, embedding: List[float], 
                          filters: Dict[str, Any], top_k: int) -> str:
        """Generate cache key for search query."""
        key_data = {
            'embedding_hash': hashlib.md5(str(embedding).encode()).hexdigest(),
            'filters': filters or {},
            'top_k': top_k
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached result is still valid."""
        if cache_key not in self.search_cache:
            return False
        
        cache_age = time.time() - self.search_cache[cache_key]['timestamp']
        return cache_age < self.cache_ttl
    
    def clear_cache(self):
        """Clear the search cache."""
        self.search_cache.clear()


# Export main classes
__all__ = [
    "VectorConfig",
    "VectorDatabase",
    "FAISSVectorDB",
    "ChromaVectorDB", 
    "PineconeVectorDB",
    "VectorFeatureStore",
    "VectorSearchEngine"
]
