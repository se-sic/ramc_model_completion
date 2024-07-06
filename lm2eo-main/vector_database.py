# general
import os, sys
from typing import Any, Iterable
from functools import partial
import logging

from typing import Dict, List

# random number stuff
import random

# config (e.g., for openai related stuff like apikey)
from config import model_prices, load_openai_key
from common import ask_for_proceed, choose_pytorch_device

# data wrangeling
import pandas as pd
import numpy as np

# chromadb vector database
import chromadb
from chromadb.config import Settings


# import langchain dependencies
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import JSONLoader
from langchain.docstore.document import Document
from langchain.math_utils import cosine_similarity

# TODO use a fixed random seed for reproducability

logger = logging.getLogger()

FALLBACK_LLM_MODEL = "all-MiniLM-L6-v2"

class ChangeGraphVectorDB():
    """
    A wrapper class for a Vector DB (currently supporting chroma as backend.)
    
    Usage Example:
    .. code-block:: python
        vector_db = ChangeGraphVectorDB(database_path, False)
        vector_db.built_up_db(input_path)
    
        count_train = len(vector_db.vector_db.get(where={"scope": "train"})["ids"])
        count_test = len(vector_db.vector_db.get(where={"scope": "test"})["ids"])
        logger.info(f"There are {vector_db.vector_db._collection.count()} graphs (train: {count_train}, test: {count_test}) in the database.")
    
        result: List[Document] = vector_db.k_most_diverse_elements_strong(2, "test")
        logger.info(result)
        logger.info(f"Found {len(result)} different documents.")  
    """
    
    
    def __init__(self, path_db: str = "./chroma_db", use_openai_embeddings: bool = False):        
        self.path_db = path_db
        self.use_openai_embeddings = use_openai_embeddings
        
        # Load the embedding model function
        if self.use_openai_embeddings:
            logger.info(f"Loading OpenAI embedding model.")
            from langchain.embeddings.openai import OpenAIEmbeddings
            self.embedding_function = OpenAIEmbeddings(show_progress_bar=True)
        else:
            device = choose_pytorch_device() # we choose the "best" hardware if we compute embeddings locally
            embedding_model_id = FALLBACK_LLM_MODEL
            logger.info(f"Loading embedding model: {embedding_model_id}")
            logger.info(f"Using device: {device}")
            # create the open-source embedding function
            self.embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model_id, model_kwargs = {'device': device})
        
        self.content_key = 'full_graph' # the key in the jsons to be used for the to-be-embedded content.
        
        self.vector_db = None # The Vector database
               
        # TODO also store config in the database that this can also be retrieved from a database (e.g., the embedding model to be used)

    def _iterate_folder(self, input_path: str, file_name:str = "train_samples.jsonl"):
        # Loop over all datasets
        for folder_name in os.listdir(input_path):
            input_ds_path = input_path + '/' + folder_name
            # Skip files in the input_path
            if not os.path.isdir(input_ds_path):
                continue
            
            file_path = input_ds_path + '/' + file_name
            if not os.path.exists(file_path):
                logger.warning(f"Did not find a corresponding file to load: {file_path}")
            yield folder_name, file_path
            
        
    def _create_documents(self, doc_gen: Iterable, scope: str = 'train') -> List[Document]:
        """
        
        This method iterates over an Iterable of folder_names and file_path and
        reads the files as jsonl, create documents from the json filed "full_graph" in the json documents.

        Args:
            doc_gen (Iterable): An Iterable of tuples (folder_name, file_path)

        Returns:
            List[Document]: A list of documents from the JSON-List.
        """
        
        docs: List[Document] = []
        for dataset, file_path in doc_gen:
            loader = JSONLoader(
                file_path=file_path,
                jq_schema='.',
                json_lines=True,
                content_key=self.content_key,
                metadata_func=partial(self._metadata_func, dataset, scope)
            )
            docs.extend(loader.load())
            
        # TODO for large dbs, we should probably proceed with a generator instead of loading the entire list.
        return docs

    def _metadata_func(self, dataset:str, scope:str, record: dict, metadata: dict) -> dict:
        """
        
        To extract metadata from given dicts and returns them. Basically a dict-to-dict transformation.

        Args:
            dataset (str): _description_
            record (dict): _description_
            metadata (dict): _description_

        Returns:
            dict: _description_
        """
        
        metadata["prompt"] = record.get("prompt")
        metadata["completion"] = record.get("completion")
        metadata["token_count"] = record.get("token_count")
        metadata["graph_id"] = record.get("graph_id")
        metadata["global_graph_id"] = dataset + str(record.get("graph_id"))
        metadata["dataset"] = dataset
        metadata["scope"] = scope
        metadata["number_of_edges_graph"] = record.get("number_of_edges_graph")
        metadata["number_of_removed_items"] = record.get("number_of_removed_items")
        metadata["change_type"] = str(record.get("change_type")) # Since we can have multiple change types (in case we have multiple edges for completion) this is a list of change types.
        metadata["last_edge_change_type"] = record.get("change_type")[-1]

        # Cut-off specific location on machine and replace by machine independent relative path
        if "source" in metadata:
            source = metadata["source"].split("/")
            source = source[source.index("experiment_samples"):]
            metadata["source"] = "/".join(source)

        return metadata
    
    def load_existing(self):
        if os.path.exists(self.path_db):
            logger.info(f"Database found... loading database: {self.path_db}")
            client = chromadb.PersistentClient(self.path_db, Settings(anonymized_telemetry=False))
            vector_db = Chroma(embedding_function=self.embedding_function, persist_directory=self.path_db, client=client)
            self.vector_db = vector_db
        else:
            logger.warn(f"Database not found: {self.path_db}")


    def built_up_db(self, path_to_datasets: str):
       
        # Load database if it exists already
        if os.path.exists(self.path_db):
            logger.info(f"Database found... loading database: {self.path_db}")
            client = chromadb.PersistentClient(self.path_db, Settings(anonymized_telemetry=False))
            vector_db = Chroma(embedding_function=self.embedding_function, persist_directory=self.path_db, client=client)
            self.vector_db = vector_db
            return
        
        # get iterator over folders
        logger.info(f"Loading train documents...")
        folder_iterator = self._iterate_folder(path_to_datasets, file_name = "train_samples.jsonl")
        documents_train: List[Document] = self._create_documents(folder_iterator, scope="train")
        
        logger.info(f"Loading test documents...")
        folder_iterator = self._iterate_folder(path_to_datasets, file_name = "test_samples.jsonl")
        documents_test: List[Document] = self._create_documents(folder_iterator, scope="test")
        
        documents = documents_test + documents_train
        
        if self.use_openai_embeddings:
            # calculate pricing
            total_tokens = sum([document.metadata['token_count'] for document in documents])
            expected_price = total_tokens * model_prices['embedding']
            print(f"The total cost for this experiment are expected to be: {expected_price}USD")
            
            ask_for_proceed()

        logger.info(f"Building up vector database.")
        vector_db = Chroma.from_documents(documents, self.embedding_function, persist_directory=self.path_db)#, client_settings=Settings(anonymized_telemetry=False)) #collection_name="SMO-SysML",
        # Persist db
        logger.info(f"Persisting db to: {self.path_db}")
        vector_db.persist()
        self.vector_db = vector_db
        
    def query_db(self, query:str , scope: str, k: int = 10, fetch_k: int = 100) -> List[Document]:
        retriever = self.vector_db.as_retriever(search_type="mmr", search_kwargs={"fetch_k": fetch_k, "k": k, "filter": {"scope": scope}})
        docs = retriever.get_relevant_documents(query)
        return docs

    def k_most_diverse_elements_mmr(self, k: int , scope: str):
        """
        For a more powerful version then mmr, see k_most_diverse_elements_strong.

        Args:
            k (int): _description_
            scope (str): _description_

        Returns:
            _type_: _description_
        """
        all_count = self.vector_db._collection.count()
        retriever = self.vector_db.as_retriever(search_type="mmr", search_kwargs={"fetch_k": all_count, "k": k, "filter": {"scope": scope}})
        query = ""
        docs = retriever.get_relevant_documents(query)
        return docs
    
    @classmethod
    def _dict_to_documents(cls, vector_db_result: dict) -> List[Document]:
        """
        Transforms the output of a vector db get call (i.e., a dictionary of lists of ids, docs, etc.) 
        to a list of Document objects.
        
        TODO A little bit ugly is the Chroma API right now, thats why we need this method.
        This should already been done via the Chroma wrapper but it isn't so we probably need an abstraction. 
        This would also allow us to migrate to other vector stores quickly

        Args:
            vector_db_result (dict): A dictionary of the texts, ids, and metadatas

        Returns:
            List[Document]: A list of langchang API documents.
        """
        # get the length of the result list:
        result_count = len(vector_db_result['ids'])
        return [Document(id=vector_db_result['ids'][idx], page_content=vector_db_result['documents'][idx], metadata=vector_db_result['metadatas'][idx]) for idx in range(result_count)]
    
    def get_via_ids(self, ids: List[str]) -> List[Document]:
        # There is a bug Chroma... if ids is empty everything is returned... we therefore have to catch this here
        if len(ids) == 0:
            return []
        
        retrieved_docs = self.vector_db._collection.get(ids, include=["documents", "metadatas"])
        # Transform to the "Document interface"
        docs = self._dict_to_documents(retrieved_docs)
        return docs
    
    def _get_by_graph_id(self, graph_id: int, scope: str):
        """
        This is an ugly workaround since similarty search via chroma does not return ids.
        TODO find a proper work around for the really ugly langchain chroma API.
        """
        retrieved_dicts =  self.vector_db._collection.get(where={"$and": [{"global_graph_id": graph_id}, {"scope": scope}]}, include=["embeddings"])
        assert len(retrieved_dicts["ids"]) == 1 # we expect graph_id to be an id...
        
        return retrieved_dicts["ids"], retrieved_dicts["embeddings"]   
    
    def query_k_most_diverse_strong(self, query: str, k:int, k_retrieve:int, scope: str, additional_filter=None, num_of_iterations: int =100, distance_function=cosine_similarity) -> list[Document]:
        if k <= 0:
            return []
        assert k_retrieve >= k
        
        # Setup filter
        if additional_filter is None:
            where={"scope": scope}
        else:
            where={"$and": [additional_filter, {"scope": scope}]}
        
        if k == 1:
            # in this case we do not need diversity sampling at all. We just return the closest match.
            return self.vector_db.similarity_search(query, k, filter=where)
            
        # First we retrieve the k_retrieve most similar documents
        retriever = self.vector_db.as_retriever(search_type="similarity", search_kwargs={"k": k_retrieve, "filter": where})
        most_similar = retriever.get_relevant_documents(query)
        
        # ugly workaround to get ids and embeddings
        graph_ids = [doc.metadata["global_graph_id"] for doc in most_similar]
        most_similar_ids = [self._get_by_graph_id(graph_id, scope)[0][0] for graph_id in graph_ids]
        most_similar_embeddings = [self._get_by_graph_id(graph_id, scope)[1][0] for graph_id in graph_ids]

        # Then we pic the k most diverse ones of them
        
        # Define the similarity function
        def _sim_function(id_1, id_2):
            idx_1 = most_similar_ids.index(id_1)
            idx_2 = most_similar_ids.index(id_2)
            embedding_1 = most_similar_embeddings[idx_1]
            embedding_2 = most_similar_embeddings[idx_2]
            return distance_function([embedding_1], [embedding_2])
        #Retrieve the k most divers among the retrieved most similar ones.
        k_most_diverse_ids = self._k_most_diverse_elements_strong(set(most_similar_ids), k, num_of_iterations, _sim_function)
        
        # Load the corrsponding Docs
        retrieved_docs = self.get_via_ids(k_most_diverse_ids)
        return retrieved_docs

    
    def k_most_diverse_elements_strong(self, k:int, scope: str, additional_filter = None, num_of_iterations: int =100, distance_function=cosine_similarity) -> list[Document]:
        # Setup filter
        if additional_filter is None:
            where={"scope": scope}
        else:
            where={"$and": [additional_filter, {"scope": scope}]}
        
        all_test_docs = self.vector_db.get(where=where, include=["embeddings"])
        all_test_embeddings = all_test_docs["embeddings"]
        all_test_ids = all_test_docs["ids"]
        
        def _sim_function(id_1, id_2):
            idx_1 = all_test_ids.index(id_1)
            idx_2 = all_test_ids.index(id_2)
            embedding_1 = all_test_embeddings[idx_1]
            embedding_2 = all_test_embeddings[idx_2]
            return distance_function([embedding_1], [embedding_2])
        k_most_diverse_ids = self._k_most_diverse_elements_strong(set(all_test_ids), k, num_of_iterations, _sim_function)
        
        retrieved_docs = self.get_via_ids(k_most_diverse_ids)
        return retrieved_docs
    
    def _k_most_diverse_elements_strong(self, element_set: set[Any], sample_size, num_of_iterations=None, distance_function=None) -> set[Any]:
        """
        Approximate algorithm to find the sample_size most distant elements (i.e. the minimal distance is maximal) in the given list of elements.
        We randomly choose sample_size nodes and then randomly select one of the nodes and maximize the minimal distance. 
        
        # TODO our algorithm might (in theory) get stuck in local maxima (e.g., orthic triangle of a isosceles triangle), maybe consider random restarts and the best of these randoms

        Exact solution could for example be done by using a CSP solver. The constraints are given by trying to sample sample_size samples and iteratively removing the smallest "pairs".
        Brute force exact solution would be to consider all possible choice of sample_size elements and compute their minimal distance. But we will have n choose sample_size choices...
        Note that our algorithm does not only maximize the minimal distance but also often to maximize the other distances.
        Another approximate solution that is often mentioned is to add the most distant points iteratively. But this won't give good solutions in many cases, e.g. the circle point set with 3 samples.

        Args:
            element_set (_type_): The original element set.
            sample_size (_type_): The number of samples to draw.
            num_of_iterations (_type_, optional): Number of iterations to run the algorithm for. Defaults to None.
            distance_function (_type_, optional): A distance function that is defined on the elements of element_set. Defaults to node_label_distance.

        Returns:
            _type_: A sample of size sample_size from element_set where the elements are as distinct as possible (approx.)
        """

        # Ensure element is set
        assert isinstance(element_set, set)
        # Ensure more than one element should be drawn
        assert sample_size > 0
        
        if len(element_set) == 0:
            # No elements in database with the given criteria.... return empty set
            logger.warning(f"Empty element set for sampling.")
            return {}
        
        logger.debug(f"Diversity Sampling: Size Elementset: {len(element_set)}. Samples: {sample_size}")
        
        # Compute number of iterations
        if num_of_iterations is None:
            num_of_iterations = int(sample_size*1.2)

        if element_set is None or len(element_set) == 0:
            return None

        if len(element_set) == 1:
            return element_set.pop()
        
        if sample_size == 1:
            logger.warning(f"Sample size only one. Choosing random element. Probably consider some other logic for sampling if sample size is equal to 1.")
            return random.sample(element_set, 1)
        
               
        # Choose some initial random sample from the list of elements
        if (len(list(element_set)) <= sample_size):
            logger.warning("Attention: less elements than number of samples to sample from.") 
            sample_size = min(sample_size, len(element_set))

        # Array to ensure that all tokens are touched
        coupons = list(range(sample_size))
        # Initial sample
        most_distant = random.sample(list(element_set), sample_size )
        
        # choose random pivot and remove it from the list from the most_distant list, sicne we want to maximize the minimal distance to all other elements
        pivot_index = random.choice(coupons)
        coupons.remove(pivot_index)

        pivot = most_distant[pivot_index]

        candidates = list(element_set - set(most_distant))
        # append the pivot
        candidates.append(pivot)
        # TODO might be slow for batches
        candidate_distances = np.array([[distance_function(candidate, selected) for selected in most_distant] for candidate in candidates])

        for _ in range(num_of_iterations):
            # TODO do we really need a copy here?
            candidate_distances_temp = np.copy(candidate_distances)
            candidate_distances_temp = np.delete(candidate_distances_temp,[pivot_index],1) # Delete the distances to the pivot in the current selection most_distant
            best_candidate_index = np.argmax(np.apply_along_axis(np.min, 1, candidate_distances_temp)) # this finds the index of the candidate that has the largest minimal distance to the currently most distant ones
            best_candidate = candidates[best_candidate_index]

            most_distant[pivot_index] = best_candidate # replace the pivot by the candidate
            candidate_distances[:, pivot_index] = np.array([distance_function(best_candidate, candidate) for candidate in candidates]) # and update the candidate distance matrix for the next iteration

            # choose new pivot and continue
            # If coupons are empty, refill
            if len(coupons) == 0:
                coupons = list(range(sample_size))
                
            # And compute the pivot for the next iteration
            pivot_index = random.choice(coupons)
            coupons.remove(pivot_index)
            pivot = most_distant[pivot_index]
            candidates[best_candidate_index] = pivot
            candidate_distances[best_candidate_index] = np.array([distance_function(selected, pivot) for selected in most_distant])
        return most_distant
    
    def get_filtered(self, where: dict={}) -> List[Document]:
        return self._dict_to_documents(self.vector_db.get(where=where, include=["documents", "metadatas"]))
    
    def get_distribution(self, attribute:str, where:dict={}) -> Dict[str, float]:
        """Get the statistics for the given field (i.e., how many documents per value of the given attribute).

        Args:
            attribute (str): The attribute for which to compute the statistics.
            where (dict, optional): Further constraint of the dataset. Defaults to {}, ie.e no constraint.
        """
        all_docs = self.get_filtered(where=where)
        all_count = len(all_docs)
        all_attribute_values_list = [doc.metadata[attribute] for doc in all_docs]
        all_docs_stats = {attribute_value: all_attribute_values_list.count(attribute_value)/all_count for attribute_value in set(all_attribute_values_list)}
        return all_docs_stats
    
def main(input_path: str, database_path: str, use_open_ai_embeddings: bool=False):
    # Logger settings
    logging.basicConfig(level=logging.INFO)
    
    vector_db = ChangeGraphVectorDB(database_path, use_open_ai_embeddings)
    vector_db.built_up_db(input_path)
    
    count_train = len(vector_db.get_filtered(where={"scope": "train"}))
    count_test = len(vector_db.get_filtered(where={"scope": "test"}))
    logger.info(f"There are {vector_db.vector_db._collection.count()} graphs (train: {count_train}, test: {count_test}) in the database.")

if __name__ == "__main__":
    """
    Executes when called as python module.
    """
    if len(sys.argv) == 3:
        use_open_ai_embeddings = False
        main(sys.argv[1], sys.argv[2], use_open_ai_embeddings)
    elif len(sys.argv) == 4:
        use_open_ai_embeddings = sys.argv[3] == "True"
        main(sys.argv[1], sys.argv[2], use_open_ai_embeddings)
    else:
        logger.error("Unexpected number of arguments. Call like python vector_database.py [input_path] [data_base_path] [True|False Use OpenAI Embeddings].")

    
#main("./../model_completion_dataset/Siemens_Mobility/results/experiment_samples/", "./../model_completion_dataset/Siemens_Mobility/results/experiment_samples/vector_db")

#def set_metadata_value(metadata: dict, metadata_key: str, new_value: str):
#    metadata[metadata_key] = new_value
#    return metadata

# Update all documents metadata (with single value)
# TODO this does currently update embeddings (which is expensive). We therefore should avoid updating meta-data this way.
#def update_metadata(db: Chroma, metadata_key: str, new_value: str):
#    all_entries = db.get() # get all documents
#    all_ids = all_entries['ids']
#    all_embeddings = all_entries['embeddings']
#    all_metadata = all_entries['metadatas']
#    all_documents = all_entries['documents']
#    # add metadata value
#    all_metadata = list(map(lambda metadata: set_metadata_value(metadata, metadata_key, new_value), all_metadata))
#    # update in db
#    db._collection.update(all_ids, all_embeddings, all_metadata, all_documents)
#update_metadata(vector_db, metadata_key='scope', new_value='train')