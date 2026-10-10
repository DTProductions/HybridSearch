from dataset import MSMarco
from rank_bm25 import BM25Okapi
from chromadb.utils import embedding_functions
import chromadb
from itertools import batched

class SemanticSearch:
    def __init__(self, dataset: MSMarco, model_name: str, collection_name: str, client: chromadb.Client):
        self.dataset = dataset

        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

        collection_exists = collection_name in [c.name for c in client.list_collections()]

        collection = client.get_or_create_collection(collection_name, embedding_function=embedding_func)

        if collection_exists:
            print("Collection already exists. Skipping seeding stage...")
        else:
            print("Collection does not exist. Seeding data...")
            passage_ids = []
            passages = []
            for passage_id in self.dataset.passages:
                passage_ids.append(passage_id)
                passages.append(self.dataset.passages[passage_id])

            BATCH_SIZE = 5000
            for ids_batch, passages_batch in zip(batched(passage_ids, BATCH_SIZE), batched(passages, BATCH_SIZE)):                
                collection.add(ids=list(ids_batch), documents=list(passages_batch))

            '''
            collection.add(
                ids=passage_ids,
                documents=passages
            )
            '''

        self.collection = collection

    def search(self, query: str, max_results: int = 10) -> list[str]:
        result = self.collection.query(query_texts=[query], n_results=max_results)

        relevant_passages = [passage for passage in result["documents"][0]]
        return relevant_passages

class LexicalSearch:
    def __init__(self, dataset: MSMarco):
        self.passages = [dataset.passages[passage_id] for passage_id in dataset.passages]
        tokenized_passages = [dataset.passages[passage_id].lower().split() for passage_id in dataset.passages]
        self.bm25 = BM25Okapi(tokenized_passages)

    def search(self, query: str, max_results: int = 10) -> list[str]:
        results = self.bm25.get_top_n(query.lower().split(), self.passages, n=max_results)
        return results

class MixedSearch():
    def __init__(self, lexical: LexicalSearch, semantic: SemanticSearch):
        self.lexical = lexical
        self.semantic = semantic

    def weighted_search(self, query: str, lexical_weight: float, semantic_weight: float, max_results: int = 10) -> list[str]:
        semantic_results = self.semantic.search(query, max_results)
        lexical_results = self.lexical.search(query, max_results)

        normalized_semantic_results = self.normalize_scores(semantic_results)
        normalized_lexical_results = self.normalize_scores(lexical_results)

        rankings = {}

        for score, result in normalized_semantic_results:
            result = result.lower()
            rankings[result] = rankings.get(result, 0) + semantic_weight * score

        for score, result in normalized_lexical_results:
            result = result.lower()
            rankings[result] = rankings.get(result, 0) + lexical_weight * score

        sorted_rankings = sorted(rankings, key=rankings.get, reverse=False)
        return sorted_rankings[:max_results]

    def normalize_scores(self, results: list[str]) -> list[tuple[float, str]]:
        x_max = len(results)
        x_min = 1

        normalized_results = []
        for rank, result in enumerate(results, start=1):
            normalized_rank = (rank - x_min) / (x_max - x_min)
            normalized_results.append((normalized_rank, result))

        return normalized_results

    def rrf_search(self, query: str, k: int = 60, max_results: int = 10) -> list[str]:
        semantic_results = self.semantic.search(query, max_results)
        lexical_results = self.lexical.search(query, max_results)

        rankings = {}
        for results in [semantic_results, lexical_results]:
            for rank, passage in enumerate(results, start=1):
                # This may make it so that multiple different passages
                # yield the same string, which is OK for the proposed evaluation.
                passage = passage.lower()

                score = 1 / (k + rank)
                rankings[passage] = rankings.get(passage, 0) + score

        rankings = sorted(
            rankings.items(),
            key=lambda item: item[1],
            reverse=True
        )

        ranked_passages = [passage for passage, score in rankings]
        return ranked_passages[:max_results]