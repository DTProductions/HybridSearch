from helpers import download_file, extract_tar
import pickle
from pathlib import Path

class MSMarco:
    # The Passage ranking dataset is the one used in this code.
    # Download and further description available on https://github.com/microsoft/msmarco/blob/master/Datasets.md

    def __init__(self, max_queries=None):
        download_file("https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.train.tsv", "qrels.train.tsv")
        download_file("https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz", "collection.tar.gz")
        if not Path("collection.tsv").exists():
            print("Extracting passages...")
            extract_tar("collection.tar.gz", "./")

        download_file("https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz", "queries.tar.gz")
        if not Path("queries.train.tsv").exists():
            print("Extracting queries...")
            extract_tar("queries.tar.gz", "./")

        self.cache_file = Path("dataset_cache.pkl")
        if self.cache_file.exists():
            print("Using cache")
            qrels, queries, passages = self.load_cache()
        else:
            qrels, queries, passages = self.cache_dataset(max_queries)

        self.qrels = qrels
        self.queries = queries
        self.passages = passages

    def load_query_relevance_judgments(self, max_queries=None) -> dict[str, list[str]]:
        '''Load all available mappings between queries and their respective relevant documents.

        No relevance ordering should be assumed.

        Args:
            max_queries: maximum number of mappings returned. If None, all mappings will be returned.

        Returns:
            A dict mapping query ids to lists containing their matching documents.
        '''

        qrels = {}

        with open("qrels.train.tsv", encoding="utf-8") as f:
            for line in f:
                if max_queries != None and len(qrels) > max_queries:
                    break

                # relevance is either 0 (the document is irrelevant to the query) or 1 (document is relevant to the query)
                query_id, _, document_id, relevance = line.strip().split("\t")

                if int(relevance) > 0:
                    qrels.setdefault(query_id, []).append(document_id)

        return qrels

    def load_passages(self, ids: set[str]) -> dict[str, str]:
        '''Loads passages from the MSMARCO dataset.

        Args:
            ids: set of passage ids to be loaded from the dataset.

        Returns:
            a dict with **passage_id -> passage** mappings.
        '''

        selected_passages = {}
        with open("collection.tsv", encoding="utf-8") as f:
            for line in f:
                passage_id, passage = line.strip().split("\t")

                if passage_id in ids:
                    selected_passages[passage_id] = passage

        return selected_passages

    def load_queries(self, ids: set[str]) -> dict[str, str]:
        '''Loads queries from the MSMARCO dataset.

        Args:
            ids: set of query ids to be loaded from the dataset.

        Returns:
            a dict with **query_id -> query** mappings.
        '''
        selected_queries = {}
        with open("queries.train.tsv", encoding="utf-8") as f:
            for line in f:
                query_id, query = line.strip().split("\t")

                if query_id in ids:
                    selected_queries[query_id] = query

        return selected_queries

    def cache_dataset(self, max_queries=None):
        qrels = self.load_query_relevance_judgments(max_queries)

        query_ids = set(qrels.keys())

        passage_ids = {
            passage_id
            for passages in qrels.values()
            for passage_id in passages
        }

        queries = self.load_queries(query_ids)
        passages = self.load_passages(passage_ids)

        with self.cache_file.open("wb") as f:
            pickle.dump((qrels, queries, passages), f)
            return (qrels, queries, passages)

    def load_cache(self):
        with self.cache_file.open("rb") as f:
            qrels, queries, passages = pickle.load(f)
            return (qrels, queries, passages)
