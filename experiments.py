from dataset import MSMarco
from search import MixedSearch, SemanticSearch, LexicalSearch
from evaluation import calculate_reciprocal_rank
import chromadb
from itertools import islice

def main():
    dataset = MSMarco(10000)
    client = chromadb.PersistentClient("./chroma_data")

    multilingual_search = SemanticSearch(dataset, "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "multilingual", client)
    bge_search = SemanticSearch(dataset, "BAAI/bge-m3", "bge", client)

    lexical_search = LexicalSearch(dataset)

    multilingual_lexical = MixedSearch(lexical_search, multilingual_search)
    bge_lexical = MixedSearch(lexical_search, bge_search)

    for k in [25, 60, 80]:
        multilingual_rrf_mrr10 = rrf_experiments(dataset, multilingual_lexical, k)
        append_result(f"method=multilingual_rrf, k={k}, MRR@10={multilingual_rrf_mrr10}")

        bge_rrf_mrr10 = rrf_experiments(dataset, bge_lexical, k)
        append_result(f"method=bge_rrf, k={k}, MRR@10={bge_rrf_mrr10}")

    for lexical_weight, semantic_weight in [(0.75, 0.25), (0.50, 0.50), (0.25, 0.75)]:
        multilingual_weighted_mrr10 = weighted_experiments(dataset, multilingual_lexical, lexical_weight, semantic_weight)
        append_result(f"method=multilingual_weighted, lexical_weight={lexical_weight}, semantic_weight={semantic_weight}, MRR@10={multilingual_weighted_mrr10}")

        bge_weighted_mrr10 = weighted_experiments(dataset, bge_lexical, lexical_weight, semantic_weight)
        append_result(f"method=bge_weighted, lexical_weight={lexical_weight}, semantic_weight={semantic_weight}, MRR@10={bge_weighted_mrr10}")

    multilingual_mrr10 = semantic_experiments(dataset, multilingual_search)
    append_result(f"method=multilingual, MRR@10={multilingual_mrr10}")

    bge_mrr10 = semantic_experiments(dataset, bge_search)
    append_result(f"method=bge, MRR@10={bge_mrr10}")

    lexical_mrr10 = lexical_experiments(dataset, lexical_search)
    append_result(f"method=lexical, MRR@10={lexical_mrr10}")

def rrf_experiments(dataset: MSMarco, mixed: MixedSearch, k: int) -> float:
    reciprocal_ranks = []
    for query_id in islice(dataset.qrels, 1000):
        query = dataset.queries[query_id]

        results = mixed.rrf_search(query, k)
        normalized_results = [result.lower() for result in results]

        normalized_relevant_passages = [dataset.passages[relevant_passage_id].lower() for relevant_passage_id in dataset.qrels[query_id]]

        reciprocal_rank = calculate_reciprocal_rank(normalized_relevant_passages, normalized_results)
        reciprocal_ranks.append(reciprocal_rank)

    mrr10 = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return mrr10

def weighted_experiments(dataset: MSMarco, mixed: MixedSearch, lexical_weight: float, semantic_weight: float) -> float:
    reciprocal_ranks = []
    for query_id in islice(dataset.qrels, 1000):
        query = dataset.queries[query_id]

        results = mixed.weighted_search(query, lexical_weight, semantic_weight)
        normalized_results = [result.lower() for result in results]

        normalized_relevant_passages = [dataset.passages[relevant_passage_id].lower() for relevant_passage_id in dataset.qrels[query_id]]

        reciprocal_rank = calculate_reciprocal_rank(normalized_relevant_passages, normalized_results)
        reciprocal_ranks.append(reciprocal_rank)

    mrr10 = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return mrr10

def semantic_experiments(dataset: MSMarco, semantic: SemanticSearch) -> float:
    reciprocal_ranks = []
    for query_id in islice(dataset.qrels, 1000):
        query = dataset.queries[query_id]

        results = semantic.search(query)
        normalized_results = [result.lower() for result in results]

        normalized_relevant_passages = [dataset.passages[relevant_passage_id].lower() for relevant_passage_id in dataset.qrels[query_id]]

        reciprocal_rank = calculate_reciprocal_rank(normalized_relevant_passages, normalized_results)
        reciprocal_ranks.append(reciprocal_rank)

    mrr10 = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return mrr10

def lexical_experiments(dataset: MSMarco, lexical: LexicalSearch) -> float:
    reciprocal_ranks = []
    for query_id in islice(dataset.qrels, 1000):
        query = dataset.queries[query_id]

        results = lexical.search(query)
        normalized_results = [result.lower() for result in results]

        normalized_relevant_passages = [dataset.passages[relevant_passage_id].lower() for relevant_passage_id in dataset.qrels[query_id]]

        reciprocal_rank = calculate_reciprocal_rank(normalized_relevant_passages, normalized_results)
        reciprocal_ranks.append(reciprocal_rank)

    mrr10 = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return mrr10

def append_result(result: str):
    with open("results.txt", "a") as f:
        f.write(f"{result}\n")

if __name__ == "__main__":
    main()