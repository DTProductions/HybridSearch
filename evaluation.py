def calculate_reciprocal_rank(relevant_passages: list[str], results: list[str], limit: int = 10):
    reciprocal_rank = 0
    for rank, result in enumerate(results, start=1):
        if rank > limit:
            break

        if result in relevant_passages:
            reciprocal_rank = 1 / rank
            break

    return reciprocal_rank
