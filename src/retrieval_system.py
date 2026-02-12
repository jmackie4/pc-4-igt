import heapq



def get_word_overlap_score(query, example):
    query_set = set(query.lower().split())
    example_set = set(example.lower().split())
    return len(query_set & example_set)


def get_idx_of_nlargest_values(list_of_values, n):
    idx_value_pairs = {idx: value for idx, value in enumerate(list_of_values)}
    return heapq.nlargest(n, idx_value_pairs, key=idx_value_pairs.get)


def get_examples_from_dataset(dataset, idx_list):
    return dataset[idx_list]


class InformationRetrievalSystem:
    def __init__(self, dataset, num_examples=3):
        self.dataset = dataset
        self.num_examples = num_examples

    def get_examples_from_query(self, query):
        overlap_scores = [get_word_overlap_score(query, sentence) for sentence in self.dataset['source']]
        largest_idxs = get_idx_of_nlargest_values(overlap_scores, self.num_examples)
        return get_examples_from_dataset(self.dataset, largest_idxs)
