# We'll produce the modular arithmetic datasets here.
import random 
import torch

def deterministic_shuffle(lst, seed):
    random.seed(seed)
    random.shuffle(lst)
    return lst


def get_all_pairs(p):
    pairs = []
    for i in range(p):
        for j in range(p):
            pairs.append((i, j))
    return set(pairs)


def make_dataset(p):
    data = []
    pairs = get_all_pairs(p)
    for a, b in pairs:
        data.append((torch.tensor([a, b]), torch.tensor((a + b) % p)))
    return data


def train_test_split(dataset, train_split_proportion, seed):
    """
    Splits the dataset into a training set and a test set.
    Args:
        dataset (list): The dataset to split.
        train_split_proportion (float): The proportion of the dataset to use for training.
        seed (int): The seed for the random number generator to ensure reproducibility.
    Returns:
        tuple: A tuple containing the training set and the test set.
    """
    l = len(dataset)
    train_len = int(train_split_proportion * l)
    idx = list(range(l))
    idx = deterministic_shuffle(idx, seed)
    train_idx = idx[:train_len]
    test_idx = idx[train_len:]
    return [dataset[i] for i in train_idx], [dataset[i] for i in test_idx]