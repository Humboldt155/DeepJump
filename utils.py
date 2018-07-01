import numpy as np
import torch
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import PackedSequence

def get_item_to_ix(data: list):
    '''
    Создает словарь элементов: в качестве ключа выступает сам элемент, в качестве значения - уникальный индекс.
    Словарь содержит элемент <PAD> с индексом 0

    :param data: list  # Лист: состоящий либо из самих элементов, либо являющийся листом из листов
    :return: dict
    '''

    item_to_ix = {'<PAD>': 0}

    assert type(data) == list, 'Входные данные должы иметь тип <class \'list\'>, а получен {}'.format(type(data))

    for elem in data:
        assert type(elem) == str or type(elem) == list, 'Элементы должны иметь тип str или list, а получен {}'.format(elem)

        if type(elem) == list:
            for e in elem:
                assert type(e) == str, 'Элементы последовательности должны иметь <class \'list\'>, а получен {}'.format(type(e))
                if e not in item_to_ix:
                    item_to_ix[e] = len(item_to_ix)
        else:
            if elem not in item_to_ix:
                item_to_ix[elem] = len(item_to_ix)

    return item_to_ix

def items_to_ix(data: list, item_to_ix: dict):
    assert len(data) > 0, 'входные данные пусты'
    new_data = []
    if type(data[0]) == list:
        for elem in data:
            new_data.append([item_to_ix[e] for e in elem])
    else:
        new_data = [item_to_ix[e] for e in data]
    return new_data


def mini_batch_sort_split_prepare(mini_batch: list):

    assert len(mini_batch) > 1, 'Последовательность должна содержать более 1 знака'
    mini_batch_lengths = {len(mini_batch[i]) * (-10000) + i / len(mini_batch): i for i in range(len(mini_batch))}

    sorted_order = [mini_batch_lengths[s] for s in sorted(mini_batch_lengths)]
    mini_batch_sorted = [mini_batch[i] for i in sorted_order]

    X = [seq[:-1] for seq in mini_batch_sorted]
    X_lengths = [len(sentence) for sentence in X]
    longest_sent_X = max(X_lengths)
    batch_size = len(X)
    padded_X = np.zeros((batch_size, longest_sent_X))

    # copy over the actual sequences
    for i, x_len in enumerate(X_lengths):
        sequence = X[i]
        padded_X[i, 0:x_len] = sequence[:x_len]

    Y = [seq[1:] for seq in mini_batch_sorted]
    Y_lengths = [len(sentence) for sentence in Y]
    longest_sent_Y = max(Y_lengths)
    batch_size = len(Y)
    padded_Y = np.zeros((batch_size, longest_sent_Y))

    # copy over the actual sequences
    for i, y_len in enumerate(Y_lengths):
        sequence = Y[i]
        padded_Y[i, 0:y_len] = sequence[:y_len]

    return padded_X, padded_Y, X_lengths
