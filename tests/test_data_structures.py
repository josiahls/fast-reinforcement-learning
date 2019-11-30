from fast_rl.core.data_structures import print_tree, SumTree


def test_sum_tree_with_max_size():
    memory = SumTree(10)

    values = [1, 1, 1, 1, 1, 1]
    data = [f'data with priority: {i}' for i in values]

    for element, value in zip(data, values):
        memory.add(value, element)

    print_tree(memory)


