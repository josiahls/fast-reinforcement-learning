from fast_rl.core.agent_core import PriorityExperienceReplay
import numpy as np

from fast_rl.core.data_structures import SumTree, Node, print_tree


def test_sum_tree():
    memory = SumTree(30)

    values = [1, 2, 5, 2, 2, 4]
    data = [f'data with priority: {i}' for i in values]

    print('\n')
    for element, value in zip(data, values):
        print(f'Adding {value}\n')
        memory.insert(element, value)
        print_tree(memory.root)
        print('\n')
