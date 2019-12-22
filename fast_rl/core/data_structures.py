"""


Notices:
  [1] SumTree implementation belongs to: https://github.com/rlcode/per
  As of 8/23/2019, does not have a license provided. As another note, this code is modified.


"""

import numpy as np


class SumTree(object):
    write = 0

    def __init__(self, capacity):
        """
        Used for PER.

        References:
              [1] SumTree implementation belongs to: https://github.com/rlcode/per

        Notes:
            As of 8/23/2019, does not have a license provided. As another note, this code is modified.


        Args:
            capacity:
        """

        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def _propagate(self, idx, change):
        """ Update to the root node """
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if (np.isscalar(parent) and parent != 0) or (not np.isscalar(parent) and all(parent != 0)):
            if not np.isscalar(parent): change[parent == 0] = 0
            self._propagate(parent, change)

    def get_left(self, index):
        return 2 * index + 1

    def get_right(self, index):
        return self.get_left(index) + 1

    def _retrieve(self, idx, s):
        """ Finds sample on leaf node """
        left = self.get_left(idx)
        right = self.get_right(idx)

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        """ Store priority and sample """
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        """ Update priority """
        p = p.flatten() if not np.isscalar(p) else p
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        """ Get priority and sample """
        idx = self._retrieve(0, s)
        data_index = idx - self.capacity + 1

        return idx, self.tree[idx], self.data[data_index]

    def anneal_weights(self, priorities, beta):
        sampling_probabilities = priorities / self.total()
        is_weight = np.power(self.n_entries * sampling_probabilities, -beta)
        is_weight /= is_weight.max()
        return is_weight.astype(float)

    def batch_get(self, ss):
        return np.array(list(zip(*list([self.get(s) for s in ss if self.get(s)[2] != 0]))))


def print_tree(tree: SumTree):
    print('\n')
    if tree.n_entries == 0:
        print('empty')
        return

    max_d = int(np.log2(len(tree.tree)))
    string_len_max = len(str(tree.tree[-1]))

    tree_strings = []
    display_values = None
    display_indexes = None
    for layer in range(max_d + 1):
        # Get the indexes in the current layer d
        if display_indexes is None:
            display_indexes = [[0]]
        else:
            local_list = []
            for i in [_ for _ in display_indexes[-1] if _ < len(tree.tree)]:
                if tree.get_left(i) < len(tree.tree): local_list.append(tree.get_left(i))
                if tree.get_right(i) < len(tree.tree): local_list.append(tree.get_right(i))
            display_indexes.append(local_list)

    for layer in display_indexes:
        # Get the v contained in current layer d
        if display_values is None:
            display_values = [[tree.tree[i] for i in layer]]
        else:
            display_values.append([tree.tree[i] for i in layer])

    tab_sizes = []
    spacings = []
    for i, layer in enumerate(display_values):
        # for now ignore string length
        tab_sizes.append(0 if i == 0 else (tab_sizes[-1] + 1) * 2)
        spacings.append(3 if i == 0 else (spacings[-1] * 2 + 1))

    for i, layer in enumerate(display_values):
        # tree_strings.append('*' * list(reversed(tab_sizes))[i])
        values = ''.join(str(v) + ' ' * (string_len_max * list(reversed(spacings))[i]) for v in layer)
        tree_strings.append(' ' * (string_len_max * list(reversed(tab_sizes))[i]) + values)

    for tree_string in tree_strings:
        print(tree_string)
