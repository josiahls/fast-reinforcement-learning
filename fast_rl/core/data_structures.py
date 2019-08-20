from collections import deque

from typing import Dict


class Node(object):
    def __init__(self, data, value, parent):
        self.data = data
        self.value = value
        self.d = 0

        self.ref = {'parent': parent, 'right': None, 'left': None}  # type: Dict[str: Node, str: Node, str: Node]

    def __str__(self):
        elements = [str(self.value), str(self.ref['right']) if self.ref['right'] is not None else None,
                    str(self.ref['left']) if self.ref['left'] is not None else None]
        return ','.join([el for el in elements if el is not None])

    def add_element(self, node):
        """
        Adds a Node to the tree.

        Args:
            node:

        Returns:
        """
        raise NotImplementedError('Needs to be subclassed')

    def __eq__(self, other):
        return self.value == other.value

    def __le__(self, other):
        return self.value <= other.value

    def __lt__(self, other):
        return self.value < other.value


class SumTreeNode(Node):
    def add_element(self, node):
        """
        When adding an element we have 3 steps:
        - add element to either right or left node
        - update current value (sum) if data is not None

        Notes:
            Leaf nodes should never have their values changed if their data fields are not None.
            If the data fields are not None, they should be leaf nodes.

        Args:
            node:

        Returns:

        """
        # Add
        node.d += 1
        if self.data is None: self.value += node.value
        # If the current node is (should be) a leaf then create a summation node.
        if self.data is not None:
            # Create summation node and change over parents.
            parent = PriorityItem(None, self.value + node.value)
            if self.ref['parent'] is not None:
                parent.ref['parent'] = self.ref['parent']
                # Rewire the parent's child pointers
                if parent.ref['parent'].ref['left'] is self: parent.ref['parent'].ref['left'] = parent
                if parent.ref['parent'].ref['right'] is self: parent.ref['parent'].ref['right'] = parent
            parent.ref['left'] = self if self < node else node
            parent.ref['right'] = self if self >= node else node
        else:
            # If the current node is a summation node, then propagate down the tree like usual.
            if self.ref['left'] is None:
                node.ref['parent'] = self
                self.ref['left'] = node
            elif self.ref['left'] <= node and self.ref['right'] is None:
                node.ref['parent'] = self
                self.ref['right'] = node
            elif self.ref['left'] > node:
                self.ref['left'].add_element(node)
            else:
                self.ref['right'].add_element(node)


class PriorityItem(SumTreeNode):

    def __init__(self, data, value, parent=None):
        super().__init__(data, value, parent)
        self.priority = value


class SumTree(object):
    def __init__(self, max_size):
        """
        Should be a regular sum tree however, the parents are the sums of their children where the leaf nodes are the
        actual actionable items. What we can do is keep a list of of a actual items and have the tree contain the
        indices.

        if we have the root list as a queue, then we can remove elements that do not appear as often.

        """
        self.max_size = max_size
        self.root = None

    def insert(self, item, value):
        """
        On insert:
        if root is None: make root
        if larger than root, swap

        Args:
            item:
            value:

        Returns:

        """
        node = PriorityItem(item, value)
        if self.root is None:
            self.root = node
        elif node > self.root:
            # If the node is greater than the root,
            if node.data is None:
                # -- If node.data is None, replace root
                node.ref = self.root.ref
                node.add_element(self.root)
                self.root = node
            else:
                # -- If node.data is not None and root.data is None, create SumNode, make both children
                root = PriorityItem(None, 0)
                root.add_element(self.root)
                root.add_element(node)
                self.root = root
        else:
            self.root.add_element(node)

    def __str__(self):
        if self.root is not None:
            return '[' + str(self.root) + ']'
        else:
            return '[ ] Tree is empty.'


def print_tree(root: Node):
    if root is None:
        print('')
        return
    temp = root
    d = temp.d
    while temp.ref['left'] is not None:
        temp = temp.ref['left']
        d = temp.d

    string_len_max = len(str(root.value))
    temp = root
    current_layer_nodes = []

    while True:
        tabbing = ' ' * (string_len_max * d + 1)
        space = ' ' * string_len_max

        if not current_layer_nodes:
            print(tabbing, '', temp.value)
            if temp.ref['left'] is not None: current_layer_nodes.append(temp.ref['left'])
            if temp.ref['right'] is not None: current_layer_nodes.append(temp.ref['right'])
        else:
            d -= 1
            printable = [str(_.value) for _ in current_layer_nodes]
            print(tabbing, space.join(printable))
            temp_list = []
            for node in current_layer_nodes:
                if node.ref['left'] is not None: temp_list.append(node.ref['left'])
                if node.ref['right'] is not None: temp_list.append(node.ref['right'])
            current_layer_nodes = temp_list
        if not current_layer_nodes: break



