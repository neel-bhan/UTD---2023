import copy
from queue import PriorityQueue

def makemove(empty, move, arr):
    arr[empty[0]][empty[1]] = arr[move[0]][move[1]]
    arr[move[0]][move[1]] = 0
    return arr
'''def func_distance(board):
    manhattan = 0
    goal_board = [[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 0]]
    goal_coordinates = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]

    for i in range(len(self.board)):
        for j in range(len(self.board)):
            if self.board[i][j] != goal_board[i][j]:
                manhattan += abs(i - goal_coordinates[self.board[i][j] - 1][0]) + abs(
                    j - goal_coordinates[self.board[i][j] - 1][1])

    return manhattan'''


def update_neighbors(arr):
    empty = []
    Queue = []
    for i in range(3):
        for j in range(3):
            if arr[i][j] == 0:
                empty.append(i)
                empty.append(j)
                if i > 0:
                    Queue.append([i - 1, j])
                if i < 2:
                    Queue.append([i + 1, j])
                if j > 0:
                    Queue.append([i, j - 1])
                if j < 2:
                    Queue.append([i, j + 1])

    return Queue


def solve(board):
    open_queue = board.update_neighbors(board)


class Node():
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g =0
        self.h =0
        self.f = 0

    def astar(board,start,end):
        start_node = Node(None, start)
        start_node.g = start_node.h = start_node.f = 0
        end_node = Node(None, end)
        end_node.g = end_node.h = end_node.f = 0

        open_list = []
        closed_list = []

        open_list.append(start_node)

        while len(open_list) > 0:
            current_node = open_list[0]
            current_idx = 0

            for index, item in enumerate(open_list):
                if item.f < current_node.f:
                    current_idx = index
                    current_item = item
            open_list.pop(current_idx)
            closed_list.append(current_node)

            if current_node == end_node:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                return path[::-1]

            steps = update_neighbors(current_node)

            for i in range(steps):
                children = []
                makemove()








