
import copy


def calculate_manhattan_distance(board):
    manhattan = 0
    goal_board = [[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 0]]
    goal_coordinates = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]

    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] != goal_board[i][j] and board[i][j] != 0:
                manhattan += abs(i - goal_coordinates[board[i][j] - 1][0]) + abs(
                    j - goal_coordinates[board[i][j] - 1][1])

    return manhattan
def available_moves(board):
    moves = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                if i > 0:  # Can move up
                    moves.append((i - 1, j))
                if i < 2:  # Can move down
                    moves.append((i + 1, j))
                if j > 0:  # Can move left
                    moves.append((i, j - 1))
                if j < 2:  # Can move right
                    moves.append((i, j + 1))
    return moves

def make_move(board, move):
    new_board = copy.deepcopy(board)
    empty = getempty(new_board)
    new_board[empty[0]][empty[1]] = new_board[move[0]][move[1]]
    new_board[move[0]][move[1]] = 0
    return new_board


def available(arr, Queue):
    empty = []

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


def getempty(arr):
    empty = []

    for i in range(3):
        for j in range(3):
            if arr[i][j] == 0:
                empty.append(i)
                empty.append(j)

    return empty


def makemove(empty, move, arr):
    arr[empty[0]][empty[1]] = arr[move[0]][move[1]]
    arr[move[0]][move[1]] = 0
    return arr


answer = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
solve =  [[8, 6, 3], [2, 5, 7], [0, 4, 1]]


count = 1
seen = []


count =1
import heapq

def calculate_cost(board, moves_made):
    return calculate_manhattan_distance(board) + moves_made

def a_star_search(start, goal):
    heap = []
    heapq.heappush(heap, (calculate_cost(start, 0), 0, start, []))
    seen = {str(start): 0}

    while heap:
        (cost, moves_made, current, path) = heapq.heappop(heap)

        if current == goal:
            return path + [current]

        for move in available_moves(current):
            new_board = make_move(current, move)
            new_cost = calculate_cost(new_board, moves_made + 1)

            if str(new_board) not in seen or new_cost < seen[str(new_board)]:
                seen[str(new_board)] = new_cost
                heapq.heappush(heap, (new_cost, moves_made + 1, new_board, path + [current]))

    return None  # No solution found
count = 1
# In your main loop:
path = a_star_search(solve, answer)
if path is None:
    print("No solution found")
else:
    for board in path:
        for row in board:
            print(row)
        print(f'iteration: {count}, Loss = {calculate_manhattan_distance(board)}')
        count += 1

print('\n\ngreedy\n\n')
count = 0
#old greddy first solution
'''
while answer != solve:
    print(f'Iteration: {count}')
    Queue = []
    distances = []
    empty = []
    Queue = available(solve, Queue)

    for i in range(len(Queue)):
        temp = copy.deepcopy(solve)
        empty = getempty(temp)
        temp = makemove(empty, Queue[i], temp)
        distances.append(calculate_manhattan_distance(temp))

    small = distances.index(min(distances))
    temp = copy.deepcopy(solve)
    temp = makemove(getempty(temp), Queue[small], temp)

    while temp in seen:
        small += 1
        if small > len(Queue) - 1:
            break

        temp = copy.deepcopy(solve)
        temp = makemove(empty, Queue[small], temp)

    if small >= len(Queue):
        small = 0  # or some other appropriate value

    empty = getempty(solve)
    solve = makemove(empty, Queue[small], solve)


    seen.append(copy.deepcopy(solve))
    for row in solve:
        print(row)
    print(f'loss: {calculate_manhattan_distance(solve)}')
    print()
    count += 1'''