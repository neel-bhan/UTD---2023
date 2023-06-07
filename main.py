
import pandas as pd
import random
from tkinter import Tk, Button, Frame
from tkinter import messagebox

# Function to print the board on the GUI
def print_board(board):
    for i in range(3):
        for j in range(3):
            button_text = board[i][j]
            buttons[i][j].config(text=button_text)

# Function to check for a win
def check_win(board):
    # Check rows
    for row in board:
        if row[0] == row[1] == row[2] != " ":
            return True

    # Check columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != " ":
            return True

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] != " ":
        return True
    if board[0][2] == board[1][1] == board[2][0] != " ":
        return True

    return False

# Function for the computer's move
def make_computer_move():
    available_moves = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == " ":
                available_moves.append((i, j))
    if available_moves:
        row, col = random.choice(available_moves)
        board[row][col] = "O"
        print_board(board)
        if check_win(board):
            messagebox.showinfo("Game Over", "Computer wins!")
            reset_game()
        elif all(board[i][j] != " " for i in range(3) for j in range(3)):
            messagebox.showinfo("Game Over", "It's a tie!")
            reset_game()

# Function to handle button click events
def button_click(row, col):
    if board[row][col] == " ":
        board[row][col] = "X"
        print_board(board)
        if check_win(board):
            messagebox.showinfo("Game Over", "Player X wins!")
            reset_game()
        elif all(board[i][j] != " " for i in range(3) for j in range(3)):
            messagebox.showinfo("Game Over", "It's a tie!")
            reset_game()
        else:
            make_computer_move()
    else:
        messagebox.showwarning("Invalid Move", "Invalid move! Try again.")

# Function to reset the game
def reset_game():
    for i in range(3):
        for j in range(3):
            board[i][j] = " "
            buttons[i][j].config(text=" ")
    print_board(board)

# Function to save the game state
def save_game(buttons):
    board_state = [[button["text"] for button in row] for row in buttons]
    df = pd.DataFrame(board_state)
    df.to_csv("saved_game.csv", index=False)
    messagebox.showinfo("Game Saved", "Game saved successfully!")
    root.quit()

# Function to load the saved game state
def load_game(buttons):
    df = pd.read_csv("saved_game.csv")
    board_state = df.values.tolist()
    for i in range(3):
        for j in range(3):
            button_text = board_state[i][j]
            buttons[i][j].config(text=button_text)
            board[i][j] = button_text
    print_board(board)
    messagebox.showinfo("Game Loaded", "Game loaded successfully!")
# Create the main window
root = Tk()
root.title("Tic-Tac-Toe")

# Create the buttons for the game board
lower_frame = Frame(root)
lower_frame.grid(row=4, column=0, columnspan=3, padx=4, pady=4)

close_button = Button(lower_frame, text="Close", font=("Arial", 18), width=15, height=5, command=root.quit)
close_button.grid(row=0, column=0, padx=2, pady=4)

save_button = Button(lower_frame, text="Save", font=("Arial", 18), width=15, height=5,
                     command=lambda: save_game(buttons))
save_button.grid(row=0, column=1, padx=2, pady=4)

topframe = Frame(root)
topframe.grid(row=0, column=0, columnspan=5, padx=4, pady=4)
button_load = Button(topframe, text="Load Game", font=("Arial", 18), width=30, height=2, command=lambda: load_game(buttons))
button_load.grid(row=0, column=0, padx=5, pady=4)

buttons = []
for i in range(3):
    row_buttons = []
    for j in range(3):
        button = Button(root, text=" ", font=("Arial", 18), width=9, height=3,
                        command=lambda row=i, col=j: button_click(row, col))
        button.grid(row=i+1, column=j, padx=4, pady=4)
        row_buttons.append(button)
    buttons.append(row_buttons)

# Create the game board
board = [[" ", " ", " "], [" ", " ", " "], [" ", " ", " "]]

# Start the game
print_board(board)

# Run the Tkinter event loop
root.mainloop()
