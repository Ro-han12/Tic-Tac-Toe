import streamlit as st
import numpy as np
from model import TicTacToeGNN

# Load the trained model
model = TicTacToeGNN()
model.build((None, 3, 3, 1))  # Specify the input shape
model.load_weights('model_weights_epochs50.h5') #replace any model as per choice

def predict_move(board):
    # Add a channel dimension to the board
    board_input = np.expand_dims(np.expand_dims(np.array(board), axis=-1), axis=0)
    predictions = model.predict(board_input)
    return np.argmax(predictions)

def is_winner(board, player):
    # Check rows, columns and diagonals for a win
    for i in range(3):
        if np.all(board[i, :] == player) or np.all(board[:, i] == player):
            return True
    if board[0, 0] == player and board[1, 1] == player and board[2, 2] == player:
        return True
    if board[0, 2] == player and board[1, 1] == player and board[2, 0] == player:
        return True
    return False

def check_draw(board):
    return np.all(board != 0) and not (is_winner(board, 1) or is_winner(board, 2))

# Streamlit app
st.title('Tic-Tac-Toe with GNN')

# Initialize or reset board
if 'board' not in st.session_state:
    st.session_state.board = np.zeros((3, 3))
    st.session_state.player_turn = True
    st.session_state.winner = None
    st.session_state.player_moves = 0

def draw_board():
    cols = st.columns(3)
    for row in range(3):
        for col in range(3):
            with cols[col]:
                if st.session_state.board[row, col] == 0:
                    if st.button('', key=f'{row}-{col}', help='Click to place X', use_container_width=True):
                        if st.session_state.player_turn and st.session_state.player_moves < 2:
                            st.session_state.board[row, col] = 1  # Human move
                            st.session_state.player_moves += 1
                            if st.session_state.player_moves == 2:
                                st.session_state.player_turn = False
                elif st.session_state.board[row, col] == 1:
                    st.write('X', key=f'{row}-{col}', use_container_width=True)
                else:
                    st.write('O', key=f'{row}-{col}', use_container_width=True)
        st.write()  # Add a new row after each row of buttons

draw_board()

# AI Move Logic
if not st.session_state.player_turn and st.session_state.winner is None and st.session_state.player_moves == 2:
    board = st.session_state.board
    move = predict_move(board)
    row, col = divmod(move, 3)
    if board[row, col] == 0:
        board[row, col] = 2  # AI move
        st.session_state.board = board
        st.session_state.player_turn = True
        st.session_state.player_moves = 0  # Reset player move count

if is_winner(st.session_state.board, 1):
    st.session_state.winner = 'Player'
    st.write('Player wins!')
elif is_winner(st.session_state.board, 2):
    st.session_state.winner = 'AI'
    st.write('AI wins!')
elif check_draw(st.session_state.board):
    st.session_state.winner = 'Draw'
    st.write('It\'s a draw!')

# Reset button
if st.button('Reset'):
    st.session_state.board = np.zeros((3, 3))
    st.session_state.player_turn = True
    st.session_state.winner = None
    st.session_state.player_moves = 0
