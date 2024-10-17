# Import required libraries
import pygame  # For creating the game window and handling graphics
import torch   # For neural network operations
import copy    # For creating deep copies of game states
import time    # For controlling AI move timing
import random  # For fallback random moves
from neural_network_trainer import ChessEvaluator  # Custom neural network for chess position evaluation

# Initialize pygame for graphics rendering
pygame.init()

# Define color constants using RGB values
WHITE = (255, 255, 255)
OFF_WHITE = (200, 200, 200)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Game window and board configuration
WINDOW_WIDTH = 800
BOARD_SIZE = 8  # Standard 8x8 chess board
SQUARE_SIZE = WINDOW_WIDTH // BOARD_SIZE  # Size of each square on the board
WINDOW_HEIGHT = 800 + SQUARE_SIZE  # Additional space for status display

# Chess piece definitions
PIECES = ["King", "Queen", "Rook", "Bishop", "Knight", "Pawn"]
SYMBOLS = ["K", "Q", "R", "B", "N", "P"]  # Standard chess notation symbols

# Initial chess board layout using algebraic notation
# Uppercase letters represent white pieces, lowercase represent black pieces
# None represents empty squares
BOARD_LAYOUT = [
    ["r", "n", "b", "q", "k", "b", "n", "r"],  # Black back rank
    ["p", "p", "p", "p", "p", "p", "p", "p"],  # Black pawns
    [None] * 8,  # Empty ranks
    [None] * 8,
    [None] * 8,
    [None] * 8,
    ["P", "P", "P", "P", "P", "P", "P", "P"],  # White pawns
    ["R", "N", "B", "Q", "K", "B", "N", "R"]   # White back rank
]

# Initialize pygame window and game settings
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("AI Chess")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 32)

def load_model(filepath='chess_model.pth'):
    """
    Load a pre-trained neural network model for chess position evaluation.
    
    Args:
        filepath (str): Path to the saved model file
        
    Returns:
        ChessEvaluator: Loaded neural network model in evaluation mode
    """
    model = ChessEvaluator()
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set model to evaluation mode
    return model

class Square:
    """
    Represents a single square on the chess board.
    
    Attributes:
        x (int): X-coordinate on the board (0-7)
        y (int): Y-coordinate on the board (0-7)
        color (tuple): RGB color value for the square
    """
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color

    def draw(self):
        """Draw the square on the game window"""
        pygame.draw.rect(window, self.color, 
                        (self.x * SQUARE_SIZE, self.y * SQUARE_SIZE, 
                         SQUARE_SIZE, SQUARE_SIZE))

class Piece:
    """
    Represents a chess piece.
    
    Attributes:
        x (int): X-coordinate on the board (0-7)
        y (int): Y-coordinate on the board (0-7)
        color (str): Color of the piece ("white" or "black")
        type (str): Type of piece (K, Q, R, B, N, P)
    """
    def __init__(self, x, y, color, type):
        self.x = x
        self.y = y
        self.color = color
        self.type = type

    def draw(self):
        """Draw the piece on the game window using text representation"""
        text_color = OFF_WHITE if self.color == "white" else BLACK
        window.blit(font.render(self.type, True, text_color), 
                   (self.x * SQUARE_SIZE, self.y * SQUARE_SIZE))

class Board:
    """
    Represents the chess board and manages game state.
    
    Attributes:
        squares (list): 2D list of Square objects
        pieces (list): 2D list of Piece objects
        turn (str): Current player's turn ("white" or "black")
        selected_square (Square): Currently selected square
        selected_piece (Piece): Currently selected piece
        state (str): Current game state ("running" or victory message)
    """
    def __init__(self):
        # Initialize board squares with alternating colors
        self.squares = [[Square(j, i, (WHITE if (i + j) % 2 == 0 else GRAY)) 
                        for j in range(BOARD_SIZE)] for i in range(BOARD_SIZE)]
        
        # Initialize pieces according to starting position
        self.pieces = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if BOARD_LAYOUT[i][j]:
                    self.pieces[i][j] = Piece(j, i,
                                            "white" if BOARD_LAYOUT[i][j].isupper() else "black",
                                            BOARD_LAYOUT[i][j].upper())

        self.turn = "white"
        self.selected_square = None
        self.selected_piece = None
        self.mouse_last_down = False
        self.state = "running"

    def draw(self):
        """Draw the complete board state including squares, pieces, and game status"""
        # Draw board squares
        for row in self.squares:
            for square in row:
                square.draw()

        # Draw pieces
        for row in self.pieces:
            for piece in row:
                if piece:
                    piece.draw()

        # Draw turn indicator at the bottom
        pygame.draw.rect(window, WHITE, 
                        (0, WINDOW_HEIGHT - SQUARE_SIZE, WINDOW_WIDTH, SQUARE_SIZE))
        window.blit(font.render(f"{self.turn.capitalize()}'s turn", True, BLACK),
                   (SQUARE_SIZE, WINDOW_HEIGHT - SQUARE_SIZE))

        # Draw game state if game is over
        if self.state != "running":
            window.blit(font.render(f"{self.state}", True, RED),
                       (WINDOW_WIDTH // 2, WINDOW_HEIGHT - SQUARE_SIZE))

        # Highlight selected square
        if self.selected_square:
            pygame.draw.rect(window, RED,
                           (self.selected_square.x * SQUARE_SIZE,
                            self.selected_square.y * SQUARE_SIZE,
                            SQUARE_SIZE, SQUARE_SIZE), 5)

    def update(self):
        """
        Handle user input and update game state.
        Manages piece selection and movement based on mouse input.
        """
        mouse_x, mouse_y = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()

        # Handle mouse click events
        if mouse_pressed[0]:
            self.mouse_last_down = True

        if not mouse_pressed[0] and self.mouse_last_down:
            self.mouse_last_down = False

            # Convert mouse position to board coordinates
            col = mouse_x // SQUARE_SIZE
            row = mouse_y // SQUARE_SIZE

            # Ensure click is within board boundaries
            if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                square = self.squares[row][col]
                piece = self.pieces[row][col]

                # Handle piece selection and movement
                if square == self.selected_square:
                    # Deselect if clicking same square
                    self.selected_square = None
                    self.selected_piece = None
                else:
                    if piece is not None and piece.color == self.turn:
                        # Select piece if it belongs to current player
                        self.selected_square = square
                        self.selected_piece = piece
                    elif self.selected_piece is not None:
                        # Attempt to move selected piece
                        if self.is_valid_move(self.selected_piece, square):
                            self.move_piece(self.selected_piece, square)
                            
                            # Check for game over
                            if self.is_game_over():
                                self.state = f"{self.turn.capitalize()} wins!"
                            else:
                                self.switch_turn()

                            self.selected_square = None
                            self.selected_piece = None

    def is_valid_move(self, piece, square):
        """
        Check if a proposed move is valid according to chess rules.
        
        Args:
            piece (Piece): The piece to move
            square (Square): The destination square
            
        Returns:
            bool: True if the move is valid, False otherwise
        """
        # Get the piece type and color
        piece_type = piece.type
        piece_color = piece.color

        # Get the piece and square coordinates
        piece_x = piece.x
        piece_y = piece.y
        square_x = square.x
        square_y = square.y

        # Get the relative coordinates
        dx = square_x - piece_x
        dy = square_y - piece_y

        target_piece = self.pieces[square_y][square_x]  # Get the piece on the target square

        #Make sure the piece isn't moving to the same square
        if dx == 0 and dy == 0:
            return False

        # Check the piece type
        if piece_type == "P": # Pawn
            # Check the piece color
            if piece_color == "black":
                # Check if the pawn moves one square forward
                if dx == 0 and dy == 1 and self.pieces[square_y][square_x] is None:
                    return True
                # Check if the pawn moves two squares forward from the initial position
                if dx == 0 and dy == 2 and piece_y == 1 and self.pieces[square_y][square_x] is None and self.pieces[square_y-1][square_x] is None:
                    return True
                # Check if the pawn captures a black piece diagonally
                if abs(dx) == 1 and dy == 1 and self.pieces[square_y][square_x] is not None and self.pieces[square_y][square_x].color == "white":
                    return True
            else: # White
                # Check if the pawn moves one square forward
                if dx == 0 and dy == -1 and self.pieces[square_y][square_x] is None:
                    return True
                # Check if the pawn moves two squares forward from the initial position
                if dx == 0 and dy == -2 and piece_y == 6 and self.pieces[square_y][square_x] is None and self.pieces[square_y+1][square_x] is None:
                    return True
                # Check if the pawn captures a white piece diagonally
                if abs(dx) == 1 and dy == -1 and self.pieces[square_y][square_x] is not None and self.pieces[square_y][square_x].color == "black":
                    return True
        elif piece_type == "R": # Rook
            # Check if the rook moves horizontally or vertically
            if dx == 0 or dy == 0:
                # Check if there is no piece in the way
                if self.is_clear_path(piece, square):
                    if target_piece is None or target_piece.color != piece.color:
                        return True
        elif piece_type == "B": # Bishop
            # Check if the bishop moves diagonally
            if abs(dx) == abs(dy):
                # Check if there is no piece in the way
                if self.is_clear_path(piece, square):
                    if target_piece is None or target_piece.color != piece.color:
                        return True
        elif piece_type == "N": # Knight
            # Check if the knight moves in an L-shape
            if ((abs(dx) == 1 and abs(dy) == 2) or (abs(dx) == 2 and abs(dy) == 1)) and (target_piece is None or target_piece.color != piece.color):
                return True
        elif piece_type == "Q": # Queen
            # Check if the queen moves horizontally, vertically, or diagonally
            if dx == 0 or dy == 0 or abs(dx) == abs(dy):
                # Check if there is no piece in the way
                if self.is_clear_path(piece, square):
                    if target_piece is None or target_piece.color != piece.color:
                        return True
        elif piece_type == "K":  # King
            # Check if the king moves one square in any direction
            if abs(dx) <= 1 and abs(dy) <= 1:
                # Ensure the destination is either empty or contains an opponent's piece
                if target_piece is None or target_piece.color != piece.color:
                    return True
                

        # If none of the conditions are met, the move is invalid
        return False
    def is_clear_path(self, piece, square):
        """
        Check if there are any pieces blocking the path between the start and end position.
        Used for pieces that move in straight lines (Rook, Bishop, Queen).
        
        Args:
            piece (Piece): The piece being moved
            square (Square): The destination square
            
        Returns:
            bool: True if path is clear, False if blocked
        """
        piece_x = piece.x
        piece_y = piece.y
        square_x = square.x
        square_y = square.y

        # Calculate the direction of movement
        dx = square_x - piece_x
        dy = square_y - piece_y
      
        # Determine the direction vector (-1, 0, or 1 for each component)
        x_dir = (1 if dx > 0 else (-1 if dx < 0 else 0))
        y_dir = (1 if dy > 0 else (-1 if dy < 0 else 0))

        # Check each square along the path (excluding start and end points)
        for i in range(1, max(abs(dx), abs(dy))):
            x = piece_x + i * x_dir
            y = piece_y + i * y_dir

            # If any square contains a piece, the path is blocked
            if self.pieces[y][x] is not None:
                return False
            
        return True

    def move_piece(self, piece, square):
        """
        Move a piece to a new square on the board.
        Updates the board state to reflect the move.
        
        Args:
            piece (Piece): The piece to move
            square (Square): The destination square
        """
        print(f"Moving piece {piece.type} to square {square.x}, {square.y}")
            
        # Clear the piece's current position
        self.pieces[piece.y][piece.x] = None
        # Place the piece in its new position
        self.pieces[square.y][square.x] = piece
        
        # Update the piece's coordinates
        piece.x = square.x
        piece.y = square.y

    def switch_turn(self):
        """Switch the active player between white and black"""
        self.turn = "black" if self.turn == "white" else "white"

    def is_game_over(self):
        """
        Check if the game has ended.
        Currently only checks for king capture (simplified chess rules).
        
        Returns:
            bool: True if game is over, False otherwise
        """
        # Search for the opponent's king
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                piece = self.pieces[i][j]
                # If we find the opponent's king, the game isn't over
                if piece is not None and piece.type == "K" and piece.color != self.turn:
                    return False
        # If we didn't find the opponent's king, the game is over
        return True

class AIChessGame:
    """
    Main game class that integrates the chess board with AI opponent.
    Manages game state and AI move generation.
    """
    def __init__(self):
        self.board = Board()
        self.model = load_model()  # Load the neural network model
        self.thinking = False  # Flag for when AI is calculating
        self.move_time_limit = 2.0  # Maximum time (seconds) for AI to think
        self.search_depth = 2  # How many moves ahead to look

    def print_all_legal_moves(self, color):
        """
        Debug function to print all legal moves for a given color.
        
        Args:
            color (str): The color to check ("white" or "black")
        """
        all_moves = self.get_all_moves(color)
        print(f"\nLegal moves for {color}:")
        for piece, square in all_moves:
            print(f"{piece.type} at ({piece.x}, {piece.y}) can move to ({square.x}, {square.y})")
        
    def get_board_tensor(self):
        """
        Convert the current board state to a tensor representation for neural network input.
        Pieces are represented by signed integers:
        - Positive for black pieces, negative for white
        - Magnitude indicates piece type (1=pawn, 2=knight, etc.)
        
        Returns:
            torch.Tensor: 8x8 tensor representing the board state
        """
        board_tensor = torch.zeros((8, 8))
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                piece = self.board.pieces[i][j]
                if piece is not None:
                    # Determine piece value based on color and type
                    value = 1 if piece.color == "black" else -1
                    # Multiply by piece type weight
                    if piece.type == "P": value *= 1
                    elif piece.type == "N": value *= 2
                    elif piece.type == "B": value *= 3
                    elif piece.type == "R": value *= 4
                    elif piece.type == "Q": value *= 5
                    elif piece.type == "K": value *= 6
                    board_tensor[i][j] = value
        return board_tensor

    def evaluate_position(self):
        """
        Use the neural network to evaluate the current board position.
        
        Returns:
            float: Position evaluation score (positive favors black, negative favors white)
        """
        board_tensor = self.get_board_tensor()
        # Add batch and channel dimensions required by the model
        board_input = board_tensor.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():  # No need to compute gradients for inference
            evaluation = self.model(board_input).item()
        return evaluation

    def get_all_moves(self, color):
        """
        Generate all legal moves for a given color.
        
        Args:
            color (str): The color to generate moves for ("white" or "black")
            
        Returns:
            list: List of tuples (piece, square) representing legal moves
        """
        moves = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                piece = self.board.pieces[i][j]
                if piece is not None and piece.color == color:
                    # Check all possible destination squares
                    for x in range(BOARD_SIZE):
                        for y in range(BOARD_SIZE):
                            if self.board.is_valid_move(piece, self.board.squares[y][x]):
                                moves.append((piece, self.board.squares[y][x]))
        return moves

    def get_ai_move(self):
        """
        Calculate the best move for the AI player using the neural network evaluation.
        
        Returns:
            tuple: (start_x, start_y, end_square) representing the chosen move,
                  or None if no legal moves are available
        """
        best_move = None
        best_eval = float('-inf')  # AI plays black, so maximize evaluation
        
        # Debug print of all legal moves
        self.print_all_legal_moves("black")
        
        # Consider each possible move
        for piece, square in self.get_all_moves("black"):
            print(f"AI is considering moving piece {piece.type} at {piece.x}, {piece.y} to square {square.x}, {square.y}")
            
            # Debug prints for board state verification
            print("Board tensor before backup:")
            print(self.get_board_tensor())

            # Save current board state
            board_backup = copy.deepcopy(self.board)

            print("Board tensor after backup:")
            print(self.get_board_tensor())

            # Remember original position
            original_x = piece.x
            original_y = piece.y
            
            # Try the move
            self.board.move_piece(piece, square)
            current_eval = self.evaluate_position()

            print("Board tensor after move:")
            print(self.get_board_tensor())

            # Restore the board state
            self.board = board_backup

            print("Board tensor after restore:")
            print(self.get_board_tensor())

            # Update best move if this is the highest scoring position
            if current_eval > best_eval:
                best_eval = current_eval
                best_move = (original_x, original_y, square)
                print(f"!!!!AI found a better move: {piece.type} at ({original_x}, {original_y}) to ({square.x}, {square.y})")

        return best_move

    def update(self):
        """
        Update game state and handle AI moves when it's the AI's turn.
        Manages the AI thinking state and move execution.
        """
        # Check if it's AI's turn and AI isn't already thinking
        if self.board.turn == "black" and not self.thinking:
            self.thinking = True
            
            # Calculate AI's move
            best_move = self.get_ai_move()
            
            # If a move was found, execute it
            if best_move:
                piecex, piecey, square = best_move
                piece = self.board.pieces[piecey][piecex]

                if piece is None:
                    # Fallback to random move if something went wrong
                    all_moves = self.get_all_moves("black")
                    if all_moves:
                        piece, square = all_moves[random.randint(0, len(all_moves) - 1)]
                        piecex, piecey = piece.x, piece.y
                    else:
                        print("No valid moves found for AI.")
                        self.thinking = False
                        return

                print(f"The piece coords and square are {piece.x}, {piece.y}, {square.x}, {square.y}")
                
                # Execute the move if it's valid
                if self.board.is_valid_move(piece, square):
                    print(f"AI moves {piece.type} from ({piecex}, {piecey}) to ({square.x}, {square.y})")
                    self.board.move_piece(piece, square)
                    self.board.switch_turn()
                else:
                    print(f"Error: AI tried to make an invalid move, {piece.type} at ({piece.x}, {piece.y}) to ({square.x}, {square.y})")
            else:
                print("AI found no move to make.")
            
            self.thinking = False

        # Update the board state
        self.board.update()

    def draw(self):
        """
        Draw the game state, including the board and AI thinking indicator.
        """
        # Draw the chess board
        self.board.draw()
        
        # If AI is thinking, show an overlay with "thinking" message
        if self.thinking:
            # Create semi-transparent overlay
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            overlay.fill((128, 128, 128))
            overlay.set_alpha(128)
            window.blit(overlay, (0, 0))
            
            # Draw "thinking" text
            text = font.render("AI is thinking...", True, BLACK)
            text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
            window.blit(text, text_rect)

def main():
    """
    Main game loop. Initializes the game and runs the primary game cycle.
    Handles game events and maintains the frame rate.
    """
    game = AIChessGame()
    running = True
    
    while running:
        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update game state and redraw
        game.update()
        game.draw()
        pygame.display.flip()
        clock.tick(60)  # Limit frame rate to 60 FPS

    pygame.quit()

if __name__ == "__main__":
    main()
