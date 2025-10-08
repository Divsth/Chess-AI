import pygame
import torch
import copy
import time
import random
from neural_network_trainer import ChessEvaluator

pygame.init()

# --- UI and Display Constants ---
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 200, 0)
BLUE = (100, 150, 255)
YELLOW = (255, 215, 0)
WINDOW_WIDTH = 700
BOARD_SIZE = 8
SQUARE_SIZE = WINDOW_WIDTH // BOARD_SIZE
STATUS_HEIGHT = 100
WINDOW_HEIGHT = WINDOW_WIDTH + STATUS_HEIGHT

# --- Game Logic Constants ---
MAX_THINK_TIME = 2.0 # Seconds for Time Limit mode
BOARD_LAYOUT = [
    ["r", "n", "b", "q", "k", "b", "n", "r"],
    ["p", "p", "p", "p", "p", "p", "p", "p"],
    [None] * 8, [None] * 8, [None] * 8, [None] * 8,
    ["P", "P", "P", "P", "P", "P", "P", "P"],
    ["R", "N", "B", "Q", "K", "B", "N", "R"]
]

# --- Pygame Setup ---
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("AI Chess")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 20, bold=True)
small_font = pygame.font.SysFont("Arial", 16)

# --- Asset Loading ---
piece_images = {}
def load_piece_images():
    """
    Loads piece images from the 'pieces-basic-png' folder.
    Chess piece icons downloaded from https://greenchess.net/info.php?item=downloads.
    """
    pieces = ['king', 'queen', 'rook', 'bishop', 'knight', 'pawn']
    colors = ['white', 'black']
    for color in colors:
        for piece in pieces:
            try:
                filename = f"pieces-basic-png/{color}-{piece}.png"
                image = pygame.image.load(filename)
                image = pygame.transform.smoothscale(image, (int(SQUARE_SIZE * 0.8), int(SQUARE_SIZE * 0.8)))
                piece_images[f"{color}-{piece}"] = image
            except pygame.error:
                piece_images[f"{color}-{piece}"] = None
    print(f"Loaded {len([v for v in piece_images.values() if v is not None])} piece images.")

load_piece_images()

def load_model(filepath='chess_model.pth'):
    """Loads the trained PyTorch model from the specified file."""
    model = ChessEvaluator()
    try:
        state_dict = torch.load(filepath, map_location=torch.device('cpu'))
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
        print(f"Model loaded successfully from {filepath}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {filepath}. The AI will use material evaluation only.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None
    model.eval()
    return model

class Square:
    def __init__(self, x, y, color):
        self.x, self.y, self.color = x, y, color
    def draw(self):
        pygame.draw.rect(window, self.color, (self.x * SQUARE_SIZE, self.y * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

class Piece:
    def __init__(self, x, y, color, type):
        self.x, self.y, self.color, self.type = x, y, color, type
        
    def draw(self):
        piece_name_map = {'K': 'king', 'Q': 'queen', 'R': 'rook', 'B': 'bishop', 'N': 'knight', 'P': 'pawn'}
        piece_name = piece_name_map.get(self.type, 'pawn')
        image = piece_images.get(f"{self.color}-{piece_name}")
        
        if image:
            rect = image.get_rect(center=(self.x * SQUARE_SIZE + SQUARE_SIZE // 2, self.y * SQUARE_SIZE + SQUARE_SIZE // 2))
            window.blit(image, rect)
        else: # Fallback to text rendering if images are missing
            text_color = (200, 200, 200) if self.color == "white" else BLACK
            text = pygame.font.SysFont("Arial", 32).render(self.type, True, text_color)
            rect = text.get_rect(center=(self.x * SQUARE_SIZE + SQUARE_SIZE // 2, self.y * SQUARE_SIZE + SQUARE_SIZE // 2))
            window.blit(text, rect)

class Board:
    def __init__(self):
        self.squares = [[Square(j, i, (WHITE if (i + j) % 2 == 0 else GRAY)) for j in range(BOARD_SIZE)] for i in range(BOARD_SIZE)]
        self.pieces = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        for i, row in enumerate(BOARD_LAYOUT):
            for j, piece_char in enumerate(row):
                if piece_char:
                    color = "white" if piece_char.isupper() else "black"
                    self.pieces[i][j] = Piece(j, i, color, piece_char.upper())
        self.turn = "white"
        self.selected_piece = None
        self.mouse_last_down = False
        self.state = "running"

    def draw(self):
        for row in self.squares:
            for square in row:
                square.draw()
        if self.selected_piece:
            pygame.draw.rect(window, RED, (self.selected_piece.x * SQUARE_SIZE, self.selected_piece.y * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 5)
        for row in self.pieces:
            for piece in row:
                if piece:
                    piece.draw()

    def handle_click(self, pos):
        """Handles the logic for selecting and moving pieces."""
        col, row = pos[0] // SQUARE_SIZE, pos[1] // SQUARE_SIZE
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return

        square = self.squares[row][col]
        piece = self.pieces[row][col]

        if self.selected_piece:
            if self.is_valid_move(self.selected_piece, square):
                self.move_piece(self.selected_piece, square)
                if self.is_game_over(): self.state = f"{self.turn.capitalize()} wins!"
                else: self.switch_turn()
            self.selected_piece = None
        elif piece and piece.color == self.turn:
            self.selected_piece = piece

    def update(self):
        if self.turn == "white" and self.state == "running":
            mouse_pressed = pygame.mouse.get_pressed()[0]
            if mouse_pressed:
                self.mouse_last_down = True
            elif self.mouse_last_down:
                self.mouse_last_down = False
                self.handle_click(pygame.mouse.get_pos())

    def is_valid_move(self, piece, square):
        target_piece = self.pieces[square.y][square.x]
        if (piece.x, piece.y) == (square.x, square.y): return False
        if target_piece and target_piece.color == piece.color: return False

        dx, dy = square.x - piece.x, square.y - piece.y
        if piece.type == "P":
            direction = -1 if piece.color == "white" else 1
            start_row = 6 if piece.color == "white" else 1
            # Standard one-square move
            if dx == 0 and dy == direction and not target_piece: return True
            # Two-square move from start
            if dx == 0 and dy == 2 * direction and piece.y == start_row and not target_piece and not self.pieces[square.y - direction][square.x]: return True
            # Capture move
            if abs(dx) == 1 and dy == direction and target_piece and target_piece.color != piece.color: return True
        elif piece.type == "R":
            if (dx == 0 or dy == 0) and self.is_clear_path(piece, square): return True
        elif piece.type == "B":
            if abs(dx) == abs(dy) and self.is_clear_path(piece, square): return True
        elif piece.type == "N":
            if (abs(dx) == 1 and abs(dy) == 2) or (abs(dx) == 2 and abs(dy) == 1): return True
        elif piece.type == "Q":
            if (dx == 0 or dy == 0 or abs(dx) == abs(dy)) and self.is_clear_path(piece, square): return True
        elif piece.type == "K":
            if abs(dx) <= 1 and abs(dy) <= 1: return True
        return False

    def is_clear_path(self, piece, square):
        dx, dy = square.x - piece.x, square.y - piece.y
        x_dir = 1 if dx > 0 else -1 if dx < 0 else 0
        y_dir = 1 if dy > 0 else -1 if dy < 0 else 0
        for i in range(1, max(abs(dx), abs(dy))):
            x, y = piece.x + i * x_dir, piece.y + i * y_dir
            if self.pieces[y][x]: return False
        return True

    def move_piece(self, piece, square):
        self.pieces[piece.y][piece.x] = None
        self.pieces[square.y][square.x] = piece
        piece.x, piece.y = square.x, square.y
        
        # Pawn promotion to Queen
        if piece.type == "P" and (piece.y == 0 or piece.y == 7):
            piece.type = "Q"

    def switch_turn(self):
        self.turn = "black" if self.turn == "white" else "white"

    def is_game_over(self):
        kings = [p for row in self.pieces for p in row if p and p.type == "K"]
        return len(kings) < 2

class AIChessGame:
    def __init__(self):
        self.board = Board()
        self.model = load_model()
        self.thinking = False
        self.search_depth = 2
        self.nn_weight = 0.25
        self.time_limit_active = False # New flag for time-based search
        self.nodes_evaluated = 0
        self.last_move = "Game started"
        self.last_eval = 0.0
        self.last_material = 0.0
        self.last_nn_eval = 0.0
        self.last_think_time = 0.0
        self.search_stop_time = float('inf') # For time limiting
        self.create_ui_buttons()
    
    def create_ui_buttons(self):
        """Creates clickable UI buttons for controlling AI parameters."""
        y_offset = WINDOW_WIDTH + 35
        self.depth_buttons = []
        for i in range(1, 4):
            self.depth_buttons.append({'rect': pygame.Rect(70 + (i-1)*45, y_offset, 40, 30), 'value': i, 'label': str(i)})
        
        self.nn_buttons = []
        weights = [0.0, 0.25, 0.5, 0.75, 1.0]
        labels = ["0%", "25%", "50%", "75%", "100%"]
        start_x_nn = 360 # Adjusted for more space
        for i, (weight, label) in enumerate(zip(weights, labels)):
            self.nn_buttons.append({'rect': pygame.Rect(start_x_nn + i * 45, y_offset, 40, 30), 'value': weight, 'label': label})
            
        # Time Limit Toggle Button
        self.time_limit_button = {'rect': pygame.Rect(20, y_offset + 33, 110, 30)}
        

    def get_material_value(self, piece_type):
        return {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 0}.get(piece_type, 0)

    def calculate_material(self, board_state):
        """Calculates material balance from the current player's perspective."""
        score = 0
        for row in board_state.pieces:
            for piece in row:
                if piece:
                    value = self.get_material_value(piece.type)
                    score += value if piece.color == board_state.turn else -value
        return score

    def get_board_tensor(self, board_state):
        board_tensor = torch.zeros((8, 8))
        piece_map = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6}
        for r, row in enumerate(board_state.pieces):
            for c, piece in enumerate(row):
                if piece:
                    value = 1 if piece.color == "white" else -1
                    board_tensor[r][c] = value * piece_map.get(piece.type, 0)
        return board_tensor

    def evaluate_position(self, board_state):
        """
        Evaluates a position using a weighted combination of the neural network and material count.
        Returns a tuple: (combined_eval, nn_eval, material_eval).
        """
        kings = {"white": False, "black": False}
        for row in board_state.pieces:
            for piece in row:
                if piece and piece.type == "K":
                    kings[piece.color] = True
        
        if not kings["white"]: return (-99999, -99999, -99999) if board_state.turn == "white" else (99999, 99999, 99999)
        if not kings["black"]: return (99999, 99999, 99999) if board_state.turn == "white" else (-99999, -99999, -99999)
        
        material_eval = self.calculate_material(board_state)
        
        if self.model is None:
            return (material_eval, 0, material_eval)
        
        board_tensor = self.get_board_tensor(board_state)
        if board_state.turn == 'black':
            board_tensor = torch.flip(board_tensor, [0]) * -1

        with torch.no_grad():
            nn_eval = self.model(board_tensor.unsqueeze(0).unsqueeze(0)).item()
        
        final_eval = (1 - self.nn_weight) * material_eval + self.nn_weight * nn_eval
        return final_eval, nn_eval, material_eval

    def get_all_moves(self, color, board_state):
        moves = []
        for r, row in enumerate(board_state.pieces):
            for c, piece in enumerate(row):
                if piece and piece.color == color:
                    for sq_row in board_state.squares:
                        for square in sq_row:
                            if board_state.is_valid_move(piece, square):
                                moves.append((piece, square))
        return moves

    def minimax(self, board_state, depth, alpha, beta, is_maximizing_player):
        # Check time limit at the start of a new search depth
        if self.time_limit_active and time.time() >= self.search_stop_time:
            raise TimeoutError 

        self.nodes_evaluated += 1
        if depth == 0 or board_state.is_game_over():
            return self.evaluate_position(board_state)[0]
        
        moves = self.get_all_moves(board_state.turn, board_state)
        if not moves:
            return self.evaluate_position(board_state)[0]
        
        if is_maximizing_player:
            max_eval = float('-inf')
            for piece, square in moves:
                temp_board = copy.deepcopy(board_state)
                if temp_board.pieces[square.y][square.x] and temp_board.pieces[square.y][square.x].type == 'K':
                    return 99999
                    
                temp_board.move_piece(temp_board.pieces[piece.y][piece.x], temp_board.squares[square.y][square.x])
                temp_board.switch_turn()
                
                eval_score = self.minimax(temp_board, depth - 1, alpha, beta, False) 
                
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, max_eval)
                if beta <= alpha:
                    break 
            return max_eval
        else:
            min_eval = float('inf')
            for piece, square in moves:
                temp_board = copy.deepcopy(board_state)
                if temp_board.pieces[square.y][square.x] and temp_board.pieces[square.y][square.x].type == 'K':
                    return -99999
                    
                temp_board.move_piece(temp_board.pieces[piece.y][piece.x], temp_board.squares[square.y][square.x])
                temp_board.switch_turn()
                
                eval_score = self.minimax(temp_board, depth - 1, alpha, beta, True)
                
                min_eval = min(min_eval, eval_score)
                beta = min(beta, min_eval)
                if beta <= alpha:
                    break 
            return min_eval

    def run_minimax_search(self, depth, ai_moves):
        """Runs the search for a specific depth and returns (best_move, best_eval) or raises TimeoutError."""
        best_move, best_eval = None, float('-inf')
        alpha, beta = float('-inf'), float('inf')
        
        for ai_piece, ai_square in ai_moves:
            temp_board = copy.deepcopy(self.board)
            if temp_board.pieces[ai_square.y][ai_square.x] and temp_board.pieces[ai_square.y][ai_square.x].type == "K":
                return (ai_piece, ai_square), 99999

            temp_board.move_piece(temp_board.pieces[ai_piece.y][ai_piece.x], temp_board.squares[ai_square.y][ai_square.x])
            temp_board.switch_turn()
            
            move_eval = self.minimax(temp_board, depth - 1, alpha, beta, False) 
            
            if move_eval > best_eval:
                best_eval, best_move = move_eval, (ai_piece, ai_square)
            
            alpha = max(alpha, best_eval)
            
        return best_move, best_eval

    def get_ai_move(self):
        self.nodes_evaluated = 0
        start_time = time.time()
        ai_moves = self.get_all_moves("black", self.board)
        
        if not ai_moves:
            self.last_move = "Black: No legal moves"
            return None
        
        # Check for immediate king capture (not checked in minimax loop to save time)
        for ai_piece, ai_square in ai_moves:
             if self.board.pieces[ai_square.y][ai_square.x] and self.board.pieces[ai_square.y][ai_square.x].type == "K":
                self.last_move = f"Black: {ai_piece.type} to ({ai_square.x},{ai_square.y})"
                self.last_eval = 99999
                return (ai_piece, ai_square)
                
        # --- Search Logic ---
        best_move_at_any_depth = ai_moves[0], self.evaluate_position(self.board)[0] # Default to current position eval if search is too quick
        max_depth_to_search = self.search_depth if not self.time_limit_active else 99 
        
        if self.time_limit_active:
            self.search_stop_time = start_time + MAX_THINK_TIME
        
        for current_depth in range(1, max_depth_to_search + 1):
            if self.time_limit_active and time.time() >= self.search_stop_time:
                break
            
            try:
                # Store node count before starting search for this depth
                nodes_before_depth = self.nodes_evaluated
                
                # Run search
                current_best_move, current_best_eval = self.run_minimax_search(current_depth, ai_moves)
                
                # If a full depth was completed, update the best move
                best_move_at_any_depth = current_best_move, current_best_eval
                print(f"| Depth {current_depth} completed. Eval: {current_best_eval:.1f} | Nodes added: {self.nodes_evaluated - nodes_before_depth:,}")
                
            except TimeoutError:
                print(f"| Timeout reached. Stopping search at Depth {current_depth}.")
                break
            except Exception as e:
                # Handle unexpected errors, but don't stop the AI from making a move
                print(f"| Search error at Depth {current_depth}: {e}")
                break

        # --- Post-Search Updates ---
        elapsed = time.time() - start_time
        piece, square = best_move_at_any_depth[0]
        
        # Re-evaluate the final board state for stats
        final_board = copy.deepcopy(self.board)
        final_board.move_piece(final_board.pieces[piece.y][piece.x], final_board.squares[square.y][square.x])
        final_board.turn = 'white'
        _, self.last_nn_eval, self.last_material = self.evaluate_position(final_board)

        self.last_move = f"Black: {piece.type} to ({square.x},{square.y}) (D={current_depth-1})"
        self.last_eval = best_move_at_any_depth[1]
        self.last_think_time = elapsed
        
        print(f"AI chose: {self.last_move} | Eval: {self.last_eval:.1f} | Nodes: {self.nodes_evaluated:,} | Time: {elapsed:.2f}s")
        return best_move_at_any_depth[0]


    def handle_controls(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Depth buttons
            for button in self.depth_buttons:
                if not self.time_limit_active and button['rect'].collidepoint(event.pos):
                    self.search_depth = button['value']
                    print(f"Search depth set to: {self.search_depth}")
            # NN Weight buttons
            for button in self.nn_buttons:
                if button['rect'].collidepoint(event.pos):
                    self.nn_weight = button['value']
                    print(f"Neural Network weight set to: {self.nn_weight:.0%}")
            # Time Limit Toggle
            if self.time_limit_button['rect'].collidepoint(event.pos):
                self.time_limit_active = not self.time_limit_active
                print(f"Time Limit Mode {'ACTIVATED' if self.time_limit_active else 'DEACTIVATED'}. Max time: {MAX_THINK_TIME}s")
                if self.time_limit_active:
                    print("Depth controls are disabled in Time Limit Mode.")


    def update(self):
        if self.board.turn == "black" and not self.thinking and self.board.state == "running":
            self.thinking = True
            pygame.display.flip()
            move = self.get_ai_move()
            if move:
                piece, square = move
                self.board.move_piece(self.board.pieces[piece.y][piece.x], square)
                if self.board.is_game_over(): self.board.state = "Black wins!"
                else: self.board.switch_turn()
            else:
                self.board.state = "Stalemate!"
            self.thinking = False
        self.board.update()

    def draw_status(self):
        y_offset = WINDOW_WIDTH
        pygame.draw.rect(window, BLACK, (0, y_offset, WINDOW_WIDTH, STATUS_HEIGHT))
        
        turn_text = small_font.render(f"{self.board.turn.capitalize()}'s Turn", True, WHITE)
        window.blit(turn_text, (10, y_offset + 5))
        
        move_text = small_font.render(f"| {self.last_move}", True, GREEN)
        window.blit(move_text, (130, y_offset + 5))
        
        if self.board.state != "running":
            status_text = small_font.render(f"| {self.board.state}", True, RED)
            window.blit(status_text, (400, y_offset + 5))

        # Draw UI buttons: Depth
        depth_label = small_font.render("Depth:", True, WHITE)
        window.blit(depth_label, (10, y_offset + 40))
        for button in self.depth_buttons:
            is_selected = (button['value'] == self.search_depth) and not self.time_limit_active
            color = YELLOW if is_selected else RED if self.time_limit_active else GRAY
            pygame.draw.rect(window, color, button['rect'])
            pygame.draw.rect(window, WHITE, button['rect'], 2)
            label = small_font.render(button['label'], True, BLACK)
            window.blit(label, label.get_rect(center=button['rect'].center))
            
        # Draw UI buttons: NN Weight
        nn_label = small_font.render("NN Weight:", True, WHITE)
        window.blit(nn_label, (280, y_offset + 40))
        for button in self.nn_buttons:
            is_selected = (button['value'] == self.nn_weight)
            color = YELLOW if is_selected else GRAY
            pygame.draw.rect(window, color, button['rect'])
            pygame.draw.rect(window, WHITE, button['rect'], 2)
            label = small_font.render(button['label'], True, BLACK)
            window.blit(label, label.get_rect(center=button['rect'].center))

        # Draw UI button: Time Limit Toggle
        time_label_text = f"Time Limit ({MAX_THINK_TIME}s)"
        time_label = small_font.render(time_label_text, True, WHITE)
        color = GREEN if self.time_limit_active else GRAY
        pygame.draw.rect(window, color, self.time_limit_button['rect'])
        pygame.draw.rect(window, WHITE, self.time_limit_button['rect'], 2)
        window.blit(time_label, time_label.get_rect(center=self.time_limit_button['rect'].center))
        
        stats_text = small_font.render(f"Eval: {self.last_eval:.0f} (NN:{self.last_nn_eval:.0f} Mat:{self.last_material:.0f}) | {self.last_think_time:.1f}s | {self.nodes_evaluated:,} nodes", True, BLUE)
        window.blit(stats_text, (140, y_offset + 73))

    def draw(self):
        self.board.draw()
        self.draw_status()
        
        if self.thinking:
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_WIDTH), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            window.blit(overlay, (0, 0))
            text = font.render("AI is thinking...", True, WHITE)
            window.blit(text, text.get_rect(center=(WINDOW_WIDTH / 2, WINDOW_WIDTH / 2)))

def main():
    game = AIChessGame()
    running = True
    
    print("\n=== CONTROLS ===\n"
          "Click the buttons to adjust AI depth, NN weight, or toggle Time Limit.\n"
          f"Current settings: Depth={game.search_depth}, NN Weight={game.nn_weight:.0%}, Time Limit={'ON' if game.time_limit_active else 'OFF'}\n"
          "================\n")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            game.handle_controls(event)
        
        game.update()
        game.draw()
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()