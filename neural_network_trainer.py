import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import math
import time 
import sys 
import os
import signal

# --- Configuration ---
DATA_FILENAME = 'processed_data.txt'
TRAINING_LINE_LIMIT = 50000  # How many positions to use for training
MAX_ABSOLUTE_EVAL = 1000.0  # Used for normalizing centipawn scores

# --- Global flag for graceful shutdown ---
shutdown_requested = False
loading_data = False

def signal_handler(sig, frame):
    """Handles Ctrl+C interrupts to allow for graceful shutdown."""
    global shutdown_requested
    if loading_data:
        print("\n\n⚠️  Ctrl+C detected during data loading! Exiting immediately...")
        sys.exit(0)
    else:
        print("\n\n⚠️  Ctrl+C detected! Finishing current batch and saving model...")
        shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)

class ChessDataset(Dataset):
    """
    Loads chess positions from a pre-filtered text file.
    The file is expected to be created by process_data.py and contain:
    - White-to-move positions only.
    - A 90/10 split of decisive/balanced evaluations.
    """
    def __init__(self, data_file):
        global loading_data
        loading_data = True
        
        print(f"Loading positions from '{data_file}'...")
        print(f"Note: Training is limited to the first {TRAINING_LINE_LIMIT:,} lines of this file.")
        start_time = time.time()

        self.data = []
        try:
            if not os.path.exists(data_file):
                 raise FileNotFoundError

            print("Reading and filtering positions...")
            lines_read = 0
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if lines_read >= TRAINING_LINE_LIMIT:
                        break
                    lines_read += 1

                    if lines_read % 1000 == 0:
                        print(f"\rProcessed {lines_read}/{TRAINING_LINE_LIMIT} lines, loaded {len(self.data)} positions...", end="")
                        sys.stdout.flush()

                    try:
                        full_fen, eval_score_str = line.strip().split('\t')
                        fen = full_fen.split(' ')[0]
                        eval_score = float(eval_score_str)
                        
                        normalized_eval = eval_score / MAX_ABSOLUTE_EVAL
                        board = fen_to_board(fen)
                        self.data.append((fen, board.float(), torch.tensor([normalized_eval], dtype=torch.float32)))
                    except (ValueError, IndexError):
                        continue
        
        except FileNotFoundError:
            print(f"\nFATAL ERROR: The data file '{data_file}' was not found.")
            print("Please run process_data.py first.")
            exit()

        duration = time.time() - start_time
        print("\r" + " " * 80 + "\r", end="")
        print(f"Data loading complete. Took {duration:.2f} seconds.")
        print(f"Loaded {len(self.data)} positions for training.")
        loading_data = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class ChessEvaluator(nn.Module):
    """A simple CNN to evaluate a chess board position."""
    def __init__(self):
        super(ChessEvaluator, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(-1, 128 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.tanh(self.fc2(x))
        return x * MAX_ABSOLUTE_EVAL

def fen_to_board(fen):
    """Converts the board part of a FEN string to an 8x8 tensor."""
    board = torch.zeros(8, 8)
    piece_values = {
        'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6,
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6
    }
    ranks = fen.split('/')
    for rank_idx, rank in enumerate(ranks):
        file_idx = 0
        for char in rank:
            if char.isdigit():
                file_idx += int(char)
            else:
                if file_idx < 8:
                    board[rank_idx, file_idx] = piece_values.get(char, 0)
                    file_idx += 1
    return board

def display_board(board):
    """Prints a text representation of a board tensor for debugging."""
    value_to_piece = {
        -1: 'p', -2: 'n', -3: 'b', -4: 'r', -5: 'q', -6: 'k',
        1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K', 0: '.'
    }
    print("  a b c d e f g h\n  ---------------")
    for i in range(8):
        row_str = f"{8-i}|"
        for j in range(8):
            row_str += value_to_piece[int(board[i][j].item())] + " "
        print(row_str)
    print("  ---------------")

def evaluate_sample_positions(model, dataset, device, num_samples=5):
    """Shows model predictions vs. true evaluations for a few random positions."""
    if len(dataset) < num_samples:
        return
    model.eval()
    indices = random.sample(range(len(dataset)), min(len(dataset), num_samples))
    print("\nEvaluating sample positions (all are White-to-move):")
    with torch.no_grad():
        for idx in indices:
            fen, board, true_eval_tensor = dataset[idx]
            true_eval = true_eval_tensor.item() * MAX_ABSOLUTE_EVAL
            predicted_eval = model(board.unsqueeze(0).unsqueeze(0).to(device)).item()
            print("\n" + "="*50)
            print(f"Position FEN: {fen} w")
            display_board(board)
            print(f"\nTrue evaluation: {true_eval:.2f}")
            print(f"Model prediction: {predicted_eval:.2f}")
            print(f"Difference: {abs(true_eval - predicted_eval):.2f}")

def save_model(model, epoch, loss, filepath='chess_model.pth'):
    """Saves the model and prints status including epoch and loss."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }, filepath)
    print(f"✓ Model saved to {filepath} (Epoch {epoch}, RMSE: {loss:.2f} cp). Press Ctrl+C to stop.")

if __name__ == "__main__":
    # --- GPU Setup ---
    print("="*60 + "\nGPU DIAGNOSTICS\n" + "="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print("\n⚠️  WARNING: CUDA not available. Using CPU.")
    print("="*60 + "\n")
    
    dataset = ChessDataset(DATA_FILENAME)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = ChessEvaluator().to(device)
    print(f"\nModel moved to: {device}")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 99999
    SAVE_INTERVAL_SECONDS = 15 * 60
    
    print("\nStarting training...")
    print(f"Total positions: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")
    print(f"Auto-saving every {SAVE_INTERVAL_SECONDS // 60} minutes")
    print("="*60)
    
    last_save_time = time.time()
    save_counter = 0
    last_progress_time = time.time()
    last_completed_epoch_rmse = 0.0

    for epoch in range(num_epochs):
        if shutdown_requested: break
        if not dataset: 
            print("Training data is empty. Cannot train.")
            break
            
        model.train()
        epoch_loss = 0.0
        epoch_batch_count = 0
        
        for i, (fens, boards, eval_scores) in enumerate(dataloader):
            if shutdown_requested: break
                
            boards = boards.to(device)
            eval_scores = eval_scores.to(device)
            
            outputs = model(boards.unsqueeze(1))
            loss = criterion(outputs, eval_scores * MAX_ABSOLUTE_EVAL)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_batch_count += 1
            
            current_time = time.time()
            if current_time - last_progress_time >= 10:
                avg_loss = math.sqrt(epoch_loss / epoch_batch_count) if epoch_batch_count > 0 else 0
                progress_pct = (i / len(dataloader)) * 100
                print(f'\r[{progress_pct:.1f}%] Epoch {epoch + 1}, Batch {i}/{len(dataloader)}, RMSE: {avg_loss:.2f} cp', end="")
                last_progress_time = current_time
            
            if current_time - last_save_time >= SAVE_INTERVAL_SECONDS:
                save_counter += 1
                current_epoch_rmse = math.sqrt(epoch_loss / epoch_batch_count) if epoch_batch_count > 0 else 0
                print(f"\n{'='*60}\nAuto-save triggered (15 min elapsed) - Save #{save_counter}")
                save_model(model, epoch + 1, current_epoch_rmse, f'chess_model_save_{save_counter}.pth')
                last_save_time = current_time
                print(f"{'='*60}\n")
        
        if shutdown_requested: break

        last_completed_epoch_rmse = math.sqrt(epoch_loss / epoch_batch_count) if epoch_batch_count > 0 else 0.0
        print(f'\n\n{"="*60}\nEpoch {epoch + 1}/{num_epochs} complete! Final Epoch RMSE: {last_completed_epoch_rmse:.2f} cp\n{"="*60}')
        
        evaluate_sample_positions(model, dataset, device, num_samples=3)
        
        print(f'\n{"="*60}')
        save_model(model, epoch + 1, last_completed_epoch_rmse)
        print(f'{"="*60}\n')

    final_message = "Training interrupted by user" if shutdown_requested else "Training complete!"
    print(f"\n{'='*60}\n{final_message}\n{'='*60}")
    print("\nSaving final updated model...")
    save_model(model, epoch + 1, last_completed_epoch_rmse, 'chess_model.pth')