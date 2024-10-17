import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

class ChessDataset(Dataset):
    """
    Custom dataset class for chess positions and evaluations.
    Loads chess positions in FEN notation and their corresponding evaluations from a file.
    Inherits from PyTorch's Dataset class for compatibility with DataLoader.
    """
    def __init__(self, data_file):
        """
        Initialize the dataset by loading positions from file.
        
        Args:
            data_file (str): Path to the data file containing FEN positions and evaluations
                            Format: 'FEN_string\tevaluation_score' per line
        """
        self.data = []
        with open(data_file, 'r') as f:
            count = 0
            # Load first 25000 positions (limit for memory efficiency)
            for line in f.readlines()[:25000]:
                count += 1
                # Progress indicator every 1000 positions
                if count % 1000 == 0:
                    print(f"Loading data file: {count} lines processed")

                # Parse FEN and evaluation score
                fen, eval_score = line.strip().split('\t')
                board = fen_to_board(fen)  # Convert FEN to tensor representation
                self.data.append((fen, board.float(), float(eval_score)))

    def __len__(self):
        """Return the total number of positions in the dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single position by index.
        
        Args:
            idx (int): Position index
            
        Returns:
            tuple: (FEN string, board tensor, evaluation score)
        """
        return self.data[idx]

class ChessEvaluator(nn.Module):
    """
    Neural network model for chess position evaluation.
    Uses convolutional layers to process the board state followed by fully connected layers.
    """
    def __init__(self):
        """Initialize the network architecture"""
        super(ChessEvaluator, self).__init__()
        # Convolutional layers for pattern recognition
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Input -> 32 filters
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32 -> 64 filters
        # Fully connected layers for evaluation
        self.fc1 = nn.Linear(64 * 8 * 8, 256)  # Flattened conv output -> 256 neurons
        self.fc2 = nn.Linear(256, 1)  # Final evaluation score
        self.relu = nn.ReLU()  # Activation function

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor representing board state (batch_size, 1, 8, 8)
            
        Returns:
            torch.Tensor: Position evaluation scores (batch_size, 1)
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(-1, 64 * 8 * 8)  # Flatten for fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def fen_to_board(fen):
    """
    Convert a chess position from FEN notation to a tensor representation.
    
    Args:
        fen (str): Position in Forsyth-Edwards Notation
        
    Returns:
        torch.Tensor: 8x8 tensor representing the board state
        Pieces are represented by signed integers:
        - Positive for black pieces, negative for white
        - Magnitude indicates piece type (1=pawn, 2=knight, etc.)
    """
    board = torch.zeros(8, 8)
    ranks = fen.split('/')
    for rank_idx, rank in enumerate(ranks):
        file_idx = 0
        for char in rank:
            if char.isdigit():
                # Skip empty squares
                file_idx += int(char)
            elif char != ' ':
                if file_idx < 8:
                    # Convert piece character to numerical value
                    board[7 - rank_idx, file_idx] = char_to_value(char)
                    file_idx += 1
                else:
                    break
    return board

def char_to_value(char):
    """
    Convert a chess piece character to its corresponding numerical value.
    
    Args:
        char (str): Chess piece character (P/p=pawn, N/n=knight, etc.)
        
    Returns:
        int: Numerical value representing the piece
        - Positive for black pieces, negative for white
        - Magnitude indicates piece type (1=pawn, 2=knight, etc.)
    """
    piece_values = {
        'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6,  # Black pieces
        'P': -1, 'N': -2, 'B': -3, 'R': -4, 'Q': -5, 'K': -6  # White pieces
    }
    return piece_values.get(char, 0)  # 0 for empty squares or invalid characters

def display_board(board):
    """
    Display a chess board in text format.
    
    Args:
        board (torch.Tensor): 8x8 tensor representing the board state
    """
    # Dictionary to convert numerical values back to piece characters
    value_to_piece = {
        1: 'p', 2: 'n', 3: 'b', 4: 'r', 5: 'q', 6: 'k',  # Black pieces
        -1: 'P', -2: 'N', -3: 'B', -4: 'R', -5: 'Q', -6: 'K',  # White pieces
        0: '.'  # Empty squares
    }
    
    # Print board with coordinates
    print("  a b c d e f g h")
    print("  ---------------")
    for i in range(8):
        row = f"{8-i}|"
        for j in range(8):
            value = int(board[i][j].item())
            row += value_to_piece[value] + " "
        print(row)
    print("  ---------------")

def evaluate_sample_positions(model, dataset, num_samples=5):
    """
    Evaluate and display model predictions for random sample positions.
    
    Args:
        model (ChessEvaluator): Trained model
        dataset (ChessDataset): Dataset containing positions
        num_samples (int): Number of positions to evaluate
    """
    model.eval()  # Set model to evaluation mode
    indices = random.sample(range(len(dataset)), num_samples)
    
    print("\nEvaluating sample positions:")
    with torch.no_grad():  # Disable gradient computation for inference
        for idx in indices:
            fen, board, true_eval = dataset[idx]
            
            # Get model's prediction
            board_input = board.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            predicted_eval = model(board_input).item()
            
            # Display results
            print("\n" + "="*50)
            print(f"Position FEN: {fen}")
            print("\nBoard position:")
            display_board(board)
            print(f"\nTrue evaluation: {true_eval:.2f}")
            print(f"Model prediction: {predicted_eval:.2f}")
            print(f"Difference: {abs(true_eval - predicted_eval):.2f}")

def save_model(model, filepath='chess_model.pth'):
    """
    Save the trained model weights to a file.
    
    Args:
        model (ChessEvaluator): Model to save
        filepath (str): Path where to save the model
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath='chess_model.pth'):
    """
    Load a trained model from a file.
    
    Args:
        filepath (str): Path to the saved model file
        
    Returns:
        ChessEvaluator: Loaded model in evaluation mode
    """
    model = ChessEvaluator()
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model

if __name__ == "__main__":
    # Initialize dataset and model
    dataset = ChessDataset('processed_data.txt')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = ChessEvaluator()
    criterion = nn.MSELoss()  # Mean Squared Error loss for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training hyperparameters
    num_epochs = 10
    samples_per_epoch = 3  # Number of sample positions to evaluate after each epoch
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        batch_count = 0
        
        # Train on batches
        for i, (fens, boards, eval_scores) in enumerate(dataloader):
            # Forward pass
            outputs = model(boards.unsqueeze(1).float())
            loss = criterion(outputs, eval_scores.unsqueeze(1).float())

            # Backward pass and optimization
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights

            # Track training progress
            running_loss += loss.item()
            batch_count += 1

            # Print progress every 100 batches
            if i % 100 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}, Batch {i}, Loss: {loss.item():.4f}')

        # Print epoch summary
        epoch_loss = running_loss / batch_count
        print(f'\nEpoch {epoch + 1} complete. Average loss: {epoch_loss:.4f}')
        
        # Evaluate sample positions after each epoch
        evaluate_sample_positions(model, dataset, samples_per_epoch)
        print("\n" + "="*50 + "\n")

    print("Training complete!")
    
    # Final evaluation of more positions
    print("\nFinal evaluation of sample positions:")
    evaluate_sample_positions(model, dataset, num_samples=10)

    # Save the trained model
    save_model(model)