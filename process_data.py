import json
import random

# --- Configuration ---
INPUT_FILE = 'lichess_db_eval.jsonl'
OUTPUT_FILE = 'processed_data.txt'
MAX_INPUT_LINES = 2000000       # How many lines to read from the source file.
TOTAL_OUTPUT_POSITIONS = 50000  # How many positions to write to the final file.

# --- Filtering and Balancing Parameters ---
# Defines the centipawn evaluation ranges for position categories.
BALANCED_MIN, BALANCED_MAX = -300, 300
DECISIVE_MIN, DECISIVE_MAX = 300, 1000

def process_and_filter_data(jsonl_file, output_file):
    """
    Reads raw Lichess JSONL data, then filters and balances it to create
    a dataset for training the neural network.

    The final dataset will contain only White-to-move positions with a
    90/10 split between decisive and balanced evaluations.
    """
    print(f"Starting data processing from '{jsonl_file}'...")
    print(f"This may take a while. Reading up to {MAX_INPUT_LINES:,} lines.")

    balanced_positions = []
    decisive_positions = []
    lines_processed = 0

    try:
        with open(jsonl_file, 'r') as f:
            for line in f:
                if lines_processed >= MAX_INPUT_LINES:
                    break
                lines_processed += 1

                if lines_processed % 10000 == 0:
                    print(f"\rProcessed {lines_processed:,} lines. Found {len(decisive_positions):,} decisive and {len(balanced_positions):,} balanced...", end="")

                try:
                    data = json.loads(line)
                    fen = data['fen']
                    
                    # Rule 1: Keep only positions where it is White's turn to move.
                    if ' w ' not in fen:
                        continue
                    
                    eval_score = data['evals'][0]['pvs'][0]['cp']
                    
                    # Rule 2: Categorize positions based on their evaluation score.
                    abs_eval = abs(eval_score)
                    if BALANCED_MIN <= eval_score <= BALANCED_MAX:
                        balanced_positions.append(f"{fen}\t{eval_score}\n")
                    elif DECISIVE_MIN <= abs_eval <= DECISIVE_MAX:
                        decisive_positions.append(f"{fen}\t{eval_score}\n")
                except (KeyError, IndexError, json.JSONDecodeError):
                    continue # Skip malformed lines.
    
    except FileNotFoundError:
        print(f"\nFATAL ERROR: The input file '{jsonl_file}' was not found.")
        return

    print(f"\n\nFinished reading. Found {len(decisive_positions):,} decisive and {len(balanced_positions):,} balanced positions.")
    print("Creating the final balanced dataset...")
    
    # --- Balance the dataset to the desired 90/10 split ---
    target_decisive_count = int(TOTAL_OUTPUT_POSITIONS * 0.90)
    target_balanced_count = TOTAL_OUTPUT_POSITIONS - target_decisive_count
    
    print(f"\nTargeting {TOTAL_OUTPUT_POSITIONS:,} total positions:")
    print(f"  - {target_decisive_count:,} (90%) decisive positions")
    print(f"  - {target_balanced_count:,} (10%) balanced positions")
    
    random.shuffle(decisive_positions)
    random.shuffle(balanced_positions)
    
    final_decisive = decisive_positions[:target_decisive_count]
    final_balanced = balanced_positions[:target_balanced_count]
    
    final_positions = final_decisive + final_balanced
    random.shuffle(final_positions)
    
    # --- Write the final dataset to the output file ---
    with open(output_file, 'w') as out_f:
        out_f.writelines(final_positions)
        
    print(f"\n✅ Success! Wrote {len(final_positions):,} filtered and balanced positions to '{output_file}'")

if __name__ == "__main__":
    process_and_filter_data(INPUT_FILE, OUTPUT_FILE)