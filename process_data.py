import json

def convert_data(jsonl_file, output_file):
    with open(jsonl_file, 'r') as f, open(output_file, 'w') as out_f:
        for line in f:
            data = json.loads(line)
            fen = data['fen']
            try:
                eval_score = data['evals'][0]['pvs'][0]['cp']
                out_f.write(f"{fen}\t{eval_score}\n")
            except KeyError:
                continue

# Usage
convert_data('lichess_db_eval.jsonl', 'processed_data.txt')