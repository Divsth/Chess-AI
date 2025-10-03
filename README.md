# Chess-AI

Uses PyTorch to train and run a CNN Model. Trained via lichess.org database: https://database.lichess.org/#evals

## Instructions to Play Against Predownloaded Model: 

1. Download the latest release of Python
2. Run the following commands in a terminal:
   ```
      pip install pygame
      pip install torch torchvision torchaudio
   ```
4. Download and extract the GitHub files
5. Navigate to the extracted folder and run
   ```py main.py```
6. Enjoy!


## Instructions to Train Model:

1. Download the lichess.org database (https://database.lichess.org/#evals) and fully extract
2. Download and extract the GitHub files, remove the existing chess_model.pth from the folder.
3. Navigate to the extracted folder and paste the extracted lichess.org database
4. In the same directory, run
       ```
       py process_data.py
       py neural_network_trainer.py
       ```
5. You can now use the newly trained model to play against


(Originally Made 02/2024)
