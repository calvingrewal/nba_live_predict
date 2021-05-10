import torch
from datetime import datetime
from LSTM_Model import LSTM_Model
from prepare_data import cols
import pandas as pd
test_data = torch.load('test_data.pt')

state_dict = torch.load('model2.pt')
model = LSTM_Model(test_data.shape[-1], 512)
model.load_state_dict(state_dict)
model.eval()

df_test = pd.read_csv("./2020-21pbpfeatures.csv.zip")
sizes = df_test.groupby(['URL']).size()
def build_batch(data, homeLabels, awayLabels, indices, masks):
    batch_data = data[indices]
    batch_labels_home = torch.tensor(homeLabels[indices].tolist())
    batch_labels_away = torch.tensor(awayLabels[indices].tolist())

    batch_masks = masks[indices]
    for i in range(batch_data.shape[0]):
        batch_data[i, batch_masks[i]+1:] = -1000
    return batch_data, batch_labels_home, batch_labels_away, batch_masks

seconds_per_play = 1
# start_time = datetime.strptime("05/08/21 23:50", "%m/%d/%y %H:%M")
start_times = {
    0: datetime.strptime("05/09/21 22:21", "%m/%d/%y %H:%M"),
    1: datetime.strptime("05/09/21 22:42", "%m/%d/%y %H:%M"),
}
def get_live_preds():
    plays = torch.zeros(len(start_times), 700, test_data.shape[-1])
    for i, (game_idx, dt) in enumerate(start_times.items()):

        seconds_since_start = (datetime.now() - dt).total_seconds()

        num_plays = min(699, int(seconds_since_start // seconds_per_play))

        print(f"{seconds_since_start} seconds have passed since start, returning {num_plays} plays")

        plays[i] = get_k_plays(num_plays, game_idx)

    output_seq = model(plays)
        
    B = output_seq.shape[0]
    last_output = torch.zeros(B, 2) 
    for j in range(B):
        last_output[j] = output_seq[j, num_plays, :]   
    
    scaled = last_output * 15 + 100
    print(scaled)
    return scaled.tolist()

def get_k_plays(k, game_idx):
    plays = test_data[game_idx, :, :]
    plays[k+1:] = -1000

    return plays

# get_live_data()