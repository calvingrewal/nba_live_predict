import torch
from datetime import datetime
from LSTM_Model import LSTM_Model
from prepare_data import cols
import pandas as pd

test_data = torch.load('test_data.pt')
test_data_unnormalized = torch.load('test_data_unnormalized.pt')
times = torch.load('times.pt')
game_lengths = torch.load('game_lengths.pt')
state_dict25 = torch.load('model25.pt')
state_dict75 = torch.load('model75.pt')
model25 = LSTM_Model(test_data.shape[-1], 512)
model25.load_state_dict(state_dict25)
model25.eval()

model75 = LSTM_Model(test_data.shape[-1], 512)
model75.load_state_dict(state_dict75)
model75.eval()

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

seconds_per_play = 5
# start_time = datetime.strptime("05/08/21 23:50", "%m/%d/%y %H:%M")
start_times = {
<<<<<<< Updated upstream
    16: datetime.strptime("05/10/21 18:09", "%m/%d/%y %H:%M"),
    69: datetime.strptime("05/10/21 18:09", "%m/%d/%y %H:%M"),
=======
    16: datetime.strptime("05/10/21 01:54", "%m/%d/%y %H:%M"),
    69: datetime.strptime("05/09/21 22:42", "%m/%d/%y %H:%M"),
>>>>>>> Stashed changes
}
def get_live_preds():
    plays = torch.zeros(len(start_times), 700, test_data.shape[-1])
    live_scores = torch.zeros(len(start_times), 2)
    time_left = torch.zeros(len(start_times), 2)
    
    play_indices = []
    for i, (game_idx, dt) in enumerate(start_times.items()):

        seconds_since_start = (datetime.now() - dt).total_seconds()
        
        num_plays = min(699, int(seconds_since_start // seconds_per_play))

        print(f"{seconds_since_start} seconds have passed since start, returning {num_plays} plays")
        
        game_length = game_lengths[game_idx].item()

        play_idx = min(num_plays, game_length-1)
        play_indices.append(play_idx)
        live_scores[i] = test_data[game_idx, play_idx, -2:].detach().clone()
        time_left_sin = test_data[game_idx, play_idx, 0]
        time_left_cos = test_data[game_idx, play_idx, 1]

        time_left[i] = torch.FloatTensor([times[game_idx,play_idx,0], times[game_idx, play_idx, 1]])
        # print('time?', time_left[i])
        plays[i] = get_k_plays(num_plays, game_idx)


        plays[i] = get_k_plays(num_plays, game_idx)

    output_seq25 = model25(plays)
    output_seq75 = model75(plays)
        
    B = output_seq.shape[0]
    last_output = torch.zeros(B, 2) 
    for j in range(B):
        
        last_output[j] = output_seq[j, play_indices[j], :]   
    
    preds = last_output * 15 + 100
    # print(time_left)
    live_scores = live_scores * 15 + 100
    print(preds)
    return preds.tolist(), live_scores.tolist(), time_left.tolist()

def get_k_plays(k, game_idx):
    plays = test_data[game_idx, :, :].detach().clone()
    plays[k+1:] = -1000

    return plays

# get_live_data()