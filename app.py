from flask import Flask
from flask import render_template, url_for, send_from_directory
import torch
torch.set_num_threads(1)
app = Flask(__name__)
from datetime import timedelta

from live_data import get_live_preds

def format_time(t):
    a = str(t).split(":")
    a[2] = a[2].split(".")[0]

    return ":".join(a[1:])
@app.route('/', methods=['GET', 'POST'])
def hello_world():
    pred_scores, live_scores, time_lefts, asts, drbs, orbs, fts, twopt, threept = get_live_preds()
    # print(live_scores)
    games = []
    teams = [("GSW", "POR"), ("MEM", "ATL"), ("LAL","LAC"), ("NYK", "OKC"), ("TOR", "MIA"), ("SAC", "DAL")]
    for i in range(len(pred_scores)):
        time_left = time_lefts[i]
        pred_score = pred_scores[i]
        live_score = live_scores[i]

        d = timedelta(seconds=time_left[1])

        # print("time?",  str(d))
        game = {"TEAM_1_ABBREVIATION": teams[i][0], 
            "TEAM_1_PTS": live_score[0],
            "TEAM_1_PRED": int(pred_score[1]),
            "TEAM_2_ABBREVIATION": teams[i][1], 
            "TEAM_2_PTS": live_score[1],
            "TEAM_2_PRED": int(pred_score[0]),
            "TIME_LEFT": format_time(d),
            "QUARTER": time_left[0],
            "TEAM_1_AST": int(asts[i][0].item()),
            "TEAM_2_AST": int(asts[i][1].item()),
            "TEAM_1_DRB": int(drbs[i][0].item()),
            "TEAM_2_DRB": int(drbs[i][1].item()),
            "TEAM_1_ORB": int(orbs[i][0].item()),
            "TEAM_2_ORB": int(orbs[i][1].item()),
            "TEAM_1_FTS": round(fts[i][0].item() * 100, 2),
            "TEAM_2_FTS": round(fts[i][1].item()* 100, 2),
            "TEAM_1_2PT": round(twopt[i][0].item()* 100,2),
            "TEAM_2_2PT": round(twopt[i][1].item()* 100, 2),
            "TEAM_1_3PT": round(threept[i][0].item()* 100,2),
            "TEAM_2_3PT": round(threept[i][1].item()* 100,2)
        }
   
        games.append(game)
    return render_template('index.html',games=games)

@app.route('/boxscores')
def live_predict():
	# get live play by play data
	# live_pbp = get_live_data()
	return "hey bestie"

@app.route('/logos/<path:filename>')
def send_file(filename):
	return send_from_directory('logos', filename)

