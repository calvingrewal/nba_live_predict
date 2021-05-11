from flask import Flask
from flask import render_template, url_for, send_from_directory
from datetime import timedelta

from live_data import get_live_preds
app = Flask(__name__)

def format_time(t):
    a = str(t).split(":")
    a[2] = a[2].split(".")[0]

    return ":".join(a[1:])
@app.route('/', methods=['GET', 'POST'])
def hello_world():
    pred_scores, live_scores, time_lefts = get_live_preds()
    # print(live_scores)
    games = []
    for i in range(len(pred_scores)):
        time_left = time_lefts[i]
        pred_score = pred_scores[i]
        live_score = live_scores[i]

        d = timedelta(seconds=time_left[1])

        # print("time?",  str(d))
        print(live_scores)
        game = {"TEAM_1_ABBREVIATION": "LAC", 
            "TEAM_1_PTS": live_score[0],
            "TEAM_1_PRED": int(pred_score[0]),
            "TEAM_1_WINS_LOSSES": "45-27",
            "TEAM_2_ABBREVIATION": "LAL", 
            "TEAM_2_PTS": live_score[1],
            "TEAM_2_PRED": int(pred_score[1]),
            "TEAM_2_WINS_LOSSES": "45-27",
            "TIME_LEFT": format_time(d),
            "QUARTER": time_left[0]
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

