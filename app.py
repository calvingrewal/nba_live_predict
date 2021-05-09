from flask import Flask
from live_data  import get_live_data
from flask import render_template
app = Flask(__name__)

@app.route('/')
def hello_world():
	game = {"TEAM_1_ABBREVIATION": "LAC", 
		"TEAM_1_PTS": 88,
		"TEAM_1_WINS_LOSSES": "45-27",
		"TEAM_2_ABBREVIATION": "LAL", 
		"TEAM_2_PTS": 89,
		"TEAM_2_WINS_LOSSES": "45-27"}
	games = []
	games.append(game)
	return render_template('index.html',games=games)

@app.route('/boxscores')
def live_predict():
	# get live play by play data
	# live_pbp = get_live_data()
	return "hey bestie"


