from flask import Flask
from live_data  import get_live_data
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/live_predict')
def live_predict():
    # get live play by play data
    live_pbp = get_live_data()
    return live_pbp