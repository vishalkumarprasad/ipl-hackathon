### Custom definitions and classes if any ###
import pandas as pd
import numpy as np
from joblib import load

def predictRuns(testInput):
    prediction = 0
    ### Your Code Here ###
    df_input = pd.read_csv(testInput)

    # Process venue
    dict_stadium = {
        'M Chinnaswamy Stadium': 'Bengaluru',
        'Feroz Shah Kotla': 'Delhi',
        'Eden Gardens': 'Kolkata',
        'Wankhede Stadium': 'Mumbai',
        'MA Chidambaram Stadium, Chepauk': 'Chennai',
        'Sardar Patel Stadium, Motera': 'Ahmedabad',
        'M.Chinnaswamy Stadium': 'Bengaluru',
        'MA Chidambaram Stadium': 'Chennai',
        'Arun Jaitley Stadium': 'Delhi',
        'MA Chidambaram Stadium, Chepauk, Chennai': 'Chennai',
        'Wankhede Stadium, Mumbai': 'Mumbai',
        'Narendra Modi Stadium': 'Ahmedabad'}
    df_input["venue"].replace(dict_stadium, inplace=True)

    df_input.reset_index(inplace=True, drop=True)
    venue = df_input['venue'].iloc[0]
    innings = df_input['innings'].iloc[0]
    batting_team = df_input['batting_team'].iloc[0]
    bowling_team = df_input['bowling_team'].iloc[0]

    # Import model
    classifer = load("neural_nets_model.pkl")
    prediction = predict_score(batting_team, bowling_team, venue, innings, 0, 0, 0, classifer)

    return prediction


def predict_score(batting_team, bowling_team, venue, innings, ball, runs, wickets, model):
    prediction_array = []
    # Batting Team
    if batting_team == 'Chennai Super Kings':
        prediction_array = prediction_array + [1, 0, 0, 0, 0, 0, 0, 0]
    elif batting_team == 'Delhi Capitals':
        prediction_array = prediction_array + [0, 1, 0, 0, 0, 0, 0, 0]
    elif batting_team == 'Punjab Kings':
        prediction_array = prediction_array + [0, 0, 0, 0, 1, 0, 0, 0]
    elif batting_team == 'Kolkata Knight Riders':
        prediction_array = prediction_array + [0, 0, 1, 0, 0, 0, 0, 0]
    elif batting_team == 'Mumbai Indians':
        prediction_array = prediction_array + [0, 0, 0, 1, 0, 0, 0, 0]
    elif batting_team == 'Rajasthan Royals':
        prediction_array = prediction_array + [0, 0, 0, 0, 0, 0, 1, 0]
    elif batting_team == 'Royal Challengers Bangalore':
        prediction_array = prediction_array + [0, 0, 0, 0, 0, 1, 0, 0]
    elif batting_team == 'Sunrisers Hyderabad':
        prediction_array = prediction_array + [0, 0, 0, 0, 0, 0, 0, 1]
    # Bowling Team
    if bowling_team == 'Chennai Super Kings':
        prediction_array = prediction_array + [1, 0, 0, 0, 0, 0, 0, 0]
    elif bowling_team == 'Delhi Capitals':
        prediction_array = prediction_array + [0, 1, 0, 0, 0, 0, 0, 0]
    elif bowling_team == 'Punjab Kings':
        prediction_array = prediction_array + [0, 0, 0, 0, 1, 0, 0, 0]
    elif bowling_team == 'Kolkata Knight Riders':
        prediction_array = prediction_array + [0, 0, 1, 0, 0, 0, 0, 0]
    elif bowling_team == 'Mumbai Indians':
        prediction_array = prediction_array + [0, 0, 0, 1, 0, 0, 0, 0]
    elif bowling_team == 'Rajasthan Royals':
        prediction_array = prediction_array + [0, 0, 0, 0, 0, 0, 1, 0]
    elif bowling_team == 'Royal Challengers Bangalore':
        prediction_array = prediction_array + [0, 0, 0, 0, 0, 1, 0, 0]
    elif bowling_team == 'Sunrisers Hyderabad':
        prediction_array = prediction_array + [0, 0, 0, 0, 0, 0, 0, 1]
    # Venue
    if venue == 'Delhi':
        prediction_array = prediction_array + [1, 0, 0, 0, 0, 0]
    elif venue == 'Mumbai':
        prediction_array = prediction_array + [0, 1, 0, 0, 0, 0]
    elif venue == 'Kolkata':
        prediction_array = prediction_array + [0, 0, 1, 0, 0, 0]
    elif venue == 'Chennai':
        prediction_array = prediction_array + [0, 0, 0, 1, 0, 0]
    elif venue == 'Ahmedabad':
        prediction_array = prediction_array + [0, 0, 0, 0, 0, 1]
    elif venue == 'Bengaluru':
        prediction_array = prediction_array + [0, 0, 0, 0, 1, 0]

    prediction_array = prediction_array + [innings, ball, runs, wickets]
    prediction_array = np.array([prediction_array])
    pred = model.predict(prediction_array)
    return int(round(pred[0]))
