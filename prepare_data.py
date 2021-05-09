import pandas as pd
import torch
import numpy as np

cols = ['GameTimeLeftSin', 'GameTimeLeftCos', 'Home 2-pt shot', 'Away 2-pt shot',
        'Home 3-pt shot', 'Away 3-pt shot', 'HomeMade 2-pt shot',
        'AwayMade 2-pt shot', 'HomeMade 3-pt shot', 'AwayMade 3-pt shot',
        'AwayDRB', 'HomeDRB', 'AwayORB', 'HomeORB',
        'AwayFoul', 'HomeFoul',
        'AwayTurnover', 'HomeTurnover', 'AwayFreeThrow', 'HomeFreeThrow',
        'AwayMadeFreeThrow', 'HomeMadeFreeThrow',
        'Home 2-pt shotTotal', 'Away 2-pt shotTotal', 'HomeMade 2-pt shotTotal',
        'AwayMade 2-pt shotTotal', 'Home 3-pt shotTotal', 'Away 3-pt shotTotal',
        'HomeMade 3-pt shotTotal', 'AwayMade 3-pt shotTotal',
        'AwayTurnoverTotal', 'HomeTurnoverTotal', 'AwayFreeThrowTotal',
        'HomeFreeThrowTotal', 'AwayMadeFreeThrowTotal',
        'HomeMadeFreeThrowTotal', 'AwayAssistTotal', 'HomeAssistTotal',
        'AwayDRBTotal', 'HomeDRBTotal', 'AwayORBTotal', 'HomeORBTotal',
        'AwayFoulTotal', 'HomeFoulTotal']
PAD_LENGTH = 700
def pad_data(data):
    vec_length = len(data[0][0])
    blank_pad = [[-1000] * vec_length]
    # masks = [] # index of the last unpadded input for each game
    for game in data:
    #     print(len(game), len(game[0]))
        padding = PAD_LENGTH - len(game)
        # masks.append(int(0.75 * (len(game)-1)))
        game.extend(blank_pad * padding)
    # return np.array(masks)
    return None
    
def prep_data(path):
    """ loads our "realtime" data and saves it to disk to be used later """
    df_test = pd.read_csv(path)
    test_game_urls = df_test['URL'].unique()
    print(f'read csv of length {len(df_test)}')

    testHomeLabels = df_test.groupby('URL')['HomeFinalScore'].max()
    testAwayLabels = df_test.groupby('URL')['AwayFinalScore'].max()

    print(f'data has {len(testHomeLabels)} games')
    df_test['listified'] = df_test[cols].apply(list, axis=1).aggregate(list)

    test_data = df_test.groupby('URL')['listified'].aggregate(list)
    test_data = test_data.tolist()

    test_masks = pad_data(test_data)
    test_data = torch.tensor(test_data)

    torch.save(test_data, 'test_data.pt')

if __name__ == "__main__":
    print('preparing data!')
    prep_data('./2020-21pbpfeatures.csv.zip')