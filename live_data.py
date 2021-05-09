import torch
def get_live_data():
    test_data = torch.load('test_data.pt')
    print(f"test data shape: {test_data.shape}")
    return "hey eric!"



get_live_data()