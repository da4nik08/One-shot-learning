import pickle


def save_pkl(file_path, data):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


def get_pkl(file_path):
    with open(file_path, 'rb') as file:
        loaded_list = pickle.load(file)
    return loaded_list