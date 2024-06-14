from dataset import *


dataset_dict = {
    ### Bias Dataset ###
    "open_ended_dataset": open_ended_dataset(dataset_name="open_ended_dataset"),
    "close_ended_dataset": close_ended_dataset(dataset_name="close_ended_dataset")
}


def load_dataset(dataset_name):
    if dataset_name in dataset_dict.keys():
        dataset = dataset_dict[dataset_name]
    else:
        raise NotImplementedError(f"{dataset_name} not implemented")
    return dataset
