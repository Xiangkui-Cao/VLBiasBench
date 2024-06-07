from dataset import *


dataset_dict = {
    ### Bias Dataset ###
    "vlbold-g": VLBOLDDataset(dataset_name="vlbold-g", type="gender"),
    "vlbold-pi": VLBOLDDataset(dataset_name="vlbold-pi", type="political_ideology"),
    "vlbold-p": VLBOLDDataset(dataset_name="vlbold-p", type="profession"),
    "vlbold-r": VLBOLDDataset(dataset_name="vlbold-r", type="race"),
    "vlbold-ri": VLBOLDDataset(dataset_name="vlbold-ri", type="religious_ideology"),
    "open_ended_dataset": close_ended_dataset(dataset_name="close_ended_dataset")
}


def load_dataset(dataset_name):
    if dataset_name in dataset_dict.keys():
        dataset = dataset_dict[dataset_name]
    else:
        raise NotImplementedError(f"{dataset_name} not implemented")
    return dataset
