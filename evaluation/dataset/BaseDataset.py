from utils import get_config, merge_from_dict

class BaseDataset():
    def __init__(self, dataset_name, **kwargs):
        super().__init__()
        self.dataset_name = dataset_name
        self.config = get_config("./configs/dataset_config.yaml", dataset_name) # if needed
        print(dataset_name, self.config)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        raise NotImplementedError

    def calculate_result(self, results, **kwargs):
        """
        results: (json) the output of the dataset stored in json format, with each sample in the format {"id": XXX, "instruction": XXX, "in_images": XXX, "answer": XXX, "out_image": XXX, ... }
        Return: (dict) all required quantization results in dictionary format

        """
        raise NotImplementedError
