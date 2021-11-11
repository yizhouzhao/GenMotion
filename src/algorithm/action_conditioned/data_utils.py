from .datasets.humanact12poses import HumanAct12Poses

def get_datasets(data_path, parameters):
    """
    Get training/testing dataset
    """
    DATASET = HumanAct12Poses

    dataset = DATASET(split="train", datapath=data_path, **parameters)

    train = dataset

    # test: shallow copy (share the memory) but set the other indices
    from copy import copy
    test = copy(train)
    test.split = test
    test._test = test._train

    datasets = {"train": train,
                "test": test}

    # add specific parameters from the dataset loading
    dataset.update_parameters(parameters)

    return datasets