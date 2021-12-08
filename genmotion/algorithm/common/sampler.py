#sampler

class Sampler(object):
    def __init__(self, model_path, opt, device) -> None:
        self.model_path = model_path
        self.opt = opt
        self.device = device

    def sample(self, input_motion):
        pass