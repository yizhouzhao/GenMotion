import torch

from ..common.sampler import Sampler
from .models import VRNN

class HDM05Sampler(Sampler):
    def __init__(self, model_path, opt, device) -> None:
        super().__init__(model_path, opt, device)

        # load model
        self.model = VRNN(self.opt)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model = self.model.to(self.device)

    def sample(self, input_motion):
        max_target_length = self.opt.get("max_target_length", 20)
        input_motion = torch.FloatTensor(input_motion).to(self.device)
        if len(input_motion.shape) < 3:
            input_motion = input_motion.unsqueeze(0)

        input_motion = input_motion.transpose(0, 1) # [Batch x T x D] -> [T x Batch x D]

        self.model.eval()
        target_animation = []
        for i in range(max_target_length):
            input_frame_start = max(0, input_motion.shape[1] - self.opt.get("input_motion_length", 50))
            target_motion = self.model.sample_next(input_motion[input_frame_start:, :, :])
            #print(target_motion.shape)
            target_animation.append(target_motion.cpu().data.tolist())
            # print("dim check", input_motion.shape, target_motion.shape)
            input_motion = torch.cat([input_motion, target_motion.unsqueeze(1)], dim = 0)
        

        return target_animation


