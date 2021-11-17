import torch

class HumanAct12Params:
    def __init__(self, mode="train") -> None:
        self.exp_mode = mode

        # if self.exp_mode == "train":
        self.expname= 'exps'
        self.folder= 'exps/humanact12'
        self.cuda= True
        self.batch_size= 30
        self.num_epochs= 2000
        self.lr= 0.0001
        self.snapshot= 100
        self.dataset= 'humanact12'
        self.num_frames= 60
        self.sampling= 'conseq'
        self.sampling_step= 1
        self.pose_rep= 'rot6d'
        self.max_len= -1
        self.min_len= -1
        self.num_seq_max= -1
        self.glob= 'translation'
        self.glob_rot= [3.141592653589793, 0, 0]
        self.translation= True
        self.debug= False
        self.modelname= 'cvae_transformer_rc_rcxyz_kl'
        self.latent_dim= 256
        self.lambda_kl= 1e-05
        self.lambda_rc= 1.0
        self.lambda_rcxyz= 1.0
        self.jointstype= 'vertices'
        self.vertstrans= False
        self.num_layers= 8
        self.activation= 'gelu'
        self.no_vertstrans= True
        self.losses= ['rc', 'rcxyz', 'kl']
        self.lambdas= {'rc': 1.0, 'rcxyz': 1.0, 'kl': 1e-05}
        self.device= torch.device(type='cuda')

        if self.exp_mode == "sample":
            # sampling model
            self.num_samples_per_action = 10
            self.num_frames = 60
            self.fact_latent = 1
            self.jointstype = "smpl"
            self.vertstrans = False
            self.no_vertstrans = False
            self.mode = "gen"
            self.num_classes = 12
            self.nfeats = 6
            self.njoints = 25
        
        # cuda option
        self.cuda = True
        self.cpu = False

        # model
        self.modeltype= 'cvae'
        self.archiname= 'transformer'