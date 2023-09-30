
class Config:
    def __init__(self,
                 add_bp = False,
                name_addition = None,
                agent_class = "ppo",
                frame_stack = None,
                logdir = "./tensorboard/",
                num_episodes = 3_000_000,
                env_index = 1,
                learning_rate = 0.0001,
                network_architecture = "[128,128]",
                features_dim = 512,
                seed = 50,
                partial_obs = False,
                generalBT = True,
                    ):
        self.add_bp = add_bp
        self.name_addition = name_addition
        self.agent_class = agent_class
        self.frame_stack = frame_stack
        self.logdir = logdir
        self.num_episodes = num_episodes
        self.env_index = env_index
        self.learning_rate = learning_rate
        self.network_architecture = network_architecture
        self.features_dim = features_dim
        self.seed = seed
        self.partial_obs = partial_obs
        self.generalBT = generalBT

    def to_string(self):
        # return f"Config({self.__dict__})"
        string = ""
        for k,v in self.__dict__.items():
            string += f"{k}: {v}\n"

        return string
    

config_list=[

    Config(add_bp = False,
            name_addition = None,
            agent_class = "ppo",
            frame_stack = None,
            logdir = "./tensorboard/",
            num_episodes = 3_000_000,
            env_index = 1,
            learning_rate = 0.0001,
            network_architecture = "[128,128]",
            features_dim = 512,
            seed = 50,
                ),

    Config(add_bp = False,
            name_addition = None,
            agent_class = "ppo",
            frame_stack = None,
            logdir = "./tensorboard/",
            num_episodes = 3_000_000,
            env_index = 1,
            learning_rate = 0.0001,
            network_architecture = "[128,128]",
            features_dim = 512,
            seed = 50,
                ),
]



    
    

        
    