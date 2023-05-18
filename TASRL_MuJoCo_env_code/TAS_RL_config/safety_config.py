import argparse
def parse_maze():
    parser = argparse.ArgumentParser(description='TASRL RL Safety Arguments')
    parser.add_argument('--configure-env', default='none', help='')
    parser.add_argument('--env-change', type=float, default=1.0, help='multiplier for variation of env dynamics')
    parser.add_argument('--env-name', default='maze', help='Gym environment (default: maze)')
    parser.add_argument('--exp-data-dir', default='/Experimental_Data/', help='Set experiment data location')
    parser.add_argument('--device', default='', help='run on CUDA (default: False)')
    parser.add_argument('--logdir', default='runs', help='exterior log directory')
    parser.add_argument('--logdir_suffix', default='', help='log directory suffixy')
    parser.add_argument('--epoch', type=int, default=1, help='model updates per simulator step (default: 1)')   #Nav 1 (1)
    parser.add_argument('--seed', type=int, default=123456, help='random seed (default: 123456)')
    parser.add_argument('--train_start', type=int, default=10, help='No of episode to start training')
    parser.add_argument('--num_steps', type=int, default=1000000, help='maximum number of steps (default: 1000000)')
    parser.add_argument('--num_eps', type=int, default=1000000, help='maximum number of episodes (default: 1000000)')
    parser.add_argument('--model_path', default='runs', help='exterior log directory')
    #=========================================================================================================
    parser.add_argument('--hidden_size', type=int, default=256, help='hidden size (default: 256)')
    parser.add_argument('--saved_model_path', default='/Trained_Models/Safety_policy/Feb-06-2023_17_43_PM_SafetyAgent_maze/TASRL_model/Best/Feb-06-2023_Best_safety_ratio1', help='exterior log directory')
    #=========================================================================================================
    parser.add_argument('--gamma',type=float,default=0.99,help='discount factor for reward (default: 0.99)')
    parser.add_argument( '--tau',type=float,default=0.005, help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--alpha',type=float, default=0.20, help='Temperature parameter α determines the relative importance of the entropy\
                                                                  term against the reward (default: 0.2)')
    parser.add_argument('--lr', type=float, default=0.0003, help='learning rate (default: 0.0003)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size (default: 256)')
    parser.add_argument('--policy', default='Gaussian', help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--target_update_interval', type=int, default=1, help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, help='size of replay buffer (default: 1000000)')
    #=================================================================================
    #=================================================================================
    parser.add_argument('--beta', type=float, default=0.7, help='Rollout agent - adversarial sample ratio default 0.7')
    parser.add_argument('--eta', type=float, default=0.5, help='Rollout agent - expert sample ration default 0.5*(1-adversarial sample ratio)')
    #=================================================================================
    return parser.parse_args()

def parse_nav1():
    parser = argparse.ArgumentParser(description='TASRL Safety Arguments')
    parser.add_argument('--configure-env', default='none', help='')
    parser.add_argument('--env-name', default='nav1', help='Gym environment (default: maze)')
    parser.add_argument('--env-change', type=float, default=1.0, help='multiplier for variation of env dynamics')
    parser.add_argument('--exp-data-dir', default='/Experimental_Data/', help='Set experiment data location')
    parser.add_argument('--device', default='', help='run on CUDA (default: False)')
    parser.add_argument('--logdir', default='runs', help='exterior log directory')
    parser.add_argument('--logdir_suffix', default='', help='log directory suffixy')
    parser.add_argument('--epoch', type=int, default=1, help='model updates per simulator step (default: 1)')   #Nav 1 (1)
    parser.add_argument('--seed', type=int, default=123456, help='random seed (default: 123456)')
    parser.add_argument('--train_start', type=int, default=10, help='No of episode to start training')
    parser.add_argument('--num_steps', type=int, default=1000000, help='maximum number of steps (default: 1000000)')
    parser.add_argument('--num_eps', type=int, default=1000000, help='maximum number of episodes (default: 1000000)')
    parser.add_argument('--model_path', default='runs', help='exterior log directory')
    #=========================================================================================================
    parser.add_argument('--hidden_size', type=int, default=512, help='hidden size (default: 256)')
    parser.add_argument('--saved_model_path', default='/Trained_Models/Safety_policy/Feb-06-2023_05_20_AM_SafetyAgent_nav1/TASRL_model/Best/Feb-06-2023_Best_safety_ratio1', help='exterior log directory')
    #=========================================================================================================
    parser.add_argument('--gamma',type=float,default=0.99,help='discount factor for reward (default: 0.99)')
    parser.add_argument( '--tau',type=float,default=0.005, help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--alpha',type=float, default=0.20, help='Temperature parameter α determines the relative importance of the entropy\
                                                                  term against the reward (default: 0.2)')
    parser.add_argument('--lr', type=float, default=0.00003, help='learning rate (default: 0.0003)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size (default: 256)')
    parser.add_argument('--policy', default='Gaussian', help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--target_update_interval', type=int, default=1, help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, help='size of replay buffer (default: 1000000)')
    #=================================================================================
    #=================================================================================
    parser.add_argument('--beta', type=float, default=0.8, help='Rollout agent - adversarial sample ratio default 0.7')
    parser.add_argument('--eta', type=float, default=0.5, help='Rollout agent - expert sample ration default 0.5*(1-adversarial sample ratio)')
    #=================================================================================
    return parser.parse_args()

def parse_nav2():
    parser = argparse.ArgumentParser(description='TASRL Safety Arguments')
    parser.add_argument('--configure-env', default='none', help='')
    parser.add_argument('--env-name', default='nav2', help='Gym environment (default: maze)')
    parser.add_argument('--env-change', type=float, default=1.0, help='multiplier for variation of env dynamics')
    parser.add_argument('--exp-data-dir', default='/Experimental_Data/', help='Set experiment data location')
    parser.add_argument('--device', default='', help='run on CUDA (default: False)')
    parser.add_argument('--logdir', default='runs', help='exterior log directory')
    parser.add_argument('--logdir_suffix', default='', help='log directory suffixy')
    parser.add_argument('--epoch', type=int, default=1, help='model updates per simulator step (default: 1)')   #Nav 1 (1)
    parser.add_argument('--seed', type=int, default=123456, help='random seed (default: 123456)')
    parser.add_argument('--train_start', type=int, default=10, help='No of episode to start training')
    parser.add_argument('--num_steps', type=int, default=1000000, help='maximum number of steps (default: 1000000)')
    parser.add_argument('--num_eps', type=int, default=1000000, help='maximum number of episodes (default: 1000000)')
    parser.add_argument('--model_path', default='runs', help='exterior log directory')
    #=========================================================================================================
    parser.add_argument('--hidden_size', type=int, default=512, help='hidden size (default: 512)')
    parser.add_argument('--saved_model_path', default='/Trained_Models/Safety_policy/Feb-06-2023_19_40_PM_SafetyAgent_nav2/TASRL_model/Best/Feb-06-2023_Best_safety_ratio1', help='exterior log directory')
    #=========================================================================================================
    parser.add_argument('--gamma',type=float,default=0.99,help='discount factor for reward (default: 0.99)')
    parser.add_argument( '--tau',type=float,default=0.005, help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--alpha',type=float, default=0.20, help='Temperature parameter α determines the relative importance of the entropy\
                                                                  term against the reward (default: 0.2)')
    parser.add_argument('--lr', type=float, default=0.0003, help='learning rate (default: 0.0003)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size (default: 256)')
    parser.add_argument('--policy', default='Gaussian', help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--target_update_interval', type=int, default=1, help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, help='size of replay buffer (default: 1000000)')
    #=================================================================================
    parser.add_argument('--beta', type=float, default=0.7, help='Rollout agent - adversarial sample ratio default 0.7')
    parser.add_argument('--eta', type=float, default=0.5, help='Rollout agent - expert sample ration default 0.5*(1-adversarial sample ratio)')
    #=================================================================================
    return parser.parse_args()



def get_safety_args(env_name):
    if env_name=='maze':
        return parse_maze()
    elif env_name=='nav1':
        return parse_nav1()
    elif env_name=='nav2':
        return parse_nav2()


    