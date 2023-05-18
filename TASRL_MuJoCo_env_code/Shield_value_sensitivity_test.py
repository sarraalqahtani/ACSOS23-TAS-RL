from TAS_RL.sac import SAC
from TAS_RL.safety_agent_VAE import Safety_Agent_VAE
from TAS_RL_config.victim_config import get_victim_args
from TAS_RL_config.adversary_config import get_adv_args
from TAS_RL_config.safety_config import get_safety_args
import argparse
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
from matplotlib import pyplot 
import pickle
import matplotlib.pyplot as plt
import os
from RecoveryRL.recRL_comparison_exp_aaa_atk import *
import warnings
warnings.filterwarnings("ignore")
import random as random
import matplotlib.pyplot as plt


#====================================================================
def torchify(x):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    return torch.FloatTensor(x).to(device).unsqueeze(0)
#====================================================================

def use_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def plot_histogram(data, path, ext=''):
      plt.figure()
      print(data)
      plt.hist(data, 
              10,
              density=True,
              histtype='bar',
              facecolor='b',
              alpha=0.5
              )
      # plt.hist(data, density=False, bins=20)
      plt.ylabel('Frequency')
      plt.xlabel('Critic Value')
      plt.savefig(path+'/critc_val_hist{}.png'.format(ext), format='png')
      plt.close()

def plot_return(shield_threshold, return_data_vec, path, ext=''):
      fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
      plt.rcParams.update({'font.size': 25})
      ax.fill_between(shield_threshold, return_data_vec, 0, color='skyblue', 
                                alpha=.25, label='')
      ax.plot(shield_threshold, return_data_vec, color='skyblue', linewidth='3')
      ax.set_xticks(shield_threshold)
      ax.tick_params(axis='x', labelsize=20)
      ax.tick_params(axis='y', labelsize=20)
      ax.set_xlabel('Safety threshold value', fontsize=30)
      ax.set_ylabel('{}'.format(ext), fontsize=30)
      plt.savefig(path+'/sensitivity{}_wrt_different_threshold.png'.format(ext),dpi=300,format='png',bbox_inches='tight')


def eval_critic_val_episode(test_env,
                            task_agent, 
                            adv_agent=None,
                            adv_shield=None,
                            safety_policy=None,
                            use_random=False,
                            cap=500,
                            safety_shield=False
                            ):
    num_steps = 0
    reward_sum = 0
    cost_sum = 0
    critic_vec = []
    indx = 0
    capacity = cap
    ########################
    state = test_env.reset()
    done = False
    while not done:
        num_steps+=1
        if not use_random:
            action = adv_agent.select_action(state)
            # action = task_agent.select_action(state)
        else:
            action = test_env.action_space.sample()
        #-----------------------------------
        # shield_val = shield_val.cpu().detach().numpy()[0]
        next_state, reward, done, info = test_env.step(action)
        if info['constraint']>0:
            if safety_shield:
                shield_val = safety_policy.get_shield_value(torchify(state), torchify(action)).detach().cpu().numpy()[-1]
            else:
                shield_val = adv_shield.get_shield_value(torchify(state), torchify(action)).detach().cpu().numpy()[-1]
            if len(critic_vec) < capacity:
              critic_vec.append(None)
            critic_vec[indx]=(shield_val[0])
            indx = (indx+1)%capacity
        #-----------------------------------
        state = next_state
        done = done or (num_steps==test_env._max_episode_steps)
        if done:
          #  test_env.close()
           break
    return critic_vec

def eval_shield_cost_sensitivity(test_env, 
                                task_agent, 
                                adv_agent=None, 
                                adv_shield=None,
                                safety_policy=None,
                                shield_threshold = 0,
                                atk_rate=0.0,
                                safety_shield=False
                                ):
    num_steps = 0
    reward_sum = 0
    cost_sum = 0
    indx = 0
    critic_vec = []
    ########################
    state = test_env.reset()
    done = False
    while not done:
        num_steps+=1
        action_tsk = task_agent.select_action(state)
        # action_tsk = test_env.action_space.sample()
        #-----------------------------------
        if np.random.rand()<atk_rate:
            action_tsk = adv_agent.select_action(state)
        #-----------------------------------
        # 
        if safety_shield:
            shield_val = safety_policy.get_shield_value(torchify(state), torchify(action_tsk)).detach().cpu().numpy()[-1]
        else:
            shield_val = adv_shield.get_shield_value(torchify(state), torchify(action_tsk)).detach().cpu().numpy()[-1]
        # print(shield_val)
        if shield_val>=shield_threshold:
            action = safety_policy.select_action(state, eval=True)
        else:
            action = action_tsk
        #-----------------------------------
        next_state, reward, done, info = test_env.step(action)
        reward_sum += reward
        state = next_state
        done = done or (num_steps==test_env._max_episode_steps)
        if done:
           break
    return num_steps, reward_sum

def run(env_name=None, cfg=None, eval_epi_no=100):
    eval_epi_no = eval_epi_no
    shield_threshold = np.arange(0.0, 4.0, 0.1)
  #   shield_threshold = [.10, .20, .30, 0.40, .50, .60, .70, .80, .90, 1, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00]
    if env_name == "maze":
        from env.maze import MazeNavigation
        env = MazeNavigation()
    elif env_name == 'nav1':
        from env.navigation1 import Navigation1
        env = Navigation1()
    elif env_name == 'nav2':
        from env.navigation2 import Navigation2
        env = Navigation2()

    agent_cfg =  get_victim_args(env_name)
    safety_cfg = get_safety_args(env_name)
    adv_cfg = get_adv_args(env_name)
    
    current_path = os.getcwd()
    expdata_path = current_path+agent_cfg.exp_data_dir
    
    expert_agent_path = current_path + agent_cfg.saved_model_path
    safety_policy_path = current_path + safety_cfg.saved_model_path
    shield_path = current_path + adv_cfg.saved_model_path
    
    data_dir = current_path+cfg.exp_data_dir
    agent_observation_space = env.observation_space.shape[0]
    agent_action_space = env.action_space.shape[0]
    logdir = ' '
    #====================================================================
    expert_agent = SAC(agent_observation_space,
                    agent_action_space,
                    agent_cfg,
                    logdir,
                    env=env
                    )
    task_algorithm = "SAC"
    expert_agent.load_best_model(expert_agent_path)
    #====================================================================
    adv_agent = SAC(agent_observation_space,
                    agent_action_space,
                    adv_cfg,
                    logdir,
                    env=env
                    )
    adv_agent.load_best_model(shield_path)
    #====================================================================
    safety_agent = Safety_Agent_VAE(observation_space = agent_observation_space, 
                                  action_space= agent_action_space,
                                  args=safety_cfg,
                                  logdir=logdir,
                                  env = env,
                                  adv_agent=adv_agent
                                  )
    safety_agent.load_safety_model(safety_policy_path)
    TAS_RL_plot_path = os.path.join(expdata_path, env_name, 'TAS_RL', 'sensitivity','different_shield_threshold')
    Safety_data_vec = []
    success_safety_vec = []
    ##########################################################
    # shield_threshold = np.arange(0, 50, 2)
    reward_list_adv_sh = []
    cost_list_adv_sh = []

    reward_list_sfty_sh = []
    cost_list_sfty_sh = []

    critic_vec_tsk_all_adv_shield = []
    critic_vec_rand_all_adv_shield = []
    critic_vec_tsk_all_safety_shield = []
    critic_vec_rand_all_safety_shield = []
    ##########################################################
    for thres in tqdm(shield_threshold):
        reward_sum1 = 0
        cost_sum1 = 0

        reward_sum2 = 0
        cost_sum2 = 0

        critic_value_vector_rand_adv = []
        critic_value_vector_tsk_adv = []
        critic_value_vector_rand_sfty = []
        critic_value_vector_tsk_sfty = []  
        for i in range(eval_epi_no):
              c_vec_rand = eval_critic_val_episode(env,
                                              expert_agent,
                                              adv_agent=adv_agent,
                                              adv_shield = adv_agent,
                                              safety_policy = safety_agent,
                                              use_random=True
                                              )
              c_vec_tsk = eval_critic_val_episode(env,
                                                  expert_agent,
                                                  adv_agent=adv_agent,
                                                  adv_shield = adv_agent,
                                                  safety_policy = safety_agent,
                                                  use_random=False
                                                  )
              c_vec_rand2 = eval_critic_val_episode(env,
                                              expert_agent,
                                              adv_agent=adv_agent,
                                              adv_shield = adv_agent,
                                              safety_policy = safety_agent,
                                              use_random=True,
                                              safety_shield=True
                                              )
              c_vec_tsk2 = eval_critic_val_episode(env,
                                                   expert_agent,
                                                   adv_agent=adv_agent,
                                                   adv_shield = adv_agent,
                                                   safety_policy = safety_agent,
                                                   use_random=False,
                                                   safety_shield=True
                                                   )
              critic_value_vector_rand_adv.extend(c_vec_rand)   
              critic_value_vector_tsk_adv.extend(c_vec_tsk)   
              critic_value_vector_rand_sfty.extend(c_vec_rand2)  
              critic_value_vector_tsk_sfty.extend(c_vec_tsk2) 
              cost1, reward1 = eval_shield_cost_sensitivity(env,
                                                            expert_agent,
                                                            adv_agent=adv_agent,
                                                            adv_shield = adv_agent,
                                                            safety_policy=safety_agent,
                                                            shield_threshold=thres
                                                            )
              cost_sum1+=cost1
              reward_sum1+=reward1

              cost2, reward2 = eval_shield_cost_sensitivity(env,
                                                            expert_agent,
                                                            adv_agent=adv_agent,
                                                            adv_shield = adv_agent,
                                                            safety_policy=safety_agent,
                                                            shield_threshold=thres,
                                                            safety_shield=True
                                                            )
              cost_sum2+=cost2
              reward_sum2+=reward2
        avg_reward1 = float(reward_sum1/eval_epi_no)
        avg_cost1 = float(cost_sum1/eval_epi_no)
        reward_list_adv_sh.append(avg_reward1)
        cost_list_adv_sh.append(avg_cost1)

        avg_reward2 = float(reward_sum2/eval_epi_no)
        avg_cost2 = float(cost_sum2/eval_epi_no)
        reward_list_sfty_sh.append(avg_reward2)
        cost_list_sfty_sh.append(avg_cost2)
        
        critic_vec_rand_all_adv_shield.extend(critic_value_vector_rand_adv)
        critic_vec_tsk_all_adv_shield.extend(critic_value_vector_tsk_adv)
        critic_vec_tsk_all_safety_shield.extend(critic_value_vector_rand_sfty)
        critic_vec_rand_all_safety_shield.extend(critic_value_vector_tsk_sfty)
    
    plot_path= use_path(data_dir+'/Plots')
    plot_histogram(critic_vec_rand_all_adv_shield, plot_path, ext='random_adv_sh_ALL')
    plot_histogram(critic_vec_tsk_all_adv_shield, plot_path, ext='task_adv_sh_ALL')
    plot_histogram(critic_vec_rand_all_safety_shield, plot_path, ext='random_stfy_sh_ALL')
    plot_histogram(critic_vec_tsk_all_safety_shield, plot_path, ext='task_sfty_sh_ALL')
    plot_return(shield_threshold,reward_list_adv_sh,plot_path, ext='Reward_adv_shield')
    plot_return(shield_threshold,cost_list_adv_sh,plot_path, ext='Epi_length_adv_shield')
    plot_return(shield_threshold,reward_list_sfty_sh,plot_path, ext='Reward_sfty_shield')
    plot_return(shield_threshold,cost_list_sfty_sh,plot_path, ext='Epi_length_sfty_shield')
    ##########################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configure-env', default='none', help='Set test environment to setup all configuration')
    parser.add_argument('--exp-data-dir', default='/Experimental_Data/Sensitivity', help='Set experiment data location')
    arg = parser.parse_args()
    name = arg.configure_env
    test_epi_no = 5
    run(env_name=name, cfg=arg, eval_epi_no=test_epi_no)