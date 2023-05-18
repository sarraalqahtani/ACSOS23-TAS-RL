import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import os.path as osp
import math
import numpy as np
from torch.distributions import Normal
import torch
import torch.nn.functional as F
from datetime import datetime
from torch.optim import Adam, SGD
from TAS_RL.network import GaussianPolicy, QNetwork, DeterministicPolicy, QNetworkConstraint, StochasticPolicy, grad_false
from TAS_RL.sac import SAC
from TAS_RL.vae import *
from TAS_RL.utils import *
import sys
import pickle

def use_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path
    
class Safety_Agent_VAE(object):
        def __init__(self,
                    observation_space,
                    action_space,
                    args,
                    logdir,
                    env= None,
                    adv_agent_path = None,
                    im_shape = None,
                    temp_env = None,
                    policy_model_path=None,
                    adv_agent = None,
                    device = None,
                    look_ahead=2,
                    adv_shield=True
                    ):
            self.learning_steps = 0
            self.action_space = action_space
            self.observation_space = observation_space
            self.env = env
            self.cfg = args
            #---------------------------------------------
            #self.policy_type = args.policy    #Gaussian   Deterministic #else Stochastic
            self.gamma = args.gamma
            self.tau = args.tau
            self.alpha = args.alpha
            self.env_name = args.env_name
            self.adv_agent_path = adv_agent_path
            self.best_policy_loss = -np.inf
            self.best_critic_loss = np.inf
            self.adv_sheild = adv_shield
            self.target_update_interval = args.target_update_interval
            if device==None:
                self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            else:
                self.device = device

            self.logdir = logdir
            self.adv_critic = None
            self.look_ahead = look_ahead
            #=======================================================================
            if not adv_agent ==None:
                self.adv_critic = adv_agent
            #---------------------------------------------------------------------
            # self.safety_critic = Qvalue_Network(observation_space,
            #                                     action_space,
            #                                     self.cfg.hidden_size
            #                                     ).to(device=self.device)
            
            self.safety_critic = QNetwork(observation_space,
                                                action_space,
                                                self.cfg.hidden_size
                                                ).to(device=self.device)

            self.safety_policy = StochasticPolicy(self.observation_space,
                                                  self.action_space,
                                                  self.cfg.hidden_size,
                                                  action_space=self.env.action_space
                                                 ).to(self.device)
            self.safety_critic_optim = Adam(self.safety_critic.parameters(), lr=args.lr)
            self.safety_policy_optim = Adam(self.safety_policy.parameters(), lr=args.lr)
            #=====================================================================
            #=====================================================================
            
            self.VAE_state_predictor = VAE(self.observation_space,
                                           self.action_space,
                                           self.cfg.hidden_size,
                                           latent_dim = action_space
                                          ).to(self.device)
            self.VAE_criterion = nn.MSELoss()
            self.VAE_optimizer = optim.Adam(self.VAE_state_predictor.parameters(), lr=0.001)
            print("VAE SAFETY POLICY")
        #=====================================================================
        #=====================================================================
        def get_shield_value(self, state, action):
            if self.adv_sheild:
              if not self.adv_critic==None:
                  with torch.no_grad():
                      q = self.adv_critic.get_shield_value(state, action)
                      q =  q.detach().cpu().numpy()[-1]
                      return q[0]
            else:
                  with torch.no_grad():
                      q1, q2 = self.safety_critic(state, action)
                      q = torch.min(q1, q2)
                      q = q.detach().cpu().numpy()[-1]
                  return q[0]

        #************************************************************************************
        #****************ABLATION TEST WITH SHIELD+RANDOM POLICY************************************
        def select_ablation_action(self):
            action = self.env.action_space.sample()
            return action
        #************************************************************************************
        #************************************************************************************

        #************************************************************************************
        #*****************TAS-RL Safety Policy*******************************
        def select_action(self, state, eval=False):
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            if eval is False:
                action, _, _,_ = self.safety_policy.sample(state)
            else:
                _, _, action,_ = self.safety_policy.sample(state)
            action = action.detach().float().cpu().numpy()[0]
            # action = np.clip(action[0], self.env.action_space.low.min(), self.env.action_space.high.max())
            return action
        #************************************************************************************
        #************************************************************************************
        def evaluate_next_N_cost(self, constraint_batch, state_batch, action_batch, discount_factor=0.9, step=10):
            costs = []
            last_state_vec=[]
            last_action_vec = []
            for i in range(len(constraint_batch)):
                C_t = 0
                pw = 0
                last_state = None
                last_action = None
                sum_adv_val = 0
                k=0
                for j in range(i, min(i+step, len(constraint_batch))):
                    C_t += constraint_batch[j]
                    pw += 1
                    state=state_batch[j]
                    action = action_batch[j]
                    st = torch.FloatTensor(state).to(self.device)
                    act, _, _, _ = self.safety_policy.sample(st)
                    with torch.no_grad():
                        adv_val = self.adv_critic.critic(st, act)
                    sum_adv_val+= adv_val
                    last_state = state
                    last_action = action
                costs.append(C_t)
                last_state_vec.append(last_state)
                last_action_vec.append(last_action)
            return costs, last_state_vec, last_action_vec, sum_adv_val   
        
        
        def get_batch_tensor(self, memory, batch_size):
                state_batch, action_batch, reward_batch, contraint_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
                state_batch = torch.FloatTensor(state_batch).to(self.device)
                next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
                action_batch = torch.FloatTensor(action_batch).to(self.device)
                reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
                contraint_batch = torch.FloatTensor(contraint_batch).to(self.device).unsqueeze(1)
                mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
                return state_batch, action_batch, next_state_batch, reward_batch, contraint_batch, mask_batch

        def update_vae(self, state, action, next_state):
            self.VAE_optimizer.zero_grad()
            reconstructed_state = self.VAE_state_predictor(state, action)
            loss = self.VAE_criterion(reconstructed_state, next_state)
            loss.backward()
            self.VAE_optimizer.step()
            return loss

        def update_safety_policy(self, memory, batch_size, m=2): 
            state_batch, action_batch, next_state_batch, reward_batch, contraint_batch, mask_batch = self.get_batch_tensor(memory, batch_size)
            if m!=0:
               vae_loss = self.update_vae(state_batch, action_batch, next_state_batch)

            sampled_action, _, _, _ = self.safety_policy.sample(state_batch)
            next_state_q_risk1, next_state_q_risk2 = self.safety_critic(state_batch, sampled_action)
            
            # next_state_q_risk1, next_state_q_risk2 = self.safety_critic(next_state_batch, sampled_action)
            adv_val1 = 0
            adv_val2 = 0
            with torch.no_grad():
                trajectory_adv_val1, trajectory_adv_val2 = self.adv_critic.critic(state_batch, sampled_action)
                adv_val1+=trajectory_adv_val1
                adv_val2+=trajectory_adv_val2
                m_state = next_state_batch
                for _ in range(m):
                    sampled_action, _, _, _ = self.safety_policy.sample(m_state)
                    vae_predicted_states = self.VAE_state_predictor.predict(m_state, sampled_action)
                    tmp_trajectory_adv_val1, tmp_trajectory_adv_val2 = self.adv_critic.critic(vae_predicted_states, sampled_action)
                    adv_val1+=tmp_trajectory_adv_val1
                    adv_val2+=tmp_trajectory_adv_val2
                    m_state = vae_predicted_states
                     
            target_risk1 = contraint_batch+adv_val1
            target_risk2 = contraint_batch+adv_val2
            # target_risk1 = adv_val1
            # target_risk2 = adv_val2
            est_err1 = F.mse_loss(next_state_q_risk1, target_risk1) 
            est_err2 = F.mse_loss(next_state_q_risk2, target_risk2) 
            #==========Critic Loss Optim=======================
            self.safety_critic_optim.zero_grad()
            (est_err1+est_err2).backward()
            self.safety_critic_optim.step()

            curr_sampled_action, log_pi_agent, mean_rec, std_rec = self.safety_policy.sample(state_batch)
            q_risk1, q_risk2 = self.safety_critic(state_batch,curr_sampled_action)
            with torch.no_grad():
                _,log_pi_adv, mean_adv, std_adv = self.adv_critic.policy.sample(state_batch)

            log_pi_adv = log_pi_adv.float()
            log_pi_agent = log_pi_agent.float().unsqueeze(1)   
            target_max_q_pi =  torch.max(q_risk1, q_risk2)
    
            # target_max_sqf_pi = ((target_max_q_pi-log_pi_adv)-log_pi_agent)  
            # target_max_sqf_pi = -(log_pi_agent-(log_pi_adv-target_max_q_pi))  
            target_max_sqf_pi = -(log_pi_agent-(target_max_q_pi-log_pi_adv))
            safety_policy_loss = target_max_sqf_pi.mean() 
            
            self.safety_policy_optim.zero_grad()
            (safety_policy_loss).backward()
            self.safety_policy_optim.step()
            est_err = est_err1+est_err2
            if m!=0:
              vae_loss=vae_loss.item()
            else:
              vae_loss=0
            return safety_policy_loss.item(), est_err.item(), vae_loss
       
        def update_parameters(self, memory, batch_size, nu=None, safety_critic=None):
            self.learning_steps+=1
            safety_policy_loss, critic_loss, vae_loss  = self.update_safety_policy(memory, batch_size, m=self.look_ahead)
            return  safety_policy_loss, critic_loss, vae_loss 

    
        def save_best_safety_model(self, ratio=0, interval=0):
            time = datetime.now().strftime("%b-%d-%Y")
            if not interval==0:
                model_dir = os.path.join(self.logdir,'TASRL_model/Interval','TASRL_at_interval_{}_safety_{}'.format(interval, ratio))
            else:
                model_dir = os.path.join(self.logdir,'TASRL_model/Best','{}_Best_safety_ratio{}'.format( time, ratio))
            model_dir = use_path(model_dir)
            
            policy_path = os.path.join(model_dir, 'TASRL_policy_net.pth')
            self.safety_policy.save(policy_path)
            
            critic_path = os.path.join(model_dir, 'TASRL_critic_net.pth')
            self.safety_critic.save(critic_path)  
            
            self.VAE_state_predictor.save(model_dir)
          
        def load_safety_model(self, path):
                safety_policy_path = os.path.join(path, 'TASRL_policy_net.pth')
                safety_critic_path = os.path.join(path, 'TASRL_critic_net.pth')
                # print(safety_policy_path)
                # print(safety_critic_path)
                self.safety_policy.load(safety_policy_path,  self.device)
                self.safety_critic.load(safety_critic_path,  self.device)
                grad_false(self.safety_policy)
                grad_false(self.safety_critic)

        