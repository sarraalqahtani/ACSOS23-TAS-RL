import os
import os.path as ops
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
from plot_scripts.plot_utils import *

def get_stats(data):
    size = data.size
    mu = np.mean(data, axis=0)
    lb = mu - np.std(data, axis=0) / np.sqrt(size)
    ub = mu + np.std(data, axis=0) / np.sqrt(size)
    return mu, lb, ub

def Info_data(info_data, env_name):
    max_epi_len = 100
    test_violations = []
    test_rewards = []
    last_rewards = []
    violations_list = []
    safety_list = []
    for traj_stats in info_data:
            test_violations.append([])
            test_rewards.append(0)
            last_reward = 0
            for step_stats in traj_stats:
                test_violations[-1].append(step_stats['constraint'])
                test_rewards[-1] += step_stats['reward']
                last_reward = step_stats['reward']
            last_rewards.append(last_reward)
            violations_list.append(test_violations)
    # ep_lengths = np.array([len(t) for t in violations_list])[:max_epi_len]
    #--------------------------------------------------
    test_violations_vec = []
    for t in test_violations:
      if sum(t) > 0:
        safety = float(len(t)/max_epi_len)
        violation = 1-safety
        test_violations_vec.append(violation)
      elif sum(t)==0:
        test_violations_vec.append(0)
    #--------------------------------------------------    

    test_violations_vec = np.array(test_violations_vec)
    tv_mean, tv_lb, tv_ub = get_stats(test_violations_vec)
    tv_mean = round(tv_mean, 2)
    tv_lb = round(tv_lb, 2)
    tv_ub = round(tv_ub, 2)
    vec_violation = [tv_mean, tv_lb, tv_ub]

    if env_name=='maze':
        max_dist = -0.02
        min_dist = -5 
    elif env_name=='nav1':
        max_dist = 0.0
        min_dist = -100.0
    elif env_name=='nav2':
        max_dist = 0.0
        min_dist = -100.0

    tsk_agent_dist = np.array(last_rewards)
    task_success_rate = ((tsk_agent_dist-min_dist)/(max_dist-min_dist))*100
    task_violation_ratio = task_success_rate/(test_violations_vec+1)
    tr_mean, tr_lb, tr_ub = get_stats(task_violation_ratio)
    vec_reward = [tr_mean, tr_lb, tr_ub]
    return vec_violation, vec_reward

#################################################################################
def get_deadlock_data_for_our_safety(path, env_name):
    with open(path, 'rb') as f:
      loaded_data = pickle.load(f)
    deadlock_count = loaded_data['cycle_count']
    mean_deadlock_cycle = np.mean(deadlock_count)
    return mean_deadlock_cycle

def get_deadlock_data_our_model(env_data_path, env_name):
    paths = os.listdir(env_data_path)
    Data ={}
    Data['rate']={}
    for path in paths:
        name = path.split('atk_rate')[-1]
        info = name.split('_eps_')
        # print(info)
        atk_rate = info[0]
        eps = info[1]
        data_path = os.path.join(env_data_path, path, 'Info_for_deadlock_plotting_data.pkl')
        # print(data_path)
        Our_deadlock_data = get_deadlock_data_for_our_safety(data_path, env_name)
        Data['rate'][eps] = Our_deadlock_data
    return Data


def get_recoveryRL_deadlock_info(path, env_name):
  with open(path, 'rb') as f:
        loaded_data = pickle.load(f)
  deadlock_data = {}
  deadlock_data['algos'] = {}
  for algo in loaded_data['algos']:
      deadlock_count = loaded_data['algos'][algo]['result']['task_rec_agent']['cycle_count']
      mean_deadlock_cycle = np.mean(deadlock_count)
      deadlock_data['algos'][algo]= mean_deadlock_cycle
  return deadlock_data

def get_all_Recovery_RL_deadlock_data(logdir_recoveryRL, env_name):
  paths = os.listdir(logdir_recoveryRL)
  RecoveryRL_data = {}
  RecoveryRL_data['rate']={}
  for path in paths:
      recoveryRL_safety_data = None
      data_path = os.path.join(logdir_recoveryRL , path, 'saved_deadlock_data.pkl')
      recoveryRL_deadlock_data = get_recoveryRL_deadlock_info(data_path, env_name)
      name = path.split('Atk_rate')[-1]
      info = name.split('_eps')
      atk_rate = info[0]
      eps = info[1]
      RecoveryRL_data['rate'][eps]=recoveryRL_deadlock_data
  return RecoveryRL_data
#################################################################################


def plot_success(Data, plot_path, ablation=False, Our_Algo_name='TAS-RL', ext=''):
    mean_only=False
    atk_rate = [0,25,50,75,100]
    algo_name=[]
    TASRL_success_vec =[]
    Our_SAC_success_vec =[]
    unconstrained_success_vec =[]
    SQRL_success_vec =[]
    RRL_MF_success_vec =[]
    RSPO_success_vec =[]
    RP_success_vec =[]
    RCPO_success_vec =[]
    LR_success_vec =[]

    i=0
    only_mean=False
    Ignore_algo = ''
    for r in atk_rate:
      # print(Data[r])
      # print("RATE", r)
      for dic in Data[r]:
              if dic['Algorithm']==Our_Algo_name:
                  TASRL_success_vec.append(dic['Success'])

              elif dic['Algorithm']=='Our SAC':
                  Our_SAC_success_vec.append(dic['Success'])

              elif dic['Algorithm']=='unconstrained':
                  unconstrained_success_vec.append(dic['Success'])

              elif dic['Algorithm']=='SQRL':
                  SQRL_success_vec.append(dic['Success'])

              elif dic['Algorithm']=='RRL_MF':
                  RRL_MF_success_vec.append(dic['Success'])

              elif dic['Algorithm']=='RSPO':
                  RSPO_success_vec.append(dic['Success'])

              elif dic['Algorithm']=='RP':
                  RP_success_vec.append(dic['Success'])

              elif dic['Algorithm']=='RCPO':
                  RCPO_success_vec.append(dic['Success'])

              elif dic['Algorithm']=='LR':
                  LR_success_vec.append(dic['Success'])
    fig, axs = plt.subplots(figsize=(10,5), dpi=300)
    mean_only=False
    colors = ['cyan', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    ############################################ #1f77b4
    ub=[]
    lb=[]
    mean=[]
    for L in TASRL_success_vec:
        ub.append(L[2])
        lb.append(L[1])
        mean.append(L[0])
    if not mean_only:
      if not ablation:
          axs.fill_between(atk_rate, ub, lb, color=colors[0], alpha=.80, label='{}'.format(Our_Algo_name))
      else:
          axs.fill_between(atk_rate, ub, lb, color=colors[0], alpha=.80, label='{}: Shield(without the safety policy)'.format(Our_Algo_name))
    axs.plot(atk_rate, mean, color=colors[0], linewidth='4')
    ############################################
    ub=[]
    lb=[]
    mean=[]
    # print(Our_SAC_cost_vec)
    for L in Our_SAC_success_vec:
        ub.append(L[2])
        lb.append(L[1])
        mean.append(L[0])
    if not mean_only:
      axs.fill_between(atk_rate, ub, lb, color=colors[1], alpha=.80, label='SAC')
    axs.plot(atk_rate, mean, color=colors[1], linewidth='2')
    ############################################
    ub=[]
    lb=[]
    mean=[]
    for L in SQRL_success_vec:
        ub.append(L[2])
        lb.append(L[1])
        mean.append(L[0])
    if not mean_only:
      axs.fill_between(atk_rate, ub, lb, color=colors[2], alpha=.80, label='SQRL')
    axs.plot(atk_rate, mean, color=colors[2], linewidth='2')

    ############################################
    ub=[]
    lb=[]
    mean=[]
    for L in RCPO_success_vec:
        ub.append(L[2])
        lb.append(L[1])
        mean.append(L[0])
    if not mean_only:
      axs.fill_between(atk_rate, ub, lb, color=colors[3], alpha=.60, label='RCPO')
    axs.plot(atk_rate, mean, color=colors[3], linewidth='2')

    ############################################
    ub=[]
    lb=[]
    mean=[]
    for L in LR_success_vec:
        ub.append(L[2])
        lb.append(L[1])
        mean.append(L[0])
    if not mean_only:
      axs.fill_between(atk_rate, ub, lb, color=colors[4], alpha=.60, label='LR')
    axs.plot(atk_rate, mean, color=colors[4], linewidth='2')


    ############################################
    ub=[]
    lb=[]
    mean=[]
    for L in RSPO_success_vec:
        ub.append(L[2])
        lb.append(L[1])
        mean.append(L[0])
    if not mean_only:
      axs.fill_between(atk_rate, ub, lb, color=colors[5], alpha=.80, label='RSPO')
    axs.plot(atk_rate, mean, color=colors[5], linewidth='2')

    ############################################
    ub=[]
    lb=[]
    mean=[]
    for L in RRL_MF_success_vec:
        ub.append(L[2])
        lb.append(L[1])
        mean.append(L[0])
    if not mean_only:
      axs.fill_between(atk_rate, ub, lb, color=colors[6], alpha=.80, label='RRL_MF')
    axs.plot(atk_rate, mean, color=colors[6], linewidth='2')

    ############################################
    ub=[]
    lb=[]
    mean=[]
    for L in RP_success_vec:
        ub.append(L[2])
        lb.append(L[1])
        mean.append(L[0])
    if not mean_only:
      axs.fill_between(atk_rate, ub, lb, color=colors[7], alpha=.80, label='RP')
    axs.plot(atk_rate, mean, color=colors[7], linewidth='2')
    axs.set_xlabel('Perturbation rate', fontsize=18)
    axs.set_ylabel('Ratio of Success/violations', fontsize=18)

    plt.grid(color='gray', linewidth=1, axis='y', alpha=0.5)
    axs.legend(fontsize=12, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.2), fancybox=True, shadow=True)
    plt_dir = os.path.join(plot_path)
    if not os.path.exists(plt_dir):
        os.makedirs(plt_dir)
    plt.savefig(plt_dir+'/success_violation_ratio_plot{}.png'.format(ext), dpi=300,format='png',bbox_inches='tight')
    plt.show()  


def plot_cumulative_cost(Data, plot_path, ablation=False, Our_Algo_name='TAS-RL', ext=''):
    mean_only=False
    atk_rate = [0,25,50,75,100]
    algo_name=[]

    TAS_RL_cost_vec =[]
    Our_SAC_cost_vec =[]
    unconstrained_cost_vec =[]
    SQRL_cost_vec =[]
    RRL_MF_cost_vec =[]
    RSPO_cost_vec =[]
    RP_cost_vec =[]
    RCPO_cost_vec =[]
    LR_cost_vec =[]
    
    only_mean=False
    Ignore_algo = ''
    for r in atk_rate:
      # print(Data[r])
      # print("RATE", r)
      for dic in Data[r]:
              if dic['Algorithm']==Our_Algo_name:
                  TAS_RL_cost_vec.append(dic['Cost'])
              elif dic['Algorithm']=='Our SAC':
                  Our_SAC_cost_vec.append(dic['Cost'])
              elif dic['Algorithm']=='unconstrained':
                  unconstrained_cost_vec.append(dic['Cost'])
              elif dic['Algorithm']=='SQRL':
                  SQRL_cost_vec.append(dic['Cost'])
              elif dic['Algorithm']=='RRL_MF':
                  RRL_MF_cost_vec.append(dic['Cost'])
              elif dic['Algorithm']=='RSPO':
                  RSPO_cost_vec.append(dic['Cost'])
              elif dic['Algorithm']=='RP':
                  RP_cost_vec.append(dic['Cost'])
              elif dic['Algorithm']=='RCPO':
                  RCPO_cost_vec.append(dic['Cost'])
              elif dic['Algorithm']=='LR':
                  LR_cost_vec.append(dic['Cost'])
    fig, axs = plt.subplots(figsize=(10,5), dpi=300)
    colors = ['cyan', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    ############################################ #1f77b4
    ub=[]
    lb=[]
    mean=[]
    for L in TAS_RL_cost_vec:
        ub.append(L[2]*100)
        lb.append(L[1]*100)
        mean.append(L[0]*100)
    if not mean_only:
      if not ablation:
          axs.fill_between(atk_rate, ub, lb, color=colors[0], alpha=.80, label='{}'.format(Our_Algo_name))
      else:
          axs.fill_between(atk_rate, ub, lb, color=colors[0], alpha=.80, label='{}: Shield(without the safety policy)'.format(Our_Algo_name))
    axs.plot(atk_rate, mean, color=colors[0], linewidth='4')
    ############################################
    ub=[]
    lb=[]
    mean=[]
    # print(Our_SAC_cost_vec)
    for L in Our_SAC_cost_vec:
        ub.append(L[2]*100)
        lb.append(L[1]*100)
        mean.append(L[0]*100)
    if not mean_only:
      axs.fill_between(atk_rate, ub, lb, color=colors[1], alpha=.80, label='SAC')
    axs.plot(atk_rate, mean, color=colors[1], linewidth='2')
    ############################################
    ub=[]
    lb=[]
    mean=[]
    for L in SQRL_cost_vec:
        ub.append(L[2]*100)
        lb.append(L[1]*100)
        mean.append(L[0]*100)
    if not mean_only:
      axs.fill_between(atk_rate, ub, lb, color=colors[2], alpha=.80, label='SQRL')
    axs.plot(atk_rate, mean, color=colors[2], linewidth='2')
    ############################################
    ub=[]
    lb=[]
    mean=[]
    for L in RCPO_cost_vec:
        ub.append(L[2]*100)
        lb.append(L[1]*100)
        mean.append(L[0]*100)
    if not mean_only:
      axs.fill_between(atk_rate, ub, lb, color=colors[3], alpha=.60, label='RCPO')
    axs.plot(atk_rate, mean, color=colors[3], linewidth='2')
    ############################################
    ub=[]
    lb=[]
    mean=[]
    for L in LR_cost_vec:
        # print(L[2]*100)
        ub.append(L[2]*100)
        lb.append(L[1]*100)
        mean.append(L[0]*100)
    if not mean_only:
      axs.fill_between(atk_rate, ub, lb, color=colors[4], alpha=.60, label='LR')
    axs.plot(atk_rate, mean, color=colors[4], linewidth='2')
    ############################################
    ub=[]
    lb=[]
    mean=[]
    for L in RSPO_cost_vec:
        ub.append(L[2]*100)
        lb.append(L[1]*100)
        mean.append(L[0]*100)
    if not mean_only:
      axs.fill_between(atk_rate, ub, lb, color=colors[5], alpha=.80, label='RSPO')
    axs.plot(atk_rate, mean, color=colors[5], linewidth='2')

    ############################################
    ub=[]
    lb=[]
    mean=[]
    for L in RRL_MF_cost_vec:
        ub.append(L[2]*100)
        lb.append(L[1]*100)
        mean.append(L[0]*100)
    if not mean_only:
      axs.fill_between(atk_rate, ub, lb, color=colors[6], alpha=.80, label='RRL_MF')
    axs.plot(atk_rate, mean, color=colors[6], linewidth='2')

    ############################################
    ub=[]
    lb=[]
    mean=[]
    for L in RP_cost_vec:
        ub.append(L[2]*100)
        lb.append(L[1]*100)
        mean.append(L[0]*100)
    if not mean_only:
      axs.fill_between(atk_rate, ub, lb, color=colors[7], alpha=.80, label='RP')
    axs.plot(atk_rate, mean, color=colors[7], linewidth='2')


    axs.set_xlabel('Perturbation-rate', fontsize=18)
    axs.set_ylabel('Cumulative Constraint Violations', fontsize=18)

    plt.grid(color='gray', linewidth=1, axis='y', alpha=0.5)
    axs.legend(fontsize=12, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.2), fancybox=True, shadow=True)
    plt_dir = os.path.join(plot_path)
    if not os.path.exists(plt_dir):
        os.makedirs(plt_dir)
    plt.savefig(plt_dir+'/cumulative_cost_plot{}.png'.format(ext), dpi=300,format='png',bbox_inches='tight')
    plt.show()    

def get_processed_dataframe(Data):
  # print(Data)
  algo_name = []
  processed_dic1 = {}
  processed_dic2 = {}
  for r in Data:
    mean_cost_col_name = str(r)+"_mean_cost"
    mean_cost_lb_col_name = str(r)+"_lb_cost"
    mean_cost_ub_col_name = str(r)+"_ub_cost"
    processed_dic1[mean_cost_col_name]=[]
    processed_dic1[mean_cost_lb_col_name]=[]
    processed_dic1[mean_cost_ub_col_name]=[]

    mean_success_col_name = str(r)+"_mean_success_viol_ratio"
    mean_success_lb_col_name = str(r)+"_lb_success_viol_ratio"
    mean_success_ub_col_name = str(r)+"_ub_success_viol_ratio"
    processed_dic2[mean_success_col_name]=[]
    processed_dic2[mean_success_lb_col_name]=[]
    processed_dic2[mean_success_ub_col_name]=[]
    data_r = Data[r]
    # print(algos)
    for d in data_r:
      if not d['Algorithm'] in algo_name:
        algo_name.append(d['Algorithm'])
      processed_dic1[mean_cost_col_name].append(float(d['Cost'][0])*100)
      processed_dic1[mean_cost_lb_col_name].append(float(d['Cost'][1])*100)
      processed_dic1[mean_cost_ub_col_name].append(float(d['Cost'][2])*100)

      processed_dic2[mean_success_col_name].append(round(float(d['Success'][0]),2))
      processed_dic2[mean_success_lb_col_name].append(round(float(d['Success'][1]),2))
      processed_dic2[mean_success_ub_col_name].append(round(float(d['Success'][2]),2))
  
  df1 = pd.DataFrame(processed_dic1)
  df1['algos']=algo_name

  df2 = pd.DataFrame(processed_dic2)
  df2['algos']=algo_name

  return df1, df2


def get_result_data(env_name, TASRL_data_path, RecRL_data_path, plot_path, ablation=False, Our_Algo_name='TAS-RL', ext=''):
  algo_name = []
  success_value = []
  cost_value = []
  Data = {}
  #######################################
  ########################################
  TASRL_RL_paths = os.listdir(TASRL_data_path)
  for path in TASRL_RL_paths:
      name = path.split('atk_rate_')[-1]
      info = name.split('_eps_')
      atk_rate = info[0]
      # print(atk_rate)
      atk_rate_value=float(atk_rate)*100
      atk_rate_value = int(atk_rate_value)
      data_path = os.path.join(TASRL_data_path, path, 'Info_for_plotting_data.pkl')
      with open(data_path, "rb") as f:
        loaded_data = pickle.load(f)
      tasrl_info_data = loaded_data['tsk_rec_info']
      cost, success = Info_data(tasrl_info_data, env_name)
      if atk_rate_value not in Data:
          Data[atk_rate_value] = []
      data = {"Algorithm":Our_Algo_name,
              "Cost": cost,
              "Success":success
            }
      Data[atk_rate_value].append(data)

      tsk_info_data = loaded_data['tsk_info']
      cost, success = Info_data(tsk_info_data, env_name)
      if atk_rate_value not in Data:
          Data[atk_rate_value] = []
      data = {"Algorithm":"Our SAC",
              "Cost": cost,
              "Success":success
            }
      Data[atk_rate_value].append(data)
  ########################################
  ########################################
  paths = os.listdir(RecRL_data_path)
  for path in paths:
      atk_rate = path.split('_eps')[1]
      atk_rate = float(atk_rate)*100
      atk_rate_value = int(atk_rate)
      data_path_recrl = os.path.join(RecRL_data_path , path, 'saved_exp_data.pkl')
      with open(data_path_recrl, 'rb') as f:
        loaded_data = pickle.load(f)
      for algo in loaded_data['algos']:
        algo_name.append(algo)
        info = loaded_data['algos'][algo]['result']['task_rec_agent']['info']
        cost_recRL, success_recRL = Info_data(info, env_name)
        if atk_rate_value not in Data:
            Data[atk_rate_value] = []
        data = {"Algorithm":algo,
                "Cost": cost_recRL,
                "Success":success_recRL
              }
        Data[atk_rate_value].append(data)
  save_file_name_cost = os.path.join(plot_path,'test_cost_data.csv')
  save_file_name_success_viol_ratio = os.path.join(plot_path,'test_succ_viol_data.csv')
  df_cost, df_success_viol = get_processed_dataframe(Data)
  plot_cumulative_cost(Data, plot_path, ablation, Our_Algo_name,ext=ext)
  plot_success(Data, plot_path, ablation, Our_Algo_name, ext=ext)
  df_cost.to_csv(save_file_name_cost)
  df_success_viol.to_csv(save_file_name_success_viol_ratio)
  
  
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def draw_deadlock_info_plot(recRL_data, our_data, env_name, save_dir):
    deadlock_cycle_SQRL = []
    deadlock_cycle_RRL_MF = []
    deadlock_cycle_RCPO = []
    deadlock_cycle_LR = []
    deadlock_cycle_RSPO = []
    deadlock_cycle_RP = []
    deadlock_cycle_Our_model = []
    deadlock_cycle_SAC = []

    for r in recRL_data['rate']:
        deadlock_cycle_SQRL.append(recRL_data['rate'][r]['algos']['SQRL'])
        deadlock_cycle_RRL_MF.append(recRL_data['rate'][r]['algos']['RRL_MF'])
        deadlock_cycle_RCPO.append(recRL_data['rate'][r]['algos']['RCPO'])
        deadlock_cycle_LR.append(recRL_data['rate'][r]['algos']['LR'])
        deadlock_cycle_RSPO.append(recRL_data['rate'][r]['algos']['RSPO'])
        deadlock_cycle_RP.append(recRL_data['rate'][r]['algos']['RP'])
        deadlock_cycle_Our_model.append(our_data['rate'][r])
    
    size_spec = get_fig_size()
    fig, ax = plt.subplots(figsize=(int(size_spec['fig_size_x']), int(size_spec['fig_size_y'])), dpi=300)
  
    labels = ['0', '25', '50', '75', '100']
    atk_rate = np.array([0,25,50,75,100])

    # epsilon = np.array([0.0, 0.25, 0.50, 0.75, 1.00]) #[0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    x = np.array([0,25,50,75,100])  # the label locations

    width = 1  # the width of the bars

    rects2 = ax.bar(x-3*width, deadlock_cycle_SQRL, width, label='SQRL',color='hotpink' )
    rects3 = ax.bar(x-2*width, deadlock_cycle_RCPO, width, label='RCPO',color='lightgray' )
    rects4 = ax.bar(x-1*width, deadlock_cycle_LR, width, label='LR',color='pink' )
    rects5 = ax.bar(x*width, deadlock_cycle_RSPO, width, label='RSPO',color='brown' )
    rects6 = ax.bar(x+1*width, deadlock_cycle_RRL_MF, width, label='RRL_MF',color='mediumpurple' )
    rects7 = ax.bar(x+2*width, deadlock_cycle_RP, width, label='RP',color='orange' )
    rects8 = ax.bar(x+3*width, deadlock_cycle_Our_model, width, label='TAS-RL',color='cyan' )
    
    ax.set_xticks([0,25,50,75,100])
    ax.set_yticks([0,25,50,75,100])
    ax.tick_params(axis='x', labelsize=int(size_spec['ticks']))
    ax.tick_params(axis='y', labelsize=int(size_spec['ticks']))
    
    ax.set_xlabel('Perturbation rate (%)', fontsize=int(size_spec['label']))
    ax.set_ylabel('Cycle count', fontsize=int(size_spec['label']))
    
    plt.grid(color='gray', linewidth=1, axis='y', alpha=0.5)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=int(size_spec['label']), fancybox=True, shadow=True, ncol=4)
    
    plt_dir = os.path.join(save_dir, env_name)
    if not os.path.exists(plt_dir):
        os.makedirs(plt_dir)
    plt.savefig(plt_dir+'/deadlock_cycle_info_plot.png',dpi=300,format='png',bbox_inches='tight')
    plt.show()

  





