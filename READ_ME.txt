
####################################################################################
                    TAS-RL
##################################################################################### 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
             Directory Information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
After unzipping the given code zip file, the directory structure should be as follows
-------------------------------------------------------------------------------------   
   Directory                        Information
-------------------------------------------------------------------------------------
- READ_ME.txt                       (Contains instruction to run relevant code)
- TASRL_MuJoCo_env_code                     
    - TAS_RL                      (Contains all the code files relevant to TAS-RL)
        -adv_trainer.py             (Adversary trainer)
        -memory.py
        -network.py
        -sac.Python         
        -safety_agent_VAE.py        (TAS_RL safety policy)
        -safety_trainer.py          (TAS_RL trainer)
        -utils.py
        -victim_trainer.py          (task policy trainer)
    - TAS_RL_config               (Contains configuration of aversary, task and safety policy for each environment)
        -adversary_config.py
        -safety_config.py
        -victim_config.py
    - Trained_Models                (Contains trained network parameters for Maze, Navigation 1 and Navigation 2)
        - Adversary                 (Trained adversary's network parameters)
        - Safety_policy             (Trained TAS-RL network parameters)
        - Victim                    (Trained SAC Task policy network parameters)
    - config                        (Contains Environment related configuration)
    - env                           (Contains the Enviroment files)
    - Experimental_Data             (Contains our experimental results and relevant data)
    - plot_scripts                  (Contains python file relevant to plotting the experimental results)
    - RecoveryRL                    (Contains files to conduct testing on the 7 baselines)
        -  RecoveryRL_Model         (Contains 7 trained baselines model parameters. We trained the baselines using the code from "https://github.com/abalakrishna123/recovery-rl")      
            -Maze
                - LR
                - RCPO
                - RP
                - RRL_MF
                - RSPO
                - SQRL
                - unconstrained
            - Navigation1
                -...
            - Navigation2
                -...
        - network.py
        - recoveryRL_args.py
        - recoveryRL_agne.py
        - recoveryRL_models.py
        - recoveryRL_qrisk.py
        - recoveryRL_utils.py
        - recRL_comparison_exp_aaa_atk.py
        - recRL_comparison_exp_random_atk.py
        - render_RecRL_aaa_atk.py
        - render_RecRL_random_atk.py

    - requirements.txt              (Contains the list of dependencies)
    - test_random_atk_experiment.py
    - test_aaa_atk_experiment.py
    - test_random_atk_experiment_changed_dynamics.py
    - test_aaa_atk_experiment_changed_dynamics.python
    - test_ablation_random_atk.py
    - test_ablation_aaa_atk.py
    - deadlock_detection_aaa.py
    - render_episode_with_random_atk.py
    - render_episode_with_aaa_atk.py
    - test_safety_threshold_sensitivity.py
    - train_adv.py
    - train_victim.py
    - train_safety_policy.py
- TASRL_SafetyGym_env_code
    - TASRL_safetygym              (Contains all the code files relevant to TAS-RL)
          - safety_config.py
          - buffer_memory.py
          - network.py
          - safety_policy.py
          - safety_trainer.py
          - trpo_eval_agent.py
    - Trained_Models                 (Contains trained network parameters for Maze, Navigation 1 and Navigation 2)
        - CPO_models                 (Contains Trained baseline CPO model's network parameters)
        - TRPO                       (Contains Trained baseline TRPO model's network parameters)
        - SAC                        (Contains Trained SAC adversary policy network parameters)
        - TASRL                      (Contains Trained TAS-RL network parameters)
    - cpo_torch                      (Contains CPO implementation from https://github.com/dobro12/CPO/tree/master/torch)
    - pytorch_trpo                   (Contains TRPO implementation from https://github.com/ikostrikov/pytorch-trpo )
    - plot_scripts                   (Contains python file relevant to plotting the experimental results)
    - sac_agent                      (Contains SAC implementation for Adversary from https://github.com/pranz24/pytorch-soft-actor-critic)
    - safety_gym_file_replace        (Contains files related to SafetyGym Environment for TAS-RL safety policy training)
    - test_aaa.py
    - test_random.py
    - test_sensitivity.py
    - train_safety_policy.py

######################################################################
                Requirements
######################################################################
1. Handware and language compiler
- GPU 2 GB (In order to use CUDA)
- RAM 4 GB 
- Python 3.9.7

2. Software/Library requirments
    - Install MuJoCo 
	    - Create a virtual environment using:
            		python3 -m venv ./venv
    	    - Activate the virtual environment:
            		source venv/bin/activate
            - Get the mujoco200 ROM from http://www.roboti.us/
            - Specify the path to mujoco200 bin folder [ROM-bin-Path]
                export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:[ROM-bin-Path]
            - Install mujoco 2.0.2.13
                pip install mujoco-py 2.0.2.13

    - Go inside the folder TASRL_MuJoCo_code    
        - Install other dependency using the following command
                    pip install -r requirements.txt
        - Download the trained models' for TAS-RL and Task policy from "https://drive.google.com/drive/folders/1Lgl6I_RNGby-HjdAAcBWuUtTBQh0roTR?usp=sharing"
            - Extract the zip folder inside TASRL_MuJoCo_env_code (Path should be TASRL_MuJoCo_env_code/Trained_Models)

        - Download the trained models' for the baselines from "https://drive.google.com/drive/folders/1sKF7Y7TWA4qrNQ4v2pic5Cm4GDOevqfp?usp=sharing"
            - Extract the zip folder inside TASRL_MuJoCo_env_code/RecoveryRL (Path should be TASRL_MuJoCo_env_code/RecoveryRL/RecoveryRL_Model)
    
    - Go inside the folder TASRL_SafetyGym_env_code
        - Install other dependency using the following command
                pip install -r requirements.txt

        - Install safety-gym from https://github.com/openai/safety-gym
                - After installation, inside safety-gym folder go to "safety-gym/safety_gym/envs"
                - Replace the "engine.py" and "world.py" file with the files given inside "/TASRL_SafetyGym_env_code/safety_gym_file_replace"
                  (This is done to enable TAS-RL safety policy training in a demo rollout mechanism)

                install safety-gym: pip install -e .
        - Download the trained models' from "https://drive.google.com/drive/folders/1qxg5JaVaeYiOgjRM0iZXyLXpo6T5VcDp?usp=sharing"
            - Extract the downloaded zip folder inside TASRL_SafetyGym_env_code (Path should be TASRL_SafetyGym_env_code/Trained_models)
######################################################################
                  Experiments
######################################################################
1. MUJOCO ENVIRONMENT FROM https://github.com/abalakrishna123/recovery-rl
######################################################################
---------------------------------------------------------------------
            Load pretrained models to reproduce the results 
---------------------------------------------------------------------
Information about the command line arguments:

# Environment specific model configuration can be initialized by passing 
the name of the environment to the argument,

--configure-env [maze] or [nav1] or [nav2]

# Experimental result's save directory can be specified using the argument,
--exp-data-dir [directory]       


Use the following commands to run corresponding comparative experiments 
with pretrained parameters given in folder "Trained_Models" 
and "RecoveryRL_Model" (Contains trained parameters of the baselines) 

******************************************************************************************************************************
Experiment 1: Comparative test under random action perturbation with same environment dynamics as testing environment
******************************************************************************************************************************
    (1.a) To conduct experiment on maze environment run the following command:
          python test_random_atk_experiment.py --configure-env maze --exp-data-dir '/Experimental_Data/' 
    
    (1.b) To conduct experiment on Navigation 1 environment run the following command:
          python test_random_atk_experiment.py --configure-env nav1 --exp-data-dir '/Experimental_Data/' 
    
    (1.c) To conduct experiment on Navigation 2 environment run the following command:
          python test_random_atk_experiment.py --configure-env nav2  --exp-data-dir '/Experimental_Data/' 
******************************************************************************************************************************

******************************************************************************************************************************
Experiment 2: Comparative test under AAA perturbation with same environment dynamics as testing environment
******************************************************************************************************************************
    (2.a) To conduct experiment on maze environment run the following command:
          python test_aaa_atk_experiment.py --configure-env maze --exp-data-dir '/Experimental_Data/' 
    
    (2.b) To conduct experiment on Navigation 1 environment run the following command:
          python test_aaa_atk_experiment.py --configure-env nav1 --exp-data-dir '/Experimental_Data/' 
    
    (2.c) To conduct experiment on Navigation 2 environment run the following command:
          python test_aaa_atk_experiment.py --configure-env nav2 --exp-data-dir '/Experimental_Data/' 
******************************************************************************************************************************

******************************************************************************************************************************
Experiment 3: Comparative test under random action perturbation with change in environment dynamics
******************************************************************************************************************************
    (3.1.a) To conduct experiment on Navigation 1 by changing the training environment dynamics 5 times run the following command:
            python test_random_atk_experiment_changed_dynamics.py --configure-env nav1 --env-change 5.0 --exp-data-dir '/Experimental_Data/'
    
    (3.1.b) To conduct experiment on Navigation 1 by changing the training environment dynamics 10 times run the following command:
            python test_random_atk_experiment_changed_dynamics.py --configure-env nav1 --env-change 10.0 --exp-data-dir '/Experimental_Data/'
    
    (3.1.c) To conduct experiment on Navigation 1 by changing the training environment dynamics 15 times run the following command:
            python test_random_atk_experiment_changed_dynamics.py --configure-env nav1 --env-change 15.0 --exp-data-dir '/Experimental_Data/'
    
    (3.1.d) To conduct experiment on Navigation 1 by changing the training environment dynamics 20 times run the following command:
            python test_random_atk_experiment_changed_dynamics.py --configure-env nav1 --env-change 20.0 --exp-data-dir '/Experimental_Data/'
    
    -----------------------------------------------------------------------------------------------------------------------------------
    (3.2.a) To conduct experiment on Navigation 2 by changing the training environment dynamics 5 times run the following command:
            python test_random_atk_experiment_changed_dynamics.py --configure-env nav2 --env-change 5.0 --exp-data-dir '/Experimental_Data/'
    
    (3.2.b) To conduct experiment on Navigation 1 by changing the training environment dynamics 10 times run the following command:
            python test_random_atk_experiment_changed_dynamics.py --configure-env nav1 --env-change 10.0 --exp-data-dir '/Experimental_Data/'
    
    (3.2.c) To conduct experiment on Navigation 1 by changing the training environment dynamics 15 times run the following command:
            python test_random_atk_experiment_changed_dynamics.py --configure-env nav1 --env-change 15.0 --exp-data-dir '/Experimental_Data/'
    
    (3.2.d) To conduct experiment on Navigation 1 by changing the training environment dynamics 20 times run the following command:
            python test_random_atk_experiment_changed_dynamics.py --configure-env nav1 --env-change 20.0 --exp-data-dir '/Experimental_Data/'
******************************************************************************************************************************

******************************************************************************************************************************
Experiment 4: Comparative test under AAA perturbation with change in environment dynamics
******************************************************************************************************************************
    (4.1.a) To conduct experiment on Navigation 1 by changing the training environment dynamics 5 times run the following command:
            python test_aaa_atk_experiment_changed_dynamics.py --configure-env nav1 --env-change 5.0 --exp-data-dir '/Experimental_Data/'
    
    (4.1.b) To conduct experiment on Navigation 1 by changing the training environment dynamics 10 times run the following command:
            python test_aaa_atk_experiment_changed_dynamics.py --configure-env nav1 --env-change 10.0 --exp-data-dir '/Experimental_Data/'
    
    (4.1.c) To conduct experiment on Navigation 1 by changing the training environment dynamics 15 times run the following command:
            python test_aaa_atk_experiment_changed_dynamics.py --configure-env nav1 --env-change 15.0 --exp-data-dir '/Experimental_Data/'
    
    (4.1.d) To conduct experiment on Navigation 1 by changing the training environment dynamics 20 times run the following command:
            python test_aaa_atk_experiment_changed_dynamics.py --configure-env nav1 --env-change 20.0 --exp-data-dir '/Experimental_Data/'
    
    -----------------------------------------------------------------------------------------------------------------------------------
    (4.2.a) To conduct experiment on Navigation 2 by changing the training environment dynamics 5 times run the following command:
            python test_aaa_atk_experiment_changed_dynamics.py --configure-env nav2 --env-change 5.0 --exp-data-dir '/Experimental_Data/'
    
    (4.2.b) To conduct experiment on Navigation 1 by changing the training environment dynamics 10 times run the following command:
            python test_aaa_atk_experiment_changed_dynamics.py --configure-env nav1 --env-change 10.0 --exp-data-dir '/Experimental_Data/'
    
    (4.2.c) To conduct experiment on Navigation 1 by changing the training environment dynamics 15 times run the following command:
            python test_aaa_atk_experiment_changed_dynamics.py --configure-env nav1 --env-change 15.0 --exp-data-dir '/Experimental_Data/'
    
    (4.2.d) To conduct experiment on Navigation 1 by changing the training environment dynamics 20 times run the following command:
            python test_aaa_atk_experiment_changed_dynamics.py --configure-env nav1 --env-change 20.0 --exp-data-dir '/Experimental_Data/'
******************************************************************************************************************************

******************************************************************************************************************************
Experiment 5: Perform Ablation test under random action perturbation 
******************************************************************************************************************************
    (5.a) To conduct ablation experiment on maze environment run the following command:
          python test_ablation_random_atk.py --configure-env maze --exp-data-dir '/Experimental_Data/'

    (5.b) To conduct ablation experiment on Navigation 1 environment run the following command:
          python test_ablation_random_atk.py --configure-env nav1 --exp-data-dir '/Experimental_Data/'

    (5.c) To conduct ablation experiment on Navigation 2 environment run the following command:
          python test_ablation_random_atk.py --configure-env nav2 --exp-data-dir '/Experimental_Data/'

******************************************************************************************************************************

******************************************************************************************************************************
Experiment 6: Perform Ablation test under AAA perturbation 
******************************************************************************************************************************
    (6.a) To conduct ablation experiment on maze environment run the following command:
          python test_ablation_aaa_atk.py --configure-env maze --exp-data-dir '/Experimental_Data/'

    (6.b) To conduct ablation experiment on Navigation 1 environment run the following command:
          python test_ablation_aaa_atk.py --configure-env nav1 --exp-data-dir '/Experimental_Data/'

    (6.c) To conduct ablation experiment on Navigation 2 environment run the following command:
          python test_ablation_aaa_atk.py --configure-env nav2 --exp-data-dir '/Experimental_Data/'
******************************************************************************************************************************

******************************************************************************************************************************
Experiment 7: Deadlock detection Test
******************************************************************************************************************************
    (7.a) To conduct deadlock detection experiment on maze environment run the following command:
          python deadlock_detection_aaa.py --configure-env maze

    (7.b) To conduct deadlock detection experiment on Navigation 1 environment run the following command:
          python deadlock_detection_aaa.py --configure-env nav1
    
    (7.c) To conduct deadlock detection experiment on Navigation 2 environment run the following command:
          python deadlock_detection_aaa.py --configure-env nav2

******************************************************************************************************************************
Experiment 8: Safety Threshold Sensitivity Test
******************************************************************************************************************************
    (8.a) To conduct ablation experiment on maze environment run the following command:
          python test_safety_threshold_sensitivity.py --configure-env maze --exp-data-dir '/Experimental_Data/'

    (8.b) To conduct ablation experiment on Navigation 1 environment run the following command:
          python test_safety_threshold_sensitivity.py --configure-env nav1 --exp-data-dir '/Experimental_Data/'

    (8.c) To conduct ablation experiment on Navigation 2 environment run the following command:
          python test_safety_threshold_sensitivity.py --configure-env nav2 --exp-data-dir '/Experimental_Data/'
******************************************************************************************************************************

******************************************************************************************************************************
Experiment 9: To Render GIF Execution of Agent under random action perturbation
******************************************************************************************************************************
    (9.a) To conduct ablation experiment on maze environment run the following command:
        python render_episode_with_random_atk.py --configure-env maze --exp-data-dir '/Experimental_Data/'

    (9.b) To conduct ablation experiment on Navigation 1 environment run the following command:
          python render_episode_with_random_atk.py --configure-env nav1 --exp-data-dir '/Experimental_Data/'

    (9.c) To conduct ablation experiment on Navigation 2 environment run the following command:
          python render_episode_with_random_atk.py --configure-env nav2 --exp-data-dir '/Experimental_Data/'
******************************************************************************************************************************
******************************************************************************************************************************
---------------------------------------------------------------------
          Train models from Scratch
---------------------------------------------------------------------
---------------------------------------------------------------------
               Training the Baselines
---------------------------------------------------------------------
We trained all the baselines using code from the git repository: "https://github.com/abalakrishna123/recovery-rl"
This repository uses demostration data to train the models, as a result for space limitation we 
are not able to provide the code as supplimentary materials. However after training the 
models using their code we saved the model parameters of the 7 baseline which have been
provided in "/RecoveryRL/RecoveryRL_Model" folder.
---------------------------------------------------------------------

---------------------------------------------------------------------
              TAS-RL Training
---------------------------------------------------------------------
1. To change the model configuration of either the task policy, safety policy or 
adversary go to "TAS_RL_config" folder and change appropriate hyperparameter
manually

Replace [ENV_NAME] with corresponding environment: maze
                                                   nav1
                                                   nav2
(A) To train the task agent run the following command:
        python train_victim.py --configure-env [ENV_NAME]	
    
    A new directory "TAS_RL_Trained_Models_New" will be created 
    in the current root directory and the training models will be saved 
    there inside "Victim" folder

(B) To train the adversary run the following command:
        python train_adv.py	--configure-env [ENV_NAME] 

--------------------------------------------------------------------------------
Before training the TAS-RL Safety_policy, Manually specify the followings in 
the configuration files provided inside the "TAS_RL_config" folder: 
        # Set --saved_model_path inside the victim_config.py to the newly trained 
            victim/task agent's model path
        # Set --saved_model_path inside the adversary_config.py to the newly trained 
            adversary's model path
(C) To train the TAS-RL safety policy run the following command:
        python train_safety_policy.py --configure-env [ENV_NAME]     

    # Before running the experiments (1-9) discussed earlier, using the newly
    trained Safety_policy model parameter, it has to be manually specified in 
    the configuration files provided inside the "TAS_RL_config" by:
    
    Changing the --saved_model_path inside the safety_config.py to the newly 
    trained safety_policy's model model path

######################################################################
######################################################################
2. SAFETYGYM ENVIRONMENT FROM https://github.com/openai/safety-gym
#####################################################################
       
******************************************************************************************************************************
Experiment 11: Comparative test under random action perturbation 
******************************************************************************************************************************
   (11.a) To conduct robustness experiment on CarGoal environment run the following command:
          python test_random.py --env-name "Safexp-CarGoal1-v0" --shield 18.9

   (11.b) To conduct robustness experiment on CarButton environment run the following command:
          python test_random.py --env-name "Safexp-CarButton1-v0" --shield 4.2
 
 ******************************************************************************************************************************
Experiment 12: Comparative test under AAA action perturbation 
******************************************************************************************************************************
   (12.a) To conduct robustness experiment on CarGoal environment run the following command:
          python test_aaa.py --env-name "Safexp-CarGoal1-v0" --shield 18.9

   (12.b) To conduct robustness experiment on CarButton environment run the following command:
          python test_aaa.py --env-name "Safexp-CarButton1-v0" --shield 4.2


