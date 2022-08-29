import copy
import json
import os
import random
import time
from collections import deque

import gym
import pandas as pd
import torch
import numpy as np
from visdom import Visdom

import PPO_model
from env.case_generator import CaseGenerator
from validate import validate, get_validate_env

import warnings
warnings.filterwarnings('ignore')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    # PyTorch initialization
    # gpu_tracker = MemTracker()  # Used to monitor memory (of gpu)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print("\nPyTorch device:", device.type, "(" + str(torch.cuda.device_count()) + ")\n")
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
        print("\nPyTorch device:", device.type, "(" + str(os.cpu_count()) + ")\n")

    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

    # Load config and init objects
    with open("./config.json", 'r') as load_f:
        load_dict = json.load(load_f)
    env_paras = load_dict["env_paras"]
    model_paras = load_dict["model_paras"]
    train_paras = load_dict["train_paras"]
    env_paras["device"] = device
    model_paras["device"] = device
    env_valid_paras = copy.deepcopy(env_paras)
    env_valid_paras["batch_size"] = env_paras["valid_batch_size"]
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]

    num_jobs = env_paras["num_jobs"]
    num_mas = env_paras["num_mas"]
    opes_per_job_min = int(num_mas * 0.8)
    opes_per_job_max = int(num_mas * 1.2)
    

    print("-",str(num_jobs)+str(num_mas).zfill(2),"-\n")

    memories = PPO_model.Memory()
    model = PPO_model.PPO(model_paras, train_paras, num_envs=env_paras["batch_size"])
    env_valid = get_validate_env(env_valid_paras)  # Create an environment for validation
    maxlen = 1  # Save the best model
    best_models = deque()
    makespan_best = float('inf')

    # Use visdom to visualize the training process
    is_viz = train_paras["viz"]
    if is_viz:
        viz = Visdom(env=train_paras["viz_name"])

    # Generate data files and fill in the header
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    save_path = './save/train_{0}'.format(str_time[:-2])
    os.makedirs(save_path)

    valid_results = []
    valid_results_100 = []


    # Start training iteration
    start_time = time.time()
    env = None
    for i in range(1, train_paras["max_iterations"]+1):

        print("ITER_" + str(i))

        # Replace training instances every x iteration (x = 20 in paper)
        if (i - 1) % train_paras["parallel_iter"] == 0:
            # \mathcal{B} instances use consistent operations to speed up training
            nums_ope = [random.randint(opes_per_job_min, opes_per_job_max) for _ in range(num_jobs)]
            case = CaseGenerator(num_jobs, num_mas, opes_per_job_min, opes_per_job_max, nums_ope=nums_ope)
            env = gym.make('fjsp-v0', case=case, env_paras=env_paras)
            print('num_job: ', num_jobs, '\tnum_mas: ', num_mas, '\tnum_opes: ', sum(nums_ope))

        # Get state and completion signal
        state = env.state
        done = False
        dones = env.done_batch
        last_time = time.time()

        # Schedule in parallel
        while ~done:
            with torch.no_grad():
                actions = model.policy_old.act(state, memories, dones)
            state, rewards, dones = env.step(actions)
            done = dones.all()
            memories.rewards.append(rewards)
            memories.is_terminals.append(dones)
            # gpu_tracker.track()  # Used to monitor memory (of gpu)

        train_time = time.time()-last_time

        # Verify the solution
        gantt_result = env.validate_gantt()[0]
        if not gantt_result:
            print("Scheduling Error！！！！！！")
        # print("Scheduling Finish")

        # env.render(mode='draw')
        env.reset()

        # if iter mod x = 0 then update the policy (x = 1 in paper)
        if i % train_paras["update_timestep"] == 0:
            loss, reward = model.update(memories, env_paras, train_paras)
            print("reward: ", '%.3f' % reward, "; loss: ", '%.3f' % loss)
            print("train_time: {:.3f}s".format(train_time))
            memories.clear_memory()
            if is_viz:
                viz.line(X=np.array([i]), Y=np.array([reward]),
                    win='window{}'.format(0), update='append', opts=dict(title='reward of envs'))
                viz.line(X=np.array([i]), Y=np.array([loss]),
                    win='window{}'.format(1), update='append', opts=dict(title='loss of envs'))  # deprecated

        # if iter mod x = 0 then validate the policy (x = 10 in paper)
        if i % train_paras["save_timestep"] == 0:
            # print('\nStart validating')
            # Record the average results and the results on each instance
            vali_result, vali_result_100 = validate(env_valid_paras, env_valid, model.policy_old)
            valid_results.append(vali_result.item())
            valid_results_100.append(vali_result_100)

            # Save the best model
            if vali_result < makespan_best:
                makespan_best = vali_result
                if len(best_models) == maxlen:
                    delete_file = best_models.popleft()
                    os.remove(delete_file)
                save_file = '{0}/save_best_{1}_{2}_{3}.pt'.format(save_path, num_jobs, num_mas, i)
                best_models.append(save_file)
                torch.save(model.policy.state_dict(), save_file)

            if is_viz:
                viz.line(
                    X=np.array([i]), Y=np.array([vali_result.item()]),
                    win='window{}'.format(2), update='append', opts=dict(title='makespan of valid'))

    # Save the data of training curve to files
    ss = str(num_jobs)+"_"+str(num_mas)

    data_100 = pd.DataFrame([np.arange(10, 1010, 10), np.array(torch.stack(valid_results_100, dim=0).to('cpu'))]).T
    data_100.to_csv('{0}/training_100_{1}.csv'.format(save_path, ss),
                    header=['Iter', 'Makespan'], index=False)

    data_avg = pd.DataFrame([np.arange(10, 1010, 10), valid_results]).T
    data_avg.to_csv('{0}/training_avg_{1}.csv'.format(save_path, ss),
                    header=['Iter', 'Avg Makespan'], index=False)


    print("total_time: {:.3f}s".format(time.time()-start_time))

if __name__ == '__main__':
    main()
