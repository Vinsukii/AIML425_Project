import copy
import json
import os
import random
import time as time

import gym
import pandas as pd
import torch
import numpy as np

import pynvml
import PPO_model
from env.load_data import nums_detec

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
    # pynvml.nvmlInit()
    # handle = pynvml.nvmlDeviceGetHandleByIndex(0)

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
    test_paras = load_dict["test_paras"]
    env_paras["device"] = device
    model_paras["device"] = device
    env_test_paras = copy.deepcopy(env_paras)
    num_ins = test_paras["num_ins"]
    if test_paras["sample"]:
        env_test_paras["batch_size"] = test_paras["num_sample"]
    else:
        env_test_paras["batch_size"] = 1
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]

    print("-",test_paras["data_path"],"-")

    data_path = "./data_test/{0}/".format(test_paras["data_path"])
    test_files = os.listdir(data_path)
    test_files.sort(key=lambda x: x[:-4])
    test_files = test_files[:num_ins]
    mod_files = os.listdir('./model/')[:]

    memories = PPO_model.Memory()
    model = PPO_model.PPO(model_paras, train_paras)
    rules = test_paras["rules"]
    envs = []  # Store multiple environments

    # Detect and add models to "rules"
    if "DRL" in rules:
        for root, ds, fs in os.walk('./model/'):
            for f in fs:
                if f.endswith('.pt'):
                    rules.append(f)
    if len(rules) != 1:
        if "DRL" in rules:
            rules.remove("DRL")

    # Generate data files and fill in the header
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    if test_paras["sample"]:
        save_path = './save/test_{0}-S'.format(str_time[:-2])
    else:
        save_path = './save/test_{0}-G'.format(str_time[:-2])
    os.makedirs(save_path)


    # Rule-by-rule (model-by-model) testing
    start = time.time()
    for i_rules in range(len(rules)):
        rule = rules[i_rules]
        # Load trained model
        if rule.endswith('.pt'):
            if device.type == 'cuda':
                model_CKPT = torch.load('./model/' + mod_files[i_rules])
            else:
                model_CKPT = torch.load('./model/' + mod_files[i_rules], map_location='cpu')
            print('\nloading checkpoint:', mod_files[i_rules])
            model.policy.load_state_dict(model_CKPT)
            model.policy_old.load_state_dict(model_CKPT)
        print('rule:', rule)

        # Schedule instance by instance
        step_time_last = time.time()
        makespans = []
        times = []
        for i_ins in range(num_ins):
            test_file = data_path + test_files[i_ins]
            with open(test_file) as file_object:
                line = file_object.readlines()
                ins_num_jobs, ins_num_mas, _ = nums_detec(line)
            env_test_paras["num_jobs"] = ins_num_jobs
            env_test_paras["num_mas"] = ins_num_mas

            # Environment object already exists
            if len(envs) == num_ins:
                env = envs[i_ins]
            # Create environment object
            else:
                # Clear the existing environment
                # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                # if meminfo.used / meminfo.total > 0.7:
                #     envs.clear()
                # DRL-S, each env contains multiple (=num_sample) copies of one instance
                if test_paras["sample"]:
                    env = gym.make('fjsp-v0', case=[test_file] * test_paras["num_sample"],
                                   env_paras=env_test_paras, data_source='file')
                # DRL-G, each env contains one instance
                else:
                    env = gym.make('fjsp-v0', case=[test_file], env_paras=env_test_paras, data_source='file')
                envs.append(copy.deepcopy(env))
                # print("Create env[{0}]".format(i_ins))

            # Schedule an instance/environment
            # DRL-S
            if test_paras["sample"]:
                makespan, time_re = schedule(env, model, memories, flag_sample=test_paras["sample"])
                makespans.append(torch.min(makespan))
                times.append(time_re)
            # DRL-G
            else:
                time_s = []
                makespan_s = []  # In fact, the results obtained by DRL-G do not change
                for j in range(test_paras["num_average"]):
                    makespan, time_re = schedule(env, model, memories)
                    makespan_s.append(makespan)
                    time_s.append(time_re)
                    env.reset()
                makespans.append(torch.mean(torch.tensor(makespan_s)))
                times.append(torch.mean(torch.tensor(time_s)))
            print("Finish env[{0}]".format(i_ins))
        print("\nrule_spend_time: {:.3f}s".format(time.time()-step_time_last))

        # Save makespan and time data to files
        file_name = [test_files[i] for i in range(num_ins)]

        data_makespan = pd.DataFrame([file_name, torch.tensor(makespans).t().tolist()]).T
        data_makespan.to_csv('{0}/test_makespan_{1}.csv'.format(save_path, rule[5:-3]),
                             header=['File', 'Makespan'], index=False)

        data_time = pd.DataFrame([torch.tensor(makespans).t().tolist(), torch.tensor(times).t().tolist()]).T
        data_time.to_csv('{0}/test_time_{1}.csv'.format(save_path, rule[5:-3]),
                         header=['Makespan', 'Time'], index=False)

        for env in envs:
            env.reset()

    print("total_spend_time: {:.3f}s\n".format(time.time()-start))

def schedule(env, model, memories, flag_sample=False):
    # Get state and completion signal
    state = env.state
    dones = env.done_batch
    done = False  # Unfinished at the beginning
    last_time = time.time()
    i = 0
    while ~done:
        i += 1
        with torch.no_grad():
            actions = model.policy_old.act(state, memories, dones, flag_sample=flag_sample, flag_train=False)
        state, rewards, dones = env.step(actions)  # environment transit
        done = dones.all()
    spend_time = time.time() - last_time  # The time taken to solve this environment (instance)
    # print("spend_time: ", spend_time)

    # Verify the solution
    gantt_result = env.validate_gantt()[0]
    if not gantt_result:
        print("Scheduling Error！！！！！！")
    return copy.deepcopy(env.makespan_batch), spend_time


if __name__ == '__main__':
    main()
