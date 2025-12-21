import argparse

import gc
import os
import csv
from itertools import product

from tqdm import tqdm

from negmas import load_genius_domain_from_folder
from sao.my_sao import MySAOMechanism
from sao.my_negotiators import *
from envs.rl_negotiator import TestRLNegotiator
from matplotlib import pyplot as plt

import sys

ISSUE_NAMES = [
    'Laptop',
    'ItexvsCypress',
    'IS_BT_Acquisition',
    'Grocery',
    'thompson',
    'Car',
    'EnergySmall_A'
]
AGENT_LIST = [
    'Boulware',
    'Linear',
    'Conceder',
    'TitForTat1',
    'TitForTat2',
    "AgentK",
    "HardHeaded",
    "Atlas3",
    "AgentGG",
]
global LOAD_PATH
global PLOT


def a(x):
    return 'T' if x else 'F'


def run_session_trained(path, save_path, opponent, issue, domain, util1, util2, util3, det, noise, decoder_only, scale, decoder_num, is_first_turn=False): # 変更箇所
    session = MySAOMechanism(issues=domain, n_steps=80, avoid_ultimatum=False)
    my_agent = TestRLNegotiator(domain, issue, opponent, path, deterministic=det, accept_offer=False, decoder_only=decoder_only, scale=scale, decoder_num=decoder_num)
    opponent0 = get_opponent(opponent[0], add_noise=noise) # 変更箇所
    opponent1 = get_opponent(opponent[1], add_noise=noise) # 変更箇所

# この部分の対戦相手の順番を調整
    if is_first_turn:
        session.add(my_agent, ufun=util1)
        session.add(opponent0, ufun=util2) # 変更箇所
        session.add(opponent1, ufun=util3) # 変更箇所
    else:
        session.add(opponent0, ufun=util2) # 変更箇所
        session.add(my_agent, ufun=util1)
        session.add(opponent1, ufun=util3) # 変更箇所

    values = []
    for _ in session:
        my_agent.observation = my_agent.observer(session.state.asdict())
        if session.state['last_negotiator'] == 'RLAgent':
            values.append(my_agent.values)
    result = session.state
    print(result)


    if result['agreement'] is not None:
        agreement_offer = tuple(v for k, v in result['agreement'].items())
        # この部分は先行想定ごとに処理を変更しなければならない
        my_util, opp_util1, opp_util2 = util1(agreement_offer), util2(agreement_offer), util3(agreement_offer) # 変更箇所
    else:
        my_util, opp_util1, opp_util2 = 0, 0, 0 # 変更箇所

    # 変更箇所
    results = [
        my_util,
        opp_util1,
        opp_util2,
        my_util + opp_util1 + opp_util2,
        my_util * opp_util1 * opp_util2,
        result['agreement'],
        result['step'], 
        result['last_negotiator'], 
        values[-1],
    ]
    print(results)

    # 結果を描画
    if PLOT:
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(values)),values,linestyle='-')
        ax.set_xlim(0,80)
        ax.set_ylim(0,1.0)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        plt.savefig(save_path + f'values.png')

        my_agent.name = "Our Agent"
        session.plot(path=save_path + '/plot.png')
        plt.clf()
        plt.close()
        with open(save_path + f'values.tsv', 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(values)

    session.reset()
    del my_agent, session, opponent1, opponent2 # 変更箇所
    gc.collect()
    
    # 変更箇所
    return [
        my_util,
        opp_util1,
        opp_util2,
        my_util + opp_util1 + opp_util2,
        my_util * opp_util1 * opp_util2,
        result['agreement'],
        result['step'], 
        result['last_negotiator'], 
    ]


def test_trained(config):
    issue, agent, det, noise, save_path, decoder_only, scale, decoder_num, is_first = config
    results = [['my_util', 'opp_util1', 'opp_util2', 'social', 'nash', 'agreement', 'step', 'last_neg', 'value']] # 変更箇所
    scenario = load_genius_domain_from_folder('domain/' + issue)
    domain = scenario.issues
    util1 = scenario.ufuns[0].scale_max(1.0)
    util2 = scenario.ufuns[1].scale_max(1.0)
    util3 = scenario.ufuns[2].scale_max(1.0) # 変更箇所
    
    for _ in tqdm(range(1 if PLOT else 100)):
        result = run_session_trained(f'{LOAD_PATH}/checkpoint.pt', save_path, agent, issue, domain, util1, util2, util3, det, noise, decoder_only, scale, decoder_num, is_first) # 変更箇所
        results.append(result)

    if not PLOT:
        with open(f'{save_path}{issue}-{agent[0]}-{agent[1]}-d{a(det)}-n{a(noise)}.tsv', 'w') as f: # 変更箇所
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(results)
            print(results)
            print(result)
    else:
        print(results)
        print(result)


def get_opponent(opponent, add_noise=False):
    if opponent == 'Boulware':
        opponent = TimeBasedNegotiator(name='Boulware', aspiration_type=4.0, add_noise=add_noise)
    elif opponent == 'Linear':
        opponent = TimeBasedNegotiator(name='Linear', aspiration_type=1.0, add_noise=add_noise)
    elif opponent == 'Conceder':
        opponent = TimeBasedNegotiator(name='Conceder', aspiration_type=0.2, add_noise=add_noise)
    elif opponent == 'TitForTat1':
        opponent = AverageTitForTatNegotiator(name='TitForTat1', gamma=1, add_noise=add_noise)
    elif opponent == 'TitForTat2':
        opponent = AverageTitForTatNegotiator(name='TitForTat2', gamma=2, add_noise=add_noise)
    elif opponent == 'AgentK':
        opponent = AgentK(add_noise=add_noise)
    elif opponent == 'HardHeaded':
        opponent = HardHeaded(add_noise=add_noise)
    elif opponent == 'Atlas3':
        opponent = Atlas3(add_noise=add_noise)
    elif opponent == 'AgentGG':
        opponent = AgentGG(add_noise=add_noise)
    else:
        opponent = TimeBasedNegotiator(name='Linear', aspiration_type=1.0, add_noise=add_noise)
    return opponent


def main_trained():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents', '-a', nargs='*', required=True, type=str)
    parser.add_argument('--issues', '-i', nargs='*', required=True, type=str)
    parser.add_argument('--model', '-m', required=True, type=str)
    parser.add_argument('--scale', '-s', type=str, default='small')
    parser.add_argument('--decoder_only', '-do', action='store_true')
    parser.add_argument('--plot', '-p', action='store_true')
    parser.add_argument('--is_first', '-if', action='store_true')
    parser.add_argument('--decoder_num', '-dn', type=int, default=1)
    args = parser.parse_args()
    print(args)


    agents = args.agents
    issues = args.issues
    model_path = args.model
    scale = args.scale
    decoder_only = args.decoder_only
    plot = args.plot
    is_first = args.is_first
    decoder_num = args.decoder_num
    

    global LOAD_PATH
    LOAD_PATH = model_path

    global PLOT
    PLOT = plot

    if isinstance(issues, str):
        issues = [issues]
    if isinstance(agents, str):
        agents = [agents]

    agents_sum = len(agents) # 変更箇所
    agent = [None, None] # 変更箇所

# 変更箇所
    for i in range(agents_sum):
        agent[0] = agents[i]
        for j in range(i, agents_sum):
            agent[1] = agents[j]
            for issue in issues:
                for det, noise in product([False], [False]):
                    save_path = LOAD_PATH + ('/img' if PLOT else '/csv') + f'/{agent[0]}-{agent[1]}/' + f'/{issue}/' + f'/det={det}_noise={noise}/'
                    if not os.path.isdir(save_path):
                        os.makedirs(save_path)

                    test_trained((issue, agent, det, noise, save_path, decoder_only, scale, decoder_num, is_first)) 
# --- IGNORE ---
"""
    for agent in agents:
        for issue in issues:
            for det, noise in product([False], [False]):
                save_path = LOAD_PATH + ('/img' if PLOT else '/csv') + f'/{agent}/' + f'/{issue}/' + f'/det={det}_noise={noise}/'
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)

                test_trained((issue, agent, det, noise, save_path, decoder_only, scale, decoder_num, is_first))
"""


if __name__ == '__main__':
    main_trained()







#やるべきタスク
#・先行想定の処理

#コメント
#・ガチの多者間にするには、52~61の対戦相手の区別の仕方をopponent0とかopponent1とかにするのではなく、
# リストでopponent_listのようなものを作成するべきである