import gym
import numpy as np
import torch
import torch.optim as optim

import gym_anton
from gym_anton.model import TransformerQnet
from gym_anton.utils import ReplayBuffer
from gym_anton.utils import dqn_trainer


def main():
    env = gym.make('spot-v0', df=df, window_size=window_size)
    q = TransformerQnet(d_model=6, nhead=2, num_layers=2, num_seq=window_size)
    q_target = TransformerQnet(d_model=6, nhead=2, num_layers=2, num_seq=window_size)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer(buffer_limit=buffer_limit)

    optimzer = optim.Adam(q.parameters(), lr=learning_rate)
    score = 0.0

    for n_epi in range(1000):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200))
        s = env.reset()
        s = np.expand_dims(s, axis=0)
        done = False
        print(env._end_tick)

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r/100.0,s_prime,done_mask))
            s = s_prime
            score += r

            if done:
                break

        if memory.size()>2000:
            dqn_trainer(q, q_target, memory, optimzer, gamma, batch_size=batch_size)
        
        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())
            print(f"n_episode :{n_epi}, score :{score/print_interval}, n_buffer :{memory.size()}, eps :{epsilon*100}")
            score = 0.0
    
    env.close()
        


    


if __name__ == "__main__":
    # Hyperparameters
    learning_rate = 0.0005
    gamma = 0.98
    buffer_limit = 1000
    batch_size = 32
    print_interval = 20
    window_size = 24


    # load dataset
    df = gym_anton.datasets.TEST_BTCUSDT_5M.copy()
    main()