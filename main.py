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
    q = TransformerQnet(d_model=8, nhead=2, num_layers=2, num_seq=window_size)
    q_target = TransformerQnet(d_model=8, nhead=2, num_layers=2, num_seq=window_size)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer(buffer_limit=buffer_limit)

    optimzer = optim.Adam(q.parameters(), lr=learning_rate)
    score = 0.0

    terminate = False
    n_epi = -1
    while not terminate:
        n_epi += 1
        print(f"Episode: {n_epi}")
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200))
        s = env.reset()
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(np.expand_dims(s, axis=0)).float(), epsilon)
            s_prime, r, done, terminate = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r/100.0,s_prime,done_mask))
            s = s_prime
            score += r

            if done:
                break

        if memory.size()>5000:
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
    buffer_limit = 30000
    batch_size = 32
    print_interval = 100
    window_size = 24


    # load dataset
    df = gym_anton.datasets.BTCUSDT_10M.copy()
    main()