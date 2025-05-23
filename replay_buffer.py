import numpy as np

class ReplayBuffer():
    def __init__(self,max_size,input_shape,n_action):
        self.mem_size = max_size
        self.mem_contr = 0
        self.state_memory = np.zeros((self.mem_size,*input_shape))
        self.new_state_memory = np.zeros((self.mem_size,*input_shape))
        self.action_memory = np.zeros((self.mem_size,n_action))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size,dtype=np.bool)

    def store_transition(self, state, action, reward, state_,done):
        index = self.mem_contr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_contr += 1


    def sample_buffer(self, batch_size):
        max_mem = self.mem_contr if self.mem_contr < self.mem_size else self.mem_size
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, states_, dones   