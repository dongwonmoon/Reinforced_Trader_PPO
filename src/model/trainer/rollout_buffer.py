class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.agent_states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.dones = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.agent_states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.dones[:]
