class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.agent_states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.dones = []

    def push(
        self,
        state,
        agent_state,
        action,
        action_logprob,
        state_val,
        reward,
        done,
    ) -> None:
        # Ensure the inputs are proper data types before appending to the buffer.
        self.states.append(state)
        self.agent_states.append(agent_state)
        self.actions.append(action)
        self.logprobs.append(action_logprob)
        self.state_values.append(state_val)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.agent_states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.dones[:]
