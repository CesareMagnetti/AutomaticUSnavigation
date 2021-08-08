import torch
import concurrent.futures

class PrioritizedDeepQLearning():
    def __init__(self, gamma):
        """Initializes the trainer class
        Params:
        ==========
            gamma (float): discount factor for dqn.
        """
        # batch size and discount factor
        self.gamma = gamma

    def step(self, batch, local_model, target_model, criterion):
        """Update qbetwork parameters using given batch of experience tuples.
        Params
        ======
            batch (dict): contains all training inputs
            local_model (PyTorch model): local Q network (the one we update)
            target_model (PyTorch model): target Q network (the one we use to bootstrap future Q values)
            criterion (torch.nn.Module): the loss we use to update the network parameters
        Returns
        ======
            loss (torch.tensor): final TD error loss tensor to compute gradients from.
        """
        # 1. split batch
        states, actions, rewards, next_states, dones, weights = batch["states"], batch["actions"], batch["rewards"], batch["next_states"], batch["dones"], batch["weights"]
        # 2. get our value estimates Q for the current state, and our target values estimates (Qtarget) for the next state
        # note that the qnetworks have multiple heads, returned as a list, we will need to process each head separately,
        # and aggregate the individual losses
        Q = local_model(states)
        Qtarget = target_model(next_states)
        # 3. evaluate the loss (TDerror) for each head, launch in parallel for efficiency, aggregate each loss by summation
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.single_head_loss, q, q_target, act, r, dones, criterion, weights) for q, q_target, act, r in zip(Q, Qtarget, actions, rewards)]
        # 4. aggregate the deltas of each head and update buffer priorities
        deltas = sum([f.result()[1] for f in futures])
        # 5. aggregate the losses of each head and average across batch
        loss = sum([f.result()[0] for f in futures])
        return loss, deltas

    def single_head_loss(self, Q, Qtarget, actions, rewards, dones, criterion, weights):
        # 1. gather the values of the action taken
        Qa = Q.gather(1, actions).squeeze()
        # 2. get the target value of the greedy action at the next state
        MaxQ = Qtarget.max(1)[0].detach()
        # 3. backup the expected value of this action by bootstrapping on the greedy value of the next state
        Qhat = rewards.squeeze() + (1-dones.int())*self.gamma*MaxQ
        # 4. evalauate TD error as a fit function for the netwrok
        loss = criterion(Qa, Qhat)*weights
        # 5. deltas to update priorities
        deltas = torch.abs(Qa-Qhat)+1e-5
        return (loss.mean(), deltas)

class PrioritizedDoubleDeepQLearning():
    def __init__(self, gamma):
        """Initializes the trainer class
        Params:
        ==========
            gamma (float): discount factor for dqn.
        """
        # batch size and discount factor
        self.gamma = gamma
    
    def step(self, batch, local_model, target_model, criterion):
        """Update qbetwork parameters using given batch of experience tuples.
        Params
        ======
            batch (Tuple[torch.Tensor]): tuple of (s, a, r, s') tuples
            local_model (PyTorch model): local Q network (the one we update)
            target_model (PyTorch model): target Q network (the one we use to bootstrap future Q values)
            criterion (torch.nn.Module): the loss we use to update the network parameters
        Returns
        ======
            loss (torch.tensor): final TD error loss tensor to compute gradients from. 
        """
        # 1. split batch
        states, actions, rewards, next_states, dones, weights = batch["states"], batch["actions"], batch["rewards"], batch["next_states"], batch["dones"], batch["weights"]
        # 2. get our value estimates Q for the current state, and our target values estimates (Qtarget) for the next state
        # note that the qnetworks have multiple heads, returned as a list, we will need to process each head separately,
        # and aggregate the individual losses
        Q = local_model(states)
        Q_next = local_model(next_states)
        Qtarget_next = target_model(next_states)
        # 3. evaluate the loss (TDerror) for each head, launch in parallel for efficiency, aggregate each loss by summation.
        # each head will have its Q values, its actions and its rewards. Each head will share the same optimizing criterion.
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.single_head_loss, q, q_next, q_target_next, act, r, dones, criterion, weights) for q, q_next, q_target_next, act, r in zip(Q, Q_next, Qtarget_next, actions, rewards)]
        # 4. aggregate the deltas of each head and update buffer priorities
        deltas = sum([f.result()[1] for f in futures])
        # 5. aggregate the losses of each head and average across batch
        loss = sum([f.result()[0] for f in futures])
        return loss, deltas

    def single_head_loss(self, Q, Q_next, Qtarget_next, actions, rewards, dones, criterion, weights):
        # 1. gather Q values for the actions taken at the current state
        Qa = Q.gather(1, actions).squeeze()
        # 2. get the discrete action that maximizes the target value at the next state
        a_next = Q_next.max(1)[1].unsqueeze(-1)
        Qa_next = Qtarget_next.gather(1, a_next).detach().squeeze()
        # 3. backup the expected value of this action by bootstrapping on the greedy value of the next state
        Qhat = rewards.squeeze() + (1-dones.int())*self.gamma*Qa_next
        # 4. evalauate TD error as a fit function for the network
        loss = criterion(Qa, Qhat)*weights
        # 5. deltas to update priorities
        deltas = torch.abs(Qa-Qhat)+1e-5
        return (loss.mean(), deltas)