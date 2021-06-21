import torch

class DeepQLearning():
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
        """
        # 1. split batch
        states, actions, rewards, next_states = batch
        #print("training batch")
        #print(states.shape, actions.shape, rewards.shape, next_states.shape)

        # 2. get our value estimates Q for the current state, and our target values estimates (Qtarget) for the next state
        # note that the qnetworks have multiple heads, returned as a list, concatenate them
        # so that the shape is (n_agents*batch_size) X action_size
        Q = torch.cat(local_model(states), dim=0) 
        Qtarget = torch.cat(target_model(next_states), dim=0)
        #print("Q and MaxQ:", Q.shape, Qtarget.shape)
        # 3. gather Q values for the actions taken at the current state
        Q = Q.gather(1, actions).squeeze()
        #print("gathered Q values: ", Q.shape)
        # 4. get the target value of the greedy action at the next state
        MaxQ = Qtarget.max(1)[0].detach()
        #print("greedy target values: ", MaxQ.shape)
        # 5. backup the expected value of this action by bootstrapping on the greedy value of the next state
        Qhat = rewards + self.gamma*MaxQ
        #print("rewards + self.gamma*MaxQ: ", rewards.shape, MaxQ.shape, Qhat.shape)
        # evalauate TD error as a fit function for the netwrok
        loss = criterion(Q, Qhat)
        #print("loss: ", loss.shape)
        return loss