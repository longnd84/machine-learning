import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

from collections import OrderedDict

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        self.reset()
        
        self.state = OrderedDict()
        self.q_table = OrderedDict()
        self.learning_rate = 1.0 # begin as an "eager" learner, will decay overtime
        self.exploitation_factor = 0.0 #beginner has nothing to exploit
        
    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.ellapsed_time = 0

    
    @staticmethod
    def is_action_ok(next_waypoint, inputs):
        action_okay = True
        if next_waypoint == 'right':
            if inputs['light'] == 'red' and inputs['left'] == 'forward':
                action_okay = False
        elif next_waypoint == 'forward':
            if inputs['light'] == 'red':
                action_okay = False
        elif next_waypoint == 'left':
            if inputs['light'] == 'red' or (inputs['oncoming'] == 'forward' or inputs['oncoming'] == 'right'):
                action_okay = False

        return action_okay

    def update_Q_learning_table(self, inputs, state, action, state_after_action):
        learning_decay_rate = 0.002 
        min_learning_rate = 0.01
        discount_factor = 0.1 # strive more for "short-term" reward, since our grid is pretty simpleterm

        #the more you now, the less you tend to learn (learning decay)
        self.learning_rate = max(min_learning_rate, self.learning_rate - learning_decay_rate)

        agent_state = self.env.agent_states[self]

        location_before_action = state['location']
        location_after_action = state_after_action['location']

        heading_before_action = state['heading']
        heading_after_action = state_after_action['heading']

        bounds = self.env.bounds
        destination = agent_state['destination']
                
        q_key = (state['distance'][0], state['distance'][1], heading_before_action[0], heading_before_action[1], action)
        if self.q_table.has_key(q_key) == False:
            self.q_table[q_key] = 0.0

        distance_after_action = (destination[0] - location_after_action[0], destination[1] - location_after_action[1])
        max_next_state = 0
        for next_action in Environment.valid_actions[1:]:
            q_key_next_state = (distance_after_action[0], distance_after_action[1], heading_after_action[0], heading_after_action[1], next_action)
            if self.q_table.has_key(q_key_next_state) == False:
                self.q_table[q_key_next_state] = 0.0
            max_next_state = max(max_next_state, self.q_table[q_key_next_state])

        #action brings closer to the goal, positive immediate reward, further negative, no change 0

        reward = 0
        if self.env.compute_dist(location_before_action, destination) > self.env.compute_dist(location_after_action, destination):
            reward = 2
        else:
            reward = -0.5

        if state_after_action['location'] == state_after_action['destination']:
            reward += 10
            
        #should we only update when the reward is positive, otherwise we can bring the value down?  
        self.q_table[q_key] = (1 - self.learning_rate) * self.q_table[q_key] + (reward + discount_factor * max_next_state)

        print "Update Q table reward {} key {} value {} learning rate {} ".format(reward, q_key, self.q_table[q_key], self.learning_rate)

    def plan_next_way_point(self, inputs, state, useShortestPath):
        if useShortestPath:
            self.next_waypoint = self.planner.next_waypoint() #"shortest path in a grid" strategy
        else: #use Q-learning
            """ balance exploration and exploitation"""
            """TODO: create a method for exploitation update"""
            exploitation_rate = 0.005
            self.exploitation_factor = min(0.99, self.exploitation_factor + exploitation_rate)
            max_exploration_percentage = 5

            #in the beginning (exploitation_rate low) we always explore, when we learn more, we give some room to explore
            if random.randint(0, int(100 * self.exploitation_factor)) < max_exploration_percentage: 
                self.next_waypoint = random.choice(Environment.valid_actions[1:]) #random strategy
                #print "EXPLORATION ", self.exploitation_factor * 100
            else:
                #print "EXPLOITATION"
                maxQ = 0 #guarantee always at least one action chosen
                for action in Environment.valid_actions[1:]:
                    q_key = (state['distance'][0], state['distance'][1], state['heading'][0], state['heading'][1], action)
                    if self.q_table.has_key(q_key) == False:
                        self.q_table[q_key] = 0.0

                    if self.q_table[q_key] > maxQ:
                        maxQ = self.q_table[q_key]
                        self.next_waypoint = action

                if maxQ == 0:
                    #print "Q value not found, chose to EXPLORE"
                    self.next_waypoint = random.choice(Environment.valid_actions[1:])
                #print "Found max value ", maxQ

    """
        Extract the interesting state for our agent from the environment
        
    """
    def extract_state(self, agent_state):

        state = OrderedDict()
        location = agent_state['location']
        destination = agent_state['destination']

        #state['deadline'] = self.env.get_deadline(self) #visual inspection
        
        state['location'] = agent_state['location'] #store only for convinient retrieval
        state['distance'] = (destination[0] - location[0], destination[1] - location[1])
        state['heading'] = agent_state['heading'] 
        state['destination'] = agent_state['destination'] #store only for convinient retrieval
        return state

    def update(self, t):
        self.ellapsed_time += 1
        inputs = self.env.sense(self)

        deadline = self.env.get_deadline(self)
        
        self.state = self.extract_state(self.env.agent_states[self])            
            
        self.plan_next_way_point(inputs, self.state, False);

        action = None
        
        if LearningAgent.is_action_ok(self.next_waypoint, inputs):
            action = self.next_waypoint

        reward = self.env.act(self, action) #after this method agent_state(self) will be updated with action (e.g. new location)

        state_after_action = self.extract_state(self.env.agent_states[self])
        """
        we should not use the standard reward calculated by the environment, since it gives reward for every successful move
        even move that brings the agent further away. We can give the immediate reward when the agent is moved
        closer to the goal"""
        if reward > 0: #only update if we take one step forward
            self.update_Q_learning_table(inputs, self.state, action, state_after_action)

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
            

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent

    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.1, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
