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
        
    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.ellapsed_time = 0

        self.state = OrderedDict()
        self.q_table = OrderedDict()
        self.learning_rate = 1.0 # begin as an "eager" learner, will decay overtime
        self.ellapsed_time = 0 #reverse of deadline, included for convinient visual inspection
        

        self.exploitation_factor = 0.0 #beginner has nothing to exploit
        
        # TODO: Prepare for a new trip; reset any variables here, if required
        # reset is already done by Environment's reset?

    
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

    def update_Q_learning_table(self, inputs, state, reward, action):
        learning_decay_rate = 0.01
        min_learning_rate = 0.01
        discount_rate = 0.8 #depend on deadline?
        
        self.learning_rate = max(min_learning_rate, self.learning_rate - learning_decay_rate)

        agent_state = self.env.agent_states[self]


        location_before_action = state['location']
        location_after_action = agent_state['location']

        heading_before_action = state['heading']
        heading_after_action = agent_state['heading']
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
            
        self.q_table[q_key] = (1 - self.learning_rate) * self.q_table[q_key] + (reward + discount_rate * max_next_state)

        print "Update Q table {} value {} ".format(q_key, self.q_table[q_key])

    def plan_next_way_point(self, inputs, state, useShortestPath):
        if useShortestPath:
            self.next_waypoint = self.planner.next_waypoint() #"shortest path in a grid" strategy
        else: #use Q-learning
            """ balance exploration and exploitation"""
            """TODO: create a method for exploitation update"""
            exploitation_rate = 0.005
            self.exploitation_factor = min(0.99, self.exploitation_factor + exploitation_rate)
            
            
            if random.randint(0, int(self.exploitation_factor * 100)) <5:
                self.next_waypoint = random.choice(Environment.valid_actions[1:]) #random strategy
                print "EXPLORATION ", self.exploitation_factor * 100
            else:
                maxQ = -1 #guarantee always at least one action chosen
                for action in Environment.valid_actions[1:]:
                    q_key = (state['distance'][0], state['distance'][1], state['heading'][0], state['heading'][1], action)
                    if self.q_table.has_key(q_key) == False:
                        self.q_table[q_key] = 0.0

                    if self.q_table[q_key] > maxQ:
                        maxQ = self.q_table[q_key]
            

    def update_state(self, inputs):
        #self.state = self.env.agent_states[self]
        #self.state['time'] = self.ellapsed_time #for the sake of easy visual monitoring
        #self.state['light'] = inputs['light']
        agent_state = self.env.agent_states[self]

        location = agent_state['location']
        destination = agent_state['destination']

        self.state['dead'] = self.env.get_deadline(self) #visual inspection
        
        self.state['location'] = agent_state['location']
        self.state['distance'] = (destination[0] - location[0], destination[1] - location[1])
        self.state['heading'] = agent_state['heading']

    def update(self, t):
        self.ellapsed_time += 1
        inputs = self.env.sense(self)

        deadline = self.env.get_deadline(self)
        self.update_state(inputs)            
            

        self.plan_next_way_point(inputs, self.state, False);


        action = None
        
        if LearningAgent.is_action_ok(self.next_waypoint, inputs):
            action = self.next_waypoint
  
        reward = self.env.act(self, action)

        #TODO: feed in state_before_action and state_after_action objects
        self.update_Q_learning_table(inputs, self.state, reward, action)

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
            
        
"""    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        
        # TODO: Select action according to your policy
        action = None

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
"""


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    #TODO: put back enforce_deadline to True
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=1.0, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
