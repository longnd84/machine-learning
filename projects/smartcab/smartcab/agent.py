import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

from collections import OrderedDict

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    DEFAULT_Q_VALUE = 5.0
    
    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        self.total_reward = 0;
        
        self.trip_reward = 0

        self.reset()
        
        self.state = OrderedDict()
        self.q_table = OrderedDict()
        self.learning_rate = 1.0 # begin as an "eager" learner, will decay overtime
        self.exploitation_factor = 0.0 #beginner has nothing to exploit
        
    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.ellapsed_time = 0

        self.total_reward += self.trip_reward
        self.trip_reward = 0
        

        print "Trip reward ", self.trip_reward, " total reward ", self.total_reward
    def update_Q_learning_table(self, inputs, state, action, state_after_action, reward):
        learning_decay_rate = 0.002 
        min_learning_rate = 0.01
        discount_factor = 0.1 # strive more for "short-term" reward, since our grid is pretty simpleterm

        #the more you now, the less you tend to learn (learning decay)
        self.learning_rate = max(min_learning_rate, self.learning_rate - learning_decay_rate)

        q_key = (state[0], state[1], action)
        if self.q_table.has_key(q_key) == False:
            self.q_table[q_key] = LearningAgent.DEFAULT_Q_VALUE

        max_next_state = 0
        for next_action in Environment.valid_actions:
            q_key_next_state = (state_after_action[0], state_after_action[1], next_action)
            if self.q_table.has_key(q_key_next_state) == False:
                self.q_table[q_key_next_state] = LearningAgent.DEFAULT_Q_VALUE
            max_next_state = max(max_next_state, self.q_table[q_key_next_state])

        self.q_table[q_key] = (1 - self.learning_rate) * self.q_table[q_key] + self.learning_rate * (reward + discount_factor * max_next_state)

        #print "Update Q table reward {} key {} value {} learning rate {} ".format(reward, q_key, self.q_table[q_key], self.learning_rate)

    def plan_next_way_point(self, inputs, state, useShortestPath):
        next_waypoint = Environment.valid_actions[0]            
        if useShortestPath:
            next_waypoint = self.planner.next_waypoint() #"shortest path in a grid" strategy
        else: #use Q-learning
            """ balance exploration and exploitation"""
            exploitation_rate = 0.005
            self.exploitation_factor = min(0.99, self.exploitation_factor + exploitation_rate)
            max_exploration_percentage = 5

            #in the beginning (exploitation_rate low) we always explore, when we learn more, we give some room to explore
            if random.randint(0, int(100 * self.exploitation_factor)) < max_exploration_percentage: 
                next_waypoint = random.choice(Environment.valid_actions) #random strategy
                #print "EXPLORATION ", self.exploitation_factor * 100
            else:
                #print "EXPLOITATION"
                maxQ = 0 
                q_max_found = False
                for action in Environment.valid_actions:
                    q_key = (state['gps'], state['light'], action)
                    if self.q_table.has_key(q_key) == False: #default value is not exist
                        self.q_table[q_key] = LearningAgent.DEFAULT_Q_VALUE

                    if self.q_table[q_key] > maxQ:
                        maxQ = self.q_table[q_key]
                        q_max_found = True
                        next_waypoint = action #pick the action from the highest Q[state, action]

                if q_max_found == False:
                    #print "Q value not found, chose to EXPLORE"
                    next_waypoint = random.choice(Environment.valid_actions)
                #print "Found max value ", maxQ
        return next_waypoint

    """
        Extract the interesting state for our agent from the environment
        
    """
    def extract_state(self, inputs):

        state = OrderedDict()

        #state['deadline'] = self.env.get_deadline(self) #visual inspection
        
        state['gps'] = self.next_waypoint
        state['light'] = inputs['light']
        
        #TODO: maybe on coming traffic, current deadline as well? 

        state['reward'] = self.trip_reward #for convinient display
        #state['destination'] = agent_state['destination'] #store only for convinient retrieval
        return state

    def update(self, t):
        self.ellapsed_time += 1
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator

        inputs = self.env.sense(self)

        deadline = self.env.get_deadline(self)
        
        self.state = self.extract_state(inputs)            
        self.state['dead'] = deadline #visual inspection
        
        action = self.plan_next_way_point(inputs, self.state, False);

        #if LearningAgent.is_action_ok(self.next_waypoint, inputs):
        #          action = self.next_waypoint

        reward = self.env.act(self, action) #after this method agent_state(self) will be updated with action (e.g. new location)
        self.trip_reward += reward

        inputs_after_action = self.env.sense(self)
        
        self.update_Q_learning_table(inputs, (self.state['gps'], self.state['light']), action, (self.planner.next_waypoint(), inputs_after_action['light']), reward)

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
            

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent

    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.01, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
