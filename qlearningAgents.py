# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

#Ryan Brooks
#u1115093
#final version, slight code/comment cleanup, solved extra credit q4

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.Q = util.Counter() #storage for Q-values

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        #if there is no entry for state (we haven't seen it) return 0
        if(None == self.Q[(state,action)]):
            return 0.0
        #otherwise return Q(state,action)
        return self.Q[(state,action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        #find maximum Q-Value among all legal actions (which is by definition the value of a state)
        qMax = float('-inf')
        actions = self.getLegalActions(state)
        #iterate over all legal actions, looking for maximum QVal
        for action in actions:
            qMax = max(qMax,self.getQValue(state,action))
        #ensure that there is a legal action, no legal action => terminal state => return value of 0.0
        if(len(actions) == 0):
            qMax = 0.0
        return qMax #maxQ-Val == value of a state

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        bestActions = [] #set of optimal actions
        qMax = self.computeValueFromQValues(state) #qMax == value by definition
        #look over all legal actions checking for optimal actions
        for action in self.getLegalActions(state):
            if(self.getQValue(state,action) == qMax): #if (q == qMax) => action is optimal
                bestActions.append(action)
        #return a random, optimal action
        return random.choice(bestActions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
            """
        # Pick Action
        legalActions = self.getLegalActions(state)
        #with probability epsilon, pick a random action from the legal actions
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        #otherwise pick the optimal action chosen from assesing Q values
        return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        #Update our old estimate of q using the formula:
        #Q(s,a) = (1-alpha) * Q(s,a) + alpha * (R(s,a,s')+ discount * (max_a'(Q(s',a'))))
        self.Q[(state,action)] = ((1 - self.alpha) * self.getQValue(state,action)) + ((self.alpha) * (reward + self.discount * self.computeValueFromQValues(nextState)))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        q = 0.0 #Q-value retval
        featV = self.featExtractor.getFeatures(state,action)
        #sum over all (f_i * w_i) to perform dot product: Q(s,a) = w (dot) features
        for feat in featV:
            q += featV[feat] * self.weights[feat]
        return q

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        featV = self.featExtractor.getFeatures(state,action)
        #calculate the difference using formula: diff = [r + discount * max_a'(Q(s',a'))] - Q(s,a)
        diff = reward + (self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state,action)
        #update weight for each feature by an amount relative to this difference and alpha
        for feat in featV:
            #w_i = w_i + diff * alpha * f_i
            self.weights[feat] += diff * (self.alpha * featV[feat])

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            pass
