# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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

import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        #run for a given number of iterations, updating all values
        for x in range(self.iterations):
            vals = util.Counter()  #stores values during a given iteration (k)
            #look through all states, determining value
            for state in self.mdp.getStates():
                #find the max (Q-value = value) given our set of possible actions, store it in the counter
                qMax = float('-inf')
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    qMax = max(qMax,self.computeQValueFromValues(state,action)) #update qMax if calculated q is new max
                if(len(actions) == 0): # if there were no actions, the state is terminal and has a value of zero
                    qMax = 0.0
                vals[state] = qMax
            #update values for used for calculation during each iteration (new k-1)
            self.values = vals


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        q=0.0 #Q-Value retval
        #calculate Q value by summing over all transition states Q(s,a) = sum(T(s,a,s') * [R(s,a,s') + (discount * V*(s'))])
        for tStateProbPair in self.mdp.getTransitionStatesAndProbs(state,action):
            tState = tStateProbPair[0] # transition state (s')
            tProb =  tStateProbPair[1] # transition function / probability (T(s,a,s'))
            #calculate specific "T(s,a,s') * [R(s,a,s') + (discount * V*(s'))]" and append to q-Value
            q +=  tProb * (self.mdp.getReward(state,action,tState) + (self.discount * self.values[tState]))
        return q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        actions = self.mdp.getPossibleActions(state)
        #if we have no possible actions, return none
        if(len(actions) == 0):
            return None
        #otherwise, compute qVal for every possible action, and store so we can find best (maximizing) action
        Q = util.Counter() #used to store qvals for taking argmax
        for action in actions:
            Q[action] = self.computeQValueFromValues(state,action)
        return Q.argMax() #bestAction = argmax across all possible actions and associated Qvals

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        numStates = len(self.mdp.getStates())
        #run for a given number of iterations, updating the next value each time
        for x in range(self.iterations):
            #look at the next state and determine value
            state = self.mdp.getStates()[x % numStates] #x % numStates will repeatedly cycle through all indexes with incrementing x
            qMax = float('-inf')
            #find the max (Q-value = value) given our set of possible actions
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                qMax = max(qMax,self.computeQValueFromValues(state,action)) #update qMax if calculated q is new max
            if(len(actions) == 0): # if there were no actions, the state is terminal and has a value of zero
                qMax = 0.0
            #update the value of the state
            self.values[state] = qMax

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
