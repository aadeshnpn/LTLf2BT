from flloat.parser.ltlf import LTLfParser

from py_trees.composites import Sequence, Selector
from py_trees.decorators import Decorator, Inverter
from py_trees.trees import BehaviourTree
from py_trees.behaviour import Behaviour
import py_trees
from py_trees import common
import copy
import argparse


# Method that calls the BT execution node setup method
# This supplies trace at time i for the nodes
def setup_node(nodes, trace_i):
    for node in nodes:
        node.setup(0, trace_i)


# Method executes a BT passed as an argument
def execute_bt(bt, trace, nodes):
    # Args: bt -> BT to tick
    #       trace -> a trace of lenght m
    #       nodes -> Execution nodes of BT that takes trace as input
    for k in range(len(trace)):
        setup_node(nodes, trace[k])
        bt.tick()
    return bt.root.status



# Execute both BT and Ltlf with same traces for comparision
def execute_both_bt_ltlf(subtree, formula, trace, nodes, verbos=True):
    # Args:
        # Which BT class to use
        # LTL formual
        # Trace of lenght m
        # BT exeuction nodes which require trace input
    # Trace of length 1

    # Create a BT from the subtree
    root = BehaviourTree(subtree)    
    
    # Creating a LTLf parser object
    parser = LTLfParser()
    # Parsed formula
    parsed_formula = parser(formula)
    bt_status = execute_bt(root, trace, nodes)
    ltlf_status = parsed_formula.truth(trace)
    if verbos:
        print("trace: {}', 'BT status: {}', 'LTLf status: {}".format(trace, bt_status, ltlf_status))
    return bt_status, ltlf_status


class PropConditionNode(Behaviour):
    """Condition node for the atomic propositions.

    Inherits the Behaviors class from py_trees. This
    behavior implements the condition node for the atomic LTLf propositions.
    """

    def __init__(self, name):
        """Init method for the condition node."""
        super(PropConditionNode, self).__init__(name)
        self.proposition_symbol = name
    
    def setup(self, timeout, value=False):
        """Have defined the setup method.

        This method defines the other objects required for the
        condition node.
        Args:
        timeout: property from Behavior super class. Not required
        symbol: Name of the proposition symbol
        value: A dict object with key as the proposition symbol and 
               boolean value as values. Supplied by trace.
        """
        self.value = value

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass

    def update(self):
        """
        Main function that is called when BT ticks.
        """
        # if the proposition value is true
        ## return Success
        # else
        ## return Failure
        if self.value[self.proposition_symbol]:
            return common.Status.SUCCESS
        else:
            return common.Status.FAILURE


def proposition2condition(verbos=True):
    # Trace of length 1
    trace1 = [
        {'a': False}
    ]
    # Trace of length 1    
    trace2 = [
        {'a': True}
    ]    
    # Trace of length 3
    trace3 = [
        {'a': True},
        {'a': False},        
        {'a': True}        
    ]    

    expno = 1
    # It is important to create a new execution object for each trace
    # as BT are state machine. 
    for trace in [trace1, trace2, trace3]:
        # Create condition node that is semantically equivalent 
        # to atomic proposition
        cnode = PropConditionNode('a')
        if verbos:
            print('--------------')
            print('Experiment no: ', expno)
        execute_both_bt_ltlf(cnode, 'a', trace, [cnode], verbos)
        if verbos:
            print('=============')        
        expno += 1


def main(args):
    if args.test == 'P':
        proposition2condition()
    elif args.test == '~':
        pass
    elif args.test == '&':
        pass    
    elif args.test == 'X':
        pass    
    elif args.test == 'U':
        pass    
    elif args.test == 'G':
        pass    
    elif args.test == 'F':
        pass    
    elif args.test == 'ALL':
        pass    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, choices = ['P', '~', '&', 'X', 'U', 'G', 'F', 'ALL'])
    args = parser.parse_args()
    main(args)