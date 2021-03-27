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


# Function that compares retsults between LTLf and BT
def count_bt_ltlf_return_values(returnvalues):
    count = 0
    for value in returnvalues:
        if (value[0] == common.Status.SUCCESS and value[1] is True):
            count += 1
        elif (value[0] == common.Status.FAILURE and value[1] is False):
            count += 1
    return count
##  Start of BT sub-tree that implement the LTLf operators


# Just a simple condition node that implements atomic propositions
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


# Just a simple condition node that implements atomic propositions
class Negation(Decorator):
    """Decorator node for the negation of an atomic proposition.

    Inherits the Decorator class from py_trees. This
    behavior implements the negation of an atomic LTLf propositions.
    """
    def __init__(self, child, name=common.Name.AUTO_GENERATED):
        """
        Init with the decorated child.

        Args:
            child : child behaviour node
            name : the decorator name
        """
        super(Negation, self).__init__(name=name, child=child)

    def update(self):
        """
        Main function that is called when BT ticks.
        This returns the inverted status
        """        
        # if the proposition value is true
        ## return Failure
        # else
        ## return Success

        # This give access to the child class of decorator class
        if  self.decorated.status == common.Status.FAILURE:
            return common.Status.SUCCESS
        elif self.decorated.status == common.Status.SUCCESS:
            return common.Status.FAILURE


# Just a simple condition node that implements Next LTLf operator
class Next(Decorator):
    """Decorator node for the Next operator.

    Inherits the Decorator class from py_trees. This
    behavior implements the Next LTLf operator.
    """
    def __init__(self, child, name=common.Name.AUTO_GENERATED):
        """
        Init with the decorated child.

        Args:
            child : child behaviour node
            name : the decorator name
        """
        super(Next, self).__init__(name=name, child=child)
        self.idx = 0
        self.next_status = None

    def update(self):
        """
        Main function that is called when BT ticks.
        This returns the Next operator status
        """        
        # At index i, return Failure
        # At index i+1, return self.decorated
        if self.idx == 0:
            self.next_status = common.Status.FAILURE
        elif self.idx == 1:
            # This give access to the child class of decorator class
            if self.decorated.status == common.Status.SUCCESS:
                self.next_status = common.Status.SUCCESS
            elif self.decorated.status == common.Status.FAILURE:
                self.next_status = common.Status.FAILURE
        self.idx += 1
        return self.next_status


# Just a simple condition node that implements Globally LTLf operator
class Globally(Decorator):
    """Decorator node for the Globally operator.

    Inherits the Decorator class from py_trees. This
    behavior implements the Globally LTLf operator.
    """
    def __init__(self, child, name=common.Name.AUTO_GENERATED):
        """
        Init with the decorated child.

        Args:
            child : child behaviour node
            name : the decorator name
        """
        super(Globally, self).__init__(name=name, child=child)
        self.found_failure = False

    def update(self):
        """
        Main function that is called when BT ticks.
        This returns the Globally operator status
        """        
        # For globally the atomic proposition needs to hold for 
        # all trace. So if we find failure then Globally returns Failue
        # This give access to the child class of decorator class
        if self.found_failure:
            return common.Status.FAILURE
        else:
            if self.decorated.status == common.Status.SUCCESS:
                return common.Status.SUCCESS
            elif self.decorated.status == common.Status.FAILURE:
                self.found_failure = True
                return common.Status.FAILURE      


# Just a simple condition node that implements Finally LTLf operator
class Finally(Decorator):
    """Decorator node for the Finally operator.

    Inherits the Decorator class from py_trees. This
    behavior implements the Finally LTLf operator.
    """
    def __init__(self, child, name=common.Name.AUTO_GENERATED):
        """
        Init with the decorated child.

        Args:
            child : child behaviour node
            name : the decorator name
        """
        super(Finally, self).__init__(name=name, child=child)
        self.found_success = False

    def update(self):
        """
        Main function that is called when BT ticks.
        This returns the Globally operator status
        """        
        # For finally the atomic proposition needs to hold for 
        # just a state. So if we find one success then Finally returns Success
        # This give access to the child class of decorator class
        if self.found_success:
            return common.Status.SUCCESS
        else:
            if self.decorated.status == common.Status.SUCCESS:
                self.found_success = True                
                return common.Status.SUCCESS
            elif self.decorated.status == common.Status.FAILURE:
                return common.Status.FAILURE      


## Experiments to test each LTLf operator and its BT sub-tree


# Experiment 1 for simple atomic propositions
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
    # Experiment variables
    expno = 0
    returnvalueslist = []
    # It is important to create a new execution object for each trace
    # as BT are state machine. 
    for trace in [trace1, trace2, trace3]:
        # Create condition node that is semantically equivalent 
        # to atomic proposition
        cnode = PropConditionNode('a')
        if verbos:
            print('--------------')
            print('Experiment no: ', expno)
        # Call the excute function that will execute both BT and LTLf
        returnvalueslist.append(execute_both_bt_ltlf(cnode, 'a', trace, [cnode], verbos))
        if verbos:
            print('=============')        
        expno += 1
    count = count_bt_ltlf_return_values(returnvalueslist)
    print("Total Experiment Runs: {}, BT and LTLf agree: {}".format(expno, count))


# Experiment 2 for simple negation of atomic propositions
def negation2decorator(verbos=True):
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

    expno = 0
    returnvalueslist = []
    # It is important to create a new execution object for each trace
    # as BT are state machine. 
    for trace in [trace1, trace2, trace3]:
        # Create a bt sub-tree that is semantically equivalent to
        # Negation LTLf operator
        cnode = PropConditionNode('a')
        ndecorator = Negation(cnode, 'Invert')
        if verbos:
            print('--------------')
            print('Experiment no: ', expno)
        # Call the excute function that will execute both BT and LTLf
        returnvalueslist.append(execute_both_bt_ltlf(ndecorator, '!a', trace, [cnode], verbos))
        if verbos:
            print('=============')        
        expno += 1
    count = count_bt_ltlf_return_values(returnvalueslist)
    print("Total Experiment Runs: {}, BT and LTLf agree: {}".format(expno, count))



# Experiment 3 for simple and of atomic propositions
def and2sequence(verbos=True):
    # Trace of length 1
    trace1 = [
        {'a': False, 'b': False}
    ]
    # Trace of length 1    
    trace2 = [
        {'a': False, 'b': True}        
    ]    
    # Trace of length 1
    trace3 = [
        {'a': True, 'b': False}                
    ]    
    # Trace of length 1
    trace4 = [
        {'a': True, 'b': True}                
    ]        
    # Trace of length 3
    trace5 = [
        {'a': True, 'b': True},                
        {'a': False, 'b': False},                        
        {'a': True, 'b': True}                        
    ]        
    # Experiment variables
    expno = 0
    returnvalueslist = []
    # It is important to create a new execution object for each trace
    # as BT are state machine. 
    for trace in [trace1, trace2, trace3, trace4, trace5]:
        # Create BT-sub tree that is semantically equivalent 
        # to And LTLf operator
        # And sub-tree
        cnode1 = PropConditionNode('a')
        cnode2 = PropConditionNode('b')        
        sequence = Sequence('And')

        sequence.add_children([cnode1, cnode2])
        if verbos:
            print('--------------')
            print('Experiment no: ', expno)
        # Call the excute function that will execute both BT and LTLf
        returnvalueslist.append(execute_both_bt_ltlf(sequence, 'a & b', trace, [cnode1, cnode2], verbos))
        if verbos:
            print('=============')        
        expno += 1
    count = count_bt_ltlf_return_values(returnvalueslist)
    print("Total Experiment Runs: {}, BT and LTLf agree: {}".format(expno, count))


# Experiment 4 for Next operator
def next2decorator(verbos=True):
    # Trace of length 1
    trace1 = [
        {'a': False}
    ]
    # Trace of length 1    
    trace2 = [
        {'a': True}
    ]    
    # Trace of length 2
    trace3 = [
        {'a': False},
        {'a': True},        
    ]    
    # Trace of length 3
    trace4 = [
        {'a': False},
        {'a': True},        
        {'a': False}        
    ]        

    expno = 0
    returnvalueslist = []
    # It is important to create a new execution object for each trace
    # as BT are state machine. 
    for trace in [trace1, trace2, trace3, trace4]:
        # Create a bt sub-tree that is semantically equivalent to
        # Next LTLf operator
        cnode = PropConditionNode('a')
        ndecorator = Next(cnode, 'Next')
        if verbos:
            print('--------------')
            print('Experiment no: ', expno)
        # Call the excute function that will execute both BT and LTLf
        returnvalueslist.append(execute_both_bt_ltlf(ndecorator, 'X a', trace, [cnode], verbos))
        if verbos:
            print('=============')        
        expno += 1
    count = count_bt_ltlf_return_values(returnvalueslist)
    print("Total Experiment Runs: {}, BT and LTLf agree: {}".format(expno, count))



# Experiment 5 for Globally operator
def globally2decorator(verbos=True):
    # Trace of length 1
    trace1 = [
        {'a': False}
    ]
    # Trace of length 1    
    trace2 = [
        {'a': True}
    ]    
    # Trace of length 2
    trace3 = [
        {'a': False},
        {'a': True},        
    ]    
    # Trace of length 3
    trace4 = [
        {'a': False},
        {'a': True},        
        {'a': False}        
    ]   

    # Trace of length 4
    trace5 = [
        {'a': True},
        {'a': True},        
        {'a': True}        
    ]            

    expno = 0
    returnvalueslist = []
    # It is important to create a new execution object for each trace
    # as BT are state machine. 
    for trace in [trace1, trace2, trace3, trace4, trace5]:
        # Create a bt sub-tree that is semantically equivalent to
        # Globally LTLf operator
        cnode = PropConditionNode('a')
        gdecorator = Globally(cnode, 'Globally')
        if verbos:
            print('--------------')
            print('Experiment no: ', expno)
        # Call the excute function that will execute both BT and LTLf
        returnvalueslist.append(execute_both_bt_ltlf(gdecorator, 'G a', trace, [cnode], verbos))
        if verbos:
            print('=============')        
        expno += 1
    count = count_bt_ltlf_return_values(returnvalueslist)
    print("Total Experiment Runs: {}, BT and LTLf agree: {}".format(expno, count))


# Experiment 5 for Globally operator
def finally2decorator(verbos=True):
    # Trace of length 1
    trace1 = [
        {'a': False}
    ]
    # Trace of length 1    
    trace2 = [
        {'a': True}
    ]    
    # Trace of length 2
    trace3 = [
        {'a': False},
        {'a': True},        
    ]    
    # Trace of length 3
    trace4 = [
        {'a': False},
        {'a': True},        
        {'a': False}        
    ]   

    # Trace of length 4
    trace5 = [
        {'a': False},
        {'a': False},        
        {'a': False}        
    ]            

    expno = 0
    returnvalueslist = []
    # It is important to create a new execution object for each trace
    # as BT are state machine. 
    for trace in [trace1, trace2, trace3, trace4, trace5]:
        # Create a bt sub-tree that is semantically equivalent to
        # Globally LTLf operator
        cnode = PropConditionNode('a')
        gdecorator = Finally(cnode, 'Finally')
        if verbos:
            print('--------------')
            print('Experiment no: ', expno)
        # Call the excute function that will execute both BT and LTLf
        returnvalueslist.append(execute_both_bt_ltlf(gdecorator, 'F a', trace, [cnode], verbos))
        if verbos:
            print('=============')        
        expno += 1
    count = count_bt_ltlf_return_values(returnvalueslist)
    print("Total Experiment Runs: {}, BT and LTLf agree: {}".format(expno, count))



def main(args):
    if args.test == 'P':
        proposition2condition()
    elif args.test == '~':
        negation2decorator()
    elif args.test == '&':
        and2sequence()
    elif args.test == 'X':
        next2decorator()
    elif args.test == 'U':
        pass    
    elif args.test == 'G':
        globally2decorator()
    elif args.test == 'F':
        finally2decorator()
    elif args.test == 'ALL':
        pass    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, choices = ['P', '~', '&', 'X', 'U', 'G', 'F', 'ALL'])
    args = parser.parse_args()
    main(args)