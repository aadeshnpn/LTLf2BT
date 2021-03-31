from flloat.parser.ltlf import LTLfParser

from py_trees.composites import Sequence, Selector
from py_trees.decorators import Decorator, Inverter
from py_trees.trees import BehaviourTree
from py_trees.behaviour import Behaviour
import py_trees
from py_trees import common
import copy
import argparse
import numpy as np


# Method to randomly create traces
def getrandomtrace(n=2, maxtracelen=0):
    t1 = [{
        'a': False, 'b': False
    }]
    t2 = [{
        'a': False, 'b': True
    }]
    t3 = [{
        'a': True, 'b': False
    }]
    t4 = [{
        'a': True, 'b': True
    }]  
    tracelist = []
    def gettraces():
        if maxtracelen == 0:
            m = np.random.randint(1, 50)
        else:
            m = maxtracelen
        choices = {
            0: t1[0],
            1: t2[0],
            2: t3[0],
            3: t4[0]
        }
        trace = []
        for i in range(m):
            j = np.random.choice([0,1,2,3], m)
            trace.append(choices[j[i]])
        return trace    
    for k in range(n):
        tracelist.append(gettraces())

    return tracelist


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

## Until operator

# Right sub-tree Decorator for the until node
class DeltaU(Decorator):
    """Decorator node for the Until operator.

    Inherits the Decorator class from py_trees. This
    behavior implements a decorator for Until operator.
    """
    def __init__(self, child, name=common.Name.AUTO_GENERATED):
        """
        Init with the decorated child.

        Args:
            child : child behaviour node
            name : the decorator name
        """
        super(DeltaU, self).__init__(name=name, child=child)
        self.indx = 0
        self.previous_value = common.Status.SUCCESS
        self.previous_previous_value = common.Status.SUCCESS
        self.always_failure = False

    def update(self):
        """
        Main function that is called when BT ticks.
        This returns the Globally operator status
        """        
        # For until operator \psi_2 needs to be True at some point and
        # when \psi_2 is finally true, all previous \psi_1 needs to be true

        ## So the logic for the decorator node is to handle 
        # edges cases with trace lenght 1 and 2. This decorator node has three memory units
        # : first: to track \psi_1 previous value 
        # : second: to tracke \psi_1 previous previous value
        # : third: to remeber if \psi_1 every has been Failure before

        # Handels For trace lenght 1
        if self.indx == 0:
            self.beginning = False
            return_value = common.Status.SUCCESS
            self.previous_value = self.decorated.status
        # Handel for trace lenght 2
        elif self.indx == 1:
            if (self.previous_value == common.Status.SUCCESS and self.previous_previous_value == common.Status.SUCCESS):
                return_value = common.Status.SUCCESS
            else:
                return_value = common.Status.FAILURE
            self.previous_previous_value = self.previous_value                
            self.previous_value = self.decorated.status
        # Handel for any other trace lenght
        else:
            if self.always_failure:
                return_value = common.Status.FAILURE                
            elif (self.previous_value == common.Status.FAILURE or self.previous_previous_value == common.Status.FAILURE):
                self.always_failure = True
                return_value = common.Status.FAILURE                                
            elif (self.previous_value == common.Status.SUCCESS and self.previous_previous_value == common.Status.SUCCESS):
                return_value = common.Status.SUCCESS
            self.previous_previous_value = self.previous_value
            self.previous_value = self.decorated.status
        self.indx +=1        
        return return_value



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
def next2decorator(args, verbos=True):
    if args.trace == 'fixed':
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
        traces = [trace1, trace2, trace3, trace4]
    else:
        traces = getrandomtrace(n=args.no_trace_2_test, maxtracelen=args.max_trace_len)

    expno = 0
    returnvalueslist = []
    # It is important to create a new execution object for each trace
    # as BT are state machine. 
    for trace in traces:
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
def globally2decorator(args, verbos=True):
    if args.trace == 'fixed':
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
        traces = [trace1, trace2, trace3, trace4, trace5]
    else:
        traces = getrandomtrace(n=args.no_trace_2_test, maxtracelen=args.max_trace_len)
    expno = 0
    returnvalueslist = []
    # It is important to create a new execution object for each trace
    # as BT are state machine. 
    for trace in traces:
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
def finally2decorator(args, verbos=True):
    if args.trace == 'fixed':
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
        traces = [trace1, trace2, trace3, trace4, trace5]
    else:
        traces = getrandomtrace(n=args.no_trace_2_test, maxtracelen=args.max_trace_len)
    expno = 0
    returnvalueslist = []
    # It is important to create a new execution object for each trace
    # as BT are state machine. 
    for trace in traces:
        # Create a bt sub-tree that is semantically equivalent to
        # Finally LTLf operator
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


# Experiment 6 for Until operator
def until2subtree(args, verbos=True):
    if args.trace == 'fixed':
        # Trace of length 1
        t1 = [
            {'a': False, 'b': False}
        ]
        # Trace of length 1    
        t2 = [
            {'a': False, 'b': True}        
        ]    
        # Trace of length 1
        t3 = [
            {'a': True, 'b': False}                
        ]    
        # Trace of length 1
        t4 = [
            {'a': True, 'b': True}                
        ]        
        # Trace of length 3
        traces = [t1, t2, t3, t4]
    else:
        traces = getrandomtrace(n=args.no_trace_2_test, maxtracelen=args.max_trace_len)
    expno = 0
    returnvalueslist = []
    # It is important to create a new execution object for each trace
    # as BT are state machine. 
    for trace in traces:
        # Create a bt sub-tree that is semantically equivalent to
        # Until sub-tree
        seqleft = Sequence('main')
        goal1 = PropConditionNode('a')
        goal2 = PropConditionNode('b')    
        deltau = DeltaU(goal1)
        seqleft.add_children([deltau, goal2])        
        top = Finally(seqleft)        

        if verbos:
            print('--------------')
            print('Experiment no: ', expno)
        # Call the excute function that will execute both BT and LTLf
        returnvalueslist.append(execute_both_bt_ltlf(top, 'a U b', trace, [goal1, goal2], verbos))
        if verbos:
            print('=============')        
        expno += 1

    # t5 = [ 
    #      [t1[0], t1[0]],
    #      [t1[0], t2[0]],   
    #      [t1[0], t3[0]],        
    #      [t1[0], t4[0]],
    #      [t2[0], t1[0]],
    #      [t2[0], t2[0]],   
    #      [t2[0], t3[0]],        
    #      [t2[0], t4[0]],
    #      [t3[0], t1[0]],
    #      [t3[0], t2[0]],   
    #      [t3[0], t3[0]],        
    #      [t3[0], t4[0]],
    #      [t4[0], t1[0]],
    #      [t4[0], t2[0]],   
    #      [t4[0], t3[0]],        
    #      [t4[0], t4[0]]
    #      ] 

    # for trace in t5:
    #     # Create a bt sub-tree that is semantically equivalent to
    #     # Until sub-tree
    #     seqleft = Sequence('main')
    #     goal1 = PropConditionNode('a')
    #     goal2 = PropConditionNode('b')    
    #     deltau = DeltaU(goal1)
    #     seqleft.add_children([deltau, goal2])        
    #     top = Finally(seqleft)        

    #     if verbos:
    #         print('--------------')
    #         print('Experiment no: ', expno)
    #     # Call the excute function that will execute both BT and LTLf
    #     returnvalueslist.append(execute_both_bt_ltlf(top, 'a U b', trace, [goal1, goal2], verbos))
    #     if verbos:
    #         print('=============')        
    #     expno += 1    
    count = count_bt_ltlf_return_values(returnvalueslist)
    print("Total Experiment Runs: {}, BT and LTLf agree: {}".format(expno, count))


# Experiment 7 for until operator with random traces
def until2subtree_randomtrace(verbos=True):
    expno = 0
    returnvalueslist = []
    tracelist = getrandomtrace()
    # It is important to create a new execution object for each trace
    # as BT are state machine. 
    for trace in tracelist:
        # Create a bt sub-tree that is semantically equivalent to
        # Until sub-tree
        seqleft = Sequence('main')
        goal1 = PropConditionNode('a')
        goal2 = PropConditionNode('b')    
        deltau = DeltaU(goal1)
        seqleft.add_children([deltau, goal2])        
        top = Finally(seqleft)        

        if verbos:
            print('--------------')
            print('Experiment no: ', expno)
        # Call the excute function that will execute both BT and LTLf
        returnvalueslist.append(execute_both_bt_ltlf(top, 'a U b', trace, [goal1, goal2], verbos))
        if verbos:
            print('=============')        
        expno += 1
    count = count_bt_ltlf_return_values(returnvalueslist)
    print("Total Experiment Runs: {}, BT and LTLf agree: {}".format(expno, count))


# Counter example X psi1  wedge \psi2
def counter_example(args, verbos=True):
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

    trace6 =  [
        {'a': False, 'b': True},                
        {'a': True, 'b': True}                        
    ]        
    # Experiment variables
    expno = 0
    returnvalueslist = []
    # It is important to create a new execution object for each trace
    # as BT are state machine. 
    for trace in [trace1, trace2, trace3, trace4, trace5, trace6]:
        # Create BT-sub tree that is semantically equivalent 
        # to And LTLf operator
        # And sub-tree
        cnode1 = PropConditionNode('a')
        cnode2 = PropConditionNode('b')  
        # Next LTLf operator
        # cnode = PropConditionNode('a')
        ndecorator = Next(cnode1, 'Next')              
        sequence = Sequence('And')

        sequence.add_children([ndecorator, cnode2])
        if verbos:
            print('--------------')
            print('Experiment no: ', expno)
        # Call the excute function that will execute both BT and LTLf
        returnvalueslist.append(execute_both_bt_ltlf(sequence, '(X a) & (b)', trace, [cnode1, cnode2], verbos))
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
        next2decorator(args)
    elif args.test == 'U':
        until2subtree(args)
    elif args.test == 'G':
        globally2decorator(args)
    elif args.test == 'F':
        finally2decorator(args)
    elif args.test == 'C':
        counter_example((args))

    elif args.test == 'U_random':
        until2subtree_randomtrace(args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, choices = ['P', '~', '&', 'X', 'U', 'G', 'F', 'U_random', 'C'])
    parser.add_argument('--trace', type=str, choices = ['fixed', 'random'], default='fixed')
    parser.add_argument('--max_trace_len', type=int, default=3)    
    parser.add_argument('--no_trace_2_test', type=int, default=16)        

    ## Usages

    # Atomic proposition 
    ## python ltl2bt.py --test 'P' 

    # Negation 
    ## python ltl2bt.py --test '~'     

    # And 
    ## python ltl2bt.py --test '&'     

    # Next 
    ## python ltl2bt.py --test 'X'        

    # Globally 
    ## python ltl2bt.py --test 'G'            

    # Finally 
    ## python ltl2bt.py --test 'F'                

    # Until sub-tree
    # to run until expeirments with pre-defined traces: 
    ## python ltl2bt.py --test 'U'

    # to run until expeirments with random length trace: 
    ## python ltl2bt.py --test 'U' --trace 'random' --max_trace_len 0

    # to run until expeirments with 10 length trace: 
    ## python ltl2bt.py --test 'U' --trace 'random' --max_trace_len 10

    # to run until expeirments with 10 length trace and 50 different traces: 
    ## python ltl2bt.py --test 'U' --trace 'random' --max_trace_len 10 --no_trace_2_test 50

    ### These aruguments (random, max_trace_len, no_trace_2_test) is equally
    ### valid for (Next, Globally, Finally) operator as well
    args = parser.parse_args()


    main(args)