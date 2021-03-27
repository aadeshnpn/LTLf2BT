from flloat.parser.ltlf import LTLfParser

from py_trees.composites import Sequence, Selector
from py_trees.decorators import Decorator, Inverter
from py_trees.trees import BehaviourTree
# from py_trees.behaviour import Behaviour
import py_trees
from py_trees import common
import copy
import numpy as np


class Delta1(Decorator):
    """
    A decorator for the left sub-tree. 
    """
    def __init__(self, child, name=common.Name.AUTO_GENERATED):
        """
        Init with the decorated child.

        Args:
            child (:class:`~py_trees.behaviour.Behaviour`): behaviour to time
            name (:obj:`str`): the decorator name
        """
        super(Delta1, self).__init__(name=name, child=child)
        # self.last_time_step = common.Status.SUCCESS
        # self.is_first_time = True
        self.is_false_yet = False

    def update(self):
        """
        Flip :data:`~py_trees.common.Status.FAILURE` and
        :data:`~py_trees.common.Status.SUCCESS`

        Returns:
            :class:`~py_trees.common.Status`: the behaviour's new status :class:`~py_trees.common.Status`
        """
        return_status = None

        if not self.is_false_yet:
            # return_status = self.last_time_step
            if self.decorated.status == common.Status.SUCCESS:
                return_status = common.Status.SUCCESS
            elif self.decorated.status == common.Status.FAILURE:
                self.is_false_yet = True
                return_status = common.Status.FAILURE
        else:
            return_status = common.Status.FAILURE

        return return_status


class Delta3(Decorator):
    """
    A decorator for the left sub-tree. 
    """
    def __init__(self, child, name=common.Name.AUTO_GENERATED):
        """
        Init with the decorated child.

        Args:
            child (:class:`~py_trees.behaviour.Behaviour`): behaviour to time
            name (:obj:`str`): the decorator name
        """
        super(Delta3, self).__init__(name=name, child=child)
        self.is_true_yet = False

    def update(self):
        return_status = None

        if not self.is_true_yet:
            # return_status = self.last_time_step
            if self.decorated.status == common.Status.SUCCESS:
                self.is_true_yet = True                
                return_status = common.Status.SUCCESS
            elif self.decorated.status == common.Status.FAILURE:
                return_status = common.Status.FAILURE
        else:
            return_status = common.Status.SUCCESS

        return return_status


class Delta2(Decorator):
    """
    A decorator for the left sub-tree. 
    """
    def __init__(self, child, name=common.Name.AUTO_GENERATED):
        """
        Init with the decorated child.

        Args:
            child (:class:`~py_trees.behaviour.Behaviour`): behaviour to time
            name (:obj:`str`): the decorator name
        """
        super(Delta2, self).__init__(name=name, child=child)
        self.last_time_step = common.Status.SUCCESS

    def update(self):
        """
        Flip :data:`~py_trees.common.Status.FAILURE` and
        :data:`~py_trees.common.Status.SUCCESS`

        Returns:
            :class:`~py_trees.common.Status`: the behaviour's new status :class:`~py_trees.common.Status`
        """
        # return_status = self.last_time_step
        # if self.decorated.status == common.Status.SUCCESS:
        #     self.last_time_step = common.Status.SUCCESS
        # elif self.decorated.status == common.Status.FAILURE:
        #     self.last_time_step = common.Status.FAILURE
        return_status = self.decorated.status
        return return_status        


class DeltaG(Decorator):
    """
    A decorator for the left sub-tree. 
    """
    def __init__(self, child, name=common.Name.AUTO_GENERATED):
        """
        Init with the decorated child.

        Args:
            child (:class:`~py_trees.behaviour.Behaviour`): behaviour to time
            name (:obj:`str`): the decorator name
        """
        super(DeltaG, self).__init__(name=name, child=child)
        self.trace = []
        self.start_append = False

    def update(self):
        """
        Flip :data:`~py_trees.common.Status.FAILURE` and
        :data:`~py_trees.common.Status.SUCCESS`

        Returns:
            :class:`~py_trees.common.Status`: the behaviour's new status :class:`~py_trees.common.Status`
        """
        # if self.decorated.status == common.Status.SUCCESS:
        #     self.start_append = True

        # Only append when you get success
        # if self.start_append:
        self.trace.append(self.decorated.status)

        # if len(self.trace) < 1:
        #    return self.decorated.status    
        #
        # print(self.trace)        
        for val in self.trace:
            if val != common.Status.SUCCESS:
                return common.Status.FAILURE
        return common.Status.SUCCESS

class DeltaU(Decorator):
    """
    A decorator for the left sub-tree. 
    """
    def __init__(self, child, name=common.Name.AUTO_GENERATED):
        """
        Init with the decorated child.

        Args:
            child (:class:`~py_trees.behaviour.Behaviour`): behaviour to time
            name (:obj:`str`): the decorator name
        """
        super(DeltaU, self).__init__(name=name, child=child)
        self.psi1_false_yet = False
        self.indx = 0
        self.previous_value = common.Status.SUCCESS
        self.previous_previous_value = common.Status.SUCCESS
        self.always_failure = False

    def update(self):
        """
        Flip :data:`~py_trees.common.Status.FAILURE` and
        :data:`~py_trees.common.Status.SUCCESS`

        Returns:
            :class:`~py_trees.common.Status`: the behaviour's new status :class:`~py_trees.common.Status`
        """
        # if self.decorated.status == common.Status.SUCCESS:
        #     self.start_append = True
        if self.indx == 0:
            self.beginning = False
            return_value = common.Status.SUCCESS
            self.previous_value = self.decorated.status
        elif self.indx == 1:
            if (self.previous_value == common.Status.SUCCESS and self.previous_previous_value == common.Status.SUCCESS):
                return_value = common.Status.SUCCESS
            else:
                return_value = common.Status.FAILURE
            self.previous_previous_value = self.previous_value                
            self.previous_value = self.decorated.status
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
        # print('previous value',self.indx, self.previous_previous_value, self.previous_value, self.decorated.status)            
        self.indx +=1        
        return return_value



class LTLNode(py_trees.behaviour.Behaviour):
    """LTL node for the proving decomposition.

    Inherits the Behaviors class from py_trees. This
    behavior implements the LTL node for the Until LTL.
    """

    def __init__(self, name):
        """Init method for the LTL node."""
        super(LTLNode, self).__init__(name)
        self.goalspec = None
    
    def setup(self, timeout, goalspec, value=False):
        """Have defined the setup method.

        This method defines the other objects required for the
        LTL node. LTL specfication is the only property.
        """
        self.goalspec = goalspec
        self.value = value

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass

    def update(self):
        """
        Return the value.
        """
        # print('update',self.name, self.value, self.goalspec)
        if self.value[self.goalspec]:
            return common.Status.SUCCESS
        else:
            return common.Status.FAILURE


def setup_nodes(nodes, i, trace):
    # print('a,b', i, trace[i])
    nodes[0].setup(0, 'a', trace[i])
    nodes[1].setup(0, 'a', trace[i])    
    nodes[2].setup(0, 'b', trace[i])    
    nodes[3].setup(0, 'b', trace[i])        


def setup_nodesu(nodes, i, trace):
    # print('a,b', i, trace[i])
    nodes[0].setup(0, 'a', trace[i])    
    nodes[1].setup(0, 'b', trace[i])        



def neweruntil(trace):
    # Left sub-tree
    seleleft = Sequence('Left')
    goal1 = LTLNode('g1')
    goal2 = LTLNode('g2')    
    deltau = DeltaU(goal1)
    seleleft.add_children([deltau, goal2])
    
    top = Delta3(seleleft)
    root = BehaviourTree(top)
    i = 0

    # Creating a LTLf parser object
    parser = LTLfParser()
    # Until goal specification
    formula = "(a U b)"
    # Parsed formula
    parsed_formula = parser(formula)

    # py_trees.logging.level = py_trees.logging.Level.DEBUG
    output = py_trees.display.ascii_tree(root.root)
    # print(output)
    
    ltl_status = parsed_formula.truth(trace)

    for k in range(len(trace)):
        setup_nodesu([goal1, goal2], i, trace)
        root.tick()
        # print(i, common.Status.SUCCESS)
        i += 1

    bt_status = root.root.status
    
    print(ltl_status, bt_status)

    if (bt_status == common.Status.SUCCESS and ltl_status is True):
        return True
    elif (bt_status == common.Status.FAILURE and ltl_status is False):
        return True
    else:
        return False
    # if root.root.status == common.Status.SUCCESS:
    #     return True
    # else:
    #     return False
    # print(root.root.status)


def newuntil(trace):
    main = Sequence('R')

    # Left sub-tree
    seleleft = Selector('Left')
    goal1 = LTLNode('g1')
    goal11 = copy.copy(goal1)
    delta1 = Delta1(goal1)
    delta11 = copy.copy(Delta1(goal11))
    goal2 = LTLNode('g2')    
    goal22 = copy.copy(goal2)    
    deltag = DeltaG(goal2)
    seleleft.add_children([delta1, deltag])
    
    # Right sub-tree
    seqright = Sequence('Right')    
    delta2 = Delta2(goal22)

    seqright.add_children([delta2])

    # Main tree
    # main.add_children([seqleft, delta2])
    main.add_children([seleleft, seqright])

    top = Delta3(main)
    root = BehaviourTree(top)
    i = 0

    # Creating a LTLf parser object
    parser = LTLfParser()
    # Until goal specification
    formula = "(a U b)"
    # Parsed formula
    parsed_formula = parser(formula)

    # py_trees.logging.level = py_trees.logging.Level.DEBUG
    output = py_trees.display.ascii_tree(root.root)
    # print(output)

    for k in range(len(trace)):
        setup_nodes([goal1, goal11, goal2, goal22], i, trace)
        root.tick()
        # print(i, common.Status.SUCCESS)
        print(i, parsed_formula.truth(trace), root.root.status)        
        i += 1

    # if root.root.status == common.Status.SUCCESS:
    #     return True
    # else:
    #     return False
    # print(root.root.status)


def olduntil(trace):
    main = Sequence('R')

    # Left sub-tree
    seleleft = Selector('Left')
    goal1 = LTLNode('g1')
    goal11 = copy.copy(goal1)
    delta1 = Delta1(goal1)
    delta11 = copy.copy(Delta1(goal11))
    goal2 = LTLNode('g2')    
    goal22 = copy.copy(goal2)    
    deltag = DeltaG(goal2)
    seleleft.add_children([delta1, deltag])
    
    # Right sub-tree
    seqright = Sequence('Right')    
    delta2 = Delta2(goal22)

    seqright.add_children([delta2])

    # Main tree
    # main.add_children([seqleft, delta2])
    main.add_children([seleleft, seqright])

    root = BehaviourTree(main)
    i = 0
    # Creating a LTLf parser object
    parser = LTLfParser()
    # Until goal specification
    formula = "(a U b)"
    # Parsed formula
    parsed_formula = parser(formula)

    # py_trees.logging.level = py_trees.logging.Level.DEBUG
    output = py_trees.display.ascii_tree(root.root)
    # print(output)

    for k in range(len(trace)):
        setup_nodes([goal1, goal11, goal2, goal22], i, trace)
        root.tick()
        # print(i, common.Status.SUCCESS)
        print(i, parsed_formula.truth(trace, k), root.root.status)        
        i += 1

def main(fn):
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

    # First just testing four canonical boolean variables
    trace = [t1, t2, t3, t4]
    count = 0
    for t in trace:
        print('Trace', t)
        # Call the skeleton function that implements a minimal BT
        # for Until-sub tree based on the Figure from Dr.Goodrich
        # paper draft
        return1 = fn(t)
        if not return1:
            count += 1
        print('------')

    t5 = [ 
         [t1[0], t1[0]],
         [t1[0], t2[0]],   
         [t1[0], t3[0]],        
        [t1[0], t4[0]],
         [t2[0], t1[0]],
         [t2[0], t2[0]],   
         [t2[0], t3[0]],        
         [t2[0], t4[0]],
         [t3[0], t1[0]],
         [t3[0], t2[0]],   
         [t3[0], t3[0]],        
         [t3[0], t4[0]],
         [t4[0], t1[0]],
         [t4[0], t2[0]],   
         [t4[0], t3[0]],        
         [t4[0], t4[0]]
         ]
    for t in t5:
        print('Trace', t)
        return1 = fn(t)
        if not return1:
            count += 1
        print('------')         
     
    
    print("total trace: {}, Traces not agreed: {}".format(len(t5)+ len(trace), count))

    # t6 = [{
    #     'a': False, 'b': False},
    #     {'a': True, 'b': True,}
    #     ]
    # fn(t6)

def main1(fn):
    count = 0
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
    t6 = [ 
         [t1[0], t1[0], t1[0]],
         [t1[0], t1[0], t2[0]],   
         [t1[0], t1[0], t3[0]],        
         [t1[0], t1[0], t4[0]],

         [t1[0], t2[0], t1[0]],
         [t1[0], t2[0], t2[0]],   
         [t1[0], t2[0], t3[0]],        
         [t1[0], t2[0], t4[0]],         


         [t1[0], t3[0], t1[0]],
         [t1[0], t3[0], t2[0]],   
         [t1[0], t3[0], t3[0]],        
         [t1[0], t3[0], t4[0]],         


         [t1[0], t4[0], t1[0]],
         [t1[0], t4[0], t2[0]],   
         [t1[0], t4[0], t3[0]],        
         [t1[0], t4[0], t4[0]]
    ]

    t7 = [ 
         [t2[0], t2[0], t1[0]],
         [t2[0], t2[0], t2[0]],   
         [t2[0], t2[0], t3[0]],        
         [t2[0], t2[0], t4[0]],

         [t2[0], t3[0], t1[0]],
         [t2[0], t3[0], t2[0]],   
         [t2[0], t3[0], t3[0]],        
         [t2[0], t3[0], t4[0]],         


         [t2[0], t4[0], t1[0]],
         [t2[0], t4[0], t2[0]],   
         [t2[0], t4[0], t3[0]],        
         [t2[0], t4[0], t4[0]],         


         [t2[0], t1[0], t1[0]],
         [t2[0], t1[0], t2[0]],   
         [t2[0], t1[0], t3[0]],        
         [t2[0], t1[0], t4[0]]
    ]    

    t8 = [ 
         [t3[0], t3[0], t1[0]],
         [t3[0], t3[0], t2[0]],   
         [t3[0], t3[0], t3[0]],        
         [t3[0], t3[0], t4[0]],

         [t3[0], t4[0], t1[0]],
         [t3[0], t4[0], t2[0]],   
         [t3[0], t4[0], t3[0]],        
         [t3[0], t4[0], t4[0]],         


         [t3[0], t1[0], t1[0]],
         [t3[0], t1[0], t2[0]],   
         [t3[0], t1[0], t3[0]],        
         [t3[0], t1[0], t4[0]],         


         [t3[0], t2[0], t1[0]],
         [t3[0], t2[0], t2[0]],   
         [t3[0], t2[0], t3[0]],        
         [t3[0], t2[0], t4[0]]
    ]     

    t9 = [ 
         [t4[0], t4[0], t1[0]],
         [t4[0], t4[0], t2[0]],   
         [t4[0], t4[0], t3[0]],        
         [t4[0], t4[0], t4[0]],

         [t4[0], t1[0], t1[0]],
         [t4[0], t1[0], t2[0]],   
         [t4[0], t1[0], t3[0]],        
         [t4[0], t1[0], t4[0]],         


         [t4[0], t2[0], t1[0]],
         [t4[0], t2[0], t2[0]],   
         [t4[0], t2[0], t3[0]],        
         [t4[0], t2[0], t4[0]],         


         [t4[0], t3[0], t1[0]],
         [t4[0], t3[0], t2[0]],   
         [t4[0], t3[0], t3[0]],        
         [t4[0], t3[0], t4[0]]
    ]            

    for t in t6 + t7 + t8 + t9:
        print('Trace', t)
        return1 = fn(t)
        if not return1:
            count += 1
        print('------')  
    print("total trace: {}, Traces not agreed: {}".format(len(t6) + len(t7) + len(t8) + len(t9), count))


def mainrand(fn):
    count = 0
    n = 100
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
        m = np.random.randint(1, 50)
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
        return1 = fn(gettraces())
        if not return1:
            count += 1 
    
    print("total trace: {}, Traces not agreed: {}".format(n, count))


if __name__ == '__main__':
    # For old unitl 
    # main(olduntil)
    # For new until    
    # main(newuntil)
    # For newer until
    # main(neweruntil)    
    # main1(neweruntil)        
    mainrand(neweruntil)