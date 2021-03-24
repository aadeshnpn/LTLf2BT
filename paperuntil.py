from flloat.parser.ltlf import LTLfParser

from py_trees.composites import Sequence, Selector
from py_trees.decorators import Decorator, Inverter
from py_trees.trees import BehaviourTree
# from py_trees.behaviour import Behaviour
import py_trees
from py_trees import common
import copy


# Trying to implement the logic of the Decorator node defined the paper
class Delta1(Decorator):
    """
    A decorator for the left sub-tree. 
    """
    # This child is passed during Delta1 construction. For our case
    # this clid is \psi1
    def __init__(self, child, name=common.Name.AUTO_GENERATED):
        """
        Init with the decorated child.

        Args:
            child (:class:`~py_trees.behaviour.Behaviour`): behaviour to time
            name (:obj:`str`): the decorator name
        """
        super(Delta1, self).__init__(name=name, child=child)
        # Before the BT is ticked, the decorator has not yet received 
        # Failure status
        self.is_false_yet = False

    # Updated is called everytime
    def update(self):
        """
        Custom logic:
        return Failure once \psi1 evalutes to Failure
        else returns what \psi1 is currently
        """
        return_status = None

        # If the decorator has not yet see Failure from its child node
        if not self.is_false_yet:
            # Check the status of child node (\psi1)
            # self.decorated is pointing to child node \psi1
            if self.decorated.status == common.Status.SUCCESS:
                return_status = common.Status.SUCCESS
            elif self.decorated.status == common.Status.FAILURE:
                self.is_false_yet = True
                return_status = common.Status.FAILURE
        else:
            return_status = common.Status.FAILURE
        return return_status



class LTLNode(py_trees.behaviour.Behaviour):
    """LTL node for the proving decomposition.

    Inherits the Behaviors class from py_trees. This
    behavior implements the LTL node for the Until LTL.
    """

    def __init__(self, name):
        """Init method for the LTL node."""
        super(LTLNode, self).__init__(name)
    
    def setup(self, timeout, goalspec, value=False):
        """Have defined the setup method.

        This method defines the other objects required for the
        LTL node. LTL specfication is the only property.
        """
        # Input. Current trace value at time step t
        # value is a dictonary with a,b symbol values
        self.goalspec = goalspec
        self.value = value

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass
    
    # This method is called everytime by the BT function tick()
    def update(self):
        """
        Return the value.
        """
        # print('update',self.name, self.value, self.goalspec)
        # This grabs the value for the particular goal node
        # In our case we jsut have \psi1 and \psi2 where \psi1 is a symbol and
        # \psi2 is b symbol.
        ## If the value at the current timestep is True return Success 
        ## else failure
        if self.value[self.goalspec]:
            return common.Status.SUCCESS
        else:
            return common.Status.FAILURE


def setup_nodes(nodes, i, trace):
    # Passing input to the BT nodes \psi1 and \psi2
    # nodes is a list that has both execution nodes of the until tree
    # i is the time step
    nodes[0].setup(0, 'a', trace[i])
    nodes[1].setup(0, 'b', trace[i])    



def skeleton(trace):
    # Until sub-tree
    #######################
    #######Selector########
    ##########|############
    ###-------|--------####
    ###|##############|####    
    ###|##############|####        
    ##Delta1#######\Psi2### 
    ###|###################        
    ###|###################            
    ##Psi1#################
    
    # Creating the BT
    ## Main selector node
    main = Selector('R')
    # \Psi1 node
    goal1 = LTLNode('g1')
    # Delta1 is a decorator node
    # Below code wraps the goal1 node by the decorator node Delta1
    delta1 = Delta1(goal1)
    # \Psi2 node
    goal2 = LTLNode('g2')    
    # Adding left and right nodes to the root selector node
    main.add_children([delta1, goal2])
    # Creating Behavior Tree out of the root node
    root = BehaviourTree(main)

    ## This will print every tick of the Behavior Tree. The output is slightly confusing
    # py_trees.logging.level = py_trees.logging.Level.DEBUG
    
    # This is print the whoe BT in a ascii tree. 
    # Good for visulalizing the tree and making sure the BT is as intended.
    output = py_trees.display.ascii_tree(root.root)
    # print(output)
    
    # Creating a LTLf parser object
    parser = LTLfParser()
    # Until goal specification
    formula = "(a U b)"
    # Parsed formula
    parsed_formula = parser(formula)
    # Initializaing the time as 0
    i = 0
    # So based on the lenght of the trace we are ticking the BT
    for k in range(len(trace)):
        # Input need to be supplied to BT nodes before each tick
        setup_nodes([goal1, goal2], i, trace)
        # BT control flow/ tick
        root.tick()
        # Output from both the LTLf parser and BT 
        print(i, parsed_formula.truth(trace, k), root.root.status)
        # Increase the time step
        i += 1

    # Not important for now
    # if root.root.status == common.Status.SUCCESS:
    #     return True
    # else:
    #     return False
    # # print(root.root.status)


def main():
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

    for t in trace:
        print('Trace', t)
        # Call the skeleton function that implements a minimal BT
        # for Until-sub tree based on the Figure from Dr.Goodrich
        # paper draft
        skeleton(t)
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
        skeleton(t)
        print('------')         


if __name__ == '__main__':
    main()