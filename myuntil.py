from flloat.parser.ltlf import LTLfParser

from py_trees.composites import Sequence, Selector
from py_trees.decorators import Decorator, Inverter
from py_trees.trees import BehaviourTree
# from py_trees.behaviour import Behaviour
import py_trees
from py_trees import common
import copy


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
        self.last_time_step = common.Status.SUCCESS
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
            return_status = self.last_time_step
            if self.decorated.status == common.Status.SUCCESS:
                self.last_time_step = common.Status.SUCCESS
            elif self.decorated.status == common.Status.FAILURE:
                self.is_false_yet = True
                self.last_time_step = common.Status.FAILURE
        else:
            return_status = common.Status.FAILURE

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
        print(self.trace)        
        for val in self.trace:
            if val != common.Status.SUCCESS:
                return common.Status.FAILURE
        return common.Status.SUCCESS


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
        print('update',self.name, self.value, self.goalspec)
        if self.value[self.goalspec]:
            return common.Status.SUCCESS
        else:
            return common.Status.FAILURE


def setup_nodes(nodes, i, trace):
    print('a,b', i, trace[i])
    nodes[0].setup(0, 'a', trace[i])
    nodes[1].setup(0, 'a', trace[i])    
    nodes[2].setup(0, 'b', trace[i])    
    nodes[3].setup(0, 'b', trace[i])        


def skeleton(trace):
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
    # trace = [
    #     # {"a": True, "b": True}
    #     # {"a": True, "b": False},
    #     # {"a": True, "b": False},
    #     # {"a": True, "b": True},
    #     {"a": False, "b": False},
    #     # {"a": False, "b": True},        
    # ]    

    # py_trees.logging.level = py_trees.logging.Level.DEBUG
    output = py_trees.display.ascii_tree(root.root)
    print(output)

    for k in range(len(trace)):
        setup_nodes([goal1, goal11, goal2, goal22], i, trace)
        root.tick()
        print(root.root.status)
        i += 1


def ltl():

    # parse the formula
    parser = LTLfParser()
    formula = "(a U b)"
    parsed_formula = parser(formula)

    # evaluate over finite traces
    t1 = [
        {"a": True, "b": False},
        {"a": True, "b": False},    
        {"a": False, "b": False},
        {"a": True, "b": True},
        # {"a": False, "b": True},
        # {"a": False, "b": True},                
    ]
    # assert parsed_formula.truth(t1, 0)
    print('real LTL',parsed_formula.truth(t1))

    # from LTLf formula to DFA
    # dfa = parsed_formula.to_automaton()
    # assert dfa.accepts(t1)
    skeleton(t1)


def main():
    # skeleton()
    ltl()

if __name__ == '__main__':
    main()