from flloat.parser.ltlf import LTLfParser

from py_trees.composites import Sequence, Selector
from py_trees.decorators import Decorator, Inverter
from py_trees.trees import BehaviourTree
# from py_trees.behaviour import Behaviour
import py_trees
from py_trees import common


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
        self.is_first_time = True
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
            if self.decorated.status == common.Status.SUCCESS:
                # self.last_time_step = common.Status.SUCCESS
                return_status = common.Status.SUCCESS
                # return common.Status.SUCCESS
            elif self.decorated.status == common.Status.FAILURE:
                self.is_false_yet = True
                # self.last_time_step = common.Status.FAILURE
                return_status = common.Status.FAILURE                
                # return common.Status.FAILURE
            # return self.decorated.status
        else:
            return_status = common.Status.FAILURE

        if self.is_first_time:
            self.is_first_time = False
            return common.Status.SUCCESS
        else:
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
        self.last_time_step = common.Status.FAILURE
        # self.is_true_yet = False

    def update(self):
        """
        Flip :data:`~py_trees.common.Status.FAILURE` and
        :data:`~py_trees.common.Status.SUCCESS`

        Returns:
            :class:`~py_trees.common.Status`: the behaviour's new status :class:`~py_trees.common.Status`
        """
        return_status = None

        if self.last_time_step == common.Status.SUCCESS:
            return_status = common.Status.SUCCESS


        if self.decorated.status == common.Status.SUCCESS:
            curr_status = common.Status.SUCCESS
        elif self.decorated.status == common.Status.FAILURE:
            curr_status = common.Status.FAILURE
                
        if common.Status.SUCCESS in [self.last_time_step, curr_status]:
            return_status = common.Status.SUCCESS
        else:
            return_status = common.Status.FAILURE
        self.last_time_step = return_status

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

    def update(self):
        """
        Flip :data:`~py_trees.common.Status.FAILURE` and
        :data:`~py_trees.common.Status.SUCCESS`

        Returns:
            :class:`~py_trees.common.Status`: the behaviour's new status :class:`~py_trees.common.Status`
        """
        self.trace.append(self.decorated.status)
        for val in self.trace:
            if val != common.Status.SUCCESS:
                return common.Status.FAILURE
        return common.status.SUCCESS


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
        if self.value[self.goalspec]:
            return common.Status.SUCCESS
        else:
            return common.Status.FAILURE


def setup_nodes(nodes, i, trace):
    nodes[0].setup(0, 'a', trace[i])
    nodes[1].setup(0, 'b', trace[i])    


def skeleton():
    main = Selector('R')

    # Left sub-tree
    seqleft = Sequence('Left')
    goal1 = LTLNode('g1')
    delta1 = Delta1(goal1)
    seqleft.add_children([delta1])
    
    # Right sub-tree
    seqright = Sequence('Right')    
    delta2 = Delta2(seqright)
    deltainv = Inverter(goal1)
    goal2 = LTLNode('g1')
    deltag = DeltaG(goal2)
    seqright.add_children([deltainv, deltag])

    # Main tree
    main.add_children([seqleft, delta2])

    root = BehaviourTree(main)
    i = 0
    trace = [
        {"a": False, "b": False},
        {"a": True, "b": False},
        {"a": True, "b": False},
        {"a": True, "b": True},
        {"a": False, "b": False},
    ]    

    py_trees.logging.level = py_trees.logging.Level.DEBUG
    output = py_trees.display.ascii_tree(root.root)
    print(output)

    for k in range(len(trace)):
        setup_nodes([goal1, goal2], i, trace)
        root.tick()
        i += 1


def ltl():

    # parse the formula
    parser = LTLfParser()
    formula = "(a U b)"
    parsed_formula = parser(formula)

    # evaluate over finite traces
    t1 = [
        {"a": False, "b": False},
        {"a": True, "b": False},
        {"a": True, "b": False},
        {"a": True, "b": True},
        {"a": False, "b": False},
    ]
    # assert parsed_formula.truth(t1, 0)
    print(parsed_formula.truth(t1))

    # from LTLf formula to DFA
    # dfa = parsed_formula.to_automaton()
    # assert dfa.accepts(t1)



def main():
    skeleton()
    ltl()

if __name__ == '__main__':
    main()