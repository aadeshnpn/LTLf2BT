"""Algorithm to create GBT given a goal specification."""
from flloat.ltlf import LTLfAlways, LTLfAnd, LTLfEventually, LTLfNext, LTLfNot, LTLfOr, LTLfUntil
from flloat.parser.ltlf import LTLfParser, LTLfAtomic

from py_trees.composites import Sequence, Selector, Parallel
from py_trees.decorators import Decorator, Inverter
from py_trees.trees import BehaviourTree
from py_trees.behaviour import Behaviour
import py_trees
from py_trees import common
import copy
import argparse
import numpy as np

from ltl2btrevised import PropConditionNode, Negation, And, Next, Globally, Finally, UntilA, UntilB, Until


def create_recognizer(formula):
    # print(dir(formula), type(formula))
    # # print('s', formula.s)
    # # print('f', formula.f, type(formula.f))
    # print('base expression', formula.base_expression)
    # print('formulas', formula.formulas, type(formula.formulas))
    # print('delta', formula.delta)
    # print('operator', formula.operator_symbol)
    # # print('automaton', formula.to_automaton())
    # # print('ldlf', formula.to_ldlf())
    # print('str', formula.__str__)
    bt = BehaviourTree(parse_ltlf(formula))
    print(py_trees.display.ascii_tree(bt.root))


def create_generator():
    pass


def parse_ltlf(formula):
    # Just proposition
    if isinstance(formula, LTLfAtomic):
        # map to BT conditio node
        return PropConditionNode(formula.s)
    # Unary or binary operator
    else:
        if (
            isinstance(formula, LTLfEventually) or isinstance(formula, LTLfAlways)
                or isinstance(formula, LTLfNext) or isinstance(formula, LTLfNot) ):
            if isinstance(formula, LTLfEventually):
                return Finally(parse_ltlf(formula.f))
            elif isinstance(formula, LTLfAlways):
                return Globally(parse_ltlf(formula.f))
            elif isinstance(formula, LTLfNext):
                return Next(parse_ltlf(formula.f))
            elif isinstance(formula, LTLfNot):
                return Negation(parse_ltlf(formula.f))

        elif (isinstance(formula, LTLfAnd) or isinstance(formula, LTLfOr) or isinstance(formula, LTLfUntil)):
            if isinstance(formula, LTLfAnd):
                leftformual, rightformula = formula.formulas
                parll = Parallel('And')
                leftnode = parse_ltlf(leftformual)
                rightnode = parse_ltlf(rightformula)
                parll.add_children([leftnode, rightnode])
                anddecorator = And(parll)
                return anddecorator
            elif isinstance(formula, LTLfOr):
                leftformual, rightformula = formula.formulas
                ornode = Selector('Or')
                leftnode = parse_ltlf(leftformual)
                rightnode = parse_ltlf(rightformula)
                ornode.add_children([leftnode, rightnode])
                return ornode

            elif isinstance(formula, LTLfUntil):
                leftformual, rightformula = formula.formulas
                leftnode = parse_ltlf(leftformual)
                rightnode = parse_ltlf(rightformula)
                useq = Sequence('UntilSeq')
                untila = UntilA(leftnode)
                untilb = UntilB(rightnode)
                useq.add_children([untilb, untila])
                anddec2 = And(useq)
                untildecorator = Until(anddec2)
                return untildecorator


def main():
    formula_string = " (F(a) & ((F(b) U c)))"
    formula_string = " (c U a)"
    parser = LTLfParser()
    formula = parser(formula_string)
    create_recognizer(formula)


main()