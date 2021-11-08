"""Algorithm to create GBT given a goal specification."""
from untilverify import ppatasks
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
    bt = BehaviourTree(parse_ltlf(formula))
    print(py_trees.display.ascii_tree(bt.root))
    return bt


def create_ppatask(postcond, precond, taskcnstr, gcnstr):
    # PostCond | (PreCond & X (TaskBulk U PostCond))
    postbulk = gcnstr + ' & '+  postcond
    prebulk = gcnstr + ' & '+  precond
    taskbulk = gcnstr + ' & '+  taskcnstr
    ppatask = '('+ postbulk + ') | ((' + prebulk + ') & X((' + taskbulk + ') U (' +postbulk + '))' +  ')'
    print(ppatask)
    parser = LTLfParser()
    ppaformula = parser(ppatask)
    create_recognizer(ppaformula)


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
    # formulas = [
    #     '(a)', '(!a)', 'F(a)', 'G(a)', 'X(a)',
    #     '(a | b)', '(a & b)', '(a U b)']
    # for formula_string in formulas:
    #     parser = LTLfParser()
    #     formula = parser(formula_string)
    #     create_recognizer(formula)
    # ppa1 = '(!t & s) | ( (!t & !c) & X((!t) U (!t & s)))'
    # ppa2 = '(!t & c) | ( (!t & s) & X((!t & o) U (!t & c)))'
    # ppa = '('+ ppa1 + ') U (' + ppa2 + ')'
    # complexformulas = [ppa1, ppa2, ppa]

    # for formula_string in complexformulas:
    #     parser = LTLfParser()
    #     formula = parser(formula_string)
    #     create_recognizer(formula)

    create_ppatask('c', 's', 'o', '!t')


main()