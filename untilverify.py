from flloat.parser.ltlf import LTLfParser
from gbtalgo import create_recognizer
from ltl2btrevised import expriments


def test_mdp_cheese_trace():
    # trace = s31,s21,s22,s12,s13,s14,s24,s34,s44,s34,s24,s14,s13,s12,s22,s21,s31,s41
    # len(trace)=18
    trace = [
        {'c': False, 't': False, 'h': False},
        {'c': False, 't': False, 'h': False},
        {'c': False, 't': False, 'h': False},
        {'c': False, 't': False, 'h': False},
        {'c': False, 't': False, 'h': False},
        {'c': False, 't': False, 'h': False},
        {'c': False, 't': False, 'h': False},
        {'c': False, 't': False, 'h': False},
        {'c': True, 't': False, 'h': False},
        {'c': True, 't': False, 'h': False},
        {'c': True, 't': False, 'h': False},
        {'c': True, 't': False, 'h': False},
        {'c': True, 't': False, 'h': False},
        {'c': True, 't': False, 'h': False},
        {'c': True, 't': False, 'h': False},
        {'c': True, 't': False, 'h': False},
        {'c': True, 't': False, 'h': False},
        {'c': True, 't': False, 'h': True}
    ]
    psi1 = "(G (!t) & c) | ( (G (!t)) & ( (G (!t)) U ((G (!t) & c))))"
    psi2 = "(G (!t) & c & h) | ( (G (!t) & F (c)) & ( (G (!t)) U ((G (!t) & c & h))))"
    # psi = psi1 +' & ' + psi2
    # F Task1 and X F (Task2)
    psi = 'F(' + psi1 + ') & X(F(' + psi2+'))'
    parser = LTLfParser()
    formula1 = parser(psi1)
    formula2 = parser(psi2)
    formula = parser(psi)
    print(formula1.truth(trace))
    print(formula2.truth(trace))
    print(formula.truth(trace))


def new_task_grammar():
    trace = [
        {'p': False, 'g': True, 't': False, 'o': False},
        {'p': False, 'g': True, 't': False, 'o': False},
        {'p': False, 'g': True, 't': False, 'o': False},
        {'p': False, 'g': True, 't': True, 'o': False},
        {'p': False, 'g': True, 't': True, 'o': False},
        {'p': False, 'g': True, 't': True, 'o': False},
        {'p': False, 'g': True, 't': True, 'o': False},
        {'p': False, 'g': True, 't': True, 'o': False},
        {'p': True, 'g': True,  't': True,  'o': False},
        {'p': True, 'g': True,  't': True,  'o': False},
        {'p': True, 'g': True,  't': True,  'o': False},
        {'p': True, 'g': True,  't': True,  'o': False},
        {'p': True, 'g': True,  't': True,  'o': False},
        {'p': True, 'g': True,  't': True,  'o': False},
        {'p': True, 'g': True,  't': True,  'o': False},
        {'p': True, 'g': True,  't': True,  'o': False},
        {'p': True, 'g': True,  't': True,  'o': False},
        {'p': True, 'g': True,  't': True,  'o': True}
    ]
    # psi1 = "(G (g) & o) | ( (G (g)) & ( (G (g)) U ((G (g) & o))))"
    # psi1 = "(G (g) & o) | ( (G (g) & F (p)) & ( (G (g)) U ((G (g) & o))))"
    psi1 = "(F (p U t))"
    psi2 = "(G (g) & o) | ( (G (g) & F (p)) & ( (G (g) & t) U ((G (g) & o))))"
    # psi = psi1 +' & ' + psi2
    # F Task1 and X F (Task2)
    # psi = 'F(' + psi1 + ') & X(F(' + psi2+'))'
    parser = LTLfParser()
    formula1 = parser(psi1)
    formula2 = parser(psi2)
    # formula = parser(psi)
    print(formula1.truth(trace))
    print(formula2.truth(trace))
    # print(formula.truth(trace))


def mission_with_tau():
    trace = [
        {'p': True, 'g': True, 't': True, 'o': False},      # First try
        {'p': False, 'g': True, 't': True, 'o': False},     #
        {'p': False, 'g': False, 't': False, 'o': False},   # Global constraint violated
        {'p': True, 'g': True, 't': True, 'o': False},      ## GBT memory reset, 2nd Try
        {'p': False, 'g': True, 't': True, 'o': False},     ##
        {'p': False, 'g': True, 't': True, 'o': False},     ##
        {'p': False, 'g': True, 't': True, 'o': False},     ##
        {'p': False, 'g': True, 't': False, 'o': False},    ## Task constraint violated
        {'p': True, 'g': True,  't': True,  'o': False},    ### GBT memory reset, 3rd try
        {'p': False, 'g': True,  't': True,  'o': False},   ###
        {'p': False, 'g': True,  't': True,  'o': False},   ###
        {'p': False, 'g': True,  't': True,  'o': False},   ###
        {'p': False, 'g': True,  't': True,  'o': False},   ###
        {'p': False, 'g': True,  't': True,  'o': False},   ###
        {'p': False, 'g': True,  't': True,  'o': False},   ###
        {'p': False, 'g': True,  't': True,  'o': False},   ###
        {'p': False, 'g': True,  't': True,  'o': False},   ###
        {'p': False, 'g': True,  't': True,  'o': True}     ### Post condition satisfied.
    ]
    etrace = [
        {'s': False},
        {'s': False},
        {'s': True}
        ]
    # psi1 = "((G (g) & o) | ( (G (g)) & ( (G (g)) U ((G (g) & o)))))"
    # psi1 = "((G (g) & o) | ( (G (g) & (F p)) & ( (G (g)) U ((G (g) & o)))))"
    # psi1 = "(F (p U t))"
    psi2 = "F ((G (g) & o) |( (G (g) & (p)) & ( (G (g) & t) U ((G (g) & o)))))"
    psi3 = "F s"
    # psi = psi1 +' & ' + psi2
    # F Task1 and X F (Task2)
    # psi = 'F(' + psi1 + ') & X(F(' + psi2+'))'
    parser = LTLfParser()
    formula1 = parser(psi2)
    formula2 = parser(psi3)
    # formula2 = parser(psi2)
    # formula = parser(psi)
    print('state trace', formula1.truth(trace))
    print('execution trace', formula2.truth(etrace))
    # print(formula2.truth(trace))
    # print(formula.truth(trace))


def mission_with_tau_next():
    trace = [
        {'p': True, 'g': True, 't': True, 'o': False},      # First try
        {'p': False, 'g': True, 't': True, 'o': False},     #
        {'p': False, 'g': True,  't': True,  'o': True},    # <PPATask> returns success.
        {'p': False, 'g': True,  't': True,  'o': True}     ## Second time only PostBlk evaluation is needed
    ]
    etrace = [
        {'s': True},
        {'s': True}
        ]
    # psi1 = "((G (g) & o) | ( (G (g)) & ( (G (g)) U ((G (g) & o)))))"
    # psi1 = "((G (g) & o) | ( (G (g) & (F p)) & ( (G (g)) U ((G (g) & o)))))"
    # psi1 = "(F (p U t))"
    psi2 = "X ((G (g) & o) |( (G (g) & (p)) & ( (G (g) & t) U ((G (g) & o)))))"
    psi3 = "X s"
    # psi = psi1 +' & ' + psi2
    # F Task1 and X F (Task2)
    # psi = 'F(' + psi1 + ') & X(F(' + psi2+'))'
    parser = LTLfParser()
    formula1 = parser(psi2)
    formula2 = parser(psi3)
    # formula2 = parser(psi2)
    # formula = parser(psi)
    print('state trace', formula1.truth(trace))
    print('execution trace', formula2.truth(etrace))
    # print(formula2.truth(trace))
    # print(formula.truth(trace))


def mission_with_tau_seq():
    etrace = [
        {'o': True, 't': False},
        {'t': True}
        ]
    psi3 = "o & X t"
    parser = LTLfParser()
    formula2 = parser(psi3)
    print('execution trace', formula2.truth(etrace))


def test_sequentail_task():
    psi = 'F(a & X b)'
    # psi = 'F(a) & (X F(b))'
    trace = [
        {'a': False, 'b': False},
        {'a': False, 'b': False},
        {'a': False, 'b': False},
        {'a': True, 'b': False},
        {'a': False, 'b': True},
        {'a': False, 'b': False},
        {'a': False, 'b': False},
        {'a': False, 'b': False}
    ]
    parser = LTLfParser()
    formula1 = parser(psi)
    print(formula1.truth(trace))


def testing():
    parser = LTLfParser()

    # formula_string = "((X a) & (b)) U c"
    # formula_string = " (G (F a_12))"
    # formula_string = "(X(b) U X(c))"
    formula_string = "(b U c)"

    formula = parser(formula_string)        # returns a LTLfFormula

    t1 = [
        {'a_12': False, 'b': False, 'c': True},
        # {'a_12': True, 'b': False, 'c': True},
        # {'a_12': False, 'b': True, 'c': False},
        # {'a_12': False, 'b': True, 'c': False},
        # {'a_12': False, 'b': True, 'c': False},
        # {'a_12': False, 'b': False, 'c': True},
        ]

    print(formula.truth(t1))


def ppatasks():
    parser = LTLfParser()

    # formula_string = "((X a) & (b)) U c"
    # formula_string = " (X (a & b))"
    # formula_string = " (X (F a))"
    # formula_string = "(a)|(true & (X (true U a)))"
    ## t -> trap, c -> carrying, s -> found site,
    ppa1 = '(!t & s) | ( (!t & !c) & X((!t) U (!t & s)))'
    ppa2 = '(!t & c) | ( (!t & s) & X((!t) U (!t & c)))'
    formula_string = '('+ ppa1 + ') U (' + ppa2 + ')'
    # formula = parser(ppa1)        # returns a LTLfFormula
    # print(formula)
    ppabt = create_recognizer(formula_string)

    t1 = [
        {'t': False, 's': False, 'c': False},
        {'t': False, 's': False, 'c': False},
        {'t': False, 's': False, 'c': False},
        {'t': False, 's': True, 'c': False},
        {'t': False, 's': True, 'c': False},
        {'t': False, 's': True, 'c': False},
        {'t': False, 's': True, 'c': True},
        {'t': False, 's': True, 'c': True},
        ]

    # print(formula.truth(t1))
    class args:
        def __init__(self):
            self.trace ='fixed'
    args.trace = 'fixed'

    expriments([t1], ppabt, [ppabt], formula_string, args)


def ppatasksuntil():
    parser = LTLfParser()

    # formula_string = "((X a) & (b)) U c"
    # formula_string = " (X (a & b))"
    # formula_string = " (X (F a))"
    # formula_string = "(a)|(true & (X (true U a)))"
    ## t -> trap, c -> carrying, s -> found site,
    ppa1 = '(!t & s) | ( (!t & !c) & X((!t) U (!t & s)))'
    ppa2 = '(!t & h) | ( (!t & c) & X((!t) U (!t & h)))'
    formula_string = '('+ ppa1 + ') U (' + ppa2 + ')'
    # formula = parser(ppa1)        # returns a LTLfFormula
    # print(formula)
    ppabt = create_recognizer(formula_string)

    t1 = [
        {'t': False, 'h': False, 's': False, 'c': False},
        {'t': False, 'h': False, 's': False, 'c': False},
        {'t': False, 'h': False, 's': False, 'c': False},
        {'t': False, 'h': False, 's': True, 'c': True},
        {'t': False, 'h': False, 's': True, 'c': True},
        {'t': False, 'h': False, 's': True, 'c':True},
        {'t': False, 'h': False, 's': True, 'c': True},
        {'t': False, 'h': True, 's': True, 'c': True},
        ]

    # print(formula.truth(t1))
    class args:
        def __init__(self):
            self.trace ='fixed'
    args.trace = 'fixed'

    expriments([t1], ppabt, [ppabt], formula_string, args)


def main():

    parser = LTLfParser()

    # formula_string = "((X a) & (b)) U c"
    # formula_string = " (X (a & b))"
    # formula_string = " (X (F a))"
    # formula_string = "(a)|(true & (X (true U a)))"
    formula_string = " (F(a) & ((F(b))))"
    formula = parser(formula_string)        # returns a LTLfFormula

    t1 = [
        {'a': False, 'b': False, 'c': False},
        {'a': False, 'b': False, 'c': False},
        {'a': True, 'b': False, 'c': True},
        {'a': False, 'b': True, 'c': False},
        {'a': False, 'b': False, 'c': True}
        ]

    print(formula.truth(t1))

# main()

# testing()
# ppatasks()
# ppatasksuntil()
# test_mdp_cheese_trace()
# test_sequentail_task()
# new_task_grammar()
mission_with_tau_seq()
