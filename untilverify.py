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
    psi2 = "(G (!t) & c & h) | ( (G (!t) & c) & ( (G (!t)) U ((G (!t) & c & h))))"
    psi = psi1 +' & ' + psi2
    parser = LTLfParser()
    formula1 = parser(psi1)
    formula2 = parser(psi2)
    formula = parser(psi)
    print(formula1.truth(trace))
    print(formula2.truth(trace))
    print(formula.truth(trace))


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
test_mdp_cheese_trace()