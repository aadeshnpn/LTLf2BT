from flloat.parser.ltlf import LTLfParser



def testing():
    parser = LTLfParser()

    # formula_string = "((X a) & (b)) U c"
    formula_string = " (G (F a_12))"

    formula = parser(formula_string)        # returns a LTLfFormula

    t1 = [
        {'a_12': False, 'b': True, 'c': False},
        {'a_12': False, 'b': False, 'c': False},
        {'a_12': False, 'b': True, 'c': True},
        {'a_12': True, 'b': True, 'c': False},
        {'a_12': True, 'b': False, 'c': False},
        ]

    print(formula.truth(t1))


def main():

    parser = LTLfParser()

    # formula_string = "((X a) & (b)) U c"
    # formula_string = " (X (a & b))"
    # formula_string = " (X (F a))"
    formula_string = "(a)|(true & (X (true U a)))"
    formula = parser(formula_string)        # returns a LTLfFormula

    t1 = [
        {'a': False, 'b': True, 'c': False},
        {'a': True, 'b': False, 'c': False},
        {'a': False, 'b': True, 'c': True}
        ]

    print(formula.truth(t1))

# main()

testing()
