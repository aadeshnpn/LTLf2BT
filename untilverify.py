from flloat.parser.ltlf import LTLfParser



def testing():
    parser = LTLfParser()

    # formula_string = "((X a) & (b)) U c"
    formula_string = " (G (F a))"

    formula = parser(formula_string)        # returns a LTLfFormula

    t1 = [
        {'a': False, 'b': True, 'c': False},
        {'a': False, 'b': False, 'c': False},
        {'a': False, 'b': True, 'c': True},
        {'a': True, 'b': True, 'c': False},
        {'a': True, 'b': False, 'c': False},
        {'a': False, 'b': True, 'c': True}
        ]

    print(formula.truth(t1))


def main():

    parser = LTLfParser()

    # formula_string = "((X a) & (b)) U c"
    formula_string = " (X (a & b))"

    formula = parser(formula_string)        # returns a LTLfFormula

    t1 = [
        {'a': True, 'b': True, 'c': False},
        {'a': True, 'b': False, 'c': False},
        {'a': False, 'b': True, 'c': True}
        ]

    print(formula.truth(t1))

# main()

testing()
