from flloat.parser.ltlf import LTLfParser

 

def main():

    parser = LTLfParser()

    formula_string = "a U b"

    formula = parser(formula_string)        # returns a LTLfFormula

    t1 = [                                  # input trace

    {"a": False, "b": False},

    {"a": True, "b": False},

    {"a": True, "b": False},

    {"a": True, "b": True},

    {"a": False, "b": False},

    ]

 

    t2 = [

        {"a": True, "b": False},

        {"a": False, "b": True}

    ]

 

    t3 = [

        {"a": True, "b": False},

        {"a": True, "b": True},

        {"a": False, "b": True}

    ]

 

    t4 = [

        {"a": False, "b": True}

    ]

 

    t5 = [

        {"a": True, "b": False}

    ]

    print('First trace')

    for t in range(len(t1)):

        print('Given trace t1, until formula is ',formula.truth(t1, t), ' at time ',t)  # True

    print('\nSecond trace')

    for t in range(len(t2)):

        print('Given trace t2, until formula is ',formula.truth(t2, t), ' at time ',t)  # True

    print('\nThird trace')

    for t in range(len(t3)):

        print('Given trace t3, until formula is ',formula.truth(t3, t), ' at time ',t)  # True

    print('\nFourth trace')

    for t in range(len(t4)):

        print('Given trace t4, until formula is ',formula.truth(t4, t), ' at time ',t)  # True

    print('\nFifth trace')

    for t in range(len(t5)):

        print('Given trace t5, until formula is ',formula.truth(t5, t), ' at time ',t)  # True


main()        