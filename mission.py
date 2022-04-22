from flloat.parser.ltlf import LTLfParser
from gbtalgo import create_recognizer
from ltl2btrevised import expriments


def mission_only_task():
    trace = [
        {'p': True, 'g': True, 't': True, 'o': False},
        {'p': False, 'g': True, 't': True, 'o': False},
        {'p': False, 'g': True, 't': True, 'o': False},
        {'p': False, 'g': True, 't': True, 'o': True}
    ]
    psi2 = "((G (g) & o) |( (G (g) & (p)) & ( (G (g) & t) U ((G (g) & o)))))"
    parser = LTLfParser()
    formula1 = parser(psi2)
    print('Task state trace', formula1.truth(trace))


def mission_finally():
    trace = [
        {'p': True, 'g': True, 't': True, 'o': False},
        {'p': False, 'g': True, 't': True, 'o': False},
        {'p': False, 'g': True, 't': True, 'o': False},
        {'p': False, 'g': True, 't': True, 'o': False},
        {'p': False, 'g': True, 't': True, 'o': False},
        {'p': False, 'g': True, 't': True, 'o': False},
        {'p': False, 'g': True, 't': True, 'o': False},
        {'p': False, 'g': True, 't': True, 'o': True}
    ]

    psi2 = "F ((G (g) & o) |( (G (g) & (p)) & ( (G (g) & t) U ((G (g) & o)))))"
    parser = LTLfParser()
    formula1 = parser(psi2)
    print('Finally state trace', formula1.truth(trace))


def mission_or():
    # First task was success
    # p,g,t,o - > q,h,u,n
    trace1 = [
        {'p': True, 'g': True, 't': True, 'o': False,'q': True, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': True, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': True, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': True, 'q': True, 'h': True, 'u': True, 'n': False}
    ]
    # First task falied and second was success
    trace2 = [
        {'p': True, 'g': True, 't': True, 'o': False,  'q': True, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': True, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': True, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': True, 'h': True, 'u': True, 'n': False}, # First task was failure
        {'p': True, 'g': True, 't': True, 'o': False,  'q': True, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': False, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': False, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': False, 'h': True, 'u': True, 'n': True} # 2nd task was success
    ]


    # First task falied and second was failed
    trace3 = [
        {'p': True, 'g': True, 't': True, 'o': False,  'q': True, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': True, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': True, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': True, 'h': True, 'u': True, 'n': False}, # First task was failure
        {'p': True, 'g': True, 't': True, 'o': False,  'q': True, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': False, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': False, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': False, 'h': True, 'u': True, 'n': False} # 2nd task was failure
    ]

    psi1 = "((G (g) & o) |( (G (g) & (p)) & ( (G (g) & t) U ((G (g) & o)))))"
    psi2 = "((G (h) & n) |( (G (h) & (q)) & ( (G (h) & u) U ((G (h) & n)))))"
    psi = psi1 + '| ' + psi2
    parser = LTLfParser()
    formula1 = parser(psi)
    print('Task 1 success', formula1.truth(trace1))
    print('Task 1 fail and Task 2 success', formula1.truth(trace2))
    print('Task 1 fail and Task 2 failed', formula1.truth(trace3))


def mission_and():
    # p,g,t,o - > q,h,u,n
    # Task 1 failed and Task 2 succeed
    # Finally operator for precondition needed for and to work
    trace1 = [
        {'p': True, 'g': True, 't': True, 'o': False,'q': False, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': False, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': False, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': False, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False,'q': True, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': False, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': False, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': False, 'h': True, 'u': True, 'n': True},
    ]
    # Task 1 succeed and Task 2 succeed
    trace2 = [
        {'p': True, 'g': True, 't': True, 'o': False,'q': False, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': False, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': False, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': True, 'q': False, 'h': True, 'u': True, 'n': False},
        {'p': True, 'g': True, 't': True, 'o': False,'q': True, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': False, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': False, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': False, 'h': True, 'u': True, 'n': True},
    ]
    # Task 1 succed and Task 2 fail
    trace3 = [
        {'p': True, 'g': True, 't': True, 'o': False,'q': True, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': False, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': False, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': True, 'q': False, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False,'q': True, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': False, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': False, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': False, 'h': True, 'u': True, 'n': False},
    ]

    # Task 1 fail and Task 2 fail
    trace4 = [
        {'p': True, 'g': True, 't': True, 'o': False,'q': True, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': False, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': False, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': False, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False,'q': True, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': False, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': False, 'h': True, 'u': True, 'n': False},
        {'p': False, 'g': True, 't': True, 'o': False, 'q': False, 'h': True, 'u': True, 'n': False},
    ]
    psi1 = "((G (g) & o) |( (G (g) & (p)) & ( (G (g) & t) U ((G (g) & o)))))"
    psi2 = "((G (h) & n) |( (G (h) & (F(q))) & ( (G (h) & u) U ((G (h) & n)))))"
    psi = psi1 + '& ' + psi2
    parser = LTLfParser()
    formula1 = parser(psi)
    print('Task (F, T)', formula1.truth(trace1))
    print('Task (T, T)', formula1.truth(trace2))
    print('Task (T, F)', formula1.truth(trace3))
    print('Task (F, F)', formula1.truth(trace4))
    # print('Task 1 fail and Task 2 success', formula1.truth(trace2))
    # print('Task 1 fail and Task 2 failed', formula1.truth(trace3))


def main():
    # mission_only_task()
    # mission_finally()
    # mission_or()
    mission_and()


if __name__ == '__main__':
    main()