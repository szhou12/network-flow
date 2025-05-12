from ortools_solver import ortools_solver

def main():
    edges = [
        [0, 2,  12, 2],   # A -> X
        [0, 3,  12, 4],   # A -> Y
        [0, 1,  12, 1],
        [1, 2,  12, 3],   # B -> X
        [1, 3,  12, 1],   # B -> Y
    ]

    balance = [ 7, 5, -4, -8 ]

    ortools_solver(edges, balance)


if __name__ == "__main__":
    main()