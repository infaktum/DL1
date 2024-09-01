boolean = (True, False)


def wahrheitstabelle(lf,name = ""):
    print(f'   a     b   |  {name}' )
    print(f'-------------|---------')
    for a, b in [ (a,b) for a in boolean for b in boolean]:
        print(f' {a!s:5} {b!s:5} | {lf((a,b))!s}')