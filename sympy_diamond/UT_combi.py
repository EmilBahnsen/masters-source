from sympy import *
from sympy_diamond import *
import itertools

def print_latex(expr):
    print(latex(expr)
          .replace('tp', r't_{\Psip}')
          .replace('tm', r't_{\Psim}')
          .replace(r'\left[', r'\left(')
          .replace(r'\right]', r'\right)'))

print('All UT(t) commute')
t = Symbol('t', real=True)
Us = [U00(t), U11(t), Up(t), Um(t)]
for i,Ui in enumerate(Us):
    for j,Uj in enumerate(Us):
        print(i,j)
        sp.pprint(sp.simplify(Ui @ Uj - Uj @ Ui))

print('UT(t) combi:')
t00, t11, tp, tm = symbols('t00 t11 tp tm', real=True)

# Us_names = ['U00', 'U11', 'Up', 'Um']
# Us = [U00(t00), U11(t11), Up(tp), Um(tm)]
Us_names = ['U00', 'U11', 'Up']
Us = {'U00': U00(t00), 'U11': U11(t11), 'Up': Up(tp)}
print('Single:')
for i,U in Us.items():
    print(i)
    pprint(U)

print('Double:')
def print_product(i, j):
    print(i, j)
    expr = simplify(Us[i] @ Us[j])
    pprint(expr)
    print_latex(expr)
print_product('U00', 'U11')
print_product('U00', 'Up')
print_product('U11', 'Up')


print('Triple:')
print('U00', 'U11', 'Up')
expr = simplify(U00(t00) @ U11(t11) @ Up(tp))
pprint(expr)
print_latex(expr)


