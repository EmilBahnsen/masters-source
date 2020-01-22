import PySimpleGUI as sg
from qutip import *
from diamond import *
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

n = 2
N = 2**(n*4)
n_qubits = 4*n
init_state = ket('00000000')

# All the stuff inside your window.
operators = ['I','U(π)','U(π/2)', 'U(pi/4)', 'SWAP', *['X'+str(i) for i in range(8)], *['H'+str(i) for i in range(8)]]
operators_qutip = [I8, UN(pi,pi), UN(pi/2,pi/2), UN(pi/4,pi/4), swap(n_qubits,[3,6]), *[X(8,i) for i in range(8)], *[H(8,i) for i in range(8)]]
n_opers = 20
layout = [
    [sg.Text('Diamond chain length: 2 (8 qubits)')],
    [sg.Text('Operators:'), sg.Text('States:')],
    *[[sg.Combo(operators, key='oper{}'.format(i),font='Times', enable_events=True), sg.Text(state2string(n_qubits,init_state), key='state{}'.format(i), size=(100,1),font=(10))] for i in range(n_opers)],
    [sg.Button('Reset', key='Reset')]
]

# Create the Window
window = sg.Window('Diamond', layout, finalize=True)


# Fock state plot
def fock_plot(oper, title):
    state_new = oper * ket('0'*4*n)

    fig = plt.figure(figsize=(10,5))
    indecies_plot = [index2label(index,N) for index in range(N)]
    state_plot = np.abs([state_new[i][0][0] for i in range(N)])**2
    plt.bar(indecies_plot, state_plot)
    plt.title(title)
    plt.xlabel(r'state')
    plt.ylabel(r'$|a|^2$')
    return fig

def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def string2oper(string):
    index = operators.index(string)
    return operators_qutip[index]

# fig,_ = hinton(UN(pi))
# fig.set_dpi(50)
# fig_canvas_agg = draw_figure(window['unitary_plot'].TKCanvas, fig)

# fig = fock_plot(UN(pi,pi), 'Diamond')
# fig.set_dpi(50)
# fig_canvas_agg = draw_figure(window['histogram'].TKCanvas, fig)

def check_visibility():
    for i in range(n_opers-1,0,-1):
        print('ves' + str(i))
        if window['oper'+str(i)].get() == 'I' and window['oper'+str(i-1)].get() == 'I':
            print('ves2')
            window['oper' + str(i)].update(visible=False)
            window['state' + str(i)].update(visible=False)
        else:
            print('ves3')
            window['oper' + str(i)].update(visible=True)
            window['state' + str(i)].update(visible=True)

# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event is 'Reset':
        for i in range(n_opers):
            oper_combo = window['oper' + str(i)]
            oper_combo.update(set_to_index=0)
        event = 'oper0'
    if event in ['oper{}'.format(i) for i in range(n_opers)]:
        last_state = init_state
        for i in range(n_opers):
            oper_string = window['oper{}'.format(i)].get()
            oper = string2oper(oper_string)
            state_string = ''
            if oper == I8 and i != 0:
                state_string = window['state'+str(i-1)].DisplayText
            else:
                state = oper * last_state
                state_string = state2string(n_qubits, state)
            text_field = window['state' + str(i)]
            text_field.update(state_string)
            last_state = state
