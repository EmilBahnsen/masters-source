import tensorflow as tf

def print_matrix(matrix):
    tf.print(tf.math.real(matrix), summarize=100)

def print_non_zero_matrix(matrix):
    print_matrix(tf.greater(tf.math.real(matrix), 1e-5))

def unitary_test(matrix):
    return tf.transpose(matrix, conjugate=True) @ matrix

def print_unitary_test(matrix):
    tf.print(tf.math.real(unitary_test(matrix)), summarize=100)

def print_non_zero_unitary_test(matrix):
    print_non_zero_matrix(unitary_test(matrix))

def print_complex(*args):
    string = tf.as_string(tf.math.real(args[0])) + ' ' + tf.as_string(tf.math.imag(args[0])) + 'j'
    if len(args) > 1:
        tf.print(string, print_complex(*args[1:]))
    else:
        tf.print(string)