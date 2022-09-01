""" 
Mollifiers and smooth step functions

A mollifier defined on all real numbers has the property that 
    x <= 0 -> 0
    x > 0 -> e^-(1/x)

Additionally, this function is continuous everywhere,
and is inifinitely differentiable.
Using this function, f(x), we can define an analog with finite support by
g(x) = f(x) * f(1 - x)
Now g(x) has support [0, 1], and is inifinitely differentiable.

Next, defining h(x) = (integral of g(z) from z = 0 to z = x) / || g ||_L1
then h is 0 for x <= 0, 1 for x >= 1, and makes this transition in a way
that it is infinitely differentiable.

Finally if we define s(x) = h(x + 2) * h(2 - x),
then for s is infinitely differentiable and has
    x <= -2 -> s = 0
    -1 <= x <= 1 -> s = 1
    2 <= x -> s = 0

This module provides s with added modification which allows the user 
to modulate the size of the transition layers as well as how large 
the set of values where s(x) = 1 is.
"""

import numpy as np 
from scipy.integrate import quad 

from functools import singledispatch

@singledispatch
def smooth_function_exp(x: float):
    """
    Represents x -> { 0,        x <= 0
                    { e^-(1/x)  x > 0
    
    Input:
        x : float, list of floats, or numpy array of floats
            Input variable.
    Output:
        y : value determined from above mapping,
            if multiple values are passed through a list or array,
            this function is evaluated pointwise.
            Has the same datatype as x.
    """

    if x <= 0:
        return 0.0
    else:
        return np.exp(-1.0 / x)
    
# allow if statements to work with iterables
@smooth_function_exp.register
def _(x : list):
    return list(map(smooth_function_exp, x))

@smooth_function_exp.register
def _(x : np.ndarray):
    return np.array(smooth_function_exp(list(x)))

def mollifier(x, loc=0.0, scale=1.0):
    """
    Mollifier function, infinitely smooth with compact support

    Input:
        x : float,  list of floats, or numpy array of floats
            Input variable in function.
        loc : float, optional
            offset of variable x.
            The default is 0.
        scale : float > 0, optional
            scale of the support of this function.
            The default is 1.

        If both loc and scale are ommitted, the support of this function
        is [-1, 1].

        If both loc and scalar are provided, the new support of this
        function will be [loc - scale, loc + scale].

        To get a mollifier on [0, 1], use loc=0.5, scale=0.5

    Warning:
        this function is not setup to handle multiple values of loc or scale
        simultaneously, you (the user) must do this externally for 
        reliable performance

    Output:
        y : value determined by x as mollifier mapping.
            If multiple values of x are passed through a list or array,
            this function is evaulated pointwise.
            Has the same datatype as x.
    """

    # map [loc - scale, loc + scale] -> [0, 1]
    z = (x - (loc - scale)) / (2 * scale)
    return smooth_function_exp(z) * smooth_function_exp(1 - z)

l1_norm_default_mollifier_eval = quad(
    lambda x: mollifier(x, loc=0.5, scale=0.5), 0, 1)
l1_norm_default_mollifier = l1_norm_default_mollifier_eval[0]

@singledispatch
def smooth_step(x : float, end_support=0.0, begin_identity=1.0):
    """
    Smooth step function which transitions from 0 to 1 in a 
    finite interval, and is infinitely differentiable.

    Input:
        x : float, list of floats, or numpy array of floats
            input variable.
        end_support : float, optional
            x <= end_support -> 0.
            The default is 0.
        begin_identity : float, optional
            x >= begin_identity -> 1.
            The default is 1.
    
    Output:
        y : mapped values of x.
            If x is a list or array, the evaluation is done pointwise.
            This has the same datatype as x.
    """

    if end_support > begin_identity:
        # bounds are flipped, so flip x accordingly
        return smooth_step(-x, 
            end_support=begin_identity, begin_identity=end_support)

    # quick evaluations, no integral required
    if x <= end_support:
        return 0.0
    elif x >= begin_identity:
        return 1.0

    else:
        # x in [end_support, begin_identity] -> z in [0, 1]
        integral_scaling = begin_identity - end_support
        z = x - end_support / integral_scaling
        unscaled_result = quad(
            lambda x: mollifier(x, loc=0.5, scale=0.5), 0, z
        )
        return unscaled_result[0] / l1_norm_default_mollifier * \
            integral_scaling

@smooth_step.register
def _(x : list, end_support=0.0, begin_identity=1.0):
    return list(map(
        lambda z: smooth_step(z, 
            end_support=end_support, begin_identity=begin_identity), x
    ))

@smooth_step.register
def _(x : np.ndarray, end_support=0.0, begin_identity=1.0):
    return np.array(smooth_step(list(x), 
        end_support=end_support, begin_identity=begin_identity))


def smooth_indicator(x, unity_span=(-1.0, 1.0), rev_up=1.0, rev_down=1.0):
    """
    Smooth set indicator function, which is 1 on a predefined set [a, b],
    0 on a superset thereof [a - c, b + d],
    and transitions smoothly so that it is infinitely differentiable
    everywhere

    Input:
        x : float, list of floats, or numpy array of floats
            input data for function.
        unity_span : 2-tuple or list of length 2, optional
            interval [a, b] over which this function evaluates to 1.
            The default is [-1, 1]
        rev_up, rev_down : floats > 0, optional
            these define the width of the increasing and decreasing 
            intervals of the function,
            in the interval [a - c, b + d], rev_up = c, and rev_down = d.
            The defaults for each are 1, and so the support of this function
            is [-2, 2]
        
    Warning:
        this function is not setup to handle multiple values of the 
        optional parameters.  To use multiple values you should do this
        in an external loop-like structure for predictable performance.

    Output:
        y : mapped values of x.  
        In the case of a list or array input for x, this is done pointwise.
        Has the same datatype as x.
    """

    return smooth_step(x, 
        end_support=unity_span[0]-rev_up, begin_identity=unity_span[0]) * \
            smooth_step(x, 
                end_support=unity_span[1]+rev_down, begin_identity=unity_span[1])

