# TODO

## Field methods

### Calculus
- [ ] `antiderivative` — cumulative integral along an axis
- [ ] `grad` — gradient returning all partial derivatives at once
- [ ] `interpolate` — extend to multi-dimensional fields

### Arithmetic
- [X] arithmetic operators `+`, `-`, `*`, `/` between fields and scalars
- [ ] `__abs__` — absolute value
- [ ] `norm` — L2 norm integrated over the domain

### Inspection
- [ ] `max`, `min` — maximum and minimum values
- [X] `plot` — simple matplotlib plotting method
- [ ] 'plot' - add option for real and imag on the same plot
- [ ] `restrict` — restrict field to a subrange of an axis

### Transformations
- [ ] `apply` — apply an arbitrary function to the values
- [ ] `outer` — outer product of two 1D fields to produce a 2D field

## Models
- [ ] 'models.py' - tests for QHR
- [ ] `XXX` - XXX spin chain

## Physics modules
- [X] `tba.py` — dressing equation, TBA solver
- [X] `tba.py` - tests for zero_temperature
- [ ] `ghd.py` — GHD evolution

### Thermodynamics
- [ ] `sound_velocity` - sound velocity from TBA
- [ ] `Luttinger_K` - Luttinger liquid parameter from K
- [ ] `susceptibility` - charge susceptibility matrix 

## Infrastructure
- [ ] First release on PyPI