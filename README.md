# Ideas for a more general PMFRG.jl code

This repository contains:
- some sample code and ideas written in code (see `/src`)
- some notes (*at the moment* in LaTeX) that explain what the code should do,
  with reference to the literature (see `/doc`)

The aim of the code is to provide 
a minimal and simple, general but inefficient, implementation 
of the PMFRG method, 
that consists basically 
of the implementation of the derivative functions,
which aims at readability and explicitness.

Using Julia's multiple dispatch paradigm,
this implementation can be extended with *efficient implementations* 
that use symmetries to greatly reduce the number of terms,
when they are available.

The "minimal and simple" implementation 
can then be used to test the "efficient implementation"
when it makes sense.

As usual in software and with many things in life,
this is not a finished project and will change over time,
possibly even radically.
