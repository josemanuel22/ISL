```@meta
CurrentModule = ISL
```
# ISL.jl Documentation Guide

!!! compat This repository contains the Julia Flux implementation of the Invariant Statistical Loss (ISL) proposed in the paper [Training Implicit Generative Models via an Invariant Statistical Loss](https://arxiv.org/abs/2402.16435), published in the AISTATS 2024 conference.

Welcome to the documentation for `ISL.jl`, a Julia package designed for Invariant Statistical Learning. This guide provides a systematic overview of the modules, constants, types, and functions available in `ISL.jl`. Our documentation aims to help you quickly find the information you need to effectively utilize the package.



```@autodocs
Modules = [ISL]
Order   = [:module, :constant, :type]
```

```@docs
invariant_statistical_loss
auto_invariant_statistical_loss
ts_invariant_statistical_loss_one_step_prediction
ts_invariant_statistical_loss
ts_invariant_statistical_loss_multivariate
```
