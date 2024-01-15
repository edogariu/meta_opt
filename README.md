# THINGS TO RUN
- [ ] **BIG WMT EXPERMIENTS!!! And repeat CIFAR stuff for MNIST too**
- [ ] run with *many* ema values and plot preferred $\mu$ over time video-style
- [ ] check overfitting on task other than cross-entropy
- [ ] Add avg to training and run diagonal
- [ ] add MP, cosine, cyclical learning rates, hedging, AGD, DoWG, D-adaptation, adagrad?
- [ ] check my hunch that we produce same optimum even w different meta lr
- [ ] Plot along varying time to see which method performs best at a fixed time, sweeping hyperparameter vs accuracy at different times
- [ ] Investigate effect of overfitting
- [ ] Investigate training stability & couple w sequential stability
- [ ] add caching to dataloaders
- [ ] try other settings?

- [X] ~~**VERIFY BATCHING ISNT DEGENERATE ON WMT**~~
- [X] ~~**USE OPTAX GRAD CLIPPING**~~
- [X] ~~**run meta-opt with optimal initial lr instead of near-0**~~ (turns out optimal is 0 lol)
- [X] ~~fix buffer alignment issue~~
- [X] ~~Fix issue with the delta~~
- [X] ~~rerun CIFAR baselines with weight decay~~
- [X] ~~Correctly handle std (linear comb of R.V.s) for plotting when we smooth loss~~
- [X] ~~Fix eval for WMT to not take so long so that the big experiments will work~~

# NOTES ON META OPT
## CIFAR
- SGD: 0.1 is best, in “CIFAR_1-13”
- Adam: 0.001 is best, in “CIFAR_1-13”
- Scalar: adam w <= 1e-4, in “CIFAR_1-13”
- Momentum: 0.01 w 0.9, in “CIFAR_1-13”
- The lower the meta lr, the longer it takes for training loss to bottom out (and so it bottoms out lower), but on bigger datasets it seems that our method prevents overfitting and still keeps low eval loss. 
- With the correct hyperparams, we kinda track the optimum as learning progresses. So, 



