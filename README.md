# THINGS TO RUN
- [ ] **VERIFY BATCHING ISNT DEGENERATE**
- [ ] **USE OPTAX GRAD CLIPPING**
- [ ] **BIG WMT EXPERMIENTS!!! And repeat CIFAR stuff for MNIST too**
- [ ] **run meta-opt with optimal initial lr instead of near-0**
- [ ] fix buffer alignment issue
- [ ] Fix issue with the delta
- [X] rerun CIFAR baselines with weight decay
- [X] Correctly handle std (linear comb of R.V.s) for plotting when we smooth loss
- [ ] Plot along varying time to see which method performs best at a fixed time, sweeping hyperparameter vs accuracy at different times
- [X] Fix eval for WMT to not take so long so that the big experiments will work
- [ ] Try meta opt from initial lr of 0.1 instead of tiny, maybe sweep initial lr
- [ ] Add avg to training and run diagonal
- [ ] Sweep hgd
- [ ] Investigate effect of overfitting
- [ ] Investigate training stability & couple w sequential stability
- [ ] add caching to dataloaders
- [ ] add MP, cosine, cyclical learning rates, hedging, AGD, DoWG, D-adaptation, adagrad?
- [ ] try other settings?

# NOTES ON META OPT
## CIFAR
- SGD: 0.1 is best, in “CIFAR_1-13”
- Adam: 0.001 is best, in “CIFAR_1-13”
- Scalar: adam w <= 1e-4, in “CIFAR_1-13”
- Momentum: 0.01 w 0.9, in “CIFAR_1-13”
- The lower the meta lr, the longer it takes for training loss to bottom out (and so it bottoms out lower), but on bigger datasets it seems that our method prevents overfitting and still keeps low eval loss. 
- With the correct hyperparams, we kinda track the optimum as learning progresses. So, 



