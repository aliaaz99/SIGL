# SIGL

To regenerate the results in Figure 2(a) for graphon $i$, run the following code:

```sh
python Main.py --n_graphon "i"
```

To get the results in Figure 2(b) and Figure 2(c), increase the `--offset` by desired value $M$ as follows:

```sh
python Main.py --n_graphon "i" --offset "M"
```

Figure 3 plots are saved in a folder with named as `--name` in the Plots folder.
