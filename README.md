# SIGL

### Reproducing Results

To regenerate the results shown in **Figure 2(a)** for a specific graphon `i`, use the following command:

```bash
python Main.py --n_graphon i

For the results shown in **Figure 2(b)** and **Figure 2(c)**, adjust the `--offset` by your desired value `M` as follows:

```bash
python Main.py --n_graphon i --offset M

To get the results in Figure 2(b) and Figure 2(c), increase the `--offset` by desired value $M$ as follows:

```sh
python Main.py --n_graphon "i" --offset "M"
```

The plots for **Figure 3** are saved in a folder named according to the `--name` argument, located in the `Plots` directory.

