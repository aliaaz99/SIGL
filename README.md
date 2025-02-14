# SIGL

### Reproducing Results

First, you will need to install the requiered packages using `pip` as follows.

```sh
pip install -r requirements.txt
```

To regenerate the results shown in **Figure 2(a)** for a specific graphon `i`, use the following command:

```bash
python Main.py --n_graphon i
```

For the results shown in **Figure 2(b)** and **Figure 2(c)**, adjust the `--offset` by your desired value `M` as follows:

```bash
python Main.py --n_graphon i --offset M
```

The plots for **Figure 3** are saved in a folder named according to the `--name` argument, located in the `Plots` directory.

As an example `Graphon_1` is included.

You can use `p-Main.py` to reproduce the results for the parametric case explained in Appendix B.

### Learning graphon of an arbitrary set of graphs

To learn the graphon from a given set of graphs, save a list of adjacency matrices as a `.pkl` file and specify its directory using the `--Adjs_dir` parameter in the input. Since the true graphon is not available in this case, we cannot directly evaluate the reconstruction. However, the estimated graphon, along with the latent variables of the nodes and the sorted graph, will be plotted and saved as `output.jpg`.
