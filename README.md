# Code for Paper: *Deep Reinforcement Learning for Inverse Inorganic Materials Design*

Paper url: [Deep Reinforcement Learning for Inverse Inorganic Materials Design, NeurIPS AI4Mat Workshop (2022)](https://arxiv.org/abs/2210.11931)

Elton Pan, Christopher Karpovich, Elsa Olivetti

Department of Materials Science and Engineering, Massachusetts Institute of Technology, Cambridge, Massachusetts 02139, United States

A major obstacle to the realization of novel inorganic materials with desirable properties is the inability to perform efficient optimization across both materials properties and synthesis of those materials. In this work, we propose a reinforcement learning (RL) approach to inverse inorganic materials design, which can identify promising compounds with specified properties and synthesizability constraints. Our model learns chemical guidelines such as charge and electronegativity neutrality while maintaining chemical diversity and uniqueness. We demonstrate a multi-objective RL approach, which can generate novel compounds with targeted materials properties including formation energy and bulk/shear modulus alongside a lower sintering temperature synthesis objectives. Using this approach, the model can predict promising compounds of interest, while suggesting an optimized chemical design space for inorganic materials discovery.


![Alt text](/figures/dqn_overview.png "overview")

![Alt text](/figures/metrics.png "osda")

![Alt text](/figures/multi-objective.png "shap")

## Setup and installation

Run the following terminal commands 

1. Clone repo to local directory

```bash
  git clone https://github.com/eltonpan/InorganicMaterialRL.git
```

2. Set up and activate conda environment
```bash
  cd InorganicMaterialRL
```
```bash
  conda env create -f env.yml
```
```bash
  conda activate dqn
```

3. Add conda environment to Jupyter notebook
```bash
  conda install -c anaconda ipykernel
```
```bash
  python -m ipykernel install --user --name=dqn
```

4. Open jupyter notebooks
```bash
  jupyter notebook <notebook_name>.ipynb
```

make sure the `zeosyn` is the environment under dropdown menu `Kernel` > `Change kernel`

# Cite
If you use this dataset or code, please cite this paper:
```
@article{pan2022deep,
  title={Deep reinforcement learning for inverse inorganic materials design},
  author={Pan, Elton and Karpovich, Christopher and Olivetti, Elsa},
  journal={arXiv preprint arXiv:2210.11931},
  year={2022}
}
```