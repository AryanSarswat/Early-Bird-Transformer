To Reproduce The Experiments

Run the following command

```bash
python3 main.py
```

By Default the code will run CIFAR10 dataset with Vision Transformer model. To run other models and datasets, to run other datasets and models, you can change the parameters in the main.py file.

Replace the following line in the main.py file to run other models and datasets.

```python
get_cifar10_dataloader()
```

with

```python
get_cifar100_dataloader()
```

or 

```python
get_shvn_dataloader()
```