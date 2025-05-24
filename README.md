# Negative-Dominant Contrastive Learning for Imbalanced Domain Generalization


## Sample

```bash
#  run experimental trials
python3 -u sweep_im.py launch --data_dir /path/your/dataset --dataset PACS --output_dir ./logs/TotalHeavyTail --algorithm NDCL --command_launcher local --type TotalHeavyTail
```

```python
# generate an sample for the TotalHeavyTail setting (idg_generate.py)
# to generate a new configuration, you may override the Generator class to implement custom control logic.
generator = TotalHeavyTail(num_valid, percent_test, c=heavytail_paramenter)
stats = main(dataset, num_valid, thred_many, thred_few, generator, file=filestream)
```


# Acknowledgement
[DomainBed](https://github.com/facebookresearch/DomainBed)  
[Multi-Domain Long-Tailed Recognition](https://github.com/YyzHarry/multi-domain-imbalance)
