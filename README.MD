# H2OFormer
This is the official implementation of our paper [ Hybrid-supervised Hypergraph-enhanced Transformer for Micro-gesture Based Emotion Recognition.]()

# Preparation
### Install torchlight
Run `pip install -e torchlight`

### Data Processing

From datasets create `.json` file, file_name -> `{sampleID}_{video_id}_{label_id}_{class}.json`
```
{sampleID}_{video_id}_{label_id}_{class}.json
|_{"file_name": "Sample0001_0_0_1", "skeletons": [list]}
```

"skeletons" is a **list** converted from an **array** of size `[window_size, point_nums, 3]`

`window_size` and `point_nums` consistent with the `.yaml` file in **config** directory.

**eg.**  SMG Trainsets:

```
./xxx/train
|_Sample0001_0_0_1.json
|_Sample0001_0_1_1.json
|
...
|
|_Sample0040_413_77_0.json
```

# Training & Testing

### Training

```python
python main.py --work_dir ./work_dir/SMG/SSI_SMG_e10d4 --config ./config/smg_ssi.yaml --device 0
```

### Testing

```
python eval.py --result_dir ./result_dir/SMG/SSI_SMG_e10d4 --weights_dir ./work_dir/SMG/SSI_SMG_e10d4
```


## Acknowledgements

This code is based on the repo of [Hyperformer](https://github.com/ZhouYuxuanYX/Hyperformer). 

Thanks to the original authors for their work!

## Citation

Coming soon...

