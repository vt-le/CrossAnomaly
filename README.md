# CrossAnomaly
This is the code for **[Reconstruction-Guided Cross-Modal Learning for Video Anomaly Detection]()**.
 
## Related Works
> **DualAnomaly**: See [DualAnomaly: Dual Spatio-Temporal Cross-Attention Network for Robust Video Anomaly Detection](https://github.com/vt-le/DualAnomaly).

> **HSTforU**: See [HSTforU: Anomaly Detection in Aerial and Ground-based Videos with Hierarchical Spatio-Temporal Transformer for U-net](https://github.com/vt-le/HSTforU).

> **ASTNet**: See [Attention-based Residual Autoencoder for Video Anomaly Detection](https://vt-le.github.io/astnet/).

## Setup
The code can be run under any environment with Python 3.7 and above.
(It may run with lower versions, but we have not tested it).

Install the required packages:

    pip install -r requirements.txt
  
Clone this repo:

    git clone https://github.com/vt-le/CrossAnomaly.git
    cd CrossAnomaly/

We evaluate `CrossAnomaly` on:
| Dataset | Link                                                                                  |
|--|---------------------------------------------------------------------------------------|
| UCSD Ped2 | [![Google drive](https://badgen.net/static/Homepage/Ped2/blue)](http://www.svcl.ucsd.edu/projects/anomaly/dataset.html) |
| CUHK Avenue | [![Google drive](https://badgen.net/badge/Homepage/Avenue/cyan)](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html) |
| ShanghaiTech | [![Google drive](https://badgen.net/badge/Homepage/ShanghaiTech/green?)](https://svip-lab.github.io/dataset/campus_dataset.html) |

## Training
To train `CrosssAnomaly` on a dataset, run:
```bash
 python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345  train.py --cfg <config-file>
```  

## Evaluation
Please first download the pre-trained model

| Dataset | Pretrained Model                                                                                  |
|--|---------------------------------------------------------------------------------------|
| UCSD Ped2 | [![Google drive](https://badgen.net/static/Link/Ped2/blue?icon=chrome)](https://drive.google.com) |
| CUHK Avenue | [![Google drive](https://badgen.net/badge/Link/Avenue/blue?icon=chrome)](https://drive.google.com) |
| ShanghaiTech | [![Google drive](https://badgen.net/badge/Link/ShanghaiTech/blue?icon=chrome)](https://drive.google.com) |

To evaluate a pretrained `CrossAnomaly` on a dataset, run:

```bash
 python test.py \
    --cfg <path/to/config/file> \
    --pretrained </path/to/pre-trained/model> \
    [--batch-size <batch-size> --tag <job-tag>]
```      
 
 For example, to evaluate `CrossAnomaly` on Ped2:

```bash
python test.py \
    --cfg config/scripts/ped2/ped2_pvt2_hst.yaml \
    --model-file path/to/checkpoint/ckpt_ped2.pth
```
<!-- 
## Training from scratch
To train `CrossAnomaly` on a dataset, run:
```bash
python -m torch.distributed.launch \
    --nproc_per_node <num-of-gpus-to-use> \
    --master_port 12345  main.py \ 
    --cfg <path/to/config/file> \
    [--batch-size <batch-size-per-gpu> --tag <job-tag>]
```

For example, to train `CrossAnomaly` on Ped2:

```bash
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 train.py --cfg path/to/checkpoint/crossanomaly_ped2.yaml 
``` 
-->
## Configuration
 * We use [YAML](https://yaml.org/) for configuration.
 * We provide a couple preset configurations.
 * Please refer to `config.py` for documentation on what each configuration does.

## Citing
If you find our work useful, please consider citing:
```BibTeX
﻿@Article{le2026crossanomaly,
author={Le, Viet-Tuan
and Kim, Yong-Guk},
title={Reconstruction-Guided Cross-Modal Learning for Video Anomaly Detection},
}
```

## Contact
For any question, please file an [issue](https://github.com/vt-le/CrossAnomaly/issues) or contact:

    Viet-Tuan Le: vt-le@outlook.com

