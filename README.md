# Dedipeak: Peak Forecasting Enhancement with Deblurring Diffusion Models
## Getting Started

First clone this container and `cd` into it.

### Native installation

**Requirements**  
- Python **3.6+**
- Pytorch **1.10+**

**Optional**  
- PyTorch installed with a GPU to run it faster ;)

You can create a virtual environment with conda:
```shell
conda create -n dedipeak python=3.8
```

Install dependencies with:
```shell
pip install -r requirements.txt
```

**Download** datasets and create the dataset directory


You can download the datasets from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/e1ccfff39ad541908bae/) or [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing) links provided by [Autoformer](https://github.com/thuml/Autoformer) repository. For more information, please visit the repository.

Put all of the downloaded datasets in a `dataset` folder in the current directory:

```
this repo
├── dataset
│   ├── weather
|   |   └── weather.csv
│   ├── traffic
|   |   └── traffic.csv
|   └── ...
└── data_provider
└── exp
└── ...
```

### Docker usage
If you don't want to install it natively and want an easy solution, you can use Docker.

First pull the Nvidia's PyTorch image:
```shell
docker pull docker pull nvcr.io/nvidia/pytorch:23.02-py3
```

If you want to run the container with GPU, you will need to setup Docker for it
by installing the `nvidia-container-runtime` and `nvidia-container-toolkit` packages. 

Then run the container from this directory directory:
```shell
docker run -it --rm --gpus all --name dedipeak -v $(pwd):/workspace nvcr.io/nvidia/pytorch:23.02-py3
```

Once in it, install the dependencies with:
```shell
pip install -r requirements.txt
```

And download the data if not already done.

### Every models, every datasets, all at once
To run all the experiments by specifying models, you can use the `train_models.sh` script:
```shell
./train_models.sh InverseHeatDissipation Autoformer Informer Reformer
```
This will run the three models on all the datasets (detailled below).


### Run one experiment
To run one experiment:
```shell
python run.py --data_path electricity.csv --model Autoformer --batch_size 16 --pred_len 720 --save_preds
```
By default, one experiment is run five times (`--itr` option).

### Models and Datasets

Here are the valid arguments for `--data_path` for now:
- `weather.csv`
- `traffic.csv`
- `electricity.csv`
- `ETTm1.csv`
- `ETTm2.csv`

ℹ️ You can add other datasets by precising the folder path in the `--root_path` argument.

Here are the valid arguments for `--model` for now:
- `Autoformer`
- `Informer`
- `Reformer`
- `FEDformer`
- `Performer`
- `NHits`
- `FiLM`
- `Seq2Seq`
- `EVT`

Losses:
- `mse`
- `mae`
- `huber`
- `adaptive`
- `dilate`
- `evl` (for EVT model only)


### Test the deblurring model
The Inverse Heat Dissipation model is always used upon another base model. To test it, you can use the `test_models.py` script:
```shell
./test_models.sh Autoformer Informer Reformer
```
:warning: This necessitates to have run the `train_models.sh` script before, on the same models **and** on `InverseHeatDissipation`, which is implicitly used for the deblurring process.

The script loads the base models and deblur their predictions afterwards. The results are saved in the `results` folder, which can then be loaded with numpy to be measured with `utils.metrics`.


## Acknowledgement

We acknowledge the following github repositories that made the base of our work:

https://github.com/BorealisAI/scaleformer  
https://github.com/thuml/Autoformer  
https://github.com/zhouhaoyi/Informer2020  
https://github.com/MAZiqing/FEDformer  
https://github.com/jonbarron/robust_loss_pytorch.git  
https://github.com/Nixtla/neuralforecast  
https://github.com/tianzhou2011/FiLM  
https://github.com/vincent-leguen/DILATE 
