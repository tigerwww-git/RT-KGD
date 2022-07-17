# RT-KGD
This repository contains codes for our paper *A Relation Transition Aware Knowledge-Grounded Dialogue Generation* (ISWC 2022)



## Files

- `./main.py` :  Our RT-KGD model.
- `./metric`: Automatic metric functions.
- `./utils/HGT.py`: The Heterogeneous Graph Transformer encoder used in our model.
- `./utils/utils.py`: Other useful tool functions.
- `./utils/get_kdconv.py`: Data preprocessing for Knowledge Graph Embedding. 



## Environment

```
python==3.8
pytorch==1.8.1
pytorch-lightning==1.5.10
pytorch-transformers==1.2.0
```

- The recommended way to install the required packages is using pip and the provided `requirements.txt` file. Create the environment by running the following command:

  ```
  pip install -r requirements.txt
  ```

  

## Download Dataset

- Download the data from https://github.com/thu-coai/KdConv
- By default, we expect the data to be stored in `./data`.



## KGE Preparation

- Clone the OpenKE-PyTorch branch:

  ```bash
  git clone -b OpenKE-PyTorch https://github.com/thunlp/OpenKE --depth 1
  ```

- Put `./utils/get_kdconv.py` and domain kg file (e.g.,`./data/film/kb_film.json`) to directory `OpenKE-Pytorch/benchmarks/`.

- Generate data required for model training:

  ```
  python OpenKE-Pytorch/benchmarks/get_kdconv.py
  ```

- Refer to example to modify the code and save as `OpenKE-Pytorch/examples/train_transr_kdconv.py`

- Train TransR and get embedding file `OpenKE-Pytorch/embed.vec`. You can put it in the `./data/domian` directory.



## Train and inference

- For training, run the following code:

  ```
  python main.py
  ```

  - Training result will be output to `./output`, the trained model will be save to `./save_model`.

- For inference,  run the following code:

  ```
  python main.py --test
  ```

  - Generated responses will be output to `./output`.

- When modifying configration, follow the above two codes with `-- parameter_name your_parameter_value`.
