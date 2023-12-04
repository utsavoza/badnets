# BadNets

The effectiveness of backdoor attacks on Deep Neural Networks (DNNs) suggests that these networks have spare learning
capacity. Essentially, the DNN is capable of learning to respond incorrectly to inputs containing a backdoor, while
maintaining correct responses to clean inputs. This phenomenon involves certain neurons within the network,
referred to as "backdoor neurons," which are subtly hijacked by the attack to detect backdoors and trigger misbehavior.

In this lab, we evaluate a defense technique, where a defender might be able to disable a backdoor by removing
such neurons that are dormant for clean inputs. We refer to this strategy as the *pruning defense*.

## Usage

1. Clone the repository
      ```bash
      git clone git@github.com:utsavoza/badnets.git
      cd badnets
      ```

2. Download the validation and test datasets from [here](https://drive.google.com/drive/folders/1Rs68uH8Xqa4j6UxG53wzD0uyI8347dSq?usp=sharing) and place them under `data/` directory (See the [repo layout section](#repository-layout) for more details).

3. Create and activate the virtual environment
      ```bash
      python -m venv venv
      source venv/bin/activate
      ```

4. Install the required dependencies
      ```bash
      pip install -r requirements.txt
      ```

5. Execute `main.py` to reproduce the results, generate plots, etc.
      ```bash
      python -u main.py > out/logs.out
      ```

## Data

1. Download the validation and test datasets
   from [here](https://drive.google.com/drive/folders/1Rs68uH8Xqa4j6UxG53wzD0uyI8347dSq?usp=sharing) and store them
   under `data/` directory.
2. The dataset contains images from YouTube Aligned Face Dataset. We retrieve 1283 individuals and split into validation
   and test datasets.
3. `bd_valid.h5` and `bd_test.h5` contains validation and test images with sunglasses trigger respectively, that activates
   the backdoor for `bd_net.h5`.


## Evaluating the Models

1. The DNN architecture used to train the face recognition model is the state-of-the-art DeepID network.
2. To evaluate the backdoored model on **clean test dataset**, execute `eval.py` by running:
   ```bash
   python eval.py <clean validation data directory> <poisoned validation data directory> <model directory>`
   ```
   For example:
   ```bash
   python eval.py data/cl/test.h5 data/bd/bd_test.h5 models/pruned_net_10.h5
   ```

## Report

See [REPORT.md](./REPORT.md)

## Repository Layout

```
├── data
    └── cl
        └── valid.h5 // this is clean validation data used to design the defense
        └── test.h5  // this is clean test data used to evaluate the BadNet
    └── bd
        └── bd_valid.h5 // this is sunglasses poisoned validation data
        └── bd_test.h5  // this is sunglasses poisoned test data
├── out
    └── pruned_nets.out // these are the execution logs
├── plots
    └── pruned_model.png // plots for validation accuracy and asr
├── models
    └── bd_net.h5
    └── bd_weights.h5
    └── pruned_net_2.h5  // pruned badnet with acc. drop threshold = 2%
    └── pruned_net_4.h5  // pruned badnet with acc. drop threshold = 4%
    └── pruned_net_10.h5 // pruned badnet with acc. drop threshold = 10%
    └── pruned_net.h5
├── architecture.py
└── eval.py     // this is the evaluation script
└── REPORT.md   // includes results, plots and tables
```

## LICENSE
This project is licensed under MIT License. See [LICENSE](./LICENSE).
