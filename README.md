# Introduciton

This repository contains the implementation of **Heterogeneous Graph Transformer (HGT)** for the **Multi-faceted Dynamic Flexible Job Shop Scheduling Problem (MDFJSSP)**. This work introduces a novel scheduling framework that integrates multiple types of dynamic events, including job arrivals, operation time fluctuations, and machine breakdowns, to better model the complexities of real-world manufacturing systems.  

Here are the main libraries and their versions that we used, but other versions of these libraries might also work with our scripts.

```text
python 3.9.19
pytorch 2.3.0
gymnasium 0.28.1
```
# Quik start
You can quickly start training by running the `run.py` script by running the following command.
```sh
python run.py \
--iterations 150 \
--epochs_per_iter 3 \
--case_regen_iter 5 \
--num_operation_max 400 \
--num_cases 10 \
--num_layers 4 \
--d_model 128 \
--d_hidden 256 \
--num_head 8 \
--d_kv 16 \
--weight_decay 1e-2 \
--max_grad_norm 1.0 \
--dropout 0.2 \
--lr 5e-5 \
--batch_size 64 \
--warmup_step 1000 \
--gamma_m 0.9 \
--clip 0.15 \
--dynamic_prop 0.3 \
--sigma_pres_time 1.0 \
--mu_lbd 0.01 \
--num_machine 5 \
--num_job 10 \
--data_name valid_d03s1f001_m5j10 \
--checkpoint_id d03s1f001_m5j10
```
Parameter description
```text
# evironment
--iterations: number of iterations
--epochs_per_iter: number of epochs per iteration
--case_regen_iter: number of iterations to regenerate cases
--num_operation_max: maximum number of operations
--num_cases: number of cases (trajectories)
--dynamic_prop: dynamic arriving jobs proportion
--sigma_pres_time: The degree of fluctuation in operation times.
--mu_lbd: mean of the exponential distribution for the job arrival rate
--num_machine: number of machines
--num_job: number of jobs
--data_name: name of the valid dataset
--checkpoint_id: checkpoint id to save the model
# HGT policy
--num_layers: number of the HGT layers
--d_model: dimension of the model
--d_hidden: dimension of the hidden layer
--num_head: number of heads
--d_kv: dimension of the key and value
# training parameters
--weight_decay: weight decay
--max_grad_norm: maximum gradient norm
--dropout: dropout rate
--lr: learning rate
--batch_size: batch size
--warmup_step: warmup step
# PPO
--gamma_m: gamma for the moving average
--clip: clip ratio

```

# Evaluation
To evaluate the trained model, you must specify the checkpoint ID and the dataset to be used for testing. Additionally, it is essential to ensure that the parameters of the policy (HGT) remain consistent with those used during training.
```shell
python eval.py \
--num_layers 4 \
--d_model 128 \
--d_hidden 256 \
--num_head 8 \
--d_kv 16 \
--seed 0 \
--data_name test_d04s1f0005_m5j10 \
--checkpoint_id default
```


