cd ../../

DEVICE="cpu"

N_ROUNDS=200
LOCAL_STEPS=5
LOG_FREQ=5

echo "Experiment with synthetic dataset"

echo "=> Generate data.."

cd data/ || exit

rm -r synthetic

python main.py \
  --dataset synthetic \
  --n_tasks 10 \
  --n_classes 10 \
  --n_samples 10000 \
  --dimension 60 \
  --alpha 1.0 \
  --beta 1.0 \
  --availability_parameter 0.4 \
  --stability_parameter 0.0 \
  --tasks_deterministic_split \
  --tasks_proportion 0.5 \
  --save_dir synthetic \
  --seed 123

cd ../


seeds="12 62 123 1234 12345 57 1453 1927 1956 2011 6010 1457 3111 1293 2051 4649 4067 2357 9939 1656 1483 6819 4224 9506 6879 671 3530 866 3799 7687 7547 3729 7127 5600 736 3212 5699 5567 5710 1749 66 4113 8766 997 6901 7155 739 7164 9581"

echo "==> Run experiment with 'ideal' sampler, 'gradients' aggregator"
sampler="ideal"
aggregator="gradients"
for seed in $seeds;
do
  for lr in 0.001 0.003 0.01 0.03 0.1
  do
    for server_lr in 1.0
    do
      echo "=> activity=${sampler},${aggregator} | lr=${lr} | server_lr=${server_lr} | seed=${seed}"
      python run_experiment.py \
        --experiment "synthetic_leaf" \
        --cfg_file_path data/synthetic/cfg.json \
        --objective_type weighted \
        --aggregator_type "${aggregator}" \
        --clients_sampler "${sampler}" \
        --n_rounds "${N_ROUNDS}" \
        --local_steps "${LOCAL_STEPS}" \
        --local_optimizer sgd \
        --local_lr "${lr}" \
        --server_optimizer sgd \
        --server_lr "${server_lr}" \
        --train_bz 128 \
        --test_bz 1024 \
        --device "${DEVICE}" \
        --log_freq "${LOG_FREQ}" \
        --verbose 0 \
        --logs_dir "logs_tuning/synthetic/activity_${sampler}_${aggregator}/synthetic_lr_${lr}_server_${server_lr}/seed_${seed}" \
        --history_path "history_tuning/synthetic/activity_${sampler}_${aggregator}/synthetic_lr_${lr}_server_${server_lr}/seed_${seed}.json" \
        --seed "${seed}"
    done
  done
done


echo "==> Run experiment with 'unbiased' sampler, 'gradients' aggregator"
sampler="unbiased"
aggregator="gradients"
for seed in $seeds;
do
  for lr in 0.001 0.003 0.01 0.03 0.1
  do
    for server_lr in 1.0
    do
      echo "=> activity=${sampler},${aggregator} | lr=${lr} | server_lr=${server_lr} | seed=${seed}"
      python run_experiment.py \
        --experiment "synthetic_leaf" \
        --cfg_file_path data/synthetic/cfg.json \
        --objective_type weighted \
        --aggregator_type "${aggregator}" \
        --clients_sampler "${sampler}" \
        --n_rounds "${N_ROUNDS}" \
        --local_steps "${LOCAL_STEPS}" \
        --local_optimizer sgd \
        --local_lr "${lr}" \
        --server_optimizer sgd \
        --server_lr "${server_lr}" \
        --train_bz 128 \
        --test_bz 1024 \
        --device "${DEVICE}" \
        --log_freq "${LOG_FREQ}" \
        --verbose 0 \
        --logs_dir "logs_tuning/synthetic/activity_${sampler}_${aggregator}/synthetic_lr_${lr}_server_${server_lr}/seed_${seed}" \
        --history_path "history_tuning/synthetic/activity_${sampler}_${aggregator}/synthetic_lr_${lr}_server_${server_lr}/seed_${seed}.json" \
        --seed "${seed}"
    done
  done
done



echo "==> Run experiment with 'biased' sampler, 'params' aggregator"
sampler="biased"
aggregator="params"
for seed in $seeds;
do
  for lr in 0.001 0.003 0.01 0.03 0.1
  do
    server_lr=1.0
    echo "=> activity=${sampler},${aggregator} | lr=${lr} | server_lr=${server_lr} | seed=${seed}"
    python run_experiment.py \
      --experiment "synthetic_leaf" \
      --cfg_file_path data/synthetic/cfg.json \
      --objective_type weighted \
      --aggregator_type "${aggregator}" \
      --clients_sampler "${sampler}" \
      --n_rounds "${N_ROUNDS}" \
      --local_steps "${LOCAL_STEPS}" \
      --local_optimizer sgd \
      --local_lr "${lr}" \
      --server_optimizer sgd \
      --server_lr "${server_lr}" \
      --train_bz 128 \
      --test_bz 1024 \
      --device "${DEVICE}" \
      --log_freq "${LOG_FREQ}" \
      --verbose 0 \
      --logs_dir "logs_tuning/synthetic/activity_${sampler}_${aggregator}/synthetic_lr_${lr}_server_${server_lr}/seed_${seed}" \
      --history_path "history_tuning/synthetic/activity_${sampler}_${aggregator}/synthetic_lr_${lr}_server_${server_lr}/seed_${seed}.json" \
      --seed "${seed}"
  done
done


echo "==> Run experiment with 'unbiased' sampler, 'params' aggregator"
sampler="unbiased"
aggregator="params"
for seed in $seeds;
do
  for lr in 0.001 0.003 0.01 0.03 0.1
  do
    server_lr=1.0
    echo "=> activity=${sampler},${aggregator} | lr=${lr} | server_lr=${server_lr} | seed=${seed}"
    python run_experiment.py \
      --experiment "synthetic_leaf" \
      --cfg_file_path data/synthetic/cfg.json \
      --objective_type weighted \
      --aggregator_type "${aggregator}" \
      --clients_sampler "${sampler}" \
      --n_rounds "${N_ROUNDS}" \
      --local_steps "${LOCAL_STEPS}" \
      --local_optimizer sgd \
      --local_lr "${lr}" \
      --server_optimizer sgd \
      --server_lr "${server_lr}" \
      --train_bz 128 \
      --test_bz 1024 \
      --device "${DEVICE}" \
      --log_freq "${LOG_FREQ}" \
      --verbose 0 \
      --logs_dir "logs_tuning/synthetic/activity_${sampler}_${aggregator}/synthetic_lr_${lr}_server_${server_lr}/seed_${seed}" \
      --history_path "history_tuning/synthetic/activity_${sampler}_${aggregator}/synthetic_lr_${lr}_server_${server_lr}/seed_${seed}.json" \
      --seed "${seed}"
  done
done