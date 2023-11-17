cd ../../

DEVICE="cuda"

N_ROUNDS=200
LOCAL_STEPS=5
LOG_FREQ=5

echo "Experiment with MNIST dataset"

echo "=> Generate data.."

cd data/ || exit

rm -r mnist

python main.py \
  --dataset mnist \
  --frac 0.1 \
  --n_tasks 10 \
  --by_labels_split \
  --n_components -1 \
  --alpha 0.3 \
  --availability_parameter 0.4 \
  --stability_parameter 0.0 \
  --tasks_deterministic_split \
  --tasks_proportion 0.5 \
  --save_dir mnist \
  --seed 123

cd ../

seeds="12 62 123 1234 12345 57 1453 1927 1956 2011"

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
        --experiment "mnist" \
        --cfg_file_path data/mnist/cfg.json \
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
        --logs_dir "logs_tuning/mnist/activity_${sampler}_${aggregator}/mnist_lr_${lr}_server_${server_lr}/seed_${seed}" \
        --history_path "history_tuning/mnist/activity_${sampler}_${aggregator}/mnist_lr_${lr}_server_${server_lr}/seed_${seed}.json" \
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
      --experiment "mnist" \
      --cfg_file_path data/mnist/cfg.json \
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
      --logs_dir "logs_tuning/mnist/activity_${sampler}_${aggregator}/mnist_lr_${lr}_server_${server_lr}/seed_${seed}" \
      --history_path "history_tuning/mnist/activity_${sampler}_${aggregator}/mnist_lr_${lr}_server_${server_lr}/seed_${seed}.json" \
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
      --experiment "mnist" \
      --cfg_file_path data/mnist/cfg.json \
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
      --logs_dir "logs_tuning/mnist/activity_${sampler}_${aggregator}/mnist_lr_${lr}_server_${server_lr}/seed_${seed}" \
      --history_path "history_tuning/mnist/activity_${sampler}_${aggregator}/mnist_lr_${lr}_server_${server_lr}/seed_${seed}.json" \
      --seed "${seed}"
  done
done


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
        --experiment "mnist" \
        --cfg_file_path data/mnist/cfg.json \
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
        --logs_dir "logs_tuning/mnist/activity_${sampler}_${aggregator}/mnist_lr_${lr}_server_${server_lr}/seed_${seed}" \
        --history_path "history_tuning/mnist/activity_${sampler}_${aggregator}/mnist_lr_${lr}_server_${server_lr}/seed_${seed}.json" \
        --seed "${seed}"
    done
  done
done
