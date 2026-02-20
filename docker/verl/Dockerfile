FROM hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0

WORKDIR /app

# Prep GSM8K dataset at build time
COPY verl-config/prep_data.py /app/prep_data.py
RUN python3 /app/prep_data.py --local_save_dir /data/gsm8k

# Copy reward function and training script
COPY verl-config/reward.py /app/reward.py
COPY verl-config/run_grpo.sh /app/run_grpo.sh
RUN chmod +x /app/run_grpo.sh
