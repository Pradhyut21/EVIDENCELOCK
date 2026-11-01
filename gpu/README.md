# Colab environment gate

The participant supplied a Tesla T4 runtime report on November 1: Python 3.13.16, torch 2.11.0+cu130, driver 580.82.07, CUDA 13.0, about 12 GiB RAM and 120.8 GB free disk, with a successful CUDA tensor calculation. This is participant-supplied evidence, not direct agent access to the remote runtime.

Use prepare_colab.py in that selected Colab session. It creates a separate bootstrap venv and **Python 3.12 training venv**; it does not install training packages into the notebook's global Python 3.13 environment. The full 270-package candidate graph and artifact hashes are retained, including AlignTune's declared TRL/vLLM dependencies. There is no --no-deps install, dependency exclusion or standalone-TRL substitution.

System build prerequisites for the declared PyICU dependency are installed in the ephemeral Colab VM. Package installation can take time. Exact command output is preserved in environment-install.log. The final check exercises FP16 CUDA arithmetic, small NF4 quantization kernels, Cohere2 config registration and the installed AlignTune factory import/signature. It does not load a full model or prove LoRA backward/optimizer compatibility.

The source of the exact published AlignTune 0.1.28 wheel was inspected before preparing this gate. Its factory accepts completion_only_loss/assistant_only_loss settings, and its TRL SFT pathway constructs/uses TRL internally. Actual runtime behavior still requires the installed API and a genuine smoke run; source inspection alone is not integration verification.

Only curated train and validation exports are included in the transfer bundle. The frozen test records, account credentials, weights and local virtual environments are excluded. No login, model download, training, external writes or paid resource provisioning occurs in this setup stage. Failures must be reported with their logs; do not bypass the required toolkit if an import or kernel check fails.

For the subsequent T4 smoke experiment use FP16 and a conservative supported attention implementation, with an assistant-preserving context budget (the pilot requires up to 1,195 tokens). Do not use the earlier 512-token planning default. No successful Colab dependency installation or model training is claimed until the user returns the executed results.
