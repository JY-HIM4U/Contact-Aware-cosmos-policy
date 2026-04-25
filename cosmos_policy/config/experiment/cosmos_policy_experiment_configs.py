# -----------------------------------------------------------------------------
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

import os

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy._src.imaginaire.utils import log
from cosmos_policy._src.imaginaire.utils.checkpoint_db import get_checkpoint_path  # noqa: F401
from cosmos_policy.datasets.aloha_dataset import ALOHADataset
from cosmos_policy.datasets.libero_dataset import LIBERODataset
from cosmos_policy.datasets.robocasa_dataset import RoboCasaDataset
from cosmos_policy.models.policy_video2world_model import CosmosPolicyVideo2WorldModel
from cosmos_policy.modules.hybrid_edm_sde import HybridEDMSDE

cs = ConfigStore.instance()
val_sampling_size_override = dict(
    video_length=121,
    video_height=704,
    video_width=1280,
)
BASE_DATASETS_DIR = os.environ.get("BASE_DATASETS_DIR", ".")


# *** Main checkpoint ***
libero_all_4_suites_dataset = L(LIBERODataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only"),  # Successful demos
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only", "t5_embeddings.pkl"
    ),
    chunk_size=16,
    use_image_aug=True,
    use_wrist_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    use_stronger_image_aug=True,
    rollout_data_dir=os.path.join(
        BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "all_episodes"
    ),  # All demo rollouts (successes + failures)
    demonstration_sampling_prob=0.5,
    success_rollout_sampling_prob=0.5,
    return_value_function_returns=True,
    gamma=0.99,
)
cosmos_predict2_2b_480p_libero = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-Index-102-Size-2B-Res-480-Fps-16-Note-HQ_V5_from_26",
            {"override /data_train": "mock"},
            {"override /model": "policy_fsdp"},
            {"override /tokenizer": "policy_wan2pt1_tokenizer"},
            {
                "override /callbacks": [
                    "basic",
                    "long",
                    "cluster_speed",
                    "wandb",
                    "wandb_callback_actions",
                ]
            },
            "_self_",
        ],
        trainer=dict(
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=100000,
                    save_s3=False,
                    use_negative_prompt=False,
                    guidance=[0],
                    num_sampling_step=9,
                ),
            ),
            run_validation=False,
            logging_iter=5,
            max_iter=1000000,
            straggler_detection=dict(
                enabled=False,
            ),
        ),
        optimizer=dict(
            lr=1e-4,
        ),
        scheduler=dict(
            # LR decay for 30K steps in cycle #1, then decay by 5x and stay constant forever in cycle #2
            cycle_lengths=[30000, 100000000000000],
            warm_up_steps=[1000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                conditioner=dict(
                    text=dict(
                        # IMPORTANT: We don't want any text dropout; otherwise, the model may fail to follow language
                        dropout_rate=0.0,
                    ),
                ),
                state_t=9,  # Latent temporal dim (blank, proprio, wrist, primary, action, future proprio, future wrist, future primary, value)
                min_num_conditional_frames=4,  # 1 blank, 3 conditioning (proprio, wrist, primary)
                max_num_conditional_frames=4,  # 1 blank, 3 conditioning (proprio, wrist, primary)
                sigma_conditional=0.0,  # No noise on conditional latents
                conditioning_strategy="frame_replace",
                denoise_replace_gt_frames=True,
                tokenizer=dict(
                    chunk_duration=33,  # 1 blank + 32 images (4 proprio, 4 wrist image, 4 primary image, 4 action, 4 future proprio, 4 future wrist, 4 future primary, 4 value)
                ),
                ema=dict(
                    enabled=False,
                ),
                input_data_key="video",
                sde=L(HybridEDMSDE)(
                    hybrid_sigma_distribution=True,
                    p_mean=1.3862943611198906,  # Copied from base model config
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                    uniform_lower=1.0,
                    uniform_upper=85.0,
                ),
                adjust_video_noise=True,
                resize_online=True,
                resolution="224",
                high_sigma_strategy="none",
            ),
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        checkpoint=dict(
            load_path=get_checkpoint_path("hf://nvidia/Cosmos-Policy-LIBERO-Predict2-2B/Cosmos-Policy-LIBERO-Predict2-2B.pt"),
            load_training_state=False,  # This means do not load train state from the base checkpoint above (load_path); but when resuming this job, will load train state
            strict_resume=False,
            save_iter=1000,
            load_ema_to_reg=True,
            load_from_object_store=dict(
                enabled=False,
            ),
            save_to_object_store=dict(
                enabled=False,
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=libero_all_4_suites_dataset,
            sampler=L(DistributedSampler)(
                dataset=libero_all_4_suites_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=30,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_libero",
        ),
        upload_reproducible_setup=False,
    )
)
# Inference version
cosmos_predict2_2b_480p_libero__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                )
            )
        ),
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_libero__inference_only",
        ),
    )
)


# *** Main checkpoint ***
robocasa_50_demos_per_task_dataset = L(RoboCasaDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only"),  # Successful demos
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only", "t5_embeddings.pkl"
    ),
    chunk_size=32,
    use_image_aug=True,
    use_wrist_images=True,
    use_third_person_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    use_stronger_image_aug=True,
    rollout_data_dir=os.path.join(
        BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "all_episodes"
    ),  # All demo rollouts (successes + failures)
    demonstration_sampling_prob=0.5,
    success_rollout_sampling_prob=0.5,
    return_value_function_returns=True,
    gamma=0.99,
)
cosmos_predict2_2b_480p_robocasa_50_demos_per_task = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=11,  # Latent temporal dim (blank, proprio, wrist image, primary image, secondary image, action, future proprio, future wrist image, future primary image, future secondary image, value)
                min_num_conditional_frames=5,  # 1 blank, 4 conditioning (proprio, wrist image, primary image, secondary image)
                max_num_conditional_frames=5,  # 1 blank, 4 conditioning (proprio, wrist image, primary image, secondary image)
                tokenizer=dict(
                    chunk_duration=41,  # 1 blank + 40 images (4 proprio, 4 wrist image, 4 primary image, 4 secondary image, 4 action, 4 future proprio, 4 future wrist, 4 future primary, 4 future secondary, 4 value)
                ),
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            dataset=robocasa_50_demos_per_task_dataset,
            sampler=L(DistributedSampler)(
                dataset=robocasa_50_demos_per_task_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robocasa_50_demos_per_task",
        ),
    )
)
# Inference version
cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robocasa_50_demos_per_task",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                )
            )
        ),
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference",
        ),
    )
)


# *** Main checkpoint ***
aloha_cosmos_policy_dataset_185_demos = L(ALOHADataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "ALOHA-Cosmos-Policy", "preprocessed"),
    t5_text_embeddings_path=os.path.join(BASE_DATASETS_DIR, "ALOHA-Cosmos-Policy", "preprocessed", "t5_embeddings.pkl"),
    chunk_size=50,
    use_image_aug=True,
    use_stronger_image_aug=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    treat_demos_as_success_rollouts=True,  # Include demos as success rollouts
    demonstration_sampling_prob=0.5,
    success_rollout_sampling_prob=0.5,
    return_value_function_returns=True,
    gamma=0.998,  # Higher gamma for ALOHA because episodes can have up to 1.5-2.0K steps  # (s, a, s', v)
)
cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        scheduler=dict(
            # LR decay for 20K steps in cycle #1, then decay by 5x and stay constant forever in cycle #2
            cycle_lengths=[20000, 100000000000000],
            warm_up_steps=[2000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=11,  # Latent temporal dim (blank, proprio, left wrist, right wrist, primary, action, future proprio, future left wrist, future right wrist, future primary, value)
                min_num_conditional_frames=5,  # 1 blank, 4 conditioning (proprio, left wrist, right wrist, primary)
                max_num_conditional_frames=5,  # 1 blank, 4 conditioning (proprio, left wrist, right wrist, primary)
                tokenizer=dict(
                    chunk_duration=41,  # 1 blank + 40 images (4 proprio, 4 left wrist image, 4 right wrist image, 4 primary image, 4 action, 4 future proprio, 4 future left wrist, 4 future right wrist, 4 future primary, 4 value)
                ),
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=aloha_cosmos_policy_dataset_185_demos,
            sampler=L(DistributedSampler)(
                dataset=aloha_cosmos_policy_dataset_185_demos,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80",
        ),
    )
)
# Inference version
cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                )
            )
        ),
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__inference_only",
        ),
    )
)


# ALOHA planning model
# Dataset: 648 rollouts from evaluations with Cosmos Policy, pi05, pi0, OpenVLA-OFT+, Diffusion Policy
# NOTE: This rollouts dataset is not released; you will need to replace `rollout_data_dir` below with your own rollouts dataset
aloha_2025_09_18__648_rollouts__cosmos_policy__pi05__pi0__openvla_oft__diffusion_policy__dataset = L(
    ALOHADataset
)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "ALOHA-Cosmos-Policy", "preprocessed"),
    t5_text_embeddings_path=os.path.join(BASE_DATASETS_DIR, "ALOHA-Cosmos-Policy", "preprocessed", "t5_embeddings.pkl"),
    chunk_size=50,
    use_image_aug=True,
    use_stronger_image_aug=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    treat_demos_as_success_rollouts=False,  # Don't include demos as success rollouts because they have a fixed episode length + we want to focus on real policy rollouts
    demonstration_sampling_prob=0.1,  # Smaller demonstration sampling prob - more emphasis on rollouts
    success_rollout_sampling_prob=0.5,
    return_value_function_returns=True,
    gamma=0.998,  # Higher gamma for ALOHA because episodes can have up to 1.5-2.0K steps  # (s, a, s', v)
    rollout_data_dir=os.path.join(BASE_DATASETS_DIR, "PATH/TO/YOUR/ROLLOUTS/DATASET"),  # JPEG images
    use_jpeg_for_rollouts=True,  # JPEG images
)
cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80",
            "_self_",
        ],
        checkpoint=dict(
            # Resume from 50K checkpoint of base Cosmos Policy run
            load_path=get_checkpoint_path(
                "hf://nvidia/Cosmos-Policy-ALOHA-Predict2-2B/Cosmos-Policy-ALOHA-Predict2-2B.pt"
            ),
        ),
        scheduler=dict(
            # LR decay for 15K steps in cycle #1, then decay by 5x and stay constant forever in cycle #2
            cycle_lengths=[15000, 100000000000000],
            warm_up_steps=[1500, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=aloha_2025_09_18__648_rollouts__cosmos_policy__pi05__pi0__openvla_oft__diffusion_policy__dataset,
            sampler=L(DistributedSampler)(
                dataset=aloha_2025_09_18__648_rollouts__cosmos_policy__pi05__pi0__openvla_oft__diffusion_policy__dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                mask_current_state_action_for_value_prediction=True,  # Use input masking to mask out irrelevant inputs (current state and action) during value prediction
            ),
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func",
        ),
    )
)
# Inference version
cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                )
            )
        ),
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func__inference_only",
        ),
    )
)


# ── Contact-Aware smoke-test experiments (M1b / M2) ──────────────────────────
# V baseline: LIBERO checkpoint + LoRA, no F/T, 100-step smoke test.
libero_spatial_dataset_smoke = L(LIBERODataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only"),
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only", "t5_embeddings.pkl"
    ),
    chunk_size=16,
    use_image_aug=False,
    use_wrist_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    use_stronger_image_aug=False,
    rollout_data_dir="",
    demonstration_sampling_prob=1.0,
    return_value_function_returns=True,
    gamma=0.99,
)
cosmos_predict2_2b_480p_libero_smoke_v = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(
            max_iter=100,
            logging_iter=5,
            run_validation=False,
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                compile_tokenizer=dict(enabled=False),
            ),
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                use_lora=True,
                lora_rank=8,
            )
        ),
        dataloader_train=L(DataLoader)(
            num_workers=2,
            persistent_workers=False,
            pin_memory=True,
            dataset=libero_spatial_dataset_smoke,
            batch_size=1,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_contact_aware",
            name="cosmos_predict2_2b_480p_libero_smoke_v",
        ),
        upload_reproducible_setup=False,
    )
)

# V+F variant: same but with F/T injection (state_t=11) for smoke test.
libero_spatial_dataset_smoke_vf = L(LIBERODataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only"),
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only", "t5_embeddings.pkl"
    ),
    chunk_size=16,
    use_image_aug=False,
    use_wrist_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    use_stronger_image_aug=False,
    rollout_data_dir="",
    demonstration_sampling_prob=1.0,
    return_value_function_returns=True,
    gamma=0.99,
    use_ft=True,
    ft_data_dir=os.path.join(BASE_DATASETS_DIR, "libero_spatial_with_ft"),
    ft_stats_path=os.path.join(BASE_DATASETS_DIR, "libero_spatial_with_ft", "dataset_stats_all.json"),
)
cosmos_predict2_2b_480p_libero_smoke_vf = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(
            max_iter=100,
            logging_iter=5,
            run_validation=False,
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                compile_tokenizer=dict(enabled=False),
            ),
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=11,
                min_num_conditional_frames=5,  # blank + proprio + ft + wrist + primary
                max_num_conditional_frames=5,
                tokenizer=dict(chunk_duration=41),
                use_lora=True,
                lora_rank=8,
            )
        ),
        dataloader_train=L(DataLoader)(
            num_workers=2,
            persistent_workers=False,
            pin_memory=True,
            dataset=libero_spatial_dataset_smoke_vf,
            batch_size=1,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_contact_aware",
            name="cosmos_predict2_2b_480p_libero_smoke_vf",
        ),
        upload_reproducible_setup=False,
    )
)



# ── M2: LIBERO-Spatial full LoRA training (V and V+F) ────────────────────────
# 5 000 steps ≈ ~2 epochs over the 450-demo spatial dataset; checkpoints at 1k.
# data_dir points directly to libero_spatial_regen/ to avoid picking up HDF5s
# from other suites that may be present in the parent success_only/ directory.
libero_spatial_dataset_m2 = L(LIBERODataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only", "libero_spatial_regen"),
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only", "t5_embeddings.pkl"
    ),
    chunk_size=16,
    use_image_aug=False,
    use_wrist_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    use_stronger_image_aug=False,
    rollout_data_dir="",
    demonstration_sampling_prob=1.0,
    return_value_function_returns=True,
    gamma=0.99,
)

libero_spatial_dataset_m2_vf = L(LIBERODataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only", "libero_spatial_regen"),
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only", "t5_embeddings.pkl"
    ),
    chunk_size=16,
    use_image_aug=False,
    use_wrist_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    use_stronger_image_aug=False,
    rollout_data_dir="",
    demonstration_sampling_prob=1.0,
    return_value_function_returns=True,
    gamma=0.99,
    use_ft=True,
    ft_data_dir=os.path.join(BASE_DATASETS_DIR, "libero_spatial_with_ft"),
    ft_stats_path=os.path.join(BASE_DATASETS_DIR, "libero_spatial_with_ft", "dataset_stats_all.json"),
)

# V baseline: no F/T
cosmos_predict2_2b_480p_libero_m2_v = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(
            max_iter=5000,
            logging_iter=50,
            run_validation=False,
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                compile_tokenizer=dict(enabled=False),
            ),
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                use_lora=True,
                lora_rank=8,
            )
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=libero_spatial_dataset_m2,
            batch_size=1,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_contact_aware",
            name="cosmos_predict2_2b_480p_libero_m2_v",
        ),
        upload_reproducible_setup=False,
    )
)

libero_spatial_dataset_m2_vf_v2 = L(LIBERODataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only", "libero_spatial_regen"),
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only", "t5_embeddings.pkl"
    ),
    chunk_size=16,
    use_image_aug=False,
    use_wrist_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    use_stronger_image_aug=False,
    rollout_data_dir="",
    demonstration_sampling_prob=1.0,
    return_value_function_returns=True,
    gamma=0.99,
    use_ft=True,
    ft_data_dir=os.path.join(BASE_DATASETS_DIR, "libero_spatial_with_ft"),
    ft_stats_path=os.path.join(BASE_DATASETS_DIR, "libero_spatial_with_ft", "dataset_stats_p1p99.json"),
)

# V+F: real F/T from libero_spatial_with_ft
cosmos_predict2_2b_480p_libero_m2_vf = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(
            max_iter=5000,
            logging_iter=50,
            run_validation=False,
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                compile_tokenizer=dict(enabled=False),
            ),
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=11,
                min_num_conditional_frames=5,
                max_num_conditional_frames=5,
                tokenizer=dict(chunk_duration=41),
                use_lora=True,
                lora_rank=8,
            )
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=libero_spatial_dataset_m2_vf,
            batch_size=1,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_contact_aware",
            name="cosmos_predict2_2b_480p_libero_m2_vf",
        ),
        upload_reproducible_setup=False,
    )
)


# V+F v2: p1/p99 F/T normalization, 50k steps
cosmos_predict2_2b_480p_libero_m2_vf_v2 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(
            max_iter=50000,
            logging_iter=50,
            run_validation=False,
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                compile_tokenizer=dict(enabled=False),
            ),
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=11,
                min_num_conditional_frames=5,
                max_num_conditional_frames=5,
                tokenizer=dict(chunk_duration=41),
                use_lora=True,
                lora_rank=8,
            )
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=libero_spatial_dataset_m2_vf_v2,
            batch_size=1,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_contact_aware",
            name="cosmos_predict2_2b_480p_libero_m2_vf_v2",
        ),
        upload_reproducible_setup=False,
    )
)

libero_10_dataset_m2_vf = L(LIBERODataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only", "libero_10_regen"),
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only", "t5_embeddings.pkl"
    ),
    chunk_size=16,
    use_image_aug=False,
    use_wrist_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    use_stronger_image_aug=False,
    rollout_data_dir="",
    demonstration_sampling_prob=1.0,
    return_value_function_returns=True,
    gamma=0.99,
    use_ft=True,
    ft_data_dir=os.path.join(BASE_DATASETS_DIR, "libero_10_with_ft"),
    ft_stats_path=os.path.join(BASE_DATASETS_DIR, "libero_10_with_ft", "dataset_stats_p1p99.json"),
)

libero_10_dataset_m2_vf_filtered = L(LIBERODataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only", "libero_10_regen"),
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only", "t5_embeddings.pkl"
    ),
    chunk_size=16,
    use_image_aug=False,
    use_wrist_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    use_stronger_image_aug=False,
    rollout_data_dir="",
    demonstration_sampling_prob=1.0,
    return_value_function_returns=True,
    gamma=0.99,
    use_ft=True,
    ft_data_dir=os.path.join(BASE_DATASETS_DIR, "libero_10_with_ft_filtered"),
    ft_stats_path=os.path.join(BASE_DATASETS_DIR, "libero_10_with_ft_filtered", "dataset_stats_p1p99.json"),
)

cosmos_predict2_2b_480p_libero10_m2_vf_filtered = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(
            max_iter=25000,
            logging_iter=50,
            run_validation=False,
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                compile_tokenizer=dict(enabled=False),
            ),
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=11,
                min_num_conditional_frames=5,
                max_num_conditional_frames=5,
                tokenizer=dict(chunk_duration=41),
                use_lora=True,
                lora_rank=8,
            )
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=libero_10_dataset_m2_vf_filtered,
            batch_size=4,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_contact_aware",
            name="cosmos_predict2_2b_480p_libero10_m2_vf_filtered",
        ),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2_2b_480p_libero10_m2_vf_filtered__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=11,
                min_num_conditional_frames=5,
                max_num_conditional_frames=5,
                tokenizer=dict(chunk_duration=41),
                use_lora=True,
                lora_rank=8,
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                ),
            )
        ),
        job=dict(
            group="cosmos_v2_contact_aware",
            name="cosmos_predict2_2b_480p_libero10_m2_vf_filtered__inference",
        ),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2_2b_480p_libero10_m2_vf = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(
            max_iter=25000,
            logging_iter=50,
            run_validation=False,
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                compile_tokenizer=dict(enabled=False),
            ),
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=11,
                min_num_conditional_frames=5,
                max_num_conditional_frames=5,
                tokenizer=dict(chunk_duration=41),
                use_lora=True,
                lora_rank=8,
            )
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=libero_10_dataset_m2_vf,
            batch_size=4,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_contact_aware",
            name="cosmos_predict2_2b_480p_libero10_m2_vf",
        ),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2_2b_480p_libero10_m2_vf__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=11,
                min_num_conditional_frames=5,
                max_num_conditional_frames=5,
                tokenizer=dict(chunk_duration=41),
                use_lora=True,
                lora_rank=8,
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                ),
            )
        ),
        job=dict(
            group="cosmos_v2_contact_aware",
            name="cosmos_predict2_2b_480p_libero10_m2_vf__inference",
        ),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2_2b_480p_libero_m2_vf_v4 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(
            max_iter=25000,
            logging_iter=50,
            run_validation=False,
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                compile_tokenizer=dict(enabled=False),
            ),
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=11,
                min_num_conditional_frames=5,
                max_num_conditional_frames=5,
                tokenizer=dict(chunk_duration=41),
                use_lora=True,
                lora_rank=8,
            )
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=libero_spatial_dataset_m2_vf_v2,
            batch_size=4,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_contact_aware",
            name="cosmos_predict2_2b_480p_libero_m2_vf_v4",
        ),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2_2b_480p_libero_m2_vf_v4__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=11,
                min_num_conditional_frames=5,
                max_num_conditional_frames=5,
                tokenizer=dict(chunk_duration=41),
                use_lora=True,
                lora_rank=8,
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                ),
            )
        ),
        job=dict(
            group="cosmos_v2_contact_aware",
            name="cosmos_predict2_2b_480p_libero_m2_vf_v4__inference",
        ),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2_2b_480p_libero_m2_vf_v3 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(
            max_iter=25000,
            logging_iter=50,
            run_validation=False,
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                compile_tokenizer=dict(enabled=False),
            ),
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=11,
                min_num_conditional_frames=5,
                max_num_conditional_frames=5,
                tokenizer=dict(chunk_duration=41),
                use_lora=True,
                lora_rank=8,
                use_ft_encoder=True,
            )
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=libero_spatial_dataset_m2_vf_v2,
            batch_size=4,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_contact_aware",
            name="cosmos_predict2_2b_480p_libero_m2_vf_v3",
        ),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2_2b_480p_libero_m2_vf_v3__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=11,
                min_num_conditional_frames=5,
                max_num_conditional_frames=5,
                tokenizer=dict(chunk_duration=41),
                use_lora=True,
                lora_rank=8,
                use_ft_encoder=True,
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                ),
            )
        ),
        job=dict(
            group="cosmos_v2_contact_aware",
            name="cosmos_predict2_2b_480p_libero_m2_vf_v3__inference",
        ),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2_2b_480p_libero_m2_vf_v2__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=11,
                min_num_conditional_frames=5,
                max_num_conditional_frames=5,
                tokenizer=dict(chunk_duration=41),
                use_lora=True,
                lora_rank=8,
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                ),
            )
        ),
        job=dict(
            group="cosmos_v2_contact_aware",
            name="cosmos_predict2_2b_480p_libero_m2_vf_v2__inference",
        ),
        upload_reproducible_setup=False,
    )
)

# Inference configs for M2 fine-tuned checkpoints
cosmos_predict2_2b_480p_libero_m2_v__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                use_lora=True,
                lora_rank=8,
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                ),
            )
        ),
        job=dict(
            group="cosmos_v2_contact_aware",
            name="cosmos_predict2_2b_480p_libero_m2_v__inference",
        ),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2_2b_480p_libero_m2_vf__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=11,
                min_num_conditional_frames=5,
                max_num_conditional_frames=5,
                tokenizer=dict(chunk_duration=41),
                use_lora=True,
                lora_rank=8,
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                ),
            )
        ),
        job=dict(
            group="cosmos_v2_contact_aware",
            name="cosmos_predict2_2b_480p_libero_m2_vf__inference",
        ),
        upload_reproducible_setup=False,
    )
)


# ── Single-task LIBERO-90 fine-tune: KITCHEN_SCENE3_turn_on_the_stove ──────
# Pretrained cosmos-policy was trained on libero_{spatial,10,goal,object}; libero_90
# single-task variants are unseen. Fine-tune only on this one task to compare V vs V+F
# on a contact-rich task where F/T (torque under gripper occlusion) should help most.
_LIBERO90_STOVE_DATA_DIR = os.path.join(BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "libero_90_stove")
_LIBERO90_STOVE_FT_DIR = os.path.join(BASE_DATASETS_DIR, "libero_90_stove_with_ft")

libero90_stove_dataset_v = L(LIBERODataset)(
    data_dir=_LIBERO90_STOVE_DATA_DIR,
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only", "t5_embeddings_with_libero90.pkl"
    ),
    chunk_size=16,
    use_image_aug=False,
    use_wrist_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    use_stronger_image_aug=False,
    rollout_data_dir="",
    demonstration_sampling_prob=1.0,
    return_value_function_returns=True,
    gamma=0.99,
)

libero90_stove_dataset_vf = L(LIBERODataset)(
    data_dir=_LIBERO90_STOVE_DATA_DIR,
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only", "t5_embeddings_with_libero90.pkl"
    ),
    chunk_size=16,
    use_image_aug=False,
    use_wrist_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    use_stronger_image_aug=False,
    rollout_data_dir="",
    demonstration_sampling_prob=1.0,
    return_value_function_returns=True,
    gamma=0.99,
    use_ft=True,
    ft_data_dir=_LIBERO90_STOVE_FT_DIR,
    ft_stats_path=os.path.join(_LIBERO90_STOVE_FT_DIR, "dataset_stats_p1p99.json"),
)

cosmos_predict2_2b_480p_libero90_stove_v = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(
            max_iter=5000,
            logging_iter=50,
            run_validation=False,
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                compile_tokenizer=dict(enabled=False),
            ),
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                use_lora=True,
                lora_rank=8,
            )
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=libero90_stove_dataset_v,
            batch_size=4,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_contact_aware",
            name="cosmos_predict2_2b_480p_libero90_stove_v",
        ),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2_2b_480p_libero90_stove_v__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                use_lora=True,
                lora_rank=8,
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                ),
            )
        ),
        job=dict(
            group="cosmos_v2_contact_aware",
            name="cosmos_predict2_2b_480p_libero90_stove_v__inference",
        ),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2_2b_480p_libero90_stove_vf = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(
            max_iter=5000,
            logging_iter=50,
            run_validation=False,
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                compile_tokenizer=dict(enabled=False),
            ),
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=11,
                min_num_conditional_frames=5,
                max_num_conditional_frames=5,
                tokenizer=dict(chunk_duration=41),
                use_lora=True,
                lora_rank=8,
            )
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=libero90_stove_dataset_vf,
            batch_size=4,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_contact_aware",
            name="cosmos_predict2_2b_480p_libero90_stove_vf",
        ),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2_2b_480p_libero90_stove_vf__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=11,
                min_num_conditional_frames=5,
                max_num_conditional_frames=5,
                tokenizer=dict(chunk_duration=41),
                use_lora=True,
                lora_rank=8,
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                ),
            )
        ),
        job=dict(
            group="cosmos_v2_contact_aware",
            name="cosmos_predict2_2b_480p_libero90_stove_vf__inference",
        ),
        upload_reproducible_setup=False,
    )
)



# ── GH200 full fine-tune (no LoRA): paper Appendix A recipe ─────────────
# Per cosmos paper A.2.2: all weights fully fine-tuned, larger batch, longer training.
# Single-GPU GH200 (96GB) lets us run batch=8, no FSDP needed.
cosmos_predict2_2b_480p_libero90_stove_vf_full = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(
            max_iter=20000,
            logging_iter=50,
            run_validation=False,
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                compile_tokenizer=dict(enabled=False),
            ),
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=11,
                min_num_conditional_frames=5,
                max_num_conditional_frames=5,
                tokenizer=dict(chunk_duration=41),
                use_lora=False,
            )
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=libero90_stove_dataset_vf,
            batch_size=8,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_contact_aware",
            name="cosmos_predict2_2b_480p_libero90_stove_vf_full",
        ),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2_2b_480p_libero90_stove_vf_full__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=11,
                min_num_conditional_frames=5,
                max_num_conditional_frames=5,
                tokenizer=dict(chunk_duration=41),
                use_lora=False,
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                ),
            )
        ),
        job=dict(
            group="cosmos_v2_contact_aware",
            name="cosmos_predict2_2b_480p_libero90_stove_vf_full__inference",
        ),
        upload_reproducible_setup=False,
    )
)


# ── GH200 full FT from Cosmos-Predict2 base (NOT cosmos-policy LIBERO) ─────
# Single-task LIBERO-90 fine-tune to compare V vs V+F under matched conditions:
# both runs start from the same vision-only DiT prior (no LIBERO bias), so any
# V+F gain is unambiguously attributable to the F/T input.
#
# Task: KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet
# - 0/3 success on cosmos-policy LIBERO baseline (eval 2026-04-24, libero_90)
# - Drawer is already open (no sub-task) → single contact event = clean F/T attribution
# - Tall wine bottle into shallow drawer = tight geometric constraint where
#   lateral force spikes signal misalignment
# - Sibling "put the black bowl in the bottom drawer" was 3/3 → bottle geometry,
#   not drawer access, is the differentiator → exactly what F/T should solve
_LIBERO90_WINEDRAWER_DATA_DIR = os.path.join(BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "libero_90_winedrawer")
_LIBERO90_WINEDRAWER_FT_DIR = os.path.join(BASE_DATASETS_DIR, "libero_90_winedrawer_with_ft")
_COSMOS_PREDICT2_BASE_CKPT = "hf://nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt"

libero90_winedrawer_dataset_v = L(LIBERODataset)(
    data_dir=_LIBERO90_WINEDRAWER_DATA_DIR,
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only", "t5_embeddings_with_libero90.pkl"
    ),
    chunk_size=16,
    use_image_aug=False,
    use_wrist_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    use_stronger_image_aug=False,
    rollout_data_dir="",
    demonstration_sampling_prob=1.0,
    return_value_function_returns=True,
    gamma=0.99,
)

libero90_winedrawer_dataset_vf = L(LIBERODataset)(
    data_dir=_LIBERO90_WINEDRAWER_DATA_DIR,
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only", "t5_embeddings_with_libero90.pkl"
    ),
    chunk_size=16,
    use_image_aug=False,
    use_wrist_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    use_stronger_image_aug=False,
    rollout_data_dir="",
    demonstration_sampling_prob=1.0,
    return_value_function_returns=True,
    gamma=0.99,
    use_ft=True,
    ft_data_dir=_LIBERO90_WINEDRAWER_FT_DIR,
    ft_stats_path=os.path.join(_LIBERO90_WINEDRAWER_FT_DIR, "dataset_stats_p1p99.json"),
)

cosmos_predict2_2b_480p_libero90_winedrawer_v_full = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(
            max_iter=20000,
            logging_iter=50,
            run_validation=False,
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                compile_tokenizer=dict(enabled=False),
            ),
        ),
        checkpoint=dict(
            load_path=get_checkpoint_path(_COSMOS_PREDICT2_BASE_CKPT),
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                use_lora=False,
            )
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=libero90_winedrawer_dataset_v,
            batch_size=8,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_contact_aware",
            name="cosmos_predict2_2b_480p_libero90_winedrawer_v_full",
        ),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2_2b_480p_libero90_winedrawer_v_full__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                use_lora=False,
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                ),
            )
        ),
        job=dict(
            group="cosmos_v2_contact_aware",
            name="cosmos_predict2_2b_480p_libero90_winedrawer_v_full__inference",
        ),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2_2b_480p_libero90_winedrawer_vf_full = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(
            max_iter=20000,
            logging_iter=50,
            run_validation=False,
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                compile_tokenizer=dict(enabled=False),
            ),
        ),
        checkpoint=dict(
            load_path=get_checkpoint_path(_COSMOS_PREDICT2_BASE_CKPT),
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=11,
                min_num_conditional_frames=5,
                max_num_conditional_frames=5,
                tokenizer=dict(chunk_duration=41),
                use_lora=False,
            )
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=libero90_winedrawer_dataset_vf,
            batch_size=8,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_contact_aware",
            name="cosmos_predict2_2b_480p_libero90_winedrawer_vf_full",
        ),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2_2b_480p_libero90_winedrawer_vf_full__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=11,
                min_num_conditional_frames=5,
                max_num_conditional_frames=5,
                tokenizer=dict(chunk_duration=41),
                use_lora=False,
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                ),
            )
        ),
        job=dict(
            group="cosmos_v2_contact_aware",
            name="cosmos_predict2_2b_480p_libero90_winedrawer_vf_full__inference",
        ),
        upload_reproducible_setup=False,
    )
)


def register_configs():
    cs = ConfigStore.instance()
    # Register the experiments
    for _item in [
        # LIBERO
        cosmos_predict2_2b_480p_libero,  # *** Main checkpoint ***
        cosmos_predict2_2b_480p_libero__inference_only,
        # RoboCasa
        cosmos_predict2_2b_480p_robocasa_50_demos_per_task,  # *** Main checkpoint ***
        cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference,
        # ALOHA
        cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80,  # *** Main checkpoint ***
        cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__inference_only,
        cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func,  # ALOHA planning model
        cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func__inference_only,
        # Contact-Aware (M1b smoke tests)
        cosmos_predict2_2b_480p_libero_smoke_v,
        cosmos_predict2_2b_480p_libero_smoke_vf,
        # Contact-Aware (M2 full 5k-step runs)
        cosmos_predict2_2b_480p_libero_m2_v,
        cosmos_predict2_2b_480p_libero_m2_vf,
        # Contact-Aware (M2 inference configs)
        cosmos_predict2_2b_480p_libero_m2_v__inference,
        cosmos_predict2_2b_480p_libero_m2_vf__inference,
        # Contact-Aware (M2 v2: p1/p99 norm, 50k steps)
        cosmos_predict2_2b_480p_libero_m2_vf_v2,
        cosmos_predict2_2b_480p_libero_m2_vf_v2__inference,
        # Contact-Aware (M2 v3: v2 + learnable F/T encoder MLP)
        cosmos_predict2_2b_480p_libero_m2_vf_v3,
        cosmos_predict2_2b_480p_libero_m2_vf_v3__inference,
        # Contact-Aware (M2 v4: p1/p99 norm + batch=4, NO encoder — clean ablation)
        cosmos_predict2_2b_480p_libero_m2_vf_v4,
        cosmos_predict2_2b_480p_libero_m2_vf_v4__inference,
        # Contact-Aware (libero_10: V+F where contact actually matters)
        cosmos_predict2_2b_480p_libero10_m2_vf,
        cosmos_predict2_2b_480p_libero10_m2_vf__inference,
        # Contact-Aware (libero_10: V+F with offline causal median3+butter3Hz F/T preprocessing)
        cosmos_predict2_2b_480p_libero10_m2_vf_filtered,
        cosmos_predict2_2b_480p_libero10_m2_vf_filtered__inference,
        # Contact-Aware (single task: libero_90 KITCHEN_SCENE3_turn_on_the_stove, V vs V+F)
        cosmos_predict2_2b_480p_libero90_stove_v,
        cosmos_predict2_2b_480p_libero90_stove_v__inference,
        cosmos_predict2_2b_480p_libero90_stove_vf,
        cosmos_predict2_2b_480p_libero90_stove_vf__inference,
        # Contact-Aware (single task, GH200 full fine-tune, no LoRA, paper-style recipe)
        cosmos_predict2_2b_480p_libero90_stove_vf_full,
        cosmos_predict2_2b_480p_libero90_stove_vf_full__inference,
        # Contact-Aware (single task, GH200 full FT FROM COSMOS-PREDICT BASE — V vs V+F matched)
        # Task: KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet
        cosmos_predict2_2b_480p_libero90_winedrawer_v_full,
        cosmos_predict2_2b_480p_libero90_winedrawer_v_full__inference,
        cosmos_predict2_2b_480p_libero90_winedrawer_vf_full,
        cosmos_predict2_2b_480p_libero90_winedrawer_vf_full__inference,
    ]:
        experiment_name = _item["job"]["name"]
        log.info(f"Registering experiment: {experiment_name}")
        cs.store(
            group="experiment",
            package="_global_",
            name=experiment_name,
            node=_item,
        )
