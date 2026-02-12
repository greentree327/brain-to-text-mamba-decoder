"""
Kaggle data and model source downloads.
"""

from __future__ import annotations

import kagglehub


def download_all_sources() -> dict:
    """Download Kaggle datasets/models and return their local paths."""
    paths = {}

    # Data
    paths["brain_to_text_25_path"] = kagglehub.competition_download("brain-to-text-25")
    paths["heyyousum_brain_to_text_25_copytaskdata_description_path"] = kagglehub.dataset_download(
        "heyyousum/brain-to-text-25-copytaskdata-description"
    )

    # Mamba models
    paths["heyyousum_v7_57_a14b_mamba_path"] = kagglehub.dataset_download("heyyousum/v7-57-a14b-mamba")
    paths["heyyousum_v7_57_a14c_mamba_path"] = kagglehub.dataset_download("heyyousum/v7-57-a14c-mamba")
    paths["heyyousum_v7_57_a14d_mamba_path"] = kagglehub.dataset_download("heyyousum/v7-57-a14d-mamba")

    paths["heyyousum_v7_57_a14m_mamba_path"] = kagglehub.dataset_download("heyyousum/v7-57-a14m-mamba")
    paths["heyyousum_v7_57_a15n_mamba_path"] = kagglehub.dataset_download("heyyousum/v7-57-a15n-mamba")
    paths["heyyousum_v7_57_a15h_mamba_path"] = kagglehub.dataset_download("heyyousum/v7-57-a15h-mamba")

    paths["heyyousum_v7_57_a16f_mamba_path"] = kagglehub.dataset_download("heyyousum/v7-57-a16f-mamba")

    paths["heyyousum_v7_57_a14j_mamba_path"] = kagglehub.dataset_download("heyyousum/v7-57-a14j-mamba")
    paths["heyyousum_v7_57_a16g_mamba_path"] = kagglehub.dataset_download("heyyousum/v7-57-a16g-mamba")
    paths["heyyousum_v7_57_a15t_mamba_path"] = kagglehub.dataset_download("heyyousum/v7-57-a15t-mamba")

    # GRU models
    paths["heyyousum_btt_25_gru_pure_baseline_0_0898_path"] = kagglehub.dataset_download(
        "heyyousum/btt-25-gru-pure-baseline-0-0898"
    )
    paths["heyyousum_btt_25_baseline_seed_2_99_path"] = kagglehub.dataset_download(
        "heyyousum/btt-25-baseline-seed-2-99"
    )
    paths["heyyousum_btt_25_gru_size_34_stride_4_seed_3_72_path"] = kagglehub.dataset_download(
        "heyyousum/btt-25-gru-size-34-stride-4-seed-3-72"
    )
    paths["heyyousum_gru_size_22_stride_4_input_layer_drop_0_25_sed_7_1_path"] = kagglehub.dataset_download(
        "heyyousum/gru-size-22-stride-4-input-layer-drop-0-25-sed-7-1"
    )

    # N-gram and KenLM
    paths["heyyousum_quality_english_dataset_for_ngram_model_path"] = kagglehub.notebook_output_download(
        "heyyousum/fork-of-quality-english-dataset-for-ngram-model"
    )
    paths["ansonlyt_kenlm_path"] = kagglehub.dataset_download(
        "heyyousum/custom-4-gram-wiki-news-switchboard-updated-v3"
    )

    # Version and parameters
    paths["version"] = "LISA_open_version_mamba_0.02696_GRU_coherent_majority_vote_random_a14.2b"
    paths["current_drift_lambda"] = 0.01

    print("Data source import complete.")
    return paths
