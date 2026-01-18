import os
import tensorflow_datasets as tfds
from tensorflow_datasets.core.download import DownloadConfig
from multiprocessing import cpu_count


def fast_download_tfds(
    dataset_name: str,
    data_dir: str,
    num_parallel_downloads: int = 16,
    num_parallel_extracts: int = 16,
):
    """
    TFDS fast downloader compatible with tensorflow-datasets 4.9.x
    """

    cores = cpu_count()
    num_parallel_downloads = min(num_parallel_downloads, cores)
    num_parallel_extracts = min(num_parallel_extracts, cores)

    # TFDS 4.9.x parallelism via environment variables
    os.environ["TFDS_DOWNLOAD_MAX_WORKERS"] = str(num_parallel_downloads)
    os.environ["TFDS_EXTRACT_MAX_WORKERS"] = str(num_parallel_extracts)

    print(f"CPU cores detected: {cores}")
    print(f"Parallel downloads (env): {num_parallel_downloads}")
    print(f"Parallel extracts (env): {num_parallel_extracts}")

    data_dir = os.path.expanduser(data_dir)

    # IMPORTANT: no num_parallel_* arguments here
    download_config = DownloadConfig(
        try_download_gcs=True
    )

    builder = tfds.builder(dataset_name, data_dir=data_dir)
    builder.download_and_prepare(download_config=download_config)

    print(f"✅ {dataset_name} download complete")
