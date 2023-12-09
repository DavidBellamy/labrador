from huggingface_hub import snapshot_download

snapshot_download(repo_id="drbellamy/labrador", local_dir="model_weights")
