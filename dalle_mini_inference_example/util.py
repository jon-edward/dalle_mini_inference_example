import logging
import os
from pathlib import Path
import shutil

from dalle_mini import DalleBart, DalleBartProcessor
import jax.numpy as jnp
from vqgan_jax.modeling_flax_vqgan import VQModel
import wandb


_ROOT_DIR = Path(__file__).parent.parent.absolute()

DALLE_MINI_MODEL_DIR = str(_ROOT_DIR.joinpath("_dalle-mini-model/"))
DALLE_MINI_MODEL_LOCATION = "dalle-mini/dalle-mini/mega-1-fp16:latest"

VQGAN_MODEL_DIR = str(_ROOT_DIR.joinpath("_vqgan-model/"))
VQGAN_MODEL_LOCATION = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"


def _log_rmtree_error(*error_info):
    logging.warning(f"Error encountered when cleaning "
                    f"preexisting model dirs: {error_info}")


def _log_mkdir_error(error):
    logging.warning(f"Error encountered when making directory: {error}")


def _clean_model_paths():
    for dir_path in (DALLE_MINI_MODEL_DIR, VQGAN_MODEL_DIR):
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path, onerror=_log_rmtree_error)
        try:
            os.mkdir(dir_path)
        except Exception as e:
            _log_mkdir_error(e)


def update_dalle_mini_model():
    api_obj = wandb.Api()
    artifact = api_obj.artifact(DALLE_MINI_MODEL_LOCATION)
    artifact.download(DALLE_MINI_MODEL_DIR)


def update_vqgan_model():
    model = VQModel.from_pretrained(
        VQGAN_MODEL_LOCATION,
        revision=VQGAN_COMMIT_ID,
    )
    model.save_pretrained(str(VQGAN_MODEL_DIR))


def update_downloaded_models():
    _clean_model_paths()
    update_dalle_mini_model()
    update_vqgan_model()


def load_dalle_model():
    return DalleBart.from_pretrained(
        DALLE_MINI_MODEL_DIR, dtype=jnp.float16, _do_init=False
    )


def load_vqgan_model():
    return VQModel.from_pretrained(
        VQGAN_MODEL_DIR, revision=VQGAN_COMMIT_ID, _do_init=False
    )


def load_dalle_processor():
    return DalleBartProcessor.from_pretrained(DALLE_MINI_MODEL_DIR)
