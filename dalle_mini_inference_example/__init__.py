
from functools import partial
import random

from flax.training.common_utils import shard_prng_key

from dalle_mini_inference_example.util import load_dalle_model, load_vqgan_model, load_dalle_processor

from flax.jax_utils import replicate
import jax
import numpy as np
from PIL import Image


def infer_images(
        prompts,
        seed=None,
        n_predictions=8,
        gen_top_k=None,
        gen_top_p=None,
        temperature=None,
        cond_scale=10.0):

    seed = seed if seed else random.randint(0, 2**32 - 1)

    dalle_model, dalle_params = load_dalle_model()
    vqgan_model, vqgan_params = load_vqgan_model()

    dalle_params = replicate(dalle_params)
    vqgan_params = replicate(vqgan_params)

    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
    def p_generate(
            _tokenized_prompt, _key, _params, _top_k, _top_p, _temperature, _condition_scale
    ):
        return dalle_model.generate(
            **_tokenized_prompt,
            prng_key=_key,
            params=_params,
            top_k=_top_k,
            top_p=_top_p,
            temperature=_temperature,
            condition_scale=_condition_scale,
        )

    @partial(jax.pmap, axis_name="batch")
    def p_decode(_indices, _params):
        return vqgan_model.decode_code(_indices, params=_params)

    key = jax.random.PRNGKey(seed)

    processor = load_dalle_processor()
    tokenized_prompts = processor(prompts)
    tokenized_prompts = replicate(tokenized_prompts)

    images = []

    for i in range(max(n_predictions // jax.device_count(), 1)):
        key, subkey = jax.random.split(key)
        encoded_images = p_generate(
            tokenized_prompts,
            shard_prng_key(subkey),
            dalle_params,
            gen_top_k,
            gen_top_p,
            temperature,
            cond_scale,
        )
        encoded_images = encoded_images.sequences[..., 1:]
        decoded_images = p_decode(encoded_images, vqgan_params)
        decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
        for decoded_img in decoded_images:
            img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
            images.append(img)

    return images
