from importlib.metadata import version
import warnings
import transformers
from .llama_hijack_4_43 import llama_flash_attn2_forward as llama_flash_attn2_forward_4_43
from .mistral_hijack_4_43 import mistral_flash_attn2_forward as mistral_flash_attn2_forward_4_43
from .phi3_hijack_4_43 import phi3_flash_attn2_forward as phi3_flash_attn2_forward_4_43

def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    return transformers_version


def replace_llama():
    transformers_version = check_version()
    version_list = ['4.43']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(
            f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_4_43


def replace_mistral():
    transformers_version = check_version()
    version_list = ['4.43']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(
            f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")
    transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_4_43


def replace_phi3():
    transformers_version = check_version()
    version_list = ['4.43']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(
            f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")
    transformers.models.phi3.modeling_phi3.Phi3FlashAttention2.forward = phi3_flash_attn2_forward_4_43
