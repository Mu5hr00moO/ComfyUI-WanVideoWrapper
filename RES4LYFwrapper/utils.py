# code based on RES4LYF by ClownsharkBatwing (https://github.com/ClownsharkBatwing/RES4LYF)
from .samplers_res4lyf_list import RES4LYF_NAMES
from comfy.samplers import SCHEDULER_NAMES

# Dictionary construction logic
RES4LYF_samplers_map = {}
for orig_sampler_name in RES4LYF_NAMES:
    if "/" in orig_sampler_name:
        folder, sampler_name = orig_sampler_name.rsplit("/", 1)
    else:
        folder = ""
        sampler_name = orig_sampler_name
    RES4LYF_samplers_map[sampler_name] = folder

class DualFormatList(list):
    """list that can match items with or without category prefixes."""
    def __contains__(self, item):
        if super().__contains__(item):
            return True

        if isinstance(item, str) and "/" in item:
            base_name = item.split("/")[-1]
            return any(name.endswith(base_name) for name in self)

        return any(isinstance(opt, str) and opt.endswith("/" + item) for opt in self)    

# Main function to get default sampler name
def get_res4lyf_default_sampler_name():
    default_sampler_name = "res_2m"
    if default_sampler_name in RES4LYF_samplers_map:
        return default_sampler_name
    return default_sampler_name

# Function to get list of sampler names
def get_res4lyf_sampler_name_list() -> list:
    sampler_name_list = []
    for sampler_name in RES4LYF_NAMES:
        sampler_name_list.append(sampler_name) 
    return DualFormatList(sampler_name_list)
      
def get_res4lyf_scheduler_list():
    scheduler_names = SCHEDULER_NAMES.copy()
    if "beta57" not in scheduler_names:
        scheduler_names.append("beta57")
    if "bong_tangent" not in scheduler_names:
        scheduler_names.append("bong_tangent")      
    return scheduler_names        