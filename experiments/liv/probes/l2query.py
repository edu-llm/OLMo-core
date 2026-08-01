import torch
p = torch.cuda.get_device_properties(0)
print("name", p.name)
print("L2_cache_size_bytes", getattr(p,"L2_cache_size",None))
print("L2_cache_size_MiB", (getattr(p,"L2_cache_size",0) or 0)/2**20)
print("total_mem_GiB", p.total_memory/2**30)
print("multi_processor_count", p.multi_processor_count)
