
import torch

x = torch.rand(5, 3)
print(x)

print(torch.cuda.is_available())
#print(torch.cuda.get_device_name().encode('utf-8').strip())
print(torch.cuda.device_count())
