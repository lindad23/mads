import time
from tqdm import trange

# 强制慢下来测试
for j in trange(0, 100):
    time.sleep(0.1) # 暂停 0.1 秒