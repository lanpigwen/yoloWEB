import redis
import json
# 连接Redis
r = redis.Redis(host='localhost', port=6379, db=1)

# r.flushall()
# # 从Redis获取JSON字符串
keys_pattern = 'Shoots:123*'
matching_keys = r.keys(keys_pattern)

# 遍历匹配的键，并获取它们的值
for key in matching_keys:
    value = r.get(key)
    print(f"Key: {key}, Value: {json.loads(value)}")
