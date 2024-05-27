import random
from datetime import datetime, timedelta
import uuid



def generate_random_timestamps(start, end, count=10):
    # 将start和end转换为datetime对象
    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")

    def random_time():
        # 生成随机时间戳
        base_time = start_date + timedelta(seconds=random.randint(0, int((end_date - start_date).total_seconds())))
        start_time = base_time
        # 随机增加1到2小时的时间戳
        end_time = start_time + timedelta(seconds=random.randint(1, 2 * 60 * 60))
        return start_time, end_time

    def random_shoot():
        # 生成随机射击数据
        return {
            'nx': random.random(),
            'ny': random.random(),
            'score': random.random() > 0.5
        }

    def random_user_data():
        # 生成随机用户数据
        shoots = [random_shoot() for _ in range(random.randint(1, 30))]
        start_time, end_time = random_time()
        return {
            #python的时间戳要×1000才是js的时间戳
            'trainStartTime': start_time.timestamp()*1000,
            'trainEndTime': end_time.timestamp()*1000,
            'Shoots': shoots
        }

    # 生成指定数量的用户数据列表
    return [random_user_data() for _ in range(count)]

# # # 示例使用
# start = "2020-01-01"
# end = "2024-05-25"
# data = generate_random_timestamps(start, end,3000)
user=['123','456','789']

# allData=[]
# # print(len(user_data))
# # # 打印结果
# for d in data:
#     random_user = random.choice(user)
#     uuid4 = str(uuid.uuid4())
#     tempTrain={'user':random_user,'uuid':uuid4,'ShootInfo':d}
#     allData.append(tempTrain)


import json
import redis

# # 连接到Redis，默认是localhost的6379端口
# r = redis.Redis(host='localhost', port=6379, db=1)

# for item in allData:
#     user = item['user']
#     uuid = item['uuid']
#     # 构造Redis的key
#     key = f"Shoots:{user}:{uuid}"
    
#     # 将字典d序列化为JSON字符串
#     d_json = json.dumps(item['ShootInfo'])
    
#     # 存储到Redis中
#     # 使用SET命令，key是"Shoots:user:uuid"，value是序列化后的JSON字符串
#     r.set(key, d_json)

# 确认数据是否存储成功，可以打印出来或使用redis-cli进行检查
# 例如，使用redis-cli获取数据：GET Shoots:123:some_uuid

# for u in user:
#     key=f'user:{u}'
#     r.set(key,u) 

# print(r.get('user:456')==b'456')
