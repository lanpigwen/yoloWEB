import random
from datetime import datetime, timedelta

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

# # 示例使用
# start = "2024-05-10"
# end = "2024-05-24"
# user_data = generate_random_timestamps(start, end)

# # 打印结果
# for user in user_data:
#     print(user)