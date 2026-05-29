import asyncio
import time

async def producer(queue : asyncio.Queue):
    for i in range(3):
        await queue.put(f"item {i}")
        print(f"Produced item {i}")
        await asyncio.sleep(1)

async def consumer(queue: asyncio.Queue):
    while True:
        req = await queue.get()
        print(f"Consumed {req}")
        await asyncio.sleep(2)
        queue.task_done()

async def queue_main():
    q = asyncio.Queue(maxsize=5)
    #创建生产者和消费者任务
    producer_task = asyncio.create_task(producer(q))
    consumer_task = asyncio.create_task(consumer(q))
    await producer_task # 等待生产者结束
    await q.join()  # 等待队列中的所有任务完成
    consumer_task.cancel()  # 取消消费者任务

if __name__ == "__main__":
    asyncio.run(queue_main())

