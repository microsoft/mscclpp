class ChannelRegister:
    channels = {}

    @staticmethod
    def add_channel(rank, tb, tb_channel_id, channel):
        ChannelRegister.channels[(rank, tb, tb_channel_id)] = channel

    @staticmethod
    def get_channel(rank: int, threadblock: int, tb_channel_id: int):
        return ChannelRegister.channels.get((rank, threadblock, tb_channel_id))


class SemaphoreRegister:
    semaphores = {}

    @staticmethod
    def add_semaphore(semaphore):
        SemaphoreRegister.semaphores[(semaphore.rank, semaphore.id)] = semaphore

    @staticmethod
    def get_semaphore(rank: int, semaphore_id: int):
        return SemaphoreRegister.semaphores.get((rank, semaphore_id))
