
class ChannelRegister:
    channels = {}

    @staticmethod
    def add_channel(rank, tb, tb_channel_id,  channel):
        key = (rank, tb, tb_channel_id)
        ChannelRegister.channels[key] = channel

    @staticmethod
    def get_channel(rank: int, threadblock: int, tb_channel_id: int):
        return ChannelRegister.channels.get((rank, threadblock, tb_channel_id))