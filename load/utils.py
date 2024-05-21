import uuid
from typing import Literal
import random


def random_uuid(dtype: Literal['int', 'str', 'bytes', 'time'] = 'int') -> int | str | bytes:
    """生成随机uuid
    reference: https://github.com/vllm-project/vllm/blob/main/vllm/utils.py
    """
    assert dtype in ['int', 'str', 'bytes', 'time'], f"unsupported dtype: {dtype}, should be in ['int', 'str', 'bytes', 'time']"

    # uuid4: 由伪随机数得到，有一定的重复概率，该概率可以计算出来。
    uid = uuid.uuid4()
    if dtype == 'int':
        return uid.int
    elif dtype == 'str':
        return uid.hex
    elif dtype == 'bytes':
        return uid.bytes
    else:
        return uid.time


def random_uuid_int() -> int:
    """random_uuid 生成的 int uuid 会超出int64的范围,lmdeploy使用会报错"""
    return random.getrandbits(64)


if __name__ == '__main__':
    print(random_uuid('int'))
    print(random_uuid_int())
