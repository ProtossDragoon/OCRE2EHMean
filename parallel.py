# 내장
import functools

# 서드파티
import ray
import tqdm

# 프로젝트
import approach


ray.init()


@ray.remote
@functools.wraps(approach.start)
def start_parallel(*args, **kwargs):
    return approach.start(*args, **kwargs)


def wait(
    promise_li: list, 
    callback_fn = None,
) -> list:
    """Ray 기반의 promise 객체들의 실행이 모두 완료될 때까지 대기시키는 함수입니다.
    
    Args:
        promise_li (list): promise 객체들이 저장된 리스트.
        callback_fn (function): 처리 완료된 프로세스의 함수 반환값을 입력으로 받는 함수.

    Returns:
        list: 모든 프로세스에 대해 처리를 완료한 함수의 반환값들을 담고 있는 리스트.
    """
    ret = []
    n_promise = len(promise_li)
    _bar = (_ for _ in tqdm.tqdm(range(0, n_promise)))
    while promise_li:
        _before = len(promise_li)
        done_ids, promise_li = ray.wait(promise_li)
        if not callback_fn:
            callback_fn = lambda x: x
        for done_id in done_ids:
            ret.append(callback_fn(ray.get(done_id)))
        _after = len(promise_li)
        for _ in range(_before - _after):
            next(_bar)
    try:
        next(_bar)
    except StopIteration:
        pass
    return ret