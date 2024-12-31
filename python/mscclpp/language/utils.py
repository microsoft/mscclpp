from python.mscclpp.language.types import Op


def remove_op(op: Op):
    for p in op.prev:
        p.next.remove(op)
        p.next += op.next
        p.next = list(set(p.next))

    for n in op.next:
        n.prev.remove(op)
        n.prev = op.prev.union(n.prev)

    op.next = []
    op.prev = []


def merge_op(op: Op, other_op: Op):
    if other_op in op.next:
        op.next.remove(other_op)
        other_op.prev.remove(op)
    for p in other_op.prev:
        p.next.remove(other_op)
        p.next.append(op)

    for n in other_op.next:
        n.prev.remove(other_op)
        n.prev.add(op)

    op.prev = op.prev.union(other_op.prev)
    op.next = list(set(op.next + other_op.next))


def circular_dep_after_merge(op: Op, other_op: Op):
    root = set([op, other_op])
    frontier = set(op.next)
    if other_op in frontier:
        frontier.remove(other_op)
    frontier = list(frontier.union(other_op.next))
    while len(frontier) > 0:
        current = frontier[0]
        for n in current.next:
            # The root node will be visited again if there is a circular dependency
            if n in root:
                return True
            frontier.append(n)
        frontier = frontier[1:]


"""
For case: op2.prev = [op1, op3]. op1.next = [op2]. op3.next = [op2]. And op1 and op2 are satisfied to merge.
We only apply the merge if all previous ops of op2 are visited. (op1 is the last previous op of op2).
"""


def all_prevs_visited_after_merge(op: Op, other_op: Op):
    step = op.step
    for prev in other_op.prev:
        if prev.step > step:
            return False
    return True


def same_tb(op1: Op, op2: Op):
    return op1.tb == op2.tb and op1.channel == op2.channel


def same_count(op1: Op, op2: Op):
    return op1.cnt() == op2.cnt()


def same_buf_dst(op1: Op, op2: Op):
    return op1.dst.buffer == op2.dst.buffer and op1.dst.index == op2.dst.index


def same_src_dst_buffer_type(op1: Op, op2: Op):
    return op1.src.buffer == op2.src.buffer and op1.dst.buffer == op2.dst.buffer


def buf_dst_src_match(op1: Op, op2: Op):
    return op1.dst.buffer == op2.src.buffer and op1.dst.index == op2.src.index


def same_buf_src(op1: Op, op2: Op):
    return op1.src.buffer == op2.src.buffer and op1.src.index == op2.src.index


def same_chan_type(op1: Op, op2: Op):
    return op1.channel_type == op2.channel_type


def same_tb(op1: Op, op2: Op):
    return op1.tb == op2.tb
