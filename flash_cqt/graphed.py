"""cuda graph wrapper for the per slice transform.

the per slice path is launch bound, replaying a recorded graph pays one launch
for the whole chain instead of one per kernel. record once on static buffers,
then every call copies the new input in, replays, and clones the result out.

inference only, capture does not record autograd. use it on the hot per slice
functions, for example

  g = Graphed(cqt.fwd, example_slice)
  blocks, side = g(next_slice)
"""
import torch


def _tensors(t):
    #walk a nested list or tuple and yield the tensors in order
    if torch.is_tensor(t):
        yield t
    elif isinstance(t, (list, tuple)):
        for u in t:
            yield from _tensors(u)


def _clone(t):
    if torch.is_tensor(t):
        return t.clone()
    if isinstance(t, tuple):
        return tuple(_clone(u) for u in t)
    if isinstance(t, list):
        return [_clone(u) for u in t]
    return t


class Graphed:
    def __init__(self, fn, *example):
        assert torch.cuda.is_available(), "cuda graphs need a cuda device"
        self._fn = fn
        self._in = _clone(example)
        #warm up on a side stream so lazy init does not end up in the graph
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                fn(*self._in)
        torch.cuda.current_stream().wait_stream(s)
        self._graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self._graph):
            self._out = fn(*self._in)

    @torch.no_grad()
    def __call__(self, *args):
        for dst, src in zip(_tensors(self._in), _tensors(args)):
            dst.copy_(src)
        self._graph.replay()
        #clone so the caller can keep results across replays
        return _clone(self._out)
