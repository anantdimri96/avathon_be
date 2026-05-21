from .greedy import greedy_allocate
from .hungarian import hungarian_allocate
from .heuristic import heuristic_allocate
from .ml_allocator import ml_allocate

ALGORITHMS = {
    "greedy": greedy_allocate,
    "hungarian": hungarian_allocate,
    "heuristic": heuristic_allocate,
    "ml": ml_allocate,
}
