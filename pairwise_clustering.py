from typing import List, Sequence, Tuple, Callable, Optional, Union
import random
import numpy as np
from dataclasses import dataclass, field, fields as dataclass_fields
from concurrent.futures import ProcessPoolExecutor


class RandomizedSet:
    """
    A class that allows for efficient removal and random retrieval of elements.
    """

    def __init__(self, data_indices: List[int]):
        self._data_indices = data_indices
        self._index_map = {i: i for i in data_indices}

    def remove(self, val):
        if val not in self._index_map:
            return False
        idx = self._index_map[val]
        last_val = self._data_indices[-1]
        # Swap val with the last item
        self._data_indices[idx] = last_val
        self._index_map[last_val] = idx
        # Remove the last item
        self._data_indices.pop()
        del self._index_map[val]
        return True

    def get_random(self):
        return random.choice(self._data_indices)
    
    def __len__(self):
        return len(self._data_indices)
    
    def __iter__(self):
        return iter(self._data_indices)
    
    def __bool__(self):
        return bool(self._data_indices)
    

@dataclass
class PairwiseConfig:
    """
    Configuration for pairwise clustering.
    """

    minimize: bool = field(default=False, metadata={"help": "Whether to minimize (distance) or maximize (similarity)"})
    num_runs: int = field(default=5, metadata={"help": "Number of runs/samples to perform"})
    num_processes: Optional[int] = field(default=None, metadata={"help": "How many processes to use (None = use all cores)"})
    min_comp_val: Optional[float] = field(default=None, metadata={"help": "Minimum value returned by compare_func"})
    max_comp_val: Optional[float] = field(default=None, metadata={"help": "Maximum value returned by compare_func"})

    def __post_init__(self):
        self.finalize_bounds()
    
    def generate_config_doc(self) -> str:
        """Auto-generate docstring entries from a config dataclass."""
        lines = []
        for field in dataclass_fields(self):
            help_text = field.metadata.get("help", "")
            lines.append(f"    {field.name} ({field.type.__name__}): {help_text}")
        return "\n".join(lines)
    
    def finalize_bounds(self):
        if self.min_comp_val is None and self.max_comp_val is None:
            if self.minimize:
                self.min_comp_val, self.max_comp_val = 0.0, float('inf')
            else:
                self.min_comp_val, self.max_comp_val = float('-inf'), float('inf')
        elif self.min_comp_val is None or self.max_comp_val is None:
            raise ValueError("min_comp_val and max_comp_val must both be specified if one is specified")


def _create_clusters(list1: Sequence[object],
                    list2: Sequence[object],
                    compare_func: Callable[[object, object], float],
                    min_comp_val: float,
                    max_comp_val: float,
                    minimize: bool) -> Tuple[List[Tuple[object, object]], List[float]]:
    """
    Creates pairs of objects from the two lists.
    """

    matches = []
    match_scores = []
    list1_inds, list2_inds = RandomizedSet(list(range(len(list1)))), RandomizedSet(list(range(len(list2))))
    while list1_inds:
        # Get random object from list1
        rand_list1_ind = list1_inds.get_random()
        rand_list1_obj = list1[rand_list1_ind]
        # Remove object from list1 so we don't match it again in this run
        list1_inds.remove(rand_list1_ind)
        optimal_list2_obj_ind = -1
        optimal_score = max_comp_val if minimize else min_comp_val
        for list2_obj_ind in list2_inds:
            if minimize:
                # Get absolute value of score if minimizing - when comparing distance, non-negative = negative
                curr_score = abs(compare_func(rand_list1_obj, list2[list2_obj_ind]))
                # update optimal similarity/distance score and index if this element in list2 is better than we've seen
                if curr_score < optimal_score:
                    optimal_score = curr_score
                    optimal_list2_obj_ind = list2_obj_ind
            else:
                curr_score = compare_func(rand_list1_obj, list2[list2_obj_ind])
                if curr_score > optimal_score:
                    optimal_score = curr_score
                    optimal_list2_obj_ind = list2_obj_ind
        # If we found a match for the current element in list1 in list2, keep track of the best match and its info
        if optimal_list2_obj_ind >= 0:
            list2_inds.remove(optimal_list2_obj_ind)
            matches.append((rand_list1_obj, list2[optimal_list2_obj_ind]))
            match_scores.append(optimal_score)
    return matches, match_scores

def pairwise_clustering(list1: Sequence[object],
                        list2: Sequence[object],
                        compare_func: Callable[[object, object], float],
                        config: Optional[Union[dict, PairwiseConfig]] = None) -> List[Tuple[object, object]]:
    """
    Performs pairwise clustering on the two given lists.
    Assumes that there may not be a one-to-one mapping between the two lists.
    First object from second list is chosen in cases where multiple objects in second list are
    closest to the same object in first list.
    The following are tunable parameters:
    By default, the function will maximize the value returned by compare_func. It essentially assumes
    that compare_func returns a similarity metric and that its output maps from -inf to inf.
    If in config, minimze is True, the function will minimize the value returned by compare_func. It assumes
    that compare_func returns a distance metric and that its output maps from 0 to inf.
    By default, clusters are created 5 times and the best result is returned.
    The number of parallel processes used is set to the number of cores on the machine by default.

    Args:
        list1: The first list of objects.
        list2: The second list of objects.
        compare_func: A function that takes two objects and returns a float comparing the two objects.
        config: Either a PairwiseConfig object or a dict with the following fields:
        {generate_config_doc(PairwiseConfig)}
    Returns:
        A list of tuples, where each tuple contains two objects from the two lists that are closest to each other.
    """

    if config is None:
        config = PairwiseConfig()
    elif isinstance(config, dict):
        config = PairwiseConfig(**config)
    else:
        assert isinstance(config, PairwiseConfig)
    
    global_matches = []
    global_opt = float('inf') if config.minimize else float('-inf')
    with ProcessPoolExecutor(max_workers=config.num_processes) as executor:
        futures = [executor.submit(_create_clusters,
                                   list1,
                                   list2,
                                   compare_func,
                                   config.min_comp_val,
                                   config.max_comp_val,
                                   config.minimize) for _ in range(config.num_runs)]
        for future in futures:
            matches, match_scores = future.result()
            # If there are no matches in a run, we immediately return an empty list
            if len(matches) == 0:
                return []
            # The typical method of scoring a set of matches will not work if there is only one match in a run so
            # we handle this case separately
            if len(matches) == 1:
                # If there is only one match in a run and the current best match from previous runs has more than
                # one match, we skip this run - we prefer multiple matches to a single match
                if len(global_matches) > 1:
                    continue
                else:
                    # otherwise, we update as usual but the score we use is the sum of the match scores
                    curr_opt = np.sum(match_scores)
                    if config.minimize:
                        if curr_opt < global_opt:
                            global_opt = curr_opt
                            global_matches = matches
                    else:
                        if curr_opt > global_opt:
                            global_opt = curr_opt
                            global_matches = matches
            else:
                # the score for a set of matches is different depending on whether we are minimizing or maximizing
                curr_var = np.var(match_scores) if len(match_scores) > 1 else 1
                curr_tot = np.sum(match_scores)
                curr_opt = curr_var*curr_tot if config.minimize else curr_tot/curr_var
                # if the current best match from previous runs has only one match, we automatically update the best match
                # as stated above, we prefer multiple matches to a single match
                if len(global_matches) == 1:
                    global_opt = curr_opt
                    global_matches = matches
                else:
                    # otherwise, we update as usual
                    if config.minimize:
                        if curr_opt < global_opt:
                            global_opt = curr_opt
                        global_matches = matches
                    else:
                        if curr_opt > global_opt:
                            global_opt = curr_opt
                            global_matches = matches
    return global_matches
