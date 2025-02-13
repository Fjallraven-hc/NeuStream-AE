import dataclasses
from typing import List
import marshal
import random

@dataclasses.dataclass
class TestRequest:
    """
    TestRequest: A request for testing the server's performance
    """
    
    prompt: str
    prompt_len: int
    output_len: int
    
@dataclasses.dataclass
class Dataset:
    """
    Dataset: A dataset for testing the server's performance
    """
 
    dataset_name: str	# "sharegpt" / "alpaca" / ...
    reqs: List[TestRequest]
    
    def dump(self, output_path: str):
        marshal.dump({
            "dataset_name": self.dataset_name,
            "reqs": [(req.prompt, req.prompt_len, req.output_len) for req in self.reqs]
        }, open(output_path, "wb"))
    
    @staticmethod
    def load(input_path: str):
        loaded_data = marshal.load(open(input_path, "rb"))
        return Dataset(
            loaded_data["dataset_name"],
            [TestRequest(req[0], req[1], req[2]) for req in loaded_data["reqs"]]
        )
        
def sample_requests(dataset_path: str, num_prompts: int, seed: int = 0) -> List[TestRequest]:
    """
    sample_requests: Sample the given number of requests from the dataset.
    """
    dataset = Dataset.load(dataset_path)
    if num_prompts > len(dataset.reqs):
        raise ValueError(
            f"Number of prompts ({num_prompts}) is larger than the dataset size ({len(dataset.reqs)})."
        )
    random.seed(seed)
    return random.sample(dataset.reqs, num_prompts)