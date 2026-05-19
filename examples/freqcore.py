import math
import numpy as np
from typing import Iterator, Optional
import torch
from torch.utils.data.dataloader import _BaseDataLoaderIter
from torch.utils.data import Dataset, _DatasetKind
from torch.utils.data.distributed import DistributedSampler
from operator import itemgetter
import torch.distributed as dist
import warnings

__all__ = ['FreqCore']



def hack_indices(self):
    with torch.autograd.profiler.record_function(self._profile_name):
        if self._sampler_iter is None:
            # TODO(https://github.com/pytorch/pytorch/issues/76750)
            self._reset()  # type: ignore[call-arg]
        if isinstance(self._dataset, FreqCore):
            indices, data = self._next_data()
        else:
            data = self._next_data()
        self._num_yielded += 1
        if self._dataset_kind == _DatasetKind.Iterable and \
                self._IterableDataset_len_called is not None and \
                self._num_yielded > self._IterableDataset_len_called:
            warn_msg = ("Length of IterableDataset {} was reported to be {} (when accessing len(dataloader)), but {} "
                        "samples have been fetched. ").format(self._dataset, self._IterableDataset_len_called,
                                                                self._num_yielded)
            if self._num_workers > 0:
                warn_msg += ("For multiprocessing data-loading, this could be caused by not properly configuring the "
                                "IterableDataset replica at each worker. Please see "
                                "https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for examples.")
            warnings.warn(warn_msg)
        if isinstance(self._dataset, FreqCore):
            self._dataset.set_active_indices(indices)
        return data


_BaseDataLoaderIter.__next__ = hack_indices


@torch.no_grad()
def concat_all_gather(tensor, dim=0):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=dim)
    return output



class FreqCore(Dataset):
    def __init__(self, dataset: Dataset, num_epochs: int, start_epoch: int = 0,
                 prune_ratio: float = 0.5, delta: float = 0.875):
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.keep_ratio = min(1.0, max(1e-1, 1.0 - prune_ratio))
        self.delta = delta
        
        # self.scores = torch.ones(len(self.dataset)) * 3
        self.scores = torch.zeros(len(self.dataset))
        self.weights = torch.ones(len(self.dataset))
        self.num_pruned_samples = 0
        self.cur_batch_index = None
        self.iterations = start_epoch

    def __getattr__(self, name):
        return getattr(self.dataset, name)
    
    def set_active_indices(self, cur_batch_indices: torch.Tensor):
        self.cur_batch_index = cur_batch_indices
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return index, self.dataset[index] 

    def update(self, values, score):
        assert isinstance(values, torch.Tensor)
        batch_size = values.shape[0]
        assert len(self.cur_batch_index) == batch_size, 'not enough index'
        device = values.device
        weights = self.weights[self.cur_batch_index].to(device)
        indices = self.cur_batch_index.to(device)
        loss_val = values.detach().clone()
        self.cur_batch_index = []

        if dist.is_available() and dist.is_initialized():
            iv = torch.cat([indices.view(1, -1).to(device), loss_val.view(1, -1)], dim=0)
            iv_whole_group = concat_all_gather(iv, 1)
            indices = iv_whole_group[0]
            loss_val = iv_whole_group[1]
        score = torch.from_numpy(score)
        self.scores[indices.cpu().long()] = score.cpu().float()
        values.mul_(weights)
        return values.mean()

    def prune(self):
        redundant_mask = (self.scores > self.scores.mean()).numpy()
        redundant_indices = np.where(redundant_mask)[0]
        remained_indices = np.where(~redundant_mask)[0].tolist()
        selected_indices = np.random.choice(redundant_indices, int(
            self.keep_ratio * len(redundant_indices)), replace=False)
        self.reset_weights()
        if len(selected_indices) > 0:
            self.weights[selected_indices] = 1 / self.keep_ratio
            remained_indices.extend(selected_indices)
        self.num_pruned_samples += len(self.dataset) - len(remained_indices)
        np.random.shuffle(remained_indices)
        return remained_indices

    @property
    def sampler(self):
        sampler = FreqCoreSampler(self)
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedFreqCoreSampler(sampler)
        return sampler

    def no_prune(self):
        samples_indices = list(range(len(self)))
        np.random.shuffle(samples_indices)
        return samples_indices

    def mean_score(self):
        return self.scores.mean()

    def get_weights(self, indexes):
        return self.weights[indexes]

    def get_pruned_count(self):
        return self.num_pruned_samples

    @property
    def stop_prune(self):
        return self.num_epochs * self.delta

    def reset_weights(self):
        self.weights[:] = 1

# FreqCore Sampler
class FreqCoreSampler(object):
    def __init__(self, dataset: FreqCore):
        self.dataset = dataset
        self.stop_prune = dataset.stop_prune
        self.iterations = dataset.iterations
        self.sample_indices = None
        self.iter_obj = None
        self.reset()

    def __getitem__(self, idx):
        return self.sample_indices[idx]

    def reset(self):
        np.random.seed(self.iterations)
        if self.iterations > self.stop_prune:
            if self.iterations == self.stop_prune + 1:
                self.dataset.reset_weights()
            self.sample_indices = self.dataset.no_prune()
        else:
            self.sample_indices = self.dataset.prune()
        self.iter_obj = iter(self.sample_indices)
        self.iterations += 1

    def __next__(self):
        return next(self.iter_obj)  # may raise StopIteration
        
    def __len__(self):
        return len(self.sample_indices)

    def __iter__(self):
        self.reset()
        return self



class DistributedFreqCoreSampler(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler can change size during training.
    """
    class DatasetFromSampler(Dataset):
        def __init__(self, sampler: FreqCoreSampler):
            self.dataset = sampler
 
        def reset(self, ):
            self.indices = None
            self.dataset.reset()

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index: int):
            """Gets element of the dataset.
            Args:
                index: index of the element in the dataset
            Returns:
                Single element by index
            """
            # if self.indices is None:
            #    self.indices = list(self.dataset)
            return self.dataset[index]

    def __init__(self, dataset: FreqCoreSampler, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = True) -> None:
        sampler = self.DatasetFromSampler(dataset)
        super(DistributedFreqCoreSampler, self).__init__(
            sampler, num_replicas, rank, shuffle, seed, drop_last)
        self.sampler = sampler
        self.dataset = sampler.dataset.dataset  # the real dataset.
        self.iter_obj = None

    def __iter__(self) -> Iterator[int]:
        self.sampler.reset()
        if self.drop_last and len(self.sampler) % self.num_replicas != 0:  # type: ignore[arg-type]
            self.num_samples = math.ceil(
                (len(self.sampler) - self.num_replicas) /
                self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(
                len(self.sampler) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # type: ignore[arg-type]
            indices = torch.randperm(len(self.sampler), generator=g).tolist()
        else:
            indices = list(range(len(self.sampler)))  # type: ignore[arg-type]

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size /
                            len(indices)))[:padding_size]
        else:
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.iter_obj = iter(itemgetter(*indices)(self.sampler))
        return self.iter_obj
   
