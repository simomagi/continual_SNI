import importlib
from argparse import ArgumentParser
from datasets.memory_dataset import MemoryDataset


class ExemplarsDataset(MemoryDataset):
    """Exemplar storage for approaches with an interface of Dataset"""

    def __init__(self, dataset_type, dataset_path, class_indices,
                 num_exemplars=0, num_exemplars_per_class=0, exemplar_selection='random'):
                 
        super().__init__({'x': [], 'y': [], 'ids':[]}, dataset_type, dataset_path, class_indices=class_indices)

         
        self.max_num_exemplars = num_exemplars
        cls_name = "{}ExemplarsSelector".format(exemplar_selection.capitalize())
        selector_cls = getattr(importlib.import_module(name='datasets.exemplars_selection'), cls_name)
        self.exemplars_selector = selector_cls(self)

    # Returns a parser containing the approach specific parameters
    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser("Exemplars Management Parameters")
        _group = parser.add_mutually_exclusive_group()
        _group.add_argument('--num-exemplars', default=0, type=int, required=False,
                            help='Fixed memory, total number of exemplars (default=%(default)s)')
        parser.add_argument('--exemplar-selection', default='random', type=str,
                            choices=['random'],
                            required=False, help='Exemplar selection strategy (default=%(default)s)')
        return parser.parse_known_args(args)

    def _is_active(self):
        return self.max_num_exemplars > 0

    def collect_exemplars(self, model, trn_loader, selection_transform):
        if self._is_active():
            self.images, self.labels, self.ids,  self.dataset_type, self.dataset_path   = self.exemplars_selector(model, trn_loader, selection_transform)
