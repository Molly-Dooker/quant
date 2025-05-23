from abc import ABCMeta, abstractmethod
from bos_metal import op

class BaseDenseHead(op.BaseModule, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self, ):
        super(BaseDenseHead, self).__init__()

    # @abstractmethod
    def get_bboxes(self, **kwargs):
        """Transform network output for a batch into bbox predictions."""
        pass

    def simple_test(self, feats, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        return self.simple_test_bboxes(feats, img_metas, rescale=rescale)
