import logging

from anndata import AnnData

from scvi._compat import Literal
from scvi.dataloaders import AnnDataLoader
from scvi.lightning import TrainingPlan
from scvi.model.base import BaseModelClass, VAEMixin

from ._module import MyModule

logger = logging.getLogger(__name__)


class MyModel(VAEMixin, BaseModelClass):
    """
    Skeleton for an scvi-tools model.

    Please use this skeleton to create new models.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :func:`~scvi.data.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    use_gpu
        Use the GPU or not.
    **model_kwargs
        Keyword args for :class:`~scskeleton.MyModule`

    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> scvi.data.setup_anndata(adata, batch_key="batch")
    >>> vae = scskeleton.MyModel(adata)
    >>> vae.train()
    >>> adata.obsm["X_mymodel"] = vae.get_latent_representation()
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        use_gpu: bool = True,
        **model_kwargs,
    ):
        super(MyModel, self).__init__(adata, use_gpu=use_gpu)

        # self.summary_stats provides information about anndata dimensions and other tensor info

        self.module = MyModule(
            n_input=self.summary_stats["n_vars"],
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            **model_kwargs,
        )
        self._model_summary_string = "Overwrite this attribute to get an informative representation for your model"
        # necessary line to get params that will be used for saving/loading
        self.init_params_ = self._get_init_params(locals())

        logger.info("The model has been initialized")

    @property
    def _plan_class(self):
        return TrainingPlan

    @property
    def _data_loader_cls(self):
        return AnnDataLoader
