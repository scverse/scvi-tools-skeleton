import logging
from typing import Optional, Sequence

import numpy as np
import torch
from anndata import AnnData
from scvi.dataloaders import AnnDataLoader
from scvi.lightning import PyroTrainingPlan, Trainer
from scvi.model.base import BaseModelClass

from ._mypyromodule import MyPyroModule

logger = logging.getLogger(__name__)


class MyPyroModel(BaseModelClass):
    """
    Skeleton for a pyro version of a scvi-tools model.

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
        Keyword args for :class:`~mypackage.MyModule`

    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> scvi.data.setup_anndata(adata, batch_key="batch")
    >>> vae = mypackage.MyModel(adata)
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
        super(MyPyroModel, self).__init__(adata, use_gpu=use_gpu)

        # self.summary_stats provides information about anndata dimensions and other tensor info

        self.module = MyPyroModule(
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
        return PyroTrainingPlan

    @property
    def _data_loader_cls(self):
        return AnnDataLoader

    def get_latent(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
    ):
        r"""
        Return the latent representation for each cell.

        This is denoted as :math:`z_n` in our manuscripts.
        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        adata = self._validate_anndata(adata)
        scdl = self._make_scvi_dl(adata=adata, indices=indices, batch_size=batch_size)
        latent = []
        for tensors in scdl:
            qz_m = self.module.get_latent(tensors)
            latent += [qz_m.cpu()]
        return np.array(torch.cat(latent))
