from scvi.data import synthetic_iid
from scskeleton import MyModel


def test_mymodel():
    n_latent = 5
    adata = synthetic_iid()
    model = MyModel(adata, n_latent=n_latent)
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)

    # tests __repr__
    print(model)
