import scvi
from scvi.data import synthetic_iid
from mypackage import MyModel, MyPyroModel
import pyro


from mypackage import MyPyroModule
from scvi.dataloaders import AnnDataLoader
from scvi.lightning import PyroTrainingPlan, Trainer
import torch

def test_mymodel():
    n_latent = 5
    adata = synthetic_iid()
    model = MyModel(adata, n_latent=n_latent)
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.get_elbo()
    model.get_latent_representation()
    model.get_marginal_ll(n_mc_samples=5)
    model.get_reconstruction_error()
    model.history

    # tests __repr__
    print(model)

def test_pyro_scvi(): # this works
    use_gpu = int(torch.cuda.is_available())
    adata = synthetic_iid()
    train_dl = AnnDataLoader(adata, shuffle=True, batch_size=128)
    pyro.clear_param_store()
    model = MyPyroModule(adata.n_vars, 10, 128, 1)
    plan = PyroTrainingPlan(model)
    trainer = Trainer(
        gpus=use_gpu,
        max_epochs=2,
    )
    trainer.fit(plan, train_dl)

def test_pyro_scvi_mixt(): # this works
    use_gpu = int(torch.cuda.is_available())
    adata = synthetic_iid()
    train_dl = AnnDataLoader(adata, shuffle=True, batch_size=128)
    pyro.clear_param_store()
    model = MyPyroModel(adata)
    plan = PyroTrainingPlan(model.module)
    trainer = Trainer(
        gpus=use_gpu,
        max_epochs=2,
    )
    trainer.fit(plan, train_dl)

def test_pyro_scvi_mixt_2(): # this fails because it tries to send the indices as well
    use_gpu = int(torch.cuda.is_available())
    adata = synthetic_iid()
    pyro.clear_param_store()
    model = MyPyroModel(adata)
    train_dl, val_dl, test_dl = model._train_test_val_split(
            adata,
            train_size=1,
            batch_size=128,
        )
    train_indices_ = train_dl.indices
    plan = PyroTrainingPlan(model.module, len(train_indices_)) # that causes the problem
    trainer = Trainer(
        gpus=use_gpu,
        max_epochs=2,
    )
    trainer.fit(plan, train_dl)

def test_mypyromodel():
    n_latent = 5
    adata = synthetic_iid()
    pyro.clear_param_store()
    model = MyPyroModel(adata)
    model.train(max_epochs=1, train_size=1)
    model.get_elbo()
    model.get_latent_representation()
    model.get_marginal_ll(n_mc_samples=5)
    model.get_reconstruction_error()
    model.history

    # tests __repr__
    print(model)
