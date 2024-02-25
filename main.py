import logging

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2,3, 4, 5, 6, 7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
#torch.set_num_threads(12)

logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="defaults")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(42)
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    # Instantiate all modules specified in the configs
    model = hydra.utils.instantiate(
        cfg.model,  # Object to instantiate
        # Overwrite arguments at runtime that depends on other modules
        net=cfg.net,
        loss_fun=cfg.loss,
        type = cfg.type,
        patch_size = cfg.data.patch_size,
        # Don't instantiate optimizer submodules with hydra, let `configure_optimizers()` do it
        _recursive_=False,
    )

    data_module = hydra.utils.instantiate(cfg.data)

    # Let hydra manage direcotry outputs
    tensorboard = pl.loggers.TensorBoardLogger(".", "", "", log_graph=False, default_hp_metric=False)
    callbacks = [
        pl.callbacks.ModelCheckpoint(monitor='loss/val', save_top_k=3, save_last=True),
        #pl.callbacks.EarlyStopping(monitor='loss_epoch/val', patience=100),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        pl.callbacks.RichProgressBar(),
        pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-4)
    ]

    trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer),
        logger=tensorboard,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=data_module)
    # trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    main()
