


from typing import Dict,Any,Protocol, List,Tuple
import torch
import logging
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from openretina.models.core_readout import BaseCoreReadout
from openretina.models.core_readout import load_core_readout_model
from openretina.data_io.base import MoviesTrainTestSplit, ResponsesTrainTestSplit



import lightning.pytorch
import torch.utils.data as data

from openretina.data_io.base import compute_data_info
from openretina.data_io.cyclers import LongCycler, ShortCycler



import matplotlib.pyplot as plt
import numpy as np
import torch



from openretina.models.core_readout import load_core_readout_model
from openretina.modules.layers.ensemble import EnsembleModel


log = logging.getLogger(__name__)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAME_RATE_MODEL = 30.0  # Hz
STIMULUS_SHAPE = (1, 2, 50, 18, 16)




def get_seed_and_path(ckpt_dir: str) -> Dict[int,str]:
    seed_and_path = {}
    for file in os.listdir(ckpt_dir):
        if file.endswith(".ckpt") and "seed_" in file:
            seed_str = file.split("seed_")[-1].split(".ckpt")[0]
            try:
                seed = int(seed_str)
                seed_and_path[seed] = os.path.join(ckpt_dir, file)
            except ValueError:
                log.warning(f"Could not convert {seed_str} to int.")
    return seed_and_path


def load_pretrained_model(checkpoint_path: str) -> BaseCoreReadout:
    is_gru_model = "gru" in checkpoint_path
    model = load_core_readout_model(checkpoint_path,device=DEVICE, is_gru_model=is_gru_model)
    return model

def load_pretrained_ensemble_model(ckpt_dir: str,seeds: List[str],set_eval=True) -> EnsembleModel:
    models = []
    for seed in seeds:
        ckpt_path = os.path.join(ckpt_dir, f"seed_{seed}.ckpt")
        print(f"Loading model from {ckpt_path}")
        model = load_pretrained_model(ckpt_path)
        if set_eval:
            model.eval()
        models.append(model)

    ensemble_model = EnsembleModel(*models)
    return ensemble_model


def single_training_loop(dataloaders,cfg,data_info,log,load_model_path=None,seed= None):
    train_loader = data.DataLoader(
        LongCycler(dataloaders["train"], shuffle=True), batch_size=None, num_workers=0, pin_memory=True
    )
    valid_loader = ShortCycler(dataloaders["validation"])

    if seed is not None:
        lightning.pytorch.seed_everything(seed)
    elif cfg.seed is not None:
        lightning.pytorch.seed_everything(cfg.seed)

    ## Model init
    if load_model_path is None:
        load_model_path = cfg.paths.get("load_model_path")
    
    log.info(f"Loading model from <{load_model_path}>")
    is_gru_model = "gru" in cfg.model._target_.lower() if hasattr(cfg.model, "_target_") else False
    model = load_core_readout_model(load_model_path, DEVICE, is_gru_model=is_gru_model)

    
    # add new readouts and modify stored data in model
    model.readout.add_sessions(data_info["n_neurons_dict"])  # type: ignore
    model.update_model_data_info(data_info)

    if cfg.get("only_train_readout") is True:
        log.info("Only training readout, core model parameters will be frozen.")
        model.core.requires_grad_(False)

    ### Logging
    log.info("Instantiating loggers...")
    logger_array = []
    for _, logger_params in cfg.logger.items():
        logger = hydra.utils.instantiate(logger_params)
        logger_array.append(logger)

    ### Callbacks
    log.info("Instantiating callbacks...")
    callbacks = [
        hydra.utils.instantiate(callback_params) for callback_params in cfg.get("training_callbacks", {}).values()
    ]

    ### Trainer init
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: lightning.Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger_array, callbacks=callbacks)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


    # Get the checkpoint callback from your callbacks list
    checkpoint_callback = next(c for c in callbacks if isinstance(c, lightning.pytorch.callbacks.ModelCheckpoint))
    if checkpoint_callback is None:
        raise RuntimeError("No ModelCheckpoint callback found. Ensure save_top_k >= 1 and monitor are set.")


    # After training completes:
    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path:
        raise RuntimeError("best_model_path is empty. Check ModelCheckpoint(monitor=..., save_top_k>=1).")


    # load best model for testing
    log.info(f"Loading best model from {best_model_path} for testing and inference.")

    #DEBUG:MAKE SURE SESSION DATA INFO AND NNEURONS DICT IS THERE 
    model = load_core_readout_model(best_model_path, DEVICE, is_gru_model=is_gru_model)

    log.info("All dataloaders...")
    short_cyclers = [(n, ShortCycler(dl)) for n, dl in dataloaders.items()]
    dataloader_mapping = {f"DataLoader {i}": x[0] for i, x in enumerate(short_cyclers)}
    log.info(f"Dataloader mapping: {dataloader_mapping}")
    # test with model passed
    trainer.test(model, dataloaders=[c for _, c in short_cyclers], ckpt_path=None)


    ### Testing
    log.info("Testing individual neurons!")
    neuron_testset_correl = get_single_neuron_test_correlations(dataloaders, model)
    log.info(f"Test set neuron correlations statistics (mean,std,min,max): {[func(list(neuron_testset_correl.values())) for func in [np.mean, np.std, np.min, np.max]]}")
    
    return model,neuron_testset_correl, best_model_path


#@time_it
def train_model_online(cfg: DictConfig,
                       neuron_data_dict:Dict[str,ResponsesTrainTestSplit],
                       movies_dict: MoviesTrainTestSplit) -> \
                       Tuple[EnsembleModel | BaseCoreReadout, Dict[int,float], Dict[int,str] | str]:
    
    log.info("Logging full config:")
    log.info(OmegaConf.to_yaml(cfg))

    if cfg.paths.cache_dir is None:
        raise ValueError("Please provide a cache_dir for the data in the config file or as a command line argument.")

    ### Set cache folder
    os.environ["OPENRETINA_CACHE_DIRECTORY"] = cfg.paths.cache_dir

    ### Display log directory for ease of access
    log.info(f"Saving run logs at: {cfg.paths.output_dir}")

  
    if cfg.check_stimuli_responses_match:
        for _, neuron_data in neuron_data_dict.items():
            neuron_data.check_matching_stimulus(movies_dict)

    dataloaders = hydra.utils.instantiate( # dict[str, dict[str, DataLoader]]
        cfg.dataloader,
        neuron_data_dictionary=neuron_data_dict,
        movies_dictionary=movies_dict,
    )

    data_info = compute_data_info(neuron_data_dict, movies_dict)

    is_ensemble_model =cfg.get("is_ensemble_model", False)
    log.info(f"Is ensemble model: {is_ensemble_model}")
    if is_ensemble_model:
        assert os.path.isdir(cfg.paths.load_model_path), "For ensemble model, load_model_path should be a directory containing seed_x.ckpt files."
        seed_and_path = get_seed_and_path(cfg.paths.load_model_path)
        log.info(f"Found {len(seed_and_path)} seeds in {cfg.paths.load_model_path}: {list(seed_and_path.keys())}")

        all_models = []
        best_model_path = {}
        for seed, path in seed_and_path.items():
            log.info(f"Loading model for seed {seed} from {path}")

            # perform single training loop for each model
            model,_,best_path = single_training_loop(dataloaders, 
                                                        cfg, data_info, 
                                                        log, 
                                                        load_model_path=path,
                                                        seed=seed)
            all_models.append(model)
            best_model_path[seed] = best_path

        # bind to ensemble and test
        model = EnsembleModel(*all_models)
        neuron_testset_correl = get_single_neuron_test_correlations(dataloaders, model)
    else:

        model,neuron_testset_correl, best_model_path = single_training_loop(dataloaders, cfg, data_info, log)

    return model,neuron_testset_correl, best_model_path



def get_single_neuron_test_correlations(dataloaders , model: BaseCoreReadout | EnsembleModel) ->  Dict[int, float]:
    """Calculate the correlation between model predictions and targets for each neuron in each session in the test session."""

    neuron_correlations = {}
    if len(dataloaders["test"]) != 1:
        raise ValueError(f"Expected only one session in the test set for online training. but found {len(dataloaders['test'])} sessions.")


    for session_id, session_dataloader in dataloaders["test"].items():
        all_preds, all_targets = [], []
        
        # Run model on all test batches
        with torch.no_grad():
            model.eval()
            model.to(DEVICE)
            for batch in session_dataloader:
                inputs, targets = batch
                inputs = inputs.to(DEVICE)
                predictions = model(inputs, data_key=session_id)
                all_preds.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Concatenate batch results and compute correlations
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        
        # Fast vectorized correlation calculation
        session_correlations = {}
        num_neurons = all_targets.shape[-1]

        conv_eats_n_frames = all_targets.shape[1] - all_preds.shape[1]
        for i in range(num_neurons):
            pred = all_preds[..., i].flatten()
            target = all_targets[:,conv_eats_n_frames:, i].flatten()
            corr = np.corrcoef(pred, target)[0, 1] if np.var(pred) > 0 and np.var(target) > 0 else 0
            session_correlations[i] = corr
            
        neuron_correlations = session_correlations

    return neuron_correlations
        





def load_stimuli(or_config: DictConfig):

    movies_dict: MoviesTrainTestSplit = hydra.utils.call(or_config.data_io.stimuli) 
    return movies_dict
   
