


from typing import Dict,Any,Protocol, List,Tuple
import torch
import logging
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from openretina.models.core_readout import BaseCoreReadout,UnifiedCoreReadout
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




DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAME_RATE_MODEL = 30.0  # Hz
STIMULUS_SHAPE = (1, 2, 50, 18, 16)




def get_seed_and_path(ckpt_dir: str) -> Dict[int,str]:
    all_files = os.listdir(ckpt_dir)


    seed_and_path = {}
    for file in all_files:
        if not file.endswith(".ckpt"):
            raise ValueError(f"Expected .ckpt files in {ckpt_dir}, but found {file}.")
        if file.endswith(".ckpt") and "seed_" in file:
            seed_str = file.split("seed_")[-1].split(".ckpt")[0]
            seed = int(seed_str)
            seed_and_path[seed] = os.path.join(ckpt_dir, file)
    return seed_and_path


def load_pretrained_model(checkpoint_path: str) -> BaseCoreReadout:

    if not os.path.isfile(checkpoint_path) or not checkpoint_path.endswith(".ckpt"):
        raise ValueError(f"Expected a .ckpt file, but got {checkpoint_path}.")
    model = load_core_readout_model(checkpoint_path,device=DEVICE)
    return model


def prepare_model_for_refinement(model: BaseCoreReadout, data_info: Dict[str, Any]) -> BaseCoreReadout:
    # add new readouts and modify stored data in model
    model.readout.add_sessions(data_info["n_neurons_dict"])  # type: ignore
    model.update_model_data_info(data_info)
    model.core.requires_grad_(False)
    return model

def instanitate_new_model(cfg: DictConfig, data_info: Dict[str, Any]) -> BaseCoreReadout:
    # Assign missing n_neurons_dict to model
    cfg.model.n_neurons_dict = data_info["n_neurons_dict"]

    if hasattr(cfg.model, "_target_"):
        model = hydra.utils.instantiate(cfg.model, data_info=data_info)
    else:        
        model = UnifiedCoreReadout(data_info=data_info, **cfg.model)
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

def instantiate_other_training_components(cfg: DictConfig, dataloaders: Dict[str, Dict[str, data.DataLoader]]):
    """
    Instantiate trainer, train_loader, valid_loader, callbacks
    """
    
    ### Logging
    logger_array = []
    for _, logger_params in cfg.logger.items():
        logger = hydra.utils.instantiate(logger_params)
        logger_array.append(logger)

    train_loader = data.DataLoader(
        LongCycler(dataloaders["train"], shuffle=True), batch_size=None, num_workers=0, pin_memory=True
    )
    valid_loader = ShortCycler(dataloaders["validation"])
    
    ### Callbacks
    callbacks = [
        hydra.utils.instantiate(callback_params) for callback_params in cfg.get("training_callbacks", {}).values()
    ]

    ### Trainer init
    trainer: lightning.Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger_array, callbacks=callbacks)

    return trainer, train_loader, valid_loader, callbacks

def test_single_model(model: BaseCoreReadout, dataloaders: Dict[str, Dict[str, data.DataLoader]],trainer: lightning.Trainer):
    short_cyclers = [(n, ShortCycler(dl)) for n, dl in dataloaders.items()]
    trainer.test(model, dataloaders=[c for _, c in short_cyclers], ckpt_path=None)

    
    


def single_training_or_refinement_wrapper(dataloaders,
                         cfg,
                         data_info,
                         load_model_path=None,
                         seed= None) -> Tuple[BaseCoreReadout | EnsembleModel, str]:


    # in case of training a member of ensemble, we want to set different seeds for each model and not the ones in config
    if seed is not None: 
        lightning.pytorch.seed_everything(seed)
    elif cfg.seed is not None:
        lightning.pytorch.seed_everything(cfg.seed)

    ## Model init
    if load_model_path is None:
        load_model_path = cfg.paths.get("load_model_path")
    

    if  cfg.only_train_readout is True:
        model = load_pretrained_model(load_model_path)
        prepare_model_for_refinement(model, data_info)
    
    else:
        model = instanitate_new_model(cfg, data_info)
        
    trainer, train_loader, valid_loader, callbacks = instantiate_other_training_components(cfg, dataloaders)
   
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


    # get best model path from checkpoint callback
    best_model_path =  next(c for c in callbacks if isinstance(c, lightning.pytorch.callbacks.ModelCheckpoint)).best_model_path
    model = load_core_readout_model(best_model_path, DEVICE)

    test_single_model(model, dataloaders, trainer)

    return model, best_model_path


def refine_ensemble_model(dataloaders: Dict[str, Dict[str, data.DataLoader]],
                          cfg: DictConfig,
                         data_info: Dict[str, Any],
                         dir_with_ckpt: str,) -> Tuple[EnsembleModel, Dict[int,str]]:
    """
    Loads mulitple models from a dir """

    
    seed_and_path = get_seed_and_path(dir_with_ckpt)

    all_models = []
    best_model_path = {}
    for seed, path in seed_and_path.items():

        # perform single training loop for each model
        model,best_path = single_training_or_refinement_wrapper(
            dataloaders, cfg, data_info, load_model_path=path, seed=seed
        )
        all_models.append(model)
        best_model_path[seed] = best_path

    # bind to ensemble and test
    model = EnsembleModel(*all_models)

    return model, best_model_path

def get_dataloaders_and_data_info(cfg: DictConfig,
                   neuron_data_dict:Dict[str,ResponsesTrainTestSplit],
                   movies_dict: MoviesTrainTestSplit,) -> Tuple[Dict[str, Dict[str, data.DataLoader]], Dict[str, Any]]:
    """
    Instantiate dataloaders for all splits and sessions
    """


    if cfg.check_stimuli_responses_match:
        for _, neuron_data in neuron_data_dict.items():
            neuron_data.check_matching_stimulus(movies_dict)

    dataloaders = hydra.utils.instantiate( # dict[str, dict[str, DataLoader]]
        cfg.dataloader,
        neuron_data_dictionary=neuron_data_dict,
        movies_dictionary=movies_dict,
    )
    data_info = compute_data_info(neuron_data_dict, movies_dict)

    return dataloaders, data_info


def train_or_refine_member_or_ensemble(model_configs: DictConfig,
                                        dataloaders: Dict[str, Dict[str, data.DataLoader]],
                                        data_info: Dict[str, Any],
                                        ) -> \
                                        Tuple[EnsembleModel | BaseCoreReadout, Dict[int,str] | str]:
                        


    is_ensemble_model =model_configs.get("is_ensemble_model", False)
    if is_ensemble_model:
        model, best_model_path = refine_ensemble_model(dataloaders, model_configs, data_info, dir_with_ckpt=model_configs.paths.load_model_path)
    else:

        model, best_model_path = single_training_or_refinement_wrapper(dataloaders, model_configs, data_info, )

    return model, best_model_path


def get_single_neuron_split_predictions(dataloaders ,                                           
                                        model: BaseCoreReadout | EnsembleModel, 
                                           split = "test",
                                           only_this_session_id = None):
    """
    Gets predicted and actual responses for all neurons in a certain split.
    """

    assert split in dataloaders, f"Split {split} not found in dataloaders."

    
    if len(dataloaders[split]) > 1:
        if only_this_session_id is None:
            raise ValueError(f"Found more than one online training session and only_this_session_id is None.")
        
        # Create a subset with just the specified session
        sessions_to_use = {only_this_session_id: dataloaders[split][only_this_session_id]}
    else:
        # Use all sessions in this split
        sessions_to_use = dataloaders[split]

    all_preds = {}
    all_targets = {}
    for session_id, session_dataloader in sessions_to_use.items():
        all_preds_session, all_targets_session = [], []
        
        # Run model on all test batches
        with torch.no_grad():
            model.eval()
            model.to(DEVICE)
            for batch in session_dataloader:
                inputs, targets = batch
                inputs = inputs.to(DEVICE)
                predictions = model(inputs, data_key=session_id)
                all_preds_session.append(predictions.squeeze().cpu()) # remove batch dim
                all_targets_session.append(targets.squeeze().cpu())
        
        # Concatenate batch results and compute correlations
        all_preds[session_id] = torch.cat(all_preds_session, dim=0).numpy()
        all_targets[session_id] = torch.cat(all_targets_session, dim=0).numpy()
        
        # same nr of neurons 
        assert  all_preds[session_id].shape[1] == all_targets[session_id].shape[1], f"Number of neurons in predictions and targets do not match for session {session_id}."

    return all_preds, all_targets # (Dict[str, (time, neurons) ],Dict[str,np.ndarray])



def get_single_neuron_session_correlations(all_preds: Dict[str,np.ndarray],
                                           all_targets: Dict[str,np.ndarray],
                                           ) -> Dict[str,Dict[int,float]]:
    """Calculate the correlation between model predictions and targets for each neuron in each session in the session of a certain split."""
    
    all_sessions = list(all_preds.keys())


    
    neuron_correlations ={session: {} for session in all_sessions}
    for session_id in all_sessions:
        session_correlations = {}
        pred_session = all_preds[session_id]
        target_session = all_targets[session_id]
        conv_eats_n_frames = target_session.shape[0] - pred_session.shape[0]  
        num_neurons = pred_session.shape[1]

        for i in range(num_neurons):
            pred = pred_session[:,i].flatten()
            target = target_session[conv_eats_n_frames:,i].flatten()

            corr = np.corrcoef(pred, target)[0, 1] if np.var(pred) > 0 and np.var(target) > 0 else 0
            session_correlations[i] = corr
            
        neuron_correlations[session_id]= session_correlations


    
    return neuron_correlations
        





def load_stimuli(or_config: DictConfig):

    movies_dict: MoviesTrainTestSplit = hydra.utils.call(or_config.data_io.stimuli) 
    return movies_dict
   
