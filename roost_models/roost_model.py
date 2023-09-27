import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from roost.roost.model import Roost
from roost.roost.data import CompositionData, collate_batch
from roost.core import Normalizer
import pickle5 as pickle

@torch.no_grad()
def results_multitask(  # noqa: C901
    model_class,
    model_name,
    run_id,
    ensemble_folds,
    test_set,
    data_params,
    robust,
    task_dict,
    device,
    model_file, # name of model eg. 'oqmd-form-enthalpy.tar'
    eval_type="checkpoint",
    print_results=True,
    save_results=True,
):
    """
    take an ensemble of models and evaluate their performance on the test set
    """

    # assert print_results or save_results, (
    #     "Evaluating Model pointless if both 'print_results' and "
    #     "'save_results' are False."
    # )

    # print(
    #     "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
    #     "------------Evaluate model on Test Set------------\n"
    #     "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
    # )

    test_generator = DataLoader(test_set, **data_params)

    results_dict = {n: {} for n in task_dict}
    for name, task in task_dict.items():
        if task == "regression":
            results_dict[name]["pred"] = np.zeros((ensemble_folds, len(test_set)))
            if robust:
                results_dict[name]["ale"] = np.zeros((ensemble_folds, len(test_set)))

        # elif task == "classification":
        #     results_dict[name]["logits"] = []
        #     results_dict[name]["pre-logits"] = []
        #     if robust:
        #         results_dict[name]["pre-logits_ale"] = []

    for j in range(ensemble_folds):

        if ensemble_folds == 1:
            resume = f"models/" + model_file
            # print("Evaluating Model")
        # else:
        #     resume = f"models/{model_name}/{eval_type}-r{j}.pth.tar"
        #     print(f"Evaluating Model {j + 1}/{ensemble_folds}")

        try: # If script run from roost_models folder
            assert os.path.isfile(resume), f"no checkpoint found at '{resume}'"
            checkpoint = torch.load(resume, map_location=device)
        except AssertionError: # If script run from dqn folder
            checkpoint = torch.load('roost_models/'+resume, map_location=device)
        

        normalizer_dict = {}
        for task, state_dict in checkpoint["normalizer_dict"].items():
            if state_dict is not None:
                normalizer_dict[task] = Normalizer.from_state_dict(state_dict)
            else:
                normalizer_dict[task] = None

        # assert (
        #     checkpoint["model_params"]["robust"] == robust
        # ), f"robustness of checkpoint '{resume}' is not {robust}"

        # assert (
        #     checkpoint["model_params"]["task_dict"] == task_dict
        # ), f"task_dict of checkpoint '{resume}' does not match current task_dict"

        model = model_class(**checkpoint["model_params"], device=device)
        model.to(device)
        model.load_state_dict(checkpoint["state_dict"])

        y_test, output, *ids = model.predict(generator=test_generator)

        # TODO should output also be a dictionary?

        for pred, target, (name, task) in zip(output, y_test, model.task_dict.items()):
            if task == "regression":
                if model.robust:
                    mean, log_std = pred.chunk(2, dim=1)
                    pred = normalizer_dict[name].denorm(mean.data.cpu())
                    ale_std = torch.exp(log_std).data.cpu() * normalizer_dict[name].std
                    results_dict[name]["ale"][j, :] = ale_std.view(-1).numpy()
                else:
                    pred = normalizer_dict[name].denorm(pred.data.cpu())
                results_dict[name]["pred"][j, :] = pred.view(-1).numpy()

            elif task == "classification":
                if model.robust:
                    mean, log_std = pred.chunk(2, dim=1)
                    logits = (
                        sampled_softmax(mean, log_std, samples=10).data.cpu().numpy()
                    )
                    pre_logits = mean.data.cpu().numpy()
                    pre_logits_std = torch.exp(log_std).data.cpu().numpy()
                    results_dict[name]["pre-logits_ale"].append(pre_logits_std)
                else:
                    pre_logits = pred.data.cpu().numpy()
                    logits = softmax(pre_logits, axis=1)

                results_dict[name]["pre-logits"].append(pre_logits)
                results_dict[name]["logits"].append(logits)

            results_dict[name]["target"] = target

    # # TODO cleaner way to get identifier names
    # if save_results:
    #     save_results_dict(
    #         dict(zip(test_generator.dataset.dataset.identifiers, ids)),
    #         results_dict,
    #         model_name,
    #     )

    # if print_results:
    #     for name, task in task_dict.items():
    #         print(f"\nTask: '{name}' on Test Set")
    #         if task == "regression":
    #             print_metrics_regression(**results_dict[name])
    #         elif task == "classification":
    #             print_metrics_classification(**results_dict[name])
    if 'enthalpy' in model_file:
        result = results_dict['formation_energy_per_atom']['pred'][0][0]
    elif 'bulk' in model_file:
        result = results_dict['log_K_VRH']['pred'][0][0]
    elif 'shear' in model_file:
        result = results_dict['log_G_VRH']['pred'][0][0]
    elif 'band' in model_file:
        result = results_dict['band_gap']['pred'][0][0]
    return result


def predict_formation_energy(material):
    '''
    Given a material composition, predict formation energy using ROOST

    Args:
    material: Str.

    Returns:
    form_e_pred: float64. Predicted formation energy per atom
    '''
    task_dict={'formation_energy_per_atom': 'regression'}
    test_set = CompositionData(
                material = material,
                # material = 'NaCl',
                # data_path="data/datasets/roost/oqmd-form-enthalpy.csv",
                fea_path="/home/jupyter/Elton/RL/DQN/roost_models/data/el-embeddings/matscholar-embedding.json",
                task_dict=task_dict
            )

    form_e_pred = results_multitask(
                    model_class=Roost,
                    model_name=' ',
                    run_id=0,
                    ensemble_folds=1,
                    test_set=test_set, # <torch.utils.data.dataset.Subset object at 0x7f549e0a2710>
                    data_params={'batch_size': 2048, 'num_workers': 0, 'pin_memory': False, 'shuffle': False, 'collate_fn': collate_batch},
                    robust=False,
                    task_dict=task_dict,
                    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                    eval_type="checkpoint",
                    model_file = 'oqmd-form-enthalpy.tar'
                )
    del test_set
    return form_e_pred

def predict_bulk_mod(material):
    '''
    Given a material composition, predict formation energy using ROOST

    Args:
    material: Str.

    Returns:
    bulk_mod_pred: float64. Predicted bulk modulus
    '''
    task_dict={'log_K_VRH': 'regression'}
    test_set = CompositionData(
                material = material,
                # material = 'NaCl',
                # data_path="data/datasets/roost/oqmd-form-enthalpy.csv",
                fea_path="data/el-embeddings/matscholar-embedding.json",
                task_dict=task_dict
            )

    bulk_mod_pred = results_multitask(
                    model_class=Roost,
                    model_name=' ',
                    run_id=0,
                    ensemble_folds=1,
                    test_set=test_set, # <torch.utils.data.dataset.Subset object at 0x7f549e0a2710>
                    data_params={'batch_size': 2048, 'num_workers': 0, 'pin_memory': False, 'shuffle': False, 'collate_fn': collate_batch},
                    robust=False,
                    task_dict=task_dict,
                    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                    eval_type="checkpoint",
                    model_file = 'mp-bulk-mod.tar'
                )
    del test_set
    return bulk_mod_pred

def predict_shear_mod(material):
    '''
    Given a material composition, predict formation energy using ROOST

    Args:
    material: Str.

    Returns:
    shear_mod_pred: float64. Predicted shear modulus
    '''
    task_dict={'log_G_VRH': 'regression'}
    test_set = CompositionData(
                material = material,
                # material = 'NaCl',
                # data_path="data/datasets/roost/oqmd-form-enthalpy.csv",
                fea_path="data/el-embeddings/matscholar-embedding.json",
                task_dict=task_dict
            )

    shear_mod_pred = results_multitask(
                    model_class=Roost,
                    model_name=' ',
                    run_id=0,
                    ensemble_folds=1,
                    test_set=test_set, # <torch.utils.data.dataset.Subset object at 0x7f549e0a2710>
                    data_params={'batch_size': 2048, 'num_workers': 0, 'pin_memory': False, 'shuffle': False, 'collate_fn': collate_batch},
                    robust=False,
                    task_dict=task_dict,
                    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                    eval_type="checkpoint",
                    model_file = 'mp-shear-mod.tar'
                )
    return shear_mod_pred

def predict_band_gap(material):
    '''
    Given a material composition, predict band gap using ROOST

    Args:
    material: Str.

    Returns:
    band_gap_pred: float64. Predicted band gap
    '''
    task_dict={'band_gap': 'regression'}
    test_set = CompositionData(
                material = material,
                # material = 'NaCl',
                # data_path="data/datasets/roost/oqmd-form-enthalpy.csv",
                fea_path="data/el-embeddings/matscholar-embedding.json",
                task_dict=task_dict
            )

    band_gap_pred = results_multitask(
                    model_class=Roost,
                    model_name=' ',
                    run_id=0,
                    ensemble_folds=1,
                    test_set=test_set, # <torch.utils.data.dataset.Subset object at 0x7f549e0a2710>
                    data_params={'batch_size': 2048, 'num_workers': 0, 'pin_memory': False, 'shuffle': False, 'collate_fn': collate_batch},
                    robust=False,
                    task_dict=task_dict,
                    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                    eval_type="checkpoint",
                    model_file = 'mp-band-gap.tar'
                )
    return band_gap_pred

#
if __name__ == "__main__":

    # # Predict form e of valid compounds from constrained DQN (1000 cxompounds)
    # with open('../training_data/constrained_DQN_cuda/final_compounds_RF_constrained.pkl', 'rb') as f: # For final compounds generated using constrained RL (DQN + DCN)
    # # with open('../training_data/compounds_vs_iter_RF.pkl', 'rb') as f: # For final compounds generated using constrained RL (DQN + DCN)
    #     final_compounds = pickle.load(f)
    # # final_compounds = final_compounds[-1]

    # print(len(final_compounds))
    # import time
    # start = time.time()
    # form_e = []
    # for compound in final_compounds:
    #     form_e.append(predict_formation_energy(compound))
    # end = time.time()
    # print('time taken for %.0f formation energy predictions: ' % len(final_compounds), end-start)
    # print(form_e)

    # Single compound prediction
    print(predict_formation_energy('Ge2As6Pb7S14'))