import logging
import warnings
import torch
import numpy as np
from botorch.models import SingleTaskGP, fully_bayesian
from botorch.fit import fit_gpytorch_mll, fit_fully_bayesian_model_nuts
from botorch.optim import optimize_acqf_discrete
from botorch.acquisition import qNoisyExpectedImprovement, qMaxValueEntropy
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from borc.data_processing import (
    OptimizationSettings,
    process_data,
    build_X_y_representation,
    build_combination_space,
    unbuild_representation,
    recover_smiles_from_fingerprint,
    print_candidates,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


GP_LEVEL = 0
ACQF_LEVEL = 0


def main(verbose=False):

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    settings = OptimizationSettings("data/settings.json")

    logger.debug("Settings loaded\n" + settings.display_settings())

    X_raw, y_raw, X_columns, y_columns, index_ranges, smiles_mf_dict = process_data(
        settings
    )
    X_comb, smiles_mf_dict = build_combination_space(
        X_raw, X_columns, settings, smiles_mf_dict
    )
    X, y = build_X_y_representation(X_raw, y_raw)

    mm_scaler = MinMaxScaler()
    st_scaler = StandardScaler()

    Xt = mm_scaler.fit_transform(X)
    
    if y.shape[1] > 1:
        warnings.warn(
            "Multiple target columns detected. Using only the first column for regression."
        )

    yt = st_scaler.fit_transform(y[:, [0]])

    train_x = torch.tensor(Xt, dtype=torch.float64)
    train_y = torch.tensor(yt, dtype=torch.float64)

    logger.debug(f"train_X shape: {train_x.shape}")
    logger.debug(f"train_Y shape: {train_y.shape}")

    if GP_LEVEL == 0:
        logger.debug("Fitting GP model with GPyTorch")
        gp = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

    elif GP_LEVEL == 1:
        logger.debug("Fitting GP model with BoTorch")
        gp = fully_bayesian.SaasFullyBayesianSingleTaskGP(train_x, train_y)
        fit_fully_bayesian_model_nuts(gp)

    logger.debug("GP model fitted")
    logger.debug(f"Predicting with GP model: {gp}")

    test_x = torch.tensor(mm_scaler.transform(X_comb), dtype=torch.float64)

    logger.debug("Optimizing acquisition function")

    if ACQF_LEVEL == 0:
        acqf = qNoisyExpectedImprovement(gp, train_x)
    elif ACQF_LEVEL == 1:
        acqf = qMaxValueEntropy(gp, train_x)

    optim_acqf = optimize_acqf_discrete(
        acqf,
        1,
        test_x,
    )

    cands_x = mm_scaler.inverse_transform(optim_acqf[0].detach().numpy())
    cands_y = (
        st_scaler.inverse_transform(
            np.atleast_2d(gp.posterior(optim_acqf[0].detach()).mean.detach().numpy())
        ),
        st_scaler.inverse_transform(
            np.atleast_2d(gp.posterior(optim_acqf[0].detach()).variance.detach().numpy())
        ),
    )

    logger.debug(f"Optimized candidates X: {cands_x}")
    logger.debug(f"Optimized candidates Y: {cands_y}")

    logger.debug("Building representation of candidates")
    ub_cands = unbuild_representation(cands_x, index_ranges)

    logger.debug(f"Unbuilt candidates: {ub_cands}")

    print_candidates(
        ub_cands, cands_y, settings.molecular_col, settings.tabular_col, smiles_mf_dict
    )


if __name__ == "__main__":
    main(verbose=True)
