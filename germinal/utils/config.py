"""Germinal Configuration Management Module

This module provides comprehensive configuration management for the Germinal protein
design system using Hydra framework. It handles configuration processing, validation,
and initialization of design runs with proper device checking and directory structure
setup.

The module implements:
- Hydra configuration processing and conversion
- Design run initialization with device validation
- Directory structure creation and management
- Starting structure generation and validation
- CDR position computation and binder sequence extraction

Key Functions:
    process_config: Convert Hydra configuration to system format
    initialize_germinal_run: Initialize complete design run with validation

Dependencies:
    - Hydra/OmegaConf for configuration management
    - JAX/Torch for device validation
    - Germinal utilities for structure processing
    - Bio libraries for PDB manipulation
"""

import os
from typing import Dict, Any
from omegaconf import DictConfig, OmegaConf
import warnings

from germinal.utils.utils import (
    get_jax_device,
    get_torch_device,
    compute_cdr_positions,
    create_starting_structure,
    get_sequence_from_pdb,
)
from germinal.utils.io import RunLayout, IO


def process_config(cfg: DictConfig) -> Dict[str, Any]:
    """Process Hydra configuration and convert to system-expected format.
    
    Takes a Hydra DictConfig object and converts it into the standardized format
    expected by the Germinal system. Separates configuration into run parameters,
    target specifications, and filtering criteria.
    
    Args:
        cfg (DictConfig): Hydra configuration object containing nested configuration
            sections including 'target', 'filter', and run parameters.
            
    Returns:
        Dict[str, Any]: Processed configuration dictionary with four main keys:
            - 'run': Run configuration parameters (excluding target and filter sections)
            - 'target': Target-specific configuration parameters
            - 'filters_initial': Initial filtering criteria for trajectory screening
            - 'filters_final': Final filtering criteria for design acceptance
    """
    # Convert OmegaConf to regular dict for compatibility
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Extract the main sections
    target_config = config_dict.get("target", {})
    filter_config = config_dict.get("filter", {})

    # Build run config by excluding target and filter sections
    run_config = {k: v for k, v in config_dict.items() if k not in ["target", "filter"]}

    # Extract initial and final filters
    filters_initial = filter_config.get("initial", {})
    filters_final = filter_config.get("final", {})

    # Return the four-key structure as requested
    processed_cfg = {
        "run": run_config,
        "target": target_config,
        "filters_initial": filters_initial,
        "filters_final": filters_final,
    }

    return processed_cfg


def initialize_germinal_run(
    run_settings: Dict[str, Any], target_settings: Dict[str, Any]
):
    """Initialize a complete Germinal design run with validation and setup.
    
    Performs comprehensive initialization of a Germinal protein design run including
    device validation, directory structure creation, starting structure generation,
    and configuration of all necessary parameters for the design process.
    
    The function validates computational resources (JAX/CUDA availability), creates
    the required directory structure, generates or validates starting PDB complexes,
    computes CDR positions, and configures model parameters for the design run.
    
    Args:
        run_settings (Dict[str, Any]): Run configuration parameters including:
            - project_dir: Base project directory path
            - results_dir: Results directory name
            - experiment_name: Unique experiment identifier
            - cdr_lengths: List of CDR lengths
            - fw_lengths: Framework region lengths
            - type: Binder type ('nb' for nanobody, 'scfv' for single-chain Fv)
            - use_multimer_design: Whether to use multimer models
            - bias_redesign: Bias value for redesign (negative values set to False)
            
        target_settings (Dict[str, Any]): Target-specific configuration including:
            - target_name: Name of the target protein
            - target_pdb_path: Path to target PDB structure
            - binder_chain: Chain identifier for binder (default 'B')
            - target_chain: Chain identifier for target (default 'A')
            
    Returns:
        tuple: (io, run_settings) where:
            - io (IO): Initialized I/O handler for the run with directory structure
            - run_settings (Dict[str, Any]): Updated run settings with additional fields:
                * cdr_positions: Computed CDR residue positions
                * starting_binder_seq: Extracted binder sequence from starting structure
                * starting_pdb_complex: Path to the starting PDB complex
                * design_models: List of model indices for AF2 design
                
    Raises:
        AssertionError: If JAX device is not available or CUDA is not accessible
        AssertionError: If target PDB path does not exist
    """
    # Validate computational device availability for design execution
    assert get_jax_device(), "JAX device not available"
    if get_torch_device() != "cuda":
        warnings.warn("Torch device not available")
    # Construct hierarchical directory path for design outputs
    design_path = os.path.join(
        run_settings.get("project_dir", "."),
        run_settings.get("results_dir", "results"),
        run_settings.get("experiment_name", "germinal_run"),
        run_settings.get("run_config", ""),
    )
    # Initialize directory structure and I/O handler
    design_paths = RunLayout.create(design_path)
    # Persist configuration parameters to file system
    io = IO(design_paths)
    io.save_run_config(run_settings, target_settings)

    # Compute CDR residue positions from framework and CDR lengths
    cdr_str = "_".join(str(i) for i in run_settings.get("cdr_lengths"))
    cdr_positions = compute_cdr_positions(
        run_settings.get("cdr_lengths"), run_settings.get("fw_lengths")
    )
    run_settings["cdr_positions"] = cdr_positions
    # Determine path for starting PDB complex based on binder type
    binder_type = run_settings.get("type", "nb")
    if binder_type == "nb":
        complex_name = f"{target_settings.get('target_name')}_{cdr_str}_nb"
    else:  # scfv
        complex_name = f"{target_settings.get('target_name')}_{cdr_str}_scfv"
    starting_pdb_complex = os.path.join(
        run_settings.get("pdb_dir", "pdbs"), f"{complex_name}.pdb"
    )
    # Generate starting complex if not present, otherwise use existing structure
    if not os.path.exists(starting_pdb_complex):
        target_pdb_path = target_settings.get("target_pdb_path")
        assert os.path.exists(target_pdb_path), (
            f"Target PDB path does not exist: {target_pdb_path}"
        )
        template_binder_pdb = os.path.join(
            run_settings.get("pdb_dir", "pdbs"), f"{binder_type}.pdb"
        )
        # Build a combined starting complex. If multiple target chains are provided
        # (e.g., "A,B,C"), they will be concatenated into a single chain 'A' with
        # a 50-residue numbering gap between successive chains.
        create_starting_structure(
            starting_pdb_complex,
            template_binder_pdb,
            target_pdb_path,
            binder_chain=target_settings.get("binder_chain", "B"),
            target_chain=target_settings.get("target_chain", "A"),
        )

        # Normalize target chain to 'A' post-concatenation and remap hotspots
        chains_field = target_settings.get("target_chain", "A")
        # Build ordered list of chains from provided field (supports list or comma-separated string)
        if isinstance(chains_field, (list, tuple)):
            chain_order = [str(c).strip() for c in chains_field if str(c).strip()]
            multi_chain = len(chain_order) > 1
        else:
            chain_order = [c.strip() for c in str(chains_field).split(",") if c.strip()]
            multi_chain = "," in str(chains_field)

        if multi_chain:
            # Update target_chain to single 'A'
            target_settings["target_chain"] = "A"

            # Remap hotspots, if provided, from original chains to concatenated 'A'
            hotspots = target_settings.get("target_hotspots", "")
            if hotspots:
                # Compute original chain lengths from the original target PDB
                chain_seqs = get_sequence_from_pdb(target_pdb_path)
                gap = 50
                # Precompute offsets for each chain in order
                offsets = {}
                running = 0
                for idx, ch in enumerate(chain_order):
                    if idx > 0:
                        running += gap
                    offsets[ch] = running
                    running += len(chain_seqs.get(ch, ""))

                def _remap_token(tok: str) -> str:
                    tok = tok.strip()
                    if not tok:
                        return ""
                    # Determine chain label and residue spec
                    if tok[0].isalpha():
                        ch = tok[0]
                        rest = tok[1:]
                    else:
                        # default to first chain when none specified
                        ch = chain_order[0]
                        rest = tok
                    off = offsets.get(ch, 0)
                    if "-" in rest:
                        s, e = rest.split("-")
                        try:
                            s_i = int(s)
                            e_i = int(e)
                        except Exception:
                            return tok  # leave unchanged on parse error
                        return f"A{off + s_i}-A{off + e_i}"
                    else:
                        try:
                            r = int(rest)
                        except Exception:
                            return tok
                        return f"A{off + r}"

                remapped = ",".join(
                    [t for t in (_remap_token(x) for x in str(hotspots).split(",")) if t]
                )
                target_settings["target_hotspots"] = remapped

    run_settings["starting_binder_seq"] = get_sequence_from_pdb(starting_pdb_complex)[
        target_settings.get("binder_chain", "B")
    ]
    run_settings["starting_pdb_complex"] = starting_pdb_complex

    # Apply configuration updates and validation logic
    if run_settings.get("use_multimer_design"):
        run_settings["design_models"] = [0, 1, 2, 3, 4]
    else:
        run_settings["design_models"] = [0, 1]
    # Normalize negative bias values to False for consistency
    if run_settings.get("bias_redesign") < 0:
        run_settings["bias_redesign"] = False

    return io, run_settings
