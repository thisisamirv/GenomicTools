#!/usr/bin/env python
# Import required modules
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from .AliasUtils import AliasUtils
from .LoggingUtils import log


def ParseToKeyValueList(expr: Optional[str]) -> List[Tuple[str, str]]:
    """
    Parses a string expression into a list of key-value tuples.
    Example: "col1=a,col2=b" -> [("col1", "a"), ("col2", "b")]
    """
    if not expr:
        return []

    conditions: List[Tuple[str, str]] = []
    expressions = expr.split(",")

    for expr_part in expressions:
        if "=" in expr_part:
            col, val = expr_part.split("=", 1)
            col = col.strip()
            val = val.strip()
            if col:
                conditions.append((col, val))
        else:
            log.warn(f"Invalid format: {expr_part}. Expected format: 'column=value'")

    return conditions


def ParseToKeyValueDict(expr: Optional[Union[str, Dict[str, str]]]) -> Dict[str, str]:
    """
    Parses a string expression into a dictionary of key-value pairs.
    Example: "key1=value1,key2=value2" -> {"key1": "value1", "key2": "value2"}
    """
    if isinstance(expr, dict):
        return expr

    if not expr:
        return {}

    result_dict: Dict[str, str] = {}
    expressions = expr.split(",")

    for expr_part in expressions:
        if "=" in expr_part:
            key, val = expr_part.split("=", 1)
            result_dict[key.strip()] = val.strip()
        else:
            log.warn(
                f"Invalid key-value format: {expr_part}. Expected format: 'key=value'"
            )

    return result_dict


def ParseToList(s: Optional[Union[str, List[str]]]) -> List[str]:
    """
    Parses a comma-separated string into a list of strings.
    Example: "a, b, c" -> ["a", "b", "c"]
    """
    if isinstance(s, list):
        return s

    if not s:
        return []

    return [item.strip() for item in s.split(",")]


def ParseFormula(
    formula: str,
    default_data_variable: Optional[str] = None,
    build_formula: bool = False,
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], str]]:
    """
    Parses a statistical formula string into its components.
    Example: "Methylation ~ Genotype + Age + (1|Batch) + Genotype:Age"
    """
    try:
        methylation_aliases = AliasUtils.get_aliases("Methylation")
        genotype_aliases = AliasUtils.get_aliases("Genotype")
    except (AttributeError, KeyError) as e:
        log.warn(f"Could not load aliases from AliasUtils: {e}")
        methylation_aliases = ["Methylation", "methylation", "M", "m"]
        genotype_aliases = ["Genotype", "genotype", "G", "g"]

    parts = formula.split("~")
    if len(parts) != 2:
        raise ValueError("Invalid formula format. Expected format: y ~ x1 + x2 + ...")

    dependent_var = parts[0].strip()
    independent_vars = parts[1].strip()

    if not dependent_var:
        raise ValueError("Invalid formula format. Expected format: y ~ x1 + x2 + ...")

    if not independent_vars:
        raise ValueError("Invalid formula format. Expected format: y ~ x1 + x2 + ...")

    random_pattern = r"\((.*?)\|(.*?)\)"
    random_matches = re.findall(random_pattern, independent_vars)
    random_effects: Optional[str] = None
    if random_matches:
        random_effects = ",".join([match[1].strip() for match in random_matches])

    independent_vars_clean = re.sub(random_pattern, "", independent_vars)

    interaction_pattern = r"(\b\w+[:*]\w+(?:[:*]\w+)*\b)"
    interactions = re.findall(interaction_pattern, independent_vars_clean)
    interaction_term: Optional[str] = None
    if interactions:
        interaction_term = interactions[0].replace("*", ":")

    independent_vars_no_interactions = re.sub(
        r"\b\w+[:*]\w+(?:[:*]\w+)*\b", "", independent_vars_clean
    )

    independent_parts = re.split(r"[+\-]", independent_vars_no_interactions)

    all_vars: List[str] = []
    for part in independent_parts:
        var = part.strip()
        if var and var not in ["1", "0"] and not var.startswith(":"):
            all_vars.append(var)

    if dependent_var in methylation_aliases:
        dependent_var = "Methylation"
    elif dependent_var in genotype_aliases:
        dependent_var = "Genotype"

    data_variable: Optional[str] = None
    data_type_found: Optional[str] = None

    if dependent_var == "Methylation":
        data_variable = "Methylation"
        data_type_found = "Methylation"
    elif dependent_var == "Genotype":
        data_variable = "Genotype"
        data_type_found = "Genotype"

    standardized_vars: List[str] = []
    for var in all_vars:
        if var in methylation_aliases:
            standardized_var = "Methylation"
            standardized_vars.append(standardized_var)
            if not data_type_found:
                data_variable = "Methylation"
                data_type_found = "Methylation"
            elif data_type_found != "Methylation":
                log.warn(
                    f"Multiple data types found: {data_type_found} and Methylation. Using first: {data_type_found}"
                )
        elif var in genotype_aliases:
            standardized_var = "Genotype"
            standardized_vars.append(standardized_var)
            if not data_type_found:
                data_variable = "Genotype"
                data_type_found = "Genotype"
            elif data_type_found != "Genotype":
                log.warn(
                    f"Multiple data types found: {data_type_found} and Genotype. Using first: {data_type_found}"
                )
        else:
            standardized_vars.append(var)

    if not data_variable and default_data_variable:
        data_variable = default_data_variable
        log.info(
            f"No data variable specified in formula, using default: {default_data_variable}"
        )

    covariates: Optional[List[str]]
    covariates_list: List[str] = []
    for var in standardized_vars:
        if var not in ["Methylation", "Genotype"]:
            covariates_list.append(var)

    covariates = covariates_list if covariates_list else None

    if interaction_term:
        interaction_parts = interaction_term.replace("*", ":").split(":")
        standardized_interaction_parts: List[str] = []
        for part in interaction_parts:
            part = part.strip()
            if part in methylation_aliases:
                standardized_interaction_parts.append("Methylation")
            elif part in genotype_aliases:
                standardized_interaction_parts.append("Genotype")
            else:
                standardized_interaction_parts.append(part)
        interaction_term = ":".join(standardized_interaction_parts)

    result: Dict[str, Any] = {
        "dependent_var": dependent_var,
        "data_variable": data_variable,
        "data_type": data_type_found,
        "covariates": covariates,
        "interaction_term": interaction_term,
        "random_effects": random_effects,
    }

    log.debug("Formula parsing results:")
    log.debug(f"  Original formula: {formula}")
    log.debug(f"  Dependent variable: {dependent_var}")
    log.debug(f"  Data variable: {data_variable}")
    log.debug(f"  Data type: {data_type_found}")
    log.debug(f"  Covariates: {covariates}")

    if build_formula:
        if not data_variable:
            raise ValueError("data_variable is required to build formula")

        formula_parts: List[str] = []

        if dependent_var != data_variable:
            formula_parts.append(data_variable)

        if covariates:
            formula_parts.extend(covariates)
        if interaction_term:
            formula_parts.append(interaction_term)

        if not formula_parts:
            formula_parts = ["1"]

        built_formula = f"{dependent_var} ~ {' + '.join(formula_parts)}"
        if random_effects:
            built_formula += f" + (1|{random_effects})"

        log.info(f"Built full formula: {built_formula}")
        return result, built_formula

    return result
