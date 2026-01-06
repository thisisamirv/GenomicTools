#!/usr/bin/env python
import pytest
from unittest.mock import patch
from utils.LoggingUtils import log
from utils.ParsingUtils import (
    ParseFormula,
    ParseToKeyValueDict,
    ParseToKeyValueList,
    ParseToList,
)

log.setup(level="DEBUG")


@pytest.mark.unit
def test_valid_single_pair():
    result = ParseToKeyValueList("column1=value1")
    expected = [("column1", "value1")]
    assert result == expected


@pytest.mark.unit
def test_valid_multiple_pairs():
    result = ParseToKeyValueList("column1=value1,column2=value2,column3=value3")
    expected = [("column1", "value1"), ("column2", "value2"), ("column3", "value3")]
    assert result == expected


@pytest.mark.unit
def test_values_with_equals():
    result = ParseToKeyValueList("url=http://example.com,equation=x=y+z")
    expected = [("url", "http://example.com"), ("equation", "x=y+z")]
    assert result == expected


@pytest.mark.unit
def test_invalid_format_no_equals():
    result = ParseToKeyValueList("column1,column2=value2")

    expected = [("column2", "value2")]
    assert result == expected


@pytest.mark.unit
def test_mixed_valid_invalid():
    result = ParseToKeyValueList("valid=value,invalid,another=good")

    expected = [("valid", "value"), ("another", "good")]
    assert result == expected


@pytest.mark.unit
def test_special_characters_in_values():
    result = ParseToKeyValueList(
        'path=/home/user/file.txt,regex=[A-Z]+,json={"key":"value"}'
    )
    expected = [
        ("path", "/home/user/file.txt"),
        ("regex", "[A-Z]+"),
        ("json", '{"key":"value"}'),
    ]
    assert result == expected


@pytest.mark.unit
def test_empty_string_input():
    result = ParseToKeyValueList("")
    expected = []
    assert result == expected


@pytest.mark.unit
def test_whitespace_handling():
    result = ParseToKeyValueList(" key1 = value1 , key2 = value2 ")
    expected = [("key1", "value1"), ("key2", "value2")]
    assert result == expected


@pytest.mark.unit
def test_empty_key_value_pairs():
    result = ParseToKeyValueList("=value1,key2=,key3=value3")
    expected = [("key2", ""), ("key3", "value3")]
    assert result == expected


@pytest.mark.unit
def test_dict_valid_single_pair():
    result = ParseToKeyValueDict("key1=value1")
    expected = {"key1": "value1"}
    assert result == expected


@pytest.mark.unit
def test_dict_valid_multiple_pairs():
    result = ParseToKeyValueDict("key1=value1,key2=value2,key3=value3")
    expected = {"key1": "value1", "key2": "value2", "key3": "value3"}
    assert result == expected


@pytest.mark.unit
def test_dict_already_dict_input():
    input_dict = {"key1": "value1", "key2": "value2"}
    result = ParseToKeyValueDict(input_dict)
    assert result == input_dict
    assert result is input_dict


@pytest.mark.unit
def test_dict_values_with_equals():
    result = ParseToKeyValueDict("url=http://example.com,math=x=y+z")
    expected = {"url": "http://example.com", "math": "x=y+z"}
    assert result == expected


@pytest.mark.unit
def test_dict_invalid_format_no_equals():
    result = ParseToKeyValueDict("key1,key2=value2")

    expected = {"key2": "value2"}
    assert result == expected


@pytest.mark.unit
def test_dict_duplicate_keys():
    result = ParseToKeyValueDict("key1=value1,key1=value2,key2=value3")
    expected = {"key1": "value2", "key2": "value3"}
    assert result == expected


@pytest.mark.unit
def test_dict_empty_values():
    result = ParseToKeyValueDict("key1=,key2=value2,key3=")
    expected = {"key1": "", "key2": "value2", "key3": ""}
    assert result == expected


@pytest.mark.unit
def test_dict_empty_string_input():
    result = ParseToKeyValueDict("")
    expected = {}
    assert result == expected


@pytest.mark.unit
def test_dict_none_input():
    result = ParseToKeyValueDict(None)
    expected = {}
    assert result == expected


@pytest.mark.unit
def test_list_valid_comma_separated():
    result = ParseToList("item1,item2,item3")
    expected = ["item1", "item2", "item3"]
    assert result == expected


@pytest.mark.unit
def test_list_already_list_input():
    input_list = ["item1", "item2", "item3"]
    result = ParseToList(input_list)
    assert result == input_list
    assert result is input_list


@pytest.mark.unit
def test_list_single_item():
    result = ParseToList("single_item")
    expected = ["single_item"]
    assert result == expected


@pytest.mark.unit
def test_list_empty_items():
    result = ParseToList("item1,,item3,")
    expected = ["item1", "", "item3", ""]
    assert result == expected


@pytest.mark.unit
def test_list_special_characters():
    result = ParseToList("file.txt,path/to/file,data[0],func()")
    expected = ["file.txt", "path/to/file", "data[0]", "func()"]
    assert result == expected


@pytest.mark.unit
def test_list_empty_string_input():
    result = ParseToList("")
    expected = []
    assert result == expected


@pytest.mark.unit
def test_list_whitespace_handling():
    result = ParseToList(" item1 , item2 , item3 ")
    expected = ["item1", "item2", "item3"]
    assert result == expected


@pytest.mark.unit
def test_list_none_input():
    result = ParseToList(None)
    expected = []
    assert result == expected


@pytest.mark.unit
@patch("utils.ParsingUtils.AliasUtils.get_aliases")
def test_formula_simple_formula(mock_get_aliases):
    mock_get_aliases.side_effect = lambda data_type: {
        "Methylation": ["methylation", "Methylation", "M", "m"],
        "Genotype": ["genotype", "Genotype", "G", "g"],
    }.get(data_type, [])

    result = ParseFormula("y ~ x")
    expected = {
        "dependent_var": "y",
        "data_variable": None,
        "data_type": None,
        "covariates": ["x"],
        "interaction_term": None,
        "random_effects": None,
    }
    assert result == expected


@pytest.mark.unit
def test_formula_with_covariates():
    result = ParseFormula("phenotype ~ age + sex + pc1 + pc2")
    expected = {
        "dependent_var": "phenotype",
        "data_variable": None,
        "data_type": None,
        "covariates": ["age", "sex", "pc1", "pc2"],
        "interaction_term": None,
        "random_effects": None,
    }
    assert result == expected


@pytest.mark.unit
@patch("utils.ParsingUtils.AliasUtils.get_aliases")
def test_formula_with_genotype(mock_get_aliases):
    mock_get_aliases.side_effect = lambda data_type: {
        "Methylation": ["methylation", "Methylation", "M", "m"],
        "Genotype": ["genotype", "Genotype", "G", "g"],
    }.get(data_type, [])

    result = ParseFormula("trait ~ genotype + age + sex")
    expected = {
        "dependent_var": "trait",
        "data_variable": "Genotype",
        "data_type": "Genotype",
        "covariates": ["age", "sex"],
        "interaction_term": None,
        "random_effects": None,
    }
    assert result == expected


@pytest.mark.unit
@patch("utils.ParsingUtils.AliasUtils.get_aliases")
def test_formula_with_methylation(mock_get_aliases):
    mock_get_aliases.side_effect = lambda data_type: {
        "Methylation": ["methylation", "Methylation", "M", "m"],
        "Genotype": ["genotype", "Genotype", "G", "g"],
    }.get(data_type, [])

    result = ParseFormula("expression ~ methylation + batch")
    expected = {
        "dependent_var": "expression",
        "data_variable": "Methylation",
        "data_type": "Methylation",
        "covariates": ["batch"],
        "interaction_term": None,
        "random_effects": None,
    }
    assert result == expected


@pytest.mark.unit
@patch("utils.ParsingUtils.AliasUtils.get_aliases")
def test_formula_with_beta_value(mock_get_aliases):
    mock_get_aliases.side_effect = lambda data_type: {
        "Methylation": ["methylation", "Methylation", "M", "m", "beta_value"],
        "Genotype": ["genotype", "Genotype", "G", "g"],
    }.get(data_type, [])

    result = ParseFormula("outcome ~ beta_value + covar1")
    expected = {
        "dependent_var": "outcome",
        "data_variable": "Methylation",
        "data_type": "Methylation",
        "covariates": ["covar1"],
        "interaction_term": None,
        "random_effects": None,
    }
    assert result == expected


@pytest.mark.unit
def test_formula_with_interaction():
    result = ParseFormula("y ~ genotype + age + sex + genotype:age")
    expected = {
        "dependent_var": "y",
        "data_variable": "Genotype",
        "data_type": "Genotype",
        "covariates": ["age", "sex"],
        "interaction_term": "Genotype:age",
        "random_effects": None,
    }
    assert result == expected


@pytest.mark.unit
def test_formula_with_random_effects():
    result = ParseFormula("trait ~ genotype + age + (1|family)")
    expected = {
        "dependent_var": "trait",
        "data_variable": "Genotype",
        "data_type": "Genotype",
        "covariates": ["age"],
        "interaction_term": None,
        "random_effects": "family",
    }
    assert result == expected


@pytest.mark.unit
def test_formula_with_multiple_random_effects():
    result = ParseFormula("trait ~ genotype + (1|family) + (1|batch)")
    expected = {
        "dependent_var": "trait",
        "data_variable": "Genotype",
        "data_type": "Genotype",
        "covariates": None,
        "interaction_term": None,
        "random_effects": "family,batch",
    }
    assert result == expected


@pytest.mark.unit
def test_formula_complex_formula():
    result = ParseFormula(
        "height ~ genotype + age + sex + bmi + genotype:sex + (1|family) + (1|center)"
    )
    expected = {
        "dependent_var": "height",
        "data_variable": "Genotype",
        "data_type": "Genotype",
        "covariates": ["age", "sex", "bmi"],
        "interaction_term": "Genotype:sex",
        "random_effects": "family,center",
    }
    assert result == expected


@pytest.mark.unit
def test_formula_with_default_data_variable():
    result = ParseFormula("y ~ age + sex", default_data_variable="genotype")
    expected = {
        "dependent_var": "y",
        "data_variable": "genotype",
        "data_type": None,
        "covariates": ["age", "sex"],
        "interaction_term": None,
        "random_effects": None,
    }
    assert result == expected


@pytest.mark.unit
def test_formula_multiple_data_variables_warning():
    result = ParseFormula("y ~ genotype + methylation + age")

    assert result["data_variable"] == "Genotype"


@pytest.mark.unit
def test_formula_no_data_variable_with_default_info_message():
    result = ParseFormula("y ~ age + sex", default_data_variable="genotype")

    assert result["data_variable"] == "genotype"


@pytest.mark.unit
def test_formula_invalid_formula_format():
    with pytest.raises(
        ValueError,
        match="Invalid formula format. Expected format: y ~ x1 \\+ x2 \\+ ...",
    ):
        ParseFormula("invalid_formula")


@pytest.mark.unit
def test_formula_build_formula_simple():
    components, built_formula = ParseFormula(
        "y ~ age + sex", build_formula=True, default_data_variable="genotype"
    )

    expected_components = {
        "dependent_var": "y",
        "data_variable": "genotype",
        "data_type": None,
        "covariates": ["age", "sex"],
        "interaction_term": None,
        "random_effects": None,
    }

    assert components == expected_components
    assert built_formula == "y ~ genotype + age + sex"


@pytest.mark.unit
def test_formula_build_formula_complex():
    components, built_formula = ParseFormula(
        "height ~ genotype + age + sex + genotype:age + (1|family)",
        build_formula=True,
    )

    expected_components = {
        "dependent_var": "height",
        "data_variable": "Genotype",
        "data_type": "Genotype",
        "covariates": ["age", "sex"],
        "interaction_term": "Genotype:age",
        "random_effects": "family",
    }

    assert components == expected_components
    assert built_formula == "height ~ Genotype + age + sex + Genotype:age + (1|family)"


@pytest.mark.unit
def test_formula_build_formula_no_data_variable_error():
    with pytest.raises(ValueError, match="data_variable is required to build formula"):
        ParseFormula("y ~ age + sex", build_formula=True)


@pytest.mark.unit
def test_formula_build_formula_with_multiple_random_effects():
    components, built_formula = ParseFormula(
        "y ~ genotype + age + (1|family) + (1|batch)", build_formula=True
    )

    assert built_formula == "y ~ Genotype + age + (1|family,batch)"
    assert components["random_effects"] == "family,batch"


@pytest.mark.unit
def test_formula_whitespace_handling():
    result = ParseFormula("  trait  ~  genotype  +  age  +  sex  +  ( 1 | family )  ")
    expected = {
        "dependent_var": "trait",
        "data_variable": "Genotype",
        "data_type": "Genotype",
        "covariates": ["age", "sex"],
        "interaction_term": None,
        "random_effects": "family",
    }
    assert result == expected


@pytest.mark.unit
def test_formula_empty_terms_handling():
    result = ParseFormula("y ~ genotype + + age + sex")
    expected = {
        "dependent_var": "y",
        "data_variable": "Genotype",
        "data_type": "Genotype",
        "covariates": ["age", "sex"],
        "interaction_term": None,
        "random_effects": None,
    }
    assert result == expected


@pytest.mark.unit
def test_formula_with_only_data_variable_and_random_effects():
    result = ParseFormula("trait ~ genotype + (1|family)")
    expected = {
        "dependent_var": "trait",
        "data_variable": "Genotype",
        "data_type": "Genotype",
        "covariates": None,
        "interaction_term": None,
        "random_effects": "family",
    }
    assert result == expected


@pytest.mark.unit
@patch("utils.ParsingUtils.AliasUtils.get_aliases")
def test_formula_with_aliasutils_error(mock_get_aliases):
    mock_get_aliases.side_effect = AttributeError("get_aliases not available")

    result = ParseFormula("trait ~ genotype + age + sex")
    expected = {
        "dependent_var": "trait",
        "data_variable": "Genotype",
        "data_type": "Genotype",
        "covariates": ["age", "sex"],
        "interaction_term": None,
        "random_effects": None,
    }
    assert result == expected


@pytest.mark.unit
def test_formula_missing_dependent_variable():
    with pytest.raises(
        ValueError,
        match="Invalid formula format. Expected format: y ~ x1 \\+ x2 \\+ ...",
    ):
        ParseFormula("~ age + sex")


@pytest.mark.unit
def test_formula_missing_right_side():
    with pytest.raises(
        ValueError,
        match="Invalid formula format. Expected format: y ~ x1 \\+ x2 \\+ ...",
    ):
        ParseFormula("y ~")


@pytest.mark.unit
def test_formula_with_nested_random_effects():
    result = ParseFormula("trait ~ genotype + age + (age|family)")
    expected = {
        "dependent_var": "trait",
        "data_variable": "Genotype",
        "data_type": "Genotype",
        "covariates": ["age"],
        "interaction_term": None,
        "random_effects": "family",
    }
    assert result == expected


@pytest.mark.unit
def test_formula_with_complex_interaction():
    result = ParseFormula("y ~ genotype + age + sex + genotype:age:sex")
    expected = {
        "dependent_var": "y",
        "data_variable": "Genotype",
        "data_type": "Genotype",
        "covariates": ["age", "sex"],
        "interaction_term": "Genotype:age:sex",
        "random_effects": None,
    }
    assert result == expected


@pytest.mark.unit
def test_formula_with_special_characters_in_variable_names():
    result = ParseFormula("trait_1 ~ genotype + age_at_visit + sex_coded")
    expected = {
        "dependent_var": "trait_1",
        "data_variable": "Genotype",
        "data_type": "Genotype",
        "covariates": ["age_at_visit", "sex_coded"],
        "interaction_term": None,
        "random_effects": None,
    }
    assert result == expected


@pytest.mark.unit
def test_formula_case_insensitive_data_variables():
    result = ParseFormula("trait ~ GENOTYPE + age + sex")
    expected = {
        "dependent_var": "trait",
        "data_variable": "Genotype",
        "data_type": "Genotype",
        "covariates": ["age", "sex"],
        "interaction_term": None,
        "random_effects": None,
    }
    assert result == expected


@pytest.mark.unit
def test_formula_build_with_no_covariates():
    components, built_formula = ParseFormula(
        "trait ~ genotype + (1|family)", build_formula=True
    )

    assert built_formula == "trait ~ Genotype + (1|family)"
    assert components["covariates"] is None


@pytest.mark.unit
def test_formula_build_with_no_random_effects():
    components, built_formula = ParseFormula(
        "trait ~ genotype + age + sex", build_formula=True
    )

    assert built_formula == "trait ~ Genotype + age + sex"
    assert components["random_effects"] is None


@pytest.mark.integration
def test_visual_inspection():
    print("\n==== PARSING UTILS VISUAL INSPECTION ====")

    print("\nParseToKeyValueList Examples:")
    examples = [
        "col1=val1,col2=val2",
        "filter=age>30,group=treatment",
        "path=/home/user,type=csv",
        "invalid,valid=good",
        "key1=value with spaces,key2=value2",
        "empty_key=,=empty_value,normal=value",
    ]

    for example in examples:
        result = ParseToKeyValueList(example)
        print(f"  Input: '{example}'")
        print(f"  Output: {result}")

    print("\nParseToKeyValueDict Examples:")
    examples = [
        "key1=value1,key2=value2",
        {"already": "dict"},
        "",
        "duplicate=first,duplicate=second",
        None,
        "key1=,key2=value2,key3=",
    ]

    for example in examples:
        result = ParseToKeyValueDict(example)
        print(f"  Input: {example}")
        print(f"  Output: {result}")

    print("\nParseToList Examples:")
    examples = [
        "item1,item2,item3",
        ["already", "list"],
        "single_item",
        " spaced , items , here ",
        "",
        None,
        "item1,,item3,",
    ]

    for example in examples:
        result = ParseToList(example)
        print(f"  Input: {example}")
        print(f"  Output: {result}")

    print("\nParseFormula Examples:")
    formulas = [
        "height ~ genotype + age + sex",
        "trait ~ methylation + batch + (1|family)",
        "outcome ~ genotype + age + genotype:age + (1|study)",
        "expression ~ beta_value + tissue + gender + (1|individual) + (1|batch)",
        "y ~ age + sex",
        "phenotype ~ GENOTYPE + AGE + SEX",
        "trait ~ genotype + (age|family)",
    ]

    for formula in formulas:
        print(f"\n  Formula: '{formula}'")
        try:
            components = ParseFormula(formula)
            for key, value in components.items():
                print(f"    {key}: {value}")

            if components["data_variable"]:
                try:
                    _, built = ParseFormula(formula, build_formula=True)
                    print(f"    Built formula: {built}")
                except Exception as build_error:
                    print(f"    Build error: {build_error}")
        except Exception as e:
            print(f"    Error: {e}")

    print("\nEdge Cases:")
    edge_cases = [
        ("Empty string key-value", ""),
        ("Only commas", ",,"),
        ("Formula with only ~", "y ~"),
        ("Formula missing dependent", "~ age + sex"),
        ("Malformed interaction", "y ~ age + :sex"),
    ]

    for description, case in edge_cases:
        print(f"\n  {description}: '{case}'")
        try:
            if "~" in case:
                result = ParseFormula(case)
            elif "=" in case:
                result = ParseToKeyValueDict(case)
            else:
                result = ParseToList(case)
            print(f"    Result: {result}")
        except Exception as e:
            print(f"    Error: {e}")

    print("\n==============================================")

    assert True


@pytest.mark.integration
def test_parsing_with_real_world_examples():
    print("\n==== REAL-WORLD GENOMIC EXAMPLES ====")

    gwas_examples = [
        "BMI ~ SNP + age + sex + PC1 + PC2 + PC3 + (1|study)",
        "height ~ genotype + age + sex + age:sex + (1|population)",
        "diabetes ~ genotype + BMI + age + sex + smoking + (1|center)",
    ]

    ewas_examples = [
        "expression ~ methylation + age + sex + batch + (1|individual)",
        "phenotype ~ beta_value + cell_type + age + sex + (1|plate)",
        "outcome ~ M + tissue + age + sex + smoking + (1|study_id)",
    ]

    pgx_examples = [
        "drug_response ~ genotype + age + sex + weight + dose",
        "toxicity ~ G + age + sex + comorbidity + (1|hospital)",
        "efficacy ~ genotype + age + sex + genotype:age + (1|trial)",
    ]

    all_examples = [
        ("GWAS", gwas_examples),
        ("EWAS", ewas_examples),
        ("Pharmacogenomics", pgx_examples),
    ]

    for category, examples in all_examples:
        print(f"\n{category} Examples:")
        for formula in examples:
            print(f"  Formula: {formula}")
            try:
                components = ParseFormula(formula)
                print(f"    Data variable: {components['data_variable']}")
                print(f"    Data type: {components['data_type']}")
                print(f"    Covariates: {components['covariates']}")
                print(f"    Interaction: {components['interaction_term']}")
                print(f"    Random effects: {components['random_effects']}")

                if components["data_variable"]:
                    _, built = ParseFormula(formula, build_formula=True)
                    print(f"    Rebuilt: {built}")
                print()
            except Exception as e:
                print(f"    Error: {e}")
                print()

    print("Parameter Parsing Examples:")
    param_examples = [
        "maf=0.05,hwe=1e-6,geno=0.02,mind=0.02",
        "pval_threshold=5e-8,r2_threshold=0.8,window_kb=500",
        "model=additive,test=logistic,adjust=age+sex+PC1+PC2+PC3",
        "input=/path/to/data.bed,output=/path/to/results,format=plink",
    ]

    for params in param_examples:
        print(f"  Parameters: {params}")
        result_dict = ParseToKeyValueDict(params)
        result_list = ParseToKeyValueList(params)
        print(f"    As dict: {result_dict}")
        print(f"    As list: {result_list}")
        print()

    print("File List Parsing Examples:")
    file_examples = [
        "/data/chr1.bed,/data/chr2.bed,/data/chr3.bed",
        "sample1.vcf,sample2.vcf,sample3.vcf",
        "cohort_A.txt,cohort_B.txt,cohort_C.txt,cohort_D.txt",
    ]

    for files in file_examples:
        print(f"  Files: {files}")
        result = ParseToList(files)
        print(f"    Parsed: {result}")
        print()

    print("==============================================")
    assert True


if __name__ == "__main__":
    pytest.main([__file__])
