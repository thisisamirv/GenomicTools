#!/usr/bin/env python
import json
import pytest
import requests
from unittest.mock import MagicMock, patch
from utils.LoggingUtils import log
from utils.ConvertGeneID import ConvertGeneID, GeneIDConverter, ConversionConfig

log.setup(level="DEBUG")


@pytest.fixture
def sample_gene_data():
    return {
        "symbols": ["TP53", "BRCA1", "EGFR", "KRAS"],
        "entrez_ids": ["7157", "672", "1956", "3845"],
        "ensembl_ids": [
            "ENSG00000141510",
            "ENSG00000012048",
            "ENSG00000146648",
            "ENSG00000133703",
        ],
    }


@pytest.fixture
def mock_api_responses():
    return {
        "symbol_to_entrez": [
            {"query": "TP53", "entrezgene": "7157", "_id": "7157"},
            {"query": "BRCA1", "entrezgene": "672", "_id": "672"},
            {"query": "EGFR", "entrezgene": "1956", "_id": "1956"},
            {"query": "KRAS", "entrezgene": "3845", "_id": "3845"},
        ],
        "entrez_to_ensembl": [
            {
                "query": "7157",
                "ensembl": {"gene": "ENSG00000141510"},
                "_id": "7157",
            },
            {"query": "672", "ensembl": {"gene": "ENSG00000012048"}, "_id": "672"},
            {
                "query": "1956",
                "ensembl": {"gene": "ENSG00000146648"},
                "_id": "1956",
            },
            {
                "query": "3845",
                "ensembl": {"gene": "ENSG00000133703"},
                "_id": "3845",
            },
        ],
        "ensembl_to_symbol": [
            {"query": "ENSG00000141510", "symbol": "TP53", "_id": "7157"},
            {"query": "ENSG00000012048", "symbol": "BRCA1", "_id": "672"},
            {"query": "ENSG00000146648", "symbol": "EGFR", "_id": "1956"},
            {"query": "ENSG00000133703", "symbol": "KRAS", "_id": "3845"},
        ],
    }


@pytest.fixture
def mock_response():
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock()
    return mock_resp


@pytest.mark.unit
def test_conversion_config_validation():
    config = ConversionConfig(id_from="symbol", id_to="entrez")
    assert config.id_from == "symbol"
    assert config.id_to == "entrez"
    assert config.species == "human"
    assert config.batch_size == 1000

    with pytest.raises(ValueError, match="Invalid id_from"):
        ConversionConfig(id_from="invalid", id_to="entrez")

    with pytest.raises(ValueError, match="Invalid id_to"):
        ConversionConfig(id_from="symbol", id_to="invalid")

    config = ConversionConfig(id_from="symbol", id_to="entrez", batch_size=-1)
    assert config.batch_size == 1000


@pytest.mark.unit
def test_gene_id_converter_init():
    converter = GeneIDConverter()
    assert converter.config.id_from == "symbol"
    assert converter.config.id_to == "entrez"

    config = ConversionConfig(id_from="ensembl", id_to="symbol", timeout=60)
    converter = GeneIDConverter(config)
    assert converter.config.id_from == "ensembl"
    assert converter.config.id_to == "symbol"
    assert converter.config.timeout == 60


@pytest.mark.unit
@patch("requests.Session.post")
def test_symbol_to_entrez(
    mock_post, sample_gene_data, mock_api_responses, mock_response
):
    mock_response.json.return_value = mock_api_responses["symbol_to_entrez"]
    mock_post.return_value = mock_response

    result = ConvertGeneID(
        sample_gene_data["symbols"],
        id_from="symbol",
        id_to="entrez",
        show_progress=False,
    )

    assert result == sample_gene_data["entrez_ids"]
    mock_post.assert_called_once()


@pytest.mark.unit
@patch("requests.Session.post")
def test_entrez_to_ensembl(
    mock_post, sample_gene_data, mock_api_responses, mock_response
):
    mock_response.json.return_value = mock_api_responses["entrez_to_ensembl"]
    mock_post.return_value = mock_response

    result = ConvertGeneID(
        sample_gene_data["entrez_ids"],
        id_from="entrez",
        id_to="ensembl",
        show_progress=False,
    )

    assert result == sample_gene_data["ensembl_ids"]
    mock_post.assert_called_once()


@pytest.mark.unit
@patch("requests.Session.post")
def test_ensembl_to_symbol(
    mock_post, sample_gene_data, mock_api_responses, mock_response
):
    mock_response.json.return_value = mock_api_responses["ensembl_to_symbol"]
    mock_post.return_value = mock_response

    result = ConvertGeneID(
        sample_gene_data["ensembl_ids"],
        id_from="ensembl",
        id_to="symbol",
        show_progress=False,
    )

    assert result == sample_gene_data["symbols"]
    mock_post.assert_called_once()


@pytest.mark.unit
def test_empty_input():
    result = ConvertGeneID([])

    assert result == []


@pytest.mark.unit
@patch("requests.Session.post")
def test_none_values(mock_post, mock_response):
    input_with_none = ["TP53", None, "EGFR", ""]

    filtered_response = [
        {"query": "TP53", "entrezgene": "7157", "_id": "7157"},
        {"query": "EGFR", "entrezgene": "1956", "_id": "1956"},
    ]
    mock_response.json.return_value = filtered_response
    mock_post.return_value = mock_response

    result = ConvertGeneID(input_with_none, show_progress=False)

    assert len(result) == 4
    assert result[0] == "7157"
    assert result[1] is None
    assert result[2] == "1956"
    assert result[3] is None


@pytest.mark.unit
@patch("requests.Session.post")
def test_multiple_batches(mock_post):
    large_symbols = ["GENE" + str(i) for i in range(1, 2001)]
    large_entrez = [str(10000 + i) for i in range(1, 2001)]

    batch1_response = [
        {"query": f"GENE{i}", "entrezgene": f"{10000 + i}", "_id": f"{10000 + i}"}
        for i in range(1, 1001)
    ]

    batch2_response = [
        {"query": f"GENE{i}", "entrezgene": f"{10000 + i}", "_id": f"{10000 + i}"}
        for i in range(1001, 2001)
    ]

    mock_resp1 = MagicMock()
    mock_resp1.raise_for_status = MagicMock()
    mock_resp1.json.return_value = batch1_response

    mock_resp2 = MagicMock()
    mock_resp2.raise_for_status = MagicMock()
    mock_resp2.json.return_value = batch2_response

    mock_post.side_effect = [mock_resp1, mock_resp2]

    result = ConvertGeneID(large_symbols, batch_size=1000, show_progress=False)

    assert len(result) == 2000
    assert result == large_entrez
    assert mock_post.call_count == 2


@pytest.mark.unit
@patch("requests.Session.post")
def test_ensembl_list_response(mock_post, mock_response):
    list_response = [
        {
            "query": "7157",
            "ensembl": [
                {"gene": "ENSG00000141510", "transcript": "ENST00000269305"},
                {"gene": "ENSG00000141511", "transcript": "ENST00000269306"},
            ],
            "_id": "7157",
        }
    ]
    mock_response.json.return_value = list_response
    mock_post.return_value = mock_response

    result = ConvertGeneID(
        ["7157"], id_from="entrez", id_to="ensembl", show_progress=False
    )

    assert result == [["ENSG00000141510", "ENSG00000141511"]]


@pytest.mark.unit
@patch("requests.Session.post")
def test_ensembl_single_list_response(mock_post, mock_response):
    single_list_response = [
        {
            "query": "7157",
            "ensembl": [{"gene": "ENSG00000141510", "transcript": "ENST00000269305"}],
            "_id": "7157",
        }
    ]
    mock_response.json.return_value = single_list_response
    mock_post.return_value = mock_response

    result = ConvertGeneID(
        ["7157"], id_from="entrez", id_to="ensembl", show_progress=False
    )

    assert result == ["ENSG00000141510"]


@pytest.mark.unit
@patch("requests.Session.post")
def test_error_handling_with_retries(mock_post, sample_gene_data):
    mock_resp_success = MagicMock()
    mock_resp_success.raise_for_status = MagicMock()
    mock_resp_success.json.return_value = [
        {"query": "TP53", "entrezgene": "7157", "_id": "7157"}
    ]

    mock_post.side_effect = [
        requests.exceptions.Timeout("Timeout 1"),
        requests.exceptions.ConnectionError("Connection error"),
        mock_resp_success,
    ]

    result = ConvertGeneID(["TP53"], show_progress=False, max_retries=3)

    assert result == ["7157"]
    assert mock_post.call_count == 3


@pytest.mark.unit
@patch("requests.Session.post")
def test_max_retries_exceeded(mock_post, sample_gene_data):
    mock_post.side_effect = requests.exceptions.Timeout("Persistent timeout")

    result = ConvertGeneID(["TP53"], show_progress=False, max_retries=2)

    assert result == [None]
    assert mock_post.call_count == 2


@pytest.mark.unit
@patch("requests.Session.post")
def test_not_found_handling(mock_post, mock_response):
    not_found_response = [
        {"query": "TP53", "entrezgene": "7157", "_id": "7157"},
        {"query": "INVALID", "notfound": True},
    ]
    mock_response.json.return_value = not_found_response
    mock_post.return_value = mock_response

    result = ConvertGeneID(["TP53", "INVALID"], show_progress=False)

    assert result == ["7157", None]


@pytest.mark.unit
@patch("requests.Session.post")
def test_timeout_parameter(mock_post, mock_response):
    mock_response.json.return_value = [
        {"query": "TP53", "entrezgene": "7157", "_id": "7157"}
    ]
    mock_post.return_value = mock_response

    result = ConvertGeneID(["TP53"], timeout=60, show_progress=False)

    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args[1]
    assert call_kwargs["timeout"] == 60
    assert result == ["7157"]


@pytest.mark.unit
def test_invalid_conversion_type():
    with pytest.raises(ValueError, match="Invalid id_from"):
        ConvertGeneID(["TP53"], id_from="invalid", id_to="entrez")

    with pytest.raises(ValueError, match="Invalid id_to"):
        ConvertGeneID(["TP53"], id_from="symbol", id_to="invalid")


@pytest.mark.unit
@patch("requests.Session.post")
def test_batch_size_validation(mock_post, mock_response):
    mock_response.json.return_value = [
        {"query": "TP53", "entrezgene": "7157", "_id": "7157"}
    ]
    mock_post.return_value = mock_response

    result = ConvertGeneID(["TP53"], batch_size=0, show_progress=False)
    assert result == ["7157"]

    result = ConvertGeneID(["TP53"], batch_size=10000, show_progress=False)
    assert result == ["7157"]


@pytest.mark.unit
@patch("requests.Session.post")
def test_species_parameter(mock_post, mock_response):
    mock_response.json.return_value = [
        {"query": "TP53", "entrezgene": "7157", "_id": "7157"}
    ]
    mock_post.return_value = mock_response

    result = ConvertGeneID(["TP53"], species="mouse", show_progress=False)

    mock_post.assert_called_once()
    call_args = mock_post.call_args
    payload = call_args[1]["json"]
    assert payload["species"] == "mouse"
    assert result == ["7157"]


@pytest.mark.unit
@patch("requests.Session.post")
def test_unexpected_api_response_format(mock_post):
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"error": "Unexpected format"}
    mock_post.return_value = mock_resp

    result = ConvertGeneID(["TP53"], show_progress=False)

    assert result == [None]


@pytest.mark.unit
@patch("requests.Session.post")
def test_converter_class_direct_usage(mock_post, mock_response):
    mock_response.json.return_value = [
        {"query": "TP53", "entrezgene": "7157", "_id": "7157"}
    ]
    mock_post.return_value = mock_response

    config = ConversionConfig(
        id_from="symbol", id_to="entrez", batch_size=500, timeout=45, max_retries=2
    )
    converter = GeneIDConverter(config)

    result = converter.convert(["TP53"])

    assert result == ["7157"]
    call_kwargs = mock_post.call_args[1]
    assert call_kwargs["timeout"] == 45


@pytest.mark.integration
@patch("requests.Session.post")
def test_visual_inspection(mock_post):
    print("\n==== GENE ID CONVERSION VISUAL INSPECTION ====")

    print("Sample gene IDs for reference:")
    sample_data = {
        "symbol": ["TP53", "BRCA1", "EGFR", "KRAS", "PTEN"],
        "entrez": ["7157", "672", "1956", "3845", "5728"],
        "ensembl": [
            "ENSG00000141510",
            "ENSG00000012048",
            "ENSG00000146648",
            "ENSG00000133703",
            "ENSG00000171862",
        ],
    }

    for id_type, examples in sample_data.items():
        print(f"  {id_type.capitalize()}: {', '.join(examples)}")

    print("\nAPI Request Structure:")
    url = "https://mygene.info/v3/query"
    payload = {
        "q": ["TP53", "BRCA1"],
        "scopes": "symbol",
        "fields": ["entrezgene"],
        "species": "human",
        "size": 2,
    }
    print(f"  URL: {url}")
    print("  Method: POST")
    print(f"  JSON Payload: {json.dumps(payload, indent=2)}")

    def setup_mock_for_test(mock_response_data):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = mock_response_data
        mock_post.return_value = mock_resp

    print("\nEXAMPLE 1: Symbol to Entrez Conversion")
    symbol_to_entrez_response = [
        {"query": "TP53", "entrezgene": 7157, "_id": "7157"},
        {"query": "BRCA1", "entrezgene": 672, "_id": "672"},
    ]
    print("  Input: ['TP53', 'BRCA1']")
    print(f"  API Response: {json.dumps(symbol_to_entrez_response, indent=2)}")

    setup_mock_for_test(symbol_to_entrez_response)
    symbol_result = ConvertGeneID(
        ["TP53", "BRCA1"], id_from="symbol", id_to="entrez", show_progress=False
    )
    print(f"  Output: {symbol_result}")

    print("\nEXAMPLE 2: Using GeneIDConverter Class")
    config = ConversionConfig(
        id_from="symbol",
        id_to="entrez",
        batch_size=1,
        timeout=30,
        max_retries=2,
    )
    converter = GeneIDConverter(config)

    setup_mock_for_test([{"query": "TP53", "entrezgene": 7157, "_id": "7157"}])
    class_result = converter.convert(["TP53"])
    print(f"  Using class: {class_result}")

    print("\nEXAMPLE 3: Retry Mechanism")
    print("  Simulating: Timeout -> Success")

    mock_resp_success = MagicMock()
    mock_resp_success.raise_for_status = MagicMock()
    mock_resp_success.json.return_value = [
        {"query": "TP53", "entrezgene": "7157", "_id": "7157"}
    ]

    mock_post.side_effect = [requests.exceptions.Timeout("Timeout"), mock_resp_success]

    retry_result = ConvertGeneID(["TP53"], max_retries=2, show_progress=False)
    print(f"  Result after retry: {retry_result}")

    print("\nProcess Visualization (Enhanced):")
    print("  1. Validate configuration parameters")
    print("  2. Clean input (remove None/empty values)")
    print("  3. Create HTTP session with headers")
    print("  4. Split into batches of specified size")
    print("  5. Make API request for each batch with retry logic")
    print("  6. Process each batch response with error handling")
    print("  7. Merge results maintaining original order")
    print("  8. Return combined results")

    print("=================================================")

    assert True


if __name__ == "__main__":
    pytest.main([__file__])
