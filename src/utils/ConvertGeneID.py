#!/usr/bin/env python
# Import required modules
import requests
import time
import warnings
from dataclasses import dataclass
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Union, Tuple
from .LoggingUtils import log


@dataclass
class ConversionConfig:
    id_from: str
    id_to: str
    species: str = "human"
    batch_size: int = 1000
    show_progress: bool = True
    timeout: int = 30
    max_retries: int = 3
    return_stats: bool = False

    def __post_init__(self) -> None:
        valid_id_types = {"symbol", "entrez", "ensembl"}
        if self.id_from not in valid_id_types:
            raise ValueError(
                f"Invalid id_from: {self.id_from}. Must be one of {valid_id_types}"
            )
        if self.id_to not in valid_id_types:
            raise ValueError(
                f"Invalid id_to: {self.id_to}. Must be one of {valid_id_types}"
            )
        if self.batch_size <= 0:
            log.warn(f"Invalid batch size: {self.batch_size}. Using default of 1000")
            self.batch_size = 1000


class GeneIDConverter:
    BASE_URL = "https://mygene.info/v3/query"
    FIELDS_MAP = {"symbol": "symbol", "entrez": "entrezgene", "ensembl": "ensembl.gene"}
    SCOPES_MAP = {
        "symbol": "symbol,alias",
        "entrez": "entrezgene",
        "ensembl": "ensembl.gene",
    }

    def __init__(self, config: Optional[ConversionConfig] = None) -> None:
        self.config: ConversionConfig = config or ConversionConfig("symbol", "entrez")
        self.session: requests.Session = self._create_session()
        self.last_stats: Optional[Dict[str, Any]] = None

    def _create_session(self) -> requests.Session:
        """Create and configure a requests session for API calls."""
        session = requests.Session()
        session.headers.update(
            {"Content-Type": "application/json", "User-Agent": "GeneIDConverter/1.0"}
        )
        return session

    def _clean_gene_id(self, gene_id: str) -> Optional[str]:
        """Clean and validate a gene ID."""
        if not gene_id:
            return None

        gene_str = str(gene_id).strip()

        if not gene_str or gene_str.lower() in ["nan", "none", "", "null", "na"]:
            return None

        if len(gene_str) > 30:
            return None

        if gene_str.count(";") > 2:
            return None

        return gene_str

    def _create_batches(self, gene_ids: List[str]) -> List[List[str]]:
        """Create batches of cleaned gene IDs for API requests."""
        if not gene_ids:
            return []

        cleaned_ids: List[str] = []
        for gene_id in gene_ids:
            cleaned = self._clean_gene_id(gene_id)
            if cleaned:
                cleaned_ids.append(cleaned)

        if not cleaned_ids:
            log.warn("No valid gene IDs after cleaning")
            return []

        batches: List[List[str]] = []
        for i in range(0, len(cleaned_ids), self.config.batch_size):
            batch_increment = i + self.config.batch_size
            batch = cleaned_ids[i:batch_increment]
            batches.append(batch)

        log.info(f"Created {len(batches)} batches from {len(cleaned_ids)} cleaned IDs")
        return batches

    def _make_api_request(
        self, batch: List[str], attempt: int = 1
    ) -> Optional[List[Dict[str, Any]]]:
        """Make an API request to convert gene IDs."""
        payload = {
            "q": batch,
            "scopes": self.SCOPES_MAP[self.config.id_from],
            "fields": self.FIELDS_MAP[self.config.id_to],
            "species": self.config.species,
            "size": len(batch),
        }

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                response = self.session.post(
                    self.BASE_URL,
                    json=payload,
                    timeout=self.config.timeout,
                    verify=True,
                )

            response.raise_for_status()
            results = response.json()

            if not isinstance(results, list):
                log.error(f"Unexpected API response format: {type(results)}")
                return None

            log.debug(f"Received {len(results)} results from API")

            time.sleep(0.1)

            return results

        except requests.exceptions.Timeout:
            log.warn(f"Request timeout on attempt {attempt}")
        except requests.exceptions.RequestException as e:
            log.warn(f"Request failed on attempt {attempt}: {e}")
        except Exception as e:
            log.error(f"Unexpected error on attempt {attempt}: {e}")

        if attempt < self.config.max_retries:
            wait_time = attempt * 2
            log.info(
                f"Waiting {wait_time}s before retry {attempt + 1}/{self.config.max_retries}"
            )
            time.sleep(wait_time)
            return self._make_api_request(batch, attempt + 1)

        log.error(f"Failed to get results after {self.config.max_retries} attempts")
        return None

    def _process_ensembl_result(
        self, result: Dict[str, Any]
    ) -> Optional[Union[str, List[str]]]:
        """Process the Ensembl field from the API result."""
        ensembl_data = result.get("ensembl")
        if not ensembl_data:
            return None

        if isinstance(ensembl_data, dict):
            return ensembl_data.get("gene")
        elif isinstance(ensembl_data, list) and ensembl_data:
            genes = [
                item["gene"]
                for item in ensembl_data
                if isinstance(item, dict) and "gene" in item
            ]
            if not genes:
                return None
            return genes if len(genes) > 1 else genes[0]

        return None

    def _extract_converted_id(
        self, result: Dict[str, Any]
    ) -> Optional[Union[str, List[str]]]:
        """Extract the converted gene ID from the API result."""
        if result.get("notfound"):
            return None

        if self.config.id_to == "ensembl":
            return self._process_ensembl_result(result)
        else:
            field_name = self.FIELDS_MAP[self.config.id_to]
            converted_id = result.get(field_name)

            if converted_id is not None:
                if isinstance(converted_id, list):
                    return str(converted_id[0]) if converted_id else None
                return str(converted_id)

        return None

    def _process_batch_results(
        self, batch: List[str], results: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Process the API results for a batch of gene IDs."""
        batch_results: Dict[str, str] = {}

        if not results or not batch:
            return batch_results

        query_map: Dict[str, List[tuple[int, Dict[str, Any]]]] = {}
        indexed_results: List[tuple[int, Dict[str, Any], str]] = []

        for idx, result in enumerate(results):
            query_raw = result.get("query", "")
            if isinstance(query_raw, str):
                query_key = query_raw.strip()
            elif query_raw is None:
                query_key = ""
            else:
                query_key = str(query_raw).strip()

            indexed_results.append((idx, result, query_key))

            if query_key:
                query_map.setdefault(query_key, []).append((idx, result))

        used_indices: set[int] = set()

        def _select_from_candidates(
            candidates: List[tuple[int, Dict[str, Any]]],
        ) -> Tuple[Optional[str], Optional[int]]:
            """Select the best converted ID from candidate results."""
            if not candidates:
                return None, None

            sorted_candidates = sorted(
                candidates,
                key=lambda item: (
                    bool(item[1].get("notfound", False)),
                    -float(item[1].get("_score", 0) or 0),
                ),
            )

            for idx, candidate in sorted_candidates:
                try:
                    converted = self._extract_converted_id(candidate)
                except Exception as exc:
                    log.debug(
                        f"Error extracting conversion from candidate at index {idx}: {exc}"
                    )
                    continue
                if converted:
                    return converted, idx
            return None, None

        for gene_id in batch:
            gene_key = str(gene_id).strip()

            candidates: List[tuple[int, Dict[str, Any]]] = []
            for key_variant in {gene_key, gene_key.upper(), gene_key.lower()}:
                if key_variant and key_variant in query_map:
                    candidates = query_map[key_variant]
                    break

            converted_id: Optional[str]
            chosen_idx: Optional[int]

            converted_raw, chosen_idx = _select_from_candidates(candidates)
            converted_id = converted_raw

            if converted_id is None:
                for idx, result, _ in indexed_results:
                    if idx in used_indices:
                        continue
                    try:
                        converted_raw = self._extract_converted_id(result)
                    except Exception as exc:
                        log.debug(
                            f"Error processing fallback result at index {idx}: {exc}"
                        )
                        converted_raw = None
                    if converted_raw:
                        converted_id = (
                            converted_raw
                            if isinstance(converted_raw, str)
                            else str(converted_raw)
                        )
                        chosen_idx = idx
                        break

            if converted_id:
                batch_results[gene_id] = converted_id
                if chosen_idx is not None:
                    used_indices.add(chosen_idx)

        return batch_results

    def convert(
        self, gene_ids: List[str]
    ) -> Union[List[Optional[str]], Tuple[List[Optional[str]], dict]]:
        """Convert a list of gene IDs from one type to another."""
        if not gene_ids:
            log.info("No gene IDs provided for conversion")
            self.last_stats = {
                "success_count": 0,
                "total_count": 0,
                "failure_count": 0,
                "success_rate": 0,
            }
            return []

        original_to_cleaned: Dict[str, str] = {}
        cleaned_ids: List[str] = []

        for original_id in gene_ids:
            cleaned = self._clean_gene_id(original_id)
            if cleaned:
                original_to_cleaned[original_id] = cleaned
                if cleaned not in cleaned_ids:
                    cleaned_ids.append(cleaned)

        if not cleaned_ids:
            log.info("No valid gene IDs found after cleaning")
            self.last_stats = {
                "success_count": 0,
                "total_count": len(gene_ids),
                "failure_count": len(gene_ids),
                "success_rate": 0,
            }
            return [None] * len(gene_ids)

        log.info(
            f"Converting {len(cleaned_ids)} cleaned gene IDs from {self.config.id_from} to {self.config.id_to}"
        )

        batches = self._create_batches(cleaned_ids)
        all_results: Dict[str, str] = {}

        batch_iterator = (
            tqdm(
                batches,
                desc="Converting gene IDs",
                disable=not self.config.show_progress,
            )
            if self.config.show_progress
            else batches
        )

        for i, batch in enumerate(batch_iterator, 1):
            log.debug(f"Processing batch {i}/{len(batches)} with {len(batch)} IDs")

            api_results = self._make_api_request(batch)
            if api_results is None:
                log.warn(f"Failed to get results for batch {i}")
                continue

            batch_results = self._process_batch_results(batch, api_results)
            all_results.update(batch_results)

        final_results: List[Optional[str]] = []
        failure_count = 0

        for original_id in gene_ids:
            cleaned_id = original_to_cleaned.get(original_id)
            if cleaned_id and cleaned_id in all_results:
                final_results.append(all_results[cleaned_id])
            else:
                final_results.append(None)
                failure_count += 1

        success_count = len(gene_ids) - failure_count

        stats: Dict[str, Union[int, float]] = {
            "success_count": success_count,
            "total_count": len(gene_ids),
            "failure_count": failure_count,
            "success_rate": (success_count / len(gene_ids) * 100) if gene_ids else 0,
        }

        self.last_stats = stats

        if self.config.return_stats:
            return final_results, stats

        return final_results


def ConvertGeneID(
    gene_ids: List[str],
    id_from: str = "symbol",
    id_to: str = "entrez",
    species: str = "human",
    batch_size: int = 1000,
    show_progress: bool = True,
    timeout: int = 30,
    max_retries: int = 3,
    return_stats: bool = False,
) -> Union[List[Optional[str]], Tuple[List[Optional[str]], dict]]:
    """Convert a list of gene IDs from one type to another."""
    config = ConversionConfig(
        id_from=id_from,
        id_to=id_to,
        species=species,
        batch_size=batch_size,
        show_progress=show_progress,
        timeout=timeout,
        max_retries=max_retries,
        return_stats=return_stats,
    )
    converter = GeneIDConverter(config)
    return converter.convert(gene_ids)
