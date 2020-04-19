import itertools
import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

import fire
from tqdm import tqdm

# TODO (John): Filter to PMCIDs only (or at least provide the option)
# TODO (John): Filter out abstracts will less than a certain number of words?
# Query this: https://tinyurl.com/tbcatjg
# And if there is no PMCID under "articleids", this article is not in PMC.


def main(
    input_dir: str,
    output_dir: str,
    max_triplets: float,
    pos_threshold: float = 0.2,
    neg_threshold: float = 0.001,
    neg_margin: float = 0.015,
    cache: bool = False,
) -> None:
    """Generates a set of triplets of PMIDs which, among other things, can be used to evaluate document embeddings.

    Randomly generates triplets of PMIDs, where the first PMID (the "anchor") is more similar to the second PMID
    (the "positive" example) than the third (the "negative" example). This standard can be used to evaluate
    document embeddings, i.e. by computing the fraction of triplets where the document embeddings for the anchor
    and the positive example are more similar than the anchor and the negative example, according to some
    similarity metric (e.g. cosine).

    The similarity of two PMIDs is computed as

            similarity(pmid_i, pmid_j) = no. of shared references + citations / total no. of references + citations

    A candidate triplet is retained only if all of the following equalities hold:

        1. similarity(anchor, positive) >= `pos_threshold` and
        2. similarity(anchor, negative) >= `neg_threshold` and
        3. similarity(anchor, negative) <= `pos_threshold` - `neg_margin`

    The first threshold simply ensures that we choose valid positive examples. The second threshold allows us to
    mine hard examples by ensuring that the negative example is at least partially related to the anchor. The final
    threshold helps ensure we don't choose a positive and negative example that are too close in similarity to the
    anchor. The resulting triplets are stored as a JSON Lines file and saved to `output_dir/triplets.jsonl`.

    You will need to have saved an iCite Database Snapshot locally. Snapshots can be obtained at:

        https://nih.figshare.com/collections/iCite_Database_Snapshots_NIH_Open_Citation_Collection_/4586573

    Specifically, you need to download and extract a metadata snapshot in JSON lines format. E.g. The December
    2019 Snapshot:

        https://nih.figshare.com/ndownloader/files/18535490

    Args:
        input_dir (str): Path to a directory containing a iCite metadata dump in JSON lines format OR path to a
            directory containing the cached citation info. To create this cache, run this script with `cache=True`.
        output_dir (str): Path to save the triplets as a JSON Lines file.
        max_triplets (int, optional): The maximum number of triplets to generate.
        pos_threshold (float, optional): The minimum similarity score between an example and the anchor for that
            example to be considered "positive". Defaults to `0.01`.
        neg_threshold (float, optional): Only triplets whose positive and negative examples have a difference
            in similarity score to the anchor less than or equal to this value are retained. This helps
            in mining difficult examples. If `1.0`, all triplets are retained. Defaults to `0.01`.
        neg_margin (float, optional): TODO.
        cache (bool, optional): True if the citation data extracted from the iCite metadata dump at `input_dir`
            should be pickled and saved to disk at `output_dir/citations_cache.pickle`. After caching,
            you can re-run this script with `input_dir` set to the directory that contains the pickle. Defaults to
            False.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    triplets_filepath = output_dir / "triplets.jsonl"

    citations = _load_and_cache_icite_metadata(input_dir, output_dir, cache)
    triplets = _generate_triplets(citations, max_triplets, pos_threshold, neg_threshold, neg_margin)
    _save_triplets_to_disk(triplets_filepath, triplets)


def _load_and_cache_icite_metadata(
    input_dir: Path, output_dir: Path, cache: bool
) -> Dict[str, Dict[str, set]]:
    """"Load citations from disk if they are cached. Otherwise, parse the input dump."

    Args:
        input_dir (Path): input_dir (str): Path to a dictionary containing a iCite metadata dump in JSON lines
            format OR path to a directory containing the cached citation info. To create this cache, run this
            script with `cache=True`.
        cache_filepath (Path): Path to the pickled citation data.
        cache (bool): True if the citation data extracted from the iCite metadata dump at `input_dir`
            should be pickled and saved to disk at `output_dir/citations_cache.pickle`. This has no effect
            if `cache_filepath.is_file()`. Defaults to False.

    Returns:
        dict: A dictionary, keyed by PMID, containing the citations and refrences for a given PMID.
    """

    # Load citations from disk if they are cached. Otherwise, parse the input dump
    cache_filepath = input_dir / "citations_cache.pickle"
    if cache_filepath.is_file():
        print(f"Loading cached citations from {cache_filepath}...", end=" ", flush=True)
        with open(cache_filepath, "rb") as f:
            citations = pickle.load(f)
        print("Done.")
    else:
        print(f"Loading citations from {input_dir}...")
        citations = _parse_icite_metadata(input_dir)
        if cache:
            cache_filepath = output_dir / "citations_cache.pickle"
            print(f"Caching loaded citations to {cache_filepath}...", end=" ", flush=True)
            with open(cache_filepath, "wb") as f:
                pickle.dump(citations, f)
            print("Done.")

    return citations


def _parse_icite_metadata(
    input_dir: Path, research_only: bool = True, citation_count_threshold: int = 10
) -> Dict[str, set]:

    citations = {}
    for path in input_dir.iterdir():
        if not path.name.endswith(".json"):
            continue
        with open(path, "r") as f:
            for line in tqdm(f):
                contents = json.loads(line)

                # Filter anything that isn't a reasearch article or has less than citation_count_threshold
                # citations
                if (not contents["is_research_article"] and research_only) or contents[
                    "citation_count"
                ] < citation_count_threshold:
                    continue

                pmid = str(contents["pmid"])
                citations[pmid] = set()

                # Either references or cited_by can be null
                if contents["references"] is not None:
                    citations[pmid].update(
                        {str(pmid) for pmid in contents["references"].strip().split()}
                    )
                if contents["cited_by"] is not None:
                    citations[pmid].update(
                        {str(pmid) for pmid in contents["cited_by"].strip().split()}
                    )

    return citations


def _compute_similarity(
    candidate_pmids: itertools.combinations,
    citations: Dict[str, Dict[str, List]],
    pos_threshold: float,
    neg_threshold: float,
    neg_margin: float,
) -> List[Tuple[str, str, str]]:
    """Returns triplets of PMIDs from `pmid_pairs` where the first two PMIDs are most closely related. Only PMIDs
    with similarity scores greatert than or equal to `pos_threshold` are retained. The similarity score is computed
    as:

        similarity(pmid_i, pmid_j) = no. of shared references + citations / total no. of references + citations

    Args:
        candidate_pmids (TODO): TODO.
        citations (TODO): TODO.
        pos_threshold (float): A threshold value for storing the similarity of two papers. Only pairs of
            papers with similarity scores greater than or equal to this threshold will be stored and saved to disk.
            This is useful to prevent the pickled dictionary saved to `output_dir` from being too large. If `0.0`,
            all pairwise similarity scores are saved.
        pos_neg_threshold (float): TODO.


    Returns:
        List: A list of tuples containing tripets of PMIDs from `pmid_pairs` where the first two PMIDs are most
            closely related.
    """
    triplets = []

    for pmid_i, pmid_j, pmid_k in candidate_pmids:
        # TODO (John): Not exactly sure why this occurs.
        if any(pmid not in citations for pmid in (pmid_i, pmid_j, pmid_k)):
            continue

        shared_citations = [
            len(citations[pmid_i] & citations[pmid_j]),
            len(citations[pmid_i] & citations[pmid_k]),
        ]
        total_citations = [
            len(citations[pmid_i] | citations[pmid_j]),
            len(citations[pmid_i] | citations[pmid_k]),
        ]
        # If a pair of papers has no citations, default to similarity score of 0
        similarity_scores = [
            shared_citations[0] / total_citations[0] if total_citations[0] else 0,
            shared_citations[1] / total_citations[1] if total_citations[1] else 0,
        ]

        # A triplet is valid if the difference in similarity score of the positive and negative examples
        # is less than pos_neg_threshold and the positive examples similarity score exceeds pos_threshold
        # pmid_i is the positive example and pmid_j is the negative example
        if similarity_scores[0] >= pos_threshold:
            if (
                similarity_scores[1] >= neg_threshold
                and similarity_scores[1] <= pos_threshold - neg_margin
            ):
                triplets.append((pmid_i, pmid_j, pmid_k))
        # pmid_j is the positive example and pmid_i is the negative example
        elif similarity_scores[1] >= pos_threshold:
            if (
                similarity_scores[0] >= neg_threshold
                and similarity_scores[0] <= pos_threshold - neg_margin
            ):
                triplets.append((pmid_i, pmid_k, pmid_j))

    return triplets


def _generate_triplets(
    citations: Dict[str, List[str]],
    max_triplets: int,
    pos_threshold: float,
    neg_threshold: float,
    neg_margin: float,
) -> List[Tuple[str, str, str]]:

    pmids = set(citations.keys())
    spent_pmids = set()  # Accumulator of PMIDs that have been included in at least one triplet
    triplets = []

    # Because it is not feasible to compute all pariwise similarity scores, we instead randomly choose a
    # PMID and produce all combinations of this PMID, its citations and its references. The thinking is that
    # these PMID pairs are most likely to represnt similar papers, making the search over PubMed much faster.
    pbar = tqdm(desc="Generating triplets", total=max_triplets, unit="triplets", dynamic_ncols=True)
    while True:
        # Faster than random.sample on a set, see: https://stackoverflow.com/a/24949742/6578628
        random_pmid = random.choice(tuple(pmids - spent_pmids))
        sampled_pmids = [random_pmid] + list(citations[random_pmid] - spent_pmids)
        candidate_pmids = itertools.combinations(sampled_pmids, r=3)

        candidate_triplets = _compute_similarity(
            candidate_pmids, citations, pos_threshold, neg_threshold, neg_margin
        )

        for triplet in candidate_triplets:
            # Only retain triplets with unspent PMIDs
            if all(pmid not in spent_pmids for pmid in triplet):
                triplets.append(triplet)
                spent_pmids.update({*triplet})

                pbar.update()

            # Exist the loop if we have accumulated the requested number of triplets
            if len(triplets) >= max_triplets:
                break

        # A little ugly, but we need a second break here to exit the while loop
        if len(triplets) >= max_triplets:
            pbar.close()
            triplets = triplets[:max_triplets]  # drop triplets over the max number requested
            break

    print(
        (
            f"Generated {len(triplets)} triplets of PMIDs with a positive threshold of {pos_threshold}, a"
            f" negative threshold of {neg_threshold} and a negative margin of {neg_margin}"
        )
    )

    return triplets


def _save_triplets_to_disk(triplets_filepath: Path, triplets: Tuple[str, str, str]) -> None:
    print(f"Writing triplets to {triplets_filepath}...", end=" ", flush=True)
    with open(triplets_filepath, "w") as f:
        for anchor, positive, negative in triplets:
            json.dump({"anchor": anchor, "positive": positive, "negative": negative}, f)
            f.write("\n")
    print("Done.")


if __name__ == "__main__":
    fire.Fire(main)
