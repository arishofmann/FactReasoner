# coding=utf-8
# Copyright 2023-present the International Business Machines.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Main runner script

import os
import json
import argparse
import pandas as pd

# Local imports
from src.fact_reasoner.factreasoner import FactReasoner
from src.fact_reasoner.baselines.factscore import FactScore
from src.fact_reasoner.baselines.factverify import FactVerify
from src.fact_reasoner.baselines.veriscore import VeriScore
from src.fact_reasoner.atom_extractor import AtomExtractor
from src.fact_reasoner.atom_reviser import AtomReviser
from src.fact_reasoner.context_retriever import ContextRetriever
from src.fact_reasoner.query_builder import QueryBuilder
from src.fact_reasoner.context_summarizer import ContextSummarizer
from src.fact_reasoner.nli_extractor import NLIExtractor
from src.fact_reasoner.fact_graph import FactGraph
from src.fact_reasoner.fact_utils import (
    Atom, 
    Context, 
    Relation, 
    build_atoms, 
    build_contexts, 
    build_relations,
    remove_duplicated_contexts, 
    remove_duplicated_atoms, 
    is_relevant_context, 
    PRIOR_PROB_ATOM, 
    PRIOR_PROB_CONTEXT
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_file',
        type=str,
        default=None,
        help="Path to the input dataset (jsonl)."
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help="Path to the output directory."
    )

    parser.add_argument(
        '--cache_dir',
        type=str,
        default=None,
        help="Path to the cache directory."
    )

    parser.add_argument(
        '--dataset_name',
        type=str,
        default=None,
        help="Name of the dataset."
    )

    parser.add_argument(
        '--service_type',
        type=str,
        default="google",
        help="Service type (langchain, chromadb, google)."
    )

    parser.add_argument(
        '--model_id',
        type=str,
        default=None,
        help="Name of the model used by the pipeline."
    )

    parser.add_argument(
        '--pipeline',
        type=str,
        default="factreasoner",
        required=True,
        help="Factuality pipeline (factreasoner, factscore, veriscore, factverify)."
    )

    parser.add_argument(
        '--version',
        type=int,
        default=2,
        help="FactReasoner version: 1, 2 or 3"
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="rits",
        help="The model's backend (rits, hf, wx)"
    )

    parser.add_argument(
        '--top_k', 
        type=int, 
        default=3, 
        help="Top k results retrieved as contexts per atom."
    )

    parser.add_argument(
        '--use_priors', 
        default=False, 
        action='store_true', 
        help="Use the atom and context priors in the factor definition."
    )

    parser.add_argument(
        '--use_query_builder', 
        default=False, 
        action='store_true', 
        help="Use the QueryBuilder to generate queries for Google search."
    )

    parser.add_argument(
        '--text_only', 
        default=False, 
        action='store_true', 
        help="Contexts are considered text only."
    )

    parser.add_argument(
        '--nli_prompt_version', 
        type=str, 
        default="v1", 
        help="NLI prompt version: v1 (original) or v2 (more recent - some reasoning)"
    )

    parser.add_argument(
        '--atomizer_prompt_version', 
        type=str, 
        default="v2", 
        help="Atomizer prompt version: v1 (original) or v2 (newer)"
    )

    parser.add_argument(
        '--reviser_prompt_version', 
        type=str, 
        default="v1", 
        help="Reviser prompt version: v1 (newer) or v2 (original)"
    )

    parser.add_argument(
        '--merlin_path',
        type=str,
        default="/home/radu/git/fm-factual/lib/merlin",
        help="Path to the probabilistic inference engine merlin."
    )

    # Parse the CLI arguments
    args = parser.parse_args()

    # FactReasoner versions:
    if args.version == 1:  # 1 - context-atom relationships only, allow duplicated contexts
        rel_context_context = False
        remove_duplicates = False
        contexts_per_atom_only = True
        option = "1"
    elif args.version == 2:  # 2 - context-atom relationships only, no duplicated contexts
        rel_context_context = False
        remove_duplicates = True
        contexts_per_atom_only = False
        option = "2"
    elif args.version == 3:  # 3 - context-atom and context-context relationships, no duplicated contexts
        rel_context_context = True
        remove_duplicates = True
        contexts_per_atom_only = False
        option = "3"
    else:
        raise ValueError(f"Unknown FactReasoner version: {args.version}")

    # Create the atom extractor
    atom_extractor = AtomExtractor(
        model_id=args.model_id, 
        prompt_version=args.atomizer_prompt_version,
        backend=args.backend
    )
    
    # Create the atom reviser
    atom_reviser = AtomReviser(
        model_id=args.model_id, 
        prompt_version=args.reviser_prompt_version,
        backend=args.backend
    )

    # Create the NLI extractor
    nli_extractor = NLIExtractor(
        model_id=args.model_id, 
        prompt_version=args.nli_prompt_version, 
        backend=args.backend
    )

    # Create the Query Builder
    if args.use_query_builder:
        query_builder = QueryBuilder(model_id=args.model_id, backend=args.backend)
    else:
        query_builder = None

    # Create context retriever
    context_retriever = ContextRetriever(
        service_type=args.service_type, 
        top_k=args.top_k, 
        cache_dir=args.cache_dir,
        query_builder=query_builder
    )

    print(f"Processing input dataset: {args.input_file}")
    filename = args.input_file  # a jsonl file

    with open(filename) as f:
        lines = f.read().splitlines()
    df_inter = pd.DataFrame(lines)
    df_inter.columns = ['json_element']
    df_inter['json_element'].apply(json.loads)
    df = pd.json_normalize(df_inter['json_element'].apply(json.loads))
    dataset = df.to_dict('records')

    print(f"Loading data from: {filename}")
    print(f"Found {len(dataset)} elements")

    if args.pipeline in ["factscore", "factverify", "veriscore"]:
        pipeline_name = args.pipeline
    elif args.pipeline == "factreasoner":
        pipeline_name = f"{args.pipeline}{args.version}"
    else:
        raise ValueError(f"Unknown pipeline: {args.pipeline}. Aborting.")

    # Check if previous results exist. If yes, load them and skip over them
    # when processing the input dataset.
    filename = "eval_results_{}_{}_{}_{}.jsonl".format(
        pipeline_name,
        args.service_type,
        args.dataset_name,
        args.model_id
    )

    output_filename = os.path.join(args.output_dir, filename)
    print(f"Reading previous results from: {output_filename}")
    evaluation_data = []
    if os.path.isfile(output_filename):
        with open(output_filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                evaluation_data.append(json.loads(line))

    print(f"Found {len(evaluation_data)} existing evaluations data.")
    print(f"Using factuality pipeline: {pipeline_name}")

    # Loop over the data points in the dataset
    for input_data in dataset:
        # Check if current data has been processed already
        processed = False
        for eval_data in evaluation_data:
            if eval_data["input"] == input_data["input"]:
                processed = True
                break
        if processed:
            prompt = input_data["input"]
            print(f"Input: {prompt} already processed.")
            continue

        # Process the data point with the FactReasoner pipeline
        if args.pipeline == "factreasoner":
            pipeline = FactReasoner(
                context_retriever=context_retriever,
                atom_extractor=atom_extractor,
                atom_reviser=atom_reviser,
                nli_extractor=nli_extractor,
                query_builder=query_builder,
                merlin_path=args.merlin_path,
                use_priors=args.use_priors
            )
        elif args.pipeline == "factscore":
            pipeline = FactScore(
                context_retriever=context_retriever,
                atom_extractor=atom_extractor,
                atom_reviser=atom_reviser,
                model_id=args.model_id,
                debug_mode=False,
                add_topic=True,
                backend=args.backend
            )
        elif args.pipeline == "veriscore":
            pipeline = VeriScore(
                context_retriever=context_retriever,
                atom_extractor=atom_extractor,
                atom_reviser=atom_reviser,
                model_id=args.model_id,
                debug_mode=False,
                binary_output=False,
                backend=args.backend
            )
        elif args.pipeline == "factverify":
            pipeline = FactVerify(
                context_retriever=context_retriever,
                atom_extractor=atom_extractor,
                atom_reviser=atom_reviser,
                model_id=args.model_id,
                backend=args.backend
            )

        # Load the problem instance from a file or dict
        ok = pipeline.from_dict_with_contexts(input_data)
        if not ok:
            continue  # annotations are null (ignore)

        # Build the FactReasoner pipeline
        if args.pipeline == "factreasoner":
            pipeline.build(
                remove_duplicates=remove_duplicates,
                contexts_per_atom_only=contexts_per_atom_only,
                has_atoms=True,
                has_contexts=True,
                revise_atoms=False,
                rel_atom_context=True,
                rel_context_context=rel_context_context,
                text_only=args.text_only
            )

            results, marginals = pipeline.score()
            results["model_name"] = args.model_id
            evaluation_data.append(results)
            print(f"[FactReasoner] Marginals: {marginals}")
            print(f"[FactReasoner] Results: {results}")
        elif args.pipeline == "factscore":
            pipeline.build(
                has_atoms=True,
                has_contexts=True,
                decontextualize_atoms=False
            )

            # Print the results
            results = pipeline.score()
            results["model_name"] = args.model_id
            evaluation_data.append(results)
            print(f"[FactScore] Results: {results}")
        elif args.pipeline == "veriscore":
            pipeline.build(
                has_atoms=True,
                has_contexts=True,
                decontextualize_atoms=False
            )

            # Print the results
            results = pipeline.score()
            results["model_name"] = args.model_id
            evaluation_data.append(results)
            print(f"[VeriScore] Results: {results}")
        elif args.pipeline == "factverify":
            pipeline.build(
                has_atoms=True,
                has_contexts=True,
                decontextualize_atoms=False
            )

            # Print the results
            results = pipeline.score()
            results["model_name"] = args.model_id
            evaluation_data.append(results)
            print(f"[FactVerify] Results: {results}")

        # Save results to a file
        filename = "eval_results_{}_{}_{}_{}.jsonl".format(
            pipeline_name,
            args.service_type,
            args.dataset_name,
            args.model_id
        )
        output_filename = os.path.join(args.output_dir, filename)
        print(f"Writing results to: {output_filename}")
        with open(output_filename, "w") as f:
            for res in evaluation_data:
                f.write(f"{json.dumps(res)}\n")

    print("Done.")
