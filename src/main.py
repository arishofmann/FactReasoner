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

import argparse

from fact_reasoner.pipeline import FactReasoner

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
        help="Name of the model used internally"
    )

    parser.add_argument(
        '--version',
        type=int,
        default=1,
        help="FactReasoner version: 1, 2 or 3"
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="rits"
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
        '--test', 
        default=False, 
        action='store_true', 
        help="Debugging mode."
    )

    parser.add_argument(
        '--merlin_path',
        type=str,
        default="/home/radu/git/fm-factual/lib/merlin",
        help="Path to the probabilistic inference merlin."
    )

    args = parser.parse_args()

    if args.test:
        test()
        sys.exit(0)
