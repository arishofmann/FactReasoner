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

# Atomic fact decontextualization using LLMs

import string

from typing import List, Any
from tqdm import tqdm

# Local imports
from src.fact_reasoner.utils import strip_string, extract_first_code_block, extract_last_wrapped_response
from src.fact_reasoner.llm_handler import LLMHandler
from src.fact_reasoner.prompts import ATOM_REVISER_PROMPT_V1, ATOM_REVISER_PROMPT_V2

class AtomReviser:
    """
    Atomic unit decontextualization using LLMs.
    
    """

    def __init__(
            self,
            model_id: str = "llama-3.3-70b-instruct",
            prompt_version: str = "v1",
            backend: str = "rits"
    ):
        """
        Initialize the AtomReviser with the specified model and prompt version.

        Args:
            mode_id: str
                The name/id of the model.
            prompt_version: str
                The prompt version used. Allowed values are v1 - newer, v2 - original.
            backend: str
                The model's backend.
        """
        
        self.model_id = model_id
        self.backend = backend
        self.prompt_version = prompt_version
        self.llm_handler = LLMHandler(model_id, backend)

        self.prompt_begin = self.llm_handler.get_prompt_begin()
        self.prompt_end = self.llm_handler.get_prompt_end()
            
        print(f"[AtomReviser] Using LLM on {self.backend}: {self.model_id}")
        print(f"[AtomReviser] Using prompt version: {self.prompt_version}")

    def make_prompt(self, unit: str, response: str):
        """
        Create a prompt for the LLM to decontextualize the atomic unit.
        
        Args:
            unit: str
                The atomic unit to be decontextualized.
            response: str
                The context or response from which the atomic unit is extracted.
        
        Returns:
            str: The formatted prompt for the LLM.
        """
        
        if self.prompt_version == "v1":
            prompt = ATOM_REVISER_PROMPT_V1.format(
                _UNIT_PLACEHOLDER=unit,
                _RESPONSE_PLACEHOLDER=response,
                _PROMPT_BEGIN_PLACEHOLDER=self.prompt_begin,
                _PROMPT_END_PLACEHOLDER=self.prompt_end
            )
        elif self.prompt_version == "v2":
            prompt = ATOM_REVISER_PROMPT_V2.format(
                _UNIT_PLACEHOLDER=unit,
                _RESPONSE_PLACEHOLDER=response,
                _PROMPT_BEGIN_PLACEHOLDER=self.prompt_begin,
                _PROMPT_END_PLACEHOLDER=self.prompt_end
            )
        else:
            raise ValueError(f"Unknown prompt version: {self.prompt_version}. "
                             f"Supported versions are: 'v1', 'v2'.")
        
        prompt = strip_string(prompt)
        
        return prompt
        
    def run(self, atoms: List[str], response: str):
        """
        Decontextualize a list of atomic units from a given response.
        
        Args:
            atoms: List[str]
                A list of atomic units to be decontextualized.
            response: str
                The response from which the atomic units are decontextualized.
        """
        
        results = []
        prompts = [self.make_prompt(atom, response) for atom in atoms]
        print(f"[AtomReviser] Prompts created: {len(prompts)}")

        for _, response in tqdm(
            enumerate(
                self.llm_handler.batch_completion(
                    prompts    
                )
            ),
            total=len(prompts),
            desc="Decontextualization",
            unit="prompts",
            ):
                results.append(response.choices[0].message.content)

        revised_atoms = []
        if self.prompt_version == "v1":
            revised_atoms = [extract_first_code_block(output, ignore_language=True) for output in results]
        elif self.prompt_version == "v2":
            revised_atoms = [extract_last_wrapped_response(output) for output in results]

        for revised_atom in revised_atoms:
            if len(revised_atom) > 0:
                if not revised_atom[-1] in string.punctuation:
                    revised_atom += "."

        final_revised_atoms = []
        for i in range(len(atoms)):
            if len(revised_atoms[i]) > 0:
                final_revised_atoms.append(dict(revised_atom=revised_atoms[i], atom=atoms[i]))
            else:
                final_revised_atoms.append(dict(revised_atom=atoms[i], atom=atoms[i]))

        return final_revised_atoms
    
    def runall(self, atoms: List[List[str]], responses: List[str]) -> List[List[dict[str, Any]]]:
        """
        Decontextualize a list of atomic units from multiple responses.

        Args:
            atoms: List[List[str]]
                A list of lists, where each sublist contains atomic units to be decontextualized.
            responses: List[str]
                A list of responses corresponding to each sublist of atomic units.
        Returns:
            List[List[dict]]:
                A list of lists, where each sublist contains dictionaries with 'revised_atom' and 'atom' keys.
        """

        n = len(responses)
        results = []
        prompts = [self.make_prompt(atom, response) for i, response in enumerate(responses) for atom in atoms[i]]
        print(f"[AtomReviser] Prompts created: {len(prompts)}")

        for _, response in tqdm(
            enumerate(
                self.llm_handler.batch_completion(
                    prompts
                )
            ),
            total=len(prompts),
            desc="Decontextualization",
            unit="prompts",
            ):
                results.append(response.choices[0].message.content)

        revised_atoms = []
        if self.prompt_version == "v1":
            revised_atoms = [extract_first_code_block(output, ignore_language=True) for output in results]
        elif self.prompt_version == "v2":
            revised_atoms = [extract_last_wrapped_response(output) for output in results]

        # TODO: need to fix the problematic revised atoms!!
        for revised_atom in revised_atoms:
            if len(revised_atom) > 0 and not revised_atom[-1] in string.punctuation:
                revised_atom += "."

        k = 0        
        outputs = []
        for j in range(n):
            output = [{'revised_atom': revised_atoms[k + i], 'atom': atoms[j][i]} for i in range(len(atoms[j]))]
            outputs.append(output)
            k += len(atoms[j])

        return outputs
        
if __name__ == "__main__":
    
    model_id = "granite-3.2-8b-instruct"
    prompt_version = "v2"
    backend = "rits"

    response = "Lanny Flaherty is an American actor born on December 18, 1949, \
        in Pensacola, Florida. He has appeared in numerous films, television \
        shows, and theater productions throughout his career, which began in the \
        late 1970s. Some of his notable film credits include \"King of New York,\" \
        \"The Abyss,\" \"Natural Born Killers,\" \"The Game,\" and \"The Straight Story.\" \
        On television, he has appeared in shows such as \"Law & Order,\" \"The Sopranos,\" \
        \"Boardwalk Empire,\" and \"The Leftovers.\" Flaherty has also worked \
        extensively in theater, including productions at the Public Theater and \
        the New York Shakespeare Festival. He is known for his distinctive looks \
        and deep gravelly voice, which have made him a memorable character \
        actor in the industry."

    atoms = [
        "He has appeared in numerous films.",
        "He has appeared in numerous television shows.",
        "He has appeared in numerous theater productions.",
        "His career began in the late 1970s."
    ]

    reviser = AtomReviser(model_id=model_id, prompt_version=prompt_version, backend=backend)
    results = reviser.run(atoms, response)
    for elem in results:
        orig_atom = elem["atom"]
        revised_atom = elem["revised_atom"]
        print(f"{orig_atom} --> {revised_atom}")

    responses = [
        "Gerhard Fischer is an inventor and entrepreneur who is best known \
        for inventing the first handheld, battery-operated metal detector in 1931. \
        He was born on July 23, 1904, in Frankfurt, Germany, and moved to the \
        United States in 1929, where he became a citizen in 1941.\n\nFischer's metal \
        detector was originally designed to find and remove nails and other metal \
        debris from wood used in construction projects. However, it soon became \
        popular among treasure hunters looking for buried artifacts and coins.\n\nIn addition \
        to his work on metal detectors, Fischer also invented a number of other \
        devices, including a waterproof flashlight and a portable radio receiver. \
        He founded the Fischer Research Laboratory in 1936, which became one of the \
        leading manufacturers of metal detectors in the world.\n\nFischer received \
        numerous awards and honors for his inventions, including the Thomas A. \
        Edison Foundation Gold Medal in 1987. He passed away on February 23, 1995, \
        leaving behind a legacy of innovation and entrepreneurship.",
        
        "Lanny Flaherty is an American actor born on December 18, 1949, in \
        Pensacola, Florida. He has appeared in numerous films, television shows, \
        and theater productions throughout his career, which began in the late 1970s. \
        Some of his notable film credits include \"King of New York,\" \"The Abyss,\" \
        \"Natural Born Killers,\" \"The Game,\" and \"The Straight Story.\" On television, \
        he has appeared in shows such as \"Law & Order,\" \"The Sopranos,\" \"Boardwalk Empire,\" \
        and \"The Leftovers.\" Flaherty has also worked extensively in theater, \
        including productions at the Public Theater and the New York Shakespeare \
        Festival. He is known for his distinctive looks and deep gravelly voice, \
        which have made him a memorable character actor in the industry."
    ]

    atoms = [
        [
            "He was born on July 23, 1904.",
            "He was born in Frankfurt, Germany.",
            "He moved to the United States in 1929.",
            "He became a citizen in the United States in 1941.",
        ],
        [
            "He has appeared in numerous films.",
            "He has appeared in numerous television shows.",
            "He has appeared in numerous theater productions.",
            "His career began in the late 1970s."
        ] 
    ]

    results = reviser.runall(atoms, responses)
    print(f"Number of results: {len(results)}")
    for result in results:
        for elem in result:
            orig_atom = elem["atom"]
            revised_atom = elem["revised_atom"]
            print(f"{orig_atom} --> {revised_atom}")


    print("Done.")
