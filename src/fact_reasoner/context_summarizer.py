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

# Context summarization using LLMs

import string
import numpy as np

from typing import List
from tqdm import tqdm

# Local
from src.fact_reasoner.utils import strip_string, extract_first_code_block, dotdict
from src.fact_reasoner.llm_handler import LLMHandler
from src.fact_reasoner.prompts import CONTEXT_SUMMARIZATION_PROMPT_V1

class ContextSummarizer:
    """
    Context summarization given the atom.
    """
    def __init__(
            self,
            model_id: str = "llama-3.3-70b-instruct",
            prompt_version: str = "v1",
            backend: str = "rits"
    ):
        """
        Initialize the ContextSummarizer.

        Args:
            model_id: str
                The name/id of the model.
            prompt_version: str
                The prompt version used. Allowed values are v1.
            backend: str
                The model's backend.
        """
        
        self.model_id = model_id
        self.prompt_version = prompt_version
        self.backend = backend
        self.llm_handler = LLMHandler(model_id, backend)

        self.prompt_begin = self.llm_handler.get_prompt_begin()
        self.prompt_end = self.llm_handler.get_prompt_end()

        print(f"[ContextSummarizer] Using LLM on {self.backend}: {self.model_id}")
        print(f"[ContextSummarizer] Using prompt version: {self.prompt_version}")

    def make_prompt(self, atom: str, context: str):
        """        
        Create the prompt for a given atom and context.
        
        Args:
            atom: str
                The input atom (e.g., an atomic unit).
            context: str
                The input context, i.e., a list of previuos queries and results.
        Return:
            A string containing the prompt for the LLM.
        """

        if self.prompt_version == "v1":
            prompt = CONTEXT_SUMMARIZATION_PROMPT_V1.format(
                _ATOM_PLACEHOLDER=atom,
                _CONTEXT_PLACEHOLDER=context,
                _PROMPT_BEGIN_PLACEHOLDER=self.prompt_begin,
                _PROMPT_END_PLACEHOLDER=self.prompt_end
            )
        else:
            raise ValueError(f"Unknown prompt version: {self.prompt_version}. "
                             f"Allowed values are: v1.")
        
        prompt = strip_string(prompt)
        return prompt
        
    def run(self, contexts: List[str], atom: str):
        """
        Generate summaries for a given atom and a list of contexts.
        
        Args:
            contexts: List[str]
                A list of contexts (strings) for which to generate summaries.
            atom: str
                The input atom (e.g., an atomic unit).
        Returns:
            List[dict]
                A list of dictionaries, each containing a summary, context, and probability.
        """

        generated_texts = []
        generated_logprobs = []
        prompts = [self.make_prompt(atom, context) for context in contexts if context != ""]
        print(f"[ContextSummarizer] Prompts created: {len(prompts)}")

        for _, response in tqdm(
            enumerate(
                self.llm_handler.batch_completion(
                    prompts,
                    logprobs=True,
                    temperature=0,
                    seed=42,

                )
            ),
            total=len(prompts),
            desc="Summarization",
            unit="prompts",
            ):
                generated_texts.append(response.choices[0].message.content)
                generated_logprobs.append(response.choices[0].logprobs['content'])

        summaries = []
        for text, logprobs in zip(generated_texts, generated_logprobs):

            if text is not None and logprobs is not None:
                summary = extract_first_code_block(text, ignore_language=True)
                logprob_sum = 0.0
                generated_tokens = logprobs[:-1]
                for token in generated_tokens: #last token is just <|eot_id|>
                    token = dotdict(token)
                    logprob_sum += token.logprob
                probability = np.exp(logprob_sum/len(generated_tokens))
            else: 
                summary = ""
                probability = 0.
            summaries.append({"summary": summary, "probability": probability})
            

        final_summaries = [{"summary": context, "probability": 1.0} for context in contexts]
        j = 0
        for i in range(len(final_summaries)):
            if final_summaries[i]["summary"] != "":
                final_summaries[i]["summary"] = summaries[j]["summary"]
                final_summaries[i]["probability"] = summaries[j]["probability"]
                j += 1
       
        for summary in final_summaries:
            if (len(summary["summary"]) > 0) and (summary["summary"] != "None") and (not summary["summary"][-1] in string.punctuation):
                summary["summary"] += "."

        outputs = []
        for i in range(len(contexts)):
            if len(final_summaries[i]["summary"]) > 0 and final_summaries[i]["summary"] != "None":
                outputs.append(dict(summary=final_summaries[i]["summary"], context=contexts[i], probability=final_summaries[i]["probability"]))
            else:
                outputs.append(dict(summary="", context=contexts[i], probability=final_summaries[i]["probability"]))

        return outputs
    

    def runall(self, contexts: List[List[str]], atoms: List[str]):
        """
        Generate summaries for a list of atoms and a list of contexts.
        
        Args:
            contexts: List[List[str]]
                A list of lists of contexts (strings) for which to generate summaries.
            atoms: List[str]
                A list of atoms (e.g., atomic units).
        Returns:
            List[List[dict]]
                A list of lists of dictionaries, each containing a summary, context, and probability.
        """
        
        n = len(contexts)
        generated_texts = []
        generated_logprobs = []
        prompts = [self.make_prompt(atom, context) for i, atom in enumerate(atoms) for context in contexts[i] if context != ""]
        messages = [[dict(role="user", content=prompt)] for prompt in prompts]
        print(f"[ContextSummarizer] Prompts created: {len(prompts)}")

        for _, response in tqdm(
            enumerate(
                self.llm_handler.batch_completion(
                    prompts,
                    logprobs=True,
                    temperature=0,
                    seed=42
                )
            ),
            total=len(messages),
            desc="Summarization",
            unit="prompts",
            ):
                generated_texts.append(response.choices[0].message.content)
                generated_logprobs.append(response.choices[0].logprobs['content'])


        summaries = []
        for text, logprobs in zip(generated_texts, generated_logprobs):

            if text is not None and logprobs is not None:
                summary = extract_first_code_block(text, ignore_language=True)
                logprob_sum = 0.0
                generated_tokens = logprobs[:-1]
                for token in generated_tokens: #last token is just <|eot_id|>
                    token = dotdict(token)
                    logprob_sum += token.logprob
                probability = np.exp(logprob_sum/len(generated_tokens))
            else: 
                summary = ""
                probability = 0.
            summaries.append({"summary": summary, "probability": probability})
            

        final_summaries = [{"summary": context, "probability": 1.0} for contex in contexts for context in contex]

        j = 0
        for i in range(len(final_summaries)):
            if final_summaries[i]["summary"] != "":
                final_summaries[i]["summary"] = summaries[j]["summary"]
                final_summaries[i]["probability"] = summaries[j]["probability"]
                j += 1

        
        for summary in final_summaries:
            if (len(summary["summary"]) > 0) and (summary["summary"] != "None") and (not summary["summary"][-1] in string.punctuation):
                summary["summary"] += "."


        k = 0        
        outputs = []
        for j in range(n):
            output = [{'summary': final_summaries[k + i]["summary"], 'context': contexts[j][i], 'probability': final_summaries[k + i]["probability"]} if (len(final_summaries[k + i]["summary"]) > 0 and final_summaries[k + i]["summary"] != "None") else {'summary': "", 'context': contexts[j][i], 'probability': final_summaries[k + i]["probability"]} for i in range(len(contexts[j]))]
            outputs.append(output)
            k += len(contexts[j])
        return outputs
        
if __name__ == "__main__":
    
    model_id = "llama-3.3-70b-instruct"
    prompt_version = "v1"
    backend = "rits"
    summarizer = ContextSummarizer(model_id=model_id, prompt_version=prompt_version, backend=backend)

    atom = "The city council has approved new regulations for electric scooters."
    contexts = ["In the past year, the city had seen a rapid increase in the use of electric scooters. They seemed like a perfect solution to reduce traffic and provide an eco-friendly transportation option. However, problems arose quickly. Riders often ignored traffic laws, riding on sidewalks, and causing accidents. Additionally, the scooters were frequently left haphazardly around public spaces, obstructing pedestrians. City officials were under increasing pressure to act, and after numerous public consultations and debates, the council finally passed new regulations. The new rules included mandatory helmet use, restricted riding areas, and designated parking zones for scooters. The implementation of these regulations was expected to improve safety and the overall experience for both scooter users and pedestrians.",
        "With the rise of shared electric scooters and bikes in cities across the country, municipal governments have been scrambling to develop effective policies to handle this new form of transportation. Many cities, including the local area, were caught off guard by the sudden popularity of scooters, and their original infrastructure was ill-prepared for this new trend. Early attempts to regulate the scooters were chaotic and ineffective, often leading to public frustration. Some cities took drastic steps, such as banning scooters altogether, while others focused on infrastructure improvements, like adding dedicated lanes for scooters and bicycles. The city council's recent approval of new regulations was part of a larger effort to stay ahead of the curve and provide a balanced approach to regulating modern transportation options while encouraging their growth. These regulations were designed not only to ensure the safety of riders but also to integrate the scooters more seamlessly into the city's broader transportation network.",
        "",
        "The sun hung low in the sky, casting a warm golden glow over the city as Emily wandered through the bustling streets, her mind drifting between thoughts of the past and the uncertain future. She passed the familiar old bookstore that always smelled like aged paper and adventure, a place she used to frequent with her grandmother, whose absence still left a hollow ache in her chest. The air was thick with the scent of coffee wafting from nearby cafés, mingling with the earthy smell of rain that had yet to fall. Despite the noise of the traffic, the chatter of pedestrians, and the hum of city life, there was a strange sense of stillness around her. It was as if time had slowed down, giving her a moment to breathe, to collect her scattered thoughts. She glanced up at the towering buildings that seemed to stretch endlessly into the sky, their glass facades reflecting the fading light. Everything around her was in constant motion, yet she felt an unexpected calm. Her phone buzzed in her pocket, pulling her back to reality, and she sighed, reluctantly slipping it out. It was a message from her best friend, asking if they still wanted to meet up later."
    ]

    results = summarizer.run(contexts, atom)
    for i, elem in enumerate(results):
        context = elem["context"]
        summary = elem["summary"]
        probability = elem["probability"]
        print(f"\n\nContext #{i + 1}: {context}\n--> Summary #{i + 1}: {summary}\n--> Probability #{i + 1}: {probability}")


    print()


    atoms = [
        "The city council has approved new regulations for electric scooters.",        
        "The team announced a new partnership with a major tech company.",
        "They've developed a new app that helps manage personal finances more effectively."
    ]

    contexts = [
        [
            "In the past year, the city had seen a rapid increase in the use of electric scooters. They seemed like a perfect solution to reduce traffic and provide an eco-friendly transportation option. However, problems arose quickly. Riders often ignored traffic laws, riding on sidewalks, and causing accidents. Additionally, the scooters were frequently left haphazardly around public spaces, obstructing pedestrians. City officials were under increasing pressure to act, and after numerous public consultations and debates, the council finally passed new regulations. The new rules included mandatory helmet use, restricted riding areas, and designated parking zones for scooters. The implementation of these regulations was expected to improve safety and the overall experience for both scooter users and pedestrians.",
            "",
            "As the sun began to set, Sarah made her way to the park to meet up with friends after work. As she walked past the entrance, she noticed several electric scooters parked in random spots. A few of them were right in the middle of the sidewalk, forcing pedestrians to step around them. She rolled her eyes, knowing that the city had been discussing new regulations for scooters for months. Sarah, who had lived in the city for several years, had witnessed how technology could both improve and complicate life. She remembered the early days of rideshare programs like Uber, which were initially unregulated and caused a similar public uproar. Just like with scooters, city officials had scrambled to come up with solutions that balanced convenience and safety. The new scooter regulations were an important step, but Sarah couldn't help but wonder if it would be enough to prevent further accidents. She had heard stories of people crashing into trees or getting hurt due to careless riders. With a sigh, she grabbed her phone to send a quick text to her friends, secretly hoping they wouldn't decide to ride scooters tonight.",
        ],
        [
            "When the announcement was made, it sent ripples of excitement through both the sports and tech communities. The team had been in talks with several major companies, but this deal with the tech giant was unexpected. The new partnership was part of a larger strategy to modernize the team's infrastructure and fan experience. With the help of the tech company, the team would implement advanced analytics to improve training techniques, player performance tracking, and even fan engagement through cutting-edge virtual reality experiences. As part of the deal, the team also planned to unveil a revamped app that would offer fans personalized content, live stats, and direct interactions with players. For the tech company, this partnership was a prime opportunity to showcase its innovative solutions on a global stage, potentially leading to millions of new customers in the sports sector.",
            "Behind the scenes, the negotiations had been intense. The team’s management, along with advisors from the tech company, spent months hammering out the terms of the deal. Initially, there had been resistance on both sides, with each party trying to secure the most advantageous terms. The team was looking for more than just financial support; they wanted access to cutting-edge technologies that could set them apart from their competitors. The tech company, on the other hand, was eager to tap into the rapidly growing sports market, which had proven to be highly lucrative. After several rounds of talks, including visits to the tech company’s headquarters and multiple brainstorming sessions, a partnership was finally agreed upon. Both parties celebrated the deal, knowing that this collaboration could change the way the team trained and interacted with its fans. The team would be the first in their league to introduce such a robust tech-driven approach to player development, and the partnership was expected to serve as a model for other organizations to follow.",
            "Jonathan had always been skeptical of corporate sponsorships in sports. To him, they felt like a distraction from the real essence of the game. He had grown up watching his favorite teams battle it out on the field without the constant bombardment of tech ads or virtual reality experiences. As he sat in the stadium, surrounded by fans excited about the team's new partnership, he couldn't help but feel uneasy. While he understood the business side of things, he worried that the essence of the sport would be lost in the shuffle of corporate interests. Jonathan's concerns were not unique; many fans shared his belief that sports should remain a pure form of entertainment, untainted by outside influences. But as the announcement about the new partnership came over the loudspeaker, he tried to push aside his doubts. Maybe, just maybe, the team would find a way to balance innovation with tradition.",
        ],
        [
            "",
            "Ellen had always been diligent about saving for the future, but when her financial advisor retired, she found herself struggling to keep track of her savings and investments. The spreadsheets she once relied on seemed outdated, and she couldn't find a budgeting tool that worked for her lifestyle. It was then that a friend recommended the new app, which promised to simplify everything. Intrigued, Ellen downloaded it and began the process of linking her bank accounts. To her surprise, the app immediately pulled in her transaction history and categorized her expenses. She was impressed by the level of detail the app provided. It offered insights into her spending habits, helping her identify areas where she could cut back. The best part? The app also included tips on how to invest her savings, using algorithms to recommend strategies that aligned with her risk tolerance. Ellen felt empowered by the app’s comprehensive approach to personal finance and began using it religiously. She also recommended it to her friends and family, confident that it could help them take control of their financial futures as well.",
            "Brian had never been particularly interested in finances. As a freelance graphic designer, his income varied from month to month, making it difficult to plan his spending. He lived in a small apartment, often struggled to pay bills on time, and would occasionally splurge on a new piece of equipment for his studio without thinking about the long-term impact on his budget. While he had heard about the new finance app, he was skeptical. After all, he didn’t want a program telling him what to do with his money. However, after an unexpected tax bill arrived, Brian realized that he needed to take a more serious approach to managing his finances. He decided to give the app a try. At first, he found it annoying that the app tracked every single transaction, but soon he began to appreciate its guidance. The app helped him set realistic financial goals, and with its alerts and reminders, he managed to avoid late fees. While Brian still wasn’t thrilled by the idea of budgeting, he had to admit that the app had made his financial life significantly easier."
        ] 

    ]

    results = summarizer.runall(contexts, atoms)
    print(f"Number of results: {len(results)}")
    for i, result in enumerate(results):
        for j, elem in enumerate(result):
            context = elem["context"]
            summary = elem["summary"]
            probability = elem["probability"]
            print(f"\n\nContext #{i + 1}.{j + 1}: {context}\n--> Summary #{i + 1}.{j + 1}: {summary}\n--> Probability #{i + 1}.{j + 1}: {probability}")

    print()

    print("Done.")
