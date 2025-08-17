import asyncio
import json
import os
import argparse
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from mistralai import Mistral
from .prompts_fluency import FLUENCY_PROMPT_TEMPLATE

load_dotenv()

class EvalResponse(BaseModel):
    answer: int = Field(..., description="The category number that best matches the evaluation of the text.")
    reasoning: str = Field(..., description="A short explanation of the evaluation.")

async def evaluate_fluency_mistral(continuation: str, client: Mistral, semaphore: asyncio.Semaphore, model_name: str, retries: int = 3) -> EvalResponse:
    """
    Sends a single text to the Mistral API for evaluation.
    """
    async with semaphore:
        prompt = FLUENCY_PROMPT_TEMPLATE.format(continuation=continuation)
        current_attempt = 0
        while current_attempt < retries:
            try:
                print(f"Processing continuation: \"{continuation[:50]}...\"")
                response = await client.chat.complete_async(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    response_format={"type": "json_object"},
                    temperature=0
                )
                print(f"Response: {response.choices[0].message.content[:50]}...{response.choices[0].message.content[-20:]}")
                fluency_result = response.choices[0].message.content
                
                if fluency_result.startswith("```json"):
                    json_content = fluency_result.removeprefix("```json\n").removesuffix("\n```").strip()
                elif fluency_result.startswith("```"):
                    json_content = fluency_result.removeprefix("```\n").removesuffix("\n```").strip()
                else:
                    json_content = fluency_result

                data = json.loads(json_content)
                return data

            except Exception as e:
                print(f"API error or unexpected issue for sentence: \"{continuation[:50]}...\". Attempt {current_attempt + 1}/{retries}. Error: {e}")
                current_attempt += 1
                if current_attempt == retries:
                    raise e
                await asyncio.sleep(2**current_attempt)

async def eval_main_generations(args, client, semaphore):
    with open(args.input_file, "r") as f:
        data = json.load(f)

    output = []
    tasks = []
    for item in data:
        context = item.get("context", "")
        continuation = item.get("completion", "")

        if not context or not continuation:
            raise ValueError("Context and continuation must not be empty.")
        
        tasks.append(evaluate_fluency_mistral(continuation, client, semaphore, args.model_name))
    
    results = await asyncio.gather(*tasks)
    for i in range(len(data)):
        fluency_result = results[i]
        
        output.append({
            "context": data[i].get("context", ""),
            "completion": data[i].get("completion", ""),
            "fluency": fluency_result["answer"],
            "fluency_reasoning": fluency_result["reasoning"]
        })
    
    print(f"--- Completed Evaluation. Evaluated {len(output)} samples. ---")
    with open(args.output_json_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model-name", type=str,
                        help="Name of the model to use.")
    parser.add_argument("--input-file", type=str, required=True,
                        help="Path to the input file.")
    parser.add_argument("--output-json-file", type=str,
                        help="Path to the output JSON file.")
    parser.add_argument("--max-concurrent-requests", type=int, default=10,
                        help="Maximum number of concurrent API requests.")
    
    parsed_args = parser.parse_args()
    client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
    semaphore = asyncio.Semaphore(parsed_args.max_concurrent_requests)
    asyncio.run(eval_main_generations(parsed_args, client, semaphore))
