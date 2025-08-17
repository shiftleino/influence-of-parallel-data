import asyncio
import json
import os
import argparse
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from mistralai import Mistral
from .prompts import COMPARISON_PROMPT_TEMPLATE

load_dotenv()

class EvalResponse(BaseModel):
    answer: str = Field(..., description="The completion that fits better to the style.")
    reasoning: str = Field(..., description="A short explanation of the evaluation.")

async def evaluate_comparison_mistral(context: str, continuation1: str, continuation2: str, client: Mistral, semaphore: asyncio.Semaphore, model_name: str, retries: int = 3) -> EvalResponse:
    """
    Sends a single text to the Mistral API for evaluation.
    """
    async with semaphore:
        prompt = COMPARISON_PROMPT_TEMPLATE.format(context=context, continuation1=continuation1, continuation2=continuation2)
        current_attempt = 0
        while current_attempt < retries:
            try:
                print(f"Processing  context: \"{context[:50]}...\"")
                response = await client.chat.complete_async(
                    model= model_name,
                    messages = [
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=1200
                )
                print(f"Response: {response.choices[0].message.content[:50]}...{response.choices[0].message.content[-20:]}")
                result = response.choices[0].message.content
                if result.startswith("```json"):
                    json_content = result.removeprefix("```json\n").removesuffix("\n```").strip()
                elif result.startswith("```"):
                    json_content = result.removeprefix("```\n").removesuffix("\n```").strip()
                else:
                    json_content = result
                
                data = json.loads(json_content)
                return data

            except Exception as e:
                print(f"API error or unexpected issue for sentence: \"{context[:50]}...\". Attempt {current_attempt + 1}/{retries}. Error: {e}")
                current_attempt += 1
                if current_attempt == retries:
                    raise e
                await asyncio.sleep(2**current_attempt)

async def eval_generations(args, client, semaphore):
    with open(args.controlled_file, "r") as f:
        data_controlled = json.load(f)

    with open(args.uncontrolled_file, "r") as f:
        data_uncontrolled = json.load(f)

    output = []
    tasks = []
    for i in range(len(data_controlled)):
        item_controlled = data_controlled[i]
        item_uncontrolled = data_uncontrolled[i]
        context = item_controlled.get("context", "")
        completion_controlled = item_controlled.get("completion", "")
        completion_uncontrolled = item_uncontrolled.get("completion", "")

        if not context or not completion_controlled or not completion_uncontrolled:
            raise ValueError("Context and completions must not be empty.")
        
        tasks.append(evaluate_comparison_mistral(context, completion_controlled, completion_uncontrolled, client, semaphore, args.model_name))
        tasks.append(evaluate_comparison_mistral(context, completion_uncontrolled, completion_controlled, client, semaphore, args.model_name))

    results = await asyncio.gather(*tasks)
    for i in range(0, len(data_controlled)):
        result_index = i*2
        output_item = {
            "context": data_controlled[i].get("context", ""),
            "completion1": data_controlled[i].get("completion", ""),
            "completion2": data_uncontrolled[i].get("completion", ""),
            "reasoning":  "1. " + results[result_index].get("reasoning") + "\n\n2. " + results[result_index+1].get("reasoning")
        }

        if results[result_index].get("answer") == "A" and results[result_index+1].get("answer") == "B":
            output_item["evaluation"] = "win"
        elif results[result_index].get("answer") == "B" and results[result_index+1].get("answer") == "A":
            output_item["evaluation"] = "lose"
        else:
            output_item["evaluation"] = "tie"
        output.append(output_item)
    
    print(f"--- Completed Evaluation. Evaluated {len(output)} samples. ---")
    with open(args.output_json_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model-name", type=str,
                        help="Name of the model to use.")
    parser.add_argument("--controlled-file", type=str, required=True,
                        help="Path to the input file.")
    parser.add_argument("--uncontrolled-file", type=str, required=True,
                        help="Path to the uncontrolled input file.")
    parser.add_argument("--output-json-file", type=str,
                        help="Path to the output JSON file.")
    parser.add_argument("--max-concurrent-requests", type=int, default=10,
                        help="Maximum number of concurrent API requests.")
    
    parsed_args = parser.parse_args()
    client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
    semaphore = asyncio.Semaphore(parsed_args.max_concurrent_requests)
    
    asyncio.run(eval_generations(parsed_args, client, semaphore))
    