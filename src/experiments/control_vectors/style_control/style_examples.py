import pandas as pd
import os
import asyncio
import json
import argparse
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

PROMPT_TEMPLATE = """Transform the style of the following text snippet to be extremely positive, cheerful, and enthusiastic. The goal is an 'overly happy' version.

Key requirements:

Preserve Core Content: The fundamental message, topic, key information, and facts from the original snippet must be accurately retained.
Amplify Positivity: Inject exuberant language, positive adjectives/adverbs, exclamation points (where appropriate), and a generally effusive tone.
Focus on Style: The changes should be purely stylistic; do not add new information or alter the original meaning.
Output format: The output format should be JSON with the following structure:
```json
{{
  "original": "Original text snippet here",
  "transformed": "Transformed text snippet here"
}}
```json

Here is the origi nal text snippet:
{original_text}
"""

async def process_sentence(sentence: str, client: Mistral, semaphore: asyncio.Semaphore, model_name: str, retries: int = 3) -> dict | None:
    """
    Sends a single sentence to the Mistral API for transformation.
    """
    async with semaphore:
        prompt = PROMPT_TEMPLATE.format(original_text=sentence)
        current_attempt = 0
        while current_attempt < retries:
            try:
                print(f"Processing sentence: \"{sentence[:50]}...\"")
                chat_response = await client.chat.complete_async(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    response_format={"type": "json_object"}
                )
                
                received_content = chat_response.choices[0].message.content

                if received_content.startswith("```json"):
                    json_content = received_content.removeprefix("```json\n").removesuffix("\n```").strip()
                elif received_content.startswith("```"):
                    json_content = received_content.removeprefix("```\n").removesuffix("\n```").strip()
                else:
                    json_content = received_content
                
                data = json.loads(json_content)
                
                if "original" in data and "transformed" in data:
                    return {
                        "original": data.get("original", sentence),
                        "transformed": data["transformed"]
                    }
                else:
                    print(f"Error: 'original' or 'transformed' key missing in response for: \"{sentence[:50]}...\"")
                    return None 

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for sentence: \"{sentence[:50]}...\". Attempt {current_attempt + 1}/{retries}. Error: {e}")
                current_attempt += 1
                if current_attempt == retries:
                    return None
                await asyncio.sleep(2**current_attempt)

            except Exception as e:
                print(f"API error or unexpected issue for sentence: \"{sentence[:50]}...\". Attempt {current_attempt + 1}/{retries}. Error: {e}")
                current_attempt += 1
                if current_attempt == retries:
                    return None
                await asyncio.sleep(2**current_attempt)
        return None

async def main(args):
    api_key = args.api_key if args.api_key else os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("Error: MISTRAL_API_KEY not provided via --api-key argument or MISTRAL_API_KEY environment variable.")
        return

    try:
        print(f"Loading sentences from {args.input_parquet_file}...")
        sentences_to_process = pd.read_parquet(args.input_parquet_file)["text"].to_list()[:args.num_examples]
        print(f"Loaded {len(sentences_to_process)} sentences to process (up to {args.num_examples} requested).")
    except Exception as e:
        print(f"Error loading Parquet file '{args.input_parquet_file}': {e}")
        return

    if not sentences_to_process:
        print("No sentences to process.")
        return

    client = Mistral(api_key=api_key)
    semaphore = asyncio.Semaphore(args.max_concurrent_requests)
    
    all_results = []
    tasks = []

    print("Creating tasks for sentence transformation...")
    for sentence_text in sentences_to_process:
        tasks.append(process_sentence(sentence_text, client, semaphore, args.model_name))

    print(f"Processing {len(tasks)} tasks concurrently (max {args.max_concurrent_requests} at a time using model {args.model_name})...")
    processed_results = await asyncio.gather(*tasks)

    for result in processed_results:
        if result:
            all_results.append(result)
    
    print(f"Successfully processed {len(all_results)} out of {len(sentences_to_process)} sentences.")

    try:
        with open(args.output_json_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        print(f"Results successfully written to {args.output_json_file}")
    except IOError as e:
        print(f"Error writing JSON file '{args.output_json_file}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Asynchronously transform sentences using Mistral API.")
    
    parser.add_argument("--api-key", type=str, default=None,
                        help="Mistral API Key. If not provided, uses MISTRAL_API_KEY environment variable.")
    parser.add_argument("--model-name", type=str, default="mistral-medium-latest",
                        help="Name of the Mistral model to use.")
    parser.add_argument("--input-parquet-file", type=str, required=True,
                        help="Path to the input Parquet file.")
    parser.add_argument("--output-json-file", type=str, default="transformed_sentences.json",
                        help="Path to the output JSON file.")
    parser.add_argument("--num-examples", type=int, default=100,
                        help="Number of sentences to process from the Parquet file.")
    parser.add_argument("--max-concurrent-requests", type=int, default=10,
                        help="Maximum number of concurrent API requests.")
    
    parsed_args = parser.parse_args()
    
    asyncio.run(main(parsed_args))
