from openai import OpenAI
import json
import os

class LLMClient:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.model_name = "gpt-4o-mini" # User requested model
        self.embedding_model = "text-embedding-3-small"

    def generate_fewshot_examples(self, retrieved_items):
        """
        検索されたエンティティを元に、Few-Shot（Q&A）形式の例を生成する
        """
        examples_str = ""
        for item in retrieved_items[:2]: # Use top 2 as examples
            examples_str += f"""
Example Q: Tell me about {item['name']}.
Example A: {item['name']} (URI: {item['uri']}) is a {', '.join(item['type'])}. It is located in {item['name'][:2]}... (Detailed attributes from ontology).
"""
        return examples_str

    def generate_response(self, prompt, context_str="", retrieved_items=None):
        """
        コンテキスト（および検索結果からのFew-Shot例）を用いて回答を生成する
        """
        few_shot_section = ""
        if retrieved_items:
            examples = self.generate_fewshot_examples(retrieved_items)
            few_shot_section = f"\nRefer to these similar cases for style:\n{examples}\n"

        system_instruction = "You are an expert architect assistant. Use the provided context to answer. If no context is provided, answer based on your general knowledge."
        
        user_content = f"""
Context from Ontology:
{context_str}

{few_shot_section}

Question: {prompt}
Answer:
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def get_embedding(self, text):
        """
        テキストの埋め込みベクトルを取得
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []

    def parse_building_info(self, text, options=None):
        """
        自然言語の記述から建築物の属性を抽出してJSONで返す
        """
        schema_hint = "Make sure to return valid JSON with these keys: '名称', '建築年', '高さ_m', '階数'. For object properties like '構造種別を持つ', return the closest matching string from the options provided."
        
        prompt = f"""
        Extract building information from the following text and format it as JSON.
        
        Text: "{text}"
        
        Requirements:
        1. Keys: "名称" (string), "建築年" (int), "高さ_m" (float), "階数" (int), "場所にある" (string), "構造種別を持つ" (string), "用途を持つ" (string), "耐震技術を持つ" (string).
        2. For numeric values, infer reasonable numbers if approximate (e.g., "about 100m" -> 100.0).
        3. For "場所にある", "構造種別を持つ", etc., try to match these provided options if possible:
           Options: {json.dumps(options, ensure_ascii=False) if options else "None"}
        4. If a field is missing, set it to null or empty string.
        5. Return ONLY the JSON string, no code blocks.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a data extraction assistant. Return purely JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"} # Force JSON mode
            )
            clean_text = response.choices[0].message.content.strip()
            return json.loads(clean_text)
        except Exception as e:
            print(f"Error parsing building info: {e}")
            return {}
