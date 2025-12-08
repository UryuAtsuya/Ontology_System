import google.generativeai as genai
import json
import os

class LLMClient:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model_name = 'gemini-2.5-flash' # Updated to a known valid model name just in case, or keep 'gemini-flash-latest'
        self.model = genai.GenerativeModel(self.model_name)

    def generate_response(self, prompt, context, system_instruction=None):
        """
        Geminiを使用したテキスト生成（JSON制約なし）
        """
        system_prompt = """
        あなたは建築の専門家であるAIアシスタントです。
        提供されたコンテキスト（オントロジー情報）を使用して、ユーザーの質問に答えてください。

        ルール：
        1. 自然な日本語で回答してください。
        2. オントロジーに含まれる情報（クラス、個体、プロパティ）に基づいて回答し、事実に基づかない捏造は避けてください。
        3. 専門用語や固有名詞は正確に使用してください。
        4. コンテキストに情報がない場合は、正直に「情報がありません」と答えてください。
        """

        if system_instruction:
            # カスタムシステムプロンプトが提供された場合、それを使用
            final_system_prompt = system_instruction
        else:
            # デフォルトのシステムプロンプト
            final_system_prompt = system_prompt

        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=final_system_prompt,
            generation_config={"temperature": 0.2}
        )
        
        try:
            response = model.generate_content(f"Context:\n{context}\n\nQuestion: {prompt}")
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def get_embedding(self, text):
        """
        テキストの埋め込みベクトルを取得
        """
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document",
                title="Ontology Entity"
            )
            return result['embedding']
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
            response = self.model.generate_content(prompt)
            clean_text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except Exception as e:
            print(f"Error parsing building info: {e}")
            return {}
