import os
import json
from dataclasses import dataclass
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


@dataclass
class CampaignData:
    id: int
    name: str
    type: str  
    description: str
    product_name: str = ""
    product_price: str = ""
    product_desc: str = ""
    role_title: str = ""
    experience_years: int = 0
    location: str = ""
    skills_required: str = ""
    relationship_goal: str = ""
    intro_context: str = ""
    target_industry: str = ""


EMBEDDING_MODEL = OllamaEmbeddings(model="qwen3-embedding:4b")

def get_valid_filename(name):
    return "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()

def save_prospect_json(prospect_data: dict, campaign_data: CampaignData):
    base_path = Path("contacts")
    folder_name = get_valid_filename(prospect_data['full_name'])
    prospect_folder = base_path / folder_name
    os.makedirs(prospect_folder, exist_ok=True)
    
    llm_context = {
        "prospect_info": {
            "name": prospect_data['full_name'],
            "profession": prospect_data.get('profession'),
            "location": prospect_data.get('region'),
            "interests": prospect_data.get('interests_hobbies'),
            "recent_post_analysis": prospect_data.get('previous_post_text')
        },
        "campaign_context": {
            "campaign_name": campaign_data.name,
            "type": campaign_data.type,
            "description": campaign_data.description,
            "why_approaching": "" 
        }
    }

  
    if campaign_data.type == "sales":
        llm_context["campaign_context"]["why_approaching"] = (
            f"Selling {campaign_data.product_name} ({campaign_data.product_price}). {campaign_data.product_desc}"
        )
    elif campaign_data.type == "hiring":
        llm_context["campaign_context"]["why_approaching"] = (
            f"Hiring {campaign_data.role_title}, {campaign_data.experience_years}y exp in {campaign_data.location}."
        )

 
    json_file_path = prospect_folder / "prospect.json"
    json_content_str = json.dumps(llm_context, indent=4, ensure_ascii=False)
    with open(json_file_path, 'w', encoding='utf-8') as f:
        f.write(json_content_str)

 
    try:
        vector_db_path = prospect_folder / "vector_db"
        doc = Document(
            page_content=json_content_str,
            metadata={"source": str(json_file_path), "prospect_name": prospect_data['full_name']}
        )
   
        Chroma.from_documents(
            documents=[doc],
            embedding=EMBEDDING_MODEL,
            persist_directory=str(vector_db_path)
        )
        print(f"✅ Success! Data saved for {prospect_data['full_name']}")
    except Exception as e:
        print(f"❌ Error: {e}")

    return str(json_file_path)


def generate_prospect_essay(json_path):
    """
    Reads the prospect JSON and uses Llama3:8b to write a 300-word essay.
    """
    # 1. Load the data from the JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        prospect_data = json.load(f)

    # 2. Initialize the Llama3 model
    llm = ChatOllama(
        model="llama3:8b",
        temperature=0.7,
    )

    # 3. Create the Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional biographer and recruitment strategist."),
        ("user", (
            "Write a 300-word essay about the following prospect based on their data and the campaign context. "
            "Highlight their professional background, interests, and why they are a great fit for the campaign. "
            "Data: {data}"
        ))
    ])

    # 4. Chain and Invoke
    print(f"\n✍️ Generating essay for {prospect_data['prospect_info']['name']}...")
    chain = prompt | llm
    response = chain.invoke({"data": json.dumps(prospect_data, indent=2)})

    return response.content

if __name__ == "__main__":
 
    sample_prospect = {
        "full_name": "Jane Doe",
        "profession": "Senior Software Engineer",
        "region": "San Francisco, CA",
        "interests_hobbies": "Hiking, Open Source, Photography",
        "previous_post_text": "Just finished a major migration to microservices!"
    }

    
    sample_campaign = CampaignData(
        id=101,
        name="Q1 Tech Talent Search",
        type="hiring",
        description="Looking for top-tier backend engineers.",
        role_title="Backend Lead",
        experience_years=8,
        location="Remote/SF",
        skills_required="Python, Kubernetes, Go"
    )

 
    print("🚀 Starting local embedding process...")
    save_prospect_json(sample_prospect, sample_campaign)

    json_path = save_prospect_json(sample_prospect, sample_campaign)

    # 2. Run the LLM to generate the essay
    try:
        essay = generate_prospect_essay(json_path)
        
        print("\n--- GENERATED ESSAY ---")
        print(essay)
        
        # Optional: Save the essay to a file
        essay_path = Path(json_path).parent / "prospect_essay.txt"
        with open(essay_path, "w", encoding="utf-8") as f:
            f.write(essay)
        print(f"\n💾 Essay saved to: {essay_path}")

    except Exception as e:
        print(f"❌ Error during LLM generation: {e}")