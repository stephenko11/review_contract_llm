import autogen
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from typing_extensions import Annotated
import dotenv
import os
import utilies

# Save the current (original) working directory
original_path = os.getcwd()
# Change the working directory to the current folder of the script
current_folder = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_folder)
print("Current working directory:", os.getcwd())


# Vector Database parameters (Using PGvetor, Postgresql)
postgres_username = 'postgres'
database_name = 'llm_vector_1'
collection_name = 'autogen_contract'
llm_model = 'gpt-4o-mini-2024-07-18'

# Instructions 
task = 'agreement'
acting_party = 'Manufacturer'
agreement_to_review_file_name = "agreement.docx"

agreement = utilies.read_docx(os.path.join(os.path.abspath(""), "documents", "agreement", f'{agreement_to_review_file_name}'))

playbook = """
Background information and playbook:
<Basic Transaction Mode>

<Financial Term>

<Delivery>

<Restriction>

<Governing Law & Dispute Resolution>

"""

PROBLEM = f"""Review the {task} as an attorney. Since we are acting for the {acting_party}, the clause should be in the favor of the 
{acting_party} whenever it is possible and reasonable. 
Add any new clause to the agreement if necesary in accordinance to the playbook if it does not mentioned in the agreement. 
Here is the playbook: {playbook}. Here is the {task}: {agreement} """



#### Codes for review

dotenv.load_dotenv()
llm_api_key = os.environ.get('OPENAI_API_KEY')
postgres_password = os.environ.get('POSTGRES_KEY')

llm_config = {
    "cache_seed": 42,  # change the cache_seed for different trials
    "temperature": 0,
    #"config_list": config_list, # this is OpenAi Config
    "timeout": 120,  # in seconds,
    "model": f"{llm_model}", 
    "api_key": llm_api_key, 
}


user_proxy = autogen.UserProxyAgent(
    name="admin",
    system_message=f"""A human admin. Interact with the planner to discuss the plan. 
    Plan execution needs to be approved by this admin.""",
    code_execution_config={
        "work_dir": "code",
        "use_docker": False
    },
    human_input_mode="TERMINATE",
)

planner = autogen.AssistantAgent(
    name="planner",
    system_message=f"""Planner. Suggest a plan. Revise the plan based on feedback from admin, until admin approval.
    The {task} should first review by the attorney_chatbot. Then involve a playbook analyst who analyze the playbook to look for parts that need to be amended and updated on the
    {task} and check if the amendment has already include all the changes as instructed in the playbook. 
    The business consultant should review the {task} to see clauses are favorable to the {acting_party} from business perspective based 
    on the business contemplated on the {task}.
    Be clear which step is performed by playbook analyst, and which step is performed by the attorney_chatbot and 
    the business consultant. say TERMINATE once all the step of the plan is completed. 
    
    """,
    llm_config=llm_config,
)

ragproxyagent = RetrieveUserProxyAgent(
    name="rag_retrieval_agent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "code",
        "docs_path": [
            os.path.join(os.path.abspath(""), "documents", f"{agreement_to_review_file_name}"),
        ],
        "vector_db": "pgvector",
        "collection_name": collection_name,
        "db_config": {
            "connection_string": f"postgresql://{postgres_username}:{postgres_password}@localhost:5432/{database_name}", 
        },
        "custom_text_types": ["docx"],
        "chunk_token_size": 1000,
        "overwrite": True, 
        "model": f"{llm_model}",
        "get_or_create": True,
    },
    code_execution_config=False,
)


attorney = autogen.AssistantAgent(
    name="attorney_chatbot",
    llm_config=llm_config,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    system_message=f"""You are a professional lawyer, known for your insightful, drafting and reviewing qaulity agreement.
    You follow an approved plan.
    You are acting on behalf of the {acting_party}. You aim to enhance the {task} to minimize dispute and enhance clarity. 
    You should also follow the playbook.
    You should improve the quality of comment on the {task} based on the feedback from the Business Consultant and 
    playbook_analyst. You should provide your proposed amendment to the clause and providing the proposed chagnes to the revised clauses.
    Highlight the changes you made for easy reference. 
    """
)


playbook_analyst = autogen.AssistantAgent(
    name="playbook_analyst",
    llm_config=llm_config,
    system_message=f"""Playbook analyst. You follow an approved plan. Analyze the playbook and extract  
    the parts that need to be amended and updated on the {task}. Ensure the playbook instruction is followed. 
    Only do extraction.""",
)


business_consultant = autogen.AssistantAgent(
    name="business_consultant",
    system_message=f"""You are a critic and a business consultant, 
    known for your expert in the type of transaction contemplated in the {task} and the background information.
    You follow an approved plan.
    Your task is to scrutinize content for any harmful elements, regulatory violations, clarity of the contract terms, ensuring
    all materials are aligned and the {acting_party} is protected. Your aim is to minimize dispute and 
    ensure the party you represented is able to execute the business transaction without undue risk. You should ensure the agreement 
    follows the information from the playbook. Make sure the {acting_party} obtains all the IP rights. Make sure you follow 
    the background informaton and the playbook and include all the terms it provides and reflect them in the agreement. """,
    llm_config=llm_config,
)

### Running Mode

from typing import Dict, List
from autogen import Agent

def custom_speaker_selection_func(last_speaker: Agent, groupchat: autogen.GroupChat):
    """Define a customized speaker selection function.
    A recommended way is to define a transition for each speaker in the groupchat.

    Returns:
        Return an `Agent` class or a string from ['auto', 'manual', 'random', 'round_robin'] to select a default method to use.
    """
    messages = groupchat.messages

    if len(messages) <= 1:
        # first, let the engineer retrieve relevant data
        return planner
    
    else:
        # default to auto speaker selection method
        return "auto"


def _reset_agents():
    user_proxy.reset()
    planner.reset()
    ragproxyagent.reset()
    attorney.reset()
    playbook_analyst.reset()
    business_consultant.reset()



def rag_chat():
    _reset_agents()
    groupchat = autogen.GroupChat(
        agents=[planner, ragproxyagent, attorney, playbook_analyst, business_consultant], messages=[], 
        max_round=12, speaker_selection_method="round_robin"
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # Start chatting with boss_aid as this is the user proxy agent.
    ragproxyagent.initiate_chat(
        manager,
        message=ragproxyagent.message_generator,
        problem=PROBLEM,
        n_results=3,
    )


def norag_chat():
    _reset_agents()
    groupchat = autogen.GroupChat(
        agents=[planner, attorney, playbook_analyst, business_consultant],
        messages=[],
        max_round=12,
        speaker_selection_method=custom_speaker_selection_func,
        allow_repeat_speaker=False,
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # Start chatting with the boss as this is the user proxy agent.
    user_proxy.initiate_chat(
        manager,
        message=PROBLEM,
    )


def call_rag_chat():
    _reset_agents()

    # In this case, we will have multiple user proxy agents and we don't initiate the chat
    # with RAG user proxy agent.
    # In order to use RAG user proxy agent, we need to wrap RAG agents in a function and call
    # it from other agents.
    def retrieve_content(
        message: Annotated[
            str,
            "Refined message which keeps the original meaning and can be used to retrieve content for code generation and question answering.",
        ],
        n_results: Annotated[int, "number of results"] = 20,
    ) -> str:
        ragproxyagent.n_results = n_results  # Set the number of results to be retrieved.
        _context = {"problem": message, "n_results": n_results}
        ret_msg = ragproxyagent.message_generator(ragproxyagent, None, _context)
        return ret_msg or message

    ragproxyagent.human_input_mode = "NEVER"  # Disable human input for boss_aid since it only retrieves content.

    for caller in [planner, attorney, playbook_analyst, business_consultant]:
        d_retrieve_content = caller.register_for_llm(
            description="retrieve content for code generation and question answering.", api_style="function"
        )(retrieve_content)

    for executor in [planner, attorney, playbook_analyst, business_consultant]:
        executor.register_for_execution()(d_retrieve_content)

    groupchat = autogen.GroupChat(
        agents=[user_proxy, planner, attorney, playbook_analyst, business_consultant],
        messages=[],
        max_round=30,
        speaker_selection_method=custom_speaker_selection_func,
        allow_repeat_speaker=False,
    )

    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # Start chatting with the boss as this is the user proxy agent.
    user_proxy.initiate_chat(
        manager,
        message=PROBLEM,
    )




norag_chat()

