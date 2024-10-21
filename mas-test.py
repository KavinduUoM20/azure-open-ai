import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env
load_dotenv()
azure_oai_endpoint = os.getenv("AZURE_OAI_ENDPOINT")
azure_oai_key = os.getenv("AZURE_OAI_KEY")
azure_oai_deployment = os.getenv("AZURE_OAI_DEPLOYMENT")
azure_oai_api_version = os.getenv("API_VERSION")

# Initialize the Azure ChatOpenAI model
llm = AzureChatOpenAI(
    api_key=azure_oai_key,
    api_version=azure_oai_api_version,
    azure_endpoint=azure_oai_endpoint,
    model=azure_oai_deployment
)

# Define the prompt template
base_prompt = '''Act as an agent working on matching order descriptions on Apparel manufacturing.
Your task is to match the descriptions from two lists.
Consider term PO is used to identify Purchase order and there would be many instances that descriptions do not match 100% but their meaning is similar.
Based on the given information, provide matching columns to list A.

list A:
{column_names}

Here are the column names from the data frame:
{df_head}

[Guidelines]
- Specifically show me List A column name and matching column from List B only.
- If there is no matching column in List B, keep it as blank.
- Provide a similarity score from 0 to 100 for each matching column, based on similarity.
- Consider List A column name Style is usually named as Collection.
- Make sure to only return the processed result as JSON
- Do not return code.
[Guidelines Ends]

Return output in the following JSON format:
result: [[list_a_column_name: List A column name goes here,  list_b_column_name: List B column name goes here, similarity_score: similarity score goes here],
[list_a_column_name: List A column name goes here,  list_b_column_name: List B column name goes here, similarity_score: similarity score goes here], ]
'''

# Create a PromptTemplate
prompt = PromptTemplate(
    input_variables=["column_names", "df_head"],
    template=base_prompt
)

# Output parser (this assumes you're outputting simple strings)
output_parser = StrOutputParser()

# Define the inputs
list_a = """PO Number, Style, Quantity, Price"""
df_string = """Purchase Order Number, Collection, Qty, Unit Price"""

# Construct the inputs for the chain
inputs = {
    "column_names": list_a,  # String representation of List A columns
    "df_head": df_string     # String representation of DataFrame columns
}

# Call the LLM with the formatted prompt
formatted_prompt = prompt.format(column_names=inputs["column_names"], df_head=inputs["df_head"])
result = llm(formatted_prompt)  # This sends the formatted prompt to the Azure OpenAI model

# Parse the result with the output parser
parsed_result = output_parser.parse(result)

# Print the final formatted output
print(parsed_result)
