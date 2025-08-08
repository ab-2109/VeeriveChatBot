import os

import dotenv
from langchain_community.graphs import Neo4jGraph


# Load environment variables from .env file
dotenv.load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)
# get graph schema and store it in a file called schema.txt
graph.refresh_schema()         # optional but good to refresh
schema = graph.schema          # <-- property, no parentheses
print(schema)
with open("schema.txt", "w") as f:
    f.write(str(schema))
print("Schema saved to schema.txt")