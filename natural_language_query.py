import psycopg2
import json
from openai import OpenAI

class NaturalLanguageQuery:
    def __init__(self, client: OpenAI, model: str, db_config: dict):
        self.client = client
        self.model = model
        self.db_config = db_config

        self.tools_definition = [
            {
                "type": "function",
                "function": {
                    "name": "execute_sql_query",
                    "description": "Executes an SQL query on a PostgreSQL database.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sql_query": {
                                "type": "string",
                                "description": "The SQL query you wish to execute."
                            }
                        },
                        "required": ["sql_query"]
                    }
                }
            }
        ]

        self.tools_map = {
            "execute_sql_query": self._execute_sql_query
        }

        self.forbidden_commands = [
            "DROP",
            "DELETE",
            "UPDATE",
            "INSERT",
            "CREATE",
            "ALTER",
            "TRUNCATE",
            "GRANT",
            "REVOKE",
            "COMMIT",
            "ROLLBACK"
        ]

        self.messages = []

        self._initialize_ai()

    def _initialize_ai(self):
        system_prompt = {
            "role": "system",
            "content": """You are an assistant who answers questions about a postgres database that you can access.
                    You can execute any SELECT query using execute_sql_query tool.
                    If you don't have enough information you can query schema information and sample data from tables.
                    Don't speculate and don't assume anything about the schema, read actual schema before referencing any tables or columns.
                    Don't ask for clarification from the user, this is not an interactive chat, explore database until you can provide a human readable answer.
                        """
        }
        
        self.messages.append(system_prompt)

        schema = self._execute_sql_query("SELECT table_schema, table_name FROM information_schema.tables")
        self.messages.append({
            "role": "user",
            "content": schema
        })

    def _execute_sql_query(self, sql_query):
        """
        Executes an arbitrary SQL query on a PostgreSQL database.
        
        Args:
            sql_query (str): The SQL query to execute.
        
        Returns:
            str: A JSON string representing the results or an error message.
        """
        print(f"Executing query: {sql_query}")
       
        try:
            for cmd in self.forbidden_commands:
                if cmd in sql_query.upper():
                    return json.dumps({"error": "You are not allowed to execute this command."})

            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            cur.execute(sql_query)
            
            # If the query returns rows, fetch them
            if cur.description:
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                results = [dict(zip(columns, row)) for row in rows]
            else:
                results = {"message": "Query executed successfully."}
            conn.commit()
            return json.dumps(results)
        except Exception as e:
            return json.dumps({"error": str(e)})
        finally:
            if 'cur' in locals():
                cur.close()
            if 'conn' in locals():
                conn.close()

    def run_query(self, query: str):
        
        self.messages.append({
            "role": "user",
            "content": query
        })

        # First iteration: Force tool usage to start database exploration
        force_tool = True

        while True:
            # Use different tool_choice strategy based on iteration
            tool_choice_param = {"type": "function", "function": {"name": "execute_sql_query"}} if force_tool else "auto"
            
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=self.messages,
                tools=self.tools_definition,
                tool_choice=tool_choice_param,
            )

            message = response.choices[0].message

            if message.content:
                print(message.content)
                
            self.messages.append(message)

            # After first iteration, let the model decide whether to use tools
            force_tool = False

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    # Get the tool function name and arguments Grok wants to call
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    # Call one of the tool function defined earlier with arguments
                    result = self.tools_map[function_name](**function_args)

                    # Append the result from tool function call to the chat message history,
                    # with "role": "tool"
                    self.messages.append(
                        {
                            "role": "tool",
                            "content": json.dumps(result),
                            "tool_call_id": tool_call.id  # tool_call.id supplied in Grok's response
                        }
                    )
            else:
                # If no tool calls, we have our final answer
                return message.content
