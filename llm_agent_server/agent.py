import asyncio
import subprocess
from typing import List, Dict
from google.generativeai.types import HarmCategory, HarmBlockThreshold

class LLMCommandAgent:
    def __init__(self, llm):
        self.llm = llm

    async def execute_command(self, commands: List[str]) -> str:
        """Executes multiple commands in the shell sequentially and returns their combined output."""
        print("commands got: ", commands)
        results = []
        for command in commands:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if stderr:
                results.append(f"Error: {stderr.decode()}")
            else:
                results.append(stdout.decode())
        
        return "\n".join(results)

    async def handle_request(self, request: str) -> Dict[str, str]:
        """Processes a request by calling the LLM and executing commands."""

        # Construct the prompt with clear instructions for the LLM
        prompt = f"""
        You are a command-line expert tasked with providing simple and effective CLI commands based on user requests.

        User Request:
        "{request}"

        Guidelines:
        - Begin by carefully interpreting the user's request to grasp the main objective.
        - Deconstruct the task into straightforward steps that can be easily executed.
        - For each step, craft a corresponding CLI command that directly addresses that part of the task.
        - Present your commands in a clean format, with each command on a new line.
        - Ensure the sequence of commands is logical and follows the workflow needed to complete the task efficiently.
        - Keep your commands simple; avoid using complex constructs or loops.

        Example Output:
        ```
        cp source_dir/*.txt destination_dir/
        grep -l "example" destination_dir/*.txt > files_with_example.txt
        grep -L "example" destination_dir/*.txt | xargs rm -f
        ```
        """
        
        # set HARM_CATEGORY_DANGEROUS_CONTENT blockage to high and above since we got MEDIUM for lots of prompts
        response = self.llm.generate_content(
            prompt, 
            safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH
            }
        )
        if response.text is None:
            return {"output": "Cannot execute the command for safety reasons imposed by the LLM Operator."}
        
        # Remove any triple backticks and split by lines
        commands = response.text.replace("```", "").strip().split('\n')
        result = await self.execute_command(commands)
        return {"output": result}
