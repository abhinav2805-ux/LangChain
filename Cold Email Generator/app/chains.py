import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

# === Load API Key from File ===
api_key_path = r"C:\Users\gupta\groq_api_key.txt"

with open(api_key_path, "r") as file:
    api_key = file.read().strip()

# Optional: store it in env just in case other libs look for it
os.environ["GROQ_API_KEY"] = api_key

class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=api_key,
            model_name="llama3-70b-8192"
        )

    def extract_jobs(self, cleaned_text: str) -> list[dict]:
        """
        Extract job postings from scraped text.
        """
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}

            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format 
            with keys: `role`, `experience`, `skills`, and `description`.

            ### OUTPUT FORMAT (STRICT JSON ONLY, NO TEXT OUTSIDE):
            """
        )

        chain_extract = prompt_extract | self.llm

        try:
            res = chain_extract.invoke({"page_data": cleaned_text})
            json_parser = JsonOutputParser()
            parsed = json_parser.parse(res.content)
            return parsed if isinstance(parsed, list) else [parsed]
        except OutputParserException:
            raise OutputParserException("Unable to parse jobs. Possibly too much context or invalid JSON.")

    def write_mail(self, job: dict, links: list[dict]) -> str:
        """
        Generate a cold email for a given job description and portfolio links.
        """
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Mohan, a Business Development Executive at AtliQ.
            AtliQ is an AI & Software Consulting company focused on streamlining businesses
            using automation and tailored AI solutions.

            Write a cold email to a potential client describing how AtliQ can help with the job needs above.
            Include the most relevant links from this portfolio: {link_list}
            Do not add greetings or preambles.

            ### EMAIL:
            """
        )

        flat_links = [meta['links'] for meta in links if isinstance(meta, dict) and 'links' in meta]
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({
            "job_description": str(job),
            "link_list": "\n".join(flat_links)
        })

        return res.content
