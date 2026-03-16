import os
import json
import time
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.prompts import PromptTemplate

os.environ["GOOGLE_API_KEY"] = "GOOGLE_API_KEY"

@dataclass
class AnswerRecord:
    question_text: str
    value: str
    references: List[Dict[str, Any]]


@dataclass
class SubmissionPackage:
    team_email: str
    submission_name: str
    answers: List[Dict[str, Any]]


class ConfigManager:
    REQUIRED_ENV_VARS = ["GOOGLE_API_KEY"]

    def __init__(self):
        self._validate_environment()
        self.google_api_key = os.environ.get("GOOGLE_API_KEY")
        self.submission_name = "Dryagalova_v1"
        self.email = "st119022@student.spbu.ru"
        self.output_filename = f"submission_{self.submission_name}.json"
        self.index_directory = "faiss_index"
        self.data_directory = "/content"
        self.questions_file = "questions.json"

    def _validate_environment(self):
        missing_vars = [var for var in self.REQUIRED_ENV_VARS if
                        not os.environ.get(var)]
        if missing_vars:
            raise EnvironmentError(
                f"Missing environment variables: {', '.join(missing_vars)}")


class DocumentProcessor:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def extract_pdf_content(self) -> List[Any]:
        pdf_files = list(Path(self.data_dir).glob("*.pdf"))
        
        if not pdf_files:
            print(f"Warning: No PDF files found in {self.data_dir}")
            return []
            
        documents = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_pdf = {
                executor.submit(self._load_single_pdf, pdf_path): pdf_path
                for pdf_path in pdf_files
            }

            for future in as_completed(future_to_pdf):
                pdf_path = future_to_pdf[future]
                try:
                    pages = future.result()
                    documents.extend(pages)
                    print(f"Loaded: {pdf_path.name} - {len(pages)} pages")
                except Exception as e:
                    print(f"Failed to load {pdf_path.name}: {e}")

        return documents

    def _load_single_pdf(self, pdf_path: Path) -> List[Any]:
        loader = PyMuPDFLoader(str(pdf_path))
        pages = loader.load()

        file_hash = self._calculate_file_hash(pdf_path)
        for idx, page in enumerate(pages):
            page.metadata["source_file"] = pdf_path.name
            page.metadata["file_hash"] = file_hash
            page.metadata["page_number"] = idx + 1

        return pages

    def _calculate_file_hash(self, file_path: Path) -> str:
        sha1 = hashlib.sha1()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha1.update(chunk)
        return sha1.hexdigest()

    @staticmethod
    def chunk_documents(documents: List[Any], chunk_size: int = 2000,
                        overlap: int = 50) -> List[Any]:
        if not documents:
            print("Warning: No documents to chunk")
            return []
            
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks


class VectorIndexManager:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embeddings = SentenceTransformerEmbeddings(
            model_name=embedding_model)

    def create_index(self, documents: List[Any], save_path: str) -> Optional[FAISS]:
        if not documents:
            print("Error: Cannot create index - no documents provided")
            return None
            
        non_empty_docs = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
        if not non_empty_docs:
            print("Error: All documents are empty")
            return None
            
        print(f"Creating index with {len(non_empty_docs)} non-empty documents")
        print("This may take several minutes for large document sets...")
        
        try:
            batch_size = 1000
            total_batches = (len(non_empty_docs) + batch_size - 1) // batch_size
            
            print(f"Processing in {total_batches} batches of {batch_size} documents...")
            
            first_batch = non_empty_docs[:batch_size]
            vectorstore = FAISS.from_documents(first_batch, self.embeddings)
            print(f"Batch 1/{total_batches} completed")
            
            for i in range(batch_size, len(non_empty_docs), batch_size):
                batch = non_empty_docs[i:i+batch_size]
                vectorstore.add_documents(batch)
                batch_num = i//batch_size + 1
                print(f"Batch {batch_num}/{total_batches} completed")
            
            vectorstore.save_local(save_path)
            print(f"Index saved to {save_path}")
            return vectorstore
            
        except Exception as e:
            print(f"Error creating index: {e}")
            return None

    def load_index(self, index_path: str) -> Optional[FAISS]:
        try:
            return FAISS.load_local(index_path, self.embeddings,
                                    allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Error loading index: {e}")
            return None

    def get_retriever(self, vectorstore: FAISS, k_documents: int = 5): 
        if not vectorstore:
            return None
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k_documents}
        )


class AnswerCleaner:
    @staticmethod
    def process(raw_answer: Optional[str]) -> str:
        if not raw_answer:
            return "N/A"

        answer = str(raw_answer).strip()
        answer = AnswerCleaner._remove_prefixes(answer)
        answer = AnswerCleaner._normalize_boolean(answer)
        answer = AnswerCleaner._extract_number(answer)
        answer = AnswerCleaner._truncate_long_text(answer)

        return answer

    @staticmethod
    def _remove_prefixes(text: str) -> str:
        patterns = [
            r'^(Answer|Output|Result):\s*',
            r'^(The answer is|The result is|Value:)\s*'
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return text.strip()

    @staticmethod
    def _normalize_boolean(text: str) -> str:
        text_lower = text.lower()
        if text_lower in ["true", "yes", "correct", "1"]:
            return "True"
        if text_lower in ["false", "no", "incorrect", "0"]:
            return "False"
        if text_lower in ["n/a", "no data", "not found", "missing"]:
            return "N/A"
        return text

    @staticmethod
    def _extract_number(text: str) -> str:
        text = re.sub(r'[$€£]', '', text)

        number_match = re.search(r"(-?[\d,]+(\.\d+)?)", text)

        if number_match:
            clean_number = number_match.group(1).replace(',', '')
            try:
                float(clean_number)
                return clean_number
            except ValueError:
                pass

        return text

    @staticmethod
    def _truncate_long_text(text: str, max_words: int = 5) -> str:
        if len(text.split()) > max_words and text not in ["True", "False",
                                                          "N/A"]:
            return "N/A"
        return text


class AnswerExtractor:
    def __init__(self, retriever, llm_model: str = "models/gemini-1.5-flash"):
        if not retriever:
            raise ValueError("Retriever cannot be None")
            
        self.llm = ChatGoogleGenerativeAI(
            model=llm_model, 
            temperature=0.0,
            max_output_tokens=50
        )
        self.retriever = retriever
        self.qa_chain = self._setup_qa_chain()

    def _setup_qa_chain(self) -> RetrievalQA:
        template = """You are a financial auditor. Extract the exact value from the context.

Context: {context}
Question: {question}

Rules:
- Return only the number, True, False, or N/A
- No explanations
- Check units (thousands/millions) and multiply if needed

Answer:"""

        prompt = PromptTemplate.from_template(template)

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    def get_answer(self, question: str) -> Tuple[str, List[Any]]:
        try:
            result = self.qa_chain.invoke({"query": question})
            return result["result"], result["source_documents"]
        except Exception as e:
            print(f"Error getting answer: {e}")
            return "Error", []


class ResultManager:
    def __init__(self, output_file: str):
        self.output_file = output_file

    def load_existing(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.output_file):
            return []

        try:
            with open(self.output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("answers", [])
        except Exception as e:
            print(f"Error loading progress: {e}")
            return []

    def save(self, answers: List[Dict[str, Any]], email: str,
             submission_name: str):
        submission = SubmissionPackage(
            team_email=email,
            submission_name=submission_name,
            answers=answers
        )

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(asdict(submission), f, ensure_ascii=False, indent=2)


class QuestionLoader:
    def __init__(self, questions_file: str):
        self.questions_file = questions_file

    def load(self) -> List[Dict[str, str]]:
        try:
            with open(self.questions_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                questions = data if isinstance(data, list) else data.get(
                    "questions", [])

            normalized = []
            for q in questions:
                if isinstance(q, dict):
                    normalized.append(q)
                else:
                    normalized.append({"text": str(q)})

            return normalized
        except Exception as e:
            raise RuntimeError(f"Failed to load questions: {e}")


class Application:
    def __init__(self):
        self.config = ConfigManager()
        self.result_manager = ResultManager(self.config.output_filename)
        self.answer_cleaner = AnswerCleaner()

    def run(self):
        start_time = time.time()
        
        doc_processor = DocumentProcessor(self.config.data_directory)
        index_manager = VectorIndexManager()

        if os.path.exists(self.config.index_directory):
            print("Loading existing vector index...")
            vectorstore = index_manager.load_index(self.config.index_directory)
            if not vectorstore:
                print("Failed to load existing index, will create new one")
                vectorstore = None
        else:
            vectorstore = None

        if not vectorstore:
            print("Creating new vector index...")
            documents = doc_processor.extract_pdf_content()
            
            if not documents:
                print("Error: No documents found. Please check the content directory.")
                return
                
            chunks = doc_processor.chunk_documents(documents)
            
            if not chunks:
                print("Error: No chunks created from documents.")
                return
            
            max_chunks = 20000
            if len(chunks) > max_chunks:
                print(f"Too many chunks ({len(chunks)}). Limiting to {max_chunks}...")
                chunks = chunks[::2][:max_chunks]
                print(f"Reduced to {len(chunks)} chunks")
                
            vectorstore = index_manager.create_index(chunks,
                                                     self.config.index_directory)
            if not vectorstore:
                print("Error: Failed to create vector index.")
                return
                
            print(f"Created index with {len(chunks)} chunks")

        retriever = index_manager.get_retriever(vectorstore, k_documents=5)
        if not retriever:
            print("Error: Failed to create retriever")
            return

        try:
            extractor = AnswerExtractor(retriever)
        except Exception as e:
            print(f"Error creating answer extractor: {e}")
            return

        question_loader = QuestionLoader(self.config.questions_file)
        questions = question_loader.load()
        
        if not questions:
            print("Error: No questions loaded")
            return

        existing_answers = self.result_manager.load_existing()
        start_index = len(existing_answers)

        print(f"\nTotal questions: {len(questions)}")
        print(f"Already answered: {start_index}")
        print(f"Questions to process: {len(questions) - start_index}")

        if start_index >= len(questions):
            print("All questions already answered.")
            return

        answers = existing_answers.copy()
        processed = 0
        failed = 0

        for idx in range(start_index, len(questions)):
            question = questions[idx]
            question_text = question["text"]

            print(
                f"\n[{idx + 1}/{len(questions)}] Processing: {question_text[:80]}...")

            try:
                raw_answer, source_docs = extractor.get_answer(question_text)
                cleaned_answer = self.answer_cleaner.process(raw_answer)

                references = self._extract_references(source_docs, max_refs=1)

                print(f"Answer: {cleaned_answer}")
                processed += 1

                answer_record = AnswerRecord(
                    question_text=question_text,
                    value=cleaned_answer,
                    references=references
                )
                answers.append(asdict(answer_record))
                
                if (idx + 1) % 10 == 0 or idx == len(questions) - 1:
                    self.result_manager.save(answers, self.config.email,
                                             self.config.submission_name)
                    elapsed = time.time() - start_time
                    print(f"Progress saved. Elapsed time: {elapsed/60:.1f} minutes")

            except Exception as e:
                failed += 1
                if not self._handle_error(e, answers, question_text, idx):
                    break

        elapsed = time.time() - start_time
        print(f"\nCompleted! Processed: {processed}, Failed: {failed}")
        print(f"Total time: {elapsed/60:.1f} minutes")

    def _extract_references(self, source_docs: List[Any], max_refs: int = 1) -> List[Dict[str, Any]]:
        seen = set()
        references = []

        for doc in source_docs:
            source = doc.metadata.get("source_file", "")
            page = doc.metadata.get("page_number", 1)
            key = (source, page)

            if key in seen:
                continue

            seen.add(key)
            references.append({
                "pdf_sha1": doc.metadata.get("file_hash", source),
                "page_index": page
            })

            if len(references) >= max_refs:
                break

        return references

    def _handle_error(self, error: Exception, answers: List,
                      question_text: str, idx: int) -> bool:
        error_msg = str(error).lower()

        if any(code in error_msg for code in
               ["429", "quota", "resource_exhausted"]):
            print("\nAPI quota exhausted. Stopping.")
            print(f"Error: {error}")
            return False

        print(f"Error processing question {idx + 1}: {error}")

        error_answer = AnswerRecord(
            question_text=question_text,
            value="error",
            references=[]
        )
        answers.append(asdict(error_answer))
        time.sleep(1)
        return True


def main():
    try:
        print("Starting optimized version with reduced chunk count...")
        print("Make sure you're using GPU in Colab (Runtime -> Change runtime type -> GPU)")
        app = Application()
        app.run()
    except Exception as e:
        print(f"Application failed: {e}")
        raise


if __name__ == "__main__":
    main()
