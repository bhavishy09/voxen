import requests
import json
import os
from typing import List

# Try to import LangChain components with better error handling
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.prompts import ChatPromptTemplate
    from langchain_groq import ChatGroq
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain.schema import Document
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    LANGCHAIN_AVAILABLE = True
    print(" LangChain components imported successfully")
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    print(f" LangChain not available: {e}")

class LLMService:
    def __init__(self):
        # Using free Hugging Face API as fallback
        self.api_url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
        self.headers = {"Authorization": "Bearer hf_dummy"}
          # Initialize RAG system if LangChain is available
        self.rag_enabled = False
        self.rag_error = None
        
        if LANGCHAIN_AVAILABLE:
            try:
                self._setup_rag_system()
                self.rag_enabled = True
                print(" RAG system initialized successfully")
            except Exception as e:
                self.rag_error = str(e)
                print(f" RAG system failed to initialize: {e}")
                print("Falling back to rule-based responses")
        else:
            self.rag_error = "LangChain packages not installed"
    
    def _setup_rag_system(self):
        """Setup RAG system with course knowledge base"""
        try:
            # Load course information
            knowledge_file = "ai_bootcamp_info.txt"
            if os.path.exists(knowledge_file):
                with open(knowledge_file, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                content = self._get_default_course_content()
            
            print(f" Loaded knowledge base: {len(content)} characters")
            
            # Split text into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=100,
                separators=["\n\n", "\n", " ", ""]
            )
            documents = [Document(page_content=content)]
            split_docs = splitter.split_documents(documents)
            
            print(f" Split into {len(split_docs)} chunks")
              # Create embeddings and vector store with reduced model for faster loading
            print(" Loading embeddings model...")
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            print(" Creating vector store...")
            self.vector_store = FAISS.from_documents(split_docs, embeddings)
            # Use fewer retrieval results for more focused context
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})
              # Enhanced prompt template for concise, focused responses
            self.prompt = ChatPromptTemplate.from_template("""
            You are a professional AI Sales Agent for the AI Mastery Bootcamp. Provide concise, direct responses that are informative but brief.

            RESPONSE GUIDELINES:
            - Keep responses under 4 sentences maximum
            - Be direct and to the point
            - Focus on the most important information
            - Include specific details when relevant
            - End with a brief call-to-action if appropriate
            - Be professional and helpful

            COURSE CONTEXT:
            {context}

            Customer Question: {input}

            Provide a brief, focused response (maximum 3-4 sentences) that directly addresses their question:
            """)
            
            # Setup LLM with settings optimized for concise responses
            groq_api_key = os.getenv('GROQ_API_KEY')
            if groq_api_key:
                print(" Initializing Groq LLM...")
                self.llm = ChatGroq(
                    model="llama3-8b-8192", 
                    api_key=groq_api_key,
                    temperature=0.3,  # Lower temperature for more focused responses
                    max_tokens=200    # Limit tokens for shorter responses
                )
                
                # Create QA Chain
                document_chain = create_stuff_documents_chain(self.llm, self.prompt)
                self.retrieval_chain = create_retrieval_chain(self.retriever, document_chain)
                print("RAG chain created successfully")
            else:
                print(" No GROQ API key found")
                self.llm = None
                self.retrieval_chain = None
                raise Exception("GROQ API key not found")
                
        except Exception as e:
            print(f" Error in RAG setup: {e}")
            raise e
    
    def _get_default_course_content(self):
        """Default course content if file is not found"""
        return """        AI Mastery Bootcamp - 12-week comprehensive AI training program
        Price: $499 (Special offer: $299)
        Features: LLMs, Computer Vision, MLOps, Job placement assistance
        Hands-on projects, Industry mentorship, Certificate completion
        Flexible learning, 24/7 support, Money-back guarantee
        """
    
    def generate_response(self, conversation_history: list, customer_message: str) -> str:
        """Generate response using RAG system or fallback rules"""
        
        # Try RAG system first
        if self.rag_enabled and self.retrieval_chain:
            try:
                response = self.retrieval_chain.invoke({'input': customer_message})
                answer = response.get('answer', '').strip()
                
                # Clean up any unwanted tags
                if "</think>" in answer:
                    answer = answer.split("</think>")[-1].strip()
                
                if answer and len(answer) > 5:  # Valid response
                    return answer
            except Exception as e:
                print(f"RAG system error: {e}")
        
        # Fallback to rule-based responses
        return self._get_fallback_response(customer_message)
    
    def get_rag_response(self, customer_message: str) -> str:
        """Force RAG system response only"""
        if self.rag_enabled and self.retrieval_chain:
            try:
                response = self.retrieval_chain.invoke({'input': customer_message})
                answer = response.get('answer', '').strip()
                
                # Clean up any unwanted tags
                if "</think>" in answer:
                    answer = answer.split("</think>")[-1].strip()
                
                if answer and len(answer) > 5:  # Valid response
                    return answer
                else:
                    return "I apologize, but I couldn't find specific information about that. Could you please rephrase your question or ask about our AI Mastery Bootcamp features, pricing, or curriculum?"
            except Exception as e:
                print(f"RAG system error: {e}")
                return f"I'm experiencing some technical difficulties accessing the course information. Please try again or contact us directly at info@aimasterybootcamp.com"
        else:
            return "RAG system is not available. Please switch to Custom mode for responses."
    
    def get_rag_status(self) -> dict:
        """Get the status of the RAG system"""
        return {
            "available": self.rag_enabled,
            "error": self.rag_error,
            "message": "RAG system ready" if self.rag_enabled else f"RAG system not available: {self.rag_error}"
        }
    
    def generate_rag_response(self, customer_message: str) -> str:
        """Generate concise response using only RAG system"""
        if not self.rag_enabled or not self.retrieval_chain:
            return f"RAG system is not available. Please switch to Custom mode for responses."
        
        try:
            response = self.retrieval_chain.invoke({'input': customer_message})
            answer = response.get('answer', '').strip()
            
            # Clean up any unwanted tags
            if "</think>" in answer:
                answer = answer.split("</think>")[-1].strip()
            
            # Ensure response is concise (under 4 sentences)
            if answer:
                sentences = answer.split('. ')
                if len(sentences) > 4:
                    # Keep only first 3-4 sentences
                    answer = '. '.join(sentences[:3]) + '.'
                return answer
            else:
                return "I couldn't find specific information about that. Could you please rephrase your question?"
                
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def _get_fallback_response(self, customer_message: str) -> str:
        """Fallback responses for common scenarios"""
        message_lower = customer_message.lower()
        
        # Enhanced objection handling with course-specific details
        if any(word in message_lower for word in ["expensive", "cost", "price", "money", "afford"]):
            return "I understand the cost concern! Let me share some great news - we're offering the AI Mastery Bootcamp at a special discount of $299 instead of the regular $499. That's 40% off! Plus, we offer payment plans and a 30-day money-back guarantee. Consider this: the average AI engineer salary is $120,000+ annually, so this investment pays for itself quickly!"
        
        if any(word in message_lower for word in ["time", "busy", "schedule", "work"]):
            return "I totally understand - everyone's busy! That's why our AI Mastery Bootcamp is designed for working professionals. You only need 2-3 hours per week, with flexible scheduling including evening and weekend options. We also provide lifetime access to materials, so you can learn at your own pace. Many of our students completed it while working full-time!"
        
        if any(word in message_lower for word in ["already", "took", "course", "learned", "experience"]):
            return "That's fantastic - having some background will actually help you excel even more! Our AI Mastery Bootcamp is different because it focuses on the latest technologies like Large Language Models (LLMs), advanced MLOps, and real industry projects. Plus, you get job placement assistance and networking opportunities you won't find elsewhere. What specific AI area are you most experienced in?"
        
        if any(word in message_lower for word in ["not interested", "no thanks", "goodbye", "not now"]):
            return "No problem at all! I appreciate your time today. Before I go, can I ask what might make an AI course more appealing to you in the future? We're always improving our offerings. Either way, I wish you the best in your career journey!"
        
        if any(word in message_lower for word in ["tell me more", "details", "interested", "yes", "learn", "course"]):
            return "Excellent! The AI Mastery Bootcamp is a comprehensive 12-week program covering everything from machine learning fundamentals to cutting-edge LLMs and computer vision. You'll work on real projects, get personal mentorship, and receive job placement assistance. We have a 95% job placement rate! The regular price is $499, but today it's just $299. Would you like to know more about the curriculum or career outcomes?"
        
        if any(word in message_lower for word in ["job", "career", "employment", "hire", "work"]):
            return "Great question! Our job placement assistance is one of our strongest features. We have partnerships with 200+ companies actively hiring AI professionals. Our career support includes resume building, interview prep, portfolio development, and direct referrals. 95% of our graduates find AI roles within 6 months, with average starting salaries of $85,000-$120,000. Would you like to hear about specific success stories?"
        
        if any(word in message_lower for word in ["certificate", "certification", "credential"]):
            return "Yes! You'll receive an industry-recognized certificate upon completion. Our certificate is valued by employers because it represents hands-on project experience, not just theoretical knowledge. You'll have a portfolio of real AI projects to showcase. Many of our graduates mention that their certificate and project portfolio were key factors in landing their AI roles!"
        
        if any(word in message_lower for word in ["curriculum", "syllabus", "topics", "learn", "cover"]):
            return "Our curriculum is comprehensive and current! Week 1-3: AI/ML foundations, Week 4-6: Machine learning algorithms, Week 7-9: Deep learning and neural networks, Week 10-12: LLMs, computer vision, and MLOps. You'll work with Python, TensorFlow, PyTorch, and cloud platforms. Each week includes hands-on projects with real datasets. Would you like details about any specific topic?"
        
        # Default enthusiastic response
        return "Thank you for your interest in AI! The AI Mastery Bootcamp is a 12-week comprehensive program that transforms beginners into AI professionals. We cover LLMs, computer vision, MLOps, and provide hands-on projects with job placement assistance. The regular price is $499, but we're offering it for $299 today - that's a $200 savings! What aspect of AI interests you most?"
    
    def should_end_call(self, message: str) -> bool:
        """Determine if the call should end"""
        end_phrases = [
            "not interested", "no thanks", "goodbye", "stop calling", 
            "remove me", "don't call", "not now", "maybe later"
        ]
        return any(phrase in message.lower() for phrase in end_phrases)
