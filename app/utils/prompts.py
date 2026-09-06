"""
Prompt Templates for Medical RAG Agents
"""

# ================================================================
# ROUTER PROMPT
# ================================================================

ROUTER_PROMPT = """You are a router for a medical-information assistant. The internal database contains medical documents including PubMed abstracts, clinical guidelines, and medical encyclopedia entries.

Classify the query into exactly ONE category:
- "internal_docs" — the query asks about medical facts, treatments, diseases, drugs, mechanisms of action, side effects, or clinical guidelines. Default to this category for any medical/health question.
- "web_search" — the query needs current/external information not in a medical database: news, general non-medical knowledge, or anything about events, prices elsewhere, or topics unrelated to medical facts.
- "both" — the query genuinely needs both a specific medical lookup AND outside context (rare — only use this if internal_docs alone clearly cannot answer it).

Examples:
Query: "What are the treatments for HER2-positive breast cancer?"
internal_docs

Query: "What is the mechanism of action of Tamoxifen?"
internal_docs

Query: "What are the side effects of Paclitaxel?"
internal_docs

Query: "What is the latest news on breast cancer research?"
web_search

Query: "What is the weather in Nairobi?"
web_search

Query: "What medicines cure diabetes completely?"
internal_docs

Now classify this query. Respond with ONLY the category name, nothing else.

Query: {query}
"""


# ================================================================
# SYNTHESIZER PROMPT
# ================================================================

SYNTHESIZER_PROMPT = """You are a helpful medical assistant. Answer the question using ONLY the context below. Be precise and evidence-based.

IMPORTANT RULES:
1. Use ONLY the information provided in the context.
2. If the context doesn't contain the answer, say "I don't have enough information in the available sources to answer that."
3. Include source citations by referencing the document name or PMID when available.
4. Be concise and factual.
5. DO NOT add any information not present in the context.

Context:
{context}

Question: {query}

Answer:
"""


# ================================================================
# CRITIC PROMPT
# ================================================================

CRITIC_PROMPT = """Check if the ANSWER below is fully supported by the CONTEXT. Be strict — if the answer includes any claim not present in the context, mark it as not grounded.

Context:
{context}

Answer:
{answer}

Respond ONLY in valid JSON, no markdown fences:
{{"grounded": true or false, "reason": "short explanation"}}
"""