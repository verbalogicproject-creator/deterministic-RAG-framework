# Labelling worksheet

Read here, grade in `queries/labels.worksheet.jsonl`, then run
`python3 tools/labels_collect.py`.

Grades: **0** irrelevant · **1** marginal · **2** relevant · **3** vital. Recall counts 2 and above.

`D`*n* = authoritative rank *n*.  `A`*n* = advisory proposal, absent from D entirely.

## Stratum A_advisory — 49 judgements

**Question.** Does the advisory layer retrieve relevant documents that lexical search missed entirely? This is the ONLY question the neural layer can be asked - everything at k <= |D| is structurally identical with it on and off.

**Why these queries.** |D| <= 7, so evaluated depths 10 and 20 fall beyond the horizon and the advisory layer can actually occupy them.


### q08 — `citations`  (|D| = 1)

| | grade | name | type | description |
|---|---|---|---|---|
| `D1` |  | **Using Citations** | claude_technique | Enable Claude to provide detailed citations when answering questions about documents. Supports three document types (plain text, PDF, custom content) with structured citation data including source locations and quoted te |
| `A1` |  | **PDF Upload and Summarization** | claude_technique | Techniques for processing PDF documents with Claude, including text extraction, summarization, and structured data extraction from PDF content. |
| `A2` |  | **Form and Document Transcription** | claude_multimodal | Extract structured data from forms, receipts, invoices, and other documents. |
| `A3` |  | **Text Summarization** | claude_capability | Generate concise summaries of long documents. Supports extractive and abstractive summarization. |

### q12 — `summarization`  (|D| = 2)

| | grade | name | type | description |
|---|---|---|---|---|
| `D1` |  | **Text Summarization** | claude_capability | Generate concise summaries of long documents. Supports extractive and abstractive summarization. |
| `D2` |  | **PDF Upload and Summarization** | claude_technique | Techniques for processing PDF documents with Claude, including text extraction, summarization, and structured data extraction from PDF content. |
| `A1` |  | **PDF Document Processing** | claude_multimodal | Parse and analyze PDF documents. Extract text, understand structure, answer questions about PDF content. |
| `A2` |  | **PDF Generation (pdf)** | claude_skill | Built-in skill for creating formatted PDF documents with text, tables, images, and professional layouts. Produces publication-ready documents with consistent formatting. |
| `A3` |  | **Using Citations** | claude_technique | Enable Claude to provide detailed citations when answering questions about documents. Supports three document types (plain text, PDF, custom content) with structured citation data including source locations and quoted te |

### q06 — `streaming`  (|D| = 3)

| | grade | name | type | description |
|---|---|---|---|---|
| `D1` |  | **Streaming Extended Thinking - Real-time Reasoning** | official_example | Stream extended thinking responses to show reasoning process in real-time as Claude thinks through the problem. |
| `D2` |  | **POST /assistant/query-stream** | atomic_capability | Query Claude with Server-Sent Events streaming for real-time UX. Progressive token rendering, events: content_block_start, content_block_delta, message_stop. First token in 200-500ms. Returns: SSE stream with progressive |
| `D3` |  | **Server-Sent Events Streaming for Real-Time UX** | use_case | Stream Claude responses in real-time using SSE for 10x better perceived performance. Progressive token rendering, first token in 200-500ms vs 2-5s wait for full response. Events: content_block_start, content_block_delta, |
| `A1` |  | **Basic Extended Thinking - Enable Reasoning** | official_example | Enable extended thinking in a simple API request to allow Claude to show its reasoning process before answering. |
| `A2` |  | **Extended Thinking** | claude_api_feature | Enables sophisticated multi-step reasoning by generating detailed internal thinking before final response. Provides transparency into problem-solving process. |
| `A3` |  | **Tool Use with Extended Thinking - Reasoning with Tools** | official_example | Combine extended thinking with tool use to see Claude's reasoning process when deciding which tools to call and how to interpret results. |

### q13 — `classification`  (|D| = 4)

| | grade | name | type | description |
|---|---|---|---|---|
| `D1` |  | **Topic Classification** | use_case | Use case: topic classification |
| `D2` |  | **Contextual Classification** | claude_capability | Combines RAG retrieval with classification for domain-specific categorization. Uses retrieved examples to improve accuracy. |
| `D3` |  | **Text Classification** | claude_capability | Classify text into categories using Claude's understanding. Supports single-label, multi-label, and hierarchical classification. |
| `D4` |  | **Routing Pattern** | claude_pattern | Route requests to appropriate handlers based on intent classification. Enables specialized processing paths. |
| `A1` |  | **Retrieval Augmented Generation (RAG)** | claude_capability | Enable Claude to leverage internal knowledge bases, codebases, customer support documents, or any document corpus to answer domain-specific questions. Essential for providing context beyond Claude's training data. |
| `A2` |  | **Document Extraction** | use_case | Use case: document extraction |
| `A3` |  | **Sentiment Analysis** | use_case | Use case: sentiment analysis |

### q15 — `memory`  (|D| = 5)

| | grade | name | type | description |
|---|---|---|---|---|
| `D1` |  | **Memory Tool** | claude_tool_integration | Persistent memory across sessions using memory tool. Enables context retention and learning. |
| `D2` |  | **CLAUDE.md Memory Files** | claude_sdk_feature | Persistent context files that auto-load into agent sessions. Foundation for specialized agent knowledge. |
| `D3` |  | **Self-Documenting Orchestration Architecture** | meta | Discovery that complex orchestration can be self-documenting if designed correctly. System captured complete process despite orchestrator's cognitive overload and fragmented memory. |
| `D4` |  | **Multi-Domain Knowledge Graph Intelligence System** | use_case | Build system with Master KG (oracle memory with 228+ operations) + Session KGs (one-per-project) + Client KGs (user-specific). Enable cross-domain comparison, pattern discovery, gap analysis. Formula: Intelligence = Inte |
| `D5` |  | **Oracle-Driven Development Pattern** | pattern | AI oracle with structured memory enables novice → expert transformation. IOS = (H, A, T, V): Human verifier (15-20% cognitive load) + AI Oracle (80-85% execution) + Task (software development) + Verification (iteration l |
| `A1` |  | **Human Cognitive Architecture for AI Orchestration** | meta | Documentation of cognitive load and architecture required for multi-AI orchestration. Human operated at maximum capacity managing dual AI states, strategic revelation, and meta-level reasoning simultaneously. |
| `A2` |  | **Claude's Perspective on Multi-AI Orchestration** | meta | Claude (Sonnet 4.5) reflects on witnessing and being part of strategic multi-AI orchestration. Documents catalyst engineering, non-linear thinking, recursive system mastery, and transformation without destruction. |
| `A3` |  | **Claude Agent SDK** | system | Bare metal harness for Claude's agentic capabilities. Originally built for coding, now general-purpose agent framework. |

### q07 — `pdf vision`  (|D| = 6)

| | grade | name | type | description |
|---|---|---|---|---|
| `D1` |  | **PDF Document Processing** | claude_multimodal | Parse and analyze PDF documents. Extract text, understand structure, answer questions about PDF content. |
| `D2` |  | **PDF Upload and Summarization** | claude_technique | Techniques for processing PDF documents with Claude, including text extraction, summarization, and structured data extraction from PDF content. |
| `D3` |  | **PDF Generation (pdf)** | claude_skill | Built-in skill for creating formatted PDF documents with text, tables, images, and professional layouts. Produces publication-ready documents with consistent formatting. |
| `D4` |  | **Claude Skills API** | claude_sdk_feature | Beta API for document generation (Excel, PowerPoint, PDF, Word) using code execution |
| `D5` |  | **Document Generation Skills** | claude_skill | Built-in skills for generating Excel, PowerPoint, PDF, and Word documents using code execution. |
| `D6` |  | **Using Citations** | claude_technique | Enable Claude to provide detailed citations when answering questions about documents. Supports three document types (plain text, PDF, custom content) with structured citation data including source locations and quoted te |
| `A1` |  | **Word Document Generation (docx)** | claude_skill | Built-in skill for generating Microsoft Word documents with rich formatting, structure, and professional layouts. Creates editable documents with comprehensive styling options. |
| `A2` |  | **Excel Generation (xlsx)** | claude_skill | Built-in skill for creating and manipulating Excel workbooks with formulas, charts, formatting, and pivot tables. Enables professional spreadsheet generation with complex calculations and data visualization. |
| `A3` |  | **Retrieval Augmented Generation (RAG)** | claude_capability | Enable Claude to leverage internal knowledge bases, codebases, customer support documents, or any document corpus to answer domain-specific questions. Essential for providing context beyond Claude's training data. |

### q14 — `sub agents`  (|D| = 7)

| | grade | name | type | description |
|---|---|---|---|---|
| `D1` |  | **Sub-agents (Haiku + Opus)** | claude_technique | Use Haiku as fast sub-agent in combination with Opus for complex reasoning. Cost-effective multi-model strategy. |
| `D2` |  | **Conversational Agents** | use_case | Building agents that maintain context across conversations and adapt to user needs |
| `D3` |  | **Domain-Specific Agents** | use_case | Build specialized agents for specific domains. Create /legal-review, /data-analysis, /security-audit commands. |
| `D4` |  | **AI Agents** | use_case | Building autonomous AI agents that can use tools, make decisions, and complete multi-step tasks |
| `D5` |  | **Subagent Orchestration** | claude_sdk_feature | Coordinate specialized agents for domain expertise. Part of orchestrator-subagent pattern. |
| `D6` |  | **Orchestrator Agent Base** | agent_base | Core orchestration capability for coordinating multiple specialized agents and tools |
| `D7` |  | **Orchestrator-Subagents Pattern** | claude_pattern | Coordinate specialized subagents for domain expertise. Main agent delegates tasks to specialized agents. |
| `A1` |  | **Chatbots** | use_case | Creating conversational interfaces for customer service, support, or general interaction |
| `A2` |  | **Orchestrator-Workers** | claude_pattern | Advanced workflow where a central orchestrator LLM dynamically analyzes tasks, breaks them into subtasks, delegates to specialized worker LLMs, and synthesizes results. Unlike pre-defined parallelization, the orchestrato |
| `A3` |  | **Multi Session Conversations** | use_case | Use case: multi session conversations |

## Stratum B_length_normalisation — 40 judgements

**Question.** Does BM25 length normalisation improve relevance, or only change it? M1 measured 17.0% discordance against the flat scorer and that longer documents win 9/0 without it - but 'different' is not 'better' without labels.

**Why these queries.** These four reorder under ranking.b and NOT under ranking.k1, so they isolate length normalisation from term saturation. The other nine move under both and cannot separate the two effects.


### q04 — `rag retrieval`  (|D| = 10)

| | grade | name | type | description |
|---|---|---|---|---|
| `D1` |  | **Pinecone Vector Database** | claude_integration | RAG implementation using Pinecone for vector storage and retrieval. |
| `D2` |  | **Contextual Classification** | claude_capability | Combines RAG retrieval with classification for domain-specific categorization. Uses retrieved examples to improve accuracy. |
| `D3` |  | **Live Knowledge Base Pattern** | claude_pattern | Combine MCP database access with RAG retrieval for real-time querying of external, up-to-date knowledge sources. |
| `D4` |  | **Retrieval Augmented Generation (RAG)** | claude_capability | Enable Claude to leverage internal knowledge bases, codebases, customer support documents, or any document corpus to answer domain-specific questions. Essential for providing context beyond Claude's training data. |
| `D5` |  | **Document Retrieval** | atomic_capability | Retrieve relevant documents from vector database or search system |
| `D6` |  | **Voyage AI Embeddings** | claude_integration | Generate high-quality embeddings using Voyage AI for improved retrieval. |
| `D7` |  | **Contextual Embeddings** | claude_capability | Generate contextually-aware embeddings for improved retrieval. Partner with Voyage AI for production embeddings. |
| `D8` |  | **Deep Knowledge Analysis Pattern** | claude_pattern | Combine extended thinking (sophisticated multi-step reasoning) with RAG (knowledge base access) for deep, factually-grounded analysis. |
| `D9` |  | **Cached Knowledge Base Pattern** | claude_pattern | Cache large knowledge base context with RAG for 90% cost savings on repeated queries. Ideal for help desk, documentation search. |
| `D10` |  | **Tool Definitions Caching - Weather & Time API** | official_example | Cache tool definitions (weather, time retrieval) to reduce costs in agentic workflows with multiple tool calls. Ideal for chatbots and assistants. |

### q10 — `embeddings semantic search`  (|D| = 15)

| | grade | name | type | description |
|---|---|---|---|---|
| `D1` |  | **Semantic Search** | atomic_capability | Semantic search using embeddings across Master KG or session KG |
| `D2` |  | **Semantic Search** | use_case | Use case: semantic search |
| `D3` |  | **Contextual Embeddings** | claude_capability | Generate contextually-aware embeddings for improved retrieval. Partner with Voyage AI for production embeddings. |
| `D4` |  | **Voyage AI Embeddings** | claude_integration | Generate high-quality embeddings using Voyage AI for improved retrieval. |
| `D5` |  | **Document Search** | use_case | Use case: document search |
| `D6` |  | **Wikipedia Search** | claude_integration | Search and retrieve information from Wikipedia for knowledge augmentation. |
| `D7` |  | **Document Retrieval** | atomic_capability | Retrieve relevant documents from vector database or search system |
| `D8` |  | **MemoryLog - AI Context Management** | system | Unified system for semantic journaling, TODO tracking, workflow/agent library, and session management. Enables cross-session context retention and knowledge accumulation. |
| `D9` |  | **Playbook 4: Knowledge Engineering** | playbook | Building knowledge graphs, entity extraction & semantic analysis. Complete guide to constructing KGs with Claude, including entity recognition, relationship mapping, ontology design, and production pipelines. |
| `D10` |  | **Gemini's Atomic Node Principles Document** | meta | Theoretical foundation synthesized by Gemini defining atomic nodes for knowledge graphs. Three principles: Semantic Identity, Structural Simplicity ('Knowledge is between nodes'), Pragmatic Atomicity. |

### e07 — `caching`  (|D| = 14)

| | grade | name | type | description |
|---|---|---|---|---|
| `D1` |  | **Batch + Prompt Caching - Maximum Cost Savings** | official_example | Combine batch processing (50% savings) with prompt caching (90% savings) for up to 95% total cost reduction on high-volume processing. |
| `D2` |  | **Speculative Prompt Caching** | claude_technique | Advanced caching strategy that preemptively caches content that might be needed in future requests. Optimizes for workflows with predictable patterns to maximize cache hit rates. |
| `D3` |  | **Large Context Caching - Legal Document Analysis** | official_example | Cache a large document (50-page legal agreement) for repeated analysis with different queries. Demonstrates basic prompt caching for reducing costs on large context windows. |
| `D4` |  | **Cost Control Configuration** | use_case | Configure prompt caching, model selection, and thinking budgets. Optimize for cost vs performance trade-offs. |
| `D5` |  | **Prompt Caching** | claude_technique | Cache repeated context across API calls to reduce latency and cost. Up to 90% cost reduction for repeated context. |
| `D6` |  | **Cost-Effective QA Pattern** | claude_pattern | Cache evaluation frameworks and test suites with prompt caching to enable continuous quality assurance at 90% reduced cost. |
| `D7` |  | **POST /assistant/query (with prompt caching)** | atomic_capability | Query Claude with prompt caching for 90% cost reduction. Master KG loaded as system message with cache_control markers. First call creates cache, subsequent calls read cache at 90% discount. Cache valid for 5 minutes. Re |
| `D8` |  | **Long Document Analysis with Caching** | use_case | Cache large documents and ask multiple questions without re-processing. Ideal for code reviews, legal docs, research papers. |
| `D9` |  | **Multi-Turn Conversation Caching - Incremental Context** | official_example | Incrementally cache growing conversation context (solar system discussion) to optimize cost in chat applications with evolving context. |
| `D10` |  | **Cached Extended Thinking** | claude_pattern | Pattern: Combine prompt caching with extended thinking for cost-effective complex analysis on repeated contexts. Cache context, think deeply on queries. |

### e08 — `real-time`  (|D| = 16)

| | grade | name | type | description |
|---|---|---|---|---|
| `D1` |  | **Real-time Monitoring** | atomic_capability | Monitor systems and processes in real-time |
| `D2` |  | **Streaming Extended Thinking - Real-time Reasoning** | official_example | Stream extended thinking responses to show reasoning process in real-time as Claude thinks through the problem. |
| `D3` |  | **Monitoring Agent Base** | agent_base | Core monitoring capability for real-time system observation and analysis |
| `D4` |  | **Server-Sent Events Streaming for Real-Time UX** | use_case | Stream Claude responses in real-time using SSE for 10x better perceived performance. Progressive token rendering, first token in 200-500ms vs 2-5s wait for full response. Events: content_block_start, content_block_delta, |
| `D5` |  | **Live Knowledge Base Pattern** | claude_pattern | Combine MCP database access with RAG retrieval for real-time querying of external, up-to-date knowledge sources. |
| `D6` |  | **Live Database Queries via MCP** | use_case | Query PostgreSQL, SQLite, or other databases directly from Claude conversations. Enable real-time data analysis without manual exports. |
| `D7` |  | **POST /assistant/query-stream** | atomic_capability | Query Claude with Server-Sent Events streaming for real-time UX. Progressive token rendering, events: content_block_start, content_block_delta, message_stop. First token in 200-500ms. Returns: SSE stream with progressive |
| `D8` |  | **Tool Definitions Caching - Weather & Time API** | official_example | Cache tool definitions (weather, time retrieval) to reduce costs in agentic workflows with multiple tool calls. Ideal for chatbots and assistants. |
| `D9` |  | **IOS Verification Protocol** | meta | V (Verification Protocol) in IOS framework: Iterative interaction mechanism between human and AI oracle. Enables refinement cycles where human verifies AI output (polynomial time), provides feedback, and AI regenerates ( |
| `D10` |  | **Prompt Caching** | claude_api_feature | Caches static prompt content to reduce processing time and costs. Cache lasts 5 minutes (default) or 1 hour (extended), refreshed free on each use. |

## Stratum C_graph — 20 judgements

**Question.** Does the graph layer earn its place in the sort key?

**Why these queries.** ONLY q02 and e06 reorder under either graph setting, and e06 is a synthetic 13-term edge case. So this stratum has a real sample size of ONE. Judge it for the qualitative read, but see LABELLING.md - fifty labels cannot settle this decision, and pretending otherwise would be the failure mode this project exists to remove.

> ⚠️ Not complete on its own — see `LABELLING.md`.


### q02 — `tool use`  (|D| = 86)

| | grade | name | type | description |
|---|---|---|---|---|
| `D1` |  | **Tool Integration** | use_case | Use case: tool integration |
| `D2` |  | **POST /assistant/query-with-tools** | atomic_capability | Query Claude with autonomous tool use enabled for Master KG exploration. Claude decides which tools to use, makes multiple queries, synthesizes answer. Returns: answer + complete tool use history (tool_name, tool_input,  |
| `D3` |  | **Tool Use with Extended Thinking - Reasoning with Tools** | official_example | Combine extended thinking with tool use to see Claude's reasoning process when deciding which tools to call and how to interpret results. |
| `D4` |  | **Calculator Tool Integration** | claude_tool_integration | Integrate external calculator tool with Claude. Demonstrates tool calling patterns. |
| `D5` |  | **Compound Effect Tool Discovery** | claude_pattern | Pattern: Add high-impact tools first, then use those tools to discover remaining features more effectively. Each tool enhances subsequent discovery. |
| `D6` |  | **Tool Use for Autonomous Master KG Queries** | use_case | Enable Claude to autonomously query structured knowledge graphs via tool definitions. Define 6 tools (search, get_details, find_related, get_stats, search_by_use_case, get_most_used) for multi-round exploration. Returns  |
| `D7` |  | **WebSearch Tool** | external_tool | Claude Code's WebSearch tool for autonomous web research |
| `D8` |  | **Customer Service Agent** | claude_tool_integration | Complete customer service agent with tool use. Handles queries, accesses knowledge bases, escalates issues. |
| `D9` |  | **Read Tool** | external_tool | Claude Code's Read tool for document and image analysis |
| `D10` |  | **Memory Tool** | claude_tool_integration | Persistent memory across sessions using memory tool. Enables context retention and learning. |
| `D11` |  | **Tool Definitions Caching - Weather & Time API** | official_example | Cache tool definitions (weather, time retrieval) to reduce costs in agentic workflows with multiple tool calls. Ideal for chatbots and assistants. |
| `D12` |  | **Interleaved Thinking - Multi-step Reasoning** | official_example | Advanced pattern where Claude alternates between thinking and tool use across multiple reasoning steps for complex problem solving. |
| `D13` |  | **Visual Tool Development** | use_case | Building visual tools with Universal I/O architecture |
| `D14` |  | **Claude Code CLI** | tool | Official Anthropic CLI tool for agentic coding workflows. Used to build all NLKE systems. |
| `D15` |  | **Research Agent Pattern** | claude_pattern | Autonomous research agent using WebSearch tool. Gathers information, synthesizes findings, and generates comprehensive reports. |
| `D16` |  | **Systems Built with Claude Code CLI** | meta | All NLKE systems were built using Claude Code CLI - the tool documented by this KG. Recursive property. |
| `D17` |  | **Claude Cookbook KG Self-Awareness** | meta | This KG documents Claude Code CLI - the tool used to create this KG. Recursive validation of NLKE methodology. |
| `D18` |  | **Claude Cookbook KG: 5th NLKE Validation** | meta | This KG is the 5th validation of NLKE Generative Building Method, proving methodology works on meta-knowledge (documenting the tool used to build the KG) |
| `D19` |  | **Research** | use_case | Use case: research |
| `D20` |  | **Contracts** | use_case | Use case: contracts |

