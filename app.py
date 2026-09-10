import logging
import requests
import gradio as gr


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("glucodoc")


# ============================================================
# Configuration
# ============================================================

FASTAPI_URL = "http://127.0.0.1:8000"


# ============================================================
# API Client
# ============================================================

def serialize_history(history):
    """
    Convert Gradio chatbot history (type="messages" dicts) into the
    clean {role, content} dicts our FastAPI expects.
    """
    messages = []
    if not history:
        return messages

    for item in history:
        # type="messages" format: each item is already a {role, content} dict
        if isinstance(item, dict):
            role = item.get("role", "")
            content = item.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": str(content)})
        # Legacy fallback: [user_msg, bot_msg] pairs
        elif isinstance(item, (list, tuple)):
            user_msg, bot_msg = item[0], item[1]
            if user_msg is not None:
                messages.append({"role": "user", "content": str(user_msg)})
            if bot_msg is not None:
                messages.append({"role": "assistant", "content": str(bot_msg)})

    return messages


def ask_glucodoc(question: str, history: list = None, summarize: bool = False):
    """
    Send the user's question and history to the FastAPI backend.
    """

    if not question or not question.strip():
        return {
            "answer": "Please enter a question so I can help.",
            "confidence": "N/A",
            "evidence": "No question was provided.",
            "citations": [],
        }

    question = question.strip()
    clean_history = serialize_history(history)

    logger.info("Sending question to FastAPI: %s", question)

    try:
        response = requests.post(
            f"{FASTAPI_URL}/chat",
            json={
                "question": question,
                "history": clean_history,
                "summarize": summarize,
            },
            timeout=120,
        )

        response.raise_for_status()
        data = response.json()
        logger.info("Response received from FastAPI.")

        return {
            "answer": data.get("answer", "No answer was returned."),
            "confidence": data.get("confidence", "N/A"),
            "evidence": data.get("evidence", "No evidence was returned."),
            "citations": data.get("citations", []),
        }

    except requests.exceptions.ConnectionError:
        logger.exception("Could not connect to FastAPI.")
        return {
            "answer": "⚠️ Unable to connect to the GlucoDoc API. Please make sure FastAPI is running (`uvicorn api.fastapi:app --port 8000`).",
            "confidence": "N/A",
            "evidence": "FastAPI is not reachable.",
            "citations": [],
        }

    except requests.exceptions.Timeout:
        logger.exception("FastAPI request timed out.")
        return {
            "answer": "⏱️ The request took too long. Please try again.",
            "confidence": "N/A",
            "evidence": "Request timeout.",
            "citations": [],
        }

    except Exception:
        logger.exception("Failed to communicate with FastAPI.")
        return {
            "answer": "Something went wrong while processing your question. Please try again.",
            "confidence": "N/A",
            "evidence": "An unexpected API error occurred.",
            "citations": [],
        }


# ============================================================
# Formatting
# ============================================================

def format_confidence(value):
    """Format confidence for display."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        percentage = value * 100
        if percentage >= 70:
            return f"🟢 {percentage:.1f}%"
        elif percentage >= 40:
            return f"🟡 {percentage:.1f}%"
        else:
            return f"🔴 {percentage:.1f}%"
    if isinstance(value, int):
        return str(value)
    return str(value)


def format_evidence(evidence, citations):
    """Format evidence and citations for the UI."""
    output = ""

    if evidence:
        output += str(evidence)

    if citations:
        output += "\n\n### 📚 Sources\n"
        for citation in citations:
            if isinstance(citation, dict):
                source = citation.get("source", citation.get("file_name", "Unknown source"))
                page = citation.get("page", "Unknown page")
                output += f"- **{source}** — Page {page}\n"
            else:
                output += f"- {citation}\n"

    if not output.strip():
        output = "No evidence information was returned."

    return output


# ============================================================
# Chat Handler
# ============================================================

def respond(message, history, summarize=False):
    """Handle a chat message through FastAPI."""

    if not message or not message.strip():
        return ("", history or [], "", "N/A", "No question was provided.")

    question = message.strip()
    result = ask_glucodoc(question, history=history, summarize=summarize)

    answer = result["answer"]
    confidence = format_confidence(result["confidence"])
    evidence = format_evidence(result["evidence"], result["citations"])

    # Use messages format required by gr.Chatbot(type="messages")
    history = history or []
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})

    return ("", history, answer, confidence, evidence)


# ============================================================
# Clear
# ============================================================

def clear_conversation():
    return ([], "", "", "N/A", "")


# ============================================================
# Custom CSS — Dark & Vibrant
# ============================================================

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* =========================================================
   Global & Theme
   ========================================================= */
:root {
    --primary:        #38bdf8;
    --primary-hover:  #0ea5e9;
    --primary-glow:   rgba(56, 189, 248, 0.3);
    --accent:         #818cf8;
    --bg-root:        #0b0f19;
    --bg-card:        #111827;
    --bg-card-hover:  #1a2236;
    --bg-input:       #1e2d3d;
    --text-main:      #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted:     #64748b;
    --border:         rgba(148, 163, 184, 0.12);
    --border-focus:   rgba(56, 189, 248, 0.5);
    --shadow-glow:    0 0 20px rgba(56, 189, 248, 0.15);
    --radius-sm:      8px;
    --radius-md:      14px;
    --radius-lg:      20px;
    --radius-xl:      28px;
}

body, .gradio-container {
    background: var(--bg-root) !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    color: var(--text-main) !important;
}

.gradio-container {
    max-width: 1440px !important;
    margin: 0 auto !important;
    padding: 24px !important;
}

/* Remove default Gradio white backgrounds */
.gr-box, .gr-form, .gr-panel {
    background: transparent !important;
}

/* =========================================================
   Sidebar
   ========================================================= */
#sidebar {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    padding: 24px 20px;
    box-shadow: var(--shadow-glow);
    display: flex;
    flex-direction: column;
    gap: 12px;
    height: 100%;
}

.brand-lockup {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 8px;
}

.brand-mark {
    width: 48px;
    height: 48px;
    border-radius: 14px;
    display: grid;
    place-items: center;
    font-size: 24px;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
    box-shadow: 0 4px 16px rgba(56, 189, 248, 0.4);
    flex-shrink: 0;
}

.brand-name {
    font-size: 22px;
    font-weight: 800;
    color: var(--text-main);
    letter-spacing: -0.02em;
}

.brand-caption {
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 2px;
}

.section-label {
    color: var(--text-muted) !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    margin: 10px 0 6px !important;
}

.examples-column {
    display: flex !important;
    flex-direction: column !important;
    gap: 8px !important;
}

.example-chip button {
    border: 1px solid var(--border) !important;
    background: rgba(30, 45, 61, 0.6) !important;
    color: var(--primary) !important;
    border-radius: 10px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 10px 14px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    transition: all 0.2s ease !important;
    line-height: 1.4 !important;
}

.example-chip button:hover {
    background: rgba(56, 189, 248, 0.08) !important;
    border-color: var(--primary) !important;
    transform: translateX(3px) !important;
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.1) !important;
}

.divider-line {
    border: none;
    border-top: 1px solid var(--border);
    margin: 4px 0;
}

#clear-button {
    border: 1px solid rgba(239, 68, 68, 0.3) !important;
    color: #f87171 !important;
    background: transparent !important;
    border-radius: 10px !important;
    font-size: 13px !important;
    transition: all 0.2s ease !important;
    margin-top: 4px !important;
}

#clear-button:hover {
    background: rgba(239, 68, 68, 0.1) !important;
    border-color: #f87171 !important;
    box-shadow: 0 0 12px rgba(239, 68, 68, 0.15) !important;
}

.info-panel {
    background: rgba(56, 189, 248, 0.05);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: var(--radius-md);
    padding: 14px;
}

.info-panel h3 {
    margin: 0 0 6px;
    font-size: 13px;
    color: var(--primary);
    font-weight: 600;
}

.info-panel p {
    margin: 0;
    color: var(--text-secondary);
    font-size: 12px;
    line-height: 1.5;
}

.disclaimer {
    padding: 14px;
    border: 1px solid rgba(234, 179, 8, 0.25);
    border-radius: var(--radius-md);
    background: rgba(234, 179, 8, 0.05);
    color: #fde68a;
    font-size: 11.5px;
    line-height: 1.5;
}

.footer {
    text-align: center;
    color: var(--text-muted);
    font-size: 11px;
    margin-top: 4px;
}

/* =========================================================
   Main Panel
   ========================================================= */
#main-panel {
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.hero-card {
    padding: 28px 32px;
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    background: var(--bg-card);
    position: relative;
    overflow: hidden;
}

.hero-card::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(56,189,248,0.12) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-card::after {
    content: '';
    position: absolute;
    bottom: -80px; left: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(129,140,248,0.08) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-title {
    font-size: clamp(22px, 3vw, 30px);
    font-weight: 800;
    line-height: 1.25;
    color: var(--text-main);
    margin-bottom: 10px;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #f1f5f9 0%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    position: relative;
    z-index: 1;
}

.hero-copy {
    max-width: 580px;
    color: var(--text-secondary);
    font-size: 15px;
    line-height: 1.6;
    position: relative;
    z-index: 1;
}

.badge-row {
    display: flex;
    gap: 10px;
    margin-top: 14px;
    flex-wrap: wrap;
    position: relative;
    z-index: 1;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 10px;
    border-radius: 100px;
    font-size: 11.5px;
    font-weight: 600;
    border: 1px solid;
}

.badge-blue  { background: rgba(56,189,248,0.08); border-color: rgba(56,189,248,0.3); color: #38bdf8; }
.badge-violet{ background: rgba(129,140,248,0.08); border-color: rgba(129,140,248,0.3); color: #818cf8; }
.badge-green { background: rgba(52,211,153,0.08); border-color: rgba(52,211,153,0.3); color: #34d399; }

/* =========================================================
   Chatbot
   ========================================================= */
#conversation {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-xl) !important;
    background: var(--bg-card) !important;
    box-shadow: var(--shadow-glow);
}

/* User bubbles */
.message-wrap .user {
    background: rgba(56, 189, 248, 0.1) !important;
    border: 1px solid rgba(56, 189, 248, 0.25) !important;
    color: var(--text-main) !important;
    border-radius: 18px 18px 4px 18px !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
    padding: 14px 18px !important;
}

/* Bot bubbles */
.message-wrap .bot {
    background: var(--bg-card-hover) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-main) !important;
    border-radius: 18px 18px 18px 4px !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
    padding: 14px 18px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
}

.message-wrap .bot p { color: var(--text-main) !important; }
.message-wrap .bot code {
    background: rgba(56,189,248,0.1) !important;
    color: var(--primary) !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-size: 13px !important;
}

/* =========================================================
   Composer (input area)
   ========================================================= */
.composer {
    margin-top: 2px !important;
    padding: 6px 6px 6px 14px !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    background: var(--bg-input) !important;
    transition: all 0.25s ease !important;
    align-items: center !important;
}

.composer:focus-within {
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 3px var(--primary-glow) !important;
}

.composer textarea {
    border: 0 !important;
    box-shadow: none !important;
    color: var(--text-main) !important;
    background: transparent !important;
    padding: 10px 4px !important;
    font-size: 15px !important;
    resize: none !important;
    caret-color: var(--primary) !important;
}

.composer textarea::placeholder {
    color: var(--text-muted) !important;
}

#send-button {
    border-radius: 12px !important;
    background: linear-gradient(135deg, #38bdf8, #0ea5e9) !important;
    color: #0f172a !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 10px rgba(56, 189, 248, 0.35) !important;
}

#send-button:hover {
    background: linear-gradient(135deg, #7dd3fc, #38bdf8) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(56, 189, 248, 0.45) !important;
}

/* =========================================================
   Summarize Toggle
   ========================================================= */
.summarize-row {
    padding: 4px 0 !important;
}

.summarize-row label {
    color: var(--text-secondary) !important;
    font-size: 13px !important;
    cursor: pointer !important;
}

.summarize-row input[type=checkbox] {
    accent-color: var(--primary) !important;
}

/* =========================================================
   Accordion & Details
   ========================================================= */
.accordion-wrapper {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    background: var(--bg-card) !important;
    overflow: hidden !important;
}

.accordion-wrapper > .gr-accordion-header {
    color: var(--text-secondary) !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 14px 18px !important;
}

#confidence-box textarea, #confidence-box input {
    background: var(--bg-input) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
}

#evidence-box, #answer-box {
    background: var(--bg-input) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 16px !important;
}

#evidence-box p, #answer-box p { color: var(--text-main) !important; }
#evidence-box h3 { color: var(--primary) !important; font-size: 15px !important; }
#evidence-box strong { color: var(--text-main) !important; }

/* =========================================================
   General Gradio overrides
   ========================================================= */
label span {
    color: var(--text-secondary) !important;
}

.svelte-1gfkn6j {
    color: var(--text-main) !important;
}

@media (max-width: 768px) {
    #main-panel { order: -1; }
    #sidebar { margin-top: 16px; }
    .badge-row { display: none; }
}
"""


# ============================================================
# Gradio Application
# ============================================================

theme = gr.themes.Base(
    primary_hue="sky",
    secondary_hue="violet",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

with gr.Blocks(title="GlucoDoc | Diabetes Information Assistant") as demo:

    with gr.Row():

        # ------------------------------------------------
        # Sidebar
        # ------------------------------------------------
        with gr.Column(scale=3, elem_id="sidebar"):

            gr.HTML("""
                <div class="brand-lockup">
                    <div class="brand-mark">✚</div>
                    <div>
                        <div class="brand-name">GlucoDoc</div>
                        <div class="brand-caption">Your diabetes knowledge companion</div>
                    </div>
                </div>
            """)

            gr.Markdown("Suggested Questions", elem_classes=["section-label"])

            with gr.Column(elem_classes=["examples-column"]):
                example_1 = gr.Button("🔍  Diagnostic criteria for diabetes", elem_classes=["example-chip"])
                example_2 = gr.Button("🩸  What is type 2 diabetes?",         elem_classes=["example-chip"])
                example_3 = gr.Button("⚖️  Type 1 vs. Type 2 differences",   elem_classes=["example-chip"])
                example_4 = gr.Button("⚠️  Common symptoms of diabetes",      elem_classes=["example-chip"])

            gr.HTML("<hr class='divider-line'>")

            gr.HTML("""
                <div class="info-panel">
                    <h3>💡 How to get the best answer</h3>
                    <p>Ask one focused question at a time. GlucoDoc retrieves evidence from its curated medical knowledge base and cites sources inline.</p>
                </div>

                <div class="disclaimer" style="margin-top:12px;">
                    <b>⚠️ Medical Disclaimer:</b> GlucoDoc is an <b>educational tool</b>. It is not a substitute for a qualified healthcare professional.
                </div>
            """)

            clear_button = gr.Button("🗑  Clear conversation", variant="secondary", elem_id="clear-button")
            gr.HTML('<div class="footer">GlucoDoc · RAG-Powered Diabetes Assistant</div>')

        # ------------------------------------------------
        # Main Panel
        # ------------------------------------------------
        with gr.Column(scale=9, elem_id="main-panel"):

            gr.HTML("""
                <div class="hero-card">
                    <div class="hero-title">Clear answers for better health conversations.</div>
                    <div class="hero-copy">Ask GlucoDoc about diabetes, diagnosis, symptoms, and care. Every response is grounded in the medical knowledge base with inline citations.</div>
                    <div class="badge-row">
                        <span class="badge badge-blue">🧠 RAG-Powered</span>
                        <span class="badge badge-violet">💬 Memory-Enabled</span>
                        <span class="badge badge-green">✅ Safety Guardrails</span>
                    </div>
                </div>
            """)

            chatbot = gr.Chatbot(
                label="Conversation",
                height=520,
                show_label=False,
                elem_id="conversation",
                avatar_images=[
                    "https://api.iconify.design/fluent-emoji:person-light.svg",
                    "https://api.iconify.design/fluent-emoji:stethoscope.svg",
                ],
                placeholder="<div style='text-align:center;color:#64748b;padding:40px 0;'>Welcome to <b>GlucoDoc</b>. Ask a question about diabetes below.</div>",
            )

            with gr.Row(elem_classes=["composer"]):
                message = gr.Textbox(
                    placeholder="Type your question about diabetes…",
                    show_label=False,
                    lines=1,
                    max_lines=5,
                    scale=8,
                    autofocus=True,
                    container=False,
                )
                send_button = gr.Button("Send ↑", variant="primary", scale=1, elem_id="send-button")

            with gr.Row(elem_classes=["summarize-row"]):
                summarize_checkbox = gr.Checkbox(
                    label="⚡  Summarize response (concise mode)",
                    value=False,
                )

            with gr.Accordion("📊 Answer Details & Evidence", open=False, elem_classes=["accordion-wrapper"]):
                answer_box = gr.Markdown(label="Answer Output", elem_id="answer-box", visible=False)
                with gr.Row():
                    confidence_box = gr.Textbox(
                        label="Retrieval Confidence",
                        value="N/A",
                        interactive=False,
                        elem_id="confidence-box",
                    )
                evidence_box = gr.Markdown(label="Supporting Evidence", elem_id="evidence-box")


    # ========================================================
    # Events
    # ========================================================

    send_button.click(
        respond,
        inputs=[message, chatbot, summarize_checkbox],
        outputs=[message, chatbot, answer_box, confidence_box, evidence_box],
    )

    message.submit(
        respond,
        inputs=[message, chatbot, summarize_checkbox],
        outputs=[message, chatbot, answer_box, confidence_box, evidence_box],
    )

    clear_button.click(
        clear_conversation,
        outputs=[chatbot, message, answer_box, confidence_box, evidence_box],
    )

    # ========================================================
    # Examples
    # ========================================================
    example_questions = {
        example_1: "What are the diagnostic criteria for diabetes mellitus?",
        example_2: "What is type 2 diabetes?",
        example_3: "What is the difference between type 1 and type 2 diabetes?",
        example_4: "What are the symptoms of diabetes?",
    }

    for button, question in example_questions.items():
        button.click(
            lambda q=question: q,
            outputs=message,
        ).then(
            respond,
            inputs=[message, chatbot, summarize_checkbox],
            outputs=[message, chatbot, answer_box, confidence_box, evidence_box],
        )


# ============================================================
# Launch
# ============================================================
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        theme=theme,
        css=CSS,
    )
