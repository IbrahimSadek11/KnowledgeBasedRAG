"""
SportLLM - Beautiful Working Interface
=======================================
Custom styling with trending articles and improved UX
"""

import streamlit as st
import os
import json
import sys
from datetime import datetime
from pathlib import Path

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from backend.graph_rag.llm_service import init_graph_chain
from backend.news_service import EquestrianNewsScraper

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="SportLLM",
    page_icon="🐴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

#MainMenu {visibility: hidden;}
footer {visibility: hidden !important;}
header {visibility: hidden;}
.stDeployButton {display: none;}

.main {background: #FDFBF7;}
.block-container {padding: 2rem 3rem; max-width: 1400px;}

h1, h2, h3 {font-family: 'Playfair Display', serif; color: #C9A5A0; font-weight: 600;}
body, p, div, span, button, input {font-family: 'Inter', sans-serif;}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F5EFE7 0%, #E8DDD0 100%);
    border-right: 2px solid #D4ACA7;
}

[data-testid="stSidebar"] h2 {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    color: #C9A5A0;
    font-weight: 600;
    margin-bottom: 1.5rem;
}

[data-testid="stSidebar"] button {
    background: linear-gradient(135deg, #C9A5A0, #8B6F5C);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-weight: 500;
    transition: all 0.3s ease;
    box-shadow: 0 2px 4px rgba(201, 165, 160, 0.2);
}

[data-testid="stSidebar"] button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(201, 165, 160, 0.4);
}

[data-testid="stSidebar"] button[kind="primary"] {
    background: linear-gradient(135deg, #C9A5A0, #8B6F5C);
}

[data-testid="stSidebar"] button[kind="secondary"] {
    background: white;
    color: #2C3E50;
    border: 1px solid #D4ACA7;
}

[data-testid="stSidebar"] button[kind="secondary"]:hover {
    background: #E8DDD0;
}

/* Icon buttons in sidebar */
.icon-button {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
}

[data-testid="stSidebar"] button[data-icon="true"] {
    padding: 0.5rem;
    min-width: unset;
    font-size: 1.1rem;
}

/* Delete button styling */
[data-testid="stSidebar"] button[data-delete="true"] {
    background: #dc3545 !important;
    color: white !important;
}

[data-testid="stSidebar"] button[data-delete="true"]:hover {
    background: #c82333 !important;
    box-shadow: 0 4px 12px rgba(220, 53, 69, 0.4) !important;
}

/* Trending articles cards */
.trending-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    border-left: 4px solid #C9A5A0;
    transition: all 0.3s ease;
    cursor: pointer;
    text-decoration: none;
    display: block;
    color: inherit;
}

.trending-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 20px rgba(201, 165, 160, 0.3);
}

.trending-badge {
    display: inline-block;
    background: linear-gradient(135deg, #C9A5A0, #8B6F5C);
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.trending-title {
    font-size: 1rem;
    font-weight: 600;
    color: #2C3E50;
    margin-bottom: 0.5rem;
    line-height: 1.4;
    overflow: hidden;
    text-overflow: ellipsis;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
}

.trending-source {
    font-size: 0.85rem;
    color: #95A5A6;
}

/* Chat messages */
.stChatMessage {
    border-radius: 20px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.5rem;
    animation: fadeIn 0.3s ease;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}

[data-testid="stChatMessageContainer"] > div:has([data-testid="user-message"]) .stChatMessage {
    background: linear-gradient(135deg, #C9A5A0 0%, #B89590 100%);
    color: white;
    border: none;
    margin-left: auto;
    max-width: 70%;
    border-radius: 20px 20px 4px 20px;
    box-shadow: 0 4px 12px rgba(201, 165, 160, 0.3);
}

[data-testid="stChatMessageContainer"] > div:has([data-testid="user-message"]) .stChatMessage p {
    color: white;
}

[data-testid="stChatMessageContainer"] > div:has([data-testid="assistant-message"]) .stChatMessage {
    background: white;
    border-left: 4px solid #C9A5A0;
    color: #2C3E50;
    max-width: 70%;
    border-radius: 20px 20px 20px 4px;
}

/* AVATARS - Icônes dans les messages */
[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"] {
    background: linear-gradient(135deg, #C9A5A0 0%, #8B6F5C 100%) !important;
}

/* SVG dans les avatars - icônes blanches */
[data-testid="chatAvatarIcon-user"] svg,
[data-testid="chatAvatarIcon-assistant"] svg {
    color: white !important;
    fill: white !important;
}

.stChatInputContainer {
    border-top: 2px solid #D4ACA7;
    background: white;
    padding: 1.5rem 0;
    box-shadow: 0 -4px 12px rgba(0, 0, 0, 0.05);
}

.stChatInput > div {
    border: 2px solid #D4ACA7;
    border-radius: 25px;
    background: #FDFBF7;
    transition: all 0.3s ease;
}

.stChatInput > div:focus-within {
    border-color: #C9A5A0;
    box-shadow: 0 0 0 4px rgba(201, 165, 160, 0.1);
    background: white;
}

.stChatInput input {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    padding: 0.5rem 1rem;
}

/* BOUTON D'ENVOI - Flèche */
.stChatInput button,
button[data-testid="stChatInputSubmitButton"],
.stChatInputContainer button {
    background: linear-gradient(135deg, #C9A5A0 0%, #8B6F5C 100%) !important;
    border: none !important;
    transition: all 0.3s ease !important;
}

.stChatInput button:hover,
button[data-testid="stChatInputSubmitButton"]:hover,
.stChatInputContainer button:hover {
    background: linear-gradient(135deg, #8B6F5C 0%, #7A5F4D 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 6px rgba(201, 165, 160, 0.15) !important;
}

.stChatInput button svg,
button[data-testid="stChatInputSubmitButton"] svg,
.stChatInputContainer button svg {
    color: white !important;
    fill: white !important;
}

.stSpinner > div {
    border-top-color: #C9A5A0 !important;
    border-right-color: #C9A5A0 !important;
}

hr {border-color: #D4ACA7; margin: 1.5rem 0;}

.streamlit-expanderHeader {
    background: #F5EFE7;
    border: 1px solid #D4ACA7;
    border-radius: 8px;
    color: #2C3E50;
    transition: all 0.3s ease;
}

.streamlit-expanderHeader:hover {
    background: #E8DDD0;
    border-color: #C9A5A0;
}

.stTextInput > div > div > input {
    border: 2px solid #D4ACA7;
    border-radius: 8px;
    padding: 0.75rem;
    transition: all 0.3s ease;
}

.stTextInput > div > div > input:focus {
    border-color: #C9A5A0;
    box-shadow: 0 0 0 3px rgba(201, 165, 160, 0.1);
}

.stAlert {border-radius: 8px; border-left-width: 4px;}
.stSuccess {background: #E8F5E9; border-left-color: #4CAF50;}
.stWarning {background: #FFF3E0; border-left-color: #FF9800;}
.stError {background: #FFEBEE; border-left-color: #F44336;}
.stInfo {background: #FFF8F0; border-left-color: #C9A5A0;}

::-webkit-scrollbar {width: 8px;}
::-webkit-scrollbar-track {background: #F5EFE7;}
::-webkit-scrollbar-thumb {background: #C9A5A0; border-radius: 4px;}
::-webkit-scrollbar-thumb:hover {background: #B89590;}

.professional-header {
    background: linear-gradient(135deg, #C9A5A0 0%, #8B6F5C 100%);
    color: white;
    padding: 2.5rem 3rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 12px rgba(201, 165, 160, 0.3);
}

.professional-header h1 {
    margin: 0;
    color: white;
    font-size: 2.5rem;
    font-weight: 600;
}

.professional-header p {
    margin: 0.5rem 0 0 0;
    color: rgba(255, 255, 255, 0.95);
    font-size: 1.1rem;
}

.text-muted {color: #95A5A6; font-size: 0.9rem;}

/* Modal overlay for delete confirmation */
.modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9999;
}

.modal-content {
    background: white;
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
    max-width: 400px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA SETUP
# ============================================================================

CONVERSATIONS_DIR = Path("../data/conversations")
CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_FILE = CONVERSATIONS_DIR / "config.json"

def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"last_conversation": None}

def save_config(config):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

def get_conversations():
    conversations = []
    for file in CONVERSATIONS_DIR.glob("conv_*.json"):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                conversations.append({
                    "id": file.stem,
                    "title": data.get("title", "Sans titre"),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at")
                })
        except:
            continue
    return sorted(conversations, key=lambda x: x.get("updated_at", ""), reverse=True)

def load_conversation(conv_id):
    file_path = CONVERSATIONS_DIR / f"{conv_id}.json"
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def save_conversation(conv_id, title, messages):
    file_path = CONVERSATIONS_DIR / f"{conv_id}.json"
    data = {
        "title": title,
        "messages": messages,
        "updated_at": datetime.now().isoformat()
    }
    if not file_path.exists():
        data["created_at"] = datetime.now().isoformat()
    else:
        existing = load_conversation(conv_id)
        if existing and "created_at" in existing:
            data["created_at"] = existing["created_at"]
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def create_new_conversation():
    return f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def delete_conversation(conv_id):
    file_path = CONVERSATIONS_DIR / f"{conv_id}.json"
    if file_path.exists():
        file_path.unlink()
        return True
    return False

@st.cache_resource
def get_chain_and_graph():
    try:
        chain, graph = init_graph_chain()
        return chain, graph
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation : {str(e)}")
        return None, None

@st.cache_data(ttl=3600)
def get_trending_articles():
    """Fetch top 3 trending articles"""
    try:
        scraper = EquestrianNewsScraper(os.getenv("OPENAI_API_KEY"))
        articles = scraper.fetch_news(max_articles=10)
        return articles[:3] if articles else []
    except:
        return []

# ============================================================================
# SESSION STATE
# ============================================================================

if "current_conversation" not in st.session_state:
    config = load_config()
    st.session_state.current_conversation = config.get("last_conversation") or create_new_conversation()

if "messages" not in st.session_state:
    conv_data = load_conversation(st.session_state.current_conversation)
    st.session_state.messages = conv_data.get("messages", []) if conv_data else []

if "conversation_title" not in st.session_state:
    conv_data = load_conversation(st.session_state.current_conversation)
    st.session_state.conversation_title = conv_data.get("title", "Nouvelle conversation") if conv_data else "Nouvelle conversation"

if "editing_title" not in st.session_state:
    st.session_state.editing_title = False

if "editing_conv_id" not in st.session_state:
    st.session_state.editing_conv_id = None

if "delete_error" not in st.session_state:
    st.session_state.delete_error = None

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown('<h2>Conversations</h2>', unsafe_allow_html=True)
    
    if st.button("Nouvelle conversation", use_container_width=True, type="primary"):
        save_conversation(st.session_state.current_conversation, st.session_state.conversation_title, st.session_state.messages)
        st.session_state.current_conversation = create_new_conversation()
        st.session_state.messages = []
        st.session_state.conversation_title = "Nouvelle conversation"
        st.session_state.delete_error = None
        st.rerun()
    
    st.divider()
    
    conversations = get_conversations()
    
    if conversations:
        st.markdown('<h3 style="font-size: 1rem; margin-bottom: 1rem;">Historique</h3>', unsafe_allow_html=True)
        
        for conv in conversations:
            is_current = conv["id"] == st.session_state.current_conversation
            
            # Check if this conversation is being edited
            if st.session_state.editing_conv_id == conv["id"]:
                new_title = st.text_input("", value=conv["title"], key=f"edit_{conv['id']}", label_visibility="collapsed")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Enregistrer", key=f"save_{conv['id']}", use_container_width=True):
                        conv_data = load_conversation(conv["id"])
                        if conv_data:
                            save_conversation(conv["id"], new_title, conv_data.get("messages", []))
                            if is_current:
                                st.session_state.conversation_title = new_title
                        st.session_state.editing_conv_id = None
                        st.rerun()
                with col2:
                    if st.button("Annuler", key=f"cancel_{conv['id']}", use_container_width=True):
                        st.session_state.editing_conv_id = None
                        st.rerun()
            else:
                col1, col2, col3 = st.columns([5, 1, 1])
                
                with col1:
                    button_label = f"{'• ' if is_current else ''}{conv['title'][:25]}"
                    button_type = "primary" if is_current else "secondary"
                    
                    if st.button(button_label, key=f"load_{conv['id']}", use_container_width=True, type=button_type):
                        if not is_current:
                            save_conversation(st.session_state.current_conversation, st.session_state.conversation_title, st.session_state.messages)
                            conv_data = load_conversation(conv["id"])
                            if conv_data:
                                st.session_state.current_conversation = conv["id"]
                                st.session_state.messages = conv_data.get("messages", [])
                                st.session_state.conversation_title = conv_data.get("title", "Sans titre")
                                save_config({"last_conversation": conv["id"]})
                                st.session_state.delete_error = None
                                st.rerun()
                
                with col2:
                    if st.button("✎", key=f"edit_{conv['id']}", help="Modifier le titre"):
                        st.session_state.editing_conv_id = conv["id"]
                        st.rerun()
                
                with col3:
                    if st.button("✕", key=f"del_{conv['id']}", help="Supprimer"):
                        if is_current:
                            st.session_state.delete_error = "Impossible de supprimer la conversation active"
                            st.rerun()
                        else:
                            if delete_conversation(conv["id"]):
                                st.session_state.delete_error = None
                                st.rerun()
    else:
        st.info("Aucune conversation sauvegardée")

# ============================================================================
# MAIN INTERFACE
# ============================================================================

st.markdown("""
<div class="professional-header">
    <h1>SportLLM</h1>
    <p>Posez-moi vos questions sur le monde équestre et je ferai de mon mieux pour y répondre</p>
</div>
""", unsafe_allow_html=True)

# Display delete error if exists
if st.session_state.delete_error:
    st.error(st.session_state.delete_error)
    if st.button("Fermer"):
        st.session_state.delete_error = None
        st.rerun()

# Trending articles
st.markdown("### 📰 Actualités Équestres")
trending_articles = get_trending_articles()

if trending_articles:
    cols = st.columns(3)
    for idx, article in enumerate(trending_articles):
        with cols[idx]:
            # Make the entire card clickable
            link = article.get('link', '#')
            title = article.get('title', 'Sans titre')[:60]
            source = article.get('source', 'Source inconnue')
            
            st.markdown(f"""
            <a href="{link}" target="_blank" class="trending-card">
                <span class="trending-badge">TENDANCE</span>
                <div class="trending-title">{title}</div>
                <div class="trending-source">{source}</div>
            </a>
            """, unsafe_allow_html=True)
else:
    st.info("Chargement des actualités...")

st.divider()

# Title display
st.markdown(f'<p class="text-muted">Conversation : {st.session_state.conversation_title}</p>', unsafe_allow_html=True)

st.divider()

# Check config
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    st.error("Configuration manquante : Veuillez configurer votre clé OpenAI")
    st.stop()

# Initialize
with st.spinner("Connexion..."):
    chain, graph = get_chain_and_graph()

if not chain or not graph:
    st.error("Impossible de se connecter au graphe Neo4j")
    st.stop()

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Posez votre question..."):
    if len(st.session_state.messages) == 0 and st.session_state.conversation_title == "Nouvelle conversation":
        st.session_state.conversation_title = prompt[:45] + ("..." if len(prompt) > 45 else "")
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Traitement..."):
            try:
                # Appel direct à la chaîne sans mémoire
                result = chain.invoke({"query": prompt})
                answer = result.get("result", "Désolé, je n'ai pas pu trouver de réponse.")
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_str = str(e)
                if "SyntaxError" in error_str or "Invalid input" in error_str:
                    st.warning("Erreur de génération de requête")
                    friendly_msg = "Je n'ai pas pu générer une requête valide. Essayez de reformuler."
                    st.info(friendly_msg)
                    st.session_state.messages.append({"role": "assistant", "content": friendly_msg})
                else:
                    st.error(f"Erreur : {error_str}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Erreur : {error_str}"})
            
            save_conversation(st.session_state.current_conversation, st.session_state.conversation_title, st.session_state.messages)
            save_config({"last_conversation": st.session_state.current_conversation})