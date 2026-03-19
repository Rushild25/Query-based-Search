from __future__ import annotations

from src.app.layout import root_layout


CHAT_PAGE_HTML = """
<h1>Chat with Your Data (Python Port)</h1>
<p><a href=\"/\">Home</a></p>
<div class=\"card\"><div class=\"card-content\">
  <form onsubmit=\"return sendMessage(event)\">
    <div class=\"row\">
      <input id=\"msg\" class=\"input\" placeholder=\"Ask a question about your content...\" />
      <button class=\"button\" type=\"submit\">Send</button>
    </div>
  </form>
  <pre id=\"chat_out\"></pre>
</div></div>
<script>
let messages = [];
async function sendMessage(e){
  e.preventDefault();
  const content = document.getElementById('msg').value;
  if(!content) return false;
  messages.push({role:'user', content});
  const res = await fetch('/api/vespa/search', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({query: content})
  });
  const data = await res.json();
  messages.push({role:'assistant', content: data.answer || data.detail || 'No answer'});
  document.getElementById('chat_out').textContent = JSON.stringify(messages, null, 2);
  document.getElementById('msg').value = '';
  return false;
}
</script>
"""


def render_chat_page() -> str:
    return root_layout(CHAT_PAGE_HTML, title="Chat Page")
