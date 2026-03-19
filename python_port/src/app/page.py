from __future__ import annotations

from src.app.layout import root_layout


HOME_PAGE_HTML = """
<h1>ArXiv Research Extractor (Python Port)</h1>
<p><a href=\"/chat\">Go to Chat</a></p>
<div class=\"card\"><div class=\"card-content\">
  <h3>Extract Single Paper</h3>
  <div class=\"row\">
    <input id=\"url\" class=\"input\" placeholder=\"https://arxiv.org/html/2502.09457v1\" />
    <button class=\"button\" onclick=\"extractPaper()\">Extract</button>
  </div>
  <pre id=\"out\"></pre>
</div></div>
<div class=\"card\"><div class=\"card-content\">
  <h3>Auto Ingest</h3>
  <div class=\"row\">
    <input id=\"query\" class=\"input\" placeholder=\"machine learning\" />
    <button class=\"button\" onclick=\"autoIngest()\">Run</button>
  </div>
  <pre id=\"ingest\"></pre>
</div></div>
<script>
async function extractPaper() {
  const url = document.getElementById('url').value;
  const res = await fetch('/api/extract?url=' + encodeURIComponent(url));
  document.getElementById('out').textContent = JSON.stringify(await res.json(), null, 2);
}
async function autoIngest() {
  const query = document.getElementById('query').value;
  const res = await fetch('/api/auto-ingest', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({query, limit: 20, existingUrls: []})
  });
  document.getElementById('ingest').textContent = JSON.stringify(await res.json(), null, 2);
}
</script>
"""


def render_research_page() -> str:
    return root_layout(HOME_PAGE_HTML, title="Research Page")
