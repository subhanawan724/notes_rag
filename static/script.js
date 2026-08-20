const askBtn = document.getElementById("ask-btn");
const queryInput = document.getElementById("query");
const messagesDiv = document.getElementById("messages");

let threadId = localStorage.getItem("thread_id");
if (!threadId) {
  threadId = crypto.randomUUID();
  localStorage.setItem("thread_id", threadId);
}

function addBubble(text, sender) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${sender}`;
  if (sender === "bot") {
    bubble.innerHTML = marked.parse(text);
    bubble.querySelectorAll("pre code").forEach(b => hljs.highlightElement(b));
  } else {
    bubble.textContent = text;
  }
  messagesDiv.appendChild(bubble);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
  return bubble;
}

askBtn.addEventListener("click", async () => {
  const q = queryInput.value.trim();
  if (!q) return;
  addBubble(q, "user");
  queryInput.value = "";
  const loading = addBubble("Thinking...", "bot");

  const res = await fetch("/question", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_query: q, thread_id: threadId })
  });
  const data = await res.json();
  loading.remove();
  addBubble(data.answer, "bot");
});

queryInput.addEventListener("keydown", (e) => { if (e.key === "Enter") askBtn.click(); });
