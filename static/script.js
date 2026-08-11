let threadId = localStorage.getItem("thread_id");
if (!threadId) {
  threadId = crypto.randomUUID();
  localStorage.setItem("thread_id", threadId);
}
const askBtn = document.getElementById("ask-btn");
const queryInput = document.getElementById("query");
const answerBox = document.getElementById("answer-box");

askBtn.addEventListener("click", async () => {
  const userQuery = queryInput.value.trim();
  if (!userQuery) return;

  answerBox.innerHTML = "Thinking...";

  const response = await fetch("/question", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_query: userQuery, thread_id: threadId })
  });

  const data = await response.json();
  const rawMarkdown = data.answer;

  const html = marked.parse(rawMarkdown);
  answerBox.innerHTML = html;

  document.querySelectorAll("pre code").forEach((block) => {
    hljs.highlightElement(block);
  });
});
